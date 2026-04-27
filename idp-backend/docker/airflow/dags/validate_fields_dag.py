from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.mysql.hooks.mysql import MySqlHook
from datetime import datetime, timedelta
from pdf2image import convert_from_path

import pytesseract
import openai
import os
import json
import re
import requests
from dotenv import load_dotenv
from transaction_status import sync_stage_status
from ocr_services.ocr_cache_utils import ensure_ocr_cache, get_cached_document_text

load_dotenv() 

AUTO_EXECUTE_NEXT_NODE = 0

# === DAG Trigger CONFIG === #
AIRFLOW_API_URL = "http://airflow-airflow-apiserver-1:8080/api/v2"  # or localhost in local mode
AIRFLOW_USERNAME = os.getenv("AIRFLOW_USERNAME")
AIRFLOW_PASSWORD = os.getenv("AIRFLOW_PASSWORD")
LOCAL_MODE = os.getenv("LOCAL_MODE", "false").lower() == "true"

if LOCAL_MODE:
    AIRFLOW_API_URL = "http://localhost:8080/api/v2"

# === CONFIG ===
LOCAL_DOWNLOAD_DIR = "/opt/airflow/downloaded_docs"

# Set your OpenAI API Key
openai.api_key = os.getenv("OPENAI_API_KEY")  # from .env
if not openai.api_key or not openai.api_key.startswith("sk-") and not openai.api_key.startswith("sk-proj-"):
    raise EnvironmentError("❌ OpenAI API key missing or invalid. Please set OPENAI_API_KEY as an environment variable.")   
    
def get_auth_token():
    """Get JWT token from Airflow API"""
    auth_url = f"{AIRFLOW_API_URL.replace('/api/v2', '')}/auth/token"
    response = requests.post(
        auth_url,
        json={"username": AIRFLOW_USERNAME, "password": AIRFLOW_PASSWORD},
        headers={"Content-Type": "application/json"},
        timeout=10
    )
    response.raise_for_status()
    return response.json()["access_token"]

def build_ocr_logger():
    def _logger(level, message):
        print(message)

    return _logger


def extract_text_from_pdf(pdf_path, logger_callback=None):
    process_instance_dir = os.path.dirname(pdf_path)
    cached_text = get_cached_document_text(process_instance_dir, os.path.basename(pdf_path))
    if cached_text and cached_text.strip():
        return cached_text

    cache_payload = ensure_ocr_cache(
        pdf_path=pdf_path,
        process_instance_dir=process_instance_dir,
        ocr_engine="paddle",
        config={"dpi": 300},
        logger_callback=logger_callback,
    )
    cached_text = cache_payload.get("cleaned_text") or cache_payload.get("raw_text") or ""
    if cached_text.strip():
        return cached_text

    try:
        images = convert_from_path(pdf_path)
        text = ""
        for img in images:
            text += pytesseract.image_to_string(img)
        return text
    except Exception as e:
        print(f"OCR failed for {pdf_path}: {e}")
        return ""

def validate_extracted_fields(**context):
    # Get process instance ID from DAG run configuration
    process_instance_id = context["dag_run"].conf.get("id")
    if not process_instance_id:
        raise ValueError("Missing process_instance_id in dag_run.conf")
    
    process_instance_dir_path = os.path.join(LOCAL_DOWNLOAD_DIR, "process-instance-" + str(process_instance_id))
    os.makedirs(process_instance_dir_path, exist_ok=True)
    EXTRACTED_FIELDS_PATH = os.path.join(process_instance_dir_path, "cleaned_extracted_fields.json")

    # Step 1: Update currentStage in ProcessInstances
    hook = MySqlHook(mysql_conn_id="idp_mysql")
    conn = hook.get_conn()
    cursor = conn.cursor()
    transaction_id = sync_stage_status(cursor, process_instance_id, "Validation", 1)
    conn.commit()
    print(f"🟢 Updated ProcessInstances to 'Validation' stage for process_instance_id={process_instance_id}")

    # Step 2: Load extracted fields
    if not os.path.exists(EXTRACTED_FIELDS_PATH):
        raise FileNotFoundError("❌ cleaned_extracted_fields.json not found")

    with open(EXTRACTED_FIELDS_PATH, "r") as f:
        extracted_data = json.load(f)

    # Step 3: Validate each document
    for doc in extracted_data:
        document_meta = doc.get("documentDetails", {})
        file_name = document_meta.get("documentName")
        doc_type = document_meta.get("documentType")
        fields = doc.get("extractedFields", {})
        process_instance_id = doc.get("processInstanceId")

        if not file_name or not doc_type or not process_instance_id:
            print("❌ Missing critical data in entry, skipping...")
            continue

        doc_path = os.path.join(process_instance_dir_path, file_name)
        if not os.path.exists(doc_path):
            print(f"❌ File missing: {file_name}")
            continue

        ocr_text = extract_text_from_pdf(doc_path, logger_callback=build_ocr_logger())
        score_sum = 0
        valid_fields = 0
        validated_fields = []

        for field_name, extracted_value in fields.items():
            if extracted_value.strip().upper() == "N/A":
                continue

            prompt = f"""
            You are validating extracted data from a scanned document.

            Document Type: {doc_type}

            OCR Text:
            {ocr_text[:1500]}

            Field Name: {field_name}
            Extracted Value: {extracted_value}

            Validation Rules:
            - Look for the extracted value in the OCR text.
            - Minor spelling errors or OCR typos should not reduce the score.
            - Only consider the format or presence of value.
            - Do NOT penalize spelling mistakes unless they alter the actual meaning.

            Return result in this format:
            Score: <number between 0-100>.
            """

            try:
                response = openai.ChatCompletion.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2
                )
                content = response["choices"][0]["message"]["content"].strip()
                score_match = re.search(r"Score\s*:\s*(\d+)", content)
                if score_match:
                    score = int(score_match.group(1))
                    validated_fields.append({
                        "fieldName": field_name,
                        "fieldValue": extracted_value,
                        "fieldScore": score
                    })
                    score_sum += score
                    valid_fields += 1
                else:
                    print(f"⚠️ Score not found in response: {content}")
            except Exception as e:
                print(f"❌ Validation failed for {field_name}: {e}")

        # Replace raw dict with validated list
        doc["extractedFields"] = validated_fields
        overall_score = round(score_sum / valid_fields) if valid_fields else 0

        # Step 4: Save documentDetails + overAllScore
        document_details_json = json.dumps({
            "documentName": file_name,
            "documentType": doc_type
        })

        cursor.execute("""
            INSERT INTO ProcessInstanceDocuments (processInstancesId, documentDetails, overAllScore, updatedAt)
            VALUES (%s, %s, %s, NOW())
            ON DUPLICATE KEY UPDATE
                documentDetails = VALUES(documentDetails),
                overAllScore = VALUES(overAllScore),
                updatedAt = NOW()
        """, (process_instance_id, document_details_json, overall_score))
        conn.commit()

        print(f"✅ Stored metadata for {file_name} with avg score {overall_score}%")

    # Step 5: Write updated fields with fieldScore back to file
    with open(EXTRACTED_FIELDS_PATH, "w") as f:
        json.dump(extracted_data, f, indent=2)
    print(f"✅ Updated extracted_fields with fieldScore in {EXTRACTED_FIELDS_PATH}")

    # Trigger highlight_extracted_documents_dag
    if AUTO_EXECUTE_NEXT_NODE == 1:
        print("🚀 Triggering highlight_extracted_documents_dag...")
        token = get_auth_token()
        trigger_url = f"{AIRFLOW_API_URL}/dags/highlight_extracted_fields_dag/dagRuns"
        run_id = f"triggered_by_validation_{process_instance_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        payload = {
            "dag_run_id": run_id,
            "logical_date": datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
            "conf": {"id": process_instance_id}
        }

        response = requests.post(trigger_url, json=payload, headers=headers, timeout=10)
        response.raise_for_status()
        print(f"✅ Successfully triggered highlight_extracted_documents_dag with ID {process_instance_id}")


# === DAG Definition ===
with DAG(
    dag_id="validate_documents_dag",
    start_date=datetime.now() - timedelta(days=1),
    schedule=None,
    catchup=False,
    tags=["idp", "validation"]
) as dag:

    validate_task = PythonOperator(
        task_id="validate_extracted_fields",
        python_callable=validate_extracted_fields,
    )

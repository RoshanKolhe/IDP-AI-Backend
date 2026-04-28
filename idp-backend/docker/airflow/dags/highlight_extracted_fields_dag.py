from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pytesseract
from pdf2image import convert_from_path
from PIL import Image, ImageDraw
import os
import json
import re
import requests
from airflow.providers.mysql.hooks.mysql import MySqlHook
from openai import OpenAI
from opik.integrations.openai import track_openai
from dotenv import load_dotenv
from pymongo import MongoClient
from transaction_status import sync_stage_status
from ocr_services.ocr_cache_utils import ensure_ocr_cache, get_cached_document_text

load_dotenv() 

AUTO_EXECUTE_NEXT_NODE = 1
AIRFLOW_API_URL = "http://airflow-airflow-apiserver-1:8080/api/v2"
AIRFLOW_USERNAME = os.getenv("AIRFLOW_USERNAME")
AIRFLOW_PASSWORD = os.getenv("AIRFLOW_PASSWORD")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # from .env
OpenAI.api_key = OPENAI_API_KEY
OPIK_API_KEY = os.getenv("OPIK_API_KEY")  
MONGO_URI = os.getenv("MONGO_URI")
LOCAL_MODE = os.getenv("LOCAL_MODE", "false").lower() == "true"

if LOCAL_MODE:
    AIRFLOW_API_URL = "http://localhost:8080/api/v2"

LOCAL_DOWNLOAD_DIR = "/opt/airflow/downloaded_docs"
UPLOAD_URL = "https://api.docognize.ai/files"

MONGO_DB_NAME = "idp"
MONGO_COLLECTION = "LogEntry"
mongo_client = MongoClient(MONGO_URI)
mongo_collection = mongo_client[MONGO_DB_NAME][MONGO_COLLECTION]
openai_client = OpenAI()  
openai_client = track_openai(openai_client, project_name="my-idp-project")

def _get_transaction_id(process_instance_id):
    tid_path = os.path.join(LOCAL_DOWNLOAD_DIR, f"process-instance-{process_instance_id}", "tid.json")
    if not os.path.exists(tid_path):
        return None
    try:
        with open(tid_path, "r", encoding="utf-8") as f:
            return json.load(f).get("transactionId")
    except Exception as exc:
        print(f"Warning: failed to read transaction id from {tid_path}: {exc}")
        return None

if not OpenAI.api_key or not OpenAI.api_key.startswith("sk-") and not OpenAI.api_key.startswith("sk-proj-"):
    raise EnvironmentError("❌ OpenAI API key missing or invalid. Please set OPENAI_API_KEY as an environment variable.")

def log_to_mongo(process_instance_id, node_name, message, log_type=1, remark=""):
    try:
        log_entry = {
            "processInstanceId": process_instance_id,
            "processInstanceTransactionId": _get_transaction_id(process_instance_id),
            "nodeName": node_name,
            "logsDescription": message,
            "logType": log_type,  # 0=info, 1=error, 2=success, 3=warning
            "isDeleted": False,
            "isActive": True,
            "remark": remark,
            "createdAt": datetime.utcnow()
        }
        mongo_collection.insert_one(log_entry)
        print(f"📝 Logged to MongoDB: {message}")
    except Exception as mongo_err:
        print(f"⚠️ Failed to log to MongoDB: {mongo_err}")

def build_ocr_logger(process_instance_id, node_name):
    def _logger(level, message):
        level_map = {
            "info": 0,
            "error": 1,
            "success": 2,
            "warning": 3,
        }
        print(message)
        log_to_mongo(
            process_instance_id,
            node_name=node_name,
            message=message,
            log_type=level_map.get(level, 0),
        )

    return _logger


def get_auth_token():
    auth_url = f"{AIRFLOW_API_URL.replace('/api/v2', '')}/auth/token"
    response = requests.post(
        auth_url,
        json={"username": AIRFLOW_USERNAME, "password": AIRFLOW_PASSWORD},
        headers={"Content-Type": "application/json"},
        timeout=10
    )
    response.raise_for_status()
    return response.json()["access_token"]

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
        fallback_ocr_engine="tesseract",
    )
    cached_text = cache_payload.get("cleaned_text") or cache_payload.get("raw_text") or ""
    if cached_text.strip():
        return cached_text

    try:
        images = convert_from_path(pdf_path)
        return "".join(pytesseract.image_to_string(img) for img in images)
    except Exception as e:
        print(f"OCR failed for {pdf_path}: {e}")
        log_to_mongo(process_instance_id, message = f"OCR failed for {pdf_path}: {e}", node_name = "Validation", log_type=1)
        return ""

def highlight_and_upload(**context):
    process_instance_id = context["dag_run"].conf.get("id")
    is_orchestrated = bool(context["dag_run"].conf.get("orchestrated", False))
    print("highlight_and_upload() function has started.")
    if not process_instance_id:
        raise ValueError("Missing process_instance_id in dag_run.conf")
        log_to_mongo(process_instance_id, message = f"Missing process_instance_id in dag_run.conf", node_name = "Validation", log_type=1)

    dir_path = os.path.join(LOCAL_DOWNLOAD_DIR, f"process-instance-{process_instance_id}")
    os.makedirs(dir_path, exist_ok=True)
    cleaned_fields_path = os.path.join(dir_path, "cleaned_extracted_fields.json")
    highlighted_dir = os.path.join(dir_path, "highlighted_docs")
    os.makedirs(highlighted_dir, exist_ok=True)

    response_body_path = os.path.join(dir_path, "response_body.json")

    hook = MySqlHook(mysql_conn_id="idp_mysql")
    conn = hook.get_conn()
    cursor = conn.cursor()
    transaction_id = sync_stage_status(cursor, process_instance_id, "Validation", 1)
    conn.commit()
    print(f"🟢 Updated ProcessInstances to 'Validation' stage for process_instance_id={process_instance_id}")
    log_to_mongo(process_instance_id, message = f"Updated ProcessInstances to 'Validation' stage for process_instance_id={process_instance_id}", node_name = "Validation", log_type=2)

    if not os.path.exists(cleaned_fields_path):
        raise FileNotFoundError("cleaned_extracted_fields.json not found")
        log_to_mongo(process_instance_id, message = f"cleaned_extracted_fields.json not found. Please Run Extraction first.", node_name = "Validation", log_type=1)

    with open(cleaned_fields_path, "r") as f:
        extracted_data = json.load(f)

    for doc in extracted_data:
        document_name = doc["documentDetails"]["documentName"]
        document_type = doc["documentDetails"]["documentType"]
        raw_fields = doc.get("extractedFields", {})
        fields = []

        if isinstance(raw_fields, dict):
            for k, v in raw_fields.items():
                fields.append({"fieldName": k, "fieldValue": v})
        else:
            print(f"⚠️ Unexpected format for extractedFields: {type(raw_fields)}")
            log_to_mongo(process_instance_id, message = f"Unexpected format for extractedFields: {type(raw_fields)}", node_name = "Validation", log_type=3)

        pid = doc.get("processInstanceId")

        pdf_path = os.path.join(dir_path, document_name)
        if not os.path.exists(pdf_path):
            continue

        ocr_text = extract_text_from_pdf(pdf_path, logger_callback=build_ocr_logger(process_instance_id, "Validation"))
        validated_fields, score_sum = [], 0

        for field in fields:
            field_name = field.get("fieldName", "")
            value = field.get("fieldValue", "")

            if str(value).strip().upper() == "N/A":
                continue
            prompt = f"""
            You are validating extracted data from a scanned document.
            Document Type: {document_type}
            OCR Text: {ocr_text[:1500]}
            Field Name: {field_name}
            Extracted Value: {value}
            Validation Rules:
            - Compare the extracted value with the OCR text semantically, not word-by-word.
	    - Minor paraphrasing, reordering, or punctuation differences should still count as a strong match.
	    - Return a confidence score between 0–100:
  	    - 90–100: Strong match (semantically equivalent).
  	    - 60–89: Partial match (some overlap but not full).
  	    - 30–59: Weak match (topic is related but value is incomplete or vague).
  	    - 0–29: Not found.
	    Return result in this format:
	    Score: <number between 0-100>
            """
            try:
                response = openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2
                )
                content = response.choices[0].message.content.strip()
                match = re.search(r"Score\s*:\s*(\d+)", content)
                if match:
                    score = int(match.group(1))
                    validated_fields.append({
                        "fieldName": field_name,
                        "fieldValue": value,
                        "fieldScore": score
                    })
                    score_sum += score
            except Exception as e:
                print(f"Validation failed for {field_name}: {e}")
                log_to_mongo(process_instance_id, message = f"Validation failed for {field_name}: {e}", node_name = "Validation", log_type=1)

        overall_score = round(score_sum / len(validated_fields)) if validated_fields else 0
        doc["extractedFields"] = validated_fields

        with open(cleaned_fields_path, "w") as f:
            json.dump(extracted_data, f, indent=2)
        print(f"✅ Updated extracted_fields with fieldScore in {cleaned_fields_path}")
        log_to_mongo(process_instance_id, message = f"Updated extracted_fields with fieldScore in {cleaned_fields_path}", node_name = "Validation", log_type=2)

        document_details_json = json.dumps(doc["documentDetails"])

################################################## HIGHLIGHT DOCUMENT LOGIC BELOW THIS POINT ########################################

        images = convert_from_path(pdf_path)
        highlighted_images = []

        for image in images:
            overlay = Image.new('RGBA', image.size, (255, 255, 255, 0))
            draw_overlay = ImageDraw.Draw(overlay)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            draw = ImageDraw.Draw(image)
            ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

            text_blocks = [
                {
                    "text": ocr_data["text"][i].lower(),
                    "left": ocr_data["left"][i],
                    "top": ocr_data["top"][i],
                    "width": ocr_data["width"][i],
                    "height": ocr_data["height"][i]
                }
                for i in range(len(ocr_data["text"])) if str(ocr_data["text"][i]).strip() and int(ocr_data["conf"][i]) > 60
            ]

            for field in validated_fields:
                val_words = re.findall(r'[a-z0-9]+', str(field["fieldValue"]).lower())
                for i in range(len(text_blocks) - len(val_words) + 1):
                    if all(val_words[j] == text_blocks[i + j]["text"] for j in range(len(val_words))):
                        x = text_blocks[i]["left"]
                        y = text_blocks[i]["top"]
                        x_end = text_blocks[i + len(val_words) - 1]["left"] + text_blocks[i + len(val_words) - 1]["width"]
                        y_end = text_blocks[i + len(val_words) - 1]["top"] + text_blocks[i + len(val_words) - 1]["height"]
                        draw_overlay.rectangle([(x, y), (x_end, y_end)], fill=(255, 255, 102, 100), outline=(255, 204, 0, 200))
                        break

            image = Image.alpha_composite(image.convert('RGBA'), overlay).convert('RGB')
            highlighted_images.append(image)

        output_pdf = os.path.join(highlighted_dir, document_name)
        highlighted_images[0].save(output_pdf, save_all=True, append_images=highlighted_images[1:], format="PDF")
        print(f"✅ Highlighted PDF saved: {output_pdf}")
        log_to_mongo(process_instance_id, message = f"Highlighted PDF saved: {output_pdf}", node_name = "Validation", log_type=2)

        with open(output_pdf, "rb") as f:
            response = requests.post(UPLOAD_URL, files={"file": (document_name, f, "application/pdf")})
            file_response = response.json() if response.status_code == 200 else {}

        os.remove(output_pdf)

        file_details = json.dumps(file_response.get("files", [{}])[0])
        extracted_fields_json = json.dumps(validated_fields)

        cursor.execute("""
            SELECT id FROM ProcessInstanceDocuments
            WHERE processInstancesId = %s AND JSON_UNQUOTE(JSON_EXTRACT(documentDetails, '$.documentName')) = %s
        """, (pid, document_name))
        exists = cursor.fetchone()

        if exists:
            cursor.execute("""
                UPDATE ProcessInstanceDocuments
                SET documentDetails = %s,
                    extractedFields = %s,
                    fileDetails = %s,
                    overAllScore = %s,
                    updatedAt = NOW()
                WHERE id = %s
            """, (document_details_json, extracted_fields_json, file_details, overall_score, exists[0]))
        else:
            cursor.execute("""
                INSERT INTO ProcessInstanceDocuments
                (processInstancesId, documentDetails, extractedFields, fileDetails, overAllScore, createdAt, updatedAt, isActive, isDeleted)
                VALUES (%s, %s, %s, %s, %s, NOW(), NOW(), 1, 0)
            """, (pid, document_details_json, extracted_fields_json, file_details, overall_score))

        conn.commit()
        print("✅ Saved extractedFields and fileDetails to DB")
        log_to_mongo(process_instance_id, message = f"Saved extractedFields and fileDetails to DB", node_name = "Validation", log_type=2)
        print(f"✅ Stored metadata for file with avg score {overall_score}%")
        log_to_mongo(process_instance_id, message = f"Stored metadata for file with avg score {overall_score}%", node_name = "Validation", log_type=2)

    if AUTO_EXECUTE_NEXT_NODE == 1 and not is_orchestrated:
        token = get_auth_token()
        trigger_url = f"{AIRFLOW_API_URL}/dags/deliver_dag/dagRuns"
        run_id = f"triggered_by_validate_human_{process_instance_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        payload = {
            "dag_run_id": run_id,
            "logical_date": datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
            "conf": {"id": process_instance_id}
        }
        requests.post(trigger_url, json=payload, headers=headers, timeout=10)

with DAG(
    dag_id="highlight_extracted_fields_dag",
    start_date=datetime.now() - timedelta(days=1),
    schedule=None,
    catchup=False,
    tags=["idp", "highlighting"]
) as dag:

    highlight_task = PythonOperator(
        task_id="highlight_extracted_fields",
        python_callable=highlight_and_upload,
    )

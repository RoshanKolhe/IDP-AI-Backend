from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.mysql.hooks.mysql import MySqlHook
from datetime import datetime, timedelta
from openai import OpenAI
from opik.integrations.openai import track_openai
import os
import json
import pytesseract
import requests
from pdf2image import convert_from_path
from airflow.utils.log.logging_mixin import LoggingMixin
import re
from PyPDF2 import PdfReader
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import signal
from contextlib import contextmanager
from sentence_transformers import SentenceTransformer
log = LoggingMixin().log

from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv() 

AUTO_EXECUTE_NEXT_NODE = 1

# === DAG Trigger CONFIG === #
AIRFLOW_API_URL = "http://airflow-airflow-apiserver-1:8080/api/v2"  # or localhost in local mode
AIRFLOW_USERNAME = os.getenv("AIRFLOW_USERNAME")
AIRFLOW_PASSWORD = os.getenv("AIRFLOW_PASSWORD")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # from .env
OpenAI.api_key = OPENAI_API_KEY
OPIK_API_KEY = os.getenv("OPIK_API_KEY")  
LOCAL_MODE = os.getenv("LOCAL_MODE", "false").lower() == "true"
MAX_PAGES_TO_SCAN = 100

if LOCAL_MODE:
    AIRFLOW_API_URL = "http://localhost:8080/api/v2"

# === CONFIG ===
LOCAL_DOWNLOAD_DIR = "/opt/airflow/downloaded_docs"
ML_MODELS_DIR = "/opt/airflow/dags/ml_models"
MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB_NAME = "idp"
MONGO_COLLECTION = "LogEntry"
mongo_client = MongoClient(MONGO_URI)
mongo_collection = mongo_client[MONGO_DB_NAME][MONGO_COLLECTION]
openai_client = OpenAI()  
openai_client = track_openai(openai_client, project_name="my-idp-project")

if not OpenAI.api_key or not OpenAI.api_key.startswith("sk-") and not OpenAI.api_key.startswith("sk-proj-"):
    raise EnvironmentError("❌ OpenAI API key missing or invalid. Please set OPENAI_API_KEY as an environment variable.")


# ---------------- Timeout ----------------
class TimeoutException(Exception):
    pass

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

# ---------------- Load ML field vectors ----------------
try:
    VECTOR_PATH = os.path.join(ML_MODELS_DIR, "field_vectors.pkl")
    print(f"🔍 Loading ML vectors from {VECTOR_PATH}")
    with open(VECTOR_PATH, "rb") as f:
        ml_data = pickle.load(f)
    print("✅ Loaded field_vectors.pkl for ML extraction")
except Exception as e:
    print(f"❌ Failed to load field_vectors.pkl: {e}")
    ml_data = {}

ml_vectorizer = ml_data.get("tfidf_vectorizer")
ml_X_tfidf = ml_data.get("X_tfidf")
ml_labels = ml_data.get("labels", [])
ml_X_emb = ml_data.get("X_emb", None)

embedding_model_name = "all-MiniLM-L6-v2"
ml_model = None
if ml_X_emb is not None and embedding_model_name:
    try:
        ml_model = SentenceTransformer(embedding_model_name)
        print(f"✅ Embedding model {embedding_model_name} loaded for ML extraction")
    except Exception as e:
        print(f"⚠️ Failed to load embedding model: {e}")

# ---------------- Preprocess ----------------
def preprocess_text(text):
    if not text:
        return ""
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    return text.lower()

# ---------------- Extract text from PDF (ML) ----------------
def ml_extract_text_from_pdf(pdf_path, max_pages=5):
    text_content = []
    try:
        reader = PdfReader(pdf_path)
        num_pages = min(len(reader.pages), max_pages)

        for i in range(num_pages):
            text = reader.pages[i].extract_text()
            if text and text.strip():
                text_content.append(text)
            else:
                try:
                    with time_limit(30):
                        images = convert_from_path(pdf_path, first_page=i+1, last_page=i+1)
                        if images:
                            text = pytesseract.image_to_string(images[0])
                            if text.strip():
                                text_content.append(text)
                except Exception as e:
                    print(f"⚠️ OCR error on page {i+1} of {pdf_path}: {e}")
    except Exception as e:
        print(f"⚠️ Error reading {pdf_path}: {e}")

    return "\n".join(text_content)

# ---------------- ML Field Extraction ----------------
def classify_text_for_field(field_name, text, threshold=0.3):
    if not text.strip():
        return None

    candidates = re.split(r"[\n\t:;|]", text)
    candidates = [preprocess_text(c) for c in candidates if len(c.strip()) > 3]

    if not candidates:
        return None

    best_val, best_score = None, -1

    # Embedding similarity
    if ml_model and ml_X_emb is not None:
        try:
            cand_emb = ml_model.encode(candidates, convert_to_numpy=True, normalize_embeddings=True)
            sims = cosine_similarity(cand_emb, ml_X_emb)
            for i, cand in enumerate(candidates):
                for j, label in enumerate(ml_labels):
                    if label == field_name and sims[i][j] > best_score:
                        best_score = sims[i][j]
                        best_val = cand
            if best_score >= threshold:
                return best_val
        except Exception as e:
            print(f"⚠️ Embedding error for {field_name}: {e}")

    # TF–IDF fallback
    if ml_vectorizer is not None and ml_X_tfidf is not None:
        cand_tfidf = ml_vectorizer.transform(candidates)
        sims = cosine_similarity(cand_tfidf, ml_X_tfidf)
        for i, cand in enumerate(candidates):
            for j, label in enumerate(ml_labels):
                if label == field_name and sims[i][j] > best_score:
                    best_score = sims[i][j]
                    best_val = cand

    return best_val if best_score >= threshold else None

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

def log_to_mongo(process_instance_id, node_name, message, log_type=1, remark=""):
    try:
        log_entry = {
            "processInstanceId": process_instance_id,
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

def extract_text_from_pdf(pdf_path):
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(pdf_path)
        total_pages = len(reader.pages)

        texts = []
        for i in range(1, total_pages + 1):
            images = convert_from_path(pdf_path, first_page=i, last_page=i)
            if images:
                text = pytesseract.image_to_string(images[0])
                texts.append(text)
        return texts
    except Exception as e:
        print(f"❌ OCR failed for {pdf_path}: {e}")
        log_to_mongo(process_instance_id, message = f"OCR failed for {pdf_path}: {e}", node_name = "Extraction", log_type=1)
        return []

def correct_typos_with_genai(extracted_data):
    try:
        prompt = f"""
        You are a helpful assistant. The following JSON contains field names and their extracted values from OCR-scanned documents.
        Some values may contain spelling errors. Correct typos only in field values.
        Do not make unneccesary corrections in Name unless it is of english Origin.
        Do not change field names, structure or formatting. Return only corrected JSON.

        JSON:
        {json.dumps(extracted_data, indent=2)}
        """
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            timeout=60
        )
        content = response.choices[0].message.content.strip()
        if content.startswith("```json"):
            content = content.replace("```json", "").replace("```", "").strip()
        return json.loads(content)
    except Exception as e:
        print(f"❌ GenAI typo correction failed: {e}")
        log_to_mongo(process_instance_id, message = f"GenAI typo correction failed: {e}", node_name = "Extraction", log_type=1)
        return extracted_data

def extract_fields_from_documents(**context):
    # Get process instance ID from DAG run configuration
    process_instance_id = context["dag_run"].conf.get("id")
    is_orchestrated = bool(context["dag_run"].conf.get("orchestrated", False))
    if not process_instance_id:
        raise ValueError("Missing process_instance_id in dag_run.conf")
    
    process_instance_dir_path = os.path.join(LOCAL_DOWNLOAD_DIR, "process-instance-" + str(process_instance_id))
    os.makedirs(process_instance_dir_path, exist_ok=True)
    EXTRACTED_FIELDS_PATH = os.path.join(process_instance_dir_path, "extracted_fields.json")
    CLEANED_FIELDS_PATH = os.path.join(process_instance_dir_path, "cleaned_extracted_fields.json")
    CLASSIFIED_JSON_PATH = os.path.join(process_instance_dir_path, "classified_documents.json")
    BLUEPRINT_PATH = os.path.join(process_instance_dir_path, "blueprint.json")

    # MySQL connection
    hook = MySqlHook(mysql_conn_id="idp_mysql")
    conn = hook.get_conn()
    cursor = conn.cursor()

    # Load blueprint
    if not os.path.exists(BLUEPRINT_PATH):
        raise FileNotFoundError(f"❌ Missing blueprint.json at {BLUEPRINT_PATH}")
        log_to_mongo(process_instance_id, message = f"Missing blueprint.json at {BLUEPRINT_PATH}", node_name = "Extraction", log_type=1)
    with open(BLUEPRINT_PATH, "r") as f:
        blueprint = json.load(f)

    # Load classification results
    if not os.path.exists(CLASSIFIED_JSON_PATH):
        raise FileNotFoundError("❌ classified_documents.json not found.")
        log_to_mongo(process_instance_id, message = f"classified_documents.json not found.", node_name = "Extraction", log_type=1)
    with open(CLASSIFIED_JSON_PATH, "r") as f:
        classified_docs = json.load(f)

    # Update current stage
    cursor.execute("""
        UPDATE ProcessInstances
        SET currentStage = %s, isInstanceRunning = %s, updatedAt = NOW()
        WHERE id = %s
    """, ("Extraction", 1, process_instance_id))
    conn.commit()


    extract_node = next((n for n in blueprint if n["nodeName"].lower() == "extract"), None)
    if not extract_node:
        raise ValueError("❌ No extract node found in blueprint.")
        log_to_mongo(process_instance_id, message = f"No extract node found in blueprint.", node_name = "Extraction", log_type=1)

    rules = extract_node["component"]
    categories = {c["documentType"].lower(): c["id"] for c in rules["categories"]}
    extractors = rules["extractors"]
    extractor_fields = rules["extractorFields"]

    structured_results = []

    for file_name, doc_type in classified_docs.items():
        doc_path = os.path.join(process_instance_dir_path, file_name)
        if not os.path.exists(doc_path):
            print(f"⚠️ File not found: {file_name}")
            log_to_mongo(process_instance_id, message = f"File not found: {file_name}", node_name = "Extraction", log_type=3)
            continue

        doc_type_lower = doc_type.lower()
        doc_type_id = categories.get(doc_type_lower)
        if not doc_type_id or str(doc_type_id) not in extractor_fields:
            print(f"⚠️ No extraction rules for {doc_type}")
            log_to_mongo(process_instance_id, message = f"No extraction rules for {doc_type}", node_name = "Extraction", log_type=3)
            continue

        ocr_text = extract_text_from_pdf(doc_path)
        field_prompts = extractor_fields[str(doc_type_id)]
        extracted = {}

        ### Choose extraction method based on config ###
        extractor_method = extractors.get(str(doc_type_id), "genai").lower()

        if extractor_method == "ml":
            # Run ML-based extraction
            try:
                print(f"🤖 Using ML extractor for {file_name} ({doc_type})")
                ocr_text_full = ml_extract_text_from_pdf(doc_path, max_pages=5)
                for field in field_prompts:
                    field_label = field["field_to_extract"]
                    field_var = field["variableName"]

                    val = classify_text_for_field(field_label, ocr_text_full, threshold=0.3)
                    extracted[field_var] = val if val else "Not Found"
            except Exception as e:
                print(f"❌ ML extraction failed for {file_name}: {e}")
                log_to_mongo(process_instance_id, message = f"ML extraction failed for {file_name}: {e}", node_name = "Extraction", log_type=1)
        else:
            # Run GenAI-based extraction
            print(f"🤖 Using GenAI extractor for {file_name} ({doc_type})")
            for field in field_prompts:
                field_name = field["variableName"]
                inp_prompt = field["prompt"]
                extracted_value = "N/A"
                for page_num in range(1, MAX_PAGES_TO_SCAN + 1):
                    try:
                        images = convert_from_path(doc_path, first_page=page_num, last_page=page_num)
                        if not images:
                            continue
                        page_image = images[0]
                        page_text = pytesseract.image_to_string(page_image)
                        
                        if not inp_prompt:
                            prompt = f"""
                                The following is OCR-extracted text (Page {page_num} of the document). 
                                Extract the value for field: "{field_name}".
                                Return only the value without any additional text or explanation. If not found, return "N/A".
				If the value is numeric, do not ever include the value in words. e.g. If value is Fifty, return '50' and not 'Fifty'.
                                when value is supposed to be in multiple outputs, create an array of objects. 

                                Text:
                                {page_text[:1500]}
                                                """
                        else:
                            prompt = f"{inp_prompt}\nText:\n{page_text[:1500]}"

                        print("Prompt Used: " + prompt)

                        log.info(f"🔍 Searching {field_name} from page {page_num} of {file_name}")
                        response = openai_client.chat.completions.create(
                            model="gpt-4o",
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.2,
                            timeout=30
                        )

                        value = response.choices[0].message.content.strip()
                        extracted[field_name] = value

                        if value and value.upper() != "N/A":
                            break  # ✅ Stop once value is found

                    except Exception as e:
                        print(f"⚠️ Error extracting {field_name} from page {page_num}: {e}")
                        log_to_mongo(process_instance_id, message = f"Error extracting {field_name} from page {page_num}: {e}", node_name = "Extraction", log_type=1)
                        extracted[field_name] = f"Error: {e}"
                        break

        # Compose final JSON structure
        structured_results.append({
            "documentDetails": {
                "documentName": file_name,
                "documentType": doc_type
            },
            "extractedFields": extracted,
            "processInstanceId": process_instance_id
        }) 

    # Save raw extraction result
    with open(EXTRACTED_FIELDS_PATH, "w") as f:
        json.dump(structured_results, f, indent=2)
    print(f"✅ Saved raw extracted_fields.json")
    log_to_mongo(process_instance_id, message = f"Saved raw extracted_fields.json", node_name = "Extraction", log_type=2)

    # Run GenAI-based typo correction
    cleaned_data = correct_typos_with_genai(structured_results)
    with open(CLEANED_FIELDS_PATH, "w") as f:
        json.dump(cleaned_data, f, indent=2)
    print(f"✅ Saved cleaned_extracted_fields.json")
    log_to_mongo(process_instance_id, message = f"Saved cleaned_extracted_fields.json", node_name = "Extraction", log_type=2)

    # Trigger validate_documents_dag
    if AUTO_EXECUTE_NEXT_NODE == 1 and not is_orchestrated:
        print("🚀 Triggering validate_fields_dag...")
        token = get_auth_token()
        trigger_url = f"{AIRFLOW_API_URL}/dags/highlight_extracted_fields_dag/dagRuns"
        run_id = f"triggered_by_extraction_{process_instance_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
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
        print(f"✅ Successfully triggered validate_fields_dag with ID {process_instance_id}")
        log_to_mongo(process_instance_id, message = f"Successfully triggered validate_fields_dag with ID {process_instance_id}", node_name = "Extraction", log_type=2)


# === DAG DEFINITION ===
with DAG(
    dag_id="extract_documents_dag",
    start_date=datetime.now() - timedelta(days=1),
    schedule=None,
    catchup=False,
    tags=["idp", "extraction"]
) as dag:

    extract_task = PythonOperator(
        task_id="extract_fields_from_documents",
        python_callable=extract_fields_from_documents
    )

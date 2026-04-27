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
from transaction_status import sync_stage_status
import ast
from ocr_services.ocr_cache_utils import ensure_ocr_cache, get_cached_document_text, get_cached_page_texts

log = LoggingMixin().log

from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

AUTO_EXECUTE_NEXT_NODE = 1

# === DAG Trigger CONFIG === #
AIRFLOW_API_URL = "http://airflow-airflow-apiserver-1:8080/api/v2"
AIRFLOW_USERNAME = os.getenv("AIRFLOW_USERNAME")
AIRFLOW_PASSWORD = os.getenv("AIRFLOW_PASSWORD")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OpenAI.api_key = OPENAI_API_KEY
OPIK_API_KEY = os.getenv("OPIK_API_KEY")
LOCAL_MODE = os.getenv("LOCAL_MODE", "false").lower() == "true"
MAX_PAGES_TO_SCAN = 20

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
rapidocr_engine = None


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


if not OpenAI.api_key or (
    not OpenAI.api_key.startswith("sk-") and not OpenAI.api_key.startswith("sk-proj-")
):
    raise EnvironmentError("OpenAI API key missing or invalid. Please set OPENAI_API_KEY.")


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


try:
    VECTOR_PATH = os.path.join(ML_MODELS_DIR, "field_vectors.pkl")
    print(f"Loading ML vectors from {VECTOR_PATH}")
    with open(VECTOR_PATH, "rb") as f:
        ml_data = pickle.load(f)
    print("Loaded field_vectors.pkl for ML extraction")
except Exception as e:
    print(f"Failed to load field_vectors.pkl: {e}")
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
        print(f"Embedding model {embedding_model_name} loaded for ML extraction")
    except Exception as e:
        print(f"Failed to load embedding model: {e}")


def preprocess_text(text):
    if not text:
        return ""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text.lower()


def preprocess_image(image):
    import cv2

    img = np.array(image)
    if len(img.shape) == 2:
        gray = img
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.equalizeHist(gray)
    blur = cv2.medianBlur(gray, 3)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


def rapidocr_extract(image):
    global rapidocr_engine

    if rapidocr_engine is None:
        from rapidocr_onnxruntime import RapidOCR

        rapidocr_engine = RapidOCR()

    result, _ = rapidocr_engine(np.array(image))
    if not result:
        return ""

    return " ".join(item[1] for item in result if len(item) > 1 and item[1])


def perform_ocr_on_image(image):
    processed_image = preprocess_image(image)
    tesseract_text = pytesseract.image_to_string(
        processed_image,
        config="--oem 3 --psm 6"
    )

    if len(preprocess_text(tesseract_text)) >= 50:
        return tesseract_text

    rapidocr_text = rapidocr_extract(processed_image)
    if len(preprocess_text(rapidocr_text)) > len(preprocess_text(tesseract_text)):
        return rapidocr_text

    return tesseract_text


def normalize_extracted_value(value):
    if not isinstance(value, str):
        return value

    normalized = re.sub(r"\s+", " ", value).strip()
    if re.fullmatch(r"[\d,\.\- ]+", normalized):
        normalized = normalized.replace(",", "")
        normalized = normalized.replace(" ", "")
    return normalized


def normalize_extracted_payload(payload):
    if isinstance(payload, dict):
        return {key: normalize_extracted_payload(val) for key, val in payload.items()}
    if isinstance(payload, list):
        return [normalize_extracted_payload(item) for item in payload]
    return normalize_extracted_value(payload)


def page_contains_relevant_keywords(page_text, field_prompts, doc_type):
    keywords = {"agreement", "contract", "amount", "date", "ref"}
    keywords.add((doc_type or "").lower())

    for field in field_prompts:
        for key in [field.get("variableName", ""), field.get("field_to_extract", "")]:
            for token in re.split(r"[^a-zA-Z0-9]+", key.lower()):
                if len(token) > 2:
                    keywords.add(token)

    page_text_lower = page_text.lower()
    return any(keyword and keyword in page_text_lower for keyword in keywords)


def is_table_like_page(page_text):
    lines = [line.strip() for line in page_text.splitlines() if line.strip()]
    if not lines:
        return False

    numeric_tokens = re.findall(r"\b\d+(?:[.,]\d+)?\b", page_text)
    lines_with_many_numbers = sum(1 for line in lines if len(re.findall(r"\d+", line)) >= 3)
    separator_lines = sum(1 for line in lines if "|" in line or "\t" in line)

    return (
        len(numeric_tokens) >= 25 and lines_with_many_numbers >= 5
    ) or separator_lines >= 4


def parse_json_response(content):
    cleaned = content.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned.replace("```json", "", 1).replace("```", "").strip()
    elif cleaned.startswith("```"):
        cleaned = cleaned.replace("```", "").strip()

    try:
        return json.loads(cleaned)
    except Exception:
        try:
            return ast.literal_eval(cleaned)
        except Exception:
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            raise


def build_multi_field_prompt(doc_type, field_prompts, page_text):
    field_lines = []
    for field in field_prompts:
        variable_name = field.get("variableName", "")
        field_label = field.get("field_to_extract", variable_name)
        field_prompt = field.get("prompt", "")
        if field_prompt:
            field_lines.append(f'- "{variable_name}": {field_prompt}')
        else:
            field_lines.append(f'- "{variable_name}": extract "{field_label}"')

    fields_block = "\n".join(field_lines)

    return f"""
This is a Work Order / Contract document for document type "{doc_type}".
Extract relevant structured fields carefully.

Extract the following fields from the document text.

Fields:
{fields_block}

Rules:
- Return JSON only.
- Use the variable names exactly as keys.
- If a field is not found, return "N/A".
- If a value is numeric, return digits only when possible.
- Do not infer BOQ/table rows or fabricate values.

Text:
{page_text[:4000]}
"""


def extract_all_fields_from_page(doc_type, field_prompts, page_text):
    prompt = build_multi_field_prompt(doc_type, field_prompts, page_text)
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        timeout=30,
    )
    return parse_json_response(response.choices[0].message.content.strip())


def ml_extract_text_from_pdf(pdf_path, max_pages=5):
    process_instance_dir = os.path.dirname(pdf_path)
    cached_text = get_cached_document_text(process_instance_dir, os.path.basename(pdf_path))
    if cached_text and cached_text.strip():
        return cached_text

    cache_payload = ensure_ocr_cache(
        pdf_path=pdf_path,
        process_instance_dir=process_instance_dir,
        ocr_engine="paddle",
        config={"dpi": 300, "last_page": max_pages},
    )
    cached_text = cache_payload.get("cleaned_text") or cache_payload.get("raw_text") or ""
    if cached_text.strip():
        return cached_text

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
                        images = convert_from_path(pdf_path, first_page=i + 1, last_page=i + 1)
                        if images:
                            text = perform_ocr_on_image(images[0])
                            if text.strip():
                                text_content.append(text)
                except Exception as e:
                    print(f"OCR error on page {i + 1} of {pdf_path}: {e}")
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")

    return "\n".join(text_content)


def classify_text_for_field(field_name, text, threshold=0.3):
    if not text.strip():
        return None

    candidates = re.split(r"[\n\t:;|]", text)
    candidates = [preprocess_text(c) for c in candidates if len(c.strip()) > 3]

    if not candidates:
        return None

    best_val, best_score = None, -1

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
            print(f"Embedding error for {field_name}: {e}")

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
    auth_url = f"{AIRFLOW_API_URL.replace('/api/v2', '')}/auth/token"
    response = requests.post(
        auth_url,
        json={"username": AIRFLOW_USERNAME, "password": AIRFLOW_PASSWORD},
        headers={"Content-Type": "application/json"},
        timeout=10,
    )
    response.raise_for_status()
    return response.json()["access_token"]


def log_to_mongo(process_instance_id, node_name, message, log_type=1, remark=""):
    try:
        log_entry = {
            "processInstanceId": process_instance_id,
            "processInstanceTransactionId": _get_transaction_id(process_instance_id),
            "nodeName": node_name,
            "logsDescription": message,
            "logType": log_type,
            "isDeleted": False,
            "isActive": True,
            "remark": remark,
            "createdAt": datetime.utcnow(),
        }
        mongo_collection.insert_one(log_entry)
        print(f"Logged to MongoDB: {message}")
    except Exception as mongo_err:
        print(f"Failed to log to MongoDB: {mongo_err}")


def extract_text_from_pdf(pdf_path):
    process_instance_dir = os.path.dirname(pdf_path)
    cached_page_texts = get_cached_page_texts(process_instance_dir, os.path.basename(pdf_path))
    if cached_page_texts:
        return cached_page_texts[:MAX_PAGES_TO_SCAN]

    cache_payload = ensure_ocr_cache(
        pdf_path=pdf_path,
        process_instance_dir=process_instance_dir,
        ocr_engine="paddle",
        config={"dpi": 300, "last_page": MAX_PAGES_TO_SCAN},
    )
    generated_page_texts = [page.get("cleaned_text") or page.get("text", "") for page in cache_payload.get("pages", [])]
    if generated_page_texts:
        return generated_page_texts[:MAX_PAGES_TO_SCAN]

    try:
        reader = PdfReader(pdf_path)
        total_pages = min(len(reader.pages), MAX_PAGES_TO_SCAN)

        texts = []
        for i in range(1, total_pages + 1):
            images = convert_from_path(pdf_path, first_page=i, last_page=i)
            if images:
                texts.append(perform_ocr_on_image(images[0]))
        return texts
    except Exception as e:
        print(f"OCR failed for {pdf_path}: {e}")
        return []


def correct_typos_with_genai(extracted_data):
    try:
        prompt = f"""
        You are a helpful assistant. The following JSON contains field names and their extracted values from OCR-scanned documents.
        Some values may contain spelling errors. Correct typos only in field values.
        Do not make unnecessary corrections in Name unless it is of English origin.
        Do not change field names, structure or formatting. Return only corrected JSON.

        JSON:
        {json.dumps(extracted_data, indent=2)}
        """
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            timeout=60,
        )
        content = response.choices[0].message.content.strip()
        corrected = parse_json_response(content)
        return normalize_extracted_payload(corrected)
    except Exception as e:
        print(f"GenAI typo correction failed: {e}")
        return extracted_data


def extract_fields_from_documents(**context):
    process_instance_id = context["dag_run"].conf.get("id")
    is_orchestrated = bool(context["dag_run"].conf.get("orchestrated", False))
    if not process_instance_id:
        raise ValueError("Missing process_instance_id in dag_run.conf")

    process_instance_dir_path = os.path.join(LOCAL_DOWNLOAD_DIR, f"process-instance-{process_instance_id}")
    os.makedirs(process_instance_dir_path, exist_ok=True)
    extracted_fields_path = os.path.join(process_instance_dir_path, "extracted_fields.json")
    cleaned_fields_path = os.path.join(process_instance_dir_path, "cleaned_extracted_fields.json")
    classified_json_path = os.path.join(process_instance_dir_path, "classified_documents.json")
    blueprint_path = os.path.join(process_instance_dir_path, "blueprint.json")

    hook = MySqlHook(mysql_conn_id="idp_mysql")
    conn = hook.get_conn()
    cursor = conn.cursor()
    try:
        if not os.path.exists(blueprint_path):
            raise FileNotFoundError(f"Missing blueprint.json at {blueprint_path}")
        with open(blueprint_path, "r", encoding="utf-8") as f:
            blueprint = json.load(f)

        if not os.path.exists(classified_json_path):
            raise FileNotFoundError("classified_documents.json not found.")
        with open(classified_json_path, "r", encoding="utf-8") as f:
            classified_docs = json.load(f)

        sync_stage_status(cursor, process_instance_id, "Extraction", 1)
        conn.commit()

        extract_node = next((n for n in blueprint if n["nodeName"].lower() == "extract"), None)
        if not extract_node:
            raise ValueError("No extract node found in blueprint.")

        rules = extract_node["component"]
        categories = {c["documentType"].lower(): c["id"] for c in rules["categories"]}
        extractors = rules["extractors"]
        extractor_fields = rules["extractorFields"]

        structured_results = []
        total_api_calls = 0
        total_skipped_pages = 0

        for file_name, doc_type in classified_docs.items():
            doc_path = os.path.join(process_instance_dir_path, file_name)
            if not os.path.exists(doc_path):
                print(f"File not found: {file_name}")
                log_to_mongo(process_instance_id, message=f"File not found: {file_name}", node_name="Extraction", log_type=3)
                continue

            doc_type_lower = doc_type.lower()
            doc_type_id = categories.get(doc_type_lower)
            if not doc_type_id or str(doc_type_id) not in extractor_fields:
                print(f"No extraction rules for {doc_type}")
                log_to_mongo(process_instance_id, message=f"No extraction rules for {doc_type}", node_name="Extraction", log_type=3)
                continue

            field_prompts = extractor_fields[str(doc_type_id)]
            extracted = {field["variableName"]: "N/A" for field in field_prompts}
            extractor_method = extractors.get(str(doc_type_id), "genai").lower()

            if extractor_method == "ml":
                try:
                    print(f"Using ML extractor for {file_name} ({doc_type})")
                    ocr_text_full = ml_extract_text_from_pdf(doc_path, max_pages=MAX_PAGES_TO_SCAN)
                    print(f"ML OCR text length for {file_name}: {len(ocr_text_full)}")
                    log_to_mongo(
                        process_instance_id,
                        message=f"ML OCR text length for {file_name}: {len(ocr_text_full)}",
                        node_name="Extraction",
                        log_type=0,
                    )

                    for field in field_prompts:
                        field_label = field["field_to_extract"]
                        field_var = field["variableName"]
                        val = classify_text_for_field(field_label, ocr_text_full, threshold=0.3)
                        extracted[field_var] = normalize_extracted_value(val) if val else "Not Found"
                except Exception as e:
                    print(f"ML extraction failed for {file_name}: {e}")
                    log_to_mongo(process_instance_id, message=f"ML extraction failed for {file_name}: {e}", node_name="Extraction", log_type=1)
            else:
                print(f"Using GenAI extractor for {file_name} ({doc_type})")
                page_texts = extract_text_from_pdf(doc_path)
                remaining_fields = {field["variableName"] for field in field_prompts}

                for page_num, page_text in enumerate(page_texts, start=1):
                    if page_num > MAX_PAGES_TO_SCAN or not remaining_fields:
                        break

                    print(f"OCR text length for {file_name} page {page_num}: {len(page_text)}")
                    log_to_mongo(
                        process_instance_id,
                        message=f"OCR text length for {file_name} page {page_num}: {len(page_text)}",
                        node_name="Extraction",
                        log_type=0,
                    )

                    if not page_contains_relevant_keywords(page_text, field_prompts, doc_type):
                        total_skipped_pages += 1
                        print(f"Skipping page {page_num} of {file_name}: no relevant keywords")
                        log_to_mongo(
                            process_instance_id,
                            message=f"Skipped page {page_num} of {file_name}: no relevant keywords",
                            node_name="Extraction",
                            log_type=3,
                        )
                        continue

                    if is_table_like_page(page_text):
                        total_skipped_pages += 1
                        print(f"Skipping page {page_num} of {file_name}: table/BOQ-like page")
                        log_to_mongo(
                            process_instance_id,
                            message=f"Skipped page {page_num} of {file_name}: table/BOQ-like page",
                            node_name="Extraction",
                            log_type=3,
                        )
                        continue

                    current_fields = [field for field in field_prompts if field["variableName"] in remaining_fields]
                    if not current_fields:
                        break

                    try:
                        total_api_calls += 1
                        extracted_payload = extract_all_fields_from_page(doc_type, current_fields, page_text)
                        extracted_payload = normalize_extracted_payload(extracted_payload)

                        for field in current_fields:
                            field_name = field["variableName"]
                            value = extracted_payload.get(field_name, "N/A") if isinstance(extracted_payload, dict) else "N/A"
                            if isinstance(value, str) and value.upper() == "N/A":
                                continue
                            if value in [None, "", [], {}]:
                                continue
                            extracted[field_name] = value
                            remaining_fields.discard(field_name)

                        log_to_mongo(
                            process_instance_id,
                            message=f"GenAI extraction API call #{total_api_calls} completed for {file_name} page {page_num}",
                            node_name="Extraction",
                            log_type=0,
                        )
                    except Exception as e:
                        print(f"Error extracting fields from page {page_num} of {file_name}: {e}")
                        log_to_mongo(
                            process_instance_id,
                            message=f"Error extracting fields from page {page_num} of {file_name}: {e}",
                            node_name="Extraction",
                            log_type=1,
                        )

                if not remaining_fields:
                    print(f"All fields found early for {file_name}, stopping page scan")
                    log_to_mongo(
                        process_instance_id,
                        message=f"All fields found early for {file_name}",
                        node_name="Extraction",
                        log_type=2,
                    )

            structured_results.append(
                {
                    "documentDetails": {
                        "documentName": file_name,
                        "documentType": doc_type,
                    },
                    "extractedFields": extracted,
                    "processInstanceId": process_instance_id,
                }
            )

        with open(extracted_fields_path, "w", encoding="utf-8") as f:
            json.dump(structured_results, f, indent=2, ensure_ascii=False)
        print("Saved raw extracted_fields.json")
        log_to_mongo(process_instance_id, message="Saved raw extracted_fields.json", node_name="Extraction", log_type=2)

        cleaned_data = correct_typos_with_genai(structured_results)
        with open(cleaned_fields_path, "w", encoding="utf-8") as f:
            json.dump(cleaned_data, f, indent=2, ensure_ascii=False)
        print("Saved cleaned_extracted_fields.json")
        log_to_mongo(process_instance_id, message="Saved cleaned_extracted_fields.json", node_name="Extraction", log_type=2)

        print(f"Extraction summary: API calls={total_api_calls}, skipped_pages={total_skipped_pages}")
        log_to_mongo(
            process_instance_id,
            message=f"Extraction summary: API calls={total_api_calls}, skipped_pages={total_skipped_pages}",
            node_name="Extraction",
            log_type=0,
        )

        if AUTO_EXECUTE_NEXT_NODE == 1 and not is_orchestrated:
            print("Triggering validate_fields_dag...")
            token = get_auth_token()
            trigger_url = f"{AIRFLOW_API_URL}/dags/highlight_extracted_fields_dag/dagRuns"
            run_id = f"triggered_by_extraction_{process_instance_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            }
            payload = {
                "dag_run_id": run_id,
                "logical_date": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                "conf": {"id": process_instance_id},
            }

            response = requests.post(trigger_url, json=payload, headers=headers, timeout=10)
            response.raise_for_status()
            print(f"Successfully triggered validate_fields_dag with ID {process_instance_id}")
            log_to_mongo(
                process_instance_id,
                message=f"Successfully triggered validate_fields_dag with ID {process_instance_id}",
                node_name="Extraction",
                log_type=2,
            )
    finally:
        cursor.close()
        conn.close()


with DAG(
    dag_id="extract_documents_dag",
    start_date=datetime.now() - timedelta(days=1),
    schedule=None,
    catchup=False,
    tags=["idp", "extraction"],
) as dag:

    extract_task = PythonOperator(
        task_id="extract_fields_from_documents",
        python_callable=extract_fields_from_documents,
    )

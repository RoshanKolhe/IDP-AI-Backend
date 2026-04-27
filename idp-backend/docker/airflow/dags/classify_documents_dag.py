from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.mysql.hooks.mysql import MySqlHook
from datetime import datetime, timedelta
import os
import re
import json
from openai import OpenAI
from opik.integrations.openai import track_openai
import requests
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from pymongo import MongoClient
import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from transaction_status import sync_stage_status
from ocr_services.ocr_service_factory import get_ocr_service
from ocr_services.ocr_cache_utils import ensure_ocr_cache, get_cached_document_text, get_cached_page_texts

load_dotenv()

AUTO_EXECUTE_NEXT_NODE = 1
MONGO_URI = os.getenv("MONGO_URI")

# === DAG Trigger CONFIG === #
AIRFLOW_API_URL = "http://airflow-airflow-apiserver-1:8080/api/v2"
AIRFLOW_USERNAME = os.getenv("AIRFLOW_USERNAME")
AIRFLOW_PASSWORD = os.getenv("AIRFLOW_PASSWORD")
OPIK_API_KEY = os.getenv("OPIK_API_KEY")
OPIK_PROJECT_NAME = "idp-classification"
LOCAL_MODE = os.getenv("LOCAL_MODE", "false").lower() == "true"

if LOCAL_MODE:
    AIRFLOW_API_URL = "http://localhost:8080/api/v2"

# === CONFIG ===
LOCAL_DOWNLOAD_DIR = "/opt/airflow/downloaded_docs"
ML_MODELS_DIR = "/opt/airflow/dags/ml_models"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OpenAI.api_key = OPENAI_API_KEY
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


if not OpenAI.api_key or (
    not OpenAI.api_key.startswith("sk-") and not OpenAI.api_key.startswith("sk-proj-")
):
    raise EnvironmentError("OpenAI API key missing or invalid. Please set OPENAI_API_KEY.")

# ============================
# ML CLASSIFICATION HELPERS
# ============================

_EMB_MODEL = None
_VECTOR_CACHE = None
_OCR_SERVICE = None


def _normalize_text(text: str) -> str:
    text = text or ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _get_ocr_service():
    global _OCR_SERVICE
    if _OCR_SERVICE is None:
        _OCR_SERVICE = get_ocr_service("paddle")
    return _OCR_SERVICE


def _get_pdf_page_count(file_path: str) -> int:
    try:
        reader = PdfReader(file_path)
        return len(reader.pages)
    except Exception:
        return 0


def _get_classification_ocr_config(component=None, **overrides):
    component = component or {}
    config = {
        "language_mode": component.get("language_mode", "auto"),
        "psm": component.get("psm", 3),
        "oem": component.get("oem", 3),
        "dpi": component.get("dpi", 300),
        "thread_count": component.get("thread_count", 2),
    }
    config.update({key: value for key, value in overrides.items() if value is not None})
    return config


def _extract_pdf_text_ml(file_path: str, component=None, max_pages=None) -> str:
    """
    Extract text for ML classification using PDF-native text first and
    Tesseract OCR fallback. By default, all pages are scanned.
    """
    process_instance_dir = os.path.dirname(file_path)
    cached_text = get_cached_document_text(process_instance_dir, os.path.basename(file_path))
    if cached_text and cached_text.strip():
        return _normalize_text(cached_text)

    cache_payload = ensure_ocr_cache(
        pdf_path=file_path,
        process_instance_dir=process_instance_dir,
        ocr_engine=(component or {}).get("ocr_engine", "paddle"),
        config=_get_classification_ocr_config(component, last_page=max_pages or None),
    )
    cached_text = cache_payload.get("cleaned_text") or cache_payload.get("raw_text") or ""
    if cached_text.strip():
        return _normalize_text(cached_text)

    text_parts = []
    total_pages = _get_pdf_page_count(file_path)
    pages_to_scan = total_pages if max_pages is None else min(total_pages, max_pages)
    ocr_service = _get_ocr_service()

    try:
        reader = PdfReader(file_path)
        for i in range(pages_to_scan):
            try:
                page_text = reader.pages[i].extract_text() or ""
            except Exception:
                page_text = ""

            if page_text.strip():
                text_parts.append(page_text)
                continue

            try:
                ocr_text = ocr_service.extract_text(
                    file_path,
                    _get_classification_ocr_config(
                        component,
                        first_page=i + 1,
                        last_page=i + 1,
                    ),
                ) or ""
                if ocr_text.strip():
                    text_parts.append(ocr_text)
            except Exception:
                pass
    except Exception:
        try:
            ocr_text = ocr_service.extract_text(
                file_path,
                _get_classification_ocr_config(
                    component,
                    last_page=pages_to_scan or None,
                ),
            ) or ""
            if ocr_text.strip():
                text_parts.append(ocr_text)
        except Exception:
            pass

    return _normalize_text("\n".join(text_parts))


def _get_embedding_model():
    global _EMB_MODEL
    if _EMB_MODEL is None:
        _EMB_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _EMB_MODEL


def _load_vector_assets(base_dir: str):
    """
    Load and cache vector PKLs from process-instance folder (preferred) or CWD.
    Required: tfidf_vectors.pkl + vectorizer.pkl
    Optional: embeddings.pkl
    """
    global _VECTOR_CACHE
    if _VECTOR_CACHE and _VECTOR_CACHE.get("base_dir") == base_dir:
        return _VECTOR_CACHE

    search_dirs = [base_dir, os.getcwd()]
    print("ML models Dir:", ML_MODELS_DIR)
    tfidf_pkl = embeddings_pkl = vectorizer_pkl = None

    for _ in search_dirs:
        t = os.path.join(ML_MODELS_DIR, "classify_tfidf_vectors.pkl")
        e = os.path.join(ML_MODELS_DIR, "classify_embeddings.pkl")
        v = os.path.join(ML_MODELS_DIR, "classify_vectorizer.pkl")
        if os.path.exists(t) and os.path.exists(v):
            tfidf_pkl = t
            vectorizer_pkl = v
        if os.path.exists(e):
            embeddings_pkl = e

    if not tfidf_pkl or not vectorizer_pkl:
        raise FileNotFoundError(
            "Missing TF-IDF assets. Expecting classify_tfidf_vectors.pkl and "
            "classify_vectorizer.pkl in the ML models directory."
        )

    tfidf_data = joblib.load(tfidf_pkl)
    vectorizer = joblib.load(vectorizer_pkl)
    embeddings_data = joblib.load(embeddings_pkl) if embeddings_pkl else None

    _VECTOR_CACHE = {
        "base_dir": base_dir,
        "tfidf_data": tfidf_data,
        "vectorizer": vectorizer,
        "embeddings_data": embeddings_data,
        "embeddings_pkl": embeddings_pkl,
    }
    return _VECTOR_CACHE


def classify_document_ml(
    file_path: str,
    base_dir: str,
    component=None,
    target_labels=None,
    threshold_embed: float = 0.35,
    threshold_tfidf: float = 0.25,
    max_pages=None,
) -> str:
    """
    Classify a PDF using prebuilt vectors (embeddings -> TF-IDF fallback).
    Returns a label or "Unknown".
    """
    try:
        assets = _load_vector_assets(base_dir)
    except Exception as e:
        print(f"ML assets load failed: {e}")
        return "Unknown"

    tfidf_data = assets["tfidf_data"]
    vectorizer = assets["vectorizer"]
    embeddings_data = assets["embeddings_data"]

    text = _extract_pdf_text_ml(file_path, component=component, max_pages=max_pages)
    if not text.strip():
        return "Unknown"

    if embeddings_data:
        try:
            emb_labels = embeddings_data["labels"]
            emb_vecs = embeddings_data["vectors"]
            model = _get_embedding_model()
            qvec = model.encode([text], convert_to_numpy=True)
            sims = cosine_similarity(qvec, emb_vecs)[0]
            idx = int(np.argmax(sims))
            score = float(sims[idx])
            print(f"{os.path.basename(file_path)} | Embedding similarity = {score:.3f}")

            if score >= threshold_embed:
                pred = emb_labels[idx]
                if (not target_labels) or (pred in target_labels):
                    return pred
                print(f"Pred '{pred}' not in blueprint categories; trying TF-IDF fallback.")
        except Exception as e:
            print(f"Embedding stage failed: {e} - falling back to TF-IDF")

    try:
        tf_labels = tfidf_data["labels"]
        tf_vecs = tfidf_data["vectors"]
        qtf = vectorizer.transform([text]).toarray()
        sims = cosine_similarity(qtf, tf_vecs)[0]
        idx = int(np.argmax(sims))
        score = float(sims[idx])
        print(f"{os.path.basename(file_path)} | TF-IDF similarity = {score:.3f}")

        if score >= threshold_tfidf:
            pred = tf_labels[idx]
            if (not target_labels) or (pred in target_labels):
                return pred
    except Exception as e:
        print(f"TF-IDF stage failed: {e}")

    print(f"{os.path.basename(file_path)} -> Unknown")
    return "Unknown"


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


def extract_text_per_page(pdf_path, component=None, max_pages=None):
    try:
        process_instance_dir = os.path.dirname(pdf_path)
        cached_page_texts = get_cached_page_texts(process_instance_dir, os.path.basename(pdf_path))
        if cached_page_texts:
            page_limit = len(cached_page_texts) if max_pages is None else min(len(cached_page_texts), max_pages)
            for page_text in cached_page_texts[:page_limit]:
                yield page_text
            return

        cache_payload = ensure_ocr_cache(
            pdf_path=pdf_path,
            process_instance_dir=process_instance_dir,
            ocr_engine=(component or {}).get("ocr_engine", "paddle"),
            config=_get_classification_ocr_config(component, last_page=max_pages or None),
        )
        generated_page_texts = [page.get("cleaned_text") or page.get("text", "") for page in cache_payload.get("pages", [])]
        if generated_page_texts:
            page_limit = len(generated_page_texts) if max_pages is None else min(len(generated_page_texts), max_pages)
            for page_text in generated_page_texts[:page_limit]:
                yield page_text
            return

        total_pages = _get_pdf_page_count(pdf_path)
        pages_to_scan = total_pages if max_pages is None else min(total_pages, max_pages)
        ocr_service = _get_ocr_service()

        for i in range(pages_to_scan):
            yield ocr_service.extract_text(
                pdf_path,
                _get_classification_ocr_config(
                    component,
                    first_page=i + 1,
                    last_page=i + 1,
                ),
            )
    except Exception as e:
        print(f"OCR failed for {pdf_path}: {e}")
        return


def classify_documents(**context):
    process_instance_id = context["dag_run"].conf.get("id")
    is_orchestrated = bool(context["dag_run"].conf.get("orchestrated", False))
    if not process_instance_id:
        raise ValueError("Missing process_instance_id in dag_run.conf")

    process_instance_dir_path = os.path.join(LOCAL_DOWNLOAD_DIR, f"process-instance-{process_instance_id}")
    os.makedirs(process_instance_dir_path, exist_ok=True)
    blueprint_path = os.path.join(process_instance_dir_path, "blueprint.json")

    hook = MySqlHook(mysql_conn_id="idp_mysql")
    conn = hook.get_conn()
    cursor = conn.cursor()
    try:
        if not os.path.exists(blueprint_path):
            raise FileNotFoundError(
                f"blueprint.json not found at {blueprint_path}. Please run ingestion DAG first."
            )

        with open(blueprint_path, "r", encoding="utf-8") as f:
            blueprint = json.load(f)

        sync_stage_status(cursor, process_instance_id, "Classification", 1)
        conn.commit()
        print(
            "ProcessInstances updated -> currentStage='Classification', "
            f"isInstanceRunning=1 for process_instance_id={process_instance_id}"
        )
        log_to_mongo(
            process_instance_id,
            message=(
                "ProcessInstances updated -> currentStage='Classification', "
                f"isInstanceRunning=1 for process_instance_id={process_instance_id}"
            ),
            node_name="Classification",
            log_type=2,
        )

        classify_node = next((n for n in blueprint if n["nodeName"].lower() == "classify"), None)
        if not classify_node:
            raise ValueError("Classification node not found in blueprint")

        classify_component = classify_node.get("component", {})
        categories = classify_component.get("categories", [])
        if not categories:
            raise ValueError("No categories found in Classify component")

        target_labels = [c["documentType"] for c in categories]
        label_str = ", ".join(target_labels)
        results = {}

        for doc_type in target_labels:
            cursor.execute("SELECT id FROM DocumentType WHERE documentType = %s", (doc_type,))
            exists = cursor.fetchone()
            if exists:
                cursor.execute(
                    """
                    UPDATE DocumentType
                    SET isActive = 1, updatedAt = NOW()
                    WHERE documentType = %s
                    """,
                    (doc_type,),
                )
            else:
                cursor.execute(
                    """
                    INSERT INTO DocumentType (documentType, isActive, createdAt)
                    VALUES (%s, 1, NOW())
                    """,
                    (doc_type,),
                )
        conn.commit()
        print("DocumentType table updated -> isActive=1 for target document types")
        log_to_mongo(
            process_instance_id,
            message="DocumentType table updated for target document types",
            node_name="Classification",
            log_type=2,
        )

        model_choice = (classify_component.get("model") or "genai").strip().lower()
        use_ml = model_choice == "ml"

        if use_ml:
            print("Using ML classifier with cached OCR / PaddleOCR fallback")
            log_to_mongo(
                process_instance_id,
                node_name="Classification",
                message="Using ML classifier with cached OCR / PaddleOCR fallback",
                log_type=0,
            )
            try:
                _load_vector_assets(process_instance_dir_path)
            except Exception as e:
                err = f"ML assets not available: {e}"
                print(err)
                log_to_mongo(process_instance_id, node_name="Classification", message=err, log_type=1)
                raise
        else:
            print("Using GENAI classifier with cached OCR / PaddleOCR fallback")
            log_to_mongo(
                process_instance_id,
                node_name="Classification",
                message="Using GENAI classifier with cached OCR / PaddleOCR fallback",
                log_type=0,
            )

        for file_name in os.listdir(process_instance_dir_path):
            if not file_name.lower().endswith(".pdf"):
                continue

            file_path = os.path.join(process_instance_dir_path, file_name)

            try:
                print(f"Classifying: {file_name}")

                if use_ml:
                    classification = classify_document_ml(
                        file_path=file_path,
                        base_dir=process_instance_dir_path,
                        component=classify_component,
                        target_labels=target_labels,
                        threshold_embed=0.35,
                        threshold_tfidf=0.25,
                        max_pages=None,
                    )
                    results[file_name] = classification
                    print(f"{file_name} -> {classification} (ML)")
                    log_to_mongo(
                        process_instance_id,
                        message=f"{file_name} -> {classification} (ML)",
                        node_name="Classification",
                        log_type=2 if classification != "Unknown" else 3,
                    )
                    continue

                accumulated_text = ""
                pages_scanned = 0
                for page_number, page_text in enumerate(
                    extract_text_per_page(file_path, component=classify_component, max_pages=None),
                    start=1,
                ):
                    pages_scanned = page_number
                    accumulated_text += f"{page_text}\n"
                    log_to_mongo(
                        process_instance_id,
                        message=f"Scanned OCR text from page {page_number}",
                        node_name="Classification",
                        log_type=0,
                    )

                if accumulated_text.strip():
                    prompt = f"""
                    Classify the document based on the following content into one of these categories: {label_str}.
                    Return only the label, nothing else.

                    Content:
                    {accumulated_text[:6000]}
                    """

                    response = openai_client.chat.completions.create(
                        model="gpt-4o",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.2,
                        timeout=20,
                    )

                    classification = response.choices[0].message.content.strip()
                    if classification in target_labels:
                        results[file_name] = classification
                        print(f"{file_name} -> {classification} (classified after scanning {pages_scanned} pages)")
                        log_to_mongo(
                            process_instance_id,
                            message=f"{file_name} -> {classification} (classified after scanning {pages_scanned} pages)",
                            node_name="Classification",
                            log_type=2,
                        )
                    else:
                        results[file_name] = "Unknown"
                        print(f"{file_name} -> Unrecognized label '{classification}' after scanning all pages")
                        log_to_mongo(
                            process_instance_id,
                            message=f"{file_name} -> Unrecognized label '{classification}' after scanning all pages",
                            node_name="Classification",
                            log_type=3,
                        )
                else:
                    results[file_name] = "Unknown"
                    print(f"{file_name} -> Unable to classify after scanning all pages")
                    log_to_mongo(
                        process_instance_id,
                        message=f"{file_name} -> Unable to classify after scanning all pages",
                        node_name="Classification",
                        log_type=3,
                    )

            except Exception as e:
                results[file_name] = f"Error: {e}"
                print(f"{file_name} -> {e}")
                log_to_mongo(
                    process_instance_id,
                    message=f"{file_name} -> {e}",
                    node_name="Classification",
                    log_type=1,
                )

        with open(
            os.path.join(process_instance_dir_path, "classified_documents.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(results, f)
            print("Classification results saved to classified_documents.json")

        for doc_type in target_labels:
            cursor.execute(
                """
                UPDATE DocumentType
                SET isActive = 0, updatedAt = NOW()
                WHERE documentType = %s
                """,
                (doc_type,),
            )
        conn.commit()
        print("DocumentType table updated -> isActive=0 after classification.")

        if AUTO_EXECUTE_NEXT_NODE == 1 and not is_orchestrated:
            print("Triggering extract_documents_dag...")
            token = get_auth_token()
            trigger_url = f"{AIRFLOW_API_URL}/dags/extract_documents_dag/dagRuns"
            run_id = f"triggered_by_classify_{process_instance_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
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
            print(f"Successfully triggered extract_documents_dag with ID {process_instance_id}")
            log_to_mongo(
                process_instance_id,
                message=f"Successfully triggered extract_documents_dag with ID {process_instance_id}",
                node_name="Classification",
                log_type=2,
            )

    except Exception as e:
        conn.rollback()
        error_message = f"{type(e).__name__}: {str(e)}"
        print(f"Error in classification process: {error_message}")
        log_to_mongo(
            process_instance_id,
            message=f"Error in classification process: {error_message}",
            node_name="Classification",
            log_type=1,
        )
        log_to_mongo(
            process_instance_id=process_instance_id,
            node_name="Classification",
            message=error_message,
            log_type=1,
            remark="DAG failed at classification",
        )
        raise
    finally:
        cursor.close()
        conn.close()


with DAG(
    dag_id="classify_documents_dag",
    start_date=datetime.now() - timedelta(days=1),
    schedule=None,
    catchup=False,
    tags=["idp", "classification"],
) as dag:

    classify_task = PythonOperator(
        task_id="classify_documents_task",
        python_callable=classify_documents,
    )

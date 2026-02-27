from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.mysql.hooks.mysql import MySqlHook
from datetime import datetime, timedelta
import os
import re
import json
from openai import OpenAI
from opik.integrations.openai import track_openai
import pytesseract
from pdf2image import convert_from_path
import tempfile
import requests
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from pymongo import MongoClient
import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

load_dotenv() 

AUTO_EXECUTE_NEXT_NODE = 1
MONGO_URI = os.getenv("MONGO_URI")

# === DAG Trigger CONFIG === #
AIRFLOW_API_URL = "http://airflow-airflow-apiserver-1:8080/api/v2"  # or localhost in local mode
AIRFLOW_USERNAME = os.getenv("AIRFLOW_USERNAME")
AIRFLOW_PASSWORD = os.getenv("AIRFLOW_PASSWORD")
OPIK_API_KEY = os.getenv("OPIK_API_KEY")  
OPIK_PROJECT_NAME= "idp-classification"
LOCAL_MODE = os.getenv("LOCAL_MODE", "false").lower() == "true"

if LOCAL_MODE:
    AIRFLOW_API_URL = "http://localhost:8080/api/v2"

# === CONFIG ===
LOCAL_DOWNLOAD_DIR = "/opt/airflow/downloaded_docs"
ML_MODELS_DIR = "/opt/airflow/dags/ml_models"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # from .env
OpenAI.api_key = OPENAI_API_KEY
MONGO_DB_NAME = "idp"
MONGO_COLLECTION = "LogEntry"
mongo_client = MongoClient(MONGO_URI)
mongo_collection = mongo_client[MONGO_DB_NAME][MONGO_COLLECTION]
openai_client = OpenAI()  
openai_client = track_openai(openai_client, project_name="my-idp-project")


if not OpenAI.api_key or not OpenAI.api_key.startswith("sk-") and not OpenAI.api_key.startswith("sk-proj-"):
    raise EnvironmentError("❌ OpenAI API key missing or invalid. Please set OPENAI_API_KEY as an environment variable.")

# ============================
# ML CLASSIFICATION HELPERS
# ============================

_EMB_MODEL = None
_VECTOR_CACHE = None  # cache loaded PKLs

def _normalize_text(text: str) -> str:
    text = text or ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def _extract_pdf_text_ml(file_path: str, max_pages: int = 10) -> str:
    """
    Efficient text extraction with OCR fallback up to `max_pages`.
    """
    text_parts = []
    try:
        reader = PdfReader(file_path)
        pages = min(len(reader.pages), max_pages)
        for i in range(pages):
            try:
                t = reader.pages[i].extract_text() or ""
            except Exception:
                t = ""
            if t.strip():
                text_parts.append(t)
            else:
                # OCR fallback (single page)
                try:
                    images = convert_from_path(file_path, first_page=i+1, last_page=i+1)
                    if images:
                        ocr_text = pytesseract.image_to_string(images[0]) or ""
                        if ocr_text.strip():
                            text_parts.append(ocr_text)
                except Exception:
                    # keep going, don't fail the run
                    pass
    except Exception:
        # if PdfReader fails, try OCR of first `max_pages` pages
        try:
            images = convert_from_path(file_path, first_page=1, last_page=max_pages)
            for img in images:
                ocr_text = pytesseract.image_to_string(img) or ""
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
    print(f'ML models Dir:', ML_MODELS_DIR)
    tfidf_pkl = embeddings_pkl = vectorizer_pkl = None

    for d in search_dirs:
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
            "Missing TF–IDF assets. Expecting tfidf_vectors.pkl and vectorizer.pkl "
            f"in {base_dir} or current working directory."
        )

    tfidf_data = joblib.load(tfidf_pkl)          # expects keys: {"labels", "vectors"}
    vectorizer = joblib.load(vectorizer_pkl)     # sklearn vectorizer
    embeddings_data = joblib.load(embeddings_pkl) if embeddings_pkl else None  # {"labels","vectors"}

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
    target_labels=None,
    threshold_embed: float = 0.35,
    threshold_tfidf: float = 0.25,
    max_pages: int = 10
) -> str:
    """
    Classify a PDF using prebuilt vectors (embeddings -> TF-IDF fallback).
    Returns a label or "Unknown".
    """
    try:
        assets = _load_vector_assets(base_dir)
    except Exception as e:
        print(f"❌ ML assets load failed: {e}")
        return "Unknown"

    tfidf_data     = assets["tfidf_data"]
    vectorizer     = assets["vectorizer"]
    embeddings_data= assets["embeddings_data"]

    text = _extract_pdf_text_ml(file_path, max_pages=max_pages)
    if not text.strip():
        return "Unknown"

    # 1) Embeddings first (if available)
    if embeddings_data:
        try:
            emb_labels = embeddings_data["labels"]
            emb_vecs   = embeddings_data["vectors"]
            model      = _get_embedding_model()
            qvec       = model.encode([text], convert_to_numpy=True)
            sims       = cosine_similarity(qvec, emb_vecs)[0]
            idx        = int(np.argmax(sims))
            score      = float(sims[idx])
            print(f"🔹 {os.path.basename(file_path)} | Embedding similarity = {score:.3f}")

            if score >= threshold_embed:
                pred = emb_labels[idx]
                if (not target_labels) or (pred in target_labels):
                    return pred
                else:
                    print(f"⚠️ Pred '{pred}' not in blueprint categories; will try TF–IDF fallback.")
        except Exception as e:
            print(f"⚠️ Embedding stage failed: {e} — falling back to TF–IDF")

    # 2) TF–IDF fallback
    try:
        tf_labels = tfidf_data["labels"]
        tf_vecs   = tfidf_data["vectors"]
        qtf       = vectorizer.transform([text]).toarray()
        sims      = cosine_similarity(qtf, tf_vecs)[0]
        idx       = int(np.argmax(sims))
        score     = float(sims[idx])
        print(f"🔹 {os.path.basename(file_path)} | TF–IDF similarity = {score:.3f}")

        if score >= threshold_tfidf:
            pred = tf_labels[idx]
            if (not target_labels) or (pred in target_labels):
                return pred
    except Exception as e:
        print(f"⚠️ TF–IDF stage failed: {e}")

    print(f"⚠️ {os.path.basename(file_path)} → Unknown")
    return "Unknown"
# ============================

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

def extract_text_per_page(pdf_path, max_pages=10):
    try:
        reader = PdfReader(pdf_path)
        total_pages = min(len(reader.pages), max_pages)
        for i in range(total_pages):
            images = convert_from_path(pdf_path, first_page=i+1, last_page=i+1)
            if not images:
                continue
            text = pytesseract.image_to_string(images[0])
            yield text
    except Exception as e:
        print(f"❌ OCR failed for {pdf_path}: {e}")
        log_to_mongo(process_instance_id, message = f"OCR failed for {pdf_path}: {e}", node_name = "Classification", log_type=1)
        return

def classify_documents(**context):
    process_instance_id = context["dag_run"].conf.get("id")
    is_orchestrated = bool(context["dag_run"].conf.get("orchestrated", False))
    if not process_instance_id:
        raise ValueError("Missing process_instance_id in dag_run.conf")

    process_instance_dir_path = os.path.join(LOCAL_DOWNLOAD_DIR, "process-instance-" + str(process_instance_id))
    os.makedirs(process_instance_dir_path, exist_ok=True)
    blueprint_path = os.path.join(process_instance_dir_path, "blueprint.json")

    hook = MySqlHook(mysql_conn_id="idp_mysql")
    conn = hook.get_conn()
    cursor = conn.cursor()
    try:
        if not os.path.exists(blueprint_path):
            raise FileNotFoundError(f"❌ blueprint.json not found at {blueprint_path}. Please run ingestion DAG first.")
            log_to_mongo(process_instance_id, message = f" blueprint.json not found at {blueprint_path}. Please run ingestion DAG first.", node_name = "Classification", log_type=1)

        with open(blueprint_path, "r") as f:
            blueprint = json.load(f)

        cursor.execute(
            """
            UPDATE ProcessInstances
            SET currentStage = %s,
                isInstanceRunning = %s,
                updatedAt = NOW()
            WHERE id = %s
            """,
            ("Classification", 1, process_instance_id)
        )
        conn.commit()
        print(f"🟢 ProcessInstances updated → currentStage='Classification', isInstanceRunning=1 for process_instance_id={process_instance_id}")
        log_to_mongo(process_instance_id, message = f"ProcessInstances updated → currentStage='Classification', isInstanceRunning=1 for process_instance_id={process_instance_id}", node_name = "Classification", log_type=2)

        classify_node = next((n for n in blueprint if n["nodeName"].lower() == "classify"), None)
        if not classify_node:
            raise ValueError("Classification node not found in blueprint")
            log_to_mongo(process_instance_id, message = f"Classification node not found in blueprint", node_name = "Classification", log_type=1)

        categories = classify_node["component"].get("categories", [])
        if not categories:
            raise ValueError("No categories found in Classify component")
            log_to_mongo(process_instance_id, message = f"No categories found in Classify component", node_name = "Classification", log_type=1)

        target_labels = [c["documentType"] for c in categories]
        label_str = ", ".join(target_labels)
        results = {}

        for doc_type in target_labels:
            cursor.execute("SELECT id FROM DocumentType WHERE documentType = %s", (doc_type,))
            exists = cursor.fetchone()
            if exists:
                cursor.execute("""
                    UPDATE DocumentType
                    SET isActive = 1, updatedAt = NOW()
                    WHERE documentType = %s
                """, (doc_type,))
            else:
                cursor.execute("""
                    INSERT INTO DocumentType (documentType, isActive, createdAt)
                    VALUES (%s, 1, NOW())
                """, (doc_type,))
        conn.commit()
        print("✅ DocumentType table updated → isActive=1 for target document types")
        log_to_mongo(process_instance_id, message = f"No categories found in Classify component", node_name = "Classification", log_type=2)

        for file_name in os.listdir(process_instance_dir_path):
            if not file_name.endswith(".pdf"):
                continue
            
            file_path = os.path.join(process_instance_dir_path, file_name)

            ###     CHECK FOR CLASSIFICATION MODEL HERE GENAI/ML     ###
            # Read classification model from blueprint ("genai" or "ml")
            model_choice = (classify_node["component"].get("model") or "genai").strip().lower()
            use_ml = (model_choice == "ml")

            if use_ml:
                print("🧠 Using ML classifier (embeddings → TF-IDF fallback)")
                log_to_mongo(process_instance_id, node_name="Classification",
                            message="Using ML classifier (vectors pipeline)", log_type=0)
                # Preload/validate vector assets once (raises if missing)
                try:
                    _ = _load_vector_assets(process_instance_dir_path)
                except Exception as e:
                    err = f"ML assets not available: {e}"
                    print(f"❌ {err}")
                    log_to_mongo(process_instance_id, node_name="Classification", message=err, log_type=1)
                    raise
            else:
                print("🤖 Using GENAI classifier")
                log_to_mongo(process_instance_id, node_name="Classification",
                            message="Using GENAI classifier", log_type=0)

            try:
                print(f"📄 Classifying: {file_name}")
                # If ML is selected, classify once with the ML pipeline and skip GENAI loop
                if use_ml:
                    classification = classify_document_ml(
                        file_path=file_path,
                        base_dir=process_instance_dir_path,
                        target_labels=target_labels,   # restrict to blueprint categories
                        threshold_embed=0.35,
                        threshold_tfidf=0.25,
                        max_pages=10
                    )
                    results[file_name] = classification
                    print(f"✅ {file_name} → {classification} (ML)")
                    log_to_mongo(process_instance_id,
                                message=f"{file_name} → {classification} (ML)",
                                node_name="Classification",
                                log_type=2 if classification != 'Unknown' else 3)
                    continue  # IMPORTANT: skip the GENAI page-by-page code below

                # --- GENAI classification (page-by-page) ---
                accumulated_text = ""
                for page_number, page_text in enumerate(extract_text_per_page(file_path, max_pages=20), start=1):
                    accumulated_text += page_text + "\n"
                    prompt = f"""
                    Classify the document based on the following content into one of these categories: {label_str}.
                    Return only the label, nothing else.
                    
                    Content:
                    {accumulated_text[:2000]}
                    """

                    response = openai_client.chat.completions.create(
                        model="gpt-4o",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.2,
                        timeout=20,
                    )

                    classification = response.choices[0].message.content.strip()
                    print(f"🔍 Page {page_number}: classified as {classification}")
                    log_to_mongo(process_instance_id, message = f"Page {page_number}: classified as {classification}", node_name = "Classification", log_type=2)

                    if classification in target_labels:
                        results[file_name] = classification
                        print(f"✅ {file_name} → {classification} (stopped early at page {page_number})")
                        log_to_mongo(process_instance_id, message = f"{file_name} → {classification} (stopped early at page {page_number})", node_name = "Classification", log_type=2)
                        break
                else:
                    results[file_name] = "Unknown"
                    print(f"⚠️ {file_name} → Unable to classify after scanning max pages")
                    log_to_mongo(process_instance_id, message = f"{file_name} → Unable to classify after scanning max pages", node_name = "Classification", log_type=3)

            except Exception as e:
                results[file_name] = f"Error: {e}"
                print(f"❌ {file_name} → {e}")
                log_to_mongo(process_instance_id, message = f"{file_name} → {e}", node_name = "Classification", log_type=1)

        with open(os.path.join(process_instance_dir_path, "classified_documents.json"), "w") as f:
            json.dump(results, f)
            print("📝 Classification results saved to classified_documents.json")

        for doc_type in target_labels:
            cursor.execute("""
                UPDATE DocumentType
                SET isActive = 0, updatedAt = NOW()
                WHERE documentType = %s
            """, (doc_type,))
        conn.commit()
        print("🔕 DocumentType table updated → isActive=0 after classification.")

        if AUTO_EXECUTE_NEXT_NODE == 1 and not is_orchestrated:
            print("🚀 Triggering extract_documents_dag...")
            token = get_auth_token()
            trigger_url = f"{AIRFLOW_API_URL}/dags/extract_documents_dag/dagRuns"
            run_id = f"triggered_by_classify_{process_instance_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
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
            print(f"✅ Successfully triggered extract_documents_dag with ID {process_instance_id}")
            log_to_mongo(process_instance_id, message = f"Successfully triggered extract_documents_dag with ID {process_instance_id}", node_name = "Classification", log_type=2)

    except Exception as e:
        conn.rollback()
        error_message = f"{type(e).__name__}: {str(e)}"
        print(f"❌ Error in classification process: {error_message}")
        log_to_mongo(process_instance_id, message = f"Error in classification process: {error_message}", node_name = "Classification", log_type=1)

        log_to_mongo(
            process_instance_id=process_instance_id,
            node_name="Classification",
            message=error_message,
            log_type=1,
            remark="DAG failed at classification"
        )

        raise

with DAG(
    dag_id="classify_documents_dag",
    start_date=datetime.now() - timedelta(days=1),
    schedule=None,
    catchup=False,
    tags=["idp", "classification"],
) as dag:

    classify_task = PythonOperator(
        task_id="classify_documents_task",
        python_callable=classify_documents
    )

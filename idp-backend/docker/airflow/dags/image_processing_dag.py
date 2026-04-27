"""
Image Processing DAG
Processes documents through OCR and optional AI cleanup
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.mysql.hooks.mysql import MySqlHook
from datetime import datetime, timedelta
import os
import json
import requests
from dotenv import load_dotenv
from pymongo import MongoClient
from openai import OpenAI
from opik.integrations.openai import track_openai
from transaction_status import sync_stage_status

# Import OCR services
from ocr_services.ocr_service_factory import get_ocr_service
from ocr_services.ocr_cache_utils import ensure_ocr_cache, get_ocr_output_dir
from ai_services.text_cleanup_service import TextCleanupService

load_dotenv()

AUTO_EXECUTE_NEXT_NODE = 1

# === DAG Trigger CONFIG === #
AIRFLOW_API_URL = "http://airflow-airflow-apiserver-1:8080/api/v2"
AIRFLOW_USERNAME = os.getenv("AIRFLOW_USERNAME")
AIRFLOW_PASSWORD = os.getenv("AIRFLOW_PASSWORD")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPIK_API_KEY = os.getenv("OPIK_API_KEY")
LOCAL_MODE = os.getenv("LOCAL_MODE", "false").lower() == "true"

if LOCAL_MODE:
    AIRFLOW_API_URL = "http://localhost:8080/api/v2"

# === CONFIG ===
LOCAL_DOWNLOAD_DIR = "/opt/airflow/downloaded_docs"
MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB_NAME = "idp"
MONGO_COLLECTION = "LogEntry"
mongo_client = MongoClient(MONGO_URI)
mongo_collection = mongo_client[MONGO_DB_NAME][MONGO_COLLECTION]

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

# Initialize OpenAI client for AI cleanup
openai_client = None
if OPENAI_API_KEY:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    openai_client = track_openai(openai_client, project_name="my-idp-project")


def log_to_mongo(process_instance_id, node_name, message, log_type=1, remark=""):
    """Log message to MongoDB"""
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


def process_documents_with_ocr(**context):
    """
    Main function to process documents through OCR and AI cleanup
    
    Args:
        context: Airflow context containing DAG run configuration
    """
    # Get process instance ID from DAG run configuration
    process_instance_id = context["dag_run"].conf.get("id")
    if not process_instance_id:
        raise ValueError("Missing process_instance_id in dag_run.conf")
        log_to_mongo(
            process_instance_id,
            "ImageProcessing",
            "Missing process_instance_id in dag_run.conf",
            log_type=1
        )
    
    process_instance_dir_path = os.path.join(
        LOCAL_DOWNLOAD_DIR,
        f"process-instance-{process_instance_id}"
    )
    os.makedirs(process_instance_dir_path, exist_ok=True)
    
    ocr_output_dir = get_ocr_output_dir(process_instance_dir_path)
    
    blueprint_path = os.path.join(process_instance_dir_path, "blueprint.json")
    
    # Initialize MySQL connection
    hook = MySqlHook(mysql_conn_id="idp_mysql")
    conn = hook.get_conn()
    cursor = conn.cursor()
    
    try:
        # Check if blueprint exists
        if not os.path.exists(blueprint_path):
            raise FileNotFoundError(
                f"❌ blueprint.json not found at {blueprint_path}. "
                f"Please run ingestion DAG first."
            )
            log_to_mongo(
                process_instance_id,
                "ImageProcessing",
                f"blueprint.json not found at {blueprint_path}",
                log_type=1
            )
        
        # Load blueprint
        with open(blueprint_path, "r") as f:
            blueprint = json.load(f)
        
        transaction_id = sync_stage_status(cursor, process_instance_id, "ImageProcessing", 1)
        conn.commit()
        print(f"✅ Updated ProcessInstances to 'ImageProcessing' stage")
        log_to_mongo(
            process_instance_id,
            "ImageProcessing",
            "ProcessInstance stage updated to 'ImageProcessing'",
            log_type=2
        )
        
        # Find image_processing node in blueprint
        image_processing_node = next(
            (node for node in blueprint
             if isinstance(node, dict) and node.get("nodeName", "").lower() == "image processing"),
            None
        )
        
        if not image_processing_node:
            log_to_mongo(
                process_instance_id,
                "ImageProcessing",
                "No image processing node found in blueprint",
                log_type=1
            )
            raise ValueError("No image processing node found in blueprint")
        
        # Get configuration from blueprint
        component = image_processing_node.get("component", {})
        ocr_engine = component.get("ocr_engine", "paddle").lower()
        language_mode = component.get("language_mode", "auto")
        ai_cleanup = component.get("ai_cleanup", False)
        output_format = component.get("output_format", "txt")  # Default to txt
        
        print(f"📋 Configuration:")
        print(f"   OCR Engine: {ocr_engine}")
        print(f"   Language Mode: {language_mode}")
        print(f"   AI Cleanup: {ai_cleanup}")
        print(f"   Output Format: {output_format}")
        
        log_to_mongo(
            process_instance_id,
            "ImageProcessing",
            f"Configuration: OCR={ocr_engine}, Language={language_mode}, AI_Cleanup={ai_cleanup}",
            log_type=0
        )
        
        # Get OCR service
        try:
            ocr_service = get_ocr_service(ocr_engine)
            print(f"✅ Initialized {ocr_engine} OCR service")
        except Exception as e:
            error_msg = f"Failed to initialize OCR service {ocr_engine}: {e}"
            print(f"❌ {error_msg}")
            log_to_mongo(process_instance_id, "ImageProcessing", error_msg, log_type=1)
            raise
        
        # Initialize AI cleanup service if needed
        cleanup_service = None
        if ai_cleanup:
            if not openai_client:
                warning_msg = "AI cleanup requested but OpenAI client not available"
                print(f"⚠️ {warning_msg}")
                log_to_mongo(process_instance_id, "ImageProcessing", warning_msg, log_type=3)
            else:
                cleanup_service = TextCleanupService(openai_client)
                print("✅ Initialized AI cleanup service")
        
        # Process each PDF file in the directory
        processed_files = []
        pdf_files = [
            f for f in os.listdir(process_instance_dir_path)
            if f.lower().endswith('.pdf')
        ]
        
        if not pdf_files:
            warning_msg = "No PDF files found to process"
            print(f"⚠️ {warning_msg}")
            log_to_mongo(process_instance_id, "ImageProcessing", warning_msg, log_type=3)
            return
        
        print(f"📄 Found {len(pdf_files)} PDF file(s) to process")
        
        for pdf_filename in pdf_files:
            pdf_path = os.path.join(process_instance_dir_path, pdf_filename)
            
            try:
                print(f"\n🔄 Processing: {pdf_filename}")
                log_to_mongo(
                    process_instance_id,
                    "ImageProcessing",
                    f"Processing {pdf_filename}",
                    log_type=0
                )
                
                # Prepare OCR config
                ocr_config = {
                    "language_mode": language_mode,
                    "psm": 3,  # Fully automatic page segmentation
                    "oem": 3,  # LSTM OCR engine
                    "dpi": component.get("dpi", 300),
                    "thread_count": component.get("thread_count", 2),
                }

                cache_payload = ensure_ocr_cache(
                    pdf_path=pdf_path,
                    process_instance_dir=process_instance_dir_path,
                    ocr_engine=ocr_engine,
                    config=ocr_config,
                    cleanup_service=cleanup_service,
                    force_refresh=True,
                )
                extracted_text = cache_payload.get("cleaned_text") or cache_payload.get("raw_text") or ""
                
                if not extracted_text or not extracted_text.strip():
                    warning_msg = f"No text extracted from {pdf_filename}"
                    print(f"⚠️ {warning_msg}")
                    log_to_mongo(process_instance_id, "ImageProcessing", warning_msg, log_type=3)
                    continue
                
                print(f"✅ Extracted {len(extracted_text)} characters from {pdf_filename}")
                
                # Apply AI cleanup if enabled
                final_text = extracted_text
                # The cache helper already performs cleanup, so skip a second pass here.
                if False and ai_cleanup and cleanup_service:
                    try:
                        print(f"🤖 Applying AI cleanup to {pdf_filename}...")
                        final_text = cleanup_service.cleanup_text(extracted_text)
                        print(f"✅ AI cleanup completed for {pdf_filename}")
                        log_to_mongo(
                            process_instance_id,
                            "ImageProcessing",
                            f"AI cleanup completed for {pdf_filename}",
                            log_type=2
                        )
                    except Exception as e:
                        error_msg = f"AI cleanup failed for {pdf_filename}: {e}"
                        print(f"⚠️ {error_msg}")
                        log_to_mongo(process_instance_id, "ImageProcessing", error_msg, log_type=3)
                        # Continue with original text if cleanup fails
                        final_text = extracted_text
                
                # Save output based on format
                base_filename = os.path.splitext(pdf_filename)[0]
                
                if output_format.lower() == "json":
                    output_path = os.path.join(ocr_output_dir, f"{base_filename}.json")
                    with open(output_path, "w", encoding="utf-8") as f:
                        json.dump(cache_payload, f, indent=2, ensure_ascii=False)
                else:
                    # Default to txt format
                    output_path = os.path.join(ocr_output_dir, f"{base_filename}.txt")
                    with open(output_path, "w", encoding="utf-8") as f:
                        f.write(final_text)
                
                print(f"💾 Saved OCR output: {output_path}")
                log_to_mongo(
                    process_instance_id,
                    "ImageProcessing",
                    f"Successfully processed {pdf_filename} → {output_path}",
                    log_type=2
                )
                
                processed_files.append(pdf_filename)
                
            except Exception as e:
                error_msg = f"Failed to process {pdf_filename}: {e}"
                print(f"❌ {error_msg}")
                log_to_mongo(process_instance_id, "ImageProcessing", error_msg, log_type=1)
                # Continue processing other files
                continue
        
        # Summary
        print(f"\n✅ Image Processing Complete!")
        print(f"   Processed: {len(processed_files)}/{len(pdf_files)} files")
        print(f"   Output directory: {ocr_output_dir}")
        
        log_to_mongo(
            process_instance_id,
            "ImageProcessing",
            f"Image processing completed: {len(processed_files)}/{len(pdf_files)} files processed",
            log_type=2
        )
        
        # Trigger next DAG if auto-execute is enabled
        if AUTO_EXECUTE_NEXT_NODE == 1:
            print("🚀 Triggering classify_documents_dag...")
            token = get_auth_token()
            trigger_url = f"{AIRFLOW_API_URL}/dags/classify_documents_dag/dagRuns"
            run_id = f"triggered_by_image_processing_{process_instance_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            }
            payload = {
                "dag_run_id": run_id,
                "logical_date": datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
                "conf": {"id": process_instance_id}
            }
            
            try:
                response = requests.post(trigger_url, json=payload, headers=headers, timeout=10)
                response.raise_for_status()
                print(f"✅ Successfully triggered classify_documents_dag with ID {process_instance_id}")
                log_to_mongo(
                    process_instance_id,
                    "ImageProcessing",
                    f"Successfully triggered classify_documents_dag",
                    log_type=2
                )
            except Exception as e:
                error_msg = f"Failed to trigger next DAG: {e}"
                print(f"❌ {error_msg}")
                log_to_mongo(process_instance_id, "ImageProcessing", error_msg, log_type=1)
    
    except Exception as e:
        conn.rollback()
        error_message = f"{type(e).__name__}: {str(e)}"
        print(f"❌ Error in image processing: {error_message}")
        
        log_to_mongo(
            process_instance_id=process_instance_id,
            node_name="ImageProcessing",
            message=error_message,
            log_type=1,
            remark="DAG failed at image processing"
        )
        
        raise
    
    finally:
        cursor.close()
        conn.close()


# === DAG Definition ===
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id="image_processing_dag",
    default_args=default_args,
    start_date=datetime.now() - timedelta(days=1),
    schedule=None,
    catchup=False,
    tags=["idp", "image_processing", "ocr"],
) as dag:

    image_processing_task = PythonOperator(
        task_id="process_documents_with_ocr",
        python_callable=process_documents_with_ocr,
    )


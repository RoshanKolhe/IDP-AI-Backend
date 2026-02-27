from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from airflow.providers.mysql.hooks.mysql import MySqlHook
import json
import requests
import os
from ftplib import FTP
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding as sym_padding
from cryptography.hazmat.backends import default_backend
import base64
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv() 


# === Secrets === #
SECRET_KEY = os.getenv("SECRET_KEY")  # Must be exactly 32 bytes
MONGO_URI = os.getenv("MONGO_URI")
INGESTION_URL = os.getenv("UI_PORTAL_INGESTION_URL") #Ingestion URL of UI portal

# === DAG Trigger CONFIG === #
AIRFLOW_API_URL = "http://airflow-airflow-apiserver-1:8080/api/v2"  # or localhost in local mode
AIRFLOW_USERNAME = os.getenv("AIRFLOW_USERNAME")
AIRFLOW_PASSWORD = os.getenv("AIRFLOW_PASSWORD")
LOCAL_MODE = os.getenv("LOCAL_MODE", "false").lower() == "true"

if LOCAL_MODE:
    AIRFLOW_API_URL = "http://localhost:8080/api/v2"

# === CONFIG === #
LOCAL_DOWNLOAD_DIR = "/opt/airflow/downloaded_docs"
AUTO_EXECUTE_NEXT_NODE = 1
MONGO_DB_NAME = "idp"
MONGO_COLLECTION = "LogEntry"
mongo_client = MongoClient(MONGO_URI)
mongo_collection = mongo_client[MONGO_DB_NAME][MONGO_COLLECTION]

def fix_base64_padding(s: str) -> str:
    return s + '=' * (-len(s) % 4)

def decrypt_password(encrypted_base64: str, secret_key: bytes) -> str:
    try:
        encrypted_base64 = fix_base64_padding(encrypted_base64)
        raw = base64.b64decode(encrypted_base64)
        iv = raw[:16]
        ciphertext = raw[16:]

        cipher = Cipher(algorithms.AES(secret_key), modes.CBC(iv), backend=default_backend())
        decryptor = cipher.decryptor()
        padded_plaintext = decryptor.update(ciphertext) + decryptor.finalize()

        unpadder = sym_padding.PKCS7(128).unpadder()
        plaintext = unpadder.update(padded_plaintext) + unpadder.finalize()

        return plaintext.decode('utf-8')
    except Exception as e:
        print("❌ Failed to decrypt FTP password:", e)
        AUTO_EXECUTE_NEXT_NODE = 0
        log_to_mongo(process_instance_id, "Failed to decrypt FTP password", node_name = "Ingestion", log_type=1)

        raise

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

def log_success(process_instance_id, step, msg):
    log_to_mongo(process_instance_id, step, msg, node_name = "Ingestion", log_type=2)

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


def fetch_blueprint_and_download_docs(**context):
    # Get process instance ID from DAG run configuration
    process_instance_id = context["dag_run"].conf.get("id")
    is_orchestrated = bool(context["dag_run"].conf.get("orchestrated", False))
    if not process_instance_id:
        raise ValueError("Missing process_instance_id in dag_run.conf")
        log_to_mongo(process_instance_id, message = "Missing process_instance_id in dag_run.conf", node_name = "Ingestion", log_type=1)
        
    global AUTO_EXECUTE_NEXT_NODE
    valid_extensions = ['.pdf']

    process_instance_dir_path = os.path.join(LOCAL_DOWNLOAD_DIR, f"process-instance-{process_instance_id}")
    os.makedirs(process_instance_dir_path, exist_ok=True)
    BLUEPRINT_JSON_PATH = os.path.join(process_instance_dir_path, "blueprint.json")
    
    # Initialize MySQL connection
    hook = MySqlHook(mysql_conn_id="idp_mysql")
    conn = hook.get_conn()
    cursor = conn.cursor()
    
    try:
        # 1. Get the process_id linked to this instance
        cursor.execute("""
            SELECT processesId 
            FROM ProcessInstances 
            WHERE id = %s
        """, (process_instance_id,))
        instance_data = cursor.fetchone()
        
        if not instance_data:
            raise ValueError(f"No process instance found with ID {process_instance_id}")
            log_to_mongo(process_instance_id, message = f"No process instance found with ID {process_instance_id}", node_name = "Ingestion", log_type=1)
        process_id = instance_data[0]  # Using index instead of dictionary access


        #get the process-instance-folder-name
        cursor.execute("""
            SELECT processInstanceFolderName 
            FROM ProcessInstances 
            WHERE id = %s
        """, (process_instance_id,))
        instance_data = cursor.fetchone()

        process_instance_folder = instance_data[0]

        # 2. Get bluePrintId from Processes
        cursor.execute("SELECT bluePrintId FROM Processes WHERE id = %s", (process_id,))
        blueprint_id_row = cursor.fetchone()
        if not blueprint_id_row or not blueprint_id_row[0]:
            raise ValueError(f"No bluePrintId found for process ID {process_id}")
            log_to_mongo(process_instance_id, message = f"No bluePrintId found for process ID {process_id}", node_name = "Ingestion", log_type=1)
        blueprint_id = blueprint_id_row[0]
        log_to_mongo(process_instance_id, "Ingestion", "Blueprint saved successfully", log_type=2)


        # 3. Get blueprint JSON from BluePrint table
        cursor.execute("SELECT bluePrint FROM BluePrint WHERE id = %s", (blueprint_id,))
        blueprint_row = cursor.fetchone()
        if not blueprint_row or not blueprint_row[0]:
            raise ValueError(f"No blueprint found for blueprint ID {blueprint_id}")
            log_to_mongo(process_instance_id, message = f"No blueprint found for blueprint ID {blueprint_id}", node_name = "Ingestion", log_type=1)
        
        blueprint_json = json.loads(blueprint_row[0])

        # Save blueprint locally for other DAGs
        os.makedirs(process_instance_dir_path, exist_ok=True)
        with open(BLUEPRINT_JSON_PATH, "w") as f:
            json.dump(blueprint_json, f, indent=2)
        print(f"✅ Blueprint saved to {BLUEPRINT_JSON_PATH}")

        # 4. Update ProcessInstances table
        cursor.execute("""
        UPDATE ProcessInstances
        SET currentStage = %s,
            isInstanceRunning = %s,
            updatedAt = NOW()
        WHERE id = %s
        """, ("Ingestion", 1, process_instance_id))
        conn.commit()
        print(f"✅ Updated ProcessInstance {process_instance_id} to Ingestion stage")
        log_to_mongo(process_instance_id, "Ingestion", "ProcessInstance stage updated to 'Ingestion'", log_type=2)


        # 5. Find ingestion node configuration
        ingestion_node = next(
            (node for node in blueprint_json 
             if isinstance(node, dict) and node.get("nodeName", "").lower() == "ingestion"), 
            None
        )
        if not ingestion_node:
            raise ValueError("No ingestion node found in blueprint")
            log_to_mongo(process_instance_id, message = f"No ingestion node found in blueprint", node_name = "Ingestion", log_type=1)

        ingestion_config = ingestion_node.get("component", {})
        ingestion_url = ingestion_config.get("url")
        old_ingestion_url = ingestion_url
        ingestion_url = ingestion_url + process_instance_folder

        if not ingestion_url:
            raise ValueError("Ingestion URL is missing in blueprint")
            log_to_mongo(process_instance_id, message = f"Ingestion URL is missing in blueprint", node_name = "Ingestion", log_type=1)

        # 6. Handle ingestion based on channelType
        channel_type = ingestion_config.get("channelType").lower()
        print(f"Ingestion Channel Type: {channel_type}")
        documents = []

        if channel_type == "ftp":

            ftp_path = ingestion_config.get("path")
            ftp_host = ingestion_config.get("host")
            ftp_user = ingestion_config.get("userName", "anonymous")
            encrypted_ftp_pass = ingestion_config.get("password", "")

            try:
                ftp_pass = decrypt_password(encrypted_ftp_pass, SECRET_KEY)
            except Exception:
                raise ValueError("Unable to decrypt FTP password")
                log_to_mongo(process_instance_id, message = f"Unable to decrypt FTP password", node_name = "Ingestion", log_type=1)
                AUTO_EXECUTE_NEXT_NODE = 0

            if not ftp_host or not ftp_path:
                raise ValueError("FTP host or path is missing in blueprint")
                log_to_mongo(process_instance_id, message = f"FTP host or path is missing in blueprint", node_name = "Ingestion", log_type=1)
                AUTO_EXECUTE_NEXT_NODE = 0

            print(f"🔌 Connecting to FTP server: {ftp_host}")
            ftp = FTP()
            ftp.connect(ftp_host, 21)
            ftp.login(ftp_user, ftp_pass)
            ftp_pi_path = (ftp_path + "/process-instance-" + str(process_instance_id))
            print(f"📂 Navigating to FTP path: {ftp_pi_path}")
            ftp.cwd(ftp_pi_path)

            documents = ftp.nlst()  # List all files in directory
            print(f"📄 Found {len(documents)} documents on FTP")
            downloaded_count = 0

            for file_name in documents:
                file_name_lower = file_name.lower()
                if not any(file_name_lower.endswith(ext) for ext in valid_extensions):
                    print(f"⚠️ Skipping unsupported file: {file_name}")
                    log_to_mongo(process_instance_id, message = f"Skipping unsupported file: {file_name}", node_name = "Ingestion", log_type=3)
                    continue

                file_path = os.path.join(process_instance_dir_path, file_name)
                print(f"⬇️ Downloading {file_name} from FTP...")
                try:
                    with open(file_path, "wb") as f:
                        ftp.retrbinary(f"RETR {file_name}", f.write)
                    downloaded_count += 1
                    log_to_mongo(process_instance_id, "Ingestion", f"Downloaded {downloaded_count} documents successfully", log_type=2)
                except Exception as e:
                    print(f"❌ Failed to download {file_name}: {str(e)}")
                    log_to_mongo(process_instance_id, "Ingestion", f"Failed to Download Documents.", log_type=1)
                    AUTO_EXECUTE_NEXT_NODE = 0

            ftp.quit()

        elif channel_type == "ui" or channel_type == "api":
            ingestion_url = INGESTION_URL
            old_ingestion_url = ingestion_url
            ingestion_url = ingestion_url + process_instance_folder

            if not ingestion_url:
                raise ValueError("Ingestion URL is missing in blueprint")
                log_to_mongo(process_instance_id, message = f"Ingestion URL is missing in blueprint", node_name = "Ingestion", log_type=1)
                AUTO_EXECUTE_NEXT_NODE = 0

            print(f"🔍 Fetching document list from HTTP: {ingestion_url}")
            response = requests.get(ingestion_url, timeout=30)
            response.raise_for_status()
            documents = response.json()
            print(f"📄 Found {len(documents)} documents at UI Portal")

            for file_name in documents:
                file_name_lower = file_name.lower()
                if not any(file_name_lower.endswith(ext) for ext in valid_extensions):
                    print(f"⚠️ Skipping unsupported file: {file_name}")
                    log_to_mongo(process_instance_id, message = f"Skipping unsupported file: {file_name}", node_name = "Ingestion", log_type=3)
                    continue

                file_url = f"{old_ingestion_url}/file/{process_instance_folder}/{file_name}"
                file_path = os.path.join(process_instance_dir_path, file_name)

                print(f"⬇️ Downloading {file_name} from UI Portal...")
                downloaded_count = 0
                try:
                    with requests.get(file_url, stream=True, timeout=30) as r:
                        r.raise_for_status()
                        with open(file_path, "wb") as f:
                            for chunk in r.iter_content(chunk_size=8192):
                                f.write(chunk)
                    downloaded_count += 1
                    log_to_mongo(process_instance_id, "Ingestion", f"Downloaded {downloaded_count} documents successfully", log_type=2)
                except Exception as e:
                    print(f"❌ Failed to download {file_name}: {str(e)}")
                    log_to_mongo(process_instance_id, "Ingestion", f"Failed to Download Documents.", log_type=1)
                    AUTO_EXECUTE_NEXT_NODE = 0

        else:
            raise ValueError(f"Unsupported channelType: {channel_type}")


        # 8. Trigger classify_documents_dag
        if AUTO_EXECUTE_NEXT_NODE == 1 and not is_orchestrated:
            print("🚀 Triggering classify_documents_dag...")
            token = get_auth_token()
            trigger_url = f"{AIRFLOW_API_URL}/dags/classify_documents_dag/dagRuns"
            run_id = f"triggered_by_ingest_{process_instance_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
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
            log_to_mongo(process_instance_id, "Ingestion", "Successfully triggered classify_documents_dag", log_type=2)

    except Exception as e:
    
        conn.rollback()
        error_message = f"{type(e).__name__}: {str(e)}"
        print(f"❌ Error in ingestion process: {error_message}")

        log_to_mongo(
            process_instance_id=process_instance_id,
            node_name="Ingestion",
            message=error_message,
            log_type=1,
            remark="DAG failed at ingestion"
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
    dag_id="ingest_documents_dag",
    default_args=default_args,
    start_date=datetime.now() - timedelta(days=1),
    schedule=None,
    catchup=False,
    tags=["idp", "ingestion"],
) as dag:

    ingest_task = PythonOperator(
        task_id="fetch_and_download_documents",
        python_callable=fetch_blueprint_and_download_docs,
    )

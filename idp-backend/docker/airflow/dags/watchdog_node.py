from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from airflow.providers.mysql.hooks.mysql import MySqlHook
import logging
import requests
import json
import os
from dotenv import load_dotenv

# Configuration
load_dotenv() 
AIRFLOW_API_URL = "http://airflow-airflow-apiserver-1:8080/api/v2"  # Use service name for Docker
AIRFLOW_USERNAME = os.getenv("AIRFLOW_USERNAME")
AIRFLOW_PASSWORD = os.getenv("AIRFLOW_PASSWORD")
LOCAL_MODE = os.getenv("LOCAL_MODE", "false").lower() == "true"

if LOCAL_MODE:
    AIRFLOW_API_URL = "http://localhost:8080/api/v2"

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

def get_auth_token():
    """Get JWT token from Airflow API"""
    auth_url = f"{AIRFLOW_API_URL.replace('/api/v2', '')}/auth/token"
    try:
        response = requests.post(
            auth_url,
            json={"username": AIRFLOW_USERNAME, "password": AIRFLOW_PASSWORD},
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        response.raise_for_status()
        logging.info(f"Auth response: {response.text}")
        return response.json()["access_token"]
    except requests.exceptions.ConnectionError as e:
        logging.error(f"Failed to connect to {auth_url}: {str(e)}")
        raise
    except requests.exceptions.HTTPError as e:
        logging.error(f"HTTP error for {auth_url}: {str(e)}")
        raise
    except KeyError as e:
        logging.error(f"Invalid response format: {str(e)}")
        raise

def check_and_trigger_ingestion():
    """Main watchdog function"""
    try:
        # 1. Get authentication token
        token = get_auth_token()
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }

        # 2. Check for running instances
        hook = MySqlHook(mysql_conn_id="idp_mysql")
        conn = hook.get_conn()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id 
            FROM ProcessInstances 
            WHERE isInstanceRunning = 1 AND currentStage IS NULL
        """)
        running_instances = cursor.fetchall()
        
        if not running_instances:
            logging.info("No running process instances found")
            return

        logging.info(f"Found {len(running_instances)} running instances")

        # 3. Trigger service orchestrator DAG for each instance
        for (instance_id,) in running_instances:
            try:
                trigger_url = f"{AIRFLOW_API_URL}/dags/idp_service_orchestrator/dagRuns"
                run_id = f"watchdog_triggered_{instance_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                payload = {
                    "dag_run_id": run_id,
                    "logical_date": datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),  # Match curl format
                    "conf": {"id": instance_id, "orchestrated": True}
                }
                
                logging.info(f"Triggering DAG with payload: {json.dumps(payload)}")
                
                response = requests.post(
                    trigger_url,
                    json=payload,
                    headers=headers,
                    timeout=10
                )
                
                if response.status_code == 403:
                    # Token might be expired, refresh and retry once
                    token = get_auth_token()
                    headers["Authorization"] = f"Bearer {token}"
                    response = requests.post(
                        trigger_url,
                        json=payload,
                        headers=headers,
                        timeout=10
                    )
                
                response.raise_for_status()
                logging.info(f"Successfully triggered orchestrator for instance {instance_id}")
                
            except requests.exceptions.HTTPError as e:
                logging.error(f"Failed to trigger DAG for instance {instance_id}: {str(e)}")
                logging.error(f"Response: {response.text}")
                continue
            except Exception as e:
                logging.error(f"Failed to trigger DAG for instance {instance_id}: {str(e)}")
                continue
                
    except Exception as e:
        logging.error(f"Watchdog failed: {str(e)}")
        raise
    finally:
        if 'conn' in locals():
            conn.close()

with DAG(
    dag_id='watchdog_node',
    default_args=default_args,
    start_date=datetime.now() - timedelta(minutes=5),
    schedule='*/5 * * * *',  # Every 5 minutes
    catchup=False,
    tags=['idp', 'watchdog'],
) as dag:

    monitor_task = PythonOperator(
        task_id='check_and_trigger_instances',
        python_callable=check_and_trigger_ingestion,
    )

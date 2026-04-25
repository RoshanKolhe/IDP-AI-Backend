from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.utils import timezone
from airflow.providers.mysql.hooks.mysql import MySqlHook
from airflow.sensors.external_task import ExternalTaskSensor
from airflow.models import Variable
from datetime import datetime, timedelta
import json
import os
from dotenv import load_dotenv
from pymongo import MongoClient
import requests
import time

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
TRANSACTION_API_URL = "https://api.docognize.ai/process-instance-transactions"
TRANSACTION_ID = ""
MONGO_DB_NAME = "idp"
MONGO_COLLECTION = "LogEntry"
mongo_client = MongoClient(MONGO_URI)
mongo_collection = mongo_client[MONGO_DB_NAME][MONGO_COLLECTION]

# ---------------- CONFIG ---------------- #
NODE_TO_DAG_MAP = {
    "Ingestion": "ingest_documents_dag",
    "Classify": "classify_documents_dag",
    "Extract": "extract_documents_dag",
    "Validate": "highlight_extracted_fields_dag",
    "Deliver": "deliver_dag",
    "External Data Sources": "external_data_sources_dag",
    "Document Index": "document_index_dag",
    "Document Query": "document_query_dag",
    "Integration": "integration_dag",
    "Image Processing": "image_processing_dag",
    "Code": "code_node_dag",
    "AI Analyser": "ai_analyser_dag",
}

EXECUTABLE_TYPES = {
    "ingestion",
    "classify",
    "extract",
    "validate",
    "deliver",
    "externalDataSources",
    "documentIndex",
    "documentQuery",
    "integration",
    "imageProcessing",
    "code",
    "aiAnalyser",
}

DEFAULT_ARGS = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 0
}

MONGO_URI = os.getenv("MONGO_URI")
mongo_client = MongoClient(MONGO_URI)
mongo_collection = mongo_client["idp"]["LogEntry"]

# --------------------------------------- #
import os
import json
import requests

TRANSACTION_API_URL = "https://api.docognize.ai/process-instance-transactions"
LOCAL_DOWNLOAD_DIR = "/opt/airflow/downloaded_docs"

def _get_transaction_id(process_instance_id):
    tid_path = os.path.join(LOCAL_DOWNLOAD_DIR, f"process-instance-{process_instance_id}", "tid.json")
    if not os.path.exists(tid_path):
        return TRANSACTION_ID or None
    try:
        with open(tid_path, "r", encoding="utf-8") as f:
            return json.load(f).get("transactionId") or TRANSACTION_ID or None
    except Exception as exc:
        print(f"Warning: failed to read transaction id from {tid_path}: {exc}")
        return TRANSACTION_ID or None

def create_process_instance_transaction(current_stage: str, **context):

    global TRANSACTION_ID

    """
    Creates a process instance transaction and stores transactionId in tid.json
    """

    process_instance_id = context["dag_run"].conf.get("id")
    if not process_instance_id:
        raise ValueError("Missing process_instance_id in dag_run.conf")

    payload = {
        "currentStage": current_stage,
        "isDeleted": False,
        "isActive": True,
        "processInstancesId": process_instance_id
    }

    response = requests.post(
        TRANSACTION_API_URL,
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=15
    )
    response.raise_for_status()

    response_data = response.json()

    # Adjust key if backend uses a different field name
    transaction_id = (
        response_data.get("transactionId")
        or response_data.get("id")
    )

    if not transaction_id:
        raise ValueError("transactionId not found in API response")

    process_instance_dir = os.path.join(
        LOCAL_DOWNLOAD_DIR,
        f"process-instance-{process_instance_id}"
    )
    os.makedirs(process_instance_dir, exist_ok=True)

    tid_path = os.path.join(process_instance_dir, "tid.json")
    with open(tid_path, "w") as f:
        json.dump(
            {
                "processInstanceId": process_instance_id,
                "transactionId": transaction_id,
                "stage": current_stage
            },
            f,
            indent=2
        )

    print(f"✅ Transaction created: {transaction_id}")
    print(f"📝 Saved to {tid_path}")

    TRANSACTION_ID = transaction_id
    return transaction_id
# --------------------------------------- #

def wait_for_dag_completion(dag_id, run_id, poll_interval=20, timeout=6*60*60):
    start = time.time()

    token = get_auth_token()
    headers = {"Authorization": f"Bearer {token}"}

    while True:
        if time.time() - start > timeout:
            raise TimeoutError(f"{dag_id} run {run_id} timed out")

        url = f"{AIRFLOW_API_URL}/dags/{dag_id}/dagRuns/{run_id}"
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()

        state = resp.json().get("state")
        print(f"⏳ {dag_id} [{run_id}] → {state}")

        if state == "success":
            return
        if state in ("failed", "upstream_failed"):
            raise Exception(f"{dag_id} failed with state {state}")

        time.sleep(poll_interval)


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

def trigger_child_dag(dag_id, process_instance_id, parent_node_name):
    print(f"🚀 Triggering {dag_id}...")

    token = get_auth_token()   # ← SAME function you already have

    trigger_url = f"{AIRFLOW_API_URL}/dags/{dag_id}/dagRuns"

    run_id = (
        f"service__{process_instance_id}__{dag_id}__"
        f"{datetime.utcnow().strftime('%Y%m%dT%H%M%S%f')}"
    )

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    payload = {
        "dag_run_id": run_id,
        "logical_date": datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
        "conf": {
            "id": process_instance_id,
            "orchestrated": True
        }
    }

    response = requests.post(
        trigger_url,
        json=payload,
        headers=headers,
        timeout=10
    )

    if response.status_code not in (200, 201):
        raise Exception(
            f"Failed to trigger {dag_id}: {response.status_code} {response.text}"
        )

    print(f"✅ Successfully triggered {dag_id} | run_id={run_id}")

    log_to_mongo(
        process_instance_id=process_instance_id,
        node_name=parent_node_name,
        message=f"Triggered {dag_id} (run_id={run_id})",
        log_type=2
    )

    return run_id

def log_to_mongo(process_instance_id, node_name, message, log_type=1, remark=""):
    mongo_collection.insert_one({
        "processInstanceId": process_instance_id,
        "processInstanceTransactionId": _get_transaction_id(process_instance_id),
        "nodeName": node_name,
        "logsDescription": message,
        "logType": log_type,
        "createdAt": datetime.utcnow()
    })

# --------------------------------------- #

def get_aggregator_setting(blueprint):
    for node in blueprint:
        if node.get("nodeName") == "Aggregator":
            return node.get("component", {}).get("aggregatorSetting", 0)
    return 0  # default strict


def read_blueprint_and_graph(**context):
    ti = context["ti"]
    process_instance_id = context["dag_run"].conf.get("id")

    if not process_instance_id:
        raise ValueError("process_instance_id missing")

    hook = MySqlHook(mysql_conn_id="idp_mysql")
    conn = hook.get_conn()
    cursor = conn.cursor()

    try:
        cursor.execute("""
            SELECT p.bluePrint, p.nodes, p.edges, p.edgeSettings
            FROM BluePrint p
            JOIN Processes pr ON pr.bluePrintId = p.id
            JOIN ProcessInstances pi ON pi.processesId = pr.id
            WHERE pi.id = %s
        """, (process_instance_id,))

        row = cursor.fetchone()
        if not row:
            raise ValueError("No blueprint record found for process instance")

        bp_raw, nodes_raw, edges_raw, edge_settings_raw = row

        if not bp_raw:
            raise ValueError("Blueprint JSON is NULL in DB")

        # Parse JSON safely
        blueprint = json.loads(bp_raw)
        nodes = json.loads(nodes_raw) if nodes_raw else []
        edges = json.loads(edges_raw) if edges_raw else []
        edge_settings = json.loads(edge_settings_raw) if edge_settings_raw else {}

        # ✅ Save blueprint.json for worker DAGs
        process_instance_dir = f"/opt/airflow/downloaded_docs/process-instance-{process_instance_id}"
        os.makedirs(process_instance_dir, exist_ok=True)

        blueprint_path = os.path.join(process_instance_dir, "blueprint.json")
        with open(blueprint_path, "w") as f:
            json.dump(blueprint, f, indent=2)

        # Push XComs
        ti.xcom_push(key="blueprint", value=blueprint)
        ti.xcom_push(key="nodes", value=nodes)
        ti.xcom_push(key="edges", value=edges)

    finally:
        cursor.close()
        conn.close()


# --------------------------------------- #

def build_execution_plan(**context):
    ti = context["ti"]
    dag_run = context["dag_run"]
    process_instance_id = dag_run.conf.get("id")
    run_ids = dag_run.conf.get("runIds", [])

    blueprint = ti.xcom_pull(task_ids="read_blueprint_and_graph", key="blueprint")
    nodes = ti.xcom_pull(task_ids="read_blueprint_and_graph", key="nodes")
    edges = ti.xcom_pull(task_ids="read_blueprint_and_graph", key="edges")

    aggregator_setting = get_aggregator_setting(blueprint)

    wait_for_aggregator(run_ids, aggregator_setting)

    merge_type = "PARTIAL" if aggregator_setting == 1 else "STRICT"

    log_to_mongo(
        process_instance_id,
        "Aggregator",
        f"{merge_type} merge activated for {len(run_ids)} branches",
        log_type=2
    )




    # -------- VALIDATION (CRITICAL) -------- #
    if not blueprint:
        log_to_mongo(
            process_instance_id,
            "Service-Orchestrator",
            "Blueprint is missing or empty",
            log_type=1
        )
        raise ValueError("Blueprint is missing")

    if not nodes:
        log_to_mongo(
            process_instance_id,
            "Service-Orchestrator",
            "Nodes JSON is missing or empty",
            log_type=1
        )
        raise ValueError("Nodes JSON is missing")

    if edges is None:
        log_to_mongo(
            process_instance_id,
            "Service-Orchestrator",
            "Edges JSON is missing (NULL)",
            log_type=1
        )
        raise ValueError("Edges JSON is missing")

    log_to_mongo(
        process_instance_id,
        "Service-Orchestrator",
        "Building execution plan (router-aware)",
        log_type=0
    )
    print("Building execution plan...")

    # -------- BUILD LOOKUPS -------- #
    node_id_to_name = {}
    for n in nodes:
        node_id = n.get("id")
        node_type = n.get("type")
        label = n.get("data", {}).get("label")

        if node_id and label and node_type in EXECUTABLE_TYPES:
            node_id_to_name[node_id] = label

    if not node_id_to_name:
        raise ValueError("No valid node labels found in nodes.json")

    router_ids = {
        n["id"]
        for n in nodes
        if n.get("type") == "router" and "id" in n
    }

    outgoing = {}
    for e in edges:
        src = e.get("source")
        tgt = e.get("target")
        if src and tgt:
            outgoing.setdefault(src, []).append(tgt)

    # -------- EXECUTION PLAN -------- #
    execution_plan = []
    visited = set()

    def dfs(node_id):
        if node_id in visited:
            return
        visited.add(node_id)

        if node_id in router_ids:
            parallel_targets = outgoing.get(node_id, [])

            log_to_mongo(
                process_instance_id,
                "Router",
                f"Router detected → parallel split to node IDs {parallel_targets}",
                log_type=3
            )
            print(f"Router detected at node {node_id}, parallel targets: {parallel_targets}")

            execution_plan.append({
                "type": "parallel",
                "nodes": [
                    node_id_to_name[t]
                    for t in parallel_targets
                    if t in node_id_to_name
                ]
            })

            for t in parallel_targets:
                for nxt in outgoing.get(t, []):
                    dfs(nxt)

        else:
            node_name = node_id_to_name.get(node_id)
            if node_name:
                execution_plan.append({
                    "type": "single",
                    "node": node_name
                })

            for nxt in outgoing.get(node_id, []):
                dfs(nxt)

    # -------- START NODE -------- #
    start_name = blueprint[0].get("nodeName")
    if not start_name:
        raise ValueError("First blueprint node has no nodeName")

    start_id = next(
        (nid for nid, name in node_id_to_name.items() if name == start_name),
        None
    )

    if not start_id:
        raise ValueError(f"Start node '{start_name}' not found in nodes.json")

    dfs(start_id)

    ti.xcom_push(key="execution_plan", value=execution_plan)

    log_to_mongo(
        process_instance_id,
        "Service-Orchestrator",
        f"Execution plan created successfully: {execution_plan}",
        log_type=2
    )
    print(f"Execution plan created: {execution_plan}")


# --------------------------------------- #

def wait_for_aggregator(run_ids, aggregator_setting):
    """
    run_ids = [(dag_id, run_id), ...]
    aggregator_setting:
        0 → strict merge
        1 → partial merge
    """
    completed = set()

    while True:
        for dag_id, run_id in run_ids:
            if run_id in completed:
                continue

            try:
                wait_for_dag_completion(dag_id, run_id, poll_interval=10)
                completed.add(run_id)

                # ✅ PARTIAL MERGE
                if aggregator_setting == 1:
                    return

            except Exception as e:
                # Strict merge fails immediately
                if aggregator_setting == 0:
                    raise
                # Partial merge ignores failures
                continue

        # ✅ STRICT MERGE
        if aggregator_setting == 0 and len(completed) == len(run_ids):
            return

        time.sleep(5)


def execute_execution_plan(**context):
    ti = context["ti"]
    process_instance_id = context["dag_run"].conf.get("id")

    execution_plan = ti.xcom_pull(
        task_ids="build_execution_plan",
        key="execution_plan"
    )

    log_to_mongo(
        process_instance_id,
        "Service-Orchestrator",
        f"Execution plan received: {execution_plan}",
        log_type=0
    )

    print(f"Executing received plan: {execution_plan}")

    for step in execution_plan:

        # ---------- SINGLE ----------
        if step["type"] == "single":
            node = step["node"]

            if node not in NODE_TO_DAG_MAP:
                log_to_mongo(
                    process_instance_id,
                    "Service-Orchestrator",
                    f"No DAG mapped for node '{node}', skipping",
                    log_type=3
                )
                continue

            dag_id = NODE_TO_DAG_MAP[node]

            run_id = trigger_child_dag(
                dag_id=dag_id,
                process_instance_id=process_instance_id,
                parent_node_name=node
            )

            wait_for_dag_completion(dag_id, run_id)


        # ---------- PARALLEL ----------
        elif step["type"] == "parallel":
            log_to_mongo(
                process_instance_id,
                "Router",
                f"Parallel split {step['nodes']}",
                log_type=3
            )

            run_ids = []

            for node in step["nodes"]:
                if node not in NODE_TO_DAG_MAP:
                    log_to_mongo(
                        process_instance_id,
                        "Service-Orchestrator",
                        f"No DAG mapped for node '{node}', skipping",
                        log_type=3
                    )
                    continue

                dag_id = NODE_TO_DAG_MAP[node]

                run_id = trigger_child_dag(
                    dag_id=dag_id,
                    process_instance_id=process_instance_id,
                    parent_node_name=node
                )
                run_ids.append((dag_id, run_id))

            for dag_id, run_id in run_ids:
                wait_for_dag_completion(dag_id, run_id)


            log_to_mongo(
                process_instance_id,
                "Router",
                "Parallel branches triggered",
                log_type=2
            )


# --------------------------------------- #

with DAG(
    dag_id="idp_service_orchestrator",
    default_args=DEFAULT_ARGS,
    start_date=datetime.now() - timedelta(days=1),
    schedule=None,
    catchup=False
) as dag:

    create_tid = PythonOperator(
        task_id="create_process_instance_transaction",
        python_callable=create_process_instance_transaction,
        op_kwargs={"current_stage": "Ingestion"},
    )

    read_graph = PythonOperator(
        task_id="read_blueprint_and_graph",
        python_callable=read_blueprint_and_graph
    )

    build_plan = PythonOperator(
        task_id="build_execution_plan",
        python_callable=build_execution_plan
    )

    execute_plan = PythonOperator(
        task_id="execute_execution_plan",
        python_callable=execute_execution_plan
    )

    create_tid >> read_graph >> build_plan >> execute_plan

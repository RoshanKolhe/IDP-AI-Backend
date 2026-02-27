from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.mysql.hooks.mysql import MySqlHook
from datetime import datetime, timedelta
from dotenv import load_dotenv
from pymongo import MongoClient
import json
import os
import requests
import time

load_dotenv()

LOCAL_DOWNLOAD_DIR = "/opt/airflow/downloaded_docs"
MCP_ENDPOINT = os.getenv("MCP_ENDPOINT", "http://localhost:8002/mcp")
MCP_AUTH_TOKEN = os.getenv("MCP_AUTH_TOKEN")
MCP_TIMEOUT_SECONDS = int(os.getenv("MCP_TIMEOUT_SECONDS", "60"))
MCP_JSONRPC_METHOD = os.getenv("MCP_JSONRPC_METHOD", "tools/call")

MONGO_URI = os.getenv("MONGO_URI")
mongo_client = MongoClient(MONGO_URI)
mongo_collection = mongo_client["idp"]["LogEntry"]


def log_to_mongo(process_instance_id, node_name, message, log_type=1, remark=""):
    try:
        mongo_collection.insert_one({
            "processInstanceId": process_instance_id,
            "nodeName": node_name,
            "logsDescription": message,
            "logType": log_type,
            "isDeleted": False,
            "isActive": True,
            "remark": remark,
            "createdAt": datetime.utcnow()
        })
    except Exception as mongo_err:
        print(f"Failed to log to MongoDB: {mongo_err}")


def _read_json(path, fallback):
    if not os.path.exists(path):
        return fallback
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _fetch_blueprint(process_instance_id, process_instance_dir, cursor):
    blueprint_path = os.path.join(process_instance_dir, "blueprint.json")
    if os.path.exists(blueprint_path):
        return _read_json(blueprint_path, [])

    cursor.execute(
        """
        SELECT b.bluePrint
        FROM ProcessInstances pi
        JOIN Processes p ON p.id = pi.processesId
        JOIN BluePrint b ON b.id = p.bluePrintId
        WHERE pi.id = %s
        """,
        (process_instance_id,)
    )
    row = cursor.fetchone()
    if not row or not row[0]:
        raise ValueError("Blueprint not found for process instance")

    blueprint = json.loads(row[0])
    _write_json(blueprint_path, blueprint)
    return blueprint


def _find_node_component(blueprint, candidate_names):
    for node in blueprint:
        node_name = str(node.get("nodeName", "")).strip().lower()
        if node_name in candidate_names:
            return node.get("component", {}) or {}
    return {}


def _get_or_create_collection_id(process_instance_id, cursor, process_instance_dir):
    context_path = os.path.join(process_instance_dir, "mcp_context.json")
    context_payload = _read_json(context_path, {})

    existing_collection_id = context_payload.get("collection_id")
    if existing_collection_id:
        return existing_collection_id, context_payload, context_path

    cursor.execute("SELECT processesId FROM ProcessInstances WHERE id = %s", (process_instance_id,))
    row = cursor.fetchone()
    if not row:
        raise ValueError(f"Process instance not found: {process_instance_id}")

    process_id = row[0]
    collection_id = f"process-{process_id}-txn-{process_instance_id}"
    context_payload["collection_id"] = collection_id

    _write_json(context_path, context_payload)
    return collection_id, context_payload, context_path


def _invoke_mcp_tool(tool_name, arguments):
    headers = {"Content-Type": "application/json"}
    if MCP_AUTH_TOKEN:
        headers["Authorization"] = f"Bearer {MCP_AUTH_TOKEN}"

    payload_variants = [
        {
            "jsonrpc": "2.0",
            "id": f"{tool_name}-{int(time.time() * 1000)}",
            "method": MCP_JSONRPC_METHOD,
            "params": {"name": tool_name, "arguments": arguments},
        },
        {
            "jsonrpc": "2.0",
            "id": f"{tool_name}-{int(time.time() * 1000)}",
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": arguments},
        },
        {
            "jsonrpc": "2.0",
            "id": f"{tool_name}-{int(time.time() * 1000)}",
            "method": "tool.call",
            "params": {"name": tool_name, "args": arguments},
        },
    ]

    last_error = None
    for payload in payload_variants:
        try:
            response = requests.post(
                MCP_ENDPOINT,
                json=payload,
                headers=headers,
                timeout=MCP_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            body = response.json()
            if body.get("error"):
                raise RuntimeError(f"MCP returned error: {body['error']}")
            return body
        except Exception as exc:
            last_error = exc

    raise RuntimeError(f"MCP call failed for tool '{tool_name}': {last_error}")


def _as_bool(value, default=False):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    if value is None:
        return default
    return bool(value)


def _extract_document_ids(mcp_response):
    candidates = []
    result = mcp_response.get("result", {}) if isinstance(mcp_response, dict) else {}

    if isinstance(result, dict):
        if isinstance(result.get("document_ids"), list):
            candidates.extend(result["document_ids"])
        if isinstance(result.get("tasks"), list):
            for task in result["tasks"]:
                doc_id = task.get("doc_index_id")
                if doc_id:
                    candidates.append(doc_id)

    if isinstance(mcp_response.get("document_ids"), list):
        candidates.extend(mcp_response["document_ids"])

    unique_ids = []
    for doc_id in candidates:
        if isinstance(doc_id, str) and doc_id not in unique_ids:
            unique_ids.append(doc_id)
    return unique_ids


def run_document_index(**context):
    process_instance_id = context["dag_run"].conf.get("id")
    if not process_instance_id:
        raise ValueError("Missing process_instance_id in dag_run.conf")

    process_instance_dir = os.path.join(
        LOCAL_DOWNLOAD_DIR,
        f"process-instance-{process_instance_id}"
    )
    os.makedirs(process_instance_dir, exist_ok=True)

    hook = MySqlHook(mysql_conn_id="idp_mysql")
    conn = hook.get_conn()
    cursor = conn.cursor()

    try:
        cursor.execute(
            """
            UPDATE ProcessInstances
            SET currentStage = %s, isInstanceRunning = 1, updatedAt = NOW()
            WHERE id = %s
            """,
            ("Document Index", process_instance_id),
        )
        conn.commit()

        blueprint = _fetch_blueprint(process_instance_id, process_instance_dir, cursor)
        node_component = _find_node_component(blueprint, {"document index", "index document"})
        if not node_component:
            raise ValueError("Document Index node not found in blueprint")

        collection_id, mcp_context, context_path = _get_or_create_collection_id(
            process_instance_id, cursor, process_instance_dir
        )

        index_mode = (node_component.get("indexMode") or "process_documents").strip()
        document_type = (node_component.get("documentType") or "digital").strip()
        is_contract = _as_bool(node_component.get("isContract", False), default=False)

        if index_mode == "index_enriched_data":
            cleaned_fields_path = os.path.join(process_instance_dir, "cleaned_extracted_fields.json")
            notes = node_component.get("notes")
            if isinstance(notes, str) and notes.strip():
                enriched_text = notes.strip()
            elif os.path.exists(cleaned_fields_path):
                enriched_text = json.dumps(_read_json(cleaned_fields_path, []))
            else:
                enriched_text = f"Indexed from process instance {process_instance_id}"

            tool_name = "index_enriched_data"
            tool_args = {
                "collection_id": collection_id,
                "enriched_text": enriched_text,
            }
        else:
            pdf_files = [
                os.path.join(process_instance_dir, name)
                for name in os.listdir(process_instance_dir)
                if name.lower().endswith(".pdf")
            ]
            if not pdf_files:
                raise FileNotFoundError("No PDF files found for Document Index node")

            tool_name = "process_documents"
            tool_args = {
                "file_paths": pdf_files,
                "collection_id": collection_id,
                "document_type": document_type,
                "is_contract": is_contract,
            }

        log_to_mongo(
            process_instance_id,
            "Document Index",
            f"Calling MCP tool '{tool_name}' with collection_id={collection_id}",
            log_type=0,
        )
        mcp_response = _invoke_mcp_tool(tool_name, tool_args)

        response_path = os.path.join(process_instance_dir, "mcp_document_index_response.json")
        _write_json(response_path, mcp_response)

        document_ids = _extract_document_ids(mcp_response)
        if document_ids:
            mcp_context["document_ids"] = document_ids
        mcp_context["document_index_response_path"] = response_path
        _write_json(context_path, mcp_context)

        log_to_mongo(
            process_instance_id,
            "Document Index",
            f"MCP indexing completed successfully via '{tool_name}'",
            log_type=2,
        )
    except Exception as exc:
        conn.rollback()
        log_to_mongo(
            process_instance_id,
            "Document Index",
            f"Document indexing failed: {type(exc).__name__}: {exc}",
            log_type=1,
            remark="document_index_dag failure",
        )
        raise
    finally:
        cursor.close()
        conn.close()


default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

with DAG(
    dag_id="document_index_dag",
    default_args=default_args,
    start_date=datetime.now() - timedelta(days=1),
    schedule=None,
    catchup=False,
    tags=["idp", "document-index"],
) as dag:
    index_task = PythonOperator(
        task_id="run_document_index",
        python_callable=run_document_index,
    )

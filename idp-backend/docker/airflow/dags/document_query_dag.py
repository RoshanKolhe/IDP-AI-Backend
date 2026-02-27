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


def _extract_document_ids_from_context(mcp_context):
    ids = mcp_context.get("document_ids", [])
    if not isinstance(ids, list):
        return []
    return [item for item in ids if isinstance(item, str)]


def run_document_query(**context):
    process_instance_id = context["dag_run"].conf.get("id")
    if not process_instance_id:
        raise ValueError("Missing process_instance_id in dag_run.conf")

    process_instance_dir = os.path.join(
        LOCAL_DOWNLOAD_DIR,
        f"process-instance-{process_instance_id}"
    )
    os.makedirs(process_instance_dir, exist_ok=True)
    mcp_context_path = os.path.join(process_instance_dir, "mcp_context.json")
    mcp_context = _read_json(mcp_context_path, {})

    collection_id = mcp_context.get("collection_id")
    if not collection_id:
        raise ValueError("collection_id not found. Run Document Index node before Document Query")

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
            ("Document Query", process_instance_id),
        )
        conn.commit()

        blueprint = _fetch_blueprint(process_instance_id, process_instance_dir, cursor)
        node_component = _find_node_component(blueprint, {"document query"})
        if not node_component:
            raise ValueError("Document Query node not found in blueprint")

        query_mode = (node_component.get("queryMode") or "query_documents").strip()
        query_text = (node_component.get("queryText") or "").strip()
        if not query_text:
            raise ValueError("Document Query node is missing query text")

        top_k = int(node_component.get("topK", 5) or 5)
        use_graph = _as_bool(node_component.get("useGraph", True), default=True)
        document_ids = _extract_document_ids_from_context(mcp_context)

        tool_args = {"query": query_text}

        if query_mode == "query_collection":
            tool_args["collection_id"] = collection_id
        elif query_mode in {
            "query_documents",
            "query_documents_hybrid",
            "query_documents_graph",
            "query_documents_by_toc",
        }:
            if not document_ids:
                raise ValueError(
                    f"Query mode '{query_mode}' requires document_ids, but none are available in context"
                )
            tool_args["document_ids"] = document_ids
            if query_mode in {"query_documents", "query_documents_hybrid"}:
                tool_args["use_graph"] = use_graph
        elif query_mode == "query_enriched_index":
            tool_args["collection_id"] = collection_id
            tool_args["top_k"] = top_k
        else:
            raise ValueError(f"Unsupported query mode '{query_mode}'")

        log_to_mongo(
            process_instance_id,
            "Document Query",
            f"Calling MCP tool '{query_mode}' with collection_id={collection_id}",
            log_type=0,
        )
        mcp_response = _invoke_mcp_tool(query_mode, tool_args)

        response_path = os.path.join(process_instance_dir, "mcp_document_query_response.json")
        _write_json(response_path, mcp_response)

        mcp_context["document_query_response_path"] = response_path
        mcp_context["last_query_mode"] = query_mode
        _write_json(mcp_context_path, mcp_context)

        log_to_mongo(
            process_instance_id,
            "Document Query",
            f"MCP query completed successfully via '{query_mode}'",
            log_type=2,
        )
    except Exception as exc:
        conn.rollback()
        log_to_mongo(
            process_instance_id,
            "Document Query",
            f"Document query failed: {type(exc).__name__}: {exc}",
            log_type=1,
            remark="document_query_dag failure",
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
    dag_id="document_query_dag",
    default_args=default_args,
    start_date=datetime.now() - timedelta(days=1),
    schedule=None,
    catchup=False,
    tags=["idp", "document-query"],
) as dag:
    query_task = PythonOperator(
        task_id="run_document_query",
        python_callable=run_document_query,
    )

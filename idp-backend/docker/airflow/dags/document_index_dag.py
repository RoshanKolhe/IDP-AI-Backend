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
MCP_ENDPOINT = os.getenv("MCP_ENDPOINT", "http://13.203.33.247:8002/mcp")
MCP_AUTH_TOKEN = os.getenv("MCP_AUTH_TOKEN")
MCP_TIMEOUT_SECONDS = int(os.getenv("MCP_TIMEOUT_SECONDS", "60"))
MCP_JSONRPC_METHOD = os.getenv("MCP_JSONRPC_METHOD", "tools/call")
MCP_PROTOCOL_VERSION = os.getenv("MCP_PROTOCOL_VERSION", "2024-11-05")
PROCESSING_BASE_URL = os.getenv("PROCESSING_BASE_URL", "http://13.203.33.247:8002")
DOCUMENT_INDEX_STATUS_POLL_INTERVAL_SECONDS = int(
    os.getenv("DOCUMENT_INDEX_STATUS_POLL_INTERVAL_SECONDS", "10")
)
DOCUMENT_INDEX_STATUS_TIMEOUT_SECONDS = int(
    os.getenv("DOCUMENT_INDEX_STATUS_TIMEOUT_SECONDS", "900")
)

MONGO_URI = os.getenv("MONGO_URI")
mongo_client = MongoClient(MONGO_URI)
mongo_collection = mongo_client["idp"]["LogEntry"]

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


def log_to_mongo(process_instance_id, node_name, message, log_type=1, remark=""):
    try:
        mongo_collection.insert_one({
            "processInstanceId": process_instance_id,
            "processInstanceTransactionId": _get_transaction_id(process_instance_id),
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


def _get_process_id(process_instance_id, cursor):
    cursor.execute("SELECT processesId FROM ProcessInstances WHERE id = %s", (process_instance_id,))
    row = cursor.fetchone()
    if not row:
        raise ValueError(f"Process instance not found: {process_instance_id}")
    return str(row[0])


def _parse_mcp_response(response):
    try:
        return response.json()
    except ValueError:
        for raw_line in response.text.splitlines():
            line = raw_line.strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if not data:
                continue
            try:
                return json.loads(data)
            except json.JSONDecodeError:
                continue
    raise RuntimeError("Unable to parse MCP response body")


def _initialize_mcp_session():
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }
    if MCP_AUTH_TOKEN:
        headers["Authorization"] = f"Bearer {MCP_AUTH_TOKEN}"

    payload = {
        "jsonrpc": "2.0",
        "id": f"initialize-{int(time.time() * 1000)}",
        "method": "initialize",
        "params": {
            "protocolVersion": MCP_PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {
                "name": "airflow-client",
                "version": "1.0",
            },
        },
    }

    response = requests.post(
        MCP_ENDPOINT,
        json=payload,
        headers=headers,
        timeout=MCP_TIMEOUT_SECONDS,
    )
    response.raise_for_status()

    mcp_session_id = response.headers.get("mcp-session-id")
    if not mcp_session_id:
        raise RuntimeError("MCP initialize succeeded but mcp-session-id header is missing")

    body = _parse_mcp_response(response)
    if body.get("error"):
        raise RuntimeError(f"MCP initialize returned error: {body['error']}")

    return mcp_session_id, body


def _get_processing_base_url():
    if PROCESSING_BASE_URL:
        return PROCESSING_BASE_URL.rstrip("/")

    endpoint = MCP_ENDPOINT.rstrip("/")
    if endpoint.endswith("/mcp"):
        return endpoint[:-4]
    return endpoint


def _invoke_mcp_tool(tool_name, arguments, mcp_session_id):
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream"
    }
    if MCP_AUTH_TOKEN:
        headers["Authorization"] = f"Bearer {MCP_AUTH_TOKEN}"
    if mcp_session_id:
        headers["mcp-session-id"] = mcp_session_id

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
        print("tool_name : ", tool_name)
        print("arguments : ", arguments)
        print("payload on invoke mcp : ", payload)
        try:
            response = requests.post(
                MCP_ENDPOINT,
                json=payload,
                headers=headers,
                timeout=MCP_TIMEOUT_SECONDS,
            )
            print("response on invoke mcp : ", response)
            response.raise_for_status()
            body = _parse_mcp_response(response)
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


def _as_mcp_bool_string(value, default=False):
    return "true" if _as_bool(value, default=default) else "false"


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


def _extract_tasks(mcp_response):
    result = mcp_response.get("result", {}) if isinstance(mcp_response, dict) else {}
    tasks = result.get("tasks", []) if isinstance(result, dict) else []
    if not isinstance(tasks, list):
        return []
    return [task for task in tasks if isinstance(task, dict)]


def _upload_documents_via_http(file_paths, process_id, document_type, is_contract):
    headers = {}
    if MCP_AUTH_TOKEN:
        headers["Authorization"] = f"Bearer {MCP_AUTH_TOKEN}"

    data = {
        "process_id": process_id,
        "doc_type": document_type,
        "is_contract": is_contract,
    }

    upload_url = f"{_get_processing_base_url()}/processing/upload-with-process-id/"
    file_handles = []
    try:
        files = []
        for file_path in file_paths:
            file_handle = open(file_path, "rb")
            file_handles.append(file_handle)
            files.append(("files", (os.path.basename(file_path), file_handle, "application/pdf")))

        response = requests.post(
            upload_url,
            headers=headers,
            data=data,
            files=files,
            timeout=MCP_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        print('response of http upload : ', response.json())        
        return response.json()
    finally:
        for file_handle in file_handles:
            try:
                file_handle.close()
            except Exception:
                pass


def _extract_mcp_tool_error(mcp_response):
    if not isinstance(mcp_response, dict):
        return ""

    result = mcp_response.get("result")
    if not isinstance(result, dict):
        return ""
    if not result.get("isError"):
        return ""

    content = result.get("content", [])
    error_lines = []
    if isinstance(content, list):
        for item in content:
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            if isinstance(text, str) and text.strip():
                error_lines.append(text.strip())

    if error_lines:
        return "\n".join(error_lines)
    return "MCP tool returned isError=true"


def _raise_if_mcp_tool_error(mcp_response):
    error_text = _extract_mcp_tool_error(mcp_response)
    if error_text:
        raise RuntimeError(error_text)


def _should_retry_with_upload(error_text):
    if not isinstance(error_text, str):
        return False

    lowered = error_text.lower()
    retry_markers = [
        "uploadfile",
        "expected uploadfile",
        "file_paths",
        "files.0",
        "no such file",
        "not found",
        "does not exist",
        "cannot access",
        "permission denied",
    ]
    return any(marker in lowered for marker in retry_markers)


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

        process_id = _get_process_id(process_instance_id, cursor)
        collection_id, mcp_context, context_path = _get_or_create_collection_id(
            process_instance_id, cursor, process_instance_dir
        )
        mcp_context["process_id"] = process_id
        mcp_session_id, initialize_response = _initialize_mcp_session()
        mcp_context["mcp_session_id"] = mcp_session_id
        mcp_context["mcp_initialize_response"] = initialize_response

        index_mode = (node_component.get("indexMode") or "process_documents").strip()
        document_type = (node_component.get("documentType") or "digital").strip()
        is_contract = _as_mcp_bool_string(node_component.get("isContract", False), default=False)

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
                "process_id": process_id,
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

            tool_name = "upload_documents"
            tool_args = {
                "file_paths": pdf_files,
                "process_id": process_id,
                "doc_type": document_type,
                "is_contract": is_contract,
            }

        log_to_mongo(
            process_instance_id,
            "Document Index",
            f"Calling MCP tool '{tool_name}' with process_id={process_id}, collection_id={collection_id} and mcp_session_id={mcp_session_id}",
            log_type=0,
        )
        if tool_name == "upload_documents":
            log_to_mongo(
                process_instance_id,
                "Document Index",
                f"Uploading {len(pdf_files)} file(s) via HTTP to processing service",
                log_type=0,
            )
            http_upload_response = _upload_documents_via_http(
                pdf_files,
                process_id,
                document_type,
                is_contract,
            )
            mcp_response = {"result": http_upload_response}

            tasks = _extract_tasks(mcp_response)
            if not tasks:
                raise RuntimeError("upload_documents succeeded but returned no tasks")

            upload_tasks = tasks
            mcp_context["upload_tasks"] = upload_tasks

            process_document_ids = []
            for task in upload_tasks:
                doc_index_id = str(task.get("doc_index_id", "")).strip()
                if not doc_index_id:
                    raise RuntimeError("upload_documents task missing doc_index_id")

                process_args = {
                    "process_id": process_id,
                    "doc_index_id": doc_index_id,
                    "document_type": document_type,
                    "is_contract": is_contract,
                }
                log_to_mongo(
                    process_instance_id,
                    "Document Index",
                    f"Calling MCP tool 'process_documents' for doc_index_id={doc_index_id}",
                    log_type=0,
                )
                process_response = _invoke_mcp_tool("process_documents", process_args, mcp_session_id)
                print('process response : ', process_response)
                _raise_if_mcp_tool_error(process_response)
                process_result = process_response.get("result", {}) if isinstance(process_response, dict) else {}
                structured_content = (
                    process_result.get("structuredContent", {}) if isinstance(process_result, dict) else {}
                )
                documents = structured_content.get("documents", []) if isinstance(structured_content, dict) else []
                if not isinstance(documents, list):
                    continue

                for document in documents:
                    if not isinstance(document, dict):
                        continue
                    document_id = str(document.get("document_id", "")).strip()
                    document_status = str(document.get("status", "")).strip().lower()
                    if document_status == "failed":
                        raise RuntimeError(
                            f"Document processing failed for doc_index_id={doc_index_id}: {document.get('error')}"
                        )
                    if document_id and document_id not in process_document_ids:
                        process_document_ids.append(document_id)

            if process_document_ids:
                result = mcp_response.setdefault("result", {})
                if isinstance(result, dict):
                    result["document_ids"] = process_document_ids
        else:
            mcp_response = _invoke_mcp_tool(tool_name, tool_args, mcp_session_id)
            _raise_if_mcp_tool_error(mcp_response)

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

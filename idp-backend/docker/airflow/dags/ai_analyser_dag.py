from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.mysql.hooks.mysql import MySqlHook
from datetime import datetime, timedelta
from dotenv import load_dotenv
from pymongo import MongoClient
import json
import os
import re
import requests
import time

load_dotenv()

LOCAL_DOWNLOAD_DIR = "/opt/airflow/downloaded_docs"
MCP_ENDPOINT = os.getenv("MCP_ENDPOINT", "http://13.203.33.247:8002/mcp")
MCP_AUTH_TOKEN = os.getenv("MCP_AUTH_TOKEN")
MCP_TIMEOUT_SECONDS = int(os.getenv("MCP_TIMEOUT_SECONDS", "60"))
MCP_JSONRPC_METHOD = os.getenv("MCP_JSONRPC_METHOD", "tools/call")
MCP_PROTOCOL_VERSION = os.getenv("MCP_PROTOCOL_VERSION", "2024-11-05")

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


def _find_node_component(blueprint):
    candidate_names = {"ai analyser", "ai analyzer"}
    for node in blueprint:
        node_name = str(node.get("nodeName", "")).strip().lower()
        if node_name in candidate_names:
            return node.get("component", {}) or {}
    return {}


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


def _invoke_mcp_tool(tool_name, arguments, mcp_session_id):
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
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
        try:
            response = requests.post(
                MCP_ENDPOINT,
                json=payload,
                headers=headers,
                timeout=MCP_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            body = _parse_mcp_response(response)
            if body.get("error"):
                raise RuntimeError(f"MCP returned error: {body['error']}")
            return body
        except Exception as exc:
            last_error = exc

    raise RuntimeError(f"MCP call failed for tool '{tool_name}': {last_error}")


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
    return "\n".join(error_lines) if error_lines else "MCP tool returned isError=true"


def _raise_if_mcp_tool_error(mcp_response):
    error_text = _extract_mcp_tool_error(mcp_response)
    if error_text:
        raise RuntimeError(error_text)


def _to_bool(value, default=False):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    if value is None:
        return default
    return bool(value)


def _clamp_int(value, default, min_value=None, max_value=None):
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = int(default)

    if min_value is not None and parsed < min_value:
        parsed = min_value
    if max_value is not None and parsed > max_value:
        parsed = max_value
    return parsed


def _split_lines(value):
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in str(value).splitlines() if item.strip()]


def _normalize_key(value):
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def _find_value_by_keys(payload, candidate_keys):
    normalized_candidates = {_normalize_key(key) for key in candidate_keys}

    def search(value):
        if isinstance(value, dict):
            for key, nested_value in value.items():
                if _normalize_key(key) in normalized_candidates and nested_value not in (None, "", [], {}):
                    return nested_value
                found = search(nested_value)
                if found not in (None, "", [], {}):
                    return found
        elif isinstance(value, list):
            for item in value:
                found = search(item)
                if found not in (None, "", [], {}):
                    return found
        return None

    return search(payload)


def _collect_person_names(payload):
    names = []

    def visit(value):
        if isinstance(value, dict):
            for key, nested_value in value.items():
                normalized = _normalize_key(key)
                if normalized in {
                    "directorname",
                    "directorsname",
                    "director",
                    "directors",
                    "ownername",
                    "ownersname",
                    "owner",
                    "owners",
                    "shareholdername",
                    "shareholdersname",
                    "personname",
                    "name",
                }:
                    if isinstance(nested_value, str):
                        stripped = nested_value.strip()
                        if stripped and len(stripped) > 2:
                            names.append(stripped)
                    elif isinstance(nested_value, list):
                        for item in nested_value:
                            if isinstance(item, str) and item.strip():
                                names.append(item.strip())
                            elif isinstance(item, dict):
                                item_name = item.get("name")
                                if isinstance(item_name, str) and item_name.strip():
                                    names.append(item_name.strip())
                visit(nested_value)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(payload)

    deduped = []
    seen = set()
    for item in names:
        normalized = item.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(item)
    return deduped


def _load_upstream_payload(process_instance_dir, mcp_context):
    payload = {}

    cleaned_fields_path = os.path.join(process_instance_dir, "cleaned_extracted_fields.json")
    if os.path.exists(cleaned_fields_path):
        payload["cleaned_extracted_fields"] = _read_json(cleaned_fields_path, [])

    extracted_fields_path = os.path.join(process_instance_dir, "extracted_fields.json")
    if os.path.exists(extracted_fields_path):
        payload["extracted_fields"] = _read_json(extracted_fields_path, [])

    response_paths = []
    if isinstance(mcp_context.get("external_data_sources_response_path"), str):
        response_paths.append(mcp_context["external_data_sources_response_path"])
    if isinstance(mcp_context.get("document_query_response_path"), str):
        response_paths.append(mcp_context["document_query_response_path"])
    if isinstance(mcp_context.get("integration_response_path"), str):
        response_paths.append(mcp_context["integration_response_path"])
    if isinstance(mcp_context.get("code_node_response_path"), str):
        response_paths.append(mcp_context["code_node_response_path"])

    upstream_responses = []
    for path in response_paths:
        if os.path.exists(path):
            upstream_responses.append(_read_json(path, {}))

    payload["upstream_responses"] = upstream_responses
    payload["mcp_context"] = mcp_context
    return payload


def _resolve_runtime_values(component, process_instance_dir, mcp_context):
    upstream_payload = _load_upstream_payload(process_instance_dir, mcp_context)

    company_name = str(component.get("riskCompanyName") or component.get("newsSubjectCompany") or "").strip()
    company_id = str(component.get("riskCompanyId") or "").strip()
    cin = str(component.get("riskCin") or "").strip()
    news_query = str(component.get("newsQuery") or "").strip()
    subject_persons = _split_lines(component.get("newsSubjectPersons"))
    director_names = _split_lines(component.get("riskDirectorNames"))

    if _to_bool(component.get("usePreviousNodeData"), default=False):
        for payload in upstream_payload.values():
            if not company_name:
                company_name = str(
                    _find_value_by_keys(payload, [
                        "company_name", "companyName", "legal_name", "entity_name", "applicant_name"
                    ]) or ""
                ).strip()
            if not company_id:
                company_id = str(
                    _find_value_by_keys(payload, [
                        "company_id", "companyId"
                    ]) or ""
                ).strip()
            if not cin:
                cin = str(_find_value_by_keys(payload, ["cin", "company_cin"]) or "").strip()
            if not news_query:
                news_query = str(
                    _find_value_by_keys(payload, ["query", "company_name", "companyName", "entity_name"]) or ""
                ).strip()
            if not subject_persons:
                subject_persons = _collect_person_names(payload)
            if not director_names:
                director_names = _collect_person_names(payload)

    if not news_query:
        news_query = company_name

    return {
        "company_name": company_name,
        "company_id": company_id,
        "cin": cin,
        "news_query": news_query,
        "subject_persons": subject_persons,
        "director_names": director_names,
    }


def _write_ai_analyser_response(process_instance_dir, analysis_type, payload):
    response_path = os.path.join(process_instance_dir, f"ai_analyser_{analysis_type}_response.json")
    _write_json(response_path, payload)

    mcp_context_path = os.path.join(process_instance_dir, "mcp_context.json")
    mcp_context = _read_json(mcp_context_path, {})
    mcp_context["ai_analyser_response_path"] = response_path
    mcp_context["ai_analyser_type"] = analysis_type
    mcp_context["ai_analyser_request"] = payload.get("request", {})
    mcp_context["ai_analyser_runtime_values"] = payload.get("runtimeValues", {})
    mcp_context["ai_analyser_result"] = payload.get("response", {})
    _write_json(mcp_context_path, mcp_context)
    return response_path


def _run_news_extraction(component, runtime_values, mcp_session_id, process_instance_id, process_instance_dir):
    query = runtime_values["news_query"]
    subject_company = runtime_values["company_name"] or str(component.get("newsSubjectCompany") or "").strip()
    subject_persons = runtime_values["subject_persons"]
    lookback_days = _clamp_int(component.get("newsLookbackDays", 365), 365, 30, 730)
    num_results = _clamp_int(component.get("newsNumResults", 10), 10, 1, 50)
    source = str(component.get("newsSource") or "newsapi").strip() or "newsapi"

    if not query:
      raise ValueError("News Extraction requires a query")
    if not subject_company:
      raise ValueError("News Extraction requires a subject company")

    tool_args = {
        "query": query,
        "subject_company": subject_company,
        "subject_persons": subject_persons or None,
        "lookback_days": lookback_days,
        "num_results": num_results,
        "source": source,
    }

    log_to_mongo(
        process_instance_id,
        "AI Analyser",
        f"Running MCP news analysis for '{subject_company}' with source '{source}'",
        log_type=0,
    )

    mcp_response = _invoke_mcp_tool("run_full_news_analysis", tool_args, mcp_session_id)
    _raise_if_mcp_tool_error(mcp_response)

    payload = {
        "analysisType": "newsExtraction",
        "toolName": "run_full_news_analysis",
        "runtimeValues": runtime_values,
        "request": tool_args,
        "response": mcp_response,
    }
    _write_ai_analyser_response(process_instance_dir, "news_extraction", payload)

    log_to_mongo(
        process_instance_id,
        "AI Analyser",
        "News extraction completed successfully",
        log_type=2,
    )


def _run_risk_assessment(component, runtime_values, mcp_session_id, process_instance_id, process_instance_dir):
    company_id = runtime_values["company_id"]
    company_name = runtime_values["company_name"]
    cin = runtime_values["cin"] or None
    director_names = runtime_values["director_names"] or None
    check_directors_litigation = _to_bool(component.get("checkDirectorsLitigation"), default=True)
    check_directors_sanctions = _to_bool(component.get("checkDirectorsSanctions"), default=True)

    if not company_id:
        raise ValueError("Risk Assessment requires company_id")
    if not company_name:
        raise ValueError("Risk Assessment requires company_name")

    tool_args = {
        "company_id": company_id,
        "company_name": company_name,
        "cin": cin,
        "director_names": director_names,
        "check_directors_litigation": check_directors_litigation,
        "check_directors_sanctions": check_directors_sanctions,
    }

    log_to_mongo(
        process_instance_id,
        "AI Analyser",
        f"Running MCP risk assessment for '{company_name}'",
        log_type=0,
    )

    mcp_response = _invoke_mcp_tool("run_full_background_check", tool_args, mcp_session_id)
    _raise_if_mcp_tool_error(mcp_response)

    payload = {
        "analysisType": "riskAssessment",
        "toolName": "run_full_background_check",
        "runtimeValues": runtime_values,
        "request": tool_args,
        "response": mcp_response,
    }
    _write_ai_analyser_response(process_instance_dir, "risk_assessment", payload)

    log_to_mongo(
        process_instance_id,
        "AI Analyser",
        "Risk assessment completed successfully",
        log_type=2,
    )


def run_ai_analyser(**context):
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
    mcp_session_id, initialize_response = _initialize_mcp_session()
    mcp_context["mcp_session_id"] = mcp_session_id
    mcp_context["mcp_initialize_response"] = initialize_response
    _write_json(mcp_context_path, mcp_context)

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
            ("AI Analyser", process_instance_id),
        )
        conn.commit()

        blueprint = _fetch_blueprint(process_instance_id, process_instance_dir, cursor)
        component = _find_node_component(blueprint)
        if not component:
            raise ValueError("AI Analyser node not found in blueprint")

        analysis_type = str(component.get("analysisType") or "newsExtraction").strip()
        runtime_values = _resolve_runtime_values(component, process_instance_dir, mcp_context)

        if analysis_type == "newsExtraction":
            _run_news_extraction(
                component,
                runtime_values,
                mcp_session_id,
                process_instance_id,
                process_instance_dir,
            )
        elif analysis_type == "riskAssessment":
            _run_risk_assessment(
                component,
                runtime_values,
                mcp_session_id,
                process_instance_id,
                process_instance_dir,
            )
        else:
            raise ValueError(f"Unsupported AI Analyser analysisType '{analysis_type}'")

    except Exception as exc:
        conn.rollback()
        log_to_mongo(
            process_instance_id,
            "AI Analyser",
            f"AI Analyser failed: {type(exc).__name__}: {exc}",
            log_type=1,
            remark="ai_analyser_dag failure",
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
    "retries": 0,
}


with DAG(
    dag_id="ai_analyser_dag",
    default_args=default_args,
    description="Run AI Analyser node using MCP news and risk tools",
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["process", "ai-analyser"],
) as dag:

    run_task = PythonOperator(
        task_id="run_ai_analyser",
        python_callable=run_ai_analyser,
    )

    run_task

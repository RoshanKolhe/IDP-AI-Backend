from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.mysql.hooks.mysql import MySqlHook
from datetime import datetime, timedelta
from dotenv import load_dotenv
from pymongo import MongoClient
import json
import os
import requests
from urllib.parse import urlencode

load_dotenv()

LOCAL_DOWNLOAD_DIR = "/opt/airflow/downloaded_docs"
INTEGRATION_TIMEOUT_SECONDS = int(os.getenv("INTEGRATION_TIMEOUT_SECONDS", "60"))

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


def _find_node_component(blueprint):
    for node in blueprint:
        node_name = str(node.get("nodeName", "")).strip().lower()
        if node_name in {"integration", "integration step"}:
            return node.get("component", {}) or {}
    return {}


def _to_dict(items):
    out = {}
    if not isinstance(items, list):
        return out
    for item in items:
        key = str(item.get("key", "")).strip()
        if not key:
            continue
        out[key] = str(item.get("value", ""))
    return out


def _method_name(method):
    method_map = {
        1: "GET",
        2: "POST",
        3: "PUT",
        4: "PATCH",
        5: "DELETE",
        "GET": "GET",
        "POST": "POST",
        "PUT": "PUT",
        "PATCH": "PATCH",
        "DELETE": "DELETE",
    }
    return method_map.get(method, method_map.get(str(method).upper(), "GET"))


def _apply_path_params(url, params_map):
    result = url
    for key, value in params_map.items():
        result = result.replace(f"{{{{{key}}}}}", value)
        result = result.replace(f":{key}", value)
    return result


def _build_body(component):
    body_type = component.get("bodyType")
    content_type = component.get("contentType")

    if body_type == 1:
        raw_content = component.get("requestContent", "")
        if content_type == 1:
            try:
                return {"json": json.loads(raw_content)}
            except Exception:
                return {"json": {"raw": raw_content}}
        return {"data": raw_content}

    if body_type == 2:
        return {"data": _to_dict(component.get("urlEncodedFields", []))}

    return {}


def run_integration(**context):
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
            ("Integration", process_instance_id),
        )
        conn.commit()

        blueprint = _fetch_blueprint(process_instance_id, process_instance_dir, cursor)
        component = _find_node_component(blueprint)
        if not component:
            raise ValueError("Integration node not found in blueprint")

        method = _method_name(component.get("method", 1))
        url = str(component.get("url", "")).strip()
        if not url:
            raise ValueError("Integration URL is missing")

        headers = _to_dict(component.get("headers", []))
        query_params = _to_dict(component.get("queryStrings", []))
        path_params = _to_dict(component.get("paramsValue", []))

        final_url = _apply_path_params(url, path_params)
        if query_params:
            separator = "&" if "?" in final_url else "?"
            final_url = f"{final_url}{separator}{urlencode(query_params)}"

        body_payload = _build_body(component)

        log_to_mongo(
            process_instance_id,
            "Integration",
            f"Calling external API {method} {final_url}",
            log_type=0,
        )

        response = requests.request(
            method=method,
            url=final_url,
            headers=headers,
            timeout=INTEGRATION_TIMEOUT_SECONDS,
            **body_payload,
        )

        response_payload = {
            "status_code": response.status_code,
            "headers": dict(response.headers),
            "body": None,
        }
        try:
            response_payload["body"] = response.json()
        except Exception:
            response_payload["body"] = response.text

        response_path = os.path.join(process_instance_dir, "integration_response.json")
        _write_json(response_path, response_payload)

        mcp_context_path = os.path.join(process_instance_dir, "mcp_context.json")
        mcp_context = _read_json(mcp_context_path, {})
        mcp_context["integration_response_path"] = response_path
        mcp_context["last_integration_status_code"] = response.status_code
        _write_json(mcp_context_path, mcp_context)

        if response.ok:
            log_to_mongo(
                process_instance_id,
                "Integration",
                f"Integration call succeeded with status {response.status_code}",
                log_type=2,
            )
        else:
            raise RuntimeError(f"Integration API returned status {response.status_code}")

    except Exception as exc:
        conn.rollback()
        log_to_mongo(
            process_instance_id,
            "Integration",
            f"Integration failed: {type(exc).__name__}: {exc}",
            log_type=1,
            remark="integration_dag failure",
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
    dag_id="integration_dag",
    default_args=default_args,
    start_date=datetime.now() - timedelta(days=1),
    schedule=None,
    catchup=False,
    tags=["idp", "integration"],
) as dag:
    integration_task = PythonOperator(
        task_id="run_integration",
        python_callable=run_integration,
    )

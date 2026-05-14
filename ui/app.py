from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from flask import Flask, jsonify, render_template_string, request


app = Flask(__name__)
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = "scripts/adapters/real_adapters_catalog.json"
TASK_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


HTML = """
<!doctype html>
<html>
  <head>
    <meta charset=\"utf-8\" />
    <title>Agent4Mat UI Prototype</title>
    <style>
      body { font-family: -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif; margin: 1.2rem; background: #f6f7fb; color: #111827; }
      .wrap { max-width: 1120px; margin: 0 auto; display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
      .card { background: white; border: 1px solid #e5e7eb; border-radius: 10px; padding: 1rem 1.2rem; }
      label { display: block; margin-top: 0.8rem; font-weight: 600; }
      input, textarea, select { width: 100%; margin-top: 0.3rem; padding: 0.5rem; }
      button { margin-top: 0.8rem; padding: 0.5rem 0.9rem; margin-right: 0.5rem; }
      pre { background: #f7f7f7; padding: 0.8rem; border-radius: 8px; overflow-x: auto; }
      h2 { margin-top: 0.2rem; }
      .full { grid-column: 1 / -1; }
    </style>
  </head>
  <body>
    <div class=\"wrap\">
      <div class=\"card\">
        <h2>Full Pipeline</h2>
        <p>Run <code>agent-run-json</code> from a request payload.</p>
        <label>Catalog path</label>
        <input id=\"catalog\" value=\"scripts/adapters/real_adapters_catalog.json\" />
        <label>Planner provider</label>
        <select id=\"planner\">
          <option value=\"rule_based_v1\">rule_based_v1</option>
          <option value=\"llm_v1\">llm_v1</option>
        </select>
        <label>Request JSON</label>
        <textarea id=\"payload\" rows=\"12\">{
  \"task_id\": \"ui_task_demo\",
  \"request_text\": \"设计470nm附近且高PLQY分子\",
  \"mode\": \"fast_screen\",
  \"targets\": [{\"property\": \"plqy\", \"objective\": \"maximize\", \"target_value\": 0.6}],
  \"budget\": {\"max_candidates\": 10}
}</textarea>
        <button onclick=\"runTask()\">Run</button>
      </div>

      <div class=\"card\">
        <h2>Single Step</h2>
        <p>Run <code>agent-run-step-json</code> for one operation.</p>
        <label>Step request JSON</label>
        <textarea id=\"step_payload\" rows=\"12\">{
  \"task\": {
    \"task_id\": \"ui_task_step_demo\",
    \"request_text\": \"单步测试\",
    \"domain\": \"oled_molecule_design\",
    \"targets\": [{\"name\": \"plqy\", \"objective\": \"maximize\", \"target_center\": 0.6, \"sigma\": 0.2, \"weight\": 1.0}],
    \"constraints\": {\"mw_min\": 150, \"mw_max\": 700, \"domain_threshold\": 0.2, \"banned_alerts\": []},
    \"model_choice\": {\"predictor_id\": \"unimol_lambda_plqy_v1\", \"generator_id\": \"reinvent4_lambda_em_v2\"},
    \"budget\": {\"max_candidates\": 10},
    \"dataset_preferences\": [\"master_database\"]
  },
  \"operation\": \"search_dataset\",
  \"args\": {\"preferences\": [\"master_database\"], \"use_web_search\": true, \"web_topk\": 3}
}</textarea>
        <button onclick=\"runStep()\">Run Step</button>
      </div>

      <div class=\"card\">
        <h2>Task Intake</h2>
        <p>Run <code>agent-intake</code> for target clarification.</p>
        <label>Task ID</label>
        <input id=\"intake_task_id\" value=\"ui_intake_demo\" />
        <label>Request text</label>
        <textarea id=\"intake_request\" rows=\"4\">设计470nm附近且高PLQY分子</textarea>
        <label>Web topk</label>
        <input id=\"intake_web_topk\" value=\"5\" />
        <button onclick=\"runIntake()\">Run Intake</button>
      </div>

      <div class=\"card\">
        <h2>Task Approve</h2>
        <p>Run <code>agent-approve</code> from intake draft task JSON.</p>
        <label>Task JSON path</label>
        <input id=\"approve_task_json_path\" value=\"runs/agent/ui_intake_demo/task.draft.json\" />
        <button onclick=\"runApprove()\">Run Approve</button>
      </div>

      <div class=\"card\">
        <h2>Task Resume</h2>
        <p>Run <code>agent-resume</code> for resumable task runs.</p>
        <label>Task ID</label>
        <input id=\"resume_task_id\" value=\"ui_task_demo\" />
        <button onclick=\"runResume()\">Run Resume</button>
      </div>

      <div class=\"card\">
        <h2>Task Inspector</h2>
        <p>Preview key artifacts under <code>runs/agent/&lt;task_id&gt;</code>.</p>
        <label>Task ID</label>
        <input id=\"inspect_task_id\" value=\"ui_task_demo\" />
        <button onclick=\"inspectTask()\">Load</button>
      </div>

      <div class=\"card full\">
        <h2>Result</h2>
        <pre id=\"out\">(waiting)</pre>
      </div>
    </div>
    <script>
      async function postJSON(url, payload) {
        const resp = await fetch(url, {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify(payload)
        });
        const data = await resp.json();
        return {status: resp.status, data};
      }

      async function runTask() {
        const out = document.getElementById('out');
        out.textContent = 'running...';
        const payloadText = document.getElementById('payload').value;
        const planner = document.getElementById('planner').value;
        const catalog = document.getElementById('catalog').value;
        const r = await postJSON('/api/run', {payload_text: payloadText, planner_provider: planner, catalog_path: catalog});
        out.textContent = JSON.stringify(r.data, null, 2);
      }

      async function runStep() {
        const out = document.getElementById('out');
        out.textContent = 'running step...';
        const payloadText = document.getElementById('step_payload').value;
        const catalog = document.getElementById('catalog').value;
        const r = await postJSON('/api/run-step', {payload_text: payloadText, catalog_path: catalog});
        out.textContent = JSON.stringify(r.data, null, 2);
      }

      async function runIntake() {
        const out = document.getElementById('out');
        out.textContent = 'running intake...';
        const taskId = document.getElementById('intake_task_id').value;
        const requestText = document.getElementById('intake_request').value;
        const webTopk = Number(document.getElementById('intake_web_topk').value || 5);
        const r = await postJSON('/api/intake', {task_id: taskId, request_text: requestText, web_topk: webTopk});
        const result = r.data && r.data.result ? r.data.result : null;
        if (result && result.task_draft_path) {
          document.getElementById('approve_task_json_path').value = result.task_draft_path;
        }
        document.getElementById('inspect_task_id').value = taskId;
        document.getElementById('resume_task_id').value = taskId;
        out.textContent = JSON.stringify(r.data, null, 2);
      }

      async function runApprove() {
        const out = document.getElementById('out');
        out.textContent = 'running approve...';
        const taskJsonPath = document.getElementById('approve_task_json_path').value;
        const planner = document.getElementById('planner').value;
        const catalog = document.getElementById('catalog').value;
        const r = await postJSON('/api/approve', {task_json_path: taskJsonPath, planner_provider: planner, catalog_path: catalog});
        out.textContent = JSON.stringify(r.data, null, 2);
      }

      async function runResume() {
        const out = document.getElementById('out');
        out.textContent = 'running resume...';
        const taskId = document.getElementById('resume_task_id').value;
        const planner = document.getElementById('planner').value;
        const catalog = document.getElementById('catalog').value;
        const r = await postJSON('/api/resume', {task_id: taskId, planner_provider: planner, catalog_path: catalog});
        out.textContent = JSON.stringify(r.data, null, 2);
      }

      async function inspectTask() {
        const out = document.getElementById('out');
        out.textContent = 'loading...';
        const taskId = document.getElementById('inspect_task_id').value;
        const resp = await fetch(`/api/task/${encodeURIComponent(taskId)}/summary`);
        const data = await resp.json();
        out.textContent = JSON.stringify(data, null, 2);
      }
    </script>
  </body>
</html>
"""


def _resolve_catalog(catalog_path: str) -> Path:
    p = Path(str(catalog_path or DEFAULT_CATALOG))
    if not p.is_absolute():
        p = (REPO_ROOT / p).resolve()
    else:
        p = p.resolve()
    return p


def _run_cli_with_json_payload(
    *,
    cli_base_args: List[str],
    payload: Dict[str, Any],
    payload_filename: str,
    payload_arg_name: str,
) -> Dict[str, Any]:
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        payload_path = td_path / payload_filename
        payload_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        cmd = [
            os.environ.get("PYTHON", "python3"),
            "-m",
            "oled_agent.cli",
            *cli_base_args,
            payload_arg_name,
            str(payload_path),
        ]
        env = dict(os.environ)
        env["PYTHONPATH"] = str(REPO_ROOT / "src")
        cp = subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True, check=False)
        stdout_text = str(cp.stdout or "").strip()
        parsed: Any = None
        if stdout_text:
            try:
                parsed = json.loads(stdout_text)
            except json.JSONDecodeError:
                parsed = None
        if cp.returncode != 0:
            return {
                "status": "fail",
                "returncode": cp.returncode,
                "command": cmd,
                "stdout": cp.stdout,
                "stderr": cp.stderr,
                "result": parsed,
            }
        return {
            "status": "pass",
            "returncode": cp.returncode,
            "command": cmd,
            "result": parsed if parsed is not None else {"raw_stdout": cp.stdout},
            "stderr": cp.stderr,
        }


def _run_cli_command(*, cli_args: List[str], ok_returncodes: Optional[List[int]] = None) -> Dict[str, Any]:
    cmd = [
        os.environ.get("PYTHON", "python3"),
        "-m",
        "oled_agent.cli",
        *cli_args,
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    cp = subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True, check=False)
    raw = str(cp.stdout or "").strip()
    parsed: Any = None
    if raw:
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = None
    allowed = set(ok_returncodes or [0])
    return {
        "status": "pass" if cp.returncode in allowed else "fail",
        "returncode": cp.returncode,
        "command": cmd,
        "result": parsed if parsed is not None else {"raw_stdout": cp.stdout},
        "stderr": cp.stderr,
    }


def _run_agent_run_json(*, payload: Dict[str, Any], planner_provider: str, catalog_path: str) -> Dict[str, Any]:
    catalog = _resolve_catalog(catalog_path)
    return _run_cli_with_json_payload(
        cli_base_args=[
            "agent-run-json",
            "--workspace-root",
            str(REPO_ROOT),
            "--catalog",
            str(catalog),
            "--planner-provider",
            str(planner_provider or "rule_based_v1"),
        ],
        payload=payload,
        payload_filename="request.json",
        payload_arg_name="--request-json",
    )


def _run_agent_step_json(*, payload: Dict[str, Any], catalog_path: str) -> Dict[str, Any]:
    catalog = _resolve_catalog(catalog_path)
    return _run_cli_with_json_payload(
        cli_base_args=[
            "agent-run-step-json",
            "--workspace-root",
            str(REPO_ROOT),
            "--catalog",
            str(catalog),
        ],
        payload=payload,
        payload_filename="step_request.json",
        payload_arg_name="--step-request-json",
    )


def _run_agent_intake(*, task_id: str, request_text: str, web_topk: int) -> Dict[str, Any]:
    return _run_cli_command(
        cli_args=[
            "agent-intake",
            "--workspace-root",
            str(REPO_ROOT),
            "--task-id",
            task_id,
            "--request",
            request_text,
            "--web-topk",
            str(max(1, int(web_topk))),
        ],
        ok_returncodes=[0, 2],
    )


def _run_agent_approve(*, task_json_path: Path, planner_provider: str, catalog_path: str) -> Dict[str, Any]:
    catalog = _resolve_catalog(catalog_path)
    return _run_cli_command(
        cli_args=[
            "agent-approve",
            "--workspace-root",
            str(REPO_ROOT),
            "--task-json",
            str(task_json_path),
            "--planner-provider",
            str(planner_provider or "rule_based_v1"),
            "--catalog",
            str(catalog),
        ],
        ok_returncodes=[0, 2],
    )


def _run_agent_resume(*, task_id: str, planner_provider: str, catalog_path: str) -> Dict[str, Any]:
    catalog = _resolve_catalog(catalog_path)
    return _run_cli_command(
        cli_args=[
            "agent-resume",
            "--workspace-root",
            str(REPO_ROOT),
            "--task-id",
            task_id,
            "--planner-provider",
            str(planner_provider or "rule_based_v1"),
            "--catalog",
            str(catalog),
        ],
        ok_returncodes=[0],
    )


def _task_artifact_path(task_id: str, filename: str) -> Path:
    return (REPO_ROOT / "runs" / "agent" / task_id / filename).resolve()


def _load_json_if_exists(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _is_safe_task_id(task_id: str) -> bool:
    tid = str(task_id or "").strip()
    if not tid:
        return False
    if not TASK_ID_PATTERN.fullmatch(tid):
        return False
    if ".." in tid or "/" in tid or "\\" in tid:
        return False
    return True


@app.get("/")
def index() -> str:
    return render_template_string(HTML)


@app.get("/api/health")
def api_health():
    return jsonify({"status": "pass", "repo_root": str(REPO_ROOT)})


@app.post("/api/run")
def api_run():
    body = request.get_json(silent=True) or {}
    payload_text = str(body.get("payload_text") or "")
    planner = str(body.get("planner_provider") or "rule_based_v1")
    catalog = str(body.get("catalog_path") or DEFAULT_CATALOG)
    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError as exc:
        return jsonify({"status": "fail", "error": f"invalid request json: {exc}"}), 400
    return jsonify(_run_agent_run_json(payload=payload, planner_provider=planner, catalog_path=catalog))


@app.post("/api/run-step")
def api_run_step():
    body = request.get_json(silent=True) or {}
    payload_text = str(body.get("payload_text") or "")
    catalog = str(body.get("catalog_path") or DEFAULT_CATALOG)
    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError as exc:
        return jsonify({"status": "fail", "error": f"invalid step request json: {exc}"}), 400
    if not isinstance(payload, dict):
        return jsonify({"status": "fail", "error": "step request must be JSON object"}), 400
    return jsonify(_run_agent_step_json(payload=payload, catalog_path=catalog))


@app.post("/api/intake")
def api_intake():
    body = request.get_json(silent=True) or {}
    task_id = str(body.get("task_id") or "").strip()
    request_text = str(body.get("request_text") or "").strip()
    web_topk = int(body.get("web_topk") or 5)
    if not task_id:
        return jsonify({"status": "fail", "error": "missing task_id"}), 400
    if not request_text:
        return jsonify({"status": "fail", "error": "missing request_text"}), 400
    return jsonify(_run_agent_intake(task_id=task_id, request_text=request_text, web_topk=web_topk))


@app.post("/api/approve")
def api_approve():
    body = request.get_json(silent=True) or {}
    task_json_path = str(body.get("task_json_path") or "").strip()
    planner = str(body.get("planner_provider") or "rule_based_v1")
    catalog = str(body.get("catalog_path") or DEFAULT_CATALOG)
    if not task_json_path:
        return jsonify({"status": "fail", "error": "missing task_json_path"}), 400
    task_path = Path(task_json_path)
    if not task_path.is_absolute():
        task_path = (REPO_ROOT / task_path).resolve()
    else:
        task_path = task_path.resolve()
    return jsonify(_run_agent_approve(task_json_path=task_path, planner_provider=planner, catalog_path=catalog))


@app.post("/api/resume")
def api_resume():
    body = request.get_json(silent=True) or {}
    task_id = str(body.get("task_id") or "").strip()
    planner = str(body.get("planner_provider") or "rule_based_v1")
    catalog = str(body.get("catalog_path") or DEFAULT_CATALOG)
    if not task_id:
        return jsonify({"status": "fail", "error": "missing task_id"}), 400
    if not _is_safe_task_id(task_id):
        return jsonify({"status": "fail", "error": "invalid task_id"}), 400
    return jsonify(_run_agent_resume(task_id=task_id, planner_provider=planner, catalog_path=catalog))


@app.get("/api/task/<task_id>/summary")
def api_task_summary(task_id: str):
    tid = str(task_id or "").strip()
    if not tid:
        return jsonify({"status": "fail", "error": "missing task_id"}), 400
    if not _is_safe_task_id(tid):
        return jsonify({"status": "fail", "error": "invalid task_id"}), 400
    run_dir = (REPO_ROOT / "runs" / "agent" / tid).resolve()
    artifacts = {
        "plan_path": _task_artifact_path(tid, "plan.json"),
        "execution_path": _task_artifact_path(tid, "execution.json"),
        "tool_state_path": _task_artifact_path(tid, "tool_state.json"),
        "decision_summary_path": _task_artifact_path(tid, "decision_summary.json"),
        "task_state_path": _task_artifact_path(tid, "task_state.json"),
        "web_evidence_path": _task_artifact_path(tid, "artifacts/web_evidence.json"),
    }
    files = {k: {"path": str(v), "exists": v.exists()} for k, v in artifacts.items()}
    execution = _load_json_if_exists(artifacts["execution_path"])
    task_state = _load_json_if_exists(artifacts["task_state_path"])
    decision = _load_json_if_exists(artifacts["decision_summary_path"])
    web_evidence = _load_json_if_exists(artifacts["web_evidence_path"])
    return jsonify(
        {
            "status": "pass" if run_dir.exists() else "missing",
            "task_id": tid,
            "run_dir": str(run_dir),
            "run_dir_exists": run_dir.exists(),
            "artifacts": files,
            "execution_summary": {
                "record_count": len(execution.get("records", [])) if isinstance(execution, dict) else 0,
                "status": execution.get("status") if isinstance(execution, dict) else None,
            },
            "task_state": task_state if isinstance(task_state, dict) else {},
            "decision_summary": decision if isinstance(decision, dict) else {},
            "web_evidence_preview": (
                web_evidence.get("results", [])[:5]
                if isinstance(web_evidence, dict) and isinstance(web_evidence.get("results"), list)
                else []
            ),
        }
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8787, debug=False)

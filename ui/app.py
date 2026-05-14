from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from flask import Flask, jsonify, render_template_string, request
from oled_agent.agent.request_contract import validate_decision_summary_payload, validate_task_state_payload


app = Flask(__name__)
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = "scripts/adapters/real_adapters_catalog.json"
TASK_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
ARTIFACT_NAME_TO_FILE = {
    "plan": "plan.json",
    "execution": "execution.json",
    "tool_state": "tool_state.json",
    "decision_summary": "decision_summary.json",
    "task_state": "task_state.json",
    "web_evidence": "artifacts/web_evidence.json",
    "experiment_trace": "artifacts/experiment_trace.json",
}


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
        <button onclick=\"loadTasks()\">Refresh Tasks</button>
        <label>Recent tasks</label>
        <select id=\"inspect_task_picker\" onchange=\"pickInspectTask()\">
          <option value=\"\">(select)</option>
        </select>
        <button onclick=\"inspectTask()\">Load</button>
        <label>Artifact</label>
        <select id=\"inspect_artifact_name\">
          <option value=\"plan\">plan</option>
          <option value=\"execution\">execution</option>
          <option value=\"decision_summary\">decision_summary</option>
          <option value=\"task_state\">task_state</option>
          <option value=\"tool_state\">tool_state</option>
          <option value=\"web_evidence\">web_evidence</option>
          <option value=\"experiment_trace\">experiment_trace</option>
        </select>
        <button onclick=\"previewArtifact()\">Preview Artifact</button>
        <button onclick=\"showTimeline()\">Show Timeline</button>
        <button onclick=\"validateTask()\">Validate Task</button>
        <button onclick=\"compareTask()\">Compare Runs</button>
        <button onclick=\"diffArtifact()\">Diff Artifact</button>
        <label>Timeline tool filter (optional)</label>
        <input id=\"timeline_tool_filter\" value=\"\" placeholder=\"e.g. score_candidates\" />
        <label>Timeline status filter</label>
        <select id=\"timeline_status_filter\">
          <option value=\"all\">all</option>
          <option value=\"failed\">failed</option>
          <option value=\"success\">success</option>
        </select>
        <label>Timeline sort</label>
        <select id=\"timeline_sort\">
          <option value=\"original\">original</option>
          <option value=\"duration_desc\">duration_desc</option>
          <option value=\"duration_asc\">duration_asc</option>
          <option value=\"name_asc\">name_asc</option>
        </select>
        <label>Compare picker</label>
        <select id=\"compare_other_task_picker\" onchange=\"pickCompareTask()\">
          <option value=\"\">(select)</option>
        </select>
        <label>Diff artifact</label>
        <select id=\"compare_artifact_name\">
          <option value=\"decision_summary\">decision_summary</option>
          <option value=\"task_state\">task_state</option>
          <option value=\"plan\">plan</option>
          <option value=\"execution\">execution</option>
          <option value=\"tool_state\">tool_state</option>
          <option value=\"web_evidence\">web_evidence</option>
          <option value=\"experiment_trace\">experiment_trace</option>
        </select>
        <label>Compare with task id</label>
        <input id=\"compare_other_task_id\" value=\"ui_task_step_demo\" placeholder=\"task id to compare against\" />
      </div>

      <div class=\"card full\">
        <h2>Result</h2>
        <button onclick=\"listExperiments()\">List Experiments</button>
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

      async function previewArtifact() {
        const out = document.getElementById('out');
        out.textContent = 'loading artifact...';
        const taskId = document.getElementById('inspect_task_id').value;
        const artifact = document.getElementById('inspect_artifact_name').value;
        const resp = await fetch(`/api/task/${encodeURIComponent(taskId)}/artifact/${encodeURIComponent(artifact)}?max_chars=20000`);
        const data = await resp.json();
        out.textContent = JSON.stringify(data, null, 2);
      }

      async function showTimeline() {
        const out = document.getElementById('out');
        out.textContent = 'loading timeline...';
        const taskId = document.getElementById('inspect_task_id').value;
        const tool = (document.getElementById('timeline_tool_filter').value || '').trim();
        const statusFilter = document.getElementById('timeline_status_filter').value;
        const sort = document.getElementById('timeline_sort').value;
        const params = new URLSearchParams();
        if (tool) params.set('tool', tool);
        if (statusFilter) params.set('status_filter', statusFilter);
        if (sort) params.set('sort', sort);
        const suffix = params.toString() ? `?${params.toString()}` : '';
        const resp = await fetch(`/api/task/${encodeURIComponent(taskId)}/timeline${suffix}`);
        const data = await resp.json();
        out.textContent = JSON.stringify(data, null, 2);
      }

      async function validateTask() {
        const out = document.getElementById('out');
        out.textContent = 'validating...';
        const taskId = document.getElementById('inspect_task_id').value;
        const resp = await fetch(`/api/task/${encodeURIComponent(taskId)}/validate`);
        const data = await resp.json();
        out.textContent = JSON.stringify(data, null, 2);
      }

      async function compareTask() {
        const out = document.getElementById('out');
        out.textContent = 'comparing...';
        const taskId = document.getElementById('inspect_task_id').value;
        const other = (document.getElementById('compare_other_task_id').value || '').trim();
        const params = new URLSearchParams();
        if (other) params.set('other_task_id', other);
        const suffix = params.toString() ? `?${params.toString()}` : '';
        const resp = await fetch(`/api/task/${encodeURIComponent(taskId)}/compare${suffix}`);
        const data = await resp.json();
        out.textContent = JSON.stringify(data, null, 2);
      }

      function fillTaskPicker(selectId, tasks, keepValue) {
        const select = document.getElementById(selectId);
        while (select.options.length > 1) {
          select.remove(1);
        }
        for (const t of tasks) {
          const tid = (t && t.task_id) ? String(t.task_id) : '';
          if (!tid) continue;
          const status = (t && t.execution_status) ? String(t.execution_status) : '';
          const records = (t && typeof t.record_count === 'number') ? t.record_count : 0;
          const failed = (t && typeof t.failed_step_count === 'number') ? t.failed_step_count : 0;
          const label = `${tid} [${status || 'unknown'}] records=${records} failed=${failed}`;
          const opt = document.createElement('option');
          opt.value = tid;
          opt.text = label;
          select.appendChild(opt);
        }
        if (keepValue) {
          select.value = keepValue;
        }
      }

      function pickInspectTask() {
        const tid = (document.getElementById('inspect_task_picker').value || '').trim();
        if (!tid) return;
        document.getElementById('inspect_task_id').value = tid;
        document.getElementById('resume_task_id').value = tid;
      }

      function pickCompareTask() {
        const tid = (document.getElementById('compare_other_task_picker').value || '').trim();
        if (!tid) return;
        document.getElementById('compare_other_task_id').value = tid;
      }

      async function loadTasks() {
        const out = document.getElementById('out');
        out.textContent = 'loading tasks...';
        const inspectNow = (document.getElementById('inspect_task_id').value || '').trim();
        const compareNow = (document.getElementById('compare_other_task_id').value || '').trim();
        const resp = await fetch('/api/tasks?limit=80');
        const data = await resp.json();
        const tasks = Array.isArray(data.tasks) ? data.tasks : [];
        fillTaskPicker('inspect_task_picker', tasks, inspectNow);
        fillTaskPicker('compare_other_task_picker', tasks, compareNow);
        out.textContent = JSON.stringify(data, null, 2);
      }

      async function diffArtifact() {
        const out = document.getElementById('out');
        out.textContent = 'diffing artifact...';
        const taskId = document.getElementById('inspect_task_id').value;
        const other = (document.getElementById('compare_other_task_id').value || '').trim();
        const artifact = document.getElementById('compare_artifact_name').value;
        const params = new URLSearchParams();
        if (other) params.set('other_task_id', other);
        if (artifact) params.set('artifact', artifact);
        const suffix = params.toString() ? `?${params.toString()}` : '';
        const resp = await fetch(`/api/task/${encodeURIComponent(taskId)}/artifact-diff${suffix}`);
        const data = await resp.json();
        out.textContent = JSON.stringify(data, null, 2);
      }

      async function listExperiments() {
        const out = document.getElementById('out');
        out.textContent = 'loading experiments...';
        const resp = await fetch('/api/experiments?limit=120');
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


def _task_artifact_paths(task_id: str) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for name, rel in ARTIFACT_NAME_TO_FILE.items():
        out[name] = _task_artifact_path(task_id, rel)
    return out


def _load_json_if_exists(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _task_updated_epoch_ms(run_dir: Path) -> int:
    latest = run_dir.stat().st_mtime if run_dir.exists() else 0.0
    try:
        for p in run_dir.rglob("*"):
            try:
                mt = p.stat().st_mtime
            except Exception:
                continue
            if mt > latest:
                latest = mt
    except Exception:
        pass
    return int(latest * 1000)


def _task_list_item(task_id: str, run_dir: Path) -> Dict[str, Any]:
    execution = _load_json_if_exists(run_dir / "execution.json")
    task_state = _load_json_if_exists(run_dir / "task_state.json")
    records = execution.get("records", []) if isinstance(execution, dict) and isinstance(execution.get("records"), list) else []
    failed_n = 0
    for rec in records:
        if isinstance(rec, dict) and str(rec.get("status") or "") != "success":
            failed_n += 1
    updated_ms = _task_updated_epoch_ms(run_dir)
    updated_at = datetime.fromtimestamp(updated_ms / 1000.0).isoformat(timespec="seconds")
    return {
        "task_id": task_id,
        "run_dir": str(run_dir),
        "updated_epoch_ms": updated_ms,
        "updated_at": updated_at,
        "execution_status": str(execution.get("status") or "") if isinstance(execution, dict) else "",
        "record_count": len(records),
        "failed_step_count": failed_n,
        "task_state_status": str(task_state.get("status") or "") if isinstance(task_state, dict) else "",
    }


def _experiment_row_from_trace(trace: Dict[str, Any], trace_path: Path) -> Dict[str, Any]:
    model_choice = trace.get("model_choice") if isinstance(trace.get("model_choice"), dict) else {}
    execution_summary = trace.get("execution_summary") if isinstance(trace.get("execution_summary"), dict) else {}
    source_artifacts = trace.get("source_artifacts") if isinstance(trace.get("source_artifacts"), dict) else {}
    candidate = source_artifacts.get("candidate_csv") if isinstance(source_artifacts.get("candidate_csv"), dict) else {}
    scored = source_artifacts.get("scored_csv") if isinstance(source_artifacts.get("scored_csv"), dict) else {}
    return {
        "task_id": str(trace.get("task_id") or ""),
        "run_label": str(trace.get("run_label") or ""),
        "generated_at": str(trace.get("generated_at") or ""),
        "execution_mode": str(trace.get("execution_mode") or ""),
        "status": str(execution_summary.get("status") or ""),
        "record_count": int(execution_summary.get("record_count") or 0),
        "failed_count": int(execution_summary.get("failed_count") or 0),
        "adapters": execution_summary.get("adapters", []) if isinstance(execution_summary.get("adapters"), list) else [],
        "predictor_id": str(model_choice.get("predictor_id") or ""),
        "generator_id": str(model_choice.get("generator_id") or ""),
        "candidate_csv_exists": bool(candidate.get("exists")),
        "scored_csv_exists": bool(scored.get("exists")),
        "trace_path": str(trace_path),
    }


def _safe_filter_token(value: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9._-]{1,128}", value))


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _preview_payload(payload: Any, *, artifact_name: str) -> Any:
    if artifact_name == "execution" and isinstance(payload, dict):
        records = payload.get("records")
        if isinstance(records, list):
            return {
                "task_id": payload.get("task_id"),
                "status": payload.get("status"),
                "started_at": payload.get("started_at"),
                "ended_at": payload.get("ended_at"),
                "record_count": len(records),
                "records_head": records[:8],
            }
    if artifact_name == "web_evidence" and isinstance(payload, dict):
        results = payload.get("results")
        if isinstance(results, list):
            lite = dict(payload)
            lite["results"] = results[:8]
            lite["result_count"] = len(results)
            return lite
    if artifact_name == "experiment_trace" and isinstance(payload, dict):
        return {
            "schema_version": payload.get("schema_version", ""),
            "task_id": payload.get("task_id", ""),
            "run_label": payload.get("run_label", ""),
            "execution_mode": payload.get("execution_mode", ""),
            "model_choice": payload.get("model_choice", {}),
            "execution_summary": payload.get("execution_summary", {}),
            "source_artifacts": payload.get("source_artifacts", {}),
        }
    return payload


def _artifact_preview(*, artifact_name: str, path: Path, max_chars: int) -> Dict[str, Any]:
    if not path.exists():
        return {
            "status": "missing",
            "artifact": artifact_name,
            "path": str(path),
            "exists": False,
        }
    text = path.read_text(encoding="utf-8", errors="replace")
    truncated = len(text) > max_chars
    text_preview = text if not truncated else text[:max_chars]
    payload = None
    parse_error = ""
    try:
        payload = json.loads(text)
    except Exception as exc:
        parse_error = f"{type(exc).__name__}: {exc}"
    return {
        "status": "pass",
        "artifact": artifact_name,
        "path": str(path),
        "exists": True,
        "size_bytes": path.stat().st_size,
        "truncated": truncated,
        "text_preview": text_preview,
        "json_preview": _preview_payload(payload, artifact_name=artifact_name) if payload is not None else None,
        "json_parse_error": parse_error,
    }


def _parse_iso_datetime(value: Any) -> Optional[datetime]:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        return datetime.fromisoformat(text)
    except Exception:
        return None


def _duration_ms(started_at: Any, ended_at: Any) -> Optional[int]:
    started = _parse_iso_datetime(started_at)
    ended = _parse_iso_datetime(ended_at)
    if started is None or ended is None:
        return None
    return int((ended - started).total_seconds() * 1000)


def _timeline_result_summary(result: Any) -> Dict[str, Any]:
    if not isinstance(result, dict):
        return {}
    out: Dict[str, Any] = {}
    for key in ["status", "adapter", "count", "rows", "output_csv", "final_output", "report_path"]:
        if key in result:
            out[key] = result.get(key)
    if "models" in result and isinstance(result.get("models"), list):
        out["models_count"] = len(result.get("models", []))
    if "results" in result and isinstance(result.get("results"), list):
        out["results_count"] = len(result.get("results", []))
    if "topn" in result:
        out["topn"] = result.get("topn")
    return out


def _filter_timeline_events(*, events: List[Dict[str, Any]], tool_filter: str, status_filter: str) -> List[Dict[str, Any]]:
    out = list(events)
    tf = str(tool_filter or "").strip().lower()
    sf = str(status_filter or "all").strip().lower()
    if tf:
        out = [e for e in out if tf in str(e.get("name") or "").lower()]
    if sf == "failed":
        out = [e for e in out if bool(e.get("is_failed"))]
    elif sf == "success":
        out = [e for e in out if not bool(e.get("is_failed"))]
    return out


def _sort_timeline_events(*, events: List[Dict[str, Any]], sort_key: str) -> List[Dict[str, Any]]:
    key = str(sort_key or "original").strip().lower()
    out = list(events)
    if key == "duration_desc":
        out.sort(key=lambda e: int(e.get("duration_ms") or -1), reverse=True)
        return out
    if key == "duration_asc":
        out.sort(key=lambda e: int(e.get("duration_ms") or 10**15))
        return out
    if key == "name_asc":
        out.sort(key=lambda e: str(e.get("name") or ""))
        return out
    return out


def _timeline_line(event: Dict[str, Any]) -> str:
    idx = int(event.get("index") or 0)
    name = str(event.get("name") or "")
    status = str(event.get("status") or "")
    dur = event.get("duration_ms")
    dur_text = f"{dur}ms" if isinstance(dur, int) and dur >= 0 else "n/a"
    adapter = str(event.get("adapter") or "")
    marker = "[FAIL]" if bool(event.get("is_failed")) else "[PASS]"
    if adapter:
        return f"{idx:02d} {marker} {name} status={status} duration={dur_text} adapter={adapter}"
    return f"{idx:02d} {marker} {name} status={status} duration={dur_text}"


def _task_compare_summary(task_id: str) -> Dict[str, Any]:
    run_dir = (REPO_ROOT / "runs" / "agent" / task_id).resolve()
    by_name = _task_artifact_paths(task_id)
    artifact_exists = {name: path.exists() for name, path in by_name.items()}
    artifact_missing = [name for name, ok in artifact_exists.items() if not ok]

    execution = _load_json_if_exists(by_name["execution"])
    records = execution.get("records", []) if isinstance(execution, dict) and isinstance(execution.get("records"), list) else []
    execution_status = str(execution.get("status") or "") if isinstance(execution, dict) else ""
    total_duration_ms = _duration_ms(execution.get("started_at"), execution.get("ended_at")) if isinstance(execution, dict) else None

    failed_steps: List[str] = []
    adapters: set[str] = set()
    for rec in records:
        if not isinstance(rec, dict):
            continue
        if str(rec.get("status") or "") != "success":
            failed_steps.append(str(rec.get("name") or ""))
        result = rec.get("result")
        if isinstance(result, dict):
            adapter = str(result.get("adapter") or "").strip()
            if adapter:
                adapters.add(adapter)

    web_evidence = _load_json_if_exists(by_name["web_evidence"])
    web_results = web_evidence.get("results", []) if isinstance(web_evidence, dict) and isinstance(web_evidence.get("results"), list) else []

    return {
        "task_id": task_id,
        "run_dir": str(run_dir),
        "run_dir_exists": run_dir.exists(),
        "artifacts_exists": artifact_exists,
        "artifacts_missing": artifact_missing,
        "execution_status": execution_status,
        "record_count": len(records),
        "failed_step_count": len(failed_steps),
        "failed_steps": failed_steps,
        "adapters": sorted(adapters),
        "total_duration_ms": total_duration_ms,
        "web_evidence_count": len(web_results),
    }


def _task_compare_diff(primary: Dict[str, Any], other: Dict[str, Any]) -> Dict[str, Any]:
    primary_adapters = set(primary.get("adapters", [])) if isinstance(primary.get("adapters"), list) else set()
    other_adapters = set(other.get("adapters", [])) if isinstance(other.get("adapters"), list) else set()
    primary_failed = set(primary.get("failed_steps", [])) if isinstance(primary.get("failed_steps"), list) else set()
    other_failed = set(other.get("failed_steps", [])) if isinstance(other.get("failed_steps"), list) else set()

    primary_rc = int(primary.get("record_count") or 0)
    other_rc = int(other.get("record_count") or 0)
    primary_fail = int(primary.get("failed_step_count") or 0)
    other_fail = int(other.get("failed_step_count") or 0)
    primary_web = int(primary.get("web_evidence_count") or 0)
    other_web = int(other.get("web_evidence_count") or 0)
    primary_dur = primary.get("total_duration_ms")
    other_dur = other.get("total_duration_ms")

    duration_delta: Optional[int] = None
    if isinstance(primary_dur, int) and isinstance(other_dur, int):
        duration_delta = primary_dur - other_dur

    return {
        "execution_status_changed": str(primary.get("execution_status") or "") != str(other.get("execution_status") or ""),
        "record_count_delta": primary_rc - other_rc,
        "failed_step_count_delta": primary_fail - other_fail,
        "web_evidence_count_delta": primary_web - other_web,
        "total_duration_ms_delta": duration_delta,
        "adapters_only_in_primary": sorted(primary_adapters - other_adapters),
        "adapters_only_in_other": sorted(other_adapters - primary_adapters),
        "failed_steps_only_in_primary": sorted(primary_failed - other_failed),
        "failed_steps_only_in_other": sorted(other_failed - primary_failed),
    }


def _task_compare_lines(primary: Dict[str, Any], other: Dict[str, Any], diff: Dict[str, Any]) -> List[str]:
    p_tid = str(primary.get("task_id") or "")
    o_tid = str(other.get("task_id") or "")
    out = [
        f"record_count {p_tid}={int(primary.get('record_count') or 0)} vs {o_tid}={int(other.get('record_count') or 0)} delta={int(diff.get('record_count_delta') or 0)}",
        f"failed_steps {p_tid}={int(primary.get('failed_step_count') or 0)} vs {o_tid}={int(other.get('failed_step_count') or 0)} delta={int(diff.get('failed_step_count_delta') or 0)}",
        f"web_evidence {p_tid}={int(primary.get('web_evidence_count') or 0)} vs {o_tid}={int(other.get('web_evidence_count') or 0)} delta={int(diff.get('web_evidence_count_delta') or 0)}",
    ]
    if isinstance(diff.get("total_duration_ms_delta"), int):
        out.append(f"duration_ms delta={int(diff.get('total_duration_ms_delta') or 0)}")
    if bool(diff.get("execution_status_changed")):
        out.append(
            f"execution_status changed: {p_tid}={str(primary.get('execution_status') or '')} vs {o_tid}={str(other.get('execution_status') or '')}"
        )
    if isinstance(diff.get("adapters_only_in_primary"), list) and diff.get("adapters_only_in_primary"):
        out.append(f"adapters only in {p_tid}: {', '.join(diff.get('adapters_only_in_primary', []))}")
    if isinstance(diff.get("adapters_only_in_other"), list) and diff.get("adapters_only_in_other"):
        out.append(f"adapters only in {o_tid}: {', '.join(diff.get('adapters_only_in_other', []))}")
    return out


def _normalize_diff_leaf(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, str) and len(value) > 240:
            return value[:240] + "...(truncated)"
        return value
    if isinstance(value, dict):
        return f"<dict:{len(value)}>"
    if isinstance(value, list):
        return f"<list:{len(value)}>"
    return str(value)


def _flatten_json_paths(
    payload: Any,
    *,
    out: Dict[str, Any],
    prefix: str = "",
    depth: int = 0,
    max_depth: int = 4,
    max_items: int = 60,
    max_nodes: int = 1800,
) -> None:
    if len(out) >= max_nodes:
        return
    if depth >= max_depth:
        key = prefix or "$"
        out[key] = "<max_depth>"
        return
    if isinstance(payload, dict):
        if not payload:
            out[prefix or "$"] = "<empty_dict>"
            return
        keys = sorted(payload.keys(), key=lambda x: str(x))
        for idx, key in enumerate(keys):
            if idx >= max_items:
                out[(prefix + "." if prefix else "") + "__truncated_keys__"] = len(keys) - max_items
                return
            k = str(key)
            next_prefix = f"{prefix}.{k}" if prefix else k
            _flatten_json_paths(
                payload.get(key),
                out=out,
                prefix=next_prefix,
                depth=depth + 1,
                max_depth=max_depth,
                max_items=max_items,
                max_nodes=max_nodes,
            )
            if len(out) >= max_nodes:
                return
        return
    if isinstance(payload, list):
        if not payload:
            out[prefix or "$"] = "<empty_list>"
            return
        limit = min(len(payload), max_items)
        for idx in range(limit):
            next_prefix = f"{prefix}[{idx}]" if prefix else f"[{idx}]"
            _flatten_json_paths(
                payload[idx],
                out=out,
                prefix=next_prefix,
                depth=depth + 1,
                max_depth=max_depth,
                max_items=max_items,
                max_nodes=max_nodes,
            )
            if len(out) >= max_nodes:
                return
        if len(payload) > limit:
            out[(prefix or "$") + ".__truncated_items__"] = len(payload) - limit
        return
    out[prefix or "$"] = _normalize_diff_leaf(payload)


def _artifact_diff_payload(primary_payload: Any, other_payload: Any) -> Dict[str, Any]:
    primary_flat: Dict[str, Any] = {}
    other_flat: Dict[str, Any] = {}
    _flatten_json_paths(primary_payload, out=primary_flat)
    _flatten_json_paths(other_payload, out=other_flat)

    primary_keys = set(primary_flat.keys())
    other_keys = set(other_flat.keys())
    only_primary = sorted(primary_keys - other_keys)
    only_other = sorted(other_keys - primary_keys)
    common = sorted(primary_keys & other_keys)
    changed: List[Dict[str, Any]] = []
    for key in common:
        if primary_flat.get(key) != other_flat.get(key):
            changed.append({"path": key, "primary": primary_flat.get(key), "other": other_flat.get(key)})

    return {
        "only_in_primary_count": len(only_primary),
        "only_in_other_count": len(only_other),
        "changed_count": len(changed),
        "only_in_primary": only_primary[:200],
        "only_in_other": only_other[:200],
        "changed": changed[:300],
        "primary_paths_total": len(primary_flat),
        "other_paths_total": len(other_flat),
    }


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


@app.get("/api/tasks")
def api_tasks():
    limit = _as_int(request.args.get("limit"), 50)
    limit = max(1, min(limit, 200))
    prefix = str(request.args.get("prefix") or "").strip()
    if prefix and not re.fullmatch(r"[A-Za-z0-9._-]{1,128}", prefix):
        return jsonify({"status": "fail", "error": "invalid prefix"}), 400
    runs_root = (REPO_ROOT / "runs" / "agent").resolve()
    if not runs_root.exists():
        return jsonify({"status": "pass", "tasks": [], "count": 0, "runs_root": str(runs_root)})

    items: List[Dict[str, Any]] = []
    for child in runs_root.iterdir():
        if not child.is_dir():
            continue
        tid = str(child.name or "").strip()
        if not _is_safe_task_id(tid):
            continue
        if prefix and not tid.startswith(prefix):
            continue
        items.append(_task_list_item(tid, child))
    items.sort(key=lambda it: int(it.get("updated_epoch_ms") or 0), reverse=True)
    limited = items[:limit]
    return jsonify(
        {
            "status": "pass",
            "runs_root": str(runs_root),
            "count": len(limited),
            "count_before_limit": len(items),
            "limit": limit,
            "prefix": prefix,
            "tasks": limited,
        }
    )


@app.get("/api/experiments")
def api_experiments():
    limit = _as_int(request.args.get("limit"), 80)
    limit = max(1, min(limit, 500))
    prefix = str(request.args.get("prefix") or "").strip()
    predictor_id = str(request.args.get("predictor_id") or "").strip()
    generator_id = str(request.args.get("generator_id") or "").strip()
    status = str(request.args.get("status") or "").strip()
    execution_mode = str(request.args.get("execution_mode") or "").strip()
    for token in (prefix, predictor_id, generator_id):
        if token and not _safe_filter_token(token):
            return jsonify({"status": "fail", "error": "invalid filter token"}), 400
    if status and status not in {"success", "failed"}:
        return jsonify({"status": "fail", "error": "invalid status"}), 400
    if execution_mode and execution_mode not in {"full_pipeline", "single_step"}:
        return jsonify({"status": "fail", "error": "invalid execution_mode"}), 400

    runs_root = (REPO_ROOT / "runs" / "agent").resolve()
    if not runs_root.exists():
        return jsonify({"status": "pass", "experiments": [], "count": 0, "runs_root": str(runs_root)})

    rows: List[Dict[str, Any]] = []
    for child in runs_root.iterdir():
        if not child.is_dir():
            continue
        task_id = str(child.name or "").strip()
        if not _is_safe_task_id(task_id):
            continue
        if prefix and not task_id.startswith(prefix):
            continue
        trace_path = child / "artifacts" / "experiment_trace.json"
        if not trace_path.exists():
            continue
        trace = _load_json_if_exists(trace_path)
        if not isinstance(trace, dict):
            continue
        row = _experiment_row_from_trace(trace, trace_path)
        if predictor_id and row.get("predictor_id") != predictor_id:
            continue
        if generator_id and row.get("generator_id") != generator_id:
            continue
        if status and row.get("status") != status:
            continue
        if execution_mode and row.get("execution_mode") != execution_mode:
            continue
        rows.append(row)
    rows.sort(key=lambda item: str(item.get("generated_at") or ""), reverse=True)
    limited = rows[:limit]
    return jsonify(
        {
            "status": "pass",
            "runs_root": str(runs_root),
            "count": len(limited),
            "count_before_limit": len(rows),
            "limit": limit,
            "filters": {
                "prefix": prefix,
                "predictor_id": predictor_id,
                "generator_id": generator_id,
                "status": status,
                "execution_mode": execution_mode,
            },
            "experiments": limited,
        }
    )


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
    by_name = _task_artifact_paths(tid)
    artifacts = {
        "plan_path": by_name["plan"],
        "execution_path": by_name["execution"],
        "tool_state_path": by_name["tool_state"],
        "decision_summary_path": by_name["decision_summary"],
        "task_state_path": by_name["task_state"],
        "web_evidence_path": by_name["web_evidence"],
        "experiment_trace_path": by_name["experiment_trace"],
    }
    files = {k: {"path": str(v), "exists": v.exists()} for k, v in artifacts.items()}
    execution = _load_json_if_exists(artifacts["execution_path"])
    task_state = _load_json_if_exists(artifacts["task_state_path"])
    decision = _load_json_if_exists(artifacts["decision_summary_path"])
    web_evidence = _load_json_if_exists(artifacts["web_evidence_path"])
    experiment_trace = _load_json_if_exists(artifacts["experiment_trace_path"])
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
            "experiment_trace_preview": (
                _preview_payload(experiment_trace, artifact_name="experiment_trace")
                if isinstance(experiment_trace, dict)
                else {}
            ),
        }
    )


@app.get("/api/task/<task_id>/artifact/<artifact_name>")
def api_task_artifact(task_id: str, artifact_name: str):
    tid = str(task_id or "").strip()
    name = str(artifact_name or "").strip()
    if not tid:
        return jsonify({"status": "fail", "error": "missing task_id"}), 400
    if not _is_safe_task_id(tid):
        return jsonify({"status": "fail", "error": "invalid task_id"}), 400
    if name not in ARTIFACT_NAME_TO_FILE:
        return jsonify({"status": "fail", "error": "invalid artifact_name"}), 400
    max_chars = max(2000, min(_as_int(request.args.get("max_chars"), 12000), 200000))
    paths = _task_artifact_paths(tid)
    payload = _artifact_preview(artifact_name=name, path=paths[name], max_chars=max_chars)
    return jsonify(payload)


@app.get("/api/task/<task_id>/timeline")
def api_task_timeline(task_id: str):
    tid = str(task_id or "").strip()
    if not tid:
        return jsonify({"status": "fail", "error": "missing task_id"}), 400
    if not _is_safe_task_id(tid):
        return jsonify({"status": "fail", "error": "invalid task_id"}), 400
    tool_filter = str(request.args.get("tool") or "").strip()
    status_filter = str(request.args.get("status_filter") or "all").strip().lower()
    sort = str(request.args.get("sort") or "original").strip().lower()
    if status_filter not in {"all", "failed", "success"}:
        return jsonify({"status": "fail", "error": "invalid status_filter"}), 400
    if sort not in {"original", "duration_desc", "duration_asc", "name_asc"}:
        return jsonify({"status": "fail", "error": "invalid sort"}), 400
    run_dir = (REPO_ROOT / "runs" / "agent" / tid).resolve()
    if not run_dir.exists():
        return jsonify({"status": "missing", "task_id": tid, "error": "run_dir_missing"}), 200
    execution_path = _task_artifact_path(tid, "execution.json")
    if not execution_path.exists():
        return jsonify({"status": "fail", "task_id": tid, "error": "missing_execution_json"}), 200
    try:
        execution = json.loads(execution_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return jsonify({"status": "fail", "task_id": tid, "error": f"invalid_execution_json: {type(exc).__name__}: {exc}"}), 200

    records = execution.get("records", []) if isinstance(execution, dict) else []
    events: List[Dict[str, Any]] = []
    for idx, rec in enumerate(records, start=1):
        if not isinstance(rec, dict):
            continue
        result = rec.get("result")
        event: Dict[str, Any] = {
            "index": idx,
            "name": str(rec.get("name") or ""),
            "status": str(rec.get("status") or ""),
            "started_at": rec.get("started_at"),
            "ended_at": rec.get("ended_at"),
            "duration_ms": _duration_ms(rec.get("started_at"), rec.get("ended_at")),
            "error": str(rec.get("error") or ""),
            "result_summary": _timeline_result_summary(result),
            "is_failed": str(rec.get("status") or "") != "success",
        }
        if isinstance(result, dict) and result.get("adapter"):
            event["adapter"] = result.get("adapter")
        event["highlight"] = "fail" if bool(event.get("is_failed")) else "normal"
        events.append(event)

    filtered = _filter_timeline_events(events=events, tool_filter=tool_filter, status_filter=status_filter)
    sorted_events = _sort_timeline_events(events=filtered, sort_key=sort)
    timeline_lines = [_timeline_line(e) for e in sorted_events]

    total_ms = _duration_ms(execution.get("started_at"), execution.get("ended_at")) if isinstance(execution, dict) else None
    success_n = sum(1 for e in sorted_events if not bool(e.get("is_failed")))
    fail_n = sum(1 for e in sorted_events if bool(e.get("is_failed")))
    return jsonify(
        {
            "status": "pass",
            "task_id": tid,
            "run_dir": str(run_dir),
            "execution_status": execution.get("status") if isinstance(execution, dict) else "",
            "started_at": execution.get("started_at") if isinstance(execution, dict) else "",
            "ended_at": execution.get("ended_at") if isinstance(execution, dict) else "",
            "total_duration_ms": total_ms,
            "summary": {
                "total_steps_before_filter": len(events),
                "total_steps": len(sorted_events),
                "success_steps": success_n,
                "failed_steps": fail_n,
                "tool_filter": tool_filter,
                "status_filter": status_filter,
                "sort": sort,
            },
            "events": sorted_events,
            "timeline_lines": timeline_lines,
        }
    )


@app.get("/api/task/<task_id>/compare")
def api_task_compare(task_id: str):
    tid = str(task_id or "").strip()
    if not tid:
        return jsonify({"status": "fail", "error": "missing task_id"}), 400
    if not _is_safe_task_id(tid):
        return jsonify({"status": "fail", "error": "invalid task_id"}), 400
    other = str(request.args.get("other_task_id") or "").strip()
    if not other:
        return jsonify({"status": "fail", "error": "missing other_task_id"}), 400
    if not _is_safe_task_id(other):
        return jsonify({"status": "fail", "error": "invalid other_task_id"}), 400
    if other == tid:
        return jsonify({"status": "fail", "error": "other_task_id must differ from task_id"}), 400

    primary = _task_compare_summary(tid)
    other_summary = _task_compare_summary(other)
    diff = _task_compare_diff(primary, other_summary)
    warnings: List[str] = []
    if not bool(primary.get("run_dir_exists")):
        warnings.append("primary_run_dir_missing")
    if not bool(other_summary.get("run_dir_exists")):
        warnings.append("other_run_dir_missing")
    return jsonify(
        {
            "status": "pass" if not warnings else "partial",
            "task_id": tid,
            "other_task_id": other,
            "warnings": warnings,
            "primary": primary,
            "other": other_summary,
            "diff": diff,
            "compare_lines": _task_compare_lines(primary, other_summary, diff),
        }
    )


@app.get("/api/task/<task_id>/artifact-diff")
def api_task_artifact_diff(task_id: str):
    tid = str(task_id or "").strip()
    if not tid:
        return jsonify({"status": "fail", "error": "missing task_id"}), 400
    if not _is_safe_task_id(tid):
        return jsonify({"status": "fail", "error": "invalid task_id"}), 400
    other = str(request.args.get("other_task_id") or "").strip()
    if not other:
        return jsonify({"status": "fail", "error": "missing other_task_id"}), 400
    if not _is_safe_task_id(other):
        return jsonify({"status": "fail", "error": "invalid other_task_id"}), 400
    if other == tid:
        return jsonify({"status": "fail", "error": "other_task_id must differ from task_id"}), 400
    artifact = str(request.args.get("artifact") or "decision_summary").strip()
    if artifact not in ARTIFACT_NAME_TO_FILE:
        return jsonify({"status": "fail", "error": "invalid artifact"}), 400

    primary_path = _task_artifact_paths(tid).get(artifact)
    other_path = _task_artifact_paths(other).get(artifact)
    if primary_path is None or other_path is None:
        return jsonify({"status": "fail", "error": "internal_artifact_resolution_error"}), 500
    if not primary_path.exists() or not other_path.exists():
        return jsonify(
            {
                "status": "missing",
                "task_id": tid,
                "other_task_id": other,
                "artifact": artifact,
                "primary_exists": primary_path.exists(),
                "other_exists": other_path.exists(),
                "primary_path": str(primary_path),
                "other_path": str(other_path),
            }
        ), 200

    try:
        primary_payload = json.loads(primary_path.read_text(encoding="utf-8"))
        other_payload = json.loads(other_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return jsonify({"status": "fail", "error": f"invalid_json: {type(exc).__name__}: {exc}"}), 200

    diff = _artifact_diff_payload(primary_payload, other_payload)
    return jsonify(
        {
            "status": "pass",
            "task_id": tid,
            "other_task_id": other,
            "artifact": artifact,
            "primary_path": str(primary_path),
            "other_path": str(other_path),
            "diff": diff,
        }
    )


@app.get("/api/task/<task_id>/validate")
def api_task_validate(task_id: str):
    tid = str(task_id or "").strip()
    if not tid:
        return jsonify({"status": "fail", "error": "missing task_id"}), 400
    if not _is_safe_task_id(tid):
        return jsonify({"status": "fail", "error": "invalid task_id"}), 400
    run_dir = (REPO_ROOT / "runs" / "agent" / tid).resolve()
    if not run_dir.exists():
        return jsonify({"status": "missing", "task_id": tid, "error": "run_dir_missing"}), 200

    checks: List[Dict[str, str]] = []
    loaded: Dict[str, Any] = {}
    required = ["plan", "execution", "tool_state", "decision_summary", "task_state"]
    by_name = _task_artifact_paths(tid)

    for name in required:
        path = by_name[name]
        if not path.exists():
            checks.append({"name": name, "status": "fail", "message": f"missing file: {path}"})
            continue
        try:
            loaded[name] = json.loads(path.read_text(encoding="utf-8"))
            checks.append({"name": name, "status": "pass", "message": "json parse ok"})
        except Exception as exc:
            checks.append({"name": name, "status": "fail", "message": f"json parse failed: {type(exc).__name__}: {exc}"})

    execution = loaded.get("execution")
    if isinstance(execution, dict) and isinstance(execution.get("records"), list) and len(execution.get("records", [])) > 0:
        checks.append({"name": "execution_records", "status": "pass", "message": "records list is non-empty"})
    else:
        checks.append({"name": "execution_records", "status": "fail", "message": "records list missing or empty"})

    decision = loaded.get("decision_summary")
    if isinstance(decision, dict):
        try:
            validate_decision_summary_payload(decision, REPO_ROOT)
            checks.append({"name": "decision_summary_schema", "status": "pass", "message": "schema valid"})
        except Exception as exc:
            checks.append({"name": "decision_summary_schema", "status": "fail", "message": str(exc)})

    task_state = loaded.get("task_state")
    if isinstance(task_state, dict):
        try:
            validate_task_state_payload(task_state, REPO_ROOT)
            checks.append({"name": "task_state_schema", "status": "pass", "message": "schema valid"})
        except Exception as exc:
            checks.append({"name": "task_state_schema", "status": "fail", "message": str(exc)})

    pass_n = sum(1 for c in checks if c.get("status") == "pass")
    fail_n = sum(1 for c in checks if c.get("status") == "fail")
    overall = "pass" if fail_n == 0 else "fail"
    return jsonify(
        {
            "status": overall,
            "task_id": tid,
            "run_dir": str(run_dir),
            "summary": {"pass": pass_n, "fail": fail_n},
            "checks": checks,
            "blocking_checks": [c.get("name") for c in checks if c.get("status") == "fail"],
        }
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8787, debug=False)

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List

from flask import Flask, jsonify, render_template_string, request


app = Flask(__name__)
REPO_ROOT = Path(__file__).resolve().parents[1]


HTML = """
<!doctype html>
<html>
  <head>
    <meta charset=\"utf-8\" />
    <title>Agent4Mat UI Prototype</title>
    <style>
      body { font-family: -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif; margin: 2rem; }
      .card { max-width: 840px; border: 1px solid #ddd; border-radius: 10px; padding: 1rem 1.2rem; }
      label { display: block; margin-top: 0.8rem; font-weight: 600; }
      input, textarea, select { width: 100%; margin-top: 0.3rem; padding: 0.5rem; }
      button { margin-top: 1rem; padding: 0.6rem 1rem; }
      pre { background: #f7f7f7; padding: 0.8rem; border-radius: 8px; overflow-x: auto; }
    </style>
  </head>
  <body>
    <div class=\"card\">
      <h2>Agent4Mat (Prototype)</h2>
      <p>Input request JSON and execute <code>agent-run-json</code>.</p>
      <label>Catalog path</label>
      <input id=\"catalog\" value=\"scripts/adapters/real_adapters_catalog.json\" />
      <label>Planner provider</label>
      <select id=\"planner\">
        <option value=\"rule_based_v1\">rule_based_v1</option>
        <option value=\"llm_v1\">llm_v1</option>
      </select>
      <label>Request JSON</label>
      <textarea id=\"payload\" rows=\"14\">{
  \"task_id\": \"ui_task_demo\",
  \"request_text\": \"设计470nm附近且高PLQY分子\",
  \"mode\": \"fast_screen\",
  \"targets\": [{\"property\": \"plqy\", \"objective\": \"maximize\", \"target_value\": 0.6}],
  \"budget\": {\"max_candidates\": 10}
}</textarea>
      <button onclick=\"runTask()\">Run</button>
      <h3>Result</h3>
      <pre id=\"out\">(waiting)</pre>
    </div>
    <script>
      async function runTask() {
        const out = document.getElementById('out');
        out.textContent = 'running...';
        const payloadText = document.getElementById('payload').value;
        const planner = document.getElementById('planner').value;
        const catalog = document.getElementById('catalog').value;
        const resp = await fetch('/api/run', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({payload_text: payloadText, planner_provider: planner, catalog_path: catalog})
        });
        const data = await resp.json();
        out.textContent = JSON.stringify(data, null, 2);
      }
    </script>
  </body>
</html>
"""


def _run_agent_run_json(*, payload: Dict[str, object], planner_provider: str, catalog_path: str) -> Dict[str, object]:
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        req = td_path / "request.json"
        req.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        cmd = [
            os.environ.get("PYTHON", "python3"),
            "-m",
            "oled_agent.cli",
            "agent-run-json",
            "--workspace-root",
            str(REPO_ROOT),
            "--catalog",
            str((REPO_ROOT / catalog_path).resolve() if not Path(catalog_path).is_absolute() else Path(catalog_path)),
            "--request-json",
            str(req),
            "--planner-provider",
            planner_provider,
        ]
        env = dict(os.environ)
        env["PYTHONPATH"] = str(REPO_ROOT / "src")
        cp = subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True, check=False)
        if cp.returncode != 0:
            return {
                "status": "fail",
                "returncode": cp.returncode,
                "stdout": cp.stdout,
                "stderr": cp.stderr,
            }
        try:
            payload_out = json.loads(cp.stdout)
        except json.JSONDecodeError:
            payload_out = {"raw_stdout": cp.stdout}
        return {
            "status": "pass",
            "result": payload_out,
        }


@app.get("/")
def index() -> str:
    return render_template_string(HTML)


@app.post("/api/run")
def api_run():
    body = request.get_json(silent=True) or {}
    payload_text = str(body.get("payload_text") or "")
    planner = str(body.get("planner_provider") or "rule_based_v1")
    catalog = str(body.get("catalog_path") or "configs/models/catalog.json")
    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError as exc:
        return jsonify({"status": "fail", "error": f"invalid request json: {exc}"}), 400
    return jsonify(_run_agent_run_json(payload=payload, planner_provider=planner, catalog_path=catalog))


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8787, debug=False)

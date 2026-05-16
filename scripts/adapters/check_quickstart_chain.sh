#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

TASK_ID="${1:-quickstart_chain_$(date +%Y%m%d_%H%M%S)}"
REQUEST_TEXT="${2:-设计470nm附近且高PLQY分子}"

RUN_DIR="runs/agent/${TASK_ID}"
REQUEST_JSON="${RUN_DIR}/request_quickstart.json"
RESULT_JSON="${RUN_DIR}/quickstart_result.json"

mkdir -p "${RUN_DIR}"

echo "[1/8] build request payload"
python3 - "${REQUEST_JSON}" "${TASK_ID}" "${REQUEST_TEXT}" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
task_id = sys.argv[2]
request_text = sys.argv[3]
payload = {
    "task_id": task_id,
    "request_text": request_text,
    "mode": "fast_screen",
    "targets": [{"property": "plqy", "objective": "maximize", "target_value": 0.6}],
    "budget": {"max_candidates": 5},
    "model_preferences": {
        "predictor_id": "pred_tpl_v1",
        "generator_id": "gen_tpl_v1",
    },
}
path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY

unset OLED_AGENT_TRAIN_CMD
unset OLED_AGENT_GENERATE_CMD
unset OLED_AGENT_SCORE_CMD

echo "[2/8] run agent-run-json with quickstart catalog"
PYTHONPATH=src python3 -m oled_agent.cli agent-run-json \
  --workspace-root . \
  --catalog scripts/adapters/quickstart_catalog.json \
  --request-json "${REQUEST_JSON}" | tee "${RESULT_JSON}"

DECISION_PATH="$(python3 - "${RESULT_JSON}" <<'PY'
import json
import pathlib
import sys

obj = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
print(obj["decision_summary_path"])
PY
)"

TASK_STATE_PATH="$(python3 - "${RESULT_JSON}" <<'PY'
import json
import pathlib
import sys

obj = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
print(obj["task_state_path"])
PY
)"

DATA_REPORT_PATH="$(python3 - "${RESULT_JSON}" <<'PY'
import json
import pathlib
import sys

obj = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
print(obj["logging_data_report_path"])
PY
)"

MODEL_REPORT_PATH="$(python3 - "${RESULT_JSON}" <<'PY'
import json
import pathlib
import sys

obj = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
print(obj["logging_model_report_path"])
PY
)"

FILTERING_REPORT_PATH="$(python3 - "${RESULT_JSON}" <<'PY'
import json
import pathlib
import sys

obj = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
print(obj["logging_filtering_report_path"])
PY
)"

EVALUATION_REPORT_PATH="$(python3 - "${RESULT_JSON}" <<'PY'
import json
import pathlib
import sys

obj = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
print(obj["logging_evaluation_report_path"])
PY
)"

echo "[3/8] validate structured artifacts schema"
python3 scripts/validate_run_artifacts.py \
  --workspace-root . \
  --decision-summary "${DECISION_PATH}" \
  --task-state "${TASK_STATE_PATH}" \
  --data-report "${DATA_REPORT_PATH}" \
  --model-report "${MODEL_REPORT_PATH}" \
  --filtering-report "${FILTERING_REPORT_PATH}" \
  --evaluation-report "${EVALUATION_REPORT_PATH}"

echo "[4/8] summarize adapters"
python3 - "${RESULT_JSON}" <<'PY'
import json
import pathlib
import sys

result = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
execution = json.loads(pathlib.Path(result["execution_path"]).read_text(encoding="utf-8"))
records = execution.get("records", [])
by_name = {r.get("name"): r for r in records if isinstance(r, dict)}
gen = by_name.get("generate_candidates", {}).get("result", {}).get("adapter", "")
score = by_name.get("score_candidates", {}).get("result", {}).get("adapter", "")

print("[PASS] quickstart chain completed")
print(f"[PASS] task_id={result.get('task_id')}")
print(f"[PASS] status={result.get('status')}")
print(f"[PASS] generate_adapter={gen}")
print(f"[PASS] score_adapter={score}")
print(f"[PASS] run_dir={pathlib.Path(result['execution_path']).parent}")
PY

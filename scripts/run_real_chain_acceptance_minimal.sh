#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_ROOT="${1:-.}"
TASK_ID="${2:-real_chain_minimal_$(date +%Y%m%d_%H%M%S)}"

cd "$WORKSPACE_ROOT"

REQ="runs/ci/request_real_chain_minimal_${TASK_ID}.json"
mkdir -p runs/ci

cat > "$REQ" <<'JSON'
{
  "task_id": "__TASK_ID__",
  "request_text": "设计470nm附近且高PLQY分子",
  "mode": "fast_screen",
  "targets": [
    {"property": "plqy", "objective": "maximize", "target_value": 0.6}
  ],
  "budget": {"max_candidates": 8},
  "model_preferences": {
    "predictor_id": "unimol_lambda_plqy_real_v1",
    "generator_id": "reinvent4_generator_real_v1"
  }
}
JSON

python3 - "$REQ" "$TASK_ID" <<'PY'
from pathlib import Path
import sys

request_path = Path(sys.argv[1])
task_id = sys.argv[2]
text = request_path.read_text(encoding="utf-8")
request_path.write_text(text.replace("__TASK_ID__", task_id), encoding="utf-8")
PY

# deterministic real-mode contract path using local stubs
export OLED_AGENT_REINVENT4_ADAPTER_MODE=real
export OLED_AGENT_REINVENT4_SOURCE_CSV="$(pwd)/configs/pipelines/demo_input.csv"
export OLED_AGENT_REINVENT4_PIPELINE_SCRIPT="$(pwd)/scripts/adapters/stub_reinvent4_pipeline.sh"
export OLED_AGENT_REINVENT4_RANKREADY_CSV="$(pwd)/runs/contract/reinvent4_real_stub_rankready_${TASK_ID}.csv"

export OLED_AGENT_UNIMOL_SCORE_MODE=real
export OLED_AGENT_UNIMOL_SCORE_SCRIPT="$(pwd)/scripts/adapters/stub_unimol_score.py"
export UNIMOL_REMOTE_HOST=stub_host
export UNIMOL_REMOTE_PY=stub_py
export UNIMOL_REMOTE_TMP_BASE=/tmp

PYTHONPATH=src python3 -m oled_agent.cli agent-run-json \
  --workspace-root . \
  --catalog scripts/adapters/real_adapters_catalog.json \
  --request-json "$REQ" > "runs/ci/agent_run_real_chain_${TASK_ID}.json"

python3 scripts/validate_run_artifacts.py \
  --workspace-root . \
  --result-json "runs/ci/agent_run_real_chain_${TASK_ID}.json"

python3 - "$TASK_ID" <<'PY'
import json
from pathlib import Path
import sys

task_id = sys.argv[1]
result_json = Path(f"runs/ci/agent_run_real_chain_{task_id}.json")
result = json.loads(result_json.read_text(encoding="utf-8"))
exec_path = Path(result["execution_path"])
execution = json.loads(exec_path.read_text(encoding="utf-8"))
records = {r.get("name"): r.get("result", {}) for r in execution.get("records", [])}
gen = records.get("generate_candidates", {})
score = records.get("score_candidates", {})
assert gen.get("adapter") == "reinvent4_generate_adapter_v1", gen
assert score.get("adapter") == "unimol_score_adapter_v1", score
assert "fallback_error" not in gen, gen
print(json.dumps({
    "status": "pass",
    "task_id": task_id,
    "generate_adapter": gen.get("adapter"),
    "score_adapter": score.get("adapter"),
    "execution_path": str(exec_path),
}, ensure_ascii=False))
PY

echo "[PASS] minimal real chain acceptance complete: $TASK_ID"

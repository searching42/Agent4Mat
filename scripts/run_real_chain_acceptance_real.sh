#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

TASK_ID="${1:-accept_real_chain_$(date +%Y%m%d_%H%M%S)}"
REQUEST_TEXT="${2:-设计470nm附近且高PLQY分子}"
CATALOG="${3:-scripts/adapters/real_adapters_catalog.json}"
DEBUG_JSON_OUT="${4:-runs/agent/${TASK_ID}/external_debug.json}"
RESULT_JSON="runs/agent/${TASK_ID}/acceptance_result.json"
REQUEST_JSON="runs/agent/${TASK_ID}/acceptance_request.json"

mkdir -p "runs/agent/${TASK_ID}"

die() {
  echo "[FAIL] $*" >&2
  exit 1
}

warn() {
  echo "[WARN] $*" >&2
}

info() {
  echo "[INFO] $*" >&2
}

require_env() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    die "missing required env: ${name}"
  fi
}

ensure_no_stub_value() {
  local name="$1"
  local val="${!name:-}"
  if [[ -z "${val}" ]]; then
    return 0
  fi
  if [[ "${val}" == *stub* ]]; then
    die "${name} points to stub-like value: ${val}"
  fi
}

ensure_no_stub_path() {
  local name="$1"
  local val="${!name:-}"
  if [[ -z "${val}" ]]; then
    return 0
  fi
  local base
  base="$(basename "${val}")"
  if [[ "${base}" == "stub_unimol_score.py" || "${base}" == "stub_reinvent4_pipeline.sh" ]]; then
    die "${name} points to forbidden stub script: ${val}"
  fi
}

require_file() {
  local path="$1"
  [[ -f "${path}" ]] || die "required file not found: ${path}"
}

cat > "${REQUEST_JSON}" <<JSON
{
  "task_id": "${TASK_ID}",
  "request_text": "${REQUEST_TEXT}",
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

info "[1/6] validate runtime env (real-mode required)"
export OLED_AGENT_USE_EXTERNAL_SCORER=1
export OLED_AGENT_UNIMOL_SCORE_MODE=real
export OLED_AGENT_REINVENT4_ADAPTER_MODE=real

require_env UNIMOL_REMOTE_HOST
require_env UNIMOL_REMOTE_PY
require_env UNIMOL_REMOTE_TMP_BASE
require_env OLED_AGENT_REINVENT4_PIPELINE_SCRIPT

ensure_no_stub_value UNIMOL_REMOTE_HOST
ensure_no_stub_path OLED_AGENT_UNIMOL_SCORE_SCRIPT
ensure_no_stub_path OLED_AGENT_REINVENT4_PIPELINE_SCRIPT

require_file "${CATALOG}"
require_file "${OLED_AGENT_REINVENT4_PIPELINE_SCRIPT}"
if [[ -n "${OLED_AGENT_UNIMOL_SCORE_SCRIPT:-}" ]]; then
  require_file "${OLED_AGENT_UNIMOL_SCORE_SCRIPT}"
fi
if [[ -n "${OLED_AGENT_REINVENT4_SOURCE_CSV:-}" ]]; then
  require_file "${OLED_AGENT_REINVENT4_SOURCE_CSV}"
fi

info "[2/6] doctor + external preflight"
PYTHONPATH=src python3 -m oled_agent.cli doctor --workspace-root .
PYTHONPATH=src python3 -m oled_agent.cli external-preflight --workspace-root .

info "[3/6] external connectivity debug (non-blocking evidence capture)"
if PYTHONPATH=src python3 -m oled_agent.cli external-connectivity-debug --workspace-root . --json-out "${DEBUG_JSON_OUT}"; then
  info "external-connectivity-debug passed"
else
  warn "external-connectivity-debug reported non-pass status; continue to run full acceptance"
fi

info "[4/6] run agent real-chain task"
PYTHONPATH=src python3 -m oled_agent.cli agent-run-json \
  --workspace-root . \
  --catalog "${CATALOG}" \
  --request-json "${REQUEST_JSON}" > "${RESULT_JSON}"

info "[5/6] validate structured artifacts"
python3 scripts/validate_run_artifacts.py \
  --workspace-root . \
  --result-json "${RESULT_JSON}"

info "[6/6] enforce non-fallback real adapters"
python3 - "${RESULT_JSON}" "${DEBUG_JSON_OUT}" <<'PY'
import json
import pathlib
import sys

result_path = pathlib.Path(sys.argv[1])
debug_path = pathlib.Path(sys.argv[2])
result = json.loads(result_path.read_text(encoding="utf-8"))
execution_path = pathlib.Path(result["execution_path"])
decision_path = pathlib.Path(result["decision_summary_path"])
execution = json.loads(execution_path.read_text(encoding="utf-8"))
decision = json.loads(decision_path.read_text(encoding="utf-8"))

records = {r.get("name"): r.get("result", {}) for r in execution.get("records", []) if isinstance(r, dict)}
gen = records.get("generate_candidates", {})
score = records.get("score_candidates", {})

gen_adapter = gen.get("adapter")
score_adapter = score.get("adapter")
if gen_adapter != "reinvent4_generate_adapter_v1":
    raise SystemExit(f"[FAIL] generate adapter mismatch: {gen_adapter}")
if score_adapter != "unimol_score_adapter_v1":
    raise SystemExit(f"[FAIL] score adapter mismatch: {score_adapter}")
if gen.get("fallback_error"):
    raise SystemExit(f"[FAIL] generate fallback detected: {gen.get('fallback_error')}")
if score.get("fallback_error"):
    raise SystemExit(f"[FAIL] score fallback detected: {score.get('fallback_error')}")

score_step = decision.get("score_step", {})
if bool(score_step.get("used_fallback")):
    raise SystemExit("[FAIL] decision_summary reports used_fallback=true")

print(json.dumps({
    "status": "pass",
    "task_id": result.get("task_id"),
    "result_json": str(result_path),
    "execution_path": str(execution_path),
    "decision_summary_path": str(decision_path),
    "generate_adapter": gen_adapter,
    "score_adapter": score_adapter,
    "external_debug_json": str(debug_path),
}, ensure_ascii=False))
PY

echo "[PASS] real chain acceptance succeeded: task_id=${TASK_ID}"

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

TASK_ID="${1:-accept_external_debug_$(date +%Y%m%d_%H%M%S)}"
REQUEST="${2:-设计470nm附近且高PLQY分子}"

RUN_DIR="runs/agent/${TASK_ID}"
DECISION_PATH="${RUN_DIR}/decision_summary.json"
DEBUG_JSON="${RUN_DIR}/external_debug.json"
mkdir -p "${RUN_DIR}"

export OLED_AGENT_USE_EXTERNAL_SCORER=1

echo "[1/4] external connectivity debug (json)"
if PYTHONPATH=src python3 -m oled_agent.cli external-connectivity-debug --workspace-root . --json-out "${DEBUG_JSON}"; then
  echo "[INFO] external-connectivity-debug passed"
else
  echo "[WARN] external-connectivity-debug reported non-pass status (continuing to capture acceptance evidence)"
fi

echo "[2/4] run agent end-to-end (external scorer expected)"
PYTHONPATH=src python3 -m oled_agent.cli agent-run --workspace-root . --task-id "${TASK_ID}" --request "${REQUEST}"

echo "[3/4] validate decision summary schema"
python3 scripts/validate_decision_summary.py "${DECISION_PATH}"

echo "[4/4] require external scorer adapter"
python3 - "${DECISION_PATH}" "${DEBUG_JSON}" <<'PY'
import json
import pathlib
import sys

decision_path = pathlib.Path(sys.argv[1])
debug_path = pathlib.Path(sys.argv[2])

obj = json.loads(decision_path.read_text(encoding="utf-8"))
score = obj.get("score_step", {})
adapter = score.get("adapter", "")
if adapter == "external_unimol_script":
    print(f"[PASS] acceptance succeeded: adapter={adapter}")
    print(f"[PASS] decision_summary={decision_path}")
    print(f"[PASS] external_debug={debug_path}")
    sys.exit(0)

print(f"[FAIL] expected external_unimol_script, got: {adapter}")
print(f"[FAIL] decision_summary={decision_path}")
print(f"[FAIL] external_debug={debug_path}")
print(
    "[FAIL] fallback="
    f"code={score.get('fallback_code')} "
    f"retryable={score.get('fallback_retryable')} "
    f"used_fallback={score.get('used_fallback')}"
)

if debug_path.exists():
    dbg = json.loads(debug_path.read_text(encoding="utf-8"))
    conn = dbg.get("connectivity", {})
    print(
        "[FAIL] connectivity="
        f"chain_ready={conn.get('chain_ready')} "
        f"runtime_source={conn.get('runtime_source')} "
        f"blocking_checks={conn.get('blocking_checks', [])} "
        f"failure_classes={conn.get('failure_classes', [])}"
    )

sys.exit(1)
PY

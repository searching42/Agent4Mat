#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

TASK_ID="${1:-accept_external_$(date +%Y%m%d_%H%M%S)}"
REQUEST="${2:-设计470nm附近且高PLQY分子}"

# This acceptance always targets external scoring.
export OLED_AGENT_USE_EXTERNAL_SCORER=1

echo "[1/3] external preflight"
PYTHONPATH=src python3 -m oled_agent.cli external-preflight --workspace-root .

echo "[2/3] run agent end-to-end (external scorer expected)"
PYTHONPATH=src python3 -m oled_agent.cli agent-run --workspace-root . --task-id "${TASK_ID}" --request "${REQUEST}"

DECISION_PATH="runs/agent/${TASK_ID}/decision_summary.json"
echo "[3/3] validate decision summary and require external scorer"
python3 scripts/validate_decision_summary.py "${DECISION_PATH}"

python3 - <<PY
import json, pathlib, sys
p = pathlib.Path("${DECISION_PATH}")
obj = json.loads(p.read_text(encoding="utf-8"))
adapter = obj.get("score_step", {}).get("adapter", "")
if adapter != "external_unimol_script":
    print(f"[FAIL] expected external_unimol_script, got: {adapter}")
    sys.exit(1)
print(f"[PASS] acceptance succeeded: adapter={adapter}, decision_summary={p}")
PY

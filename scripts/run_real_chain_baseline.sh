#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

BASE_TASK_ID="${1:-real_chain_baseline_$(date +%Y%m%d_%H%M%S)}"
REQUEST_TEXT="${2:-设计470nm附近且高PLQY分子}"
CATALOG="${3:-scripts/adapters/real_adapters_catalog.json}"
RUN_COUNT="${4:-3}"

info() {
  echo "[INFO] $*" >&2
}

die() {
  echo "[FAIL] $*" >&2
  exit 1
}

if ! [[ "${RUN_COUNT}" =~ ^[0-9]+$ ]]; then
  die "RUN_COUNT must be integer: ${RUN_COUNT}"
fi
if (( RUN_COUNT < 1 )); then
  die "RUN_COUNT must be >= 1: ${RUN_COUNT}"
fi

BASE_RUN_DIR="runs/agent/${BASE_TASK_ID}"
mkdir -p "${BASE_RUN_DIR}"

info "baseline start: base_task_id=${BASE_TASK_ID}, run_count=${RUN_COUNT}"
for i in $(seq 1 "${RUN_COUNT}"); do
  task_id="${BASE_TASK_ID}_r${i}"
  debug_json="runs/agent/${task_id}/external_debug.json"
  info "run ${i}/${RUN_COUNT}: task_id=${task_id}"
  ./scripts/run_real_chain_acceptance_real.sh "${task_id}" "${REQUEST_TEXT}" "${CATALOG}" "${debug_json}"
done

info "aggregate baseline evidence"
python3 - "${BASE_TASK_ID}" "${RUN_COUNT}" <<'PY'
import json
import pathlib
import subprocess
import sys
from datetime import datetime, timezone

base_task_id = sys.argv[1]
run_count = int(sys.argv[2])
root = pathlib.Path("runs/agent")
base_run_dir = root / base_task_id
summary_path = base_run_dir / "baseline_summary.json"

def _read_json(path: pathlib.Path):
    return json.loads(path.read_text(encoding="utf-8"))

def _git_sha() -> str:
    cp = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if cp.returncode == 0:
        return (cp.stdout or "").strip()
    return ""

entries = []
failures = []

for i in range(1, run_count + 1):
    task_id = f"{base_task_id}_r{i}"
    run_dir = root / task_id
    strict_path = run_dir / "strict_acceptance_summary.json"
    result_path = run_dir / "acceptance_result.json"
    release_json = run_dir / "release_evidence.json"

    if not strict_path.exists():
        failures.append(f"{task_id}: missing strict_acceptance_summary.json")
        continue
    if not result_path.exists():
        failures.append(f"{task_id}: missing acceptance_result.json")
        continue
    if not release_json.exists():
        failures.append(f"{task_id}: missing release_evidence.json")
        continue

    strict = _read_json(strict_path)
    if strict.get("generate_adapter") != "reinvent4_generate_adapter_v1":
        failures.append(
            f"{task_id}: generate_adapter mismatch ({strict.get('generate_adapter')})"
        )
    if strict.get("score_adapter") != "unimol_score_adapter_v1":
        failures.append(f"{task_id}: score_adapter mismatch ({strict.get('score_adapter')})")
    if str(strict.get("guardrails_strict_status") or "") != "pass":
        failures.append(
            f"{task_id}: guardrails_strict_status is not pass ({strict.get('guardrails_strict_status')})"
        )

    eval_failed_count = int(strict.get("evaluation_failed_count") or -1)
    guard_failed_count = int(strict.get("guardrails_failed_count") or -1)
    if eval_failed_count != 0:
        failures.append(f"{task_id}: evaluation_failed_count is not 0 ({eval_failed_count})")
    if guard_failed_count != 0:
        failures.append(f"{task_id}: guardrails_failed_count is not 0 ({guard_failed_count})")

    entries.append(
        {
            "task_id": task_id,
            "strict_summary": str(strict_path),
            "result_json": str(result_path),
            "release_evidence_json": str(release_json),
            "generate_adapter": strict.get("generate_adapter"),
            "score_adapter": strict.get("score_adapter"),
            "plqy_target_center": strict.get("plqy_target_center"),
            "guardrails_strict_status": strict.get("guardrails_strict_status"),
            "evaluation_failed_count": eval_failed_count,
            "guardrails_failed_count": guard_failed_count,
        }
    )

report = {
    "status": "pass" if not failures else "fail",
    "base_task_id": base_task_id,
    "run_count": run_count,
    "runs": entries,
    "failures": failures,
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "git_sha": _git_sha(),
}
summary_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

print(json.dumps({"status": report["status"], "baseline_summary": str(summary_path)}, ensure_ascii=False))
if failures:
    raise SystemExit("[FAIL] baseline check failed: " + " | ".join(failures))
PY

echo "[PASS] real-chain baseline succeeded: base_task_id=${BASE_TASK_ID}, runs=${RUN_COUNT}"

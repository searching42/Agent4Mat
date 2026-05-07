#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

required=("UNIMOL_REMOTE_HOST" "UNIMOL_REMOTE_PY" "UNIMOL_REMOTE_TMP_BASE")
missing=()

for v in "${required[@]}"; do
  if [[ -z "${!v:-}" ]]; then
    missing+=("${v}")
  fi
done

allow_default="${ALLOW_DEFAULT_UNIMOL_REMOTE:-0}"

if [[ ${#missing[@]} -gt 0 && "${allow_default}" != "1" ]]; then
  echo "[FAIL] Missing required external runtime env vars: ${missing[*]}"
  echo "       Set all UNIMOL_REMOTE_* vars, or ALLOW_DEFAULT_UNIMOL_REMOTE=1 (legacy defaults)."
  exit 1
fi

export OLED_AGENT_USE_EXTERNAL_SCORER=1

echo "[INFO] External env summary"
echo "       OLED_AGENT_USE_EXTERNAL_SCORER=${OLED_AGENT_USE_EXTERNAL_SCORER}"
echo "       ALLOW_DEFAULT_UNIMOL_REMOTE=${allow_default}"
for v in "${required[@]}"; do
  val="${!v:-}"
  if [[ -n "${val}" ]]; then
    echo "       ${v}=<set>"
  else
    echo "       ${v}=<empty>"
  fi
done

echo "[INFO] Running external preflight..."
PYTHONPATH=src python3 -m oled_agent.cli external-preflight --workspace-root .

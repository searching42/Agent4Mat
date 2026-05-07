#!/usr/bin/env bash
set -euo pipefail

PROFILE="${1:-cpu}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REQ_DIR="${ROOT_DIR}/requirements"
PYTHON_BIN="${PYTHON_BIN:-python3}"

usage() {
  cat <<'EOF'
Usage:
  ./scripts/install_profile.sh <cpu|gpu|dev|base>

Environment:
  PYTHON_BIN             Python executable (default: python3)
  TORCH_CUDA_INDEX_URL  Torch wheel index for GPU profile
                        (default: https://download.pytorch.org/whl/cu121)
  PIP_EXTRA_ARGS        Extra args passed to pip (e.g. --index-url ...)
EOF
}

if [[ "${PROFILE}" == "-h" || "${PROFILE}" == "--help" ]]; then
  usage
  exit 0
fi

REQ_FILE="${REQ_DIR}/${PROFILE}.in"
if [[ ! -f "${REQ_FILE}" ]]; then
  echo "[ERROR] Unknown profile '${PROFILE}', expected one of: base|cpu|gpu|dev" >&2
  exit 2
fi

PIP_EXTRA_ARGS="${PIP_EXTRA_ARGS:-}"

echo "[INFO] Installing profile=${PROFILE} from ${REQ_FILE}"
if [[ "${PROFILE}" == "gpu" ]]; then
  TORCH_CUDA_INDEX_URL="${TORCH_CUDA_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
  echo "[INFO] Installing GPU profile (torch index: ${TORCH_CUDA_INDEX_URL})"
  # shellcheck disable=SC2086
  "${PYTHON_BIN}" -m pip install ${PIP_EXTRA_ARGS} --extra-index-url "${TORCH_CUDA_INDEX_URL}" -r "${REQ_FILE}"
else
  # shellcheck disable=SC2086
  "${PYTHON_BIN}" -m pip install ${PIP_EXTRA_ARGS} -r "${REQ_FILE}"
fi

# Install package itself after dependency profile.
# shellcheck disable=SC2086
"${PYTHON_BIN}" -m pip install ${PIP_EXTRA_ARGS} -e . --no-deps
echo "[INFO] Verifying core package metadata"
"${PYTHON_BIN}" - <<'PY'
import importlib.metadata as md
dist = md.distribution("oled-agent")
print("oled-agent version:", dist.version)
PY

echo "[INFO] Done."

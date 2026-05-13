#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

TASK_PREFIX="${1:-molscribe_input_smoke}"
CATALOG="${2:-scripts/adapters/real_adapters_catalog.json}"

IMAGE_REQ="configs/request_templates/request_molscribe_image.json"
PDF_REQ="configs/request_templates/request_molscribe_pdf.json"

if [[ ! -f "${IMAGE_REQ}" ]]; then
  echo "[FAIL] missing request template: ${IMAGE_REQ}" >&2
  exit 1
fi
if [[ ! -f "${PDF_REQ}" ]]; then
  echo "[FAIL] missing request template: ${PDF_REQ}" >&2
  exit 1
fi
if [[ ! -f "${CATALOG}" ]]; then
  echo "[FAIL] missing catalog: ${CATALOG}" >&2
  exit 1
fi

IMAGE_STUB="runs/ci/${TASK_PREFIX}_image_stub.png"
PDF_STUB="runs/ci/${TASK_PREFIX}_pdf_stub.pdf"
mkdir -p runs/ci

python3 - "${IMAGE_STUB}" "${PDF_STUB}" <<'PY'
from pathlib import Path
import base64
import sys

image_path = Path(sys.argv[1])
pdf_path = Path(sys.argv[2])

# 1x1 transparent PNG
png_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/w8AAgMBgL8qG9kAAAAASUVORK5CYII="
image_path.write_bytes(base64.b64decode(png_b64))

# Minimal valid PDF bytes
pdf_path.write_bytes(b"%PDF-1.4\n1 0 obj<<>>endobj\ntrailer<<>>\n%%EOF\n")
PY

REQ_IMAGE="runs/ci/${TASK_PREFIX}_request_image.json"
REQ_PDF="runs/ci/${TASK_PREFIX}_request_pdf.json"

python3 - "${IMAGE_REQ}" "${REQ_IMAGE}" "${TASK_PREFIX}" "${IMAGE_STUB}" <<'PY'
import json
from pathlib import Path
import sys

src = Path(sys.argv[1])
out = Path(sys.argv[2])
task_prefix = sys.argv[3]
image_stub = Path(sys.argv[4]).resolve()

payload = json.loads(src.read_text(encoding="utf-8"))
payload["task_id"] = f"{task_prefix}_image"
g = payload.setdefault("generation_input", {})
g["source_image"] = str(image_stub)
out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY

python3 - "${PDF_REQ}" "${REQ_PDF}" "${TASK_PREFIX}" "${PDF_STUB}" <<'PY'
import json
from pathlib import Path
import sys

src = Path(sys.argv[1])
out = Path(sys.argv[2])
task_prefix = sys.argv[3]
pdf_stub = Path(sys.argv[4]).resolve()

payload = json.loads(src.read_text(encoding="utf-8"))
payload["task_id"] = f"{task_prefix}_pdf"
g = payload.setdefault("generation_input", {})
g["source_pdf"] = str(pdf_stub)
out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY

export OLED_AGENT_MOLSCRIBE_ADAPTER_MODE=smoke
export OLED_AGENT_UNIMOL_SCORE_MODE=smoke

run_one() {
  local req_json="$1"
  local result_json="$2"
  PYTHONPATH=src python3 -m oled_agent.cli agent-run-json \
    --workspace-root . \
    --catalog "${CATALOG}" \
    --request-json "${req_json}" > "${result_json}"

  python3 - "${result_json}" <<'PY'
import json
from pathlib import Path
import sys

result = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if result.get("status") != "success":
    raise SystemExit(f"result status is not success: {result.get('status')}")

execution = json.loads(Path(result["execution_path"]).read_text(encoding="utf-8"))
records = execution.get("records", [])
by_name = {r.get("name"): r for r in records if isinstance(r, dict)}
gen = ((by_name.get("generate_candidates") or {}).get("result") or {}).get("adapter", "")
score = ((by_name.get("score_candidates") or {}).get("result") or {}).get("adapter", "")
if gen != "molscribe_generate_adapter_v1":
    raise SystemExit(f"unexpected generate adapter: {gen}")
if score != "unimol_score_adapter_v1":
    raise SystemExit(f"unexpected score adapter: {score}")
print(f"[PASS] task_id={result.get('task_id')} gen={gen} score={score}")
PY
}

IMAGE_RESULT="runs/ci/${TASK_PREFIX}_result_image.json"
PDF_RESULT="runs/ci/${TASK_PREFIX}_result_pdf.json"

echo "[1/2] run MolScribe image-input smoke"
run_one "${REQ_IMAGE}" "${IMAGE_RESULT}"

echo "[2/2] run MolScribe pdf-input smoke"
run_one "${REQ_PDF}" "${PDF_RESULT}"

echo "[PASS] MolScribe input smoke completed"
echo "[PASS] image_result=${IMAGE_RESULT}"
echo "[PASS] pdf_result=${PDF_RESULT}"

# Deployment Guide

This guide is the external-user entrypoint for deploying `Agent4Mat` as a reusable repository.

## 1) Minimum support matrix
- Python: `3.10+` (CI baseline)
- OS: Linux/macOS
- Profiles:
  - CPU profile: deterministic baseline + mock/contract checks
  - GPU profile: optional heavy runtime (Uni-Mol/MinerU adapters)

## 2) Quick deployment (recommended)
```bash
git clone <repo-url>
cd Agent4Mat
python3 -m venv .venv
source .venv/bin/activate
./scripts/install_profile.sh cpu
make release-check
```

Expected:
- adapter contract checks pass
- quickstart chain passes
- llm smoke passes
- doctor returns no `fail`

## 3) Environment template
- Copy `.env.example` to your deployment env file or export equivalent variables.
- Two supported scenarios:
  - Scenario A: LLM planner route (command mode or openai_compat backend)
  - Scenario B: external adapter runtime (Uni-Mol/MinerU/remote scorer)

## 4) LLM planner deployment options

### Option A: command mode (highest priority)
```bash
export OLED_AGENT_LLM_PLANNER_CMD="python3 scripts/mock_llm_planner.py"
export MOCK_LLM_MODE=active
```

### Option B: openai_compat backend
```bash
export OLED_AGENT_LLM_BACKEND=openai_compat
export OLED_AGENT_LLM_MODEL=<model-id>
export OLED_AGENT_LLM_API_KEY=<key>
export OLED_AGENT_LLM_BASE_URL=<base-url>
```

Verify:
```bash
make llm-smoke
```

## 5) External adapter deployment (optional)

Enable remote scorer chain only when infra is ready:
```bash
export OLED_AGENT_USE_EXTERNAL_SCORER=1
export UNIMOL_REMOTE_HOST=<user@host>
export UNIMOL_REMOTE_PY=<remote-python-path>
export UNIMOL_REMOTE_TMP_BASE=<remote-tmp-dir>
```

Check:
```bash
PYTHONPATH=src python3 -m oled_agent.cli external-preflight --workspace-root .
PYTHONPATH=src python3 -m oled_agent.cli external-connectivity-debug --workspace-root . --json-out runs/external_debug.json
```

## 6) Real adapter shell validation

Contract-only smoke validation:
```bash
make real-adapter-validate
```

Note:
- `make real-adapter-validate` verifies adapter contract shape, deterministic smoke outputs, and REINVENT4 real-mode logic via a local stub pipeline; it still does not prove real remote runtime availability.
- CI values such as `UNIMOL_REMOTE_HOST=stub_host` and `UNIMOL_REMOTE_PY=stub_py` are placeholders, not real remote runtime checks.

MolScribe request entry (structured payload):
- put image/pdf inputs under `generation_input` in request JSON, e.g.:
  - `source_image` or `source_images`
  - `source_pdf` or `source_pdfs`
  - `image_paths` / `pdf_paths`
- `agent-plan-json` / `agent-run-json` will propagate these fields to `generate_candidates` tool args automatically.

Example (LLM planner + MolScribe path):
```bash
export OLED_AGENT_LLM_PLANNER_CMD="python3 scripts/mock_llm_planner.py"
export MOCK_LLM_MODE=active
export OLED_AGENT_MOLSCRIBE_ADAPTER_MODE=smoke
export OLED_AGENT_UNIMOL_SCORE_MODE=smoke
PYTHONPATH=src python3 -m oled_agent.cli agent-run-json \
  --workspace-root . \
  --catalog scripts/adapters/real_adapters_catalog.json \
  --request-json /abs/path/to/request.json \
  --planner-provider llm_v1
```

Minimal request examples:
- image input:
```json
{
  "task_id": "task_molscribe_image",
  "request_text": "从分子结构图像提取并筛选",
  "mode": "fast_screen",
  "targets": [{"property": "plqy", "objective": "maximize", "target_value": 60.0}],
  "budget": {"max_candidates": 10},
  "model_preferences": {
    "predictor_id": "unimol_lambda_plqy_real_v1",
    "generator_id": "molscribe_generator_real_v1"
  },
  "generation_input": {
    "source_image": "/abs/path/to/figure.png"
  }
}
```
- pdf input (with optional pre-extract hook):
```json
{
  "task_id": "task_molscribe_pdf",
  "request_text": "从论文PDF提取并筛选",
  "mode": "fast_screen",
  "targets": [{"property": "plqy", "objective": "maximize", "target_value": 60.0}],
  "budget": {"max_candidates": 10},
  "model_preferences": {
    "predictor_id": "unimol_lambda_plqy_real_v1",
    "generator_id": "molscribe_generator_real_v1"
  },
  "generation_input": {
    "source_pdf": "/abs/path/to/paper.pdf"
  }
}
```
```bash
export OLED_AGENT_MOLSCRIBE_PDF_EXTRACT_CMD="python3 your_pdf_extract_script.py"
```

Recommended request templates:
- `configs/request_templates/request_molscribe_image.json`
- `configs/request_templates/request_molscribe_pdf.json`
- schema guard: `make request-templates-validate`
- smoke acceptance: `make input-smoke`

Modes:
- `OLED_AGENT_UNIMOL_TRAIN_MODE=preflight|smoke|real`
- `OLED_AGENT_UNIMOL_SCORE_MODE=preflight|smoke|real`
- `OLED_AGENT_MINERU_ADAPTER_MODE=preflight|smoke`

## 7) CI acceptance matrix
- `acceptance-cpu-mock`: `make release-check`
- `acceptance-llm-mock`: `make llm-smoke`
- `acceptance external-adapter (optional)`: manual dispatch only

## 8) Release checklist
1. `make test-regressions`
2. `make release-check`
3. `make real-adapter-validate`
4. confirm `.env.example` matches current runtime knobs

# Troubleshooting

This page lists the most common failure signatures and the fastest fix path.

## 1) `make llm-smoke` fails

Symptoms:
- planner fallback unexpectedly
- command not found or invalid JSON from planner command

Checks:
```bash
echo "$OLED_AGENT_LLM_PLANNER_CMD"
python3 scripts/check_llm_planner_modes.py
```

Fix:
- for deterministic checks, set:
  - `OLED_AGENT_LLM_PLANNER_CMD="python3 scripts/mock_llm_planner.py"`
  - `MOCK_LLM_MODE=active`
- if using backend mode, verify `OLED_AGENT_LLM_*` vars are complete.

## 2) `make quickstart` fails at adapter steps

Symptoms:
- adapter command empty output / invalid JSON
- output csv missing

Checks:
```bash
make adapter-validate
```

Fix:
- ensure adapters print one JSON object to stdout
- send logs to stderr
- return/create valid `output_csv`

## 3) External scorer path falls back to local deterministic

Symptoms:
- `execution.json` shows `local_deterministic_fallback`
- decision summary has fallback code

Checks:
```bash
PYTHONPATH=src python3 -m oled_agent.cli external-preflight --workspace-root .
PYTHONPATH=src python3 -m oled_agent.cli external-connectivity-debug --workspace-root . --json-out runs/external_debug.json
```

Fix:
- set:
  - `OLED_AGENT_USE_EXTERNAL_SCORER=1`
  - `UNIMOL_REMOTE_HOST`
  - `UNIMOL_REMOTE_PY`
  - `UNIMOL_REMOTE_TMP_BASE`
- verify external workspace contains `scripts/score_unimol_property_candidates.py`

## 4) Real adapters blocked by mode

Symptoms:
- `mineru_not_configured`
- `molscribe_input_missing`
- `molscribe_input_not_found`
- `molscribe_runtime_missing` / `molscribe_checkpoint_missing`
- preflight/smoke returns but no real execution

Meaning:
- adapter shells default to safe mode:
  - `OLED_AGENT_UNIMOL_TRAIN_MODE=preflight`
  - `OLED_AGENT_UNIMOL_SCORE_MODE=preflight`
  - `OLED_AGENT_MINERU_ADAPTER_MODE=preflight`
  - `OLED_AGENT_MOLSCRIBE_ADAPTER_MODE=preflight`

Fix:
- for contract smoke: set `smoke`
- for real infra execution:
  - Uni-Mol: set `OLED_AGENT_UNIMOL_*_MODE=real` and complete remote env
  - MolScribe: set `OLED_AGENT_MOLSCRIBE_ADAPTER_MODE=real`, provide `generation_input` and runtime
    - command mode: `OLED_AGENT_MOLSCRIBE_CMD`
    - native mode: `OLED_AGENT_MOLSCRIBE_CHECKPOINT` (optional `OLED_AGENT_MOLSCRIBE_DEVICE`)

## 5) MolScribe/MinerU input path issues

Symptoms:
- `RequestValidationError` on `$.generation_input.*`
- `molscribe_input_missing` / `molscribe_input_not_found`
- MinerU remains in preflight and does not generate real candidates

Checks:
```bash
make request-templates-validate
cat configs/request_templates/request_molscribe_image.json
```

Fix:
- prefer canonical request templates under `configs/request_templates/`
- ensure `generation_input` paths exist on local filesystem
- keep PLQY target value in percent scale (`0-100`, e.g. `60.0`)
- remember current MinerU adapter is contract/preflight+smoke only (no bundled real path)

## 6) Request/plan schema errors

Symptoms:
- `RequestValidationError` with JSON path

Fix:
- validate request payload fields against `schemas/request.schema.json`
- validate tool calls against `schemas/plan.schema.json`

## 7) Fast reset checklist
1. `./scripts/install_profile.sh cpu`
2. `make release-check`
3. `make real-adapter-validate`
4. re-run target command with `PYTHONPATH=src`

# Adapter Templates

This directory provides runnable JSON-in/JSON-out adapter templates for:
- `train_predictor`
- `generate_candidates`
- `score_candidates`
- real adapter shells:
  - `train_predictor_unimol_adapter.py`
  - `score_candidates_unimol_adapter.py`
  - `generate_candidates_mineru_adapter.py`
  - `generate_candidates_reinvent4_adapter.py`
  - `generate_candidates_molscribe_adapter.py`
- `quickstart_catalog.json` (ready-to-run catalog example wired to these templates)
- `real_adapters_catalog.json` (catalog example wired to real adapter shells)
- `validate_adapter_contract.py` (offline contract checker for custom adapters)

You can wire them through model catalog (`configs/models/catalog.json`) or env overrides:
- catalog: `models[*].params.adapters.*_cmd`
- env: `OLED_AGENT_TRAIN_CMD`, `OLED_AGENT_GENERATE_CMD`, `OLED_AGENT_SCORE_CMD`

Resolution order is:
1. env override
2. catalog adapter command
3. built-in fallback behavior

## Tool Contract

All adapters receive one JSON object from `stdin` and must print one JSON object to `stdout`.

### `train_predictor` adapter
- input keys (common): `workspace_root`, `task_id`, `predictor_id`, `targets`, `target_specs`, `state`
- output keys (minimum): `status` (recommended `success`)
- recommended output keys: `adapter`, `predictor_id`, `metrics`

### `generate_candidates` adapter
- input keys (common): `workspace_root`, `task_id`, `generator_id`, `max_candidates`, `constraints`, `output_csv`, `state`
- must produce CSV file at `output_csv` (or return another `output_csv` path)
- output keys (minimum): `status`, `output_csv`
- recommended output keys: `adapter`, `rows`

Expected generated CSV columns:
- required by downstream: `smiles` (or `SMILES`)
- recommended: `candidate_id`

### `score_candidates` adapter
- input keys (common): `workspace_root`, `task_id`, `predictor_id`, `targets`, `target_specs`, `input_csv`, `output_csv`, `state`
- must produce scored CSV file at `output_csv` (or return another `output_csv` path)
- output keys (minimum): `status`, `output_csv`
- recommended output keys: `adapter`

Expected scored CSV columns:
- required by downstream: `candidate_id`, `smiles` (or `SMILES`)
- required by ranking quality: `<property>_score` (for example `plqy_score`)
- recommended: `<property>_pred`, `domain_score`, `common_prior_score`

## Failure Semantics

- `score_candidates` adapter failure:
  - agent falls back to local deterministic scoring
  - execution stays `success`
  - records `fallback_error.code=external_score_cmd_failed`

- `generate_candidates` adapter failure:
  - default bundled REINVENT4 adapter failures fall back to local generation
  - execution can stay `success` with `fallback_error.code=reinvent4_generate_cmd_failed`
  - explicitly configured `OLED_AGENT_GENERATE_CMD` failures still fail directly

## Local Smoke Example

Use repo templates directly via temporary catalog:

```bash
PYTHONPATH=src python3 -m unittest -v \
  tests.test_regressions.RegressionTests.test_agent_run_json_with_repo_adapter_templates_smoke
```

Validate adapter contract before wiring real Uni-Mol/MinerU scripts:

```bash
python3 scripts/adapters/validate_adapter_contract.py \
  --tool score_candidates \
  --cmd "python3 scripts/adapters/score_candidates_adapter_template.py" \
  --workspace-root . \
  --json
```

Validate real adapter shells in deterministic smoke mode:

```bash
make real-adapter-validate
```

Adapter shell modes:
- `OLED_AGENT_UNIMOL_TRAIN_MODE=preflight|smoke|real` (default: `preflight`)
- `OLED_AGENT_UNIMOL_SCORE_MODE=preflight|smoke|real` (default: `preflight`)
- `OLED_AGENT_MINERU_ADAPTER_MODE=preflight|smoke` (default: `preflight`)
- `OLED_AGENT_REINVENT4_ADAPTER_MODE=preflight|smoke|real` (default: `preflight`)
- `OLED_AGENT_MOLSCRIBE_ADAPTER_MODE=preflight|smoke|real` (default: `preflight`)

Real-runtime helper knobs:
- Uni-Mol score:
  - `OLED_AGENT_UNIMOL_SCORE_SCRIPT` (optional local scorer override for real-mode contract checks)
- REINVENT4:
  - `OLED_AGENT_REINVENT4_SOURCE_CSV`
  - `OLED_AGENT_REINVENT4_PIPELINE_SCRIPT`
  - `OLED_AGENT_REINVENT4_RANKREADY_CSV`
  - `OLED_AGENT_REINVENT4_ADAPTER_TIMEOUT_SEC`
- MolScribe:
  - external command mode: `OLED_AGENT_MOLSCRIBE_CMD`
  - native python mode: `OLED_AGENT_MOLSCRIBE_CHECKPOINT` (+ optional `OLED_AGENT_MOLSCRIBE_DEVICE`)
  - optional PDF extraction hook: `OLED_AGENT_MOLSCRIBE_PDF_EXTRACT_CMD`
  - `OLED_AGENT_MOLSCRIBE_ADAPTER_TIMEOUT_SEC`

Recommended rollout:
1. keep `preflight` in CI (fast fail on missing config)
2. use `smoke` locally for contract validation without remote infra
3. switch to `real` only in controlled environments with full runtime

Run the full quickstart chain:

```bash
./scripts/adapters/check_quickstart_chain.sh
# or
make quickstart
```

Use the shipped quickstart catalog:

```bash
cat > /tmp/oled_request_quickstart.json <<'JSON'
{
  "task_id": "task_quickstart_tpl",
  "request_text": "设计470nm附近且高PLQY分子",
  "mode": "fast_screen",
  "targets": [{"property": "plqy", "objective": "maximize", "target_value": 0.6}],
  "budget": {"max_candidates": 5},
  "model_preferences": {
    "predictor_id": "pred_tpl_v1",
    "generator_id": "gen_tpl_v1"
  }
}
JSON

PYTHONPATH=src python3 -m oled_agent.cli agent-run-json \
  --workspace-root . \
  --catalog scripts/adapters/quickstart_catalog.json \
  --request-json /tmp/oled_request_quickstart.json
```

## Common Troubleshooting

- `Tool command returned empty output`
  - adapter did not print JSON to `stdout`
- `Tool command output is not valid JSON`
  - adapter printed logs to `stdout`; print logs to `stderr` instead
- `generate command output csv not found`
  - returned `output_csv` path does not exist
- `score command output csv not found`
  - returned `output_csv` path does not exist
- `Missing smiles/SMILES in candidate row`
  - generated/scored CSV lacks `smiles` or `SMILES`

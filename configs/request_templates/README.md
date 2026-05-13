# Request Templates

Contract-valid request templates for `agent-plan-json` / `agent-run-json`.

## Files
- `request_molscribe_image.json`: image-conditioned generation (`generation_input.source_image`)
- `request_molscribe_pdf.json`: PDF-conditioned generation (`generation_input.source_pdf`)
- `step_request_clean_dataset.json`: step-mode template for `agent-run-step-json` clean step
- `step_request_train_predictor.json`: step-mode template for `agent-run-step-json` training step

## Notes
- PLQY follows percent scale semantics (`0-100`).
- These templates are static examples; copy and replace file paths/task ids before running.

## Validate
```bash
make request-templates-validate
make step-request-templates-validate
```

## Run (smoke)
```bash
export OLED_AGENT_MOLSCRIBE_ADAPTER_MODE=smoke
export OLED_AGENT_UNIMOL_SCORE_MODE=smoke
PYTHONPATH=src python3 -m oled_agent.cli agent-run-json \
  --workspace-root . \
  --catalog scripts/adapters/real_adapters_catalog.json \
  --request-json configs/request_templates/request_molscribe_image.json
```

## Run Step JSON (smoke)
```bash
PYTHONPATH=src python3 -m oled_agent.cli agent-run-step-json \
  --workspace-root . \
  --step-request-json configs/request_templates/step_request_clean_dataset.json
```

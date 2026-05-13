# Request Templates

Contract-valid request templates for `agent-plan-json` / `agent-run-json`.

## Files
- `request_molscribe_image.json`: image-conditioned generation (`generation_input.source_image`)
- `request_molscribe_pdf.json`: PDF-conditioned generation (`generation_input.source_pdf`)

## Notes
- PLQY follows percent scale semantics (`0-100`).
- These templates are static examples; copy and replace file paths/task ids before running.

## Validate
```bash
make request-templates-validate
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

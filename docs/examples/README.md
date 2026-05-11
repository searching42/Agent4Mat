# Example Requests (MolScribe)

This folder contains contract-valid request payload examples for image/PDF-conditioned generation via MolScribe.

## Files
- `request_molscribe_image.json`: image input via `generation_input.source_image`
- `request_molscribe_pdf.json`: PDF input via `generation_input.source_pdf`

## How to run (smoke)
```bash
export OLED_AGENT_MOLSCRIBE_ADAPTER_MODE=smoke
export OLED_AGENT_UNIMOL_SCORE_MODE=smoke
PYTHONPATH=src python3 -m oled_agent.cli agent-run-json \
  --workspace-root . \
  --catalog scripts/adapters/real_adapters_catalog.json \
  --request-json docs/examples/request_molscribe_image.json
```

## How to run (LLM planner + smoke adapters)
```bash
export OLED_AGENT_LLM_PLANNER_CMD="python3 scripts/mock_llm_planner.py"
export MOCK_LLM_MODE=active
export OLED_AGENT_MOLSCRIBE_ADAPTER_MODE=smoke
export OLED_AGENT_UNIMOL_SCORE_MODE=smoke
PYTHONPATH=src python3 -m oled_agent.cli agent-run-json \
  --workspace-root . \
  --catalog scripts/adapters/real_adapters_catalog.json \
  --request-json docs/examples/request_molscribe_pdf.json \
  --planner-provider llm_v1
```

## Expected outputs
- `runs/agent/<task_id>/plan.json`
- `runs/agent/<task_id>/execution.json`
- `runs/agent/<task_id>/decision_summary.json`
- `result/<task_id>-<timestamp>/target_structures.csv`

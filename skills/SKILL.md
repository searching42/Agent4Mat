# Agent4Mat skill router

This router maps user intent to stable CLI commands and tool-specific skills.
Business logic stays in repository Python modules.

## Routing map
- Environment or installation checks:
  - `PYTHONPATH=src python3 -m oled_agent.cli doctor --workspace-root .`
- LLM connectivity checks:
  - `PYTHONPATH=src python3 -m oled_agent.cli llm-connectivity --workspace-root .`
- Deterministic local smoke:
  - `PYTHONPATH=src python3 -m oled_agent.cli smoke --workspace-root .`
- Request planning/execution:
  - `PYTHONPATH=src python3 -m oled_agent.cli agent-plan --workspace-root . --task-id <task_id> --request "<text>"`
  - `PYTHONPATH=src python3 -m oled_agent.cli agent-run --workspace-root . --task-id <task_id> --request "<text>"`

## Tool-specific skills
- Uni-Mol:
  - `skills/unimol/SKILL.md`
- REINVENT4:
  - `skills/reinvent4/SKILL.md`
- MinerU:
  - `skills/mineru/SKILL.md`
- MolScribe:
  - `skills/molscribe/SKILL.md`

## Rules
- Always call repository CLI/adapter entrypoints; do not reimplement pipeline logic in prompt text.
- Keep outputs machine-readable in `runs/` and adapter JSON contracts.
- If a tool is not fully wired yet, return explicit preflight status and next required config.

# oled-agent skill router

This skill delegates execution to stable CLI commands in this repository.

## Use cases
- Validate environment readiness
- Run deterministic smoke tests
- Execute configured OLED pipelines

## Commands
- Environment check:
  - `PYTHONPATH=src python3 -m oled_agent.cli doctor --workspace-root .`
- Smoke test:
  - `PYTHONPATH=src python3 -m oled_agent.cli smoke --workspace-root .`
- Run pipeline:
  - `PYTHONPATH=src python3 -m oled_agent.cli run --config <config.json> --workspace-root .`

## Rules
- Do not implement business logic inside this skill.
- Always invoke repository CLI as the source of truth.
- Keep outputs machine-readable (manifest/report JSON/MD files in `runs/`).

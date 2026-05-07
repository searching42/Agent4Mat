# Agent4Mat MinerU skill

Use this skill when user intent is candidate generation via MinerU-backed adapter flow.

## Current scope
- Repository integration point: `generate_candidates` tool adapter.
- Current adapter script: `scripts/adapters/generate_candidates_mineru_adapter.py`.
- Current mode model:
  - `preflight`: fail-fast with `mineru_not_configured`.
  - `smoke`: deterministic stub CSV output.
- Real MinerU generation command is not wired in repository yet.

## Input contract
- Adapter receives JSON from stdin:
  - `workspace_root`
  - `task_id`
  - `generator_id`
  - `max_candidates`
  - `constraints`
  - `output_csv`
  - `state`

## Output contract
- Adapter writes JSON to stdout:
  - `status`
  - `adapter`
  - `output_csv`
  - `rows`
  - `mode`

## Recommended invocation
- Contract validation smoke:
  - `OLED_AGENT_MINERU_ADAPTER_MODE=smoke python3 scripts/adapters/validate_adapter_contract.py --tool generate_candidates --cmd "python3 scripts/adapters/generate_candidates_mineru_adapter.py" --workspace-root . --json`
- End-to-end with adapter catalog:
  - use `scripts/adapters/real_adapters_catalog.json` and run `agent-run-json`.

## Rules
- Do not bypass adapter contract by writing CSV directly from prompt.
- For production wiring, keep JSON-in/JSON-out contract stable and add explicit error codes for runtime failures.

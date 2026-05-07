# Agent4Mat REINVENT4 skill

Use this skill for generator-side candidate production and reuse of REINVENT4 artifacts.

## Current scope
- Primary model id in catalog: `reinvent4_lambda_em_v2`.
- Tool integration point: `generate_candidates`.
- Adapter command sources:
  - env override: `OLED_AGENT_GENERATE_CMD`
  - catalog adapter: `models[*].params.adapters.generate_candidates_cmd`
  - fallback: local deterministic generation or external artifact reuse path.

## Data contract
- Input payload (JSON stdin) follows `generate_candidates` adapter contract.
- Output must provide CSV path with at least:
  - `smiles` (or `SMILES`, normalized later)
  - recommended `candidate_id`

## Recommended invocation
- Template adapter validation:
  - `python3 scripts/adapters/validate_adapter_contract.py --tool generate_candidates --cmd "python3 scripts/adapters/generate_candidates_adapter_template.py" --workspace-root . --json`
- Full quickstart path:
  - `make quickstart`

## Notes
- In current repository, explicit REINVENT4 runtime execution is adapterized but guarded; production command wiring remains environment-dependent.
- Keep generation deterministic in CI using template/smoke adapters.

## Rules
- Do not emit malformed CSV (missing smiles).
- Keep output schema stable for downstream Uni-Mol scoring and ranking.

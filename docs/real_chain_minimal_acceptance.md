# Real Chain Minimal Acceptance

Goal: run one deterministic "real-mode logic" chain through adapters:
- generate: REINVENT4 adapter in `real` mode via local stub pipeline
- score: Uni-Mol adapter in `real` mode via local stub scorer
- filter/report: regular pipeline path

## Run
```bash
make real-chain-acceptance TASK_ID=real_chain_demo
```

This executes:
- `scripts/run_real_chain_acceptance_minimal.sh`
- `agent-run-json` with `scripts/adapters/real_adapters_catalog.json`
- `scripts/validate_run_artifacts.py`

## Pass criteria
- command exits 0
- `execution.json` has:
  - `generate_candidates.adapter == reinvent4_generate_adapter_v1`
  - `score_candidates.adapter == unimol_score_adapter_v1`
  - no `fallback_error` under `generate_candidates`

## Important
- this is a deterministic contract acceptance path (stub-backed), not proof of remote production infra availability.

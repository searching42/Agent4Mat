# adapters and fallbacks (current behavior)

## generate_candidates
Priority:
1. explicit `input_csv` provided by caller
2. reuse latest `artifacts/server_sync/reinvent4_runs/openclaw_sampling_project_v1_*.csv` (if external workspace is present)
3. local deterministic stub generator

State output:
- `tool_state.candidate_csv`

## score_candidates
Priority:
1. external Uni-Mol scorer script (`workspace/scripts/score_unimol_property_candidates.py`) only if:
   - external workspace scripts available
   - env `OLED_AGENT_USE_EXTERNAL_SCORER=1`
2. local deterministic scoring fallback (always available)

Observability:
- fallback emits machine-readable `fallback_error`:
  - `code`
  - `message`
  - `retryable`
  - `details`
- external scorer error code reference:
  - `docs/external_scorer_error_codes.md`

State output:
- `tool_state.scored_csv`

## filter_and_rank
Priority:
1. if `scored_csv` exists in state, build dynamic ranking config and run local pipeline
2. otherwise run default demo pipeline

State output:
- `tool_state.latest_manifest`
- `tool_state.final_output`

## make_report
Priority:
1. use `tool_state.final_output`
2. fallback to latest run directory scan

## Why this strategy
- keeps pipeline runnable in bare local env
- enables gradual opt-in to real external toolchains
- preserves deterministic behavior for CI/smoke tests

# Workspace Scripts Migration Whitelist

This document maps `workspace/scripts` into Agent4Mat integration status.

## How to refresh map
```bash
make script-map
```

Generated file:
- `docs/script_migration_map.json`

## Whitelist policy
- `integrated`: already wired via adapter/catalog/CLI
- `partially_integrated`: used by adapter real-mode or planned acceptance flow
- `not_in_scope`: research/offline analysis script; do not wire into runtime path

## Current priority whitelist (P0)
- `train_unimol_end2end_plqy_v2_candidate_remote.py`
  - module: `train_predictor`
  - adapter: `scripts/adapters/train_predictor_unimol_adapter.py`
- `score_unimol_property_candidates.py`
  - module: `score_candidates`
  - adapter: `scripts/adapters/score_candidates_unimol_adapter.py`
- `run_reinvent4_lambda_em_v2_pipeline.sh`
  - module: `generate_candidates`
  - adapter: `scripts/adapters/generate_candidates_reinvent4_adapter.py`
- `filter_reinvent4_lambda_em_candidates_v2.py`
  - module: `filter_and_rank` (offline/helper, not direct runtime adapter)

## Notes
- Non-whitelist scripts remain available for research use but are excluded from CI runtime guarantees.
- Promote a script into whitelist only with adapter contract + regression tests.

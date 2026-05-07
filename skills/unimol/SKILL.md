# Agent4Mat Uni-Mol skill

Use this skill for predictor training/scoring via Uni-Mol adapter chain.

## Current scope
- Training adapter: `scripts/adapters/train_predictor_unimol_adapter.py`
- Scoring adapter: `scripts/adapters/score_candidates_unimol_adapter.py`
- Remote scorer bridge script (external workspace): `scripts/score_unimol_property_candidates.py`

## Modes
- Train: `OLED_AGENT_UNIMOL_TRAIN_MODE=preflight|smoke|real`
- Score: `OLED_AGENT_UNIMOL_SCORE_MODE=preflight|smoke|real`
- In `preflight/smoke`, adapters return deterministic contract-safe outputs.
- In `real`, remote runtime env must be complete:
  - `UNIMOL_REMOTE_HOST`
  - `UNIMOL_REMOTE_PY`
  - `UNIMOL_REMOTE_TMP_BASE`

## Recommended commands
- Contract smoke:
  - `make real-adapter-validate`
- External chain readiness:
  - `PYTHONPATH=src python3 -m oled_agent.cli external-preflight --workspace-root .`
  - `PYTHONPATH=src python3 -m oled_agent.cli external-connectivity-debug --workspace-root . --json-out runs/external_debug.json`

## Failure semantics
- `score_candidates` external failure falls back to local deterministic scoring and records `fallback_error`.
- `train_predictor` adapter failure returns structured adapter error and fails that tool step.

## Rules
- Keep `candidate_id`/`smiles` columns normalized before scoring.
- Preserve adapter JSON schema and stable error codes for automation.

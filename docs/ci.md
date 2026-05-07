# CI gates

`oled-agent` uses a dedicated CI workflow:

- workflow file: `.github/workflows/oled-agent-ci.yml`
- trigger scope: changes under `oled-agent/**`

## Gate sequence
1. validate dependency profile pins:
   - `python3 scripts/validate_lockfiles.py --requirements-dir requirements`
2. install deterministic CPU profile:
   - `./scripts/install_profile.sh cpu`
3. run regression tests:
   - `PYTHONPATH=src python3 -m unittest -v tests.test_regressions`
4. run pipeline smoke:
   - `PYTHONPATH=src python3 -m oled_agent.cli smoke --workspace-root .`
5. run agent end-to-end:
   - `PYTHONPATH=src python3 -m oled_agent.cli agent-run --workspace-root . --task-id ci_smoke_task --request "设计470nm附近且高PLQY分子"`
6. validate decision summary schema:
   - `python3 scripts/validate_decision_summary.py runs/agent/ci_smoke_task/decision_summary.json`

## Acceptance matrix
- `acceptance-cpu-mock`:
  - deterministic release gate using `make release-check TASK_ID=ci_accept_cpu_mock`
- `acceptance-llm-mock`:
  - LLM integration gate without credentials using `make llm-smoke`
- `acceptance external-adapter (optional)`:
  - manual `workflow_dispatch` gate with `run_external_acceptance=true`
  - runs `./scripts/run_external_chain_acceptance_with_debug.sh`

## Real adapter contract smoke
- `make real-adapter-validate` exercises adapter shells in smoke mode only.
- It validates contract shape and deterministic outputs, but does not prove a real remote Uni-Mol or MinerU deployment is reachable.
- Treat `stub_host` / `stub_py` values in CI as placeholders for contract validation only.

## Why this gate
- catches lock drift before runtime failures
- guards regression points for fallback scoring and merge schema
- ensures minimal runnable E2E path in non-GPU CI

## Optional integration gate (real external scorer)
- Run manually or in dedicated environment:
  - `./scripts/check_external_env.sh`
  - `./scripts/run_external_chain_acceptance.sh`
- This gate expects:
  - `OLED_AGENT_USE_EXTERNAL_SCORER=1`
  - `UNIMOL_REMOTE_HOST`
  - `UNIMOL_REMOTE_PY`
  - `UNIMOL_REMOTE_TMP_BASE`
  - external workspace with `scripts/score_unimol_property_candidates.py`

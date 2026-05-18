# CI gates

`Agent4Mat` uses a dedicated CI workflow:

- workflow file: `.github/workflows/agent4mat-ci.yml`
- trigger scope: all push/pull_request changes

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
- `acceptance real-chain-minimal (manual)`:
  - manual `workflow_dispatch` gate with `run_real_chain_acceptance=true`
  - runs `make real-chain-acceptance TASK_ID=ci_real_chain_manual`
  - uploads minimal real-chain artifacts under `runs/ci/`
- `acceptance external-adapter (optional)`:
  - manual `workflow_dispatch` gate with `run_external_acceptance=true`
  - runs `./scripts/run_external_chain_acceptance_with_debug.sh`

## UI manual acceptance gates
- workflow: `.github/workflows/agent4mat-ci.yml` (`workflow_dispatch`)
- available inputs:
  - `run_ui_freeze_acceptance=true`:
    - runs `make ui-freeze-acceptance WORKSPACE_ROOT=.`
    - uploads `runs/ci/ui_freeze_acceptance.json`
    - publishes `UI Freeze Acceptance` summary
  - `run_ui_audit_acceptance=true`:
    - runs `make ui-audit-acceptance WORKSPACE_ROOT=.`
    - uploads `runs/ci/ui_audit_acceptance.json`
    - publishes `UI Audit Acceptance` summary
  - `run_ui_release_readiness=true`:
    - runs `make ui-release-readiness WORKSPACE_ROOT=.`
    - uploads `runs/ci/ui_release_readiness.json` + `runs/ci/ui_release_readiness.md`
    - uploads dependency reports `runs/ci/ui_stability_smoke.json` + `runs/ci/ui_freeze_acceptance.json` + `runs/ci/ui_audit_acceptance.json`
    - publishes `UI Release Readiness` summary
  - `run_ui_acceptance_bundle=true`:
    - unified one-click entry for all three UI jobs (`freeze + audit + release-readiness`)
    - also runs `ui-acceptance-bundle-summary` to build bundle verdict artifacts
    - then runs `ui-acceptance-bundle-verify` to re-download and validate bundle artifact schema

### Bundle behavior
- if `run_ui_acceptance_bundle=true`, all three UI jobs are triggered even when single-job inputs are `false`
- you can still run any single UI job by toggling its own input only
- bundle summary job prints:
  - `ui-freeze-acceptance` result
  - `ui-audit-acceptance` result
  - `ui-release-readiness` result
  - `build_bundle_verdict_step` result

## Real adapter contract smoke
- `make real-adapter-validate` exercises adapter shells in smoke mode, plus:
  - REINVENT4 real-mode logic through a local stub pipeline
  - Uni-Mol score real-mode logic through a local stub scorer
- It validates contract shape and deterministic outputs, but does not prove a real remote Uni-Mol, MinerU, or REINVENT4 deployment is reachable.
- Treat `stub_host` / `stub_py` and local stub scripts as placeholders for contract validation only.

## Release boundary and script map
- `make release-boundary` checks git status hygiene for release commits.
- `make script-map` generates `docs/script_migration_map.json` from `workspace/scripts`.

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

## Real baseline artifact template (manual archive)
- for strict real-chain baseline runs, use:
  - `make real-chain-baseline TASK_ID=<base_task_id>`
- to package all baseline evidence into one directory, use:
  - `make real-chain-baseline-archive TASK_ID=<base_task_id>`
  - optional compressed package: `make real-chain-baseline-archive-tgz TASK_ID=<base_task_id>`
  - validate release bundle readiness: `make real-chain-release-bundle-check TASK_ID=<base_task_id>`
- archive the following paths as one release evidence bundle:
  - `runs/agent/<base_task_id>/baseline_summary.json`
  - `runs/agent/<base_task_id>_r1/strict_acceptance_summary.json`
  - `runs/agent/<base_task_id>_r2/strict_acceptance_summary.json`
  - `runs/agent/<base_task_id>_r3/strict_acceptance_summary.json`
  - `runs/agent/<base_task_id>_r1/release_evidence.json`
  - `runs/agent/<base_task_id>_r2/release_evidence.json`
  - `runs/agent/<base_task_id>_r3/release_evidence.json`
- acceptance criteria:
  - `baseline_summary.json` has `status=pass`
  - all three runs show:
    - `generate_adapter = reinvent4_generate_adapter_v1`
    - `score_adapter = unimol_score_adapter_v1`
- packaged output:
  - `runs/archive/<base_task_id>/archive_manifest.json`
  - `runs/archive/<base_task_id>/archive_manifest.md`
  - optional: `runs/archive/<base_task_id>.tar.gz`

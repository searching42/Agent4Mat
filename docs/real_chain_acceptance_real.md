# Real Chain Acceptance (Non-Stub)

This runbook is the production-grade acceptance path after `v0.1.0-rc1`.

Goal:
- execute one full chain with real adapters (no stub):
  - `generate_candidates`: `reinvent4_generate_adapter_v1`
  - `score_candidates`: `unimol_score_adapter_v1`
- require non-fallback result in both execution and decision summary.
- enforce PLQY target semantics in percent scale (`0-100`).
- enforce strict CLI guard via `--require-real-adapters`.

## Step 1. Real runtime precheck (detailed)

### 1.1 Required env vars

Set these before running:

```bash
export OLED_AGENT_USE_EXTERNAL_SCORER=1
export OLED_AGENT_UNIMOL_SCORE_MODE=real
export OLED_AGENT_REINVENT4_ADAPTER_MODE=real

export UNIMOL_REMOTE_HOST=<user@host>
export UNIMOL_REMOTE_PY=<remote_python_path>
export UNIMOL_REMOTE_TMP_BASE=<remote_tmp_dir>

export OLED_AGENT_REINVENT4_PIPELINE_SCRIPT=/abs/path/to/run_reinvent4_lambda_em_v2_pipeline.sh
```

Optional but common:

```bash
export OLED_AGENT_REINVENT4_SOURCE_CSV=/abs/path/to/source_sampling.csv
export OLED_AGENT_UNIMOL_SCORE_SCRIPT=/abs/path/to/score_unimol_property_candidates.py
```

### 1.2 Hard constraints (must satisfy)

- Do **not** use stub values:
  - `UNIMOL_REMOTE_HOST` must not contain `stub`
  - `OLED_AGENT_UNIMOL_SCORE_SCRIPT` must not be `stub_unimol_score.py`
  - `OLED_AGENT_REINVENT4_PIPELINE_SCRIPT` must not be `stub_reinvent4_pipeline.sh`
- All required paths must exist on local machine for adapter launch.

### 1.3 Precheck commands

```bash
PYTHONPATH=src python3 -m oled_agent.cli doctor --workspace-root .
PYTHONPATH=src python3 -m oled_agent.cli external-preflight --workspace-root .
PYTHONPATH=src python3 -m oled_agent.cli external-connectivity-debug \
  --workspace-root . \
  --json-out runs/real_chain_external_debug.json
```

Expected:
- `external-preflight` should not report `overall=fail`
- `external-connectivity-debug` should produce a JSON artifact (even if warn/fail, keep as evidence)

## Step 2. Run real acceptance task (detailed)

Use the dedicated script:

```bash
./scripts/run_real_chain_acceptance_real.sh \
  accept_real_chain_001 \
  "设计470nm附近且高PLQY分子" \
  scripts/adapters/real_adapters_catalog.json \
  runs/agent/accept_real_chain_001/external_debug.json
```

What this script does:
- enforces required env and anti-stub checks
- runs `doctor` + `external-preflight`
- runs `external-connectivity-debug` for evidence
- executes `agent-run-json` with real adapter catalog
  - includes `--require-real-adapters` (fail-fast on any fallback/local adapter)
- validates structured artifacts (`decision_summary/task_state/data_report/model_report/filtering_report`)
- enforces:
  - `generate_candidates.adapter == reinvent4_generate_adapter_v1`
  - `score_candidates.adapter == unimol_score_adapter_v1`
  - no fallback in tool results
  - `decision_summary.score_step.used_fallback != true`
  - `plan.design_spec.targets[name=plqy].target_center` is numeric and in percent-scale range (`1 < center <= 100`)
- collects release evidence package:
  - `runs/agent/<task_id>/release_evidence.json`
  - `runs/agent/<task_id>/release_evidence.md`
  - `runs/agent/<task_id>/strict_acceptance_summary.json`

## Step 3. Evidence package (required for release gate)

Collect and archive:
- `runs/agent/<task_id>/acceptance_result.json`
- `runs/agent/<task_id>/external_debug.json`
- `runs/agent/<task_id>/release_evidence.json`
- `runs/agent/<task_id>/release_evidence.md`
- `runs/agent/<task_id>/strict_acceptance_summary.json`
- `runs/agent/<task_id>/decision_summary.json`
- `runs/agent/<task_id>/task_state.json`
- `runs/agent/<task_id>/execution.json`

Also record:
- exact command line used
- env snapshot (redacted keys/tokens)
- commit SHA

## Step 4. Baseline reproducibility (3 consecutive runs)

For release-level baseline stability, run:

```bash
make real-chain-baseline TASK_ID=real_chain_baseline_001
```

This command executes strict real acceptance 3 times:
- `real_chain_baseline_001_r1`
- `real_chain_baseline_001_r2`
- `real_chain_baseline_001_r3`

And aggregates into:
- `runs/agent/real_chain_baseline_001/baseline_summary.json`

Pass criteria:
- all three runs succeed
- each run contains:
  - `strict_acceptance_summary.json`
  - `release_evidence.json`
- aggregate `baseline_summary.json` has `status=pass`

## Common failure mapping

- `missing required env`
  - set all `UNIMOL_REMOTE_*` and `OLED_AGENT_REINVENT4_PIPELINE_SCRIPT`
- `stub-like value detected`
  - you are still pointing to stub host/script; replace with real endpoints
- `reinvent4_rankready_missing`
  - pipeline ran but rankready CSV missing; inspect pipeline output and run tag
- `external_runtime_config_incomplete`
  - `UNIMOL_REMOTE_HOST/PY/TMP_BASE` not set together
- `score fallback detected`
  - check remote scorer script path, remote connectivity, and stderr in execution artifacts

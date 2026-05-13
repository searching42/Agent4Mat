# Real Chain No-Fallback Quickstart

This is the shortest production acceptance path for one real task run with strict no-fallback enforcement.

## 1) Required env

```bash
export OLED_AGENT_USE_EXTERNAL_SCORER=1
export OLED_AGENT_UNIMOL_SCORE_MODE=real
export OLED_AGENT_REINVENT4_ADAPTER_MODE=real

export UNIMOL_REMOTE_HOST=<user@host>
export UNIMOL_REMOTE_PY=<remote_python_path>
export UNIMOL_REMOTE_TMP_BASE=<remote_tmp_dir>

export OLED_AGENT_REINVENT4_PIPELINE_SCRIPT=/abs/path/to/run_reinvent4_lambda_em_v2_pipeline.sh
```

Optional:

```bash
export OLED_AGENT_REINVENT4_SOURCE_CSV=/abs/path/to/source_sampling.csv
export OLED_AGENT_UNIMOL_SCORE_SCRIPT=/abs/path/to/score_unimol_property_candidates.py
```

## 2) One command

```bash
make real-chain-acceptance-real TASK_ID=real_chain_no_fallback_001
```

This command already enforces:
- real-mode env checks
- anti-stub checks
- `--require-real-adapters`
- decision summary fallback checks
- release evidence collection

## 3) Pass criteria

- command exits `0`
- `runs/agent/<task_id>/strict_acceptance_summary.json` exists and shows:
  - `generate_adapter = reinvent4_generate_adapter_v1`
  - `score_adapter = unimol_score_adapter_v1`
- `runs/agent/<task_id>/release_evidence.json` exists

## 4) Key artifacts

- `runs/agent/<task_id>/acceptance_result.json`
- `runs/agent/<task_id>/execution.json`
- `runs/agent/<task_id>/decision_summary.json`
- `runs/agent/<task_id>/strict_acceptance_summary.json`
- `runs/agent/<task_id>/release_evidence.json`
- `runs/agent/<task_id>/release_evidence.md`

## 5) Baseline reproducibility (x3)

For release baseline, run strict acceptance 3 times continuously:

```bash
make real-chain-baseline TASK_ID=real_chain_baseline_001
```

Expected:
- command exits `0`
- each run `runs/agent/<task_id>_r1|r2|r3/` has strict and release evidence artifacts
- aggregate summary exists:
  - `runs/agent/<task_id>/baseline_summary.json`
  - status is `pass`

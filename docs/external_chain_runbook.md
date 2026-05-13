# External scorer chain runbook

This runbook defines how to run and diagnose the real external Uni-Mol scoring chain.

## 1) Preconditions

1. Enable external scorer path:
   - `export OLED_AGENT_USE_EXTERNAL_SCORER=1`
2. Configure remote runtime (recommended, explicit):
   - `export UNIMOL_REMOTE_HOST=<user@host>`
   - `export UNIMOL_REMOTE_PY=<remote_python_path>`
   - `export UNIMOL_REMOTE_TMP_BASE=<remote_writable_tmp_dir>`
3. Optional legacy default fallback (not recommended for shared deployment):
   - `export ALLOW_DEFAULT_UNIMOL_REMOTE=1`
2. Ensure scorer script is present in external workspace:
   - `<external_workspace>/scripts/score_unimol_property_candidates.py`
3. Ensure remote transport is available (script currently uses `ssh/scp`).

## 2) Preflight

```bash
cd /path/to/Agent4Mat
PYTHONPATH=src python3 -m oled_agent.cli external-preflight --workspace-root .
```

or:

```bash
cd /path/to/Agent4Mat
./scripts/check_external_env.sh
```

Expected:
- `PASS` when:
  - scorer script exists and `--help` probe works
  - `ssh/scp` commands are available
  - remote runtime config is valid
  - SSH connectivity, remote python executability, and remote tmp dir writability pass

If preflight fails, inspect check names:
- `external:runtime_config`: UNIMOL env vars incomplete/missing
- `external:ssh_connectivity`: auth/network/host key failures
- `external:remote_python`: remote python path missing or not executable
- `external:remote_tmp_base`: remote tmp path permission/path issue

Machine-readable debug:

```bash
cd /path/to/Agent4Mat
PYTHONPATH=src python3 -m oled_agent.cli external-connectivity-debug --workspace-root . --json-out runs/external_debug.json
```

Read `runs/external_debug.json`:
- `connectivity.chain_ready`
- `connectivity.blocking_checks`
- `connectivity.failure_classes`

## 3) Acceptance

```bash
cd /path/to/Agent4Mat
PYTHONPATH=src ./scripts/run_external_chain_acceptance.sh <task_id>
```

or (recommended, includes auto debug artifacts):

```bash
cd /path/to/Agent4Mat
PYTHONPATH=src ./scripts/run_external_chain_acceptance_with_debug.sh <task_id>
```

This debug acceptance script always writes:
- `runs/agent/<task_id>/external_debug.json`
- `runs/agent/<task_id>/decision_summary.json`
even when acceptance fails.

Expected:
- decision summary validates
- adapter is `external_unimol_script`

## 4) Failure interpretation

If acceptance fails with fallback, inspect:

1. `runs/agent/<task_id>/decision_summary.json`
   - `score_step.fallback_code`
   - `score_step.fallback_retryable`
   - `score_step.fallback_details.errors[*].stderr_tail`
2. `runs/agent/<task_id>/execution.json`
   - `records[name=score_candidates].result.fallback_error`
3. `runs/agent/<task_id>/external_debug.json`
   - `connectivity.blocking_checks`
   - `connectivity.failure_classes`

## 5) Current known blocker in this workspace

As of `2026-05-01`, external scoring reaches the remote transport stage but fails with `scp` return code `255`, so acceptance falls back to local scoring:

- `fallback_code`: `external_command_failed`
- `fallback_retryable`: `true`

Action:
- verify SSH trust/network reachability to your configured `UNIMOL_REMOTE_HOST`
- verify remote path permissions under `/home/lbh/work/wk1/openclaw_sync`
- migrate to explicit `UNIMOL_REMOTE_*` env vars in deployment scripts

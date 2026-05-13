# Release Boundary Guide

This guide defines what can be committed to `main` for a reproducible release.

## Allowed
- source code, configs, schemas, adapter scripts, docs
- deterministic examples under `docs/examples/`

## Blocked (must not be committed)
- runtime artifacts:
  - `logging/`
  - `result/`
- ad-hoc local debug outputs

## Command
```bash
python3 scripts/check_release_boundary.py --workspace-root . --json
```

Pass criteria:
- no blocked runtime artifact paths in git status
- no disallowed untracked files

## Recommended flow
1. run `make release-boundary`
2. run `make script-map`
3. stage by topic (`contract`, `adapter`, `docs`, `ci`)
4. run `make release-check`
5. create tagged release commit
6. for real-chain release evidence, run:
   - `make real-chain-baseline TASK_ID=<base_task_id>` (recommended, 3 consecutive strict runs)
   - and verify `runs/agent/<base_task_id>/baseline_summary.json` has `status=pass`
   - and package artifacts: `make real-chain-baseline-archive TASK_ID=<base_task_id>`
   - optional compressed package: `make real-chain-baseline-archive-tgz TASK_ID=<base_task_id>`
   - or `make real-chain-acceptance-real TASK_ID=<task_id>` for a single strict run
   - or `make real-chain-evidence TASK_ID=<task_id>` on an existing acceptance run

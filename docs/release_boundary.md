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

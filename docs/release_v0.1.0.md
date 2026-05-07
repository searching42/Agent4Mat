# Release v0.1.0

Release date: `2026-05-06`

Version source of truth:
- `pyproject.toml` -> `[project].version = "0.1.0"`

## Scope

`v0.1.0` is the first reusable repository release focused on deterministic execution, adapter contracts, and CI reproducibility.

Included capability groups:
- deterministic agent CLI pipeline
- request/plan/decision schema validation and regression gates
- adapter templates + real adapter shell preflight/smoke
- short make entrypoints for external users
- CI acceptance matrix (`cpu-mock`, `llm-mock`, optional `external-adapter`)
- deployment and troubleshooting documentation

## Pre-release verification

Run in repository root:

```bash
./scripts/install_profile.sh cpu
make test-regressions
make release-check
make real-adapter-validate
```

Expected:
- tests pass
- no `fail` in `doctor`
- quickstart and llm smoke pass
- real adapter shells pass deterministic smoke contract

## Tagging and publish procedure

```bash
git checkout master
git pull --ff-only
git tag -a v0.1.0 -m "oled-agent v0.1.0"
git push origin master
git push origin v0.1.0
```

If publishing a GitHub release:
- title: `v0.1.0`
- description body:
  - copy summary bullets from `CHANGELOG.md` (`0.1.0`)
  - include quickstart snippet:
    - `./scripts/install_profile.sh cpu`
    - `make release-check`

## Post-release checks
1. Confirm tag points to intended commit.
2. Confirm CI jobs all green on tag.
3. Validate README quickstart on a fresh environment.

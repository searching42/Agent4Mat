# Release v0.1.0-rc1

Release candidate date: `2026-05-11`

Version source of truth:
- `pyproject.toml` -> `[project].version = "0.1.0"`

RC anchor commits:
- `815122e` add release boundary checks, migration map tooling, and manual real-chain acceptance gate
- `5e5ef35` integrate real adapter shells and structured artifact schemas with contract validation

## Scope

`v0.1.0-rc1` is the repository release-candidate baseline for external reproducibility.

Included capability groups:
- deterministic CLI flow (`agent-plan/agent-run` + JSON variants)
- contract validation and schema guardrails (`request/plan/decision/task_state/data_report/model_report/filtering_report`)
- adapter contract templates + real adapter shells (Uni-Mol/REINVENT4/MolScribe/MinerU)
- release boundary and migration-map controls:
  - `make release-boundary`
  - `make script-map`
- manual CI acceptance gate for minimal real-chain path

## Pre-tag verification

Run in repository root:

```bash
./scripts/install_profile.sh cpu
PYTHONPATH=src python3 -m unittest -v tests.test_regressions
make release-check TASK_ID=rc1_release_check
make real-adapter-validate
make real-chain-acceptance TASK_ID=rc1_real_chain
make release-boundary
```

Expected:
- regression suite all green
- quickstart/llm smoke/doctor chain passes
- real adapter contract smoke passes
- real-chain minimal acceptance returns:
  - `generate_adapter = reinvent4_generate_adapter_v1`
  - `score_adapter = unimol_score_adapter_v1`
- release boundary status is `pass`

## Tagging procedure

```bash
git checkout main
git pull --ff-only
git tag -a v0.1.0-rc1 -m "Agent4Mat v0.1.0-rc1"
git push origin main
git push origin v0.1.0-rc1
```

## GitHub release draft guidance

Suggested title:
- `v0.1.0-rc1`

Suggested notes:
- include CI pass statement for this RC
- include anchor commit hashes (`815122e`, `5e5ef35`)
- include quick-start commands:
  - `./scripts/install_profile.sh cpu`
  - `make release-check`
  - `make real-chain-acceptance TASK_ID=rc1_real_chain`

## Rollback point

If RC needs rollback:
- delete tag locally/remotely:
```bash
git tag -d v0.1.0-rc1
git push origin :refs/tags/v0.1.0-rc1
```
- reset release candidate baseline to commit `2114cf6` if needed (do this in a separate recovery branch, not on published `main` directly).

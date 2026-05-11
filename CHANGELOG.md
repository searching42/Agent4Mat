# Changelog

All notable changes to this project are documented in this file.

## [Unreleased]

### Release Candidate
- `v0.1.0-rc1` baseline prepared on `2026-05-11`.
- baseline scope is anchored by:
  - `815122e` (`release-boundary`/`script-map`/manual `real-chain` acceptance gate/UI smoke scaffold)
  - `5e5ef35` (real adapter shells + structured artifact schemas + contract validators)
- GitHub Actions status for `main`: all required jobs passed before RC tagging.

### Added
- LLM connectivity diagnostics:
  - new CLI command `llm-connectivity`
  - backend/command probe report (`report_type=llm_connectivity_v1`)
  - optional JSON artifact output for automation/debug.
- Makefile short entrypoint:
  - `make llm-connectivity` for quick LLM source/backend checks.
- Tool-oriented skill docs:
  - `skills/unimol/SKILL.md`
  - `skills/reinvent4/SKILL.md`
  - `skills/mineru/SKILL.md`
  - `skills/molscribe/SKILL.md`
  - updated router `skills/SKILL.md`.
- Repository-level assistant behavior policy:
  - `SOUL.md`.

### Changed
- README synchronized with current command surface:
  - documented `llm-connectivity` usage and intent
  - added Skills and `SOUL.md` references
  - clarified LLM smoke path (`make llm-smoke` based on mock planner mode checks).
- Regression suite expanded to lock LLM connectivity behavior:
  - not-configured failure path
  - command probe success path
  - backend HTTP error classification path.

## [0.1.0] - 2026-05-06

### Added
- Reusable `Agent4Mat` repository skeleton with deterministic CLI workflow:
  - `agent-plan`, `agent-run`, `agent-plan-json`, `agent-run-json`
  - machine-readable artifacts (`plan.json`, `execution.json`, `tool_state.json`, `decision_summary.json`)
- Request/plan/decision schema validation guards and regression lock suite.
- Makefile short entrypoints:
  - `quickstart`, `doctor`, `llm-smoke`, `adapter-validate`, `real-adapter-validate`, `release-check`.
- Adapter ecosystem:
  - template adapters (`train/generate/score`)
  - quickstart adapter catalog
  - adapter contract validator and quickstart self-check chain
  - real adapter shells (Uni-Mol train/score, MinerU generate preflight/smoke)
- CI guardrails:
  - schema sync check with artifact output
  - adapter contract guard
  - make entrypoint guard
  - acceptance matrix jobs (`cpu-mock`, `llm-mock`, optional `external-adapter`)
- Deployment and operations docs:
  - `docs/deploy.md`
  - `docs/troubleshooting.md`
  - external chain runbook and error-code references
  - dual-scenario `.env.example` template (LLM + external adapter runtime)

### Changed
- Packaging metadata consolidated under `pyproject.toml` with `setup.py` as compatibility shim.
- Planner provider behavior hardened:
  - command mode priority over backend mode
  - structured fallback reasons for invalid output/backend failures.
- Candidate normalization and scoring fallback behavior stabilized for mixed CSV schemas.

### Fixed
- Removed invalid/stale CI script references and moved workflow checks to repository-root Actions path.
- Fixed fallback scoring on uppercase `SMILES` source rows.
- Fixed adapter contract and command entrypoint regressions via dedicated tests.

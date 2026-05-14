# Agent4Mat current capabilities

## A. Core workflow engine
- Config-driven stage orchestration (`run` command)
- Deterministic outputs under `runs/<run_id>/`
- Machine-readable `manifest.json` per run

Implemented stages:
- `compose_multi_objective`
- `filter_multi_objective`
- `export_simple_report`

## B. Environment and deployment guards
- `doctor` command:
  - checks Python/version, writable workspace, core commands
  - checks docker compose availability
  - checks GPU visibility (`nvidia-smi`)
  - checks optional modules (`torch`, `rdkit`, `unimol_tools`, `magic_pdf`)
  - checks optional env vars (`HF_ENDPOINT`, `UNIMOL_WEIGHT_DIR`)
- `external-preflight` command:
  - validates external scorer chain availability and enablement state
  - validates remote runtime env (`UNIMOL_REMOTE_*`) completeness
  - validates `ssh/scp` availability and remote connectivity/python/tmp-base checks
- `smoke` command:
  - runs minimal demo pipeline and verifies final artifact exists

## C. Agent layer (plan + tool-calling)
- `agent-plan`:
  - converts user request to structured `DesignSpec`
  - validates model selection against model catalog
  - outputs executable `ToolCall[]`
  - supports pluggable planner-provider routing (`rule_based_v1` default; `llm_v1` supports external command invocation via env contract)
  - LLM plan payload is schema-validated (`schemas/plan.schema.json`) before acceptance
- `agent-plan-json`:
  - validates `request.json` against `schemas/request.schema.json`
  - builds plan directly from structured request payload
  - supports `generation_input` (e.g. `source_image`, `source_pdf`, `image_paths`) and propagates to `generate_candidates.args`
- `agent-run`:
  - executes tool calls in sequence
  - saves `plan.json`, `execution.json`, `tool_state.json`, `task_state.json`
  - saves `artifacts/experiment_trace.json` with run fingerprint/model+adapter/data lineage snapshot
  - saves machine-readable `decision_summary.json` (fallback usage/error code/retryability)
  - supports planner-provider routing and persists provider metadata in `design_spec.metadata`
  - mirrors normalized artifacts into:
    - `logging/<task_id>-<timestamp>/` (`task.json`, `plan.md`, `execution.log`, `data_report.json`, `model_report.json`, `filtering_report.json`, structured JSON artifacts)
    - `result/<task_id>-<timestamp>/` (`target_structures.csv`, `metadata.json`, optional `report.md`)
- `agent-run-json`:
  - validates `request.json` against schema
  - executes from structured request payload
  - writes `request.json` artifact under run directory
- decision summary validation script:
  - `scripts/validate_decision_summary.py` now reuses `validate_decision_summary_payload()`
  - single source of truth is `schemas/decision_summary.schema.json`
- task-state validation script:
  - `scripts/validate_task_state.py` now reuses `validate_task_state_payload()`
  - single source of truth is `schemas/task_state.schema.json`
- structured logging report validation scripts:
  - `scripts/validate_data_report.py` -> `validate_data_report_payload()`
  - `scripts/validate_model_report.py` -> `validate_model_report_payload()`
  - `scripts/validate_filtering_report.py` -> `validate_filtering_report_payload()`
  - single source of truth is `schemas/data_report.schema.json`, `schemas/model_report.schema.json`, `schemas/filtering_report.schema.json`
- built-in mock LLM planner script:
  - `scripts/mock_llm_planner.py` can be used with `OLED_AGENT_LLM_PLANNER_CMD` for local/CI verification of `llm_v1` path
  - `MOCK_LLM_MODE` supports deterministic scenario simulation: `active|bad_json|bad_tools|bad_model|exit_nonzero`
- CI `llm_v1` gate now covers all 5 mock modes via `scripts/check_llm_planner_modes.py`

## D. Model selection
- `configs/models/catalog.json` supports user-selectable:
  - predictor models
  - generator models
- planner and runtime validate predictor/generator IDs

## E. Tool adapters and fallbacks
### generate_candidates
- explicit input CSV (if provided)
- else reuse latest REINVENT4 artifact from external workspace
- else deterministic local stub generation
- supports image/PDF-conditioned generation args for adapters (MolScribe path):
  - `source_image`, `source_images`
  - `source_pdf`, `source_pdfs`
  - `input_image`, `input_pdf`
  - `paper_path`, `image_paths`, `pdf_paths`

### score_candidates
- tries external Uni-Mol scorer only if explicitly enabled (`OLED_AGENT_USE_EXTERNAL_SCORER=1`)
- otherwise deterministic local fallback scoring
- supports multi-target specs (`lambda_em`, `plqy`, etc.)
- fallback carries structured `fallback_error.code` for downstream automation
- remote scorer script now supports explicit remote runtime envs:
  - `UNIMOL_REMOTE_HOST`
  - `UNIMOL_REMOTE_PY`
  - `UNIMOL_REMOTE_TMP_BASE`
  - optional `ALLOW_DEFAULT_UNIMOL_REMOTE=1` for legacy defaults

### filter_and_rank
- builds dynamic ranking config from scored candidates and target specs
- runs local deterministic pipeline to produce topN/report

### make_report
- returns current task's final report path from tool state

## F. Skills / plugin-style invocation bridge
- `skills/SKILL.md` included as routing-only layer:
  - calls repository CLI commands
  - keeps business logic in Python modules

## G. Containerization scaffolding
- `Dockerfile`
- `docker-compose.yml` with `cpu` and `gpu` profiles

## H. Environment pinning and install automation
- profile-based pinned dependency files under `requirements/`:
  - `base.in` / `cpu.in` / `gpu.in` / `dev.in`
- install helper:
  - `scripts/install_profile.sh <profile>`
- lock quality checker:
  - `scripts/validate_lockfiles.py`

## I. Current constraints
- real Uni-Mol and REINVENT4 execution is partially adapterized, but still guarded/fallback-first
- `llm_v1` requires user-provided external command (`OLED_AGENT_LLM_PLANNER_CMD`) for real planning; invalid output/command failures auto-fallback to `rule_based_v1` with structured reason
- no async queue/job scheduler yet for long-running training/generation tasks

## J. Release and acceptance controls
- release boundary check:
  - `scripts/check_release_boundary.py`
  - `make release-boundary`
- workspace script migration mapping:
  - `scripts/build_script_migration_map.py`
  - `make script-map`
  - output: `docs/script_migration_map.json`
- real-chain minimal acceptance:
  - `scripts/run_real_chain_acceptance_minimal.sh`
  - `make real-chain-acceptance`
  - verifies real-mode adapter wiring with local deterministic stubs
- real-chain strict no-fallback acceptance:
  - `scripts/run_real_chain_acceptance_real.sh`
  - `make real-chain-acceptance-real`
  - enforces `--require-real-adapters` and persists `strict_acceptance_summary.json`
- experiment trace guard:
  - `scripts/check_experiment_trace.py`
  - `make experiment-trace-guard`
  - verifies full-pipeline and single-step runs both emit valid `experiment_trace` artifacts (run/logging/result)
- experiment summary:
  - `scripts/summarize_experiments.py`
  - `make experiment-summary`
  - outputs aggregated status/model/mode counters and recent run list from experiment traces
  - Markdown report prioritizes failed runs and includes score adapter/fallback signals
  - CI `agent4mat-ci` publishes `runs/ci/experiment_summary.json` + `runs/ci/experiment_summary.md` as artifact and step summary
- real-chain reproducibility baseline (3 consecutive strict runs):
  - `scripts/run_real_chain_baseline.sh`
  - `make real-chain-baseline`
  - persists aggregate `runs/agent/<task_id>/baseline_summary.json`
- lightweight UI prototype smoke:
  - `ui/app.py`
  - supports chat-first workflow (project memory + chat turn orchestration + file input entry)
  - supports project APIs (`/api/projects`, `/api/projects/<id>/history`, `/api/chat/send`, `/api/projects/<id>/upload-ref`)
  - keeps full pipeline / single-step / intake / approve / resume APIs for compatibility
  - supports experiment list endpoint (`/api/experiments`) for trace-level filtering
  - task inspector supports recent-task picker, artifact preview, timeline view (failed highlight + filter/sort), run-to-run compare, artifact key-path diff, and one-click core artifact validation
  - `make ui-smoke`

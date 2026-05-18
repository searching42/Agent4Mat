# Agent4Mat

A reusable, config-driven materials workflow runner with an agent layer (plan + tool-calling + execution records).

Version: `0.1.0`  
Release notes:
- `CHANGELOG.md`

## Environment profiles
- `requirements/base.in`: minimal runtime
- `requirements/cpu.in`: deterministic local/CI profile
- `requirements/gpu.in`: Uni-Mol + MinerU capable profile
- `requirements/dev.in`: dev/test profile

Install by profile:
```bash
cd /path/to/Agent4Mat
./scripts/install_profile.sh cpu
# or
./scripts/install_profile.sh gpu
```

Validate profile pinning:
```bash
python3 scripts/validate_lockfiles.py --requirements-dir requirements
```

## Design goals
- Reproducible runs without chat-memory dependence.
- Deterministic stage orchestration with explicit inputs/outputs.
- Machine-readable manifests for audit and replay.
- Gate/policy logic implemented in Python, not prompts.
- Agent layer that can evolve from rule planner to LLM function-calling planner.

## Quick start
```bash
cd /path/to/Agent4Mat
make quickstart
make doctor
PYTHONPATH=src python3 -m oled_agent.cli smoke --workspace-root .
```

Make targets:
- `make release-check`
- `make release-boundary`
- `make script-map`
- `make request-templates-validate`
- `make step-request-templates-validate`
- `make input-smoke`
- `make experiment-summary`
- `make intake-contract-guard`
- `make step-mode-guard`
- `make web-evidence-guard`
- `make experiment-trace-guard`
- `make real-no-fallback-gate`
- `make quickstart`
- `make doctor`
- `make llm-connectivity`
- `make adapter-validate`
- `make real-adapter-validate`
- `make real-chain-acceptance`
- `make real-chain-baseline`
- `make real-chain-baseline-archive`
- `make real-chain-baseline-archive-tgz`
- `make real-chain-release-bundle-check`
- `make ui-smoke`
- `make test-regressions`
- `make test-adapters`

## Cold-start onboarding (new machine)
Use this path to verify an external user can run the deterministic chain with minimal setup:

```bash
git clone <your-repo-url>
cd Agent4Mat
python3 -m venv .venv
source .venv/bin/activate
./scripts/install_profile.sh cpu
make release-check
make adapter-validate
make quickstart
make doctor
```

Expected outcomes:
- `make release-check` runs adapter validation + quickstart + llm smoke + doctor in one pass.
- `make adapter-validate` prints JSON success for train/generate/score adapter templates.
- `make quickstart` ends with `[PASS] quickstart chain completed`.
- `make doctor` reports environment diagnostics without fatal error for cpu profile.
- `make real-adapter-validate` validates adapter contracts with deterministic smoke outputs plus REINVENT4 `real` mode through a local stub pipeline script (still not a proof of real remote runtime availability).
 - real non-stub acceptance runbook:
   - `docs/real_chain_acceptance_real.md`

Release boundary + migration map:
```bash
make release-boundary
make script-map
```

Real-chain minimal acceptance (stub-backed real-mode logic):
```bash
make real-chain-acceptance TASK_ID=real_chain_demo
```

Real-chain production acceptance (non-stub):
```bash
./scripts/run_real_chain_acceptance_real.sh \
  accept_real_chain_001 \
  "设计470nm附近且高PLQY分子" \
  scripts/adapters/real_adapters_catalog.json \
  runs/agent/accept_real_chain_001/external_debug.json
```
This acceptance path also enforces PLQY target semantics in percent scale (`0-100`).
It also writes release evidence files:
- `runs/agent/<task_id>/release_evidence.json`
- `runs/agent/<task_id>/release_evidence.md`

Collect evidence later from an existing acceptance run:
```bash
make real-chain-evidence TASK_ID=accept_real_chain_001
```

## Recommended execution paths (single source of truth)
### Path A: strict request.json entry (fastest for CI and reproducibility)
```bash
PYTHONPATH=src python3 -m oled_agent.cli agent-run-json \
  --workspace-root . \
  --catalog configs/models/catalog.json \
  --request-json /abs/path/to/request.json
```

### Path B: task.v2 intake -> approve -> run -> resume
```bash
# 1) intake draft + missing questions (+web evidence)
PYTHONPATH=src python3 -m oled_agent.cli agent-intake \
  --workspace-root . \
  --task-id task_v2_demo \
  --request "设计470nm附近且高PLQY分子"

# 2) edit runs/agent/task_v2_demo/task.draft.json, then approve
PYTHONPATH=src python3 -m oled_agent.cli agent-approve \
  --workspace-root . \
  --task-json runs/agent/task_v2_demo/task.draft.json

# 3) execute from approved request
PYTHONPATH=src python3 -m oled_agent.cli agent-run-json \
  --workspace-root . \
  --catalog configs/models/catalog.json \
  --request-json runs/agent/task_v2_demo/request_from_task.json

# 4) idempotent resume (skip completed steps, continue from first unfinished)
PYTHONPATH=src python3 -m oled_agent.cli agent-resume \
  --workspace-root . \
  --task-id task_v2_demo \
  --catalog configs/models/catalog.json
```

### Path C: single-step operation mode
```bash
PYTHONPATH=src python3 -m oled_agent.cli agent-run-step-json \
  --workspace-root . \
  --catalog configs/models/catalog.json \
  --step-request-json configs/request_templates/step_request_clean_dataset.json
```
Supported `operation` values:
- `retrieve_candidate_data`
- `clean_dataset`
- `prepare_train_data`
- `train_predictor`
- `generate_candidates`
- `score_candidates`
- `filter_and_rank`
- `make_report`

Web evidence notes:
- `search_web_evidence` supports `time_range` values like `7d`, `30d`, `12m`, `1y`, or `YYYY-MM-DD..YYYY-MM-DD`.
- non-public sources (`file://`, localhost, private-IP URLs) are filtered before evidence is persisted.

Budget guardrails notes:
- `request.json` budget supports runtime controls:
  - `timeout_sec`
  - `max_tool_calls`
  - `max_external_calls`
  - `on_limit` (`fail` | `need_approval`)
- `task.v2` supports the same controls under `runtime_budget`; `agent-approve` will map them into `request_from_task.json`.
- when `on_limit=need_approval`, execution stops with `task_state.current_state=WAITING_APPROVAL` and can be resumed after budget adjustment.

### Path D: strict real-chain release evidence bundle
```bash
make real-chain-baseline TASK_ID=<base_task_id>
make real-chain-baseline-archive-tgz TASK_ID=<base_task_id>
make real-chain-release-bundle-check TASK_ID=<base_task_id>
```

Lightweight UI smoke check:
```bash
make ui-smoke
```

Launch local UI prototype (requires `flask`):
```bash
pip install flask
make ui-run
```
Open: `http://127.0.0.1:8787`

UI prototype API coverage:
- `GET /api/projects` -> list persisted chat projects
- `POST /api/projects` -> create/update one project session
- `GET /api/projects/<project_id>/export` -> export full project session JSON
- `POST /api/projects/import` -> import session JSON into target project id
- `GET /api/projects/<project_id>/history` -> chat history + attachments
- `POST /api/projects/<project_id>/upload-ref` -> register local file path or upload copy
- `POST /api/chat/send` -> chat orchestration (`intake -> approve -> run`) + step-mode command (`/step ...`)
- `GET /api/tasks` -> recent run list for inspector/compare picker (`limit`, `prefix`)
- `GET /api/experiments` -> experiment-trace list with filters (`limit`, `prefix`, `predictor_id`, `generator_id`, `status`, `execution_mode`)
- `POST /api/run` -> `agent-run-json` (full pipeline)
- `POST /api/run-step` -> `agent-run-step-json` (single operation)
- `POST /api/intake` -> `agent-intake` (task clarification + evidence)
- `POST /api/approve` -> `agent-approve` (task.v2 draft approval)
- `POST /api/resume` -> `agent-resume` (idempotent task resume)
- `POST /api/task/<task_id>/retry-failed-step` -> retry latest failed tool as single-step run
- `GET /api/task/<task_id>/summary` -> artifact/status preview
- `GET /api/task/<task_id>/artifact/<artifact_name>` -> artifact content preview (`plan|execution|tool_state|decision_summary|task_state|web_evidence|experiment_trace`)
- `GET /api/task/<task_id>/timeline` -> step timeline with duration/status/adapter summary (supports `tool`, `status_filter`, `sort`)
- `GET /api/task/<task_id>/compare` -> run-to-run diff vs `other_task_id` (records/failures/adapters/duration/evidence deltas)
- `GET /api/task/<task_id>/artifact-diff` -> key-path diff for one artifact vs `other_task_id` (`artifact=decision_summary|task_state|plan|execution|tool_state|web_evidence`)
- `GET /api/task/<task_id>/validate` -> one-click core artifact validation

## CLI commands
- `run`: run pipeline from config
- `doctor`: check environment/dependencies/GPU tooling
- `llm-connectivity`: check LLM source config and command/backend connectivity
- `smoke`: run minimal deterministic demo pipeline
- `agent-plan`: produce structured `DesignSpec + ToolCall[]` from user request
- `agent-run`: plan + execute tools, save plan/execution/state artifacts
- `agent-plan-json`: build plan from schema-validated request JSON
- `agent-run-json`: plan + execute from schema-validated request JSON
- `agent-intake`: build task.v2 draft + missing info questions + web evidence artifact
- `agent-approve`: validate/approve task.v2 and emit `task.json` + `request_from_task.json` + `plan.md`
- `agent-run-step`: execute one operation from `task.v2` payload
- `agent-run-step-json`: execute one operation from `step_request` payload
- `agent-resume`: idempotent resume from `runs/agent/<task_id>`, skipping already successful prefix steps

Command split:
- `agent-plan` / `agent-run`: natural language request entry
- `agent-plan-json` / `agent-run-json`: strict request-contract entry (`schemas/request.schema.json`)

Planner provider (all agent plan/run commands):
- `--planner-provider rule_based_v1` (default): current deterministic planner
- `--planner-provider llm_v1`: supports two invocation modes
  - command mode (highest priority): `OLED_AGENT_LLM_PLANNER_CMD`
  - built-in backend mode: set `OLED_AGENT_LLM_BACKEND=openai_compat` with backend env vars
  - on command/backend failure or invalid JSON output, auto-fallback to `rule_based_v1` with metadata reason

LLM planner command contract (`OLED_AGENT_LLM_PLANNER_CMD`):
- command reads JSON from stdin: `{"prompt": "...", "request": <request_payload>, "catalog_path": "..."}`
- command writes one JSON object to stdout with:
  - `summary` (string)
  - `design_spec` (object containing `targets`, optional `constraints`, optional `budget`, optional `model_choice`)
  - `tool_calls` (array of `{name, args}`)
- if command and backend are both unset, `llm_v1` falls back to `rule_based_v1` (`planner_provider_reason=llm_provider_not_implemented`)
- normalized LLM plan is validated against:
  - `schemas/request.schema.json` (request payload safety)
  - `schemas/plan.schema.json` (plan payload shape)

Built-in backend mode (`openai_compat`) env vars:
- `OLED_AGENT_LLM_BACKEND=openai_compat`
- `OLED_AGENT_LLM_MODEL=<model_name>`
- `OLED_AGENT_LLM_API_KEY=<api_key>`
- optional `OLED_AGENT_LLM_BASE_URL` (default: `https://api.openai.com/v1`)
- optional `OLED_AGENT_LLM_TIMEOUT_SEC` (default: `60`)
- optional `OLED_AGENT_LLM_BACKEND_MAX_RETRIES` (default: `0`, retryable codes: `408/409/425/429/500/502/503/504`)
- optional `OLED_AGENT_LLM_BACKEND_BACKOFF_SEC` (default: `1`, exponential backoff base seconds)
- optional `OLED_AGENT_LLM_CHAT_COMPLETIONS_PATH` (default: `/chat/completions`, for proxy-specific route)
- optional `OLED_AGENT_LLM_AUTH_HEADER` (default: `Authorization`)
- optional `OLED_AGENT_LLM_AUTH_SCHEME` (default: `Bearer`, set empty string to send raw key)
- optional `OLED_AGENT_LLM_EXTRA_HEADERS_JSON` (JSON object string, e.g. `{"X-Client":"agent4mat"}`)
- optional `OLED_AGENT_LLM_DISABLE_RESPONSE_FORMAT` (`1/true` to skip `response_format` for strict proxies)
- optional `OLED_AGENT_LLM_DEBUG_ERROR` (`1/true` to include redacted backend error detail in fallback metadata; for debugging only)

LLM connectivity probe:
```bash
PYTHONPATH=src python3 -m oled_agent.cli llm-connectivity \
  --workspace-root . \
  --catalog configs/models/catalog.json \
  --json-out runs/llm_connectivity.json
```
This command checks:
- source resolution (`command` vs `backend`)
- command mode: planner command returns JSON object
- backend mode: `openai_compat` config parsing and HTTP reachability

Example for personal proxy:
```bash
export OLED_AGENT_LLM_BACKEND=openai_compat
export OLED_AGENT_LLM_MODEL=gpt-4o-mini
export OLED_AGENT_LLM_API_KEY=sk-xxxx
export OLED_AGENT_LLM_BASE_URL=https://your-proxy.example/api
export OLED_AGENT_LLM_CHAT_COMPLETIONS_PATH=/v1/chat/completions
export OLED_AGENT_LLM_AUTH_HEADER=X-API-Key
export OLED_AGENT_LLM_AUTH_SCHEME=
export OLED_AGENT_LLM_EXTRA_HEADERS_JSON='{"X-Client":"agent4mat"}'
export OLED_AGENT_LLM_DISABLE_RESPONSE_FORMAT=1
```

Tool adapter script hooks (optional, for real train/generate/score integration):
- `OLED_AGENT_TRAIN_CMD`: JSON-in/JSON-out command for `train_predictor`
- `OLED_AGENT_GENERATE_CMD`: JSON-in/JSON-out command for `generate_candidates` (should produce output CSV)
- `OLED_AGENT_SCORE_CMD`: JSON-in/JSON-out command for `score_candidates` (should produce scored CSV)
- command resolution priority:
  1. env override (`OLED_AGENT_*_CMD`)
  2. model catalog adapter command (`configs/models/catalog.json` -> `models[*].params.adapters`)
  3. built-in local fallback logic
- adapter contract + troubleshooting guide: `scripts/adapters/README.md`
- quickstart adapter catalog: `scripts/adapters/quickstart_catalog.json`
- real adapter shell catalog: `scripts/adapters/real_adapters_catalog.json`
- adapter contract validator: `scripts/adapters/validate_adapter_contract.py`
- quickstart chain self-check: `scripts/adapters/check_quickstart_chain.sh`
- `catalog.json` adapter example:
```json
{
  "models": [
    {
      "id": "unimol_lambda_plqy_v1",
      "kind": "predictor",
      "params": {
        "adapters": {
          "train_predictor_cmd": "python3 scripts/adapters/train_predictor_unimol_adapter.py",
          "score_candidates_cmd": "python3 scripts/adapters/score_candidates_unimol_adapter.py"
        }
      }
    },
    {
      "id": "reinvent4_lambda_em_v2",
      "kind": "generator",
      "params": {
        "adapters": {
          "generate_candidates_cmd": "python3 scripts/adapters/generate_candidates_reinvent4_adapter.py"
        }
      }
    }
  ]
}
```
- failure semantics:
  - `score_candidates` adapter failure -> falls back to `local_deterministic_fallback` and records `fallback_error.code=external_score_cmd_failed`
  - `generate_candidates` default bundled REINVENT4 adapter failure -> falls back to local generation and records `fallback_error.code=reinvent4_generate_cmd_failed`
  - explicitly configured `OLED_AGENT_GENERATE_CMD` failures still fail the step directly
- optional timeouts:
  - `OLED_AGENT_TRAIN_TIMEOUT_SEC` (default `3600`)
  - `OLED_AGENT_GENERATE_TIMEOUT_SEC` (default `3600`)
  - `OLED_AGENT_SCORE_TIMEOUT_SEC` (default `3600`)

Real adapter mode controls:
- `OLED_AGENT_UNIMOL_TRAIN_MODE=preflight|smoke|real`
- `OLED_AGENT_UNIMOL_SCORE_MODE=preflight|smoke|real`
- `OLED_AGENT_MINERU_ADAPTER_MODE=preflight|smoke`
- `OLED_AGENT_REINVENT4_ADAPTER_MODE=preflight|smoke|real`
- `OLED_AGENT_MOLSCRIBE_ADAPTER_MODE=preflight|smoke|real`

Built-in mock planner for local/CI checks:
```bash
export OLED_AGENT_LLM_PLANNER_CMD="python3 scripts/mock_llm_planner.py"
export MOCK_LLM_MODE=active  # active|bad_json|bad_tools|bad_model|exit_nonzero
PYTHONPATH=src python3 -m oled_agent.cli agent-plan --workspace-root . --task-id task_llm_mock --request "设计470nm附近且高PLQY分子" --planner-provider llm_v1
```

CI-style llm mode gate:
```bash
python3 scripts/check_llm_planner_modes.py
```
This gate verifies 5 deterministic modes: `active`, `bad_json`, `bad_model`, `bad_tools`, `exit_nonzero`.

Examples:
```bash
PYTHONPATH=src python3 -m oled_agent.cli run --config configs/pipelines/demo.json --workspace-root .
PYTHONPATH=src python3 -m oled_agent.cli doctor --workspace-root . --json-out runs/doctor_report.json
PYTHONPATH=src python3 -m oled_agent.cli smoke --workspace-root .
PYTHONPATH=src python3 -m oled_agent.cli agent-plan --workspace-root . --task-id task_001 --request "设计470nm附近且高PLQY分子"
PYTHONPATH=src python3 -m oled_agent.cli agent-run --workspace-root . --task-id task_001 --request "设计470nm附近且高PLQY分子"
PYTHONPATH=src python3 -m oled_agent.cli agent-plan-json --workspace-root . --catalog configs/models/catalog.json --request-json /path/to/request.json
PYTHONPATH=src python3 -m oled_agent.cli agent-run-json --workspace-root . --catalog configs/models/catalog.json --request-json /path/to/request.json
PYTHONPATH=src python3 -m oled_agent.cli agent-plan --workspace-root . --task-id task_llm --request "设计470nm附近且高PLQY分子" --planner-provider llm_v1
```

MolScribe structured input example (`request.json`):
```json
{
  "task_id": "task_molscribe_input",
  "request_text": "从论文图像提取分子并筛选高PLQY",
  "mode": "fast_screen",
  "targets": [{"property": "plqy", "objective": "maximize", "target_value": 60.0}],
  "budget": {"max_candidates": 20},
  "model_preferences": {
    "predictor_id": "unimol_lambda_plqy_real_v1",
    "generator_id": "molscribe_generator_real_v1"
  },
  "generation_input": {
    "source_image": "/abs/path/to/figure.png"
  }
}
```

MolScribe PDF input example (`request_pdf.json`):
```json
{
  "task_id": "task_molscribe_pdf",
  "request_text": "从论文PDF提取分子并筛选高PLQY",
  "mode": "fast_screen",
  "targets": [{"property": "plqy", "objective": "maximize", "target_value": 60.0}],
  "budget": {"max_candidates": 20},
  "model_preferences": {
    "predictor_id": "unimol_lambda_plqy_real_v1",
    "generator_id": "molscribe_generator_real_v1"
  },
  "generation_input": {
    "source_pdf": "/abs/path/to/paper.pdf"
  }
}
```

Optional PDF pre-extract hook:
```bash
export OLED_AGENT_MOLSCRIBE_PDF_EXTRACT_CMD="python3 your_pdf_extract_script.py"
PYTHONPATH=src python3 -m oled_agent.cli agent-run-json \
  --workspace-root . \
  --catalog configs/models/catalog.json \
  --request-json /abs/path/to/request_pdf.json \
  --planner-provider llm_v1
```

Canonical request templates:
- `configs/request_templates/request_molscribe_image.json`
- `configs/request_templates/request_molscribe_pdf.json`

Validate templates against request schema:
```bash
make request-templates-validate
```

Run input smoke acceptance (MolScribe image/pdf):
```bash
make input-smoke
```

## Agent layer
- `src/oled_agent/agent/specs.py`: DesignSpec / ToolCall contracts
- `src/oled_agent/agent/model_catalog.py`: predictor/generator catalog + validation
- `src/oled_agent/agent/planner.py`: rule-based planner (LLM-replaceable)
- `src/oled_agent/agent/tools.py`: tool registry + adapters + fallbacks
- `src/oled_agent/agent/executor.py`: tool execution loop and step records
- `src/oled_agent/agent/session.py`: plan-only and plan+execute orchestration

## Skills
- Router: `skills/SKILL.md`
- Tool-focused skill docs:
  - `skills/unimol/SKILL.md`
  - `skills/reinvent4/SKILL.md`
  - `skills/mineru/SKILL.md`
  - `skills/molscribe/SKILL.md`

## Assistant conventions
- Repository-scoped assistant behavior policy:
  - `SOUL.md`

## Repository layout
- `src/oled_agent/cli.py`: CLI entrypoint
- `src/oled_agent/runner.py`: stage orchestration + manifest
- `src/oled_agent/diagnostics.py`: doctor checks
- `src/oled_agent/smoke.py`: smoke-run helper
- `src/oled_agent/stages/`: executable stage implementations
- `src/oled_agent/policy/`: intake/lifecycle/gate logic
- `configs/pipelines/`: pipeline config JSONs
- `configs/models/`: model catalog for user-selectable predictor/generator
- `runs/`: generated run outputs and manifests
- `docs/`: deployment and operations notes
  - deployment guide: `docs/deploy.md`
  - troubleshooting: `docs/troubleshooting.md`
  - CI guide: `docs/ci.md`

## CI gate
- workflow: `.github/workflows/agent4mat-ci.yml`
- checks: lockfile validation + regression tests + smoke + `agent-run` E2E
- optional external-chain acceptance:
  - only runs on `workflow_dispatch` with input `run_external_acceptance=true`
  - default push/pull_request CI does not run this job
- optional real-chain minimal acceptance:
  - only runs on `workflow_dispatch` with input `run_real_chain_acceptance=true`
  - runs `make real-chain-acceptance` and uploads run artifacts
- optional UI acceptance gates (manual):
  - `run_ui_freeze_acceptance=true`: run `ui-freeze-acceptance`
  - `run_ui_audit_acceptance=true`: run `ui-audit-acceptance`
  - `run_ui_release_readiness=true`: run `ui-release-readiness`
  - `run_ui_acceptance_bundle=true`: run the full UI bundle (`freeze + audit + release-readiness`) and publish a bundle summary
  - each UI job uploads `runs/ci/*` artifacts and publishes a step summary in Actions
- production non-stub acceptance:
  - see `docs/real_chain_acceptance_real.md` and run manually in real runtime environment

## Agent artifacts
`agent-run` writes:
- `plan.json`
- `execution.json`
- `tool_state.json`
- `task_state.json` (explicit task-state-machine history)
- `decision_summary.json` (machine-readable fallback decision fields)

Additional mirrored layout (for external-user readability):
- `logging/<task_id>-<timestamp>/`
  - `task.json`
  - `plan.md`
  - `execution.log`
  - `data_report.json`
  - `model_report.json`
  - `filtering_report.json`
  - `plan.json`, `execution.json`, `tool_state.json`, `decision_summary.json`, `task_state.json`
- `result/<task_id>-<timestamp>/`
  - `target_structures.csv` (from scored/candidate artifacts when available)
  - `metadata.json`
  - optional `report.md`

Decision summary validator:
- `python3 scripts/validate_decision_summary.py <path>`
- validator reuses `schemas/decision_summary.schema.json` via `request_contract` module.

Task-state validator:
- `python3 scripts/validate_task_state.py <path>`
- validator reuses `schemas/task_state.schema.json` via `request_contract` module.

Structured report validators:
- `python3 scripts/validate_data_report.py <path>`
- `python3 scripts/validate_model_report.py <path>`
- `python3 scripts/validate_filtering_report.py <path>`
- validators reuse `schemas/data_report.schema.json`, `schemas/model_report.schema.json`, and `schemas/filtering_report.schema.json` via `request_contract`.

## External scorer preflight and acceptance
- preflight command:
  - `PYTHONPATH=src python3 -m oled_agent.cli external-preflight --workspace-root .`
- extended connectivity debug (machine-readable summary):
  - `PYTHONPATH=src python3 -m oled_agent.cli external-connectivity-debug --workspace-root . --json-out runs/external_debug.json`
- one-click env + preflight check:
  - `./scripts/check_external_env.sh`
- external-chain acceptance script (expects real external scorer path):
  - `./scripts/run_external_chain_acceptance.sh <task_id>`
- external-chain acceptance with auto debug artifacts:
  - `./scripts/run_external_chain_acceptance_with_debug.sh <task_id>`

Required env vars for remote Uni-Mol scorer:
- `OLED_AGENT_USE_EXTERNAL_SCORER=1`
- `UNIMOL_REMOTE_HOST`
- `UNIMOL_REMOTE_PY`
- `UNIMOL_REMOTE_TMP_BASE`

Optional (legacy fallback defaults in scorer script):
- `ALLOW_DEFAULT_UNIMOL_REMOTE=1`

Real generator adapters:
- REINVENT4 adapter:
  - command: `scripts/adapters/generate_candidates_reinvent4_adapter.py`
  - runtime knobs:
    - `OLED_AGENT_REINVENT4_SOURCE_CSV` (optional explicit source sampling CSV)
    - `OLED_AGENT_REINVENT4_PIPELINE_SCRIPT` (optional override; default uses workspace pipeline script)
    - `OLED_AGENT_REINVENT4_RANKREADY_CSV` (optional explicit rankready path)
    - `OLED_AGENT_REINVENT4_ADAPTER_TIMEOUT_SEC`
- MolScribe adapter:
  - command: `scripts/adapters/generate_candidates_molscribe_adapter.py`
  - input comes from request payload/constraints fields such as `source_image`, `source_pdf`, `image_paths`
  - runtime knobs:
    - command mode: `OLED_AGENT_MOLSCRIBE_CMD`
    - native mode: `OLED_AGENT_MOLSCRIBE_CHECKPOINT` (+ optional `OLED_AGENT_MOLSCRIBE_DEVICE`)
    - optional PDF pre-extract hook: `OLED_AGENT_MOLSCRIBE_PDF_EXTRACT_CMD`
    - `OLED_AGENT_MOLSCRIBE_ADAPTER_TIMEOUT_SEC`

Template:
- `scripts/env_external.example`
- `.env.example` (LLM + external adapter dual-scenario template)


### LLM chain smoke test (mock planner)

Use this to validate the LLM integration chain without external model credentials:

```bash
make llm-smoke
```

This executes `scripts/check_llm_planner_modes.py` with `scripts/mock_llm_planner.py` and verifies deterministic `llm_v1` mode behavior (active + fallback modes).

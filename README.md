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
- `make quickstart`
- `make doctor`
- `make llm-connectivity`
- `make adapter-validate`
- `make real-adapter-validate`
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
- `make real-adapter-validate` is contract smoke with stub env placeholders (not a proof of real remote runtime availability).

## CLI commands
- `run`: run pipeline from config
- `doctor`: check environment/dependencies/GPU tooling
- `llm-connectivity`: check LLM source config and command/backend connectivity
- `smoke`: run minimal deterministic demo pipeline
- `agent-plan`: produce structured `DesignSpec + ToolCall[]` from user request
- `agent-run`: plan + execute tools, save plan/execution/state artifacts
- `agent-plan-json`: build plan from schema-validated request JSON
- `agent-run-json`: plan + execute from schema-validated request JSON

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
          "train_predictor_cmd": "python3 scripts/train_predictor_adapter.py",
          "score_candidates_cmd": "python3 scripts/score_candidates_adapter.py"
        }
      }
    },
    {
      "id": "reinvent4_lambda_em_v2",
      "kind": "generator",
      "params": {
        "adapters": {
          "generate_candidates_cmd": "python3 scripts/generate_candidates_adapter.py"
        }
      }
    }
  ]
}
```
- failure semantics:
  - `score_candidates` adapter failure -> falls back to `local_deterministic_fallback` and records `fallback_error.code=external_score_cmd_failed`
  - `generate_candidates` adapter failure -> step fails directly (no local generation fallback in adapter branch)
- optional timeouts:
  - `OLED_AGENT_TRAIN_TIMEOUT_SEC` (default `3600`)
  - `OLED_AGENT_GENERATE_TIMEOUT_SEC` (default `3600`)
  - `OLED_AGENT_SCORE_TIMEOUT_SEC` (default `3600`)

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

## Agent artifacts
`agent-run` writes:
- `plan.json`
- `execution.json`
- `tool_state.json`
- `decision_summary.json` (machine-readable fallback decision fields)

Decision summary validator:
- `python3 scripts/validate_decision_summary.py <path>`
- validator reuses `schemas/decision_summary.schema.json` via `request_contract` module.

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

Template:
- `scripts/env_external.example`
- `.env.example` (LLM + external adapter dual-scenario template)


### LLM chain smoke test (mock planner)

Use this to validate the LLM integration chain without external model credentials:

```bash
make llm-smoke
```

This executes `scripts/check_llm_planner_modes.py` with `scripts/mock_llm_planner.py` and verifies deterministic `llm_v1` mode behavior (active + fallback modes).

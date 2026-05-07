# Deployment Guide

This guide is the external-user entrypoint for deploying `oled-agent` as a reusable repository.

## 1) Minimum support matrix
- Python: `3.10+` (CI baseline)
- OS: Linux/macOS
- Profiles:
  - CPU profile: deterministic baseline + mock/contract checks
  - GPU profile: optional heavy runtime (Uni-Mol/MinerU adapters)

## 2) Quick deployment (recommended)
```bash
git clone <repo-url>
cd oled-agent
python3 -m venv .venv
source .venv/bin/activate
./scripts/install_profile.sh cpu
make release-check
```

Expected:
- adapter contract checks pass
- quickstart chain passes
- llm smoke passes
- doctor returns no `fail`

## 3) Environment template
- Copy `.env.example` to your deployment env file or export equivalent variables.
- Two supported scenarios:
  - Scenario A: LLM planner route (command mode or openai_compat backend)
  - Scenario B: external adapter runtime (Uni-Mol/MinerU/remote scorer)

## 4) LLM planner deployment options

### Option A: command mode (highest priority)
```bash
export OLED_AGENT_LLM_PLANNER_CMD="python3 scripts/mock_llm_planner.py"
export MOCK_LLM_MODE=active
```

### Option B: openai_compat backend
```bash
export OLED_AGENT_LLM_BACKEND=openai_compat
export OLED_AGENT_LLM_MODEL=<model-id>
export OLED_AGENT_LLM_API_KEY=<key>
export OLED_AGENT_LLM_BASE_URL=<base-url>
```

Verify:
```bash
make llm-smoke
```

## 5) External adapter deployment (optional)

Enable remote scorer chain only when infra is ready:
```bash
export OLED_AGENT_USE_EXTERNAL_SCORER=1
export UNIMOL_REMOTE_HOST=<user@host>
export UNIMOL_REMOTE_PY=<remote-python-path>
export UNIMOL_REMOTE_TMP_BASE=<remote-tmp-dir>
```

Check:
```bash
PYTHONPATH=src python3 -m oled_agent.cli external-preflight --workspace-root .
PYTHONPATH=src python3 -m oled_agent.cli external-connectivity-debug --workspace-root . --json-out runs/external_debug.json
```

## 6) Real adapter shell validation

Contract-only smoke validation:
```bash
make real-adapter-validate
```

Note:
- `make real-adapter-validate` verifies adapter contract shape and deterministic smoke outputs only.
- CI values such as `UNIMOL_REMOTE_HOST=stub_host` and `UNIMOL_REMOTE_PY=stub_py` are placeholders, not real remote runtime checks.

Modes:
- `OLED_AGENT_UNIMOL_TRAIN_MODE=preflight|smoke|real`
- `OLED_AGENT_UNIMOL_SCORE_MODE=preflight|smoke|real`
- `OLED_AGENT_MINERU_ADAPTER_MODE=preflight|smoke`

## 7) CI acceptance matrix
- `acceptance-cpu-mock`: `make release-check`
- `acceptance-llm-mock`: `make llm-smoke`
- `acceptance external-adapter (optional)`: manual dispatch only

## 8) Release checklist
1. `make test-regressions`
2. `make release-check`
3. `make real-adapter-validate`
4. confirm `.env.example` matches current runtime knobs

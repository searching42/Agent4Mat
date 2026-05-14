# UI Prototype (P2)

A minimal local web UI prototype is provided at `ui/app.py`.

## Start
```bash
pip install flask
PYTHONPATH=src python3 ui/app.py
```
Open: `http://127.0.0.1:8787`

## Scope
- full-pipeline run panel:
  - request JSON + planner provider
  - invokes `agent-run-json`
- single-step run panel:
  - step request JSON
  - invokes `agent-run-step-json`
- intake panel:
  - task id + request text + web topk
  - invokes `agent-intake`
- approve panel:
  - task draft json path
  - invokes `agent-approve`
- resume panel:
  - task id + planner
  - invokes `agent-resume`
- task inspector panel:
  - reads `runs/agent/<task_id>` artifacts
  - previews `execution/task_state/decision_summary/web_evidence`

## API endpoints
- `GET /api/health`
- `POST /api/run`
- `POST /api/run-step`
- `POST /api/intake`
- `POST /api/approve`
- `POST /api/resume`
- `GET /api/task/<task_id>/summary`

## Non-goals
- no auth
- no async queue
- no long-running job scheduler
- no replacement of CLI runtime contracts

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
  - previews `execution/task_state/decision_summary/web_evidence/experiment_trace`
  - summary endpoint includes `experiment_trace_preview`
  - supports one-click artifact preview, timeline, and core validation
  - timeline supports failed-step highlight, tool filter, and duration sort
  - supports recent-task picker (quick fill for inspect/compare task ids)
  - supports run-to-run compare (`task_id` vs `other_task_id`) with key deltas (records/failures/adapters/duration/evidence)
  - supports artifact key-path diff (`decision_summary/task_state/plan/...`)
- experiments quick panel:
  - list `experiment_trace` records with filters (`prefix/predictor/generator/status/execution_mode`)

## API endpoints
- `GET /api/health`
- `GET /api/tasks`
  - query params: `limit`, `prefix`
- `GET /api/experiments`
  - query params: `limit`, `prefix`, `predictor_id`, `generator_id`, `status`, `execution_mode`
- `POST /api/run`
- `POST /api/run-step`
- `POST /api/intake`
- `POST /api/approve`
- `POST /api/resume`
- `GET /api/task/<task_id>/summary`
- `GET /api/task/<task_id>/artifact/<artifact_name>`
- `GET /api/task/<task_id>/timeline`
  - query params: `tool`, `status_filter=all|failed|success`, `sort=original|duration_desc|duration_asc|name_asc`
- `GET /api/task/<task_id>/compare`
  - query params: `other_task_id`
- `GET /api/task/<task_id>/artifact-diff`
  - query params: `other_task_id`, `artifact`
- `GET /api/task/<task_id>/validate`

## Non-goals
- no auth
- no async queue
- no long-running job scheduler
- no replacement of CLI runtime contracts

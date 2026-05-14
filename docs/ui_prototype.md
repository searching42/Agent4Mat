# UI Prototype (P2)

A local chat-first UI is provided at `ui/app.py`.

## Start
```bash
pip install flask
PYTHONPATH=src python3 ui/app.py
```
Open: `http://127.0.0.1:8787`

## Scope
- chat-first interaction:
  - user sends natural-language messages
  - backend orchestrates `agent-intake -> agent-approve -> agent-run-json`
  - if info is missing, assistant returns structured clarification questions
- project/session memory:
  - each project persists independent chat history and runtime pointers
  - session file: `runs/ui_sessions/projects/<project_id>.json`
- file input entry:
  - register local absolute path as attachment reference
  - optional browser upload copy to `runs/ui_sessions/uploads/<project_id>/`
- output panel:
  - runtime summary (status / record_count / duration)
  - artifact preview and timeline / validation shortcuts
- compatibility:
  - legacy run/step/intake/approve/resume/task-inspector APIs remain available

## API endpoints
- project/chat/session:
  - `GET /api/projects`
  - `POST /api/projects`
  - `GET /api/projects/<project_id>/history`
  - `POST /api/projects/<project_id>/upload-ref`
  - `POST /api/chat/send`
- existing execution and inspector APIs:
  - `GET /api/health`
  - `GET /api/tasks`
  - `GET /api/experiments`
  - `POST /api/run`
  - `POST /api/run-step`
  - `POST /api/intake`
  - `POST /api/approve`
  - `POST /api/resume`
  - `GET /api/task/<task_id>/summary`
  - `GET /api/task/<task_id>/artifact/<artifact_name>`
  - `GET /api/task/<task_id>/timeline`
  - `GET /api/task/<task_id>/compare`
  - `GET /api/task/<task_id>/artifact-diff`
  - `GET /api/task/<task_id>/validate`

## Non-goals
- no auth
- no async queue
- no long-running job scheduler
- no replacement of CLI runtime contracts

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
  - center-first chat workspace layout: chat is primary canvas, advanced controls are folded into side drawers
  - backend orchestrates `agent-intake -> agent-approve -> agent-run-json`
  - if info is missing, assistant returns structured clarification questions
  - `need_user_input` 时会弹出缺失字段表单卡片，可直接填写并回传 patch
  - composer supports `Ctrl/Cmd+Enter` quick send and keeps per-project local draft
  - recent prompts are stored per project (local browser storage) and shown as reusable chips
  - supports step mode in chat via `/step <operation> {args_json}` or JSON (`{"operation":"clean_dataset","args":{...}}`)
  - includes a built-in step panel (operation dropdown + args JSON) that sends step requests into chat
- project/session memory:
  - each project persists independent chat history and runtime pointers
  - workspace URL carries `?project_id=...` and restores project context on load/back-forward navigation
  - supports `Open in New Window` and `Copy Workspace Link` for project-isolated windows
  - left drawer includes `Workspace Sessions` board with quick `Open` and `Resume` actions per project
  - session board shows `latest_failed_step` and supports `Retry Failed` one-click action per project
  - session board shows `recent_duration` and `success_ratio` with a mini progress bar
  - session board shows `failed_error` snippet, plus quick `Timeline` view and `Copy Task ID`
  - session file: `runs/ui_sessions/projects/<project_id>.json`
  - project picker/meta now includes `runtime_health` snapshot from current task execution (`status`, success/failed steps, latest failed step)
- file input entry:
  - register local absolute path as attachment reference
  - optional browser upload copy to `runs/ui_sessions/uploads/<project_id>/`
- output panel:
  - runtime summary (status / record_count / duration)
  - task stage summary (`task_state.current_stage/status`) + latest failed step hint
  - progress bar from timeline summary (`success_steps / total_steps`)
  - one-click retry current task via `agent-resume`
  - one-click retry latest failed step via `agent-run-step-json` (supports args override, target failed_tool_name, and dry-run preview)
  - `Load Suggested Retry Args` can auto-fill retry args from dry-run replay suggestion
  - task compare / artifact diff shortcuts for two-run diffing
  - recent stage events (intake/approve/run/step)
  - artifact preview and timeline / validation shortcuts
- web search interaction:
  - explicit `Web Search` action button in composer injects a search-first prompt template
- chat transcript:
  - stores per-turn event trace messages (`event_trace`) to show stage timeline inline
  - includes grouped timeline board (Running / Completed / Failed) from `/api/task/<task_id>/timeline`
  - timeline board supports scope switch:
    - `current_task`: current task grouped timeline
    - `recent_tasks`: aggregate grouped timeline across recent N tasks
  - failed-group items support inline Retry action (reuses row args by default, supports override args JSON)
- step panel:
  - operation dropdown includes `Load Args Template` to auto-fill operation-specific args skeleton
- compatibility:
  - legacy run/step/intake/approve/resume/task-inspector APIs remain available

## API endpoints
- project/chat/session:
  - `GET /api/projects`
  - `POST /api/projects`
  - `GET /api/projects/<project_id>/export`
  - `POST /api/projects/import`
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
  - `POST /api/task/<task_id>/retry-failed-step`
  - `GET /api/task/<task_id>/summary`
  - `GET /api/task/<task_id>/artifact/<artifact_name>`
  - `GET /api/task/<task_id>/timeline`
  - `GET /api/timeline-groups` (`scope=recent_tasks&limit=N`)
  - `GET /api/task/<task_id>/compare`
  - `GET /api/task/<task_id>/artifact-diff`
  - `GET /api/task/<task_id>/validate`

## Non-goals
- no auth
- no async queue
- no long-running job scheduler
- no replacement of CLI runtime contracts

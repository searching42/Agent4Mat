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
  - session board supports filter/sort controls (`project/task` text filter, health filter, sort mode)
  - session board provides quick presets (`Failed Only`, `Priority First`, `Reset`) and persists controls in browser local storage
  - session board includes `Open Top Priority` action and aggregate summary (`total/failed/success/none/avg_success_ratio`)
  - supports optional session board auto-refresh (`10/20/30/60s`)
  - supports project pin/unpin and `Pinned Only` focus mode (persisted in local storage)
  - supports `Open Next Failed` for failed-project triage
  - supports `Status Groups` mode to render failed/success/none sections
  - supports batch actions on current filtered set: `Batch Summary`, `Batch Validate`
  - supports `Batch Retry Failed` to replay each row's latest failed step in limited scope
  - supports `Batch Export JSON` to emit latest batch payload for copy/save
  - supports `Batch Limit` (1-20) to cap batch action scope
  - supports one-click health count filters (`Failed Count` / `Success Count` / `None Count`) with live counters
  - batch actions are persisted to `runs/ui_sessions/exports/<project_id>/` via API
  - supports batch history panel (`Load Batch History`) with latest export summary
  - batch history supports filter by action/status and paging (`limit/offset`, prev/next)
  - batch history includes aggregate metrics bar (`pass/partial/fail`, replay ok/fail/skipped/dry, avg elapsed)
  - batch history list items support one-click `Use ID` to fill `batch_export_id`
  - batch history list items support one-click `Use As Compare` to fill `batch_export_compare_id`
  - supports `Replay Latest Batch` to rerun latest exported batch action (`batch_summary` / `batch_validate` / `batch_retry_failed`)
  - supports `Replay Failed Latest` / `Replay Failed By ID` (`failed_only=true`) to only replay previously failed rows
  - supports export-id actions: `View Export By ID`, `Replay Export By ID`, `Delete Export By ID`
  - supports export-id compare (`Compare Export IDs`) with JSON path-level diff summary
  - supports export-id download (`Download Export JSON` / `Download Export CSV`)
  - replay supports governance controls: `dry_run`, `failed_only`, `retry_max`, `retry_backoff_ms`, `max_concurrency`
  - replay preset shortcuts: `Preset Safe`, `Preset Fast`, `Preset DryRun`, and project-level `Save Replay Defaults`
  - replay results persist `replay_metrics` (`ok/fail/skipped/dry_run`, attempts, elapsed_ms, failed_task_ids)
  - session card provides quick `Summary` and `Validate` actions for current task
  - runtime health now includes `success_ratio` and `recent_duration_ms` for smarter priority sorting
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
  - `POST /api/projects/<project_id>/batch-export`
  - `GET /api/projects/<project_id>/batch-exports` (`limit`, `offset`, `action`, `status=pass|partial|fail`)
  - `GET /api/projects/<project_id>/batch-exports/compare` (`primary_export_id`, `other_export_id`)
  - `POST /api/projects/<project_id>/batch-exports/replay-latest`
    - body supports `{ "options": { "dry_run": bool, "failed_only": bool, "retry_max": 0..3, "retry_backoff_ms": 0..5000, "max_concurrency": 1..8 } }`
  - `GET /api/projects/<project_id>/batch-exports/<export_id>`
  - `GET /api/projects/<project_id>/batch-exports/<export_id>/download` (`format=json|csv`)
  - `POST /api/projects/<project_id>/batch-exports/<export_id>/replay`
    - body supports same replay `options`
  - `DELETE /api/projects/<project_id>/batch-exports/<export_id>`
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

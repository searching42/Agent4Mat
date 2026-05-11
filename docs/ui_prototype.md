# UI Prototype (P2)

A minimal local web UI prototype is provided at:
- `ui/app.py`

## Start
```bash
PYTHONPATH=src python3 ui/app.py
```
Open: `http://127.0.0.1:8787`

## Scope
- request JSON input
- planner provider selection
- invoke `agent-run-json`
- show structured result JSON

## Non-goals
- no auth
- no async queue
- no long-running job scheduler
- no replacement of CLI runtime contracts

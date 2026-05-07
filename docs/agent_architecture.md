# Agent architecture (LLM-style tool-calling)

This repository now supports a minimal two-layer architecture:

1. Agent layer (`src/oled_agent/agent/`)
- `planner.py`: converts user request -> `DesignSpec` + tool call plan
- `tools.py`: typed tool registry and executable adapters
- `executor.py`: executes tool calls with step records
- `session.py`: plan-only or plan+execute orchestration

2. Workflow layer (`src/oled_agent/runner.py`, `src/oled_agent/stages/`)
- deterministic stage execution
- machine-readable manifest output

## CLI
- `agent-plan`: build structured plan only
- `agent-run`: build plan and execute tools

## Current status
- Planner is rule-based v1 (placeholder for LLM planner swap-in)
- Toolchain includes stubs for train/generate/score and a real local filter+report path
- Model selection is validated via `configs/models/catalog.json`

## Next integration points
1. Replace rule planner with LLM function-calling planner while preserving `DesignSpec` contract.
2. Replace `generate_candidates` / `score_candidates` stubs with adapter-backed implementations.
3. Add async job queue for long-running training/generation jobs.

# LLM integration plan (replace rule planner)

Current planner: `src/oled_agent/agent/planner.py` (rule_based_v1).

Goal: replace planner with LLM function-calling while preserving the same contracts:
- `DesignSpec`
- `ToolCall[]`
- `AgentPlan`

## 1) Keep stable contracts
Do not let LLM output free text only. LLM must return JSON matching:
- task-level spec (`DesignSpec`)
- executable tool call list (`ToolCall[]`)

## 2) Suggested LLM function schema
- `propose_design_spec(user_request, model_catalog, dataset_catalog) -> DesignSpec`
- `propose_tool_calls(design_spec) -> ToolCall[]`

## 3) Validation gate before execution
Before running tools:
1. validate model ids against catalog
2. validate objective constraints and weights
3. validate budget fields
4. reject unsafe/unsupported requests

## 4) Execution loop
- planner output -> validated plan
- execute tools in order
- collect step records (`execution.json`)
- if step fails, stop and return structured error

## 5) Human-in-the-loop mode
Add optional approval checkpoint:
- `agent-plan` generates plan
- user confirms/edits model choices
- `agent-run` executes approved plan

## 6) Minimal implementation path
1. Add `llm_planner.py` with same return type as rule planner
2. Add `--planner rule|llm` flag on `agent-plan/agent-run`
3. Keep current `executor.py` unchanged

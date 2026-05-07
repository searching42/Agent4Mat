from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from oled_agent.agent.executor import execute_plan, save_execution_result
from oled_agent.agent.planner import (
    DEFAULT_PLANNER_PROVIDER,
    build_plan,
    build_plan_from_request_payload,
)
from oled_agent.agent.request_contract import validate_request_payload
from oled_agent.agent.tools import ToolContext


DEFAULT_CATALOG = Path("configs/models/catalog.json")
DEFAULT_AGENT_OUT = Path("runs/agent")
DECISION_SUMMARY_SCHEMA_VERSION = "1.0.0"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _build_decision_summary(
    *,
    plan: Dict[str, Any],
    execution: Dict[str, Any],
    tool_state: Dict[str, Any],
) -> Dict[str, Any]:
    records = execution.get("records", []) if isinstance(execution, dict) else []
    score_record: Dict[str, Any] = {}
    for rec in records:
        if isinstance(rec, dict) and rec.get("name") == "score_candidates":
            score_record = rec
            break

    score_result = score_record.get("result", {}) if isinstance(score_record, dict) else {}
    adapter = score_result.get("adapter", "")
    fallback_error = score_result.get("fallback_error", {}) if isinstance(score_result, dict) else {}
    used_fallback = adapter == "local_deterministic_fallback"

    return {
        "schema_version": DECISION_SUMMARY_SCHEMA_VERSION,
        "generated_at": _now_iso(),
        "task_id": plan.get("design_spec", {}).get("task_id", ""),
        "status": execution.get("status", ""),
        "model_choice": plan.get("design_spec", {}).get("model_choice", {}),
        "score_step": {
            "adapter": adapter,
            "used_fallback": used_fallback,
            "fallback_reason": score_result.get("fallback_reason", ""),
            "fallback_code": fallback_error.get("code", ""),
            "fallback_retryable": bool(fallback_error.get("retryable", False)),
            "fallback_details": fallback_error.get("details", {}),
        },
        "artifacts": {
            "candidate_csv": tool_state.get("candidate_csv", ""),
            "scored_csv": tool_state.get("scored_csv", ""),
            "final_output": tool_state.get("final_output", ""),
        },
    }


def plan_request(
    *,
    workspace_root: Path,
    user_request: str,
    task_id: str,
    predictor_id: str = "",
    generator_id: str = "",
    mode: str = "fast_screen",
    planner_provider: str = DEFAULT_PLANNER_PROVIDER,
    catalog_path: Optional[Path] = None,
) -> Dict[str, Any]:
    catalog = (catalog_path or (workspace_root / DEFAULT_CATALOG)).resolve()
    plan = build_plan(
        user_request=user_request,
        task_id=task_id,
        catalog_path=catalog,
        predictor_id=predictor_id,
        generator_id=generator_id,
        mode=mode,
        planner_provider=planner_provider,
    )

    return plan.to_dict()


def plan_request_from_payload(
    *,
    workspace_root: Path,
    request_payload: Dict[str, Any],
    planner_provider: str = DEFAULT_PLANNER_PROVIDER,
    catalog_path: Optional[Path] = None,
) -> Dict[str, Any]:
    validate_request_payload(payload=request_payload, workspace_root=workspace_root)
    catalog = (catalog_path or (workspace_root / DEFAULT_CATALOG)).resolve()
    plan = build_plan_from_request_payload(
        request_payload=request_payload,
        catalog_path=catalog,
        planner_provider=planner_provider,
    )
    return plan.to_dict()


def execute_request(
    *,
    workspace_root: Path,
    user_request: str,
    task_id: str,
    predictor_id: str = "",
    generator_id: str = "",
    mode: str = "fast_screen",
    planner_provider: str = DEFAULT_PLANNER_PROVIDER,
    catalog_path: Optional[Path] = None,
) -> Dict[str, Any]:
    catalog = (catalog_path or (workspace_root / DEFAULT_CATALOG)).resolve()

    plan = build_plan(
        user_request=user_request,
        task_id=task_id,
        catalog_path=catalog,
        predictor_id=predictor_id,
        generator_id=generator_id,
        mode=mode,
        planner_provider=planner_provider,
    )

    plan_dict = plan.to_dict()

    tool_ctx = ToolContext(
        workspace_root=workspace_root.resolve(),
        catalog_path=catalog,
        task_id=task_id,
        state={},
    )
    result = execute_plan(plan, tool_ctx)
    result_dict = result.to_dict()

    out_dir = (workspace_root / DEFAULT_AGENT_OUT).resolve() / task_id
    out_dir.mkdir(parents=True, exist_ok=True)

    plan_path = out_dir / "plan.json"
    plan_path.write_text(json.dumps(plan_dict, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    result_path = out_dir / "execution.json"
    save_execution_result(result, result_path)

    state_path = out_dir / "tool_state.json"
    tool_state = dict(tool_ctx.state)
    state_path.write_text(json.dumps(tool_state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    decision_summary = _build_decision_summary(
        plan=plan_dict,
        execution=result_dict,
        tool_state=tool_state,
    )
    decision_path = out_dir / "decision_summary.json"
    decision_path.write_text(json.dumps(decision_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return {
        "task_id": task_id,
        "status": result.status,
        "plan_path": str(plan_path),
        "execution_path": str(result_path),
        "tool_state_path": str(state_path),
        "decision_summary_path": str(decision_path),
    }


def execute_request_from_payload(
    *,
    workspace_root: Path,
    request_payload: Dict[str, Any],
    planner_provider: str = DEFAULT_PLANNER_PROVIDER,
    catalog_path: Optional[Path] = None,
) -> Dict[str, Any]:
    validate_request_payload(payload=request_payload, workspace_root=workspace_root)
    catalog = (catalog_path or (workspace_root / DEFAULT_CATALOG)).resolve()
    plan = build_plan_from_request_payload(
        request_payload=request_payload,
        catalog_path=catalog,
        planner_provider=planner_provider,
    )
    plan_dict = plan.to_dict()
    task_id = str(plan.design_spec.task_id)

    tool_ctx = ToolContext(
        workspace_root=workspace_root.resolve(),
        catalog_path=catalog,
        task_id=task_id,
        state={},
    )
    result = execute_plan(plan, tool_ctx)
    result_dict = result.to_dict()

    out_dir = (workspace_root / DEFAULT_AGENT_OUT).resolve() / task_id
    out_dir.mkdir(parents=True, exist_ok=True)

    plan_path = out_dir / "plan.json"
    plan_path.write_text(json.dumps(plan_dict, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    result_path = out_dir / "execution.json"
    save_execution_result(result, result_path)

    state_path = out_dir / "tool_state.json"
    tool_state = dict(tool_ctx.state)
    state_path.write_text(json.dumps(tool_state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    decision_summary = _build_decision_summary(
        plan=plan_dict,
        execution=result_dict,
        tool_state=tool_state,
    )
    decision_path = out_dir / "decision_summary.json"
    decision_path.write_text(json.dumps(decision_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    request_path = out_dir / "request.json"
    request_path.write_text(json.dumps(request_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return {
        "task_id": task_id,
        "status": result.status,
        "plan_path": str(plan_path),
        "execution_path": str(result_path),
        "tool_state_path": str(state_path),
        "decision_summary_path": str(decision_path),
        "request_path": str(request_path),
    }

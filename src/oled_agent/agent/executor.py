from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from oled_agent.agent.specs import AgentPlan
from oled_agent.agent.tools import ToolContext, execute_tool


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


_LOCAL_ONLY_ADAPTERS = {
    "stub_generator",
    "dataset_stub_retrieval",
    "train_data_stub_builder",
    "local_cleaning_v1",
    "template_generate_cmd",
    "template_score_cmd",
    "template_train_cmd",
    "local_deterministic_fallback",
    "explicit_input_csv",
    "reuse_latest_reinvent_artifact",
}
_EXTERNAL_TOOL_CANDIDATES = {"search_web_evidence", "train_predictor", "generate_candidates", "score_candidates"}


def _is_external_tool_call(name: str, result: Optional[Dict[str, Any]]) -> bool:
    tool_name = str(name or "").strip()
    if tool_name == "search_web_evidence":
        return True
    if tool_name not in _EXTERNAL_TOOL_CANDIDATES:
        return False
    payload = result if isinstance(result, dict) else {}
    adapter = str(payload.get("adapter") or "").strip()
    if not adapter:
        return True
    return adapter not in _LOCAL_ONLY_ADAPTERS


def _budget_limit_int(value: Any, *, minimum: int) -> Optional[int]:
    if not isinstance(value, int):
        return None
    out = int(value)
    if out < minimum:
        return None
    return out


@dataclass
class ToolExecutionRecord:
    name: str
    args: Dict[str, Any]
    started_at: str
    ended_at: str
    status: str
    result: Dict[str, Any] = field(default_factory=dict)
    error: str = ""


@dataclass
class AgentExecutionResult:
    task_id: str
    status: str
    started_at: str
    ended_at: str
    records: List[ToolExecutionRecord]
    budget_control: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "status": self.status,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "records": [asdict(r) for r in self.records],
            "budget_control": self.budget_control if isinstance(self.budget_control, dict) else {},
        }


def execute_plan(plan: AgentPlan, ctx: ToolContext) -> AgentExecutionResult:
    return execute_plan_with_resume(plan, ctx, resume_records=None, resume_from_index=0)


def _record_from_dict(item: Dict[str, Any]) -> ToolExecutionRecord:
    return ToolExecutionRecord(
        name=str(item.get("name") or ""),
        args=item.get("args") if isinstance(item.get("args"), dict) else {},
        started_at=str(item.get("started_at") or ""),
        ended_at=str(item.get("ended_at") or ""),
        status=str(item.get("status") or "unknown"),
        result=item.get("result") if isinstance(item.get("result"), dict) else {},
        error=str(item.get("error") or ""),
    )


def execute_plan_with_resume(
    plan: AgentPlan,
    ctx: ToolContext,
    *,
    resume_records: Optional[List[Dict[str, Any]]] = None,
    resume_from_index: int = 0,
) -> AgentExecutionResult:
    started = _now_iso()
    started_epoch = datetime.now(timezone.utc).timestamp()
    records: List[ToolExecutionRecord] = []
    status = "success"
    budget_control: Dict[str, Any] = {}
    if isinstance(resume_records, list):
        for item in resume_records:
            if isinstance(item, dict):
                records.append(_record_from_dict(item))

    design = plan.design_spec
    budget = design.budget
    max_tool_calls = _budget_limit_int(getattr(budget, "max_tool_calls", None), minimum=1)
    timeout_sec = _budget_limit_int(getattr(budget, "timeout_sec", None), minimum=1)
    max_external_calls = _budget_limit_int(getattr(budget, "max_external_calls", None), minimum=0)
    on_limit = str(getattr(budget, "on_limit", "fail") or "fail").strip().lower()
    if on_limit not in {"fail", "need_approval"}:
        on_limit = "fail"

    external_calls_done = 0
    for item in records:
        if _is_external_tool_call(item.name, item.result):
            external_calls_done += 1

    def _elapsed_seconds() -> float:
        return max(0.0, datetime.now(timezone.utc).timestamp() - started_epoch)

    def _limit_hit(
        *,
        name: str,
        limit: int,
        observed: int,
        pending_tool_name: str,
    ) -> bool:
        nonlocal status, budget_control
        event = {
            "limit_triggered": True,
            "action": on_limit,
            "check": name,
            "limit": int(limit),
            "observed": int(observed),
            "pending_tool": str(pending_tool_name or ""),
            "at": _now_iso(),
            "workflow_state": "WAITING_APPROVAL" if on_limit == "need_approval" else "FAILED",
            "message": (
                f"budget {name} exceeded ({observed} >= {limit}); waiting for approval"
                if on_limit == "need_approval"
                else f"budget {name} exceeded ({observed} >= {limit})"
            ),
        }
        budget_control = {
            "enabled": True,
            "max_tool_calls": max_tool_calls,
            "timeout_sec": timeout_sec,
            "max_external_calls": max_external_calls,
            "on_limit": on_limit,
            "external_calls_done": external_calls_done,
            **event,
        }
        now_iso = _now_iso()
        records.append(
            ToolExecutionRecord(
                name=str(pending_tool_name or "__budget_guardrail__"),
                args={"budget_control": budget_control},
                started_at=now_iso,
                ended_at=now_iso,
                status="failed",
                result={"status": "blocked", "budget_control": budget_control},
                error=str(event.get("message") or "budget limit exceeded"),
            )
        )
        status = "failed"
        return True

    budget_control = {
        "enabled": any(v is not None for v in (max_tool_calls, timeout_sec, max_external_calls)),
        "max_tool_calls": max_tool_calls,
        "timeout_sec": timeout_sec,
        "max_external_calls": max_external_calls,
        "on_limit": on_limit,
        "external_calls_done": external_calls_done,
        "limit_triggered": False,
    }

    for idx, call in enumerate(plan.tool_calls):
        if idx < int(resume_from_index or 0):
            continue
        call_started = _now_iso()
        call_args = dict(call.args)

        if call.name == "train_predictor":
            call_args.setdefault("target_specs", [asdict(t) for t in design.targets])
        elif call.name == "generate_candidates":
            call_args.setdefault("constraints", asdict(design.constraints))
            call_args.setdefault("max_candidates", design.budget.max_candidates)
        elif call.name == "score_candidates":
            call_args.setdefault("target_specs", [asdict(t) for t in design.targets])
            call_args.setdefault("targets", [t.name for t in design.targets])
        elif call.name == "filter_and_rank":
            call_args.setdefault("topn", 10)
            call_args.setdefault("target_specs", [asdict(t) for t in design.targets])

        if max_tool_calls is not None and len(records) >= int(max_tool_calls):
            if _limit_hit(
                name="max_tool_calls",
                limit=int(max_tool_calls),
                observed=len(records),
                pending_tool_name=call.name,
            ):
                break

        if (
            max_external_calls is not None
            and call.name in _EXTERNAL_TOOL_CANDIDATES
            and external_calls_done >= int(max_external_calls)
        ):
            if _limit_hit(
                name="max_external_calls",
                limit=int(max_external_calls),
                observed=external_calls_done,
                pending_tool_name=call.name,
            ):
                break

        if timeout_sec is not None and _elapsed_seconds() >= float(timeout_sec):
            if _limit_hit(
                name="timeout_sec",
                limit=int(timeout_sec),
                observed=int(_elapsed_seconds()),
                pending_tool_name=call.name,
            ):
                break

        try:
            result = execute_tool(ctx, call.name, call_args)
            call_ended = _now_iso()
            records.append(
                ToolExecutionRecord(
                    name=call.name,
                    args=call_args,
                    started_at=call_started,
                    ended_at=call_ended,
                    status="success",
                    result=result,
                )
            )
            if _is_external_tool_call(call.name, result):
                external_calls_done += 1
                budget_control["external_calls_done"] = external_calls_done
            if timeout_sec is not None and _elapsed_seconds() >= float(timeout_sec):
                if _limit_hit(
                    name="timeout_sec",
                    limit=int(timeout_sec),
                    observed=int(_elapsed_seconds()),
                    pending_tool_name=call.name,
                ):
                    break
        except Exception as exc:
            call_ended = _now_iso()
            records.append(
                ToolExecutionRecord(
                    name=call.name,
                    args=call_args,
                    started_at=call_started,
                    ended_at=call_ended,
                    status="failed",
                    error=str(exc),
                )
            )
            status = "failed"
            break

    ended = _now_iso()
    return AgentExecutionResult(
        task_id=plan.design_spec.task_id,
        status=status,
        started_at=started,
        ended_at=ended,
        records=records,
        budget_control=budget_control,
    )


def save_execution_result(result: AgentExecutionResult, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    import json

    output_path.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

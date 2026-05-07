from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from oled_agent.agent.specs import AgentPlan
from oled_agent.agent.tools import ToolContext, execute_tool


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "status": self.status,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "records": [asdict(r) for r in self.records],
        }


def execute_plan(plan: AgentPlan, ctx: ToolContext) -> AgentExecutionResult:
    started = _now_iso()
    records: List[ToolExecutionRecord] = []
    status = "success"

    design = plan.design_spec
    for call in plan.tool_calls:
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
    )


def save_execution_result(result: AgentExecutionResult, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    import json

    output_path.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

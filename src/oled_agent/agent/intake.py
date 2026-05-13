from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from oled_agent.agent.request_contract import validate_task_v2_payload
from oled_agent.agent.task_v2 import (
    build_web_query,
    compute_missing_questions,
    dump_json,
    ensure_task_ready_for_approval,
    infer_task_draft,
    run_duckduckgo_search,
)


def run_intake(
    *,
    workspace_root: Path,
    task_id: str,
    request_text: str,
    enable_web_search: bool = True,
    web_topk: int = 5,
) -> Dict[str, Any]:
    out_dir = (workspace_root / "runs" / "agent" / task_id).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    draft = infer_task_draft(request_text=request_text, task_id=task_id)
    evidence: List[Dict[str, str]] = []
    web_error = ""
    if enable_web_search:
        try:
            evidence = run_duckduckgo_search(query=build_web_query(draft), topk=web_topk)
        except Exception as exc:
            web_error = str(exc)
            evidence = []
    draft_prov = draft.get("provenance") if isinstance(draft.get("provenance"), dict) else {}
    draft_prov["web_evidence"] = evidence
    draft_prov["web_evidence_json"] = str(out_dir / "web_evidence.json")
    draft["provenance"] = draft_prov

    missing, questions = compute_missing_questions(draft)
    draft["missing_fields"] = missing
    draft["questions"] = questions
    draft["status"] = "need_user_input" if missing else "draft"

    web_payload = {
        "task_id": task_id,
        "query": build_web_query(draft),
        "topk": web_topk,
        "evidence": evidence,
        "error": web_error,
    }
    dump_json(out_dir / "web_evidence.json", web_payload)
    dump_json(out_dir / "task.draft.json", draft)

    return {
        "task_id": task_id,
        "status": draft.get("status"),
        "task_draft_path": str(out_dir / "task.draft.json"),
        "web_evidence_path": str(out_dir / "web_evidence.json"),
        "missing_fields": missing,
        "questions": questions,
    }


def approve_task(
    *,
    workspace_root: Path,
    task_payload: Dict[str, Any],
    planner_provider: str,
    catalog_path: Optional[Path],
    plan_fn,
) -> Dict[str, Any]:
    validate_task_v2_payload(task_payload, workspace_root)

    ok, missing = ensure_task_ready_for_approval(task_payload)
    if not ok:
        return {
            "task_id": str(task_payload.get("task_id") or ""),
            "status": "need_user_input",
            "missing_fields": missing,
            "questions": task_payload.get("questions") if isinstance(task_payload.get("questions"), list) else [],
        }

    task_id = str(task_payload.get("task_id") or "task_default")
    out_dir = (workspace_root / "runs" / "agent" / task_id).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    from oled_agent.agent.task_v2 import task_v2_to_request_payload

    request_payload = task_v2_to_request_payload(task_payload)
    plan = plan_fn(
        workspace_root=workspace_root,
        request_payload=request_payload,
        planner_provider=planner_provider,
        catalog_path=catalog_path,
    )

    task_out = dict(task_payload)
    task_out["status"] = "approved"

    dump_json(out_dir / "task.json", task_out)
    dump_json(out_dir / "request_from_task.json", request_payload)
    dump_json(out_dir / "plan.json", plan)

    plan_md_lines = [
        "# Plan",
        "",
        f"- task_id: {task_id}",
        f"- planner_provider: {planner_provider}",
        f"- summary: {plan.get('summary', '')}",
        "",
        "## Tool Calls",
    ]
    for i, call in enumerate(plan.get("tool_calls", []), start=1):
        if not isinstance(call, dict):
            continue
        plan_md_lines.append(f"{i}. `{call.get('name', '')}` {call.get('args', {})}")
    (out_dir / "plan.md").write_text("\n".join(plan_md_lines) + "\n", encoding="utf-8")

    return {
        "task_id": task_id,
        "status": "approved",
        "task_path": str(out_dir / "task.json"),
        "request_path": str(out_dir / "request_from_task.json"),
        "plan_path": str(out_dir / "plan.json"),
        "plan_md_path": str(out_dir / "plan.md"),
    }

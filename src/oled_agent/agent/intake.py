from __future__ import annotations

import json
from datetime import datetime, timezone
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


TASK_STATE_SCHEMA_VERSION = "1.0.0"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_pause_task_state(
    *,
    out_dir: Path,
    task_id: str,
    current_state: str,
    status: str,
    note: str = "",
) -> Path:
    now = _now_iso()
    task_state = {
        "schema_version": TASK_STATE_SCHEMA_VERSION,
        "generated_at": now,
        "task_id": task_id,
        "status": status,
        "current_state": current_state,
        "history": [
            {"state": "INIT", "status": "completed", "at": now},
            {"state": "REQUIREMENT_COLLECTION", "status": "completed", "at": now},
            {"state": "VALIDATION", "status": "completed", "at": now},
            {"state": current_state, "status": "failed" if status == "failed" else "completed", "at": now, "note": note},
        ],
    }
    out_path = out_dir / "task_state.json"
    _write_json(out_path, task_state)
    return out_path


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
    current_state = "NEED_INFO" if missing else "WAITING_APPROVAL"
    task_state_status = "failed" if missing else "success"

    web_payload = {
        "task_id": task_id,
        "query": build_web_query(draft),
        "topk": web_topk,
        "evidence": evidence,
        "error": web_error,
    }
    dump_json(out_dir / "web_evidence.json", web_payload)
    dump_json(out_dir / "task.draft.json", draft)
    task_state_path = _write_pause_task_state(
        out_dir=out_dir,
        task_id=task_id,
        current_state=current_state,
        status=task_state_status,
        note="intake paused before execution",
    )

    return {
        "task_id": task_id,
        "status": draft.get("status"),
        "task_draft_path": str(out_dir / "task.draft.json"),
        "web_evidence_path": str(out_dir / "web_evidence.json"),
        "task_state_path": str(task_state_path),
        "current_state": current_state,
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
    task_id = str(task_payload.get("task_id") or "task_default")
    out_dir = (workspace_root / "runs" / "agent" / task_id).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if not ok:
        draft_out = dict(task_payload)
        draft_out["status"] = "need_user_input"
        draft_out["missing_fields"] = missing
        if not isinstance(draft_out.get("questions"), list):
            draft_out["questions"] = []
        dump_json(out_dir / "task.draft.json", draft_out)
        task_state_path = _write_pause_task_state(
            out_dir=out_dir,
            task_id=task_id,
            current_state="NEED_INFO",
            status="failed",
            note="approval blocked by missing required fields",
        )
        return {
            "task_id": task_id,
            "status": "need_user_input",
            "task_draft_path": str(out_dir / "task.draft.json"),
            "task_state_path": str(task_state_path),
            "current_state": "NEED_INFO",
            "missing_fields": missing,
            "questions": task_payload.get("questions") if isinstance(task_payload.get("questions"), list) else [],
        }

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
    task_state_path = _write_pause_task_state(
        out_dir=out_dir,
        task_id=task_id,
        current_state="PLAN_GENERATION",
        status="success",
        note="plan generated and ready for execution",
    )

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
        "task_state_path": str(task_state_path),
        "current_state": "PLAN_GENERATION",
    }

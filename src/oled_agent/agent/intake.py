from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from oled_agent.agent.memory_context import retrieve_memory_hints
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


def _is_true_env(name: str, default: str = "0") -> bool:
    text = str(os.environ.get(name, default) or "").strip().lower()
    return text in ("1", "true", "yes", "on")


def _inject_memory_prompt_context(*, request_text: str, memory_prompt_context: str) -> str:
    base = str(request_text or "").strip()
    memory_text = str(memory_prompt_context or "").strip()
    if not memory_text:
        return base
    marker = "Historical run memory:"
    if marker in base:
        return base
    if not base:
        return f"{marker}\n{memory_text}"
    return f"{base}\n\n{marker}\n{memory_text}"


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
    if _is_true_env("OLED_AGENT_ENABLE_BACKEND_MEMORY", "1"):
        memory_hints = retrieve_memory_hints(
            workspace_root=workspace_root,
            request_text=request_text,
            current_task_id=task_id,
            topk=5,
        )
    else:
        memory_hints = {
            "status": "disabled",
            "index_path": str((workspace_root / "runs" / "agent" / "_memory" / "memory_index.json").resolve()),
            "query": str(request_text or ""),
            "matches": [],
            "suggested_candidate_data": "",
            "prompt_context": "",
        }
    memory_hints_path = out_dir / "memory_hints.json"
    dump_json(memory_hints_path, memory_hints if isinstance(memory_hints, dict) else {})
    draft_prov["memory_hints"] = (
        memory_hints.get("matches", []) if isinstance(memory_hints, dict) and isinstance(memory_hints.get("matches"), list) else []
    )
    draft_prov["memory_hints_json"] = str(memory_hints_path)
    draft_prov["memory_prompt_context"] = str(memory_hints.get("prompt_context") or "") if isinstance(memory_hints, dict) else ""
    suggested_candidate_data = str(memory_hints.get("suggested_candidate_data") or "").strip() if isinstance(memory_hints, dict) else ""
    if suggested_candidate_data:
        draft_prov["suggested_candidate_data"] = suggested_candidate_data
        if not str(draft.get("candidate_data") or "").strip() and _is_true_env("OLED_AGENT_INTAKE_AUTOFILL_CANDIDATE_DATA_FROM_MEMORY", "0"):
            draft["candidate_data"] = suggested_candidate_data
            draft_prov["candidate_data_autofill"] = True
    draft["provenance"] = draft_prov

    missing, questions = compute_missing_questions(draft)
    if "candidate_data" in missing and suggested_candidate_data:
        questions = list(questions) + [f"历史任务常用候选数据: {suggested_candidate_data}。如可复用请直接确认。"]
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
        "memory_hints_path": str(memory_hints_path),
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
    if _is_true_env("OLED_AGENT_APPROVE_INJECT_MEMORY_PROMPT", "1"):
        provenance = task_payload.get("provenance") if isinstance(task_payload.get("provenance"), dict) else {}
        memory_prompt_context = str(provenance.get("memory_prompt_context") or "").strip()
        if memory_prompt_context:
            request_payload["request_text"] = _inject_memory_prompt_context(
                request_text=str(request_payload.get("request_text") or ""),
                memory_prompt_context=memory_prompt_context,
            )
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

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from oled_agent.agent.failure_diagnostics import execution_failure_diagnostics


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_iso(value: Any) -> Optional[datetime]:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        return datetime.fromisoformat(text)
    except Exception:
        return None


def _duration_seconds(execution_payload: Dict[str, Any]) -> Optional[float]:
    started = _parse_iso(execution_payload.get("started_at"))
    ended = _parse_iso(execution_payload.get("ended_at"))
    if started is None or ended is None:
        return None
    return max(0.0, (ended - started).total_seconds())


def _resolve_optional_path(raw: Any, workspace_root: Path) -> Optional[Path]:
    text = str(raw or "").strip()
    if not text:
        return None
    p = Path(text)
    if not p.is_absolute():
        p = (workspace_root / p).resolve()
    else:
        p = p.resolve()
    return p


def _artifact_snapshot(raw: Any, workspace_root: Path) -> Dict[str, Any]:
    p = _resolve_optional_path(raw, workspace_root)
    if p is None:
        return {"path": "", "exists": False}
    return {"path": str(p), "exists": p.exists()}


def _decision_has_fallback(decision_summary: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(decision_summary, dict):
        return False
    for key in ("score_step", "inference_step"):
        step = decision_summary.get(key)
        if not isinstance(step, dict):
            continue
        if bool(step.get("used_fallback", False)):
            return True
        if str(step.get("fallback_code") or "").strip():
            return True
        if str(step.get("fallback_reason") or "").strip():
            return True
        if isinstance(step.get("fallback_error"), dict) and step.get("fallback_error"):
            return True
    return False


def build_evaluation_report(
    *,
    task_id: str,
    execution_mode: str,
    execution_payload: Dict[str, Any],
    decision_summary: Optional[Dict[str, Any]],
    task_state: Optional[Dict[str, Any]],
    tool_state: Optional[Dict[str, Any]],
    workspace_root: Path,
) -> Dict[str, Any]:
    records = execution_payload.get("records", []) if isinstance(execution_payload, dict) else []
    records = [r for r in records if isinstance(r, dict)]
    record_count = len(records)
    success_count = sum(1 for r in records if str(r.get("status") or "") == "success")
    failed_count = record_count - success_count
    adapters = sorted(
        {
            str((r.get("result") or {}).get("adapter") or "").strip()
            for r in records
            if isinstance(r.get("result"), dict) and str((r.get("result") or {}).get("adapter") or "").strip()
        }
    )
    fallback_count = sum(
        1
        for r in records
        if (
            isinstance(r.get("result"), dict)
            and (
                str((r.get("result") or {}).get("adapter") or "") == "local_deterministic_fallback"
                or isinstance((r.get("result") or {}).get("fallback_error"), dict)
                and bool((r.get("result") or {}).get("fallback_error"))
            )
        )
    )
    decision_fallback = _decision_has_fallback(decision_summary)
    execution_status = str(execution_payload.get("status") or "").strip()
    execution_status = execution_status if execution_status in ("success", "failed") else "failed"
    failure_diag = execution_failure_diagnostics(execution_payload)

    tool_state = tool_state if isinstance(tool_state, dict) else {}
    artifact_presence = {
        "candidate_csv": _artifact_snapshot(tool_state.get("candidate_csv"), workspace_root),
        "scored_csv": _artifact_snapshot(tool_state.get("scored_csv"), workspace_root),
        "final_output": _artifact_snapshot(tool_state.get("final_output"), workspace_root),
    }

    checks: list[Dict[str, Any]] = []

    def _check(name: str, status: str, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        checks.append(
            {
                "name": name,
                "status": status,
                "message": message,
                "details": details if isinstance(details, dict) else {},
            }
        )

    if record_count > 0:
        _check("execution_records", "pass", "execution records are non-empty", {"record_count": record_count})
    else:
        _check("execution_records", "fail", "execution records are empty", {"record_count": 0})

    if execution_status == "success" and failed_count > 0:
        _check(
            "execution_status_consistency",
            "fail",
            "execution status is success but failed records exist",
            {"failed_count": failed_count},
        )
    elif execution_status == "failed" and failed_count == 0 and record_count > 0:
        _check(
            "execution_status_consistency",
            "warn",
            "execution status is failed but no failed record found",
            {"record_count": record_count},
        )
    else:
        _check("execution_status_consistency", "pass", "execution status is consistent with records")

    state_payload = task_state if isinstance(task_state, dict) else {}
    current_state = str(state_payload.get("current_state") or "").strip()
    if current_state in ("DONE", "FAILED"):
        mismatch = (current_state == "DONE" and execution_status == "failed") or (
            current_state == "FAILED" and execution_status == "success"
        )
        if mismatch:
            _check(
                "task_state_terminal",
                "fail",
                "task_state terminal state conflicts with execution status",
                {"current_state": current_state, "execution_status": execution_status},
            )
        else:
            _check("task_state_terminal", "pass", "task_state reached terminal state", {"current_state": current_state})
    else:
        _check(
            "task_state_terminal",
            "warn",
            "task_state does not show a terminal state",
            {"current_state": current_state},
        )

    if fallback_count > 0 or decision_fallback:
        _check(
            "fallback_usage",
            "warn",
            "fallback path detected in inference flow",
            {"fallback_count": fallback_count, "decision_fallback": decision_fallback},
        )
    else:
        _check("fallback_usage", "pass", "no fallback path detected")

    if failed_count > 0:
        latest_step = str(failure_diag.get("latest_failed_step") or "").strip()
        latest_kind = str(failure_diag.get("latest_failure_kind") or "").strip()
        if not latest_step:
            _check(
                "failure_diagnostics",
                "fail",
                "failed records exist but latest_failed_step is missing",
                {"failed_count": failed_count},
            )
        elif not latest_kind:
            _check(
                "failure_diagnostics",
                "warn",
                "failed records exist but latest_failure_kind is empty",
                {"latest_failed_step": latest_step},
            )
        else:
            _check(
                "failure_diagnostics",
                "pass",
                "failure diagnostics are available",
                {"latest_failed_step": latest_step, "latest_failure_kind": latest_kind},
            )
    else:
        _check("failure_diagnostics", "pass", "no failed records")

    successful_tools = {
        str(r.get("name") or "")
        for r in records
        if str(r.get("status") or "") == "success" and str(r.get("name") or "").strip()
    }
    candidate_expected_tools = {
        "retrieve_candidate_data",
        "clean_dataset",
        "prepare_train_data",
        "generate_candidates",
    }
    scored_expected_tools = {"score_candidates", "filter_and_rank"}
    report_expected_tools = {"make_report"}
    if successful_tools.intersection(candidate_expected_tools) and not bool(artifact_presence["candidate_csv"]["exists"]):
        _check("artifact_candidate_csv", "fail", "candidate_csv artifact is missing after successful preprocessing/generation")
    else:
        _check("artifact_candidate_csv", "pass", "candidate_csv artifact check passed")
    if successful_tools.intersection(scored_expected_tools) and not bool(artifact_presence["scored_csv"]["exists"]):
        _check("artifact_scored_csv", "fail", "scored_csv artifact is missing after successful scoring/filtering")
    else:
        _check("artifact_scored_csv", "pass", "scored_csv artifact check passed")
    if successful_tools.intersection(report_expected_tools) and not bool(artifact_presence["final_output"]["exists"]):
        _check("artifact_final_output", "warn", "final_output artifact is missing after successful reporting")
    else:
        _check("artifact_final_output", "pass", "final_output artifact check passed")

    pass_count = sum(1 for c in checks if c.get("status") == "pass")
    warn_count = sum(1 for c in checks if c.get("status") == "warn")
    fail_count = sum(1 for c in checks if c.get("status") == "fail")
    status = "fail" if fail_count > 0 else ("warn" if warn_count > 0 else "pass")

    return {
        "schema_version": "1.0.0",
        "generated_at": _now_iso(),
        "task_id": str(task_id or ""),
        "execution_mode": str(execution_mode or ""),
        "execution_status": execution_status,
        "status": status,
        "summary": {
            "checks_total": len(checks),
            "pass_count": pass_count,
            "warn_count": warn_count,
            "fail_count": fail_count,
        },
        "metrics": {
            "record_count": record_count,
            "success_count": success_count,
            "failed_count": failed_count,
            "fallback_count": fallback_count,
            "decision_fallback": decision_fallback,
            "adapters": adapters,
            "duration_seconds": _duration_seconds(execution_payload),
            "latest_failure_kind": str(failure_diag.get("latest_failure_kind") or ""),
            "latest_failed_step": str(failure_diag.get("latest_failed_step") or ""),
        },
        "failure_diagnostics": failure_diag,
        "artifact_presence": artifact_presence,
        "checks": checks,
    }

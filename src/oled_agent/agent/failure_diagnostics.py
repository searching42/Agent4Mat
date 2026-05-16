from __future__ import annotations

from typing import Any, Dict, List, Optional


def classify_failure_kind(
    *,
    status_text: str = "",
    token_blob: str = "",
    returncode: Any = None,
    missing_fields: Optional[List[str]] = None,
) -> str:
    status = str(status_text or "").strip().lower()
    missing = [str(x).strip() for x in (missing_fields or []) if str(x).strip()]
    blob = str(token_blob or "").lower()
    if status == "need_user_input" or len(missing) > 0:
        return "need_user_input"
    if "timeout" in blob or "timed out" in blob or "deadline" in blob or returncode in {124, 137}:
        return "timeout"
    if any(
        token in blob
        for token in (
            "adapter",
            "external scorer",
            "adapter_nonzero_exit",
            "adapter_timeout",
            "missing_output_csv",
            "invalid_json_stdin",
            "toolerror",
            "tool error",
            "fallback",
        )
    ):
        return "adapter_failure"
    return "unknown"


def execution_failure_diagnostics(execution: Any) -> Dict[str, Any]:
    if not isinstance(execution, dict):
        return {
            "failed_count": 0,
            "latest_failed_step": "",
            "latest_failed_error": "",
            "latest_failure_kind": "",
            "latest_failure_detail": "",
        }
    records = execution.get("records") if isinstance(execution.get("records"), list) else []
    failed_count = 0
    latest_failed_step = ""
    latest_failed_error = ""
    latest_failure_detail = ""
    latest_failure_kind = ""
    for rec in records:
        if not isinstance(rec, dict):
            continue
        status = str(rec.get("status") or "").strip()
        if status == "success":
            continue
        failed_count += 1
        latest_failed_step = str(rec.get("name") or latest_failed_step).strip()
        err_txt = str(rec.get("error") or "").strip()
        result = rec.get("result") if isinstance(rec.get("result"), dict) else {}
        detail_parts: List[str] = []
        if err_txt:
            detail_parts.append(err_txt[:280])
        for key in ("error", "message", "detail", "reason", "code"):
            value = str(result.get(key) or "").strip() if isinstance(result, dict) else ""
            if value:
                detail_parts.append(f"{key}={value[:220]}")
        rc = result.get("returncode") if isinstance(result, dict) else None
        if rc is not None:
            detail_parts.append(f"returncode={rc}")
        latest_failed_error = err_txt[:240] if err_txt else latest_failed_error
        latest_failure_detail = "; ".join(part for part in detail_parts if part)[:800]
        tokens = " ".join(
            [
                status,
                latest_failed_step,
                latest_failed_error,
                latest_failure_detail,
            ]
        )
        latest_failure_kind = classify_failure_kind(
            status_text=status,
            token_blob=tokens,
            returncode=rc,
            missing_fields=None,
        )
    return {
        "failed_count": failed_count,
        "latest_failed_step": latest_failed_step,
        "latest_failed_error": latest_failed_error,
        "latest_failure_kind": latest_failure_kind if failed_count > 0 else "",
        "latest_failure_detail": latest_failure_detail if failed_count > 0 else "",
    }

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse

from oled_agent.agent.failure_diagnostics import execution_failure_diagnostics


_STUB_ADAPTERS = {
    "stub_generator",
    "dataset_stub_retrieval",
    "train_data_stub_builder",
    "local_cleaning_v1",
    "template_generate_cmd",
    "template_score_cmd",
    "template_train_cmd",
}
_FALLBACK_ADAPTERS = {"local_deterministic_fallback"}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def _safe_load_json(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if path is None or not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _csv_data_rows(path: Optional[Path]) -> int:
    if path is None or not path.exists() or not path.is_file():
        return 0
    try:
        text = path.read_text(encoding="utf-8", errors="replace").strip()
    except Exception:
        return 0
    if not text:
        return 0
    lines = text.splitlines()
    if len(lines) <= 1:
        return 0
    return len(lines) - 1


def _is_non_public_url(url: str) -> bool:
    parsed = urlparse(str(url or "").strip())
    scheme = str(parsed.scheme or "").lower()
    if scheme and scheme not in ("http", "https"):
        return True
    host = str(parsed.hostname or "").strip().lower()
    if not host:
        return True
    if host in ("localhost", "::1"):
        return True
    if host.endswith(".local"):
        return True
    if host.startswith("10.") or host.startswith("127.") or host.startswith("192.168.") or host.startswith("169.254."):
        return True
    if host.startswith("172."):
        parts = host.split(".")
        if len(parts) >= 2:
            try:
                second = int(parts[1])
            except Exception:
                second = -1
            if 16 <= second <= 31:
                return True
    return False


def build_guardrails_report(
    *,
    task_id: str,
    execution_mode: str,
    execution_payload: Dict[str, Any],
    tool_state: Optional[Dict[str, Any]],
    workspace_root: Path,
    constraints: Optional[Dict[str, Any]] = None,
    web_evidence_path: Optional[Path] = None,
) -> Dict[str, Any]:
    records = execution_payload.get("records", []) if isinstance(execution_payload, dict) else []
    records = [r for r in records if isinstance(r, dict)]
    execution_status = str(execution_payload.get("status") or "").strip()
    if execution_status not in ("success", "failed"):
        execution_status = "failed"
    failure_diag = execution_failure_diagnostics(execution_payload)
    succeeded_tools = {str(r.get("name") or "") for r in records if str(r.get("status") or "") == "success"}

    tool_state = tool_state if isinstance(tool_state, dict) else {}
    constraints = constraints if isinstance(constraints, dict) else {}

    candidate_csv_path = _resolve_optional_path(tool_state.get("candidate_csv"), workspace_root)
    scored_csv_path = _resolve_optional_path(tool_state.get("scored_csv"), workspace_root)
    final_output_path = _resolve_optional_path(tool_state.get("final_output"), workspace_root)
    cleaning_report_path = _resolve_optional_path(tool_state.get("cleaning_report_json"), workspace_root)
    cleaning_report = _safe_load_json(cleaning_report_path)

    checks: list[Dict[str, Any]] = []

    def _check(name: str, status: str, message: str, *, strict_blocking: bool, details: Optional[Dict[str, Any]] = None) -> None:
        checks.append(
            {
                "name": name,
                "status": status,
                "strict_blocking": strict_blocking,
                "message": message,
                "details": details if isinstance(details, dict) else {},
            }
        )

    if execution_status == "success":
        _check("execution_status", "pass", "execution status is success", strict_blocking=True)
    else:
        _check("execution_status", "fail", "execution status is failed", strict_blocking=True)

    if execution_status == "failed":
        failed_count = int(failure_diag.get("failed_count") or 0)
        latest_step = str(failure_diag.get("latest_failed_step") or "").strip()
        latest_kind = str(failure_diag.get("latest_failure_kind") or "").strip()
        if failed_count <= 0:
            _check(
                "failure_diagnostics",
                "fail",
                "execution failed but no failed execution records were found",
                strict_blocking=True,
            )
        elif not latest_step:
            _check(
                "failure_diagnostics",
                "fail",
                "execution failed but latest_failed_step is missing",
                strict_blocking=True,
                details={"failed_count": failed_count},
            )
        elif latest_kind in ("", "unknown"):
            _check(
                "failure_diagnostics",
                "warn",
                "execution failed but failure kind classification is weak",
                strict_blocking=False,
                details={"latest_failed_step": latest_step, "latest_failure_kind": latest_kind},
            )
        else:
            _check(
                "failure_diagnostics",
                "pass",
                "failure diagnostics are available",
                strict_blocking=False,
                details={"latest_failed_step": latest_step, "latest_failure_kind": latest_kind},
            )
    else:
        _check("failure_diagnostics", "pass", "execution succeeded; no failure diagnostics required", strict_blocking=False)

    budget_control = execution_payload.get("budget_control") if isinstance(execution_payload.get("budget_control"), dict) else {}
    if bool(budget_control.get("limit_triggered")):
        action = str(budget_control.get("action") or "fail").strip().lower() or "fail"
        limit_name = str(budget_control.get("check") or "").strip()
        message = str(budget_control.get("message") or "").strip() or f"budget limit triggered ({limit_name})"
        if action == "need_approval":
            _check(
                "budget_limit",
                "warn",
                message,
                strict_blocking=False,
                details=budget_control,
            )
        else:
            _check(
                "budget_limit",
                "fail",
                message,
                strict_blocking=True,
                details=budget_control,
            )
    else:
        _check("budget_limit", "pass", "no budget limit triggered", strict_blocking=False)

    adapters = []
    stub_adapters = set()
    fallback_adapters = set()
    for rec in records:
        result = rec.get("result") if isinstance(rec.get("result"), dict) else {}
        adapter = str(result.get("adapter") or "").strip()
        if not adapter:
            continue
        adapters.append(adapter)
        if adapter in _STUB_ADAPTERS:
            stub_adapters.add(adapter)
        if adapter in _FALLBACK_ADAPTERS or (isinstance(result.get("fallback_error"), dict) and result.get("fallback_error")):
            fallback_adapters.add(adapter)
    if stub_adapters:
        _check(
            "stub_adapters",
            "warn",
            "stub/local adapters were used",
            strict_blocking=True,
            details={"adapters": sorted(stub_adapters)},
        )
    else:
        _check("stub_adapters", "pass", "no stub/local adapters used", strict_blocking=True)
    if fallback_adapters:
        _check(
            "fallback_adapters",
            "warn",
            "fallback path detected",
            strict_blocking=True,
            details={"adapters": sorted(fallback_adapters)},
        )
    else:
        _check("fallback_adapters", "pass", "no fallback path detected", strict_blocking=True)

    candidate_rows = _csv_data_rows(candidate_csv_path)
    scored_rows = _csv_data_rows(scored_csv_path)
    if succeeded_tools.intersection({"retrieve_candidate_data", "clean_dataset", "prepare_train_data", "generate_candidates"}):
        if candidate_rows > 0:
            _check("candidate_rows", "pass", "candidate csv has rows", strict_blocking=True, details={"rows": candidate_rows})
        else:
            _check("candidate_rows", "fail", "candidate csv is missing or empty", strict_blocking=True, details={"rows": candidate_rows})
    else:
        _check("candidate_rows", "pass", "candidate generation/retrieval not executed", strict_blocking=False)

    if succeeded_tools.intersection({"score_candidates", "filter_and_rank"}):
        if scored_rows > 0:
            _check("scored_rows", "pass", "scored csv has rows", strict_blocking=True, details={"rows": scored_rows})
        else:
            _check("scored_rows", "fail", "scored csv is missing or empty", strict_blocking=True, details={"rows": scored_rows})
    else:
        _check("scored_rows", "pass", "scoring/filtering not executed", strict_blocking=False)

    if "make_report" in succeeded_tools:
        if final_output_path is not None and final_output_path.exists():
            _check("final_report_output", "pass", "final report output exists", strict_blocking=False, details={"path": str(final_output_path)})
        else:
            _check("final_report_output", "warn", "final report output is missing", strict_blocking=False)
    else:
        _check("final_report_output", "pass", "report step not executed", strict_blocking=False)

    mw_threshold_set = isinstance(constraints.get("mw_min"), (int, float)) or isinstance(constraints.get("mw_max"), (int, float))
    if mw_threshold_set:
        if isinstance(cleaning_report, dict):
            hard = bool(cleaning_report.get("mw_filter_hard_applied", False))
            method = str(cleaning_report.get("mw_method") or "")
            if hard:
                _check(
                    "mw_filter_mode",
                    "pass",
                    "molecular weight threshold used hard filtering",
                    strict_blocking=False,
                    details={"mw_method": method},
                )
            else:
                _check(
                    "mw_filter_mode",
                    "warn",
                    "molecular weight threshold only soft-checked",
                    strict_blocking=False,
                    details={"mw_method": method, "warnings": cleaning_report.get("warnings", [])},
                )
        else:
            _check("mw_filter_mode", "warn", "constraints contain mw threshold but cleaning report is missing", strict_blocking=False)
    else:
        _check("mw_filter_mode", "pass", "mw threshold constraint not set", strict_blocking=False)

    banned_alerts = constraints.get("banned_alerts") if isinstance(constraints.get("banned_alerts"), list) else []
    banned_alerts = [str(x).strip() for x in banned_alerts if str(x).strip()]
    if banned_alerts and isinstance(cleaning_report, dict):
        dropped = int(cleaning_report.get("drop_banned_alert") or 0)
        if dropped > 0:
            _check(
                "banned_alerts_enforced",
                "pass",
                "banned alerts were actively filtered",
                strict_blocking=False,
                details={"dropped": dropped, "rules": banned_alerts},
            )
        else:
            _check(
                "banned_alerts_enforced",
                "warn",
                "banned alerts configured but no matches were filtered",
                strict_blocking=False,
                details={"rules": banned_alerts},
            )
    else:
        _check("banned_alerts_enforced", "pass", "no banned alerts configured", strict_blocking=False)

    web_evidence = _safe_load_json(web_evidence_path)
    if isinstance(web_evidence, dict):
        results = web_evidence.get("results") if isinstance(web_evidence.get("results"), list) else []
        bad_urls = []
        for row in results:
            if not isinstance(row, dict):
                continue
            url = str(row.get("url") or "").strip()
            if not url:
                continue
            if _is_non_public_url(url):
                bad_urls.append(url)
        if bad_urls:
            _check(
                "web_evidence_public_sources",
                "fail",
                "web evidence includes non-public or unsafe URLs",
                strict_blocking=True,
                details={"count": len(bad_urls), "sample": bad_urls[:5]},
            )
        else:
            _check(
                "web_evidence_public_sources",
                "pass",
                "web evidence URLs are public",
                strict_blocking=False,
                details={"count": len(results)},
            )
    else:
        _check("web_evidence_public_sources", "pass", "web evidence artifact not present", strict_blocking=False)

    pass_count = sum(1 for c in checks if c.get("status") == "pass")
    warn_count = sum(1 for c in checks if c.get("status") == "warn")
    fail_count = sum(1 for c in checks if c.get("status") == "fail")
    strict_blocking_checks = [
        c.get("name")
        for c in checks
        if c.get("strict_blocking") and c.get("status") in ("warn", "fail")
    ]
    blocking_checks = [c.get("name") for c in checks if c.get("status") == "fail"]
    status = "fail" if fail_count > 0 else ("warn" if warn_count > 0 else "pass")
    strict_status = "fail" if strict_blocking_checks else "pass"

    return {
        "schema_version": "1.0.0",
        "generated_at": _now_iso(),
        "task_id": str(task_id or ""),
        "execution_mode": str(execution_mode or ""),
        "execution_status": execution_status,
        "status": status,
        "strict_status": strict_status,
        "summary": {
            "checks_total": len(checks),
            "pass_count": pass_count,
            "warn_count": warn_count,
            "fail_count": fail_count,
            "strict_blocking_count": len(strict_blocking_checks),
        },
        "blocking_checks": blocking_checks,
        "strict_blocking_checks": strict_blocking_checks,
        "metrics": {
            "record_count": len(records),
            "adapters": sorted(set(adapters)),
            "stub_adapters": sorted(stub_adapters),
            "fallback_adapters": sorted(fallback_adapters),
            "candidate_rows": candidate_rows,
            "scored_rows": scored_rows,
            "latest_failure_kind": str(failure_diag.get("latest_failure_kind") or ""),
            "latest_failed_step": str(failure_diag.get("latest_failed_step") or ""),
        },
        "failure_diagnostics": failure_diag,
        "checks": checks,
    }

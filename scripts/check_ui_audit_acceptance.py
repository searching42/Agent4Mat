#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest import mock


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve_path(workspace_root: Path, raw: str) -> Path:
    path = Path(str(raw or "").strip())
    if path.is_absolute():
        return path.resolve()
    return (workspace_root / path).resolve()


def _read_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"json is not object: {path}")
    return payload


def _check_result(name: str, *, ok: bool, details: Optional[Dict[str, Any]] = None, error: str = "") -> Dict[str, Any]:
    return {
        "name": name,
        "status": "pass" if ok else "fail",
        "error": str(error or ""),
        "details": details if isinstance(details, dict) else {},
    }


def _seed_task_sandbox(root: Path, *, task_id: str, run_label: str) -> None:
    run_dir = root / "runs" / "agent" / task_id
    result_dir = root / "result" / run_label
    logging_dir = root / "logging" / run_label
    run_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    logging_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "artifacts").mkdir(parents=True, exist_ok=True)
    (run_dir / "task.draft.json").write_text(json.dumps({"task_id": task_id}, ensure_ascii=False) + "\n", encoding="utf-8")
    (run_dir / "task.json").write_text(json.dumps({"task_id": task_id}, ensure_ascii=False) + "\n", encoding="utf-8")
    (run_dir / "request_from_task.json").write_text(
        json.dumps({"task_id": task_id, "request_text": "audit acceptance seed"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (run_dir / "execution.json").write_text(
        json.dumps(
            {
                "task_id": task_id,
                "status": "success",
                "records": [
                    {"name": "search_dataset", "status": "success"},
                    {"name": "score_candidates", "status": "success"},
                ],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "decision_summary.json").write_text(
        json.dumps(
            {
                "task_id": task_id,
                "status": "success",
                "artifacts": {
                    "result_dir": str(result_dir),
                    "logging_dir": str(logging_dir),
                },
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "task_state.json").write_text(
        json.dumps({"task_id": task_id, "status": "success"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (run_dir / "artifacts" / "experiment_trace.json").write_text(
        json.dumps(
            {
                "task_id": task_id,
                "run_label": run_label,
                "execution_mode": "full_pipeline",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (result_dir / "metadata.json").write_text(
        json.dumps({"task_id": task_id, "run_label": run_label}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (logging_dir / "task.json").write_text(
        json.dumps({"task_id": task_id, "run_label": run_label}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _run_check_select_task_success(client: Any, *, task_id: str, run_label: str) -> Dict[str, Any]:
    project_id = "ui_audit_success_proj"
    c1 = client.post("/api/projects", json={"project_id": project_id, "title": "audit success"})
    c2 = client.post(f"/api/projects/{project_id}/select-task", json={"task_id": task_id})
    payload = c2.get_json() if c2.is_json else {}
    selected = payload.get("selected_task") if isinstance(payload.get("selected_task"), dict) else {}
    project = payload.get("project") if isinstance(payload.get("project"), dict) else {}
    runtime = project.get("last_runtime") if isinstance(project.get("last_runtime"), dict) else {}
    result_dir_meta = selected.get("result_dir") if isinstance(selected.get("result_dir"), dict) else {}
    messages = payload.get("messages") if isinstance(payload.get("messages"), list) else []
    has_select_note = any(str(msg.get("kind") or "") == "task_select" for msg in messages if isinstance(msg, dict))
    ok = (
        c1.status_code == 200
        and c2.status_code == 200
        and str(payload.get("status") or "") == "pass"
        and str(payload.get("task_id") or "") == task_id
        and str(selected.get("status") or "") == "pass"
        and str(selected.get("run_label") or "") == run_label
        and bool(result_dir_meta.get("exists"))
        and str(project.get("current_task_id") or "") == task_id
        and str(runtime.get("operation") or "") == "select_task"
        and str(runtime.get("task_id") or "") == task_id
        and has_select_note
    )
    return _check_result(
        "select_task_success_context_sync",
        ok=ok,
        error="" if ok else "select-task success flow did not sync project runtime/context",
        details={
            "http_codes": {"create_project": c1.status_code, "select_task": c2.status_code},
            "selected_status": str(selected.get("status") or ""),
            "selected_run_label": str(selected.get("run_label") or ""),
            "result_dir_exists": bool(result_dir_meta.get("exists")),
            "project_current_task_id": str(project.get("current_task_id") or ""),
            "runtime_operation": str(runtime.get("operation") or ""),
            "has_task_select_note": has_select_note,
        },
    )


def _run_check_audit_artifact_actions_api(client: Any, *, task_id: str) -> Dict[str, Any]:
    summary_resp = client.get(f"/api/task/{task_id}/summary")
    decision_resp = client.get(f"/api/task/{task_id}/artifact/decision_summary?max_chars=4096")
    links_resp = client.get(f"/api/task/{task_id}/artifact-links")

    summary_payload = summary_resp.get_json() if summary_resp.is_json else {}
    decision_payload = decision_resp.get_json() if decision_resp.is_json else {}
    links_payload = links_resp.get_json() if links_resp.is_json else {}

    summary_status = str(summary_payload.get("status") or "")
    decision_status = str(decision_payload.get("status") or "")
    links_status = str(links_payload.get("status") or "")
    decision_exists = bool(decision_payload.get("exists"))
    decision_artifact = str(decision_payload.get("artifact") or "")
    result_dir_meta = links_payload.get("result_dir") if isinstance(links_payload.get("result_dir"), dict) else {}
    logging_dir_meta = links_payload.get("logging_dir") if isinstance(links_payload.get("logging_dir"), dict) else {}
    has_summary_api = str(links_payload.get("summary_api") or "").endswith(f"/api/task/{task_id}/summary")
    has_decision_api = str(links_payload.get("decision_summary_api") or "").endswith(
        f"/api/task/{task_id}/artifact/decision_summary"
    )
    has_bundle_api = str(links_payload.get("bundle_api") or "").endswith(f"/api/task/{task_id}/bundle")
    ok = (
        summary_resp.status_code == 200
        and decision_resp.status_code == 200
        and links_resp.status_code == 200
        and summary_status == "pass"
        and decision_status == "pass"
        and links_status == "pass"
        and decision_exists
        and decision_artifact == "decision_summary"
        and bool(result_dir_meta.get("exists"))
        and bool(logging_dir_meta.get("exists"))
        and has_summary_api
        and has_decision_api
        and has_bundle_api
    )
    return _check_result(
        "audit_artifact_actions_api_contract",
        ok=ok,
        error="" if ok else "summary/decision/paths contract mismatch for audit actions",
        details={
            "http_codes": {
                "summary": summary_resp.status_code,
                "decision_summary": decision_resp.status_code,
                "artifact_links": links_resp.status_code,
            },
            "summary_status": summary_status,
            "decision_status": decision_status,
            "links_status": links_status,
            "decision_exists": decision_exists,
            "result_dir_exists": bool(result_dir_meta.get("exists")),
            "logging_dir_exists": bool(logging_dir_meta.get("exists")),
            "has_summary_api": has_summary_api,
            "has_decision_api": has_decision_api,
            "has_bundle_api": has_bundle_api,
        },
    )


def _run_check_select_task_missing_hints(client: Any) -> Dict[str, Any]:
    project_id = "ui_audit_missing_proj"
    recent_id = "ui_audit_recent_20260517"
    client.post("/api/projects", json={"project_id": project_id, "title": "audit missing"})
    resp = client.post(f"/api/projects/{project_id}/select-task", json={"task_id": "ui_audit_missing_20260517"})
    payload = resp.get_json() if resp.is_json else {}
    suggestions = payload.get("suggestions") if isinstance(payload.get("suggestions"), list) else []
    recents = payload.get("recent_task_ids") if isinstance(payload.get("recent_task_ids"), list) else []
    has_verify_hint = any("verify task_id" in str(x).lower() for x in suggestions)
    has_recent = any(str(item or "") == recent_id for item in recents)
    ok = (
        resp.status_code == 404
        and str(payload.get("status") or "") == "fail"
        and str(payload.get("error") or "") == "task_run_dir_not_found"
        and len(suggestions) >= 1
        and has_verify_hint
        and has_recent
    )
    return _check_result(
        "select_task_missing_run_dir_hints",
        ok=ok,
        error="" if ok else "missing-run-dir flow did not return expected hints/recent task ids",
        details={
            "http_code": resp.status_code,
            "status": str(payload.get("status") or ""),
            "error": str(payload.get("error") or ""),
            "suggestions_count": len(suggestions),
            "recent_task_ids": recents,
            "has_verify_hint": has_verify_hint,
            "has_recent_task_id": has_recent,
        },
    )


def _run_check_select_task_read_only_hints(client: Any, *, task_id: str) -> Dict[str, Any]:
    project_id = "ui_audit_read_only_proj"
    c1 = client.post(
        "/api/projects",
        json={"project_id": project_id, "title": "audit read only", "options": {"project_read_only": True}},
    )
    c2 = client.post(f"/api/projects/{project_id}/select-task", json={"task_id": task_id})
    payload = c2.get_json() if c2.is_json else {}
    suggestions = payload.get("suggestions") if isinstance(payload.get("suggestions"), list) else []
    reason = str(payload.get("reason") or "")
    has_unlock_hint = any("unlock" in str(x).lower() for x in suggestions)
    ok = (
        c1.status_code == 200
        and c2.status_code == 409
        and str(payload.get("status") or "") == "fail"
        and str(payload.get("error") or "") == "project_read_only"
        and "read-only" in reason.lower()
        and has_unlock_hint
    )
    return _check_result(
        "select_task_read_only_hints",
        ok=ok,
        error="" if ok else "read-only select-task failure hint contract mismatch",
        details={
            "http_codes": {"create_project": c1.status_code, "select_task": c2.status_code},
            "status": str(payload.get("status") or ""),
            "error": str(payload.get("error") or ""),
            "reason": reason,
            "suggestions": suggestions,
            "has_unlock_hint": has_unlock_hint,
        },
    )


def _run_check_invalid_task_id_rejected(client: Any) -> Dict[str, Any]:
    task_resp = client.get("/api/task/bad..id/artifact-links")
    task_payload = task_resp.get_json() if task_resp.is_json else {}
    project_id = "ui_audit_invalid_task_proj"
    client.post("/api/projects", json={"project_id": project_id, "title": "invalid task id"})
    select_resp = client.post(f"/api/projects/{project_id}/select-task", json={"task_id": "bad..id"})
    select_payload = select_resp.get_json() if select_resp.is_json else {}
    ok = (
        task_resp.status_code == 400
        and str(task_payload.get("status") or "") == "fail"
        and str(task_payload.get("error") or "") == "invalid task_id"
        and select_resp.status_code == 400
        and str(select_payload.get("status") or "") == "fail"
        and str(select_payload.get("error") or "") == "invalid task_id"
    )
    return _check_result(
        "invalid_task_id_rejected",
        ok=ok,
        error="" if ok else "invalid task id should be rejected by artifact-links/select-task",
        details={
            "task_http_code": task_resp.status_code,
            "task_error": str(task_payload.get("error") or ""),
            "select_http_code": select_resp.status_code,
            "select_error": str(select_payload.get("error") or ""),
        },
    )


def _run_check_audit_ui_tokens(client: Any) -> Dict[str, Any]:
    resp = client.get("/")
    html = resp.get_data(as_text=True) if resp.status_code == 200 else ""
    required_tokens = [
        "control_center_audit_filter",
        "control_center_audit_sort",
        "control_center_audit_query",
        "control_center_audit_copy_btn",
        "control_center_audit_export_btn",
        "control_center_audit_clear_btn",
        "openAuditTaskSummary(",
        "openAuditDecisionSummary(",
        "openAuditArtifactLinks(",
        "copyAuditResultDirPath(",
        "ensureAuditTaskActive(",
        "activateAuditTask(",
        "_handleAuditActionFailure(",
        "_auditFailureSuggestionLines(",
        "/api/task/${encodeURIComponent(tid)}/artifact-links",
        "/api/projects/${encodeURIComponent(pid)}/select-task",
    ]
    missing = [token for token in required_tokens if token not in html]
    ok = resp.status_code == 200 and len(missing) == 0
    return _check_result(
        "audit_ui_tokens_present",
        ok=ok,
        error="" if ok else f"missing required audit ui tokens: {missing}",
        details={"http_code": resp.status_code, "missing_tokens": missing},
    )


def _load_baseline(path: Path) -> Tuple[Optional[Dict[str, Any]], str]:
    if not path.exists():
        return None, f"baseline file missing: {path}"
    try:
        payload = _read_json(path)
    except Exception as exc:
        return None, f"baseline parse failed: {type(exc).__name__}: {exc}"
    rows = payload.get("required_checks")
    if not isinstance(rows, list):
        return None, "baseline.required_checks must be list"
    return payload, ""


def _apply_baseline(report: Dict[str, Any], baseline: Dict[str, Any]) -> None:
    checks = report.get("checks") if isinstance(report.get("checks"), list) else []
    check_map = {
        str(row.get("name") or ""): str(row.get("status") or "")
        for row in checks
        if isinstance(row, dict) and str(row.get("name") or "").strip()
    }
    required = baseline.get("required_checks") if isinstance(baseline.get("required_checks"), list) else []
    missing: List[str] = []
    mismatched: List[Dict[str, Any]] = []
    for raw in required:
        if isinstance(raw, str):
            name = str(raw).strip()
            expected = "pass"
        elif isinstance(raw, dict):
            name = str(raw.get("name") or "").strip()
            expected = str(raw.get("status") or "pass").strip() or "pass"
        else:
            continue
        if not name:
            continue
        actual = check_map.get(name)
        if actual is None:
            missing.append(name)
            continue
        if actual != expected:
            mismatched.append({"name": name, "expected": expected, "actual": actual})

    report["baseline"] = {
        "path": str(report.get("baseline_path") or ""),
        "schema_version": str(baseline.get("schema_version") or ""),
        "required_checks": required,
    }
    report["missing_required_checks"] = missing
    report["mismatched_required_checks"] = mismatched
    if missing or mismatched:
        report["status"] = "fail"
        report["failure_reason"] = "baseline_required_checks_not_satisfied"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run targeted UI audit-link acceptance checks (task switching + artifact actions + failure hints)"
    )
    parser.add_argument("--workspace-root", default=".", help="workspace root path")
    parser.add_argument("--out", default="runs/ci/ui_audit_acceptance.json", help="output json path")
    parser.add_argument(
        "--baseline",
        default="configs/acceptance/ui_audit_acceptance_baseline.json",
        help="baseline contract json path",
    )
    args = parser.parse_args()

    workspace_root = Path(args.workspace_root).resolve()
    out_path = _resolve_path(workspace_root, str(args.out or ""))
    baseline_path = _resolve_path(workspace_root, str(args.baseline or ""))

    if str(workspace_root) not in sys.path:
        sys.path.insert(0, str(workspace_root))

    report: Dict[str, Any] = {
        "status": "fail",
        "generated_at": _now_iso(),
        "workspace_root": str(workspace_root),
        "out_path": str(out_path),
        "baseline_path": str(baseline_path),
        "checks": [],
    }

    baseline_payload, baseline_error = _load_baseline(baseline_path)
    if baseline_payload is None:
        report["failure_reason"] = baseline_error
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 1

    try:
        from ui import app as ui_app_mod  # type: ignore
    except Exception as exc:
        report["failure_reason"] = f"ui import failed: {type(exc).__name__}: {exc}"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 1

    task_id = "ui_audit_task_20260517"
    run_label = f"{task_id}-20260517-121212"
    checks: List[Dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as td:
        sandbox_root = Path(td).resolve()
        _seed_task_sandbox(sandbox_root, task_id=task_id, run_label=run_label)
        # seed a recent task id for suggestion checks
        (sandbox_root / "runs" / "agent" / "ui_audit_recent_20260517").mkdir(parents=True, exist_ok=True)
        with mock.patch.object(ui_app_mod, "REPO_ROOT", sandbox_root):
            client = ui_app_mod.app.test_client()
            checks.append(_run_check_select_task_success(client, task_id=task_id, run_label=run_label))
            checks.append(_run_check_audit_artifact_actions_api(client, task_id=task_id))
            checks.append(_run_check_select_task_missing_hints(client))
            checks.append(_run_check_select_task_read_only_hints(client, task_id=task_id))
            checks.append(_run_check_invalid_task_id_rejected(client))
            checks.append(_run_check_audit_ui_tokens(client))

    report["checks"] = checks
    report["check_count"] = len(checks)
    report["passed_count"] = sum(1 for row in checks if isinstance(row, dict) and row.get("status") == "pass")
    report["failed_count"] = sum(1 for row in checks if isinstance(row, dict) and row.get("status") != "pass")
    report["status"] = "pass" if int(report["failed_count"]) == 0 else "fail"
    if report["status"] != "pass":
        report["failure_reason"] = "one_or_more_checks_failed"

    _apply_baseline(report, baseline_payload)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("status") == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

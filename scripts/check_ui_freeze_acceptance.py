#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import sys
import tarfile
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
    return json.loads(path.read_text(encoding="utf-8"))


def _check_result(name: str, *, ok: bool, details: Optional[Dict[str, Any]] = None, error: str = "") -> Dict[str, Any]:
    return {
        "name": name,
        "status": "pass" if ok else "fail",
        "error": str(error or ""),
        "details": details if isinstance(details, dict) else {},
    }


def _assert_payload_status(
    response_payload: Dict[str, Any],
    *,
    expected_status: str = "pass",
    expected_result_status: Optional[str] = None,
) -> Tuple[bool, str]:
    if str(response_payload.get("status") or "") != expected_status:
        return False, f"payload.status={response_payload.get('status')!r}, expected={expected_status!r}"
    if expected_result_status is None:
        return True, ""
    result = response_payload.get("result")
    if not isinstance(result, dict):
        return False, "payload.result is not object"
    actual = str(result.get("status") or "")
    if actual != expected_result_status:
        return False, f"payload.result.status={actual!r}, expected={expected_result_status!r}"
    return True, ""


def _run_check_full_pipeline_mock(ui_app_mod: Any, client: Any, root: Path) -> Dict[str, Any]:
    task_id = "ui_freeze_full_pipeline"
    run_dir = root / "runs" / "agent" / task_id
    run_dir.mkdir(parents=True, exist_ok=True)
    task_json = run_dir / "task.json"
    task_json.write_text(json.dumps({"task_id": task_id}, ensure_ascii=False) + "\n", encoding="utf-8")

    intake_ret = {
        "status": "pass",
        "returncode": 2,
        "result": {
            "task_id": task_id,
            "status": "need_user_input",
            "task_draft_path": str(run_dir / "task.draft.json"),
            "missing_fields": ["candidate_data"],
            "questions": ["candidate_data source?"],
        },
        "stderr": "",
    }
    approve_ret = {
        "status": "pass",
        "returncode": 0,
        "result": {
            "task_id": task_id,
            "status": "approved",
            "task_json_path": str(task_json),
            "plan_path": str(run_dir / "plan.json"),
        },
        "stderr": "",
    }
    run_ret = {
        "status": "pass",
        "returncode": 0,
        "result": {
            "task_id": task_id,
            "status": "success",
            "execution_path": str(run_dir / "execution.json"),
        },
        "stderr": "",
    }

    with (
        mock.patch.object(ui_app_mod, "_run_agent_intake", return_value=intake_ret) as intake_mock,
        mock.patch.object(ui_app_mod, "_run_agent_approve", return_value=approve_ret) as approve_mock,
        mock.patch.object(ui_app_mod, "_run_agent_run_json", return_value=run_ret) as run_mock,
    ):
        intake_resp = client.post(
            "/api/intake",
            json={
                "task_id": task_id,
                "request_text": "design 470nm and high plqy",
                "web_topk": 3,
                "web_search_enabled": True,
            },
        )
        approve_resp = client.post(
            "/api/approve",
            json={
                "task_json_path": str(task_json),
                "planner_provider": "rule_based_v1",
                "catalog_path": "configs/models/catalog.json",
            },
        )
        run_resp = client.post(
            "/api/run",
            json={
                "payload_text": json.dumps(
                    {
                        "task_id": task_id,
                        "request_text": "design 470nm and high plqy",
                    },
                    ensure_ascii=False,
                ),
                "planner_provider": "rule_based_v1",
                "catalog_path": "configs/models/catalog.json",
            },
        )

    intake_data = intake_resp.get_json() if intake_resp.is_json else {}
    approve_data = approve_resp.get_json() if approve_resp.is_json else {}
    run_data = run_resp.get_json() if run_resp.is_json else {}
    ok_i, err_i = _assert_payload_status(intake_data, expected_status="pass", expected_result_status="need_user_input")
    ok_a, err_a = _assert_payload_status(approve_data, expected_status="pass", expected_result_status="approved")
    ok_r, err_r = _assert_payload_status(run_data, expected_status="pass", expected_result_status="success")
    ok = (
        intake_resp.status_code == 200
        and approve_resp.status_code == 200
        and run_resp.status_code == 200
        and ok_i
        and ok_a
        and ok_r
        and intake_mock.call_count == 1
        and approve_mock.call_count == 1
        and run_mock.call_count == 1
    )
    err = "; ".join([x for x in [err_i, err_a, err_r] if x])
    if not ok and not err:
        err = "one or more api responses did not match expected status"
    return _check_result(
        "full_pipeline_mock",
        ok=ok,
        error=err,
        details={
            "http_codes": {
                "intake": intake_resp.status_code,
                "approve": approve_resp.status_code,
                "run": run_resp.status_code,
            },
            "mock_calls": {
                "intake": intake_mock.call_count,
                "approve": approve_mock.call_count,
                "run": run_mock.call_count,
            },
        },
    )


def _run_check_single_step_mock(ui_app_mod: Any, client: Any) -> Dict[str, Any]:
    step_ret = {
        "status": "pass",
        "returncode": 0,
        "result": {"status": "success", "operation": "clean_dataset"},
        "stderr": "",
    }
    with mock.patch.object(ui_app_mod, "_run_agent_step_json", return_value=step_ret) as step_mock:
        resp = client.post(
            "/api/run-step",
            json={
                "payload_text": json.dumps(
                    {
                        "task": {"task_id": "ui_freeze_step", "execution_mode": "single_step"},
                        "operation": "clean_dataset",
                        "args": {"input_csv": "/tmp/candidates.csv"},
                    },
                    ensure_ascii=False,
                ),
                "catalog_path": "configs/models/catalog.json",
            },
        )
    payload = resp.get_json() if resp.is_json else {}
    ok_s, err_s = _assert_payload_status(payload, expected_status="pass", expected_result_status="success")
    ok = resp.status_code == 200 and ok_s and step_mock.call_count == 1
    return _check_result(
        "single_step_mock",
        ok=ok,
        error=err_s if not ok else "",
        details={"http_code": resp.status_code, "mock_calls": step_mock.call_count},
    )


def _run_check_project_clone(client: Any) -> Dict[str, Any]:
    source_id = "ui_freeze_clone_src"
    target_id = "ui_freeze_clone_dst"
    c1 = client.post("/api/projects", json={"project_id": source_id, "title": "clone source"})
    c2 = client.post(
        f"/api/projects/{source_id}/upload-ref",
        json={"path": "/tmp/source.csv", "label": "source", "kind": "path_ref"},
    )
    clone = client.post(
        f"/api/projects/{source_id}/clone",
        json={"target_project_id": target_id},
    )
    hist = client.get(f"/api/projects/{target_id}/history?limit=300")
    hp = hist.get_json() if hist.is_json else {}
    messages = hp.get("messages") if isinstance(hp.get("messages"), list) else []
    attachments = hp.get("attachments") if isinstance(hp.get("attachments"), list) else []
    has_clone_note = any(
        "project cloned from" in str(m.get("content") or "").lower()
        for m in messages
        if isinstance(m, dict)
    )
    ok = (
        c1.status_code == 200
        and c2.status_code == 200
        and clone.status_code == 200
        and hist.status_code == 200
        and has_clone_note
        and len(attachments) >= 1
    )
    return _check_result(
        "project_clone",
        ok=ok,
        error="" if ok else "clone flow or cloned history assertions failed",
        details={
            "http_codes": {
                "create": c1.status_code,
                "attach": c2.status_code,
                "clone": clone.status_code,
                "history": hist.status_code,
            },
            "attachment_count": len(attachments),
            "has_clone_note": has_clone_note,
        },
    )


def _run_check_snapshot_roundtrip(client: Any) -> Dict[str, Any]:
    project_id = "ui_freeze_snapshot_proj"
    create_proj = client.post("/api/projects", json={"project_id": project_id, "title": "snapshot source"})
    create_snap = client.post(f"/api/projects/{project_id}/snapshots", json={"note": "freeze baseline"})
    create_payload = create_snap.get_json() if create_snap.is_json else {}
    snap = create_payload.get("snapshot") if isinstance(create_payload.get("snapshot"), dict) else {}
    snapshot_id = str(snap.get("snapshot_id") or "").strip()
    mutate_proj = client.post("/api/projects", json={"project_id": project_id, "title": "snapshot mutated"})
    restore = client.post(
        f"/api/projects/{project_id}/snapshots/{snapshot_id}/restore",
        json={"auto_snapshot_before": True, "restore_note": "rollback"},
    )
    restore_payload = restore.get_json() if restore.is_json else {}
    restored_project = restore_payload.get("project") if isinstance(restore_payload.get("project"), dict) else {}
    list_resp = client.get(f"/api/projects/{project_id}/snapshots?limit=30&offset=0")
    list_payload = list_resp.get_json() if list_resp.is_json else {}
    rows = list_payload.get("snapshots") if isinstance(list_payload.get("snapshots"), list) else []
    ids = [str(row.get("snapshot_id") or "") for row in rows if isinstance(row, dict)]
    ok = (
        create_proj.status_code == 200
        and create_snap.status_code == 200
        and bool(snapshot_id)
        and mutate_proj.status_code == 200
        and restore.status_code == 200
        and restored_project.get("title") == "snapshot source"
        and list_resp.status_code == 200
        and snapshot_id in ids
    )
    return _check_result(
        "snapshot_roundtrip",
        ok=ok,
        error="" if ok else "snapshot create/list/restore assertions failed",
        details={
            "snapshot_id": snapshot_id,
            "snapshot_count": len(ids),
            "restored_title": str(restored_project.get("title") or ""),
            "http_codes": {
                "create_project": create_proj.status_code,
                "create_snapshot": create_snap.status_code,
                "mutate_project": mutate_proj.status_code,
                "restore_snapshot": restore.status_code,
                "list_snapshots": list_resp.status_code,
            },
        },
    )


def _run_check_read_only_lock(client: Any) -> Dict[str, Any]:
    src = "ui_freeze_lock_src"
    dst = "ui_freeze_lock_dst"
    c1 = client.post("/api/projects", json={"project_id": src, "title": "read only source"})
    clone = client.post(
        f"/api/projects/{src}/clone",
        json={"target_project_id": dst, "target_options": {"project_read_only": True}},
    )
    send = client.post("/api/chat/send", json={"project_id": dst, "message": "design 470nm and high plqy"})
    upload = client.post(
        f"/api/projects/{dst}/upload-ref",
        json={"path": "/tmp/lock.csv", "label": "lock", "kind": "path_ref"},
    )
    send_payload = send.get_json() if send.is_json else {}
    upload_payload = upload.get_json() if upload.is_json else {}
    ok = (
        c1.status_code == 200
        and clone.status_code == 200
        and send.status_code == 409
        and upload.status_code == 409
        and str(send_payload.get("error") or "") == "project_read_only"
        and str(upload_payload.get("error") or "") == "project_read_only"
    )
    return _check_result(
        "read_only_lock",
        ok=ok,
        error="" if ok else "read-only project should block chat/upload with 409",
        details={
            "http_codes": {
                "create_source": c1.status_code,
                "clone_read_only": clone.status_code,
                "chat_send": send.status_code,
                "upload_ref": upload.status_code,
            }
        },
    )


def _run_check_bundle_download(client: Any, root: Path) -> Dict[str, Any]:
    task_id = "ui_freeze_bundle_case"
    run_dir = root / "runs" / "agent" / task_id
    run_dir.mkdir(parents=True, exist_ok=True)
    result_dir = root / "result" / f"{task_id}-20260515-010101"
    logging_dir = root / "logging" / f"{task_id}-20260515-010101"
    rank_dir = root / "runs" / f"agent_rank_{task_id}_20260515T010101.000000+0000"
    result_dir.mkdir(parents=True, exist_ok=True)
    logging_dir.mkdir(parents=True, exist_ok=True)
    rank_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "execution.json").write_text(
        json.dumps(
            {
                "task_id": task_id,
                "status": "success",
                "records": [
                    {
                        "name": "make_report",
                        "status": "success",
                        "result": {
                            "latest_run_dir": str(rank_dir),
                            "report": str(rank_dir / "06_report.md"),
                        },
                    }
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
                "artifacts": {
                    "final_output": str(rank_dir / "06_report.md"),
                },
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "task_state.json").write_text(json.dumps({"task_id": task_id}, ensure_ascii=False) + "\n", encoding="utf-8")
    (run_dir / "plan.json").write_text(json.dumps({"summary": "ok"}, ensure_ascii=False) + "\n", encoding="utf-8")
    (run_dir / "tool_state.json").write_text(json.dumps({"ok": True}, ensure_ascii=False) + "\n", encoding="utf-8")
    (result_dir / "metadata.json").write_text(json.dumps({"task_id": task_id}, ensure_ascii=False) + "\n", encoding="utf-8")
    (logging_dir / "task.json").write_text(json.dumps({"task_id": task_id}, ensure_ascii=False) + "\n", encoding="utf-8")
    (rank_dir / "06_report.md").write_text("# report\n", encoding="utf-8")

    resp = client.get(f"/api/task/{task_id}/bundle")
    if resp.status_code != 200:
        return _check_result(
            "bundle_download",
            ok=False,
            error=f"bundle endpoint returned status {resp.status_code}",
            details={"http_code": resp.status_code},
        )

    names: List[str] = []
    try:
        with tarfile.open(fileobj=io.BytesIO(resp.get_data()), mode="r:gz") as tf:
            names = tf.getnames()
    except Exception as exc:
        return _check_result(
            "bundle_download",
            ok=False,
            error=f"failed to parse tar.gz: {type(exc).__name__}: {exc}",
            details={},
        )

    required_suffixes = [
        "manifest.json",
        f"runs/agent/{task_id}/execution.json",
        f"result/{result_dir.name}/metadata.json",
        f"logging/{logging_dir.name}/task.json",
        f"runs/{rank_dir.name}/06_report.md",
    ]
    missing = [suffix for suffix in required_suffixes if not any(name.endswith(suffix) for name in names)]
    ok = len(missing) == 0
    return _check_result(
        "bundle_download",
        ok=ok,
        error="" if ok else f"bundle missing expected files: {missing}",
        details={
            "http_code": resp.status_code,
            "file_count": len(names),
            "missing_suffixes": missing,
        },
    )


def _run_check_batch_compare_api(client: Any) -> Dict[str, Any]:
    project_id = "ui_freeze_batch_compare_proj"
    create = client.post("/api/projects", json={"project_id": project_id, "title": "batch compare smoke"})
    save_a = client.post(
        f"/api/projects/{project_id}/batch-export",
        json={
            "payload": {
                "status": "pass",
                "action": "batch_summary",
                "limit": 2,
                "count": 2,
                "rows": [
                    {"task_id": "ui_bc_t1", "project_id": project_id, "release_gate_status": "pass"},
                    {"task_id": "ui_bc_t2", "project_id": project_id, "release_gate_status": "fail"},
                ],
                "results": [
                    {"task_id": "ui_bc_t1", "project_id": project_id, "http_status": 200, "data": {"status": "pass"}},
                    {"task_id": "ui_bc_t2", "project_id": project_id, "http_status": 404, "data": {"status": "missing"}},
                ],
                "created_at": "2026-05-17T09:00:00+08:00",
            }
        },
    )
    save_b = client.post(
        f"/api/projects/{project_id}/batch-export",
        json={
            "payload": {
                "status": "pass",
                "action": "batch_summary",
                "limit": 1,
                "count": 1,
                "rows": [
                    {"task_id": "ui_bc_t1", "project_id": project_id, "release_gate_status": "missing"},
                ],
                "results": [
                    {"task_id": "ui_bc_t1", "project_id": project_id, "http_status": 200, "data": {"status": "pass"}},
                ],
                "created_at": "2026-05-17T08:30:00+08:00",
            }
        },
    )
    listed = client.get(f"/api/projects/{project_id}/batch-exports?limit=10")
    listed_payload = listed.get_json() if listed.is_json else {}
    exports = listed_payload.get("exports") if isinstance(listed_payload.get("exports"), list) else []
    if len(exports) < 2:
        return _check_result(
            "batch_compare_api",
            ok=False,
            error="not enough batch exports saved for compare",
            details={
                "http_codes": {
                    "create": create.status_code,
                    "save_a": save_a.status_code,
                    "save_b": save_b.status_code,
                    "list": listed.status_code,
                },
                "export_count": len(exports),
            },
        )
    export_a = str(exports[0].get("export_id") or "")
    export_b = str(exports[1].get("export_id") or "")
    compare = client.get(
        f"/api/projects/{project_id}/batch-exports/compare?primary_export_id={export_a}&other_export_id={export_b}"
    )
    compare_payload = compare.get_json() if compare.is_json else {}
    gate_diff = compare_payload.get("release_gate_diff") if isinstance(compare_payload.get("release_gate_diff"), dict) else {}
    diff = compare_payload.get("diff") if isinstance(compare_payload.get("diff"), dict) else {}
    changed_rows = diff.get("changed") if isinstance(diff.get("changed"), list) else []
    release_changed = [
        row
        for row in changed_rows
        if isinstance(row, dict) and "release_gate" in str(row.get("path") or "").lower()
    ]
    status_other_resp = client.get(f"/api/projects/{project_id}/batch-exports?release_gate_status=other")
    status_other_payload = status_other_resp.get_json() if status_other_resp.is_json else {}
    status_other_rows = (
        status_other_payload.get("exports")
        if isinstance(status_other_payload.get("exports"), list)
        else []
    )
    ok = (
        create.status_code == 200
        and save_a.status_code == 200
        and save_b.status_code == 200
        and listed.status_code == 200
        and compare.status_code == 200
        and str(compare_payload.get("status") or "") == "pass"
        and isinstance(gate_diff.get("delta"), dict)
        and len(release_changed) >= 1
        and status_other_resp.status_code == 200
        and len(status_other_rows) >= 1
    )
    return _check_result(
        "batch_compare_api",
        ok=ok,
        error="" if ok else "batch compare endpoint assertions failed",
        details={
            "http_codes": {
                "create": create.status_code,
                "save_a": save_a.status_code,
                "save_b": save_b.status_code,
                "list": listed.status_code,
                "compare": compare.status_code,
                "status_other": status_other_resp.status_code,
            },
            "export_ids": [export_a, export_b],
            "release_changed_count": len(release_changed),
            "gate_delta": gate_diff.get("delta") if isinstance(gate_diff.get("delta"), dict) else {},
            "status_other_count": len(status_other_rows),
        },
    )


def _run_check_compare_ui_controls(client: Any) -> Dict[str, Any]:
    resp = client.get("/")
    html = resp.get_data(as_text=True) if resp.status_code == 200 else ""
    required_tokens = [
        "batch_compare_toggle_btn",
        "batch_compare_path_filter",
        "session_filter_readiness",
        "session_filter_resume_mode",
        "quickFilterByReadiness('fail')",
        "quickFilterByReadiness('warn')",
        "quickFilterByReadiness('pass')",
        "quickFilterByResume('full_skip')",
        "quickFilterByResume('partial_rerun')",
        "quickFilterByResume('no_resume')",
        "readiness_fail_first",
        "projectReadinessStatus(",
        "batch_history_readiness_filter",
        "batch_replay_readiness_mode",
        "pending_state_card",
        "pendingStateContinue(",
        "/api/chat/pending-continue",
        "normalizeResumeVisibility(",
        "formatResumeVisibilityLine(",
        "formatResumeVisibilityCompact(",
        "renderResumeDiagnostics(",
        "resume_diag_box",
        "resume_diag_steps",
        "resume_visibility",
        "resume(skip/partial/full/no)",
        "normalizeReadinessStats(",
        "aggregateReadinessStats(",
        "project_batch_compare_details",
        "project_batch_compare_paths",
        "project_batch_compare_selected_diff",
        "renderBatchCompareChangedPaths(",
        "renderBatchCompareSelectedDiff(",
        "copyBatchComparePath(",
        "copyBatchCompareDetails(",
        "downloadLatestBatchCompare('json')",
        "downloadLatestBatchCompare('txt')",
        "compareBatchExportsById()",
    ]
    missing = [token for token in required_tokens if token not in html]
    ok = resp.status_code == 200 and len(missing) == 0
    return _check_result(
        "compare_ui_controls",
        ok=ok,
        error="" if ok else f"missing compare ui tokens: {missing}",
        details={"http_code": resp.status_code, "missing_tokens": missing},
    )


def _load_baseline(path: Path) -> Tuple[Optional[Dict[str, Any]], str]:
    if not path.exists():
        return None, f"baseline file missing: {path}"
    try:
        payload = _read_json(path)
    except Exception as exc:
        return None, f"baseline parse failed: {type(exc).__name__}: {exc}"
    if not isinstance(payload, dict):
        return None, "baseline payload must be json object"
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
    parser = argparse.ArgumentParser(description="Run frozen UI acceptance checks and validate against baseline contract")
    parser.add_argument("--workspace-root", default=".", help="workspace root path")
    parser.add_argument("--out", default="runs/ci/ui_freeze_acceptance.json", help="output json path")
    parser.add_argument(
        "--baseline",
        default="configs/acceptance/ui_freeze_acceptance_baseline.json",
        help="baseline contract json path",
    )
    args = parser.parse_args()

    workspace_root = Path(args.workspace_root).resolve()
    out_path = _resolve_path(workspace_root, args.out)
    baseline_path = _resolve_path(workspace_root, args.baseline)
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

    checks: List[Dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as td:
        sandbox_root = Path(td).resolve()
        with mock.patch.object(ui_app_mod, "REPO_ROOT", sandbox_root):
            client = ui_app_mod.app.test_client()
            checks.append(_run_check_full_pipeline_mock(ui_app_mod, client, sandbox_root))
            checks.append(_run_check_single_step_mock(ui_app_mod, client))
            checks.append(_run_check_project_clone(client))
            checks.append(_run_check_snapshot_roundtrip(client))
            checks.append(_run_check_read_only_lock(client))
            checks.append(_run_check_bundle_download(client, sandbox_root))
            checks.append(_run_check_batch_compare_api(client))
            checks.append(_run_check_compare_ui_controls(client))

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

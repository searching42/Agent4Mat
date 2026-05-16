#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_path(raw: str, workspace_root: Path) -> Path:
    p = Path(str(raw or "").strip())
    if not p.is_absolute():
        p = (workspace_root / p).resolve()
    else:
        p = p.resolve()
    return p


def _to_int(value: Any, *, default: int = -1) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _check_bundle(
    *,
    workspace_root: Path,
    base_task_id: str,
    baseline_summary_path: Path,
    archive_manifest_path: Path,
    require_tar_gz: bool,
) -> Dict[str, Any]:
    checks = []
    failures = []

    baseline_status = ""
    baseline_run_count = -1
    baseline_runs_checked = 0
    archive_status = ""
    missing_required_count = -1

    if baseline_summary_path.exists():
        try:
            baseline_payload = _load_json(baseline_summary_path)
            baseline_status = str(baseline_payload.get("status") or "")
            if baseline_status == "pass":
                checks.append({"name": "baseline_summary_status", "status": "pass", "message": "baseline_summary.json status=pass"})
            else:
                failures.append(
                    {"name": "baseline_summary_status", "message": f"baseline_summary.json status is not pass: {baseline_status}"}
                )

            baseline_run_count = _to_int(baseline_payload.get("run_count"), default=-1)
            runs = baseline_payload.get("runs") if isinstance(baseline_payload.get("runs"), list) else []
            if baseline_run_count >= 0 and baseline_run_count == len(runs):
                checks.append(
                    {
                        "name": "baseline_summary_run_count",
                        "status": "pass",
                        "message": f"baseline run_count matches runs length: {baseline_run_count}",
                    }
                )
            else:
                failures.append(
                    {
                        "name": "baseline_summary_run_count",
                        "message": f"baseline run_count mismatch: run_count={baseline_run_count} len(runs)={len(runs)}",
                    }
                )

            for idx, item in enumerate(runs, start=1):
                if not isinstance(item, dict):
                    failures.append({"name": "baseline_run_row", "message": f"run row #{idx} is not object"})
                    continue
                baseline_runs_checked += 1
                task_id = str(item.get("task_id") or f"run_{idx}")

                strict_raw = str(item.get("strict_summary") or "").strip()
                release_raw = str(item.get("release_evidence_json") or "").strip()
                if not strict_raw:
                    failures.append({"name": f"{task_id}:strict_summary", "message": "strict_summary is missing"})
                    continue
                if not release_raw:
                    failures.append({"name": f"{task_id}:release_evidence_json", "message": "release_evidence_json is missing"})
                    continue

                strict_path = _resolve_path(strict_raw, workspace_root)
                release_path = _resolve_path(release_raw, workspace_root)
                if not strict_path.exists():
                    failures.append({"name": f"{task_id}:strict_summary", "message": f"strict summary not found: {strict_path}"})
                    continue
                if not release_path.exists():
                    failures.append({"name": f"{task_id}:release_evidence_json", "message": f"release evidence not found: {release_path}"})
                    continue

                strict_payload = _load_json(strict_path)
                release_payload = _load_json(release_path)

                guardrails_strict_status = str(
                    item.get("guardrails_strict_status")
                    or strict_payload.get("guardrails_strict_status")
                    or ""
                ).strip()
                evaluation_failed_count = _to_int(
                    item.get("evaluation_failed_count")
                    if item.get("evaluation_failed_count") is not None
                    else strict_payload.get("evaluation_failed_count"),
                    default=-1,
                )
                guardrails_failed_count = _to_int(
                    item.get("guardrails_failed_count")
                    if item.get("guardrails_failed_count") is not None
                    else strict_payload.get("guardrails_failed_count"),
                    default=-1,
                )

                if guardrails_strict_status == "pass":
                    checks.append(
                        {
                            "name": f"{task_id}:guardrails_strict_status",
                            "status": "pass",
                            "message": "guardrails_strict_status=pass",
                        }
                    )
                else:
                    failures.append(
                        {
                            "name": f"{task_id}:guardrails_strict_status",
                            "message": f"guardrails_strict_status is not pass: {guardrails_strict_status}",
                        }
                    )

                if evaluation_failed_count == 0:
                    checks.append(
                        {
                            "name": f"{task_id}:evaluation_failed_count",
                            "status": "pass",
                            "message": "evaluation_failed_count=0",
                        }
                    )
                else:
                    failures.append(
                        {
                            "name": f"{task_id}:evaluation_failed_count",
                            "message": f"evaluation_failed_count is not 0: {evaluation_failed_count}",
                        }
                    )

                if guardrails_failed_count == 0:
                    checks.append(
                        {
                            "name": f"{task_id}:guardrails_failed_count",
                            "status": "pass",
                            "message": "guardrails_failed_count=0",
                        }
                    )
                else:
                    failures.append(
                        {
                            "name": f"{task_id}:guardrails_failed_count",
                            "message": f"guardrails_failed_count is not 0: {guardrails_failed_count}",
                        }
                    )

                release_checks = release_payload.get("checks") if isinstance(release_payload.get("checks"), dict) else {}
                release_guardrails_ok = bool(release_checks.get("guardrails_strict_status_pass", False))
                release_eval_zero = bool(release_checks.get("evaluation_failure_diag_zero", False))
                release_guard_zero = bool(release_checks.get("guardrails_failure_diag_zero", False))
                if release_guardrails_ok and release_eval_zero and release_guard_zero:
                    checks.append(
                        {
                            "name": f"{task_id}:release_evidence_diagnostics",
                            "status": "pass",
                            "message": "release evidence diagnostics checks all pass",
                        }
                    )
                else:
                    failures.append(
                        {
                            "name": f"{task_id}:release_evidence_diagnostics",
                            "message": (
                                "release evidence diagnostics checks failed: "
                                f"guardrails_strict_status_pass={release_guardrails_ok}, "
                                f"evaluation_failure_diag_zero={release_eval_zero}, "
                                f"guardrails_failure_diag_zero={release_guard_zero}"
                            ),
                        }
                    )
        except Exception as exc:
            failures.append({"name": "baseline_summary_parse", "message": f"failed to parse baseline summary: {exc}"})
    else:
        failures.append({"name": "baseline_summary_exists", "message": f"missing baseline summary: {baseline_summary_path}"})

    if archive_manifest_path.exists():
        try:
            archive_payload = _load_json(archive_manifest_path)
            archive_status = str(archive_payload.get("status") or "")
            missing_required_count = int(archive_payload.get("missing_required_count", -1))
            if archive_status == "pass":
                checks.append({"name": "archive_manifest_status", "status": "pass", "message": "archive_manifest.json status=pass"})
            else:
                failures.append(
                    {"name": "archive_manifest_status", "message": f"archive_manifest.json status is not pass: {archive_status}"}
                )
            if missing_required_count == 0:
                checks.append({"name": "archive_manifest_missing_required_count", "status": "pass", "message": "missing_required_count=0"})
            else:
                failures.append(
                    {
                        "name": "archive_manifest_missing_required_count",
                        "message": f"archive manifest missing_required_count is not 0: {missing_required_count}",
                    }
                )
        except Exception as exc:
            failures.append({"name": "archive_manifest_parse", "message": f"failed to parse archive manifest: {exc}"})
    else:
        failures.append({"name": "archive_manifest_exists", "message": f"missing archive manifest: {archive_manifest_path}"})

    tar_gz_path = (workspace_root / "runs" / "archive" / f"{base_task_id}.tar.gz").resolve()
    if require_tar_gz:
        if tar_gz_path.exists():
            checks.append({"name": "archive_tar_gz_exists", "status": "pass", "message": f"tar.gz exists: {tar_gz_path}"})
        else:
            failures.append({"name": "archive_tar_gz_exists", "message": f"missing tar.gz package: {tar_gz_path}"})

    status = "pass" if not failures else "fail"
    return {
        "status": status,
        "base_task_id": base_task_id,
        "workspace_root": str(workspace_root),
        "baseline_summary_path": str(baseline_summary_path),
        "baseline_run_count": baseline_run_count,
        "baseline_runs_checked": baseline_runs_checked,
        "archive_manifest_path": str(archive_manifest_path),
        "archive_tar_gz_path": str(tar_gz_path),
        "baseline_status": baseline_status,
        "archive_status": archive_status,
        "archive_missing_required_count": missing_required_count,
        "require_tar_gz": bool(require_tar_gz),
        "checks": checks,
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate strict real-chain release bundle artifacts")
    parser.add_argument("--workspace-root", default=".", help="Workspace root")
    parser.add_argument("--base-task-id", required=True, help="Baseline task id")
    parser.add_argument("--baseline-summary", default="", help="Path to baseline_summary.json")
    parser.add_argument("--archive-manifest", default="", help="Path to archive_manifest.json")
    parser.add_argument("--require-tar-gz", action="store_true", help="Require runs/archive/<base_task_id>.tar.gz to exist")
    args = parser.parse_args()

    workspace_root = Path(args.workspace_root).resolve()
    base_task_id = str(args.base_task_id or "").strip()
    if not base_task_id:
        print(json.dumps({"status": "fail", "message": "base-task-id is required"}, ensure_ascii=False))
        return 2

    baseline_summary_path = (
        (workspace_root / str(args.baseline_summary).strip()).resolve()
        if str(args.baseline_summary or "").strip()
        else (workspace_root / "runs" / "agent" / base_task_id / "baseline_summary.json").resolve()
    )
    archive_manifest_path = (
        (workspace_root / str(args.archive_manifest).strip()).resolve()
        if str(args.archive_manifest or "").strip()
        else (workspace_root / "runs" / "archive" / base_task_id / "archive_manifest.json").resolve()
    )

    report = _check_bundle(
        workspace_root=workspace_root,
        base_task_id=base_task_id,
        baseline_summary_path=baseline_summary_path,
        archive_manifest_path=archive_manifest_path,
        require_tar_gz=bool(args.require_tar_gz),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("status") == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

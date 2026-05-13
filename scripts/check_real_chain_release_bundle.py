#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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

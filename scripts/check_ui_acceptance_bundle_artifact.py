#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


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


def _load_json_object(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("summary json must be an object")
    return payload


def _failure_category(name: str) -> str:
    key = str(name or "").strip()
    if not key:
        return "unknown"
    if key.endswith("_exists"):
        return "missing_artifact"
    if key.endswith("_parse"):
        return "invalid_json"
    if key == "summary_required_keys":
        return "schema_missing_keys"
    if key.endswith("_type"):
        return "schema_type_error"
    if key.endswith("_count"):
        return "schema_numeric_error"
    if key == "summary_status_enum":
        return "schema_enum_error"
    return "unknown"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate ui_acceptance_bundle_summary artifact schema")
    p.add_argument("--workspace-root", default=".", help="Workspace root")
    p.add_argument("--summary-json", default="runs/ci/ui_acceptance_bundle_summary.json", help="Summary JSON path")
    p.add_argument("--summary-md", default="runs/ci/ui_acceptance_bundle_summary.md", help="Summary markdown path")
    p.add_argument(
        "--out-json",
        default="runs/ci/ui_acceptance_bundle_artifact_verify.json",
        help="Verification report output JSON path",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    workspace_root = Path(args.workspace_root).resolve()
    summary_json_path = _resolve_path(str(args.summary_json or ""), workspace_root)
    summary_md_path = _resolve_path(str(args.summary_md or ""), workspace_root)
    out_json_path = _resolve_path(str(args.out_json or ""), workspace_root)

    checks: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    payload: Dict[str, Any] = {}
    summary_status = ""
    summary_check_count = -1
    summary_failure_count = -1

    if not summary_json_path.exists():
        failures.append(
            {
                "name": "summary_json_exists",
                "message": f"missing summary json artifact: {summary_json_path}",
            }
        )
    else:
        try:
            payload = _load_json_object(summary_json_path)
            checks.append({"name": "summary_json_parse", "message": "summary json parsed"})
        except Exception as exc:
            failures.append({"name": "summary_json_parse", "message": f"{summary_json_path}: {exc}"})

    if not summary_md_path.exists():
        failures.append(
            {
                "name": "summary_md_exists",
                "message": f"missing summary markdown artifact: {summary_md_path}",
            }
        )
    else:
        text = summary_md_path.read_text(encoding="utf-8")
        if str(text).strip():
            checks.append({"name": "summary_md_non_empty", "message": "summary markdown is non-empty"})
        else:
            failures.append({"name": "summary_md_non_empty", "message": "summary markdown is empty"})

    if isinstance(payload, dict) and payload:
        required = [
            "status",
            "generated_at",
            "components",
            "checks",
            "failures",
            "check_count",
            "failure_count",
        ]
        missing = [k for k in required if k not in payload]
        if missing:
            failures.append({"name": "summary_required_keys", "message": f"missing summary keys: {missing}"})
        else:
            checks.append({"name": "summary_required_keys", "message": "required keys present"})

        summary_status = str(payload.get("status") or "")
        if summary_status in {"pass", "fail"}:
            checks.append({"name": "summary_status_enum", "message": f"status={summary_status}"})
        else:
            failures.append({"name": "summary_status_enum", "message": f"unexpected summary status: {summary_status!r}"})

        for key in ("components", "checks", "failures"):
            val = payload.get(key)
            if isinstance(val, list):
                checks.append({"name": f"summary_{key}_type", "message": f"{key} is list"})
            else:
                failures.append({"name": f"summary_{key}_type", "message": f"{key} must be a list"})

        summary_check_count = _to_int(payload.get("check_count"), default=-1)
        summary_failure_count = _to_int(payload.get("failure_count"), default=-1)
        if summary_check_count >= 0:
            checks.append({"name": "summary_check_count", "message": f"check_count={summary_check_count}"})
        else:
            failures.append({"name": "summary_check_count", "message": "check_count must be an integer >= 0"})
        if summary_failure_count >= 0:
            checks.append({"name": "summary_failure_count", "message": f"failure_count={summary_failure_count}"})
        else:
            failures.append({"name": "summary_failure_count", "message": "failure_count must be an integer >= 0"})

    status = "pass" if not failures else "fail"
    failure_categories = sorted({_failure_category(str(row.get("name") or "")) for row in failures if isinstance(row, dict)})
    failure_preview = [
        {"name": str(row.get("name") or ""), "message": str(row.get("message") or "")}
        for row in failures[:3]
        if isinstance(row, dict)
    ]
    report: Dict[str, Any] = {
        "status": status,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "workspace_root": str(workspace_root),
        "summary_json_path": str(summary_json_path),
        "summary_md_path": str(summary_md_path),
        "summary_status": summary_status,
        "summary_check_count": summary_check_count,
        "summary_failure_count": summary_failure_count,
        "failure_categories": failure_categories,
        "failure_preview": failure_preview,
        "checks": checks,
        "failures": failures,
        "check_count": len(checks),
        "failure_count": len(failures),
    }

    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    out_json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

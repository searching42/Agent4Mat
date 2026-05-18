#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def _resolve_path(raw: str, workspace_root: Path) -> Path:
    p = Path(str(raw or "").strip())
    if not p.is_absolute():
        p = (workspace_root / p).resolve()
    else:
        p = p.resolve()
    return p


def _load_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("json payload must be an object")
    return payload


def _to_int(value: Any, *, default: int = -1) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _component_row(name: str, path: Path, payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    obj = payload if isinstance(payload, dict) else {}
    return {
        "name": name,
        "path": str(path),
        "exists": path.exists(),
        "status": str(obj.get("status") or "") if obj else "",
        "check_count": _to_int(obj.get("check_count"), default=-1) if obj else -1,
        "failed_count": _to_int(obj.get("failed_count"), default=-1) if obj else -1,
        "failure_count": _to_int(obj.get("failure_count"), default=-1) if obj else -1,
        "warning_count": _to_int(obj.get("warning_count"), default=-1) if obj else -1,
    }


def _build_md(report: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# UI Acceptance Bundle Summary")
    lines.append("")
    lines.append(f"- generated_at: `{report.get('generated_at')}`")
    lines.append(f"- status: `{report.get('status')}`")
    lines.append(f"- workspace_root: `{report.get('workspace_root')}`")
    lines.append("")
    lines.append("## Components")
    rows = report.get("components") if isinstance(report.get("components"), list) else []
    if rows:
        for row in rows:
            if not isinstance(row, dict):
                continue
            lines.append(
                f"- {row.get('name')}: status=`{row.get('status')}` exists=`{row.get('exists')}` "
                f"check_count=`{row.get('check_count')}` failed_count=`{row.get('failed_count')}` "
                f"failure_count=`{row.get('failure_count')}` warning_count=`{row.get('warning_count')}`"
            )
    else:
        lines.append("- none")
    lines.append("")
    lines.append("## Checks")
    checks = report.get("checks") if isinstance(report.get("checks"), list) else []
    if checks:
        for row in checks:
            if not isinstance(row, dict):
                continue
            lines.append(f"- {row.get('name')}: {row.get('message')}")
    else:
        lines.append("- none")
    lines.append("")
    lines.append("## Failures")
    failures = report.get("failures") if isinstance(report.get("failures"), list) else []
    if failures:
        for row in failures:
            if not isinstance(row, dict):
                continue
            lines.append(f"- {row.get('name')}: {row.get('message')}")
    else:
        lines.append("- none")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate UI acceptance reports into one pass/fail verdict")
    p.add_argument("--workspace-root", default=".", help="Workspace root")
    p.add_argument("--ui-freeze-json", default="runs/ci/ui_freeze_acceptance.json")
    p.add_argument("--ui-audit-json", default="runs/ci/ui_audit_acceptance.json")
    p.add_argument("--ui-release-json", default="runs/ci/ui_release_readiness.json")
    p.add_argument("--out-json", default="runs/ci/ui_acceptance_bundle_summary.json")
    p.add_argument("--out-md", default="runs/ci/ui_acceptance_bundle_summary.md")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    workspace_root = Path(args.workspace_root).resolve()
    now = datetime.now(timezone.utc).isoformat()

    checks: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    components: List[Dict[str, Any]] = []

    freeze_path = _resolve_path(str(args.ui_freeze_json or ""), workspace_root)
    audit_path = _resolve_path(str(args.ui_audit_json or ""), workspace_root)
    release_path = _resolve_path(str(args.ui_release_json or ""), workspace_root)

    def _check_component(name: str, path: Path) -> Optional[Dict[str, Any]]:
        payload: Optional[Dict[str, Any]] = None
        if not path.exists():
            failures.append({"name": f"{name}_exists", "message": f"missing required report: {path}"})
            components.append(_component_row(name, path, payload))
            return None
        try:
            payload = _load_json(path)
        except Exception as exc:
            failures.append({"name": f"{name}_parse", "message": f"{path}: {exc}"})
            components.append(_component_row(name, path, payload))
            return None

        components.append(_component_row(name, path, payload))
        status = str(payload.get("status") or "")
        if status == "pass":
            checks.append({"name": f"{name}_status", "message": "status=pass"})
        else:
            failures.append({"name": f"{name}_status", "message": f"status is not pass: {status or 'missing'}"})

        # mixed historical fields
        failed_count = _to_int(payload.get("failed_count"), default=-1)
        failure_count = _to_int(payload.get("failure_count"), default=-1)
        if failed_count >= 0 and failed_count != 0:
            failures.append({"name": f"{name}_failed_count", "message": f"failed_count={failed_count}"})
        elif failure_count >= 0 and failure_count != 0:
            failures.append({"name": f"{name}_failure_count", "message": f"failure_count={failure_count}"})
        else:
            checks.append(
                {
                    "name": f"{name}_counts",
                    "message": f"failed_count={failed_count} failure_count={failure_count}",
                }
            )
        return payload

    freeze_payload = _check_component("ui_freeze_acceptance", freeze_path)
    audit_payload = _check_component("ui_audit_acceptance", audit_path)
    release_payload = _check_component("ui_release_readiness", release_path)

    if isinstance(release_payload, dict):
        gate_rows = release_payload.get("gate_reports") if isinstance(release_payload.get("gate_reports"), list) else []
        by_name = {
            str(row.get("name") or ""): row
            for row in gate_rows
            if isinstance(row, dict) and str(row.get("name") or "").strip()
        }
        for gate in ("ui_stability_smoke", "ui_freeze_acceptance", "ui_audit_acceptance"):
            row = by_name.get(gate)
            if not isinstance(row, dict):
                failures.append({"name": f"ui_release_readiness_gate_{gate}_exists", "message": "missing gate report"})
                continue
            status = str(row.get("status") or "")
            if status == "pass":
                checks.append({"name": f"ui_release_readiness_gate_{gate}_status", "message": "status=pass"})
            else:
                failures.append(
                    {
                        "name": f"ui_release_readiness_gate_{gate}_status",
                        "message": f"status is not pass: {status or 'missing'}",
                    }
                )

    status = "pass" if not failures else "fail"
    report: Dict[str, Any] = {
        "status": status,
        "generated_at": now,
        "workspace_root": str(workspace_root),
        "components": components,
        "checks": checks,
        "failures": failures,
        "check_count": len(checks),
        "failure_count": len(failures),
    }

    out_json = _resolve_path(str(args.out_json or ""), workspace_root)
    out_md = _resolve_path(str(args.out_md or ""), workspace_root)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    out_md.write_text(_build_md(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

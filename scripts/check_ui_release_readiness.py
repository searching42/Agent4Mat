#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


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
        raise ValueError("report json must be an object")
    return payload


def _to_int(value: Any, *, default: int = -1) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _parse_timestamp(raw: Any) -> Tuple[Optional[datetime], str]:
    text = str(raw or "").strip()
    if not text:
        return None, "generated_at is missing"
    normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        dt = datetime.fromisoformat(normalized)
    except Exception:
        return None, f"generated_at is invalid: {text}"
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc), ""


def _check_gate_report(
    *,
    name: str,
    path: Path,
    payload: Dict[str, Any],
    now_utc: datetime,
    max_age_hours: int,
    required: bool,
) -> Dict[str, Any]:
    checks: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []

    status = str(payload.get("status") or "").strip()
    if status == "pass":
        checks.append({"name": f"{name}_status", "message": "status=pass"})
    else:
        target = failures if required else warnings
        target.append({"name": f"{name}_status", "message": f"status is not pass: {status or 'missing'}"})

    check_count = _to_int(payload.get("check_count"), default=-1)
    failed_count = _to_int(payload.get("failed_count"), default=-1)
    if check_count >= 0 and failed_count == 0:
        checks.append({"name": f"{name}_counts", "message": f"check_count={check_count} failed_count={failed_count}"})
    else:
        target = failures if required else warnings
        target.append(
            {
                "name": f"{name}_counts",
                "message": f"unexpected counts: check_count={check_count}, failed_count={failed_count}",
            }
        )

    ts, ts_error = _parse_timestamp(payload.get("generated_at"))
    if ts is None:
        target = failures if required else warnings
        target.append({"name": f"{name}_generated_at", "message": ts_error})
    elif max_age_hours > 0:
        age_seconds = max((now_utc - ts).total_seconds(), 0.0)
        age_hours = age_seconds / 3600.0
        if age_hours <= float(max_age_hours):
            checks.append(
                {
                    "name": f"{name}_freshness",
                    "message": f"age_hours={age_hours:.2f} <= max_age_hours={max_age_hours}",
                }
            )
        else:
            target = failures if required else warnings
            target.append(
                {
                    "name": f"{name}_freshness",
                    "message": f"age_hours={age_hours:.2f} > max_age_hours={max_age_hours}",
                }
            )

    return {
        "name": name,
        "path": str(path),
        "required": bool(required),
        "status": status or "missing",
        "checks": checks,
        "failures": failures,
        "warnings": warnings,
    }


def _build_markdown(report: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# UI Release Readiness")
    lines.append("")
    lines.append(f"- generated_at: `{report.get('generated_at')}`")
    lines.append(f"- status: `{report.get('status')}`")
    lines.append(f"- workspace_root: `{report.get('workspace_root')}`")
    lines.append(f"- max_age_hours: `{report.get('max_age_hours')}`")
    lines.append(f"- require_freeze_report: `{report.get('require_freeze_report')}`")
    lines.append(f"- require_audit_report: `{report.get('require_audit_report')}`")
    lines.append("")

    gate_rows = report.get("gate_reports") if isinstance(report.get("gate_reports"), list) else []
    lines.append("## Gate Reports")
    if gate_rows:
        for row in gate_rows:
            if not isinstance(row, dict):
                continue
            lines.append(f"- {row.get('name')}: status=`{row.get('status')}` required=`{row.get('required')}` path=`{row.get('path')}`")
    else:
        lines.append("- none")
    lines.append("")

    checks = report.get("checks") if isinstance(report.get("checks"), list) else []
    lines.append("## Checks")
    if checks:
        for row in checks:
            if not isinstance(row, dict):
                continue
            lines.append(f"- {row.get('name')}: {row.get('message')}")
    else:
        lines.append("- none")
    lines.append("")

    warnings = report.get("warnings") if isinstance(report.get("warnings"), list) else []
    lines.append("## Warnings")
    if warnings:
        for row in warnings:
            if not isinstance(row, dict):
                continue
            lines.append(f"- {row.get('name')}: {row.get('message')}")
    else:
        lines.append("- none")
    lines.append("")

    failures = report.get("failures") if isinstance(report.get("failures"), list) else []
    lines.append("## Failures")
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
    p = argparse.ArgumentParser(description="Validate UI release readiness from generated acceptance reports")
    p.add_argument("--workspace-root", default=".", help="Workspace root")
    p.add_argument("--ui-stability-json", default="runs/ci/ui_stability_smoke.json", help="Path to ui_stability_smoke.json")
    p.add_argument("--ui-freeze-json", default="runs/ci/ui_freeze_acceptance.json", help="Path to ui_freeze_acceptance.json")
    p.add_argument("--ui-audit-json", default="runs/ci/ui_audit_acceptance.json", help="Path to ui_audit_acceptance.json")
    p.add_argument("--require-freeze-report", action="store_true", help="Require ui_freeze_acceptance.json to exist and pass")
    p.add_argument("--require-audit-report", action="store_true", help="Require ui_audit_acceptance.json to exist and pass")
    p.add_argument("--max-age-hours", type=int, default=72, help="Max allowed report age in hours; <=0 disables age check")
    p.add_argument("--out-json", default="runs/ci/ui_release_readiness.json", help="Output JSON report path")
    p.add_argument("--out-md", default="runs/ci/ui_release_readiness.md", help="Output markdown report path")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    workspace_root = Path(args.workspace_root).resolve()
    now_utc = datetime.now(timezone.utc)

    failures: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []
    checks: List[Dict[str, Any]] = []
    gate_reports: List[Dict[str, Any]] = []

    stability_path = _resolve_path(str(args.ui_stability_json or ""), workspace_root)
    freeze_path = _resolve_path(str(args.ui_freeze_json or ""), workspace_root)
    audit_path = _resolve_path(str(args.ui_audit_json or ""), workspace_root)

    if not stability_path.exists():
        failures.append({"name": "ui_stability_report_exists", "message": f"missing report: {stability_path}"})
    else:
        try:
            stability_payload = _load_json(stability_path)
            gate = _check_gate_report(
                name="ui_stability_smoke",
                path=stability_path,
                payload=stability_payload,
                now_utc=now_utc,
                max_age_hours=int(args.max_age_hours),
                required=True,
            )
            gate_reports.append(gate)
            checks.extend(gate["checks"])
            failures.extend(gate["failures"])
            warnings.extend(gate["warnings"])
        except Exception as exc:
            failures.append({"name": "ui_stability_report_parse", "message": f"{stability_path}: {exc}"})

    if freeze_path.exists():
        try:
            freeze_payload = _load_json(freeze_path)
            gate = _check_gate_report(
                name="ui_freeze_acceptance",
                path=freeze_path,
                payload=freeze_payload,
                now_utc=now_utc,
                max_age_hours=int(args.max_age_hours),
                required=bool(args.require_freeze_report),
            )
            gate_reports.append(gate)
            checks.extend(gate["checks"])
            failures.extend(gate["failures"])
            warnings.extend(gate["warnings"])
        except Exception as exc:
            if args.require_freeze_report:
                failures.append({"name": "ui_freeze_report_parse", "message": f"{freeze_path}: {exc}"})
            else:
                warnings.append({"name": "ui_freeze_report_parse", "message": f"{freeze_path}: {exc}"})
    elif args.require_freeze_report:
        failures.append({"name": "ui_freeze_report_exists", "message": f"missing report: {freeze_path}"})
    else:
        warnings.append({"name": "ui_freeze_report_exists", "message": f"optional report missing: {freeze_path}"})

    if audit_path.exists():
        try:
            audit_payload = _load_json(audit_path)
            gate = _check_gate_report(
                name="ui_audit_acceptance",
                path=audit_path,
                payload=audit_payload,
                now_utc=now_utc,
                max_age_hours=int(args.max_age_hours),
                required=bool(args.require_audit_report),
            )
            gate_reports.append(gate)
            checks.extend(gate["checks"])
            failures.extend(gate["failures"])
            warnings.extend(gate["warnings"])
        except Exception as exc:
            if args.require_audit_report:
                failures.append({"name": "ui_audit_report_parse", "message": f"{audit_path}: {exc}"})
            else:
                warnings.append({"name": "ui_audit_report_parse", "message": f"{audit_path}: {exc}"})
    elif args.require_audit_report:
        failures.append({"name": "ui_audit_report_exists", "message": f"missing report: {audit_path}"})
    else:
        warnings.append({"name": "ui_audit_report_exists", "message": f"optional report missing: {audit_path}"})

    status = "pass" if not failures else "fail"
    report: Dict[str, Any] = {
        "status": status,
        "generated_at": now_utc.isoformat(),
        "workspace_root": str(workspace_root),
        "max_age_hours": int(args.max_age_hours),
        "require_freeze_report": bool(args.require_freeze_report),
        "require_audit_report": bool(args.require_audit_report),
        "gate_reports": gate_reports,
        "checks": checks,
        "warnings": warnings,
        "failures": failures,
        "check_count": len(checks),
        "warning_count": len(warnings),
        "failure_count": len(failures),
    }

    out_json = _resolve_path(str(args.out_json or ""), workspace_root)
    out_md = _resolve_path(str(args.out_md or ""), workspace_root)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    out_md.write_text(_build_markdown(report), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

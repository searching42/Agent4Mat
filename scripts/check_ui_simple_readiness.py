#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
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


def _build_markdown(report: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# UI Simple Readiness")
    lines.append("")
    lines.append(f"- generated_at: `{report.get('generated_at')}`")
    lines.append(f"- status: `{report.get('status')}`")
    lines.append(f"- workspace_root: `{report.get('workspace_root')}`")
    lines.append(f"- ui_app_path: `{report.get('ui_app_path')}`")
    lines.append(f"- check_count: `{report.get('check_count')}`")
    lines.append(f"- failed_count: `{report.get('failed_count')}`")
    lines.append(f"- warning_count: `{report.get('warning_count')}`")
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
    lines.append("## Warnings")
    warnings = report.get("warnings") if isinstance(report.get("warnings"), list) else []
    if warnings:
        for row in warnings:
            if not isinstance(row, dict):
                continue
            lines.append(f"- {row.get('name')}: {row.get('message')}")
    else:
        lines.append("- none")
    lines.append("")
    return "\n".join(lines)


def _append_pattern_check(
    *,
    content: str,
    name: str,
    pattern: str,
    checks: List[Dict[str, Any]],
    failures: List[Dict[str, Any]],
) -> None:
    if re.search(pattern, content, re.S):
        checks.append({"name": name, "message": "pattern matched"})
    else:
        failures.append({"name": name, "message": f"missing pattern: {pattern}"})


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate simple-mode UI readiness contracts")
    p.add_argument("--workspace-root", default=".", help="Workspace root")
    p.add_argument("--ui-app-path", default="ui/app.py", help="Path to ui/app.py")
    p.add_argument("--out-json", default="runs/ci/ui_simple_readiness.json", help="Output JSON report path")
    p.add_argument("--out-md", default="runs/ci/ui_simple_readiness.md", help="Output markdown report path")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    workspace_root = Path(args.workspace_root).resolve()
    ui_app_path = _resolve_path(str(args.ui_app_path or ""), workspace_root)
    out_json = _resolve_path(str(args.out_json or ""), workspace_root)
    out_md = _resolve_path(str(args.out_md or ""), workspace_root)

    checks: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []

    if not ui_app_path.exists():
        failures.append({"name": "ui_app_exists", "message": f"missing file: {ui_app_path}"})
        content = ""
    else:
        content = ui_app_path.read_text(encoding="utf-8")
        checks.append({"name": "ui_app_exists", "message": f"found file: {ui_app_path}"})

    patterns = [
        ("simple_mode_default_boot", r"applyOutputViewMode\(String\(uiPrefs\.outputViewMode \|\| 'simple'\)\);"),
        ("simple_project_strip_present", r"id=\\\"simple_project_strip\\\""),
        ("simple_input_hub_present", r"id=\\\"simple_input_hub\\\""),
        ("simple_outputs_panel_present", r"id=\\\"right_simple_core_outputs\\\""),
        ("simple_hub_summary_present", r"id=\\\"simple_input_summary\\\""),
        ("simple_hub_status_light_present", r"id=\\\"simple_input_status_light\\\""),
        ("simple_hub_summary_fn_present", r"function updateSimpleInputHubSummary\(\)"),
        (
            "simple_hub_summary_sync_main",
            r"function syncSimpleInputHubFromMain\(\)\s*\{[\s\S]*?updateSimpleInputHubSummary\(\);",
        ),
        (
            "simple_hub_summary_sync_simple",
            r"function syncMainFromSimpleInputHub\(syncPath\)\s*\{[\s\S]*?updateSimpleInputHubSummary\(\);",
        ),
        (
            "simple_hub_summary_message_hook",
            r"function setMessageInput\(text, opts\)\s*\{[\s\S]*?updateSimpleInputHubSummary\(\);",
        ),
        (
            "simple_hub_summary_composer_hook",
            r"function bindComposerShortcuts\(\)\s*\{[\s\S]*?input\.addEventListener\('input',\s*\(\)\s*=>\s*\{[\s\S]*?updateSimpleInputHubSummary\(\);",
        ),
        (
            "simple_mode_hides_advanced_drawers",
            r"body\.output-simple-mode #control_center_drawer,\s*"
            r"body\.output-simple-mode #chat_timeline_panel_drawer,\s*"
            r"body\.output-simple-mode #single_step_runner_drawer,\s*"
            r"body\.output-simple-mode #artifacts_validation_drawer,\s*"
            r"body\.output-simple-mode #file_input_drawer,\s*"
            r"body\.output-simple-mode \.chat-quick-chips,\s*"
            r"body\.output-simple-mode \.chat-quick-strip,\s*"
            r"body\.output-simple-mode #prompt_history_box,\s*"
            r"body\.output-simple-mode \.web-preset-row\s*\{\s*display:\s*none !important;\s*\}",
        ),
        (
            "simple_mode_hides_chat_status_actions",
            r"body\.output-simple-mode #chat_status_ribbon \.status-actions\s*\{\s*display:\s*none !important;\s*\}",
        ),
    ]
    for name, pattern in patterns:
        _append_pattern_check(content=content, name=name, pattern=pattern, checks=checks, failures=failures)

    action_start = content.find('<div class=\\"right-simple-actions simple-only\\" id=\\"right_simple_actions\\">')
    if action_start < 0:
        failures.append({"name": "right_simple_actions_block_exists", "message": "missing right_simple_actions block"})
    else:
        checks.append({"name": "right_simple_actions_block_exists", "message": "found right_simple_actions block"})
        action_end = content.find("</div>", action_start)
        if action_end <= action_start:
            failures.append({"name": "right_simple_actions_block_closed", "message": "unable to find right_simple_actions block end"})
        else:
            block = content[action_start:action_end]
            if ">Summary<" in block:
                failures.append({"name": "right_simple_actions_no_summary_duplicate", "message": "Summary action should not appear in right_simple_actions"})
            else:
                checks.append({"name": "right_simple_actions_no_summary_duplicate", "message": "Summary action is not duplicated"})

    status = "pass" if not failures else "fail"
    report = {
        "status": status,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "workspace_root": str(workspace_root),
        "ui_app_path": str(ui_app_path),
        "check_count": len(checks),
        "failed_count": len(failures),
        "warning_count": len(warnings),
        "checks": checks,
        "failures": failures,
        "warnings": warnings,
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    out_md.write_text(_build_markdown(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

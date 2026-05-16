#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_path(raw: str, cwd: Path) -> Path:
    p = Path(raw)
    if not p.is_absolute():
        p = (cwd / p).resolve()
    return p


def _git_sha(cwd: Path) -> str:
    try:
        cp = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
        )
        if cp.returncode == 0:
            return (cp.stdout or "").strip()
    except Exception:
        pass
    return ""


def _pick_plqy_target(plan: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    targets = plan.get("design_spec", {}).get("targets", [])
    if not isinstance(targets, list):
        return None
    for item in targets:
        if isinstance(item, dict) and item.get("name") == "plqy":
            return item
    return None


def _safe_env_snapshot() -> Dict[str, str]:
    keys = [
        "UNIMOL_REMOTE_HOST",
        "UNIMOL_REMOTE_PY",
        "UNIMOL_REMOTE_TMP_BASE",
        "OLED_AGENT_REINVENT4_PIPELINE_SCRIPT",
        "OLED_AGENT_REINVENT4_SOURCE_CSV",
        "OLED_AGENT_UNIMOL_SCORE_SCRIPT",
        "OLED_AGENT_USE_EXTERNAL_SCORER",
        "OLED_AGENT_UNIMOL_SCORE_MODE",
        "OLED_AGENT_REINVENT4_ADAPTER_MODE",
        "REMOTE_HOST",
        "REMOTE_BASE",
        "REMOTE_PY",
        "V2_FILTER_EXEC_MODE",
    ]
    out: Dict[str, str] = {}
    for key in keys:
        value = str(os.environ.get(key, "") or "")
        if value:
            out[key] = value
    return out


def _build_markdown(evidence: Dict[str, Any]) -> str:
    checks = evidence.get("checks", {})
    adapters = evidence.get("adapters", {})
    planner = evidence.get("planner", {})
    plqy = evidence.get("plqy_semantics", {})
    failure_diag = evidence.get("failure_diagnostics", {})
    artifacts = evidence.get("artifacts", {})
    env_snapshot = evidence.get("env_snapshot", {})

    lines: List[str] = []
    lines.append("# Real Chain Acceptance Evidence")
    lines.append("")
    lines.append(f"- generated_at: `{evidence.get('generated_at')}`")
    lines.append(f"- task_id: `{evidence.get('task_id')}`")
    lines.append(f"- overall: `{evidence.get('overall')}`")
    lines.append(f"- git_sha: `{evidence.get('git_sha')}`")
    lines.append("")
    lines.append("## Checks")
    for key, value in checks.items():
        lines.append(f"- {key}: `{value}`")
    lines.append("")
    lines.append("## Planner")
    lines.append(f"- provider_effective: `{planner.get('provider_effective')}`")
    lines.append(f"- provider_status: `{planner.get('provider_status')}`")
    lines.append("")
    lines.append("## Adapters")
    lines.append(f"- generate: `{adapters.get('generate')}`")
    lines.append(f"- score: `{adapters.get('score')}`")
    lines.append(f"- score_used_fallback: `{adapters.get('score_used_fallback')}`")
    lines.append(f"- guardrails_strict_status: `{adapters.get('guardrails_strict_status')}`")
    lines.append("")
    lines.append("## PLQY Semantics")
    lines.append(f"- target_center: `{plqy.get('target_center')}`")
    lines.append(f"- metadata_plqy_scale: `{plqy.get('metadata_plqy_scale')}`")
    lines.append(f"- percent_scale_check: `{plqy.get('percent_scale_check')}`")
    lines.append("")
    lines.append("## Failure Diagnostics")
    lines.append(f"- evaluation_failed_count: `{failure_diag.get('evaluation_failed_count')}`")
    lines.append(f"- guardrails_failed_count: `{failure_diag.get('guardrails_failed_count')}`")
    lines.append(f"- evaluation_latest_failure_kind: `{failure_diag.get('evaluation_latest_failure_kind')}`")
    lines.append(f"- guardrails_latest_failure_kind: `{failure_diag.get('guardrails_latest_failure_kind')}`")
    lines.append("")
    lines.append("## Artifacts")
    for key, value in artifacts.items():
        lines.append(f"- {key}: `{value}`")
    if env_snapshot:
        lines.append("")
        lines.append("## Env Snapshot")
        for key, value in env_snapshot.items():
            lines.append(f"- {key}: `{value}`")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect release evidence from a real-chain acceptance run")
    p.add_argument("--workspace-root", default=".", help="Workspace root (git repo root)")
    p.add_argument("--result-json", required=True, help="Path to acceptance_result.json")
    p.add_argument("--out-json", default="", help="Output path for evidence JSON (default: run_dir/release_evidence.json)")
    p.add_argument("--out-md", default="", help="Output path for evidence Markdown (default: run_dir/release_evidence.md)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cwd = Path.cwd()
    workspace_root = Path(args.workspace_root).resolve()
    result_json = _resolve_path(str(args.result_json), cwd)

    if not result_json.exists():
        print(f"[FAIL] result json not found: {result_json}")
        return 1

    result = _load_json(result_json)
    required_path_keys = {
        "plan": ["plan_path"],
        "execution": ["execution_path"],
        "decision_summary": ["decision_summary_path"],
        "task_state": ["task_state_path"],
        "evaluation_report": ["evaluation_report_path", "logging_evaluation_report_path"],
        "guardrails_report": ["guardrails_report_path", "logging_guardrails_report_path"],
    }
    resolved_paths: Dict[str, Path] = {}
    for logical, key_list in required_path_keys.items():
        raw = ""
        for key in key_list:
            raw = str(result.get(key) or "").strip()
            if raw:
                break
        if not raw:
            print(f"[FAIL] result json missing required key in candidates: {key_list}")
            return 1
        path = _resolve_path(raw, cwd)
        if not path.exists():
            print(f"[FAIL] artifact not found ({logical}): {path}")
            return 1
        resolved_paths[logical] = path

    plan = _load_json(resolved_paths["plan"])
    execution = _load_json(resolved_paths["execution"])
    decision = _load_json(resolved_paths["decision_summary"])
    evaluation = _load_json(resolved_paths["evaluation_report"])
    guardrails = _load_json(resolved_paths["guardrails_report"])

    records = {
        r.get("name"): r.get("result", {})
        for r in execution.get("records", [])
        if isinstance(r, dict) and isinstance(r.get("name"), str)
    }
    gen = records.get("generate_candidates", {}) if isinstance(records.get("generate_candidates", {}), dict) else {}
    score = records.get("score_candidates", {}) if isinstance(records.get("score_candidates", {}), dict) else {}
    score_step = decision.get("score_step", {}) if isinstance(decision.get("score_step", {}), dict) else {}
    eval_diag = evaluation.get("failure_diagnostics") if isinstance(evaluation.get("failure_diagnostics"), dict) else {}
    guard_diag = guardrails.get("failure_diagnostics") if isinstance(guardrails.get("failure_diagnostics"), dict) else {}
    eval_failed_count = int(eval_diag.get("failed_count") or 0)
    guard_failed_count = int(guard_diag.get("failed_count") or 0)
    guardrails_strict_status = str(guardrails.get("strict_status") or "")

    plqy_target = _pick_plqy_target(plan) or {}
    plqy_center = plqy_target.get("target_center")
    plqy_percent_ok = isinstance(plqy_center, (int, float)) and (1.0 < float(plqy_center) <= 100.0)

    checks = {
        "status_success": result.get("status") == "success",
        "generate_adapter_expected": gen.get("adapter") == "reinvent4_generate_adapter_v1",
        "score_adapter_expected": score.get("adapter") == "unimol_score_adapter_v1",
        "generate_no_fallback_error": not bool(gen.get("fallback_error")),
        "score_no_fallback_error": not bool(score.get("fallback_error")),
        "score_used_fallback_false": not bool(score_step.get("used_fallback")),
        "plqy_center_percent_scale": plqy_percent_ok,
        "evaluation_failure_diag_zero": eval_failed_count == 0,
        "guardrails_failure_diag_zero": guard_failed_count == 0,
        "guardrails_strict_status_pass": guardrails_strict_status == "pass",
    }
    overall = "pass" if all(bool(v) for v in checks.values()) else "fail"

    run_dir = result_json.parent.resolve()
    out_json = _resolve_path(args.out_json, cwd) if str(args.out_json).strip() else (run_dir / "release_evidence.json")
    out_md = _resolve_path(args.out_md, cwd) if str(args.out_md).strip() else (run_dir / "release_evidence.md")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    planner_md = plan.get("design_spec", {}).get("metadata", {}) if isinstance(plan.get("design_spec", {}), dict) else {}
    evidence = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "task_id": result.get("task_id"),
        "overall": overall,
        "git_sha": _git_sha(workspace_root),
        "checks": checks,
        "planner": {
            "provider_effective": planner_md.get("planner_provider_effective"),
            "provider_status": planner_md.get("planner_provider_status"),
        },
        "adapters": {
            "generate": gen.get("adapter"),
            "score": score.get("adapter"),
            "score_used_fallback": bool(score_step.get("used_fallback")),
            "guardrails_strict_status": guardrails_strict_status,
        },
        "failure_diagnostics": {
            "evaluation_failed_count": eval_failed_count,
            "guardrails_failed_count": guard_failed_count,
            "evaluation_latest_failure_kind": str(eval_diag.get("latest_failure_kind") or ""),
            "guardrails_latest_failure_kind": str(guard_diag.get("latest_failure_kind") or ""),
        },
        "plqy_semantics": {
            "target_center": plqy_center,
            "metadata_plqy_scale": planner_md.get("plqy_scale"),
            "percent_scale_check": plqy_percent_ok,
        },
        "artifacts": {
            "result_json": str(result_json),
            "plan_path": str(resolved_paths["plan"]),
            "execution_path": str(resolved_paths["execution"]),
            "decision_summary_path": str(resolved_paths["decision_summary"]),
            "task_state_path": str(resolved_paths["task_state"]),
            "evaluation_report_path": str(resolved_paths["evaluation_report"]),
            "guardrails_report_path": str(resolved_paths["guardrails_report"]),
        },
        "env_snapshot": _safe_env_snapshot(),
    }

    out_json.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    out_md.write_text(_build_markdown(evidence), encoding="utf-8")

    print(f"EVIDENCE_JSON={out_json}")
    print(f"EVIDENCE_MD={out_md}")
    print(json.dumps({"overall": overall, "task_id": result.get("task_id"), "git_sha": evidence["git_sha"]}, ensure_ascii=False))
    return 0 if overall == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def _run_cli(*, root: Path, args: list[str], env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )


def _extract_last_json_object(text: str) -> Optional[Dict[str, Any]]:
    raw = str(text or "")
    stripped = raw.strip()
    if not stripped:
        return None
    try:
        payload = json.loads(stripped)
        if isinstance(payload, dict):
            return payload
    except Exception:
        pass
    decoder = json.JSONDecoder()
    candidates: list[Dict[str, Any]] = []
    cursor = 0
    while cursor < len(raw):
        start = raw.find("{", cursor)
        if start < 0:
            break
        try:
            payload, _ = decoder.raw_decode(raw[start:])
        except Exception:
            cursor = start + 1
            continue
        if isinstance(payload, dict):
            candidates.append(payload)
        cursor = start + 1
    if candidates:
        return candidates[-1]
    return None


def _snippet(text: str, *, limit: int = 1200) -> str:
    value = str(text or "")
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


def _resolve_output_path(*, workspace_root: Path, out_json: str) -> Path:
    path = Path(out_json)
    if not path.is_absolute():
        path = workspace_root / path
    return path


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _finalize(*, payload: Dict[str, Any], out_json_path: Path, exit_code: int) -> int:
    final_payload = dict(payload)
    final_payload["generated_at"] = str(final_payload.get("generated_at") or datetime.now(timezone.utc).isoformat())
    status_text = str(final_payload.get("status") or "").strip().lower()
    default_failed_count = 0 if status_text == "pass" else 1
    try:
        failed_count = int(final_payload.get("failed_count"))
    except Exception:
        failed_count = default_failed_count
    final_payload["failed_count"] = max(0, failed_count)
    try:
        check_count = int(final_payload.get("check_count"))
    except Exception:
        check_count = 1
    final_payload["check_count"] = max(1, check_count)
    final_payload["out_json_path"] = str(out_json_path)
    _write_json(out_json_path, final_payload)
    print(json.dumps(final_payload, ensure_ascii=False))
    return int(exit_code)


def _resolve_output_file(*, workspace_root: Path, raw_path: str) -> Optional[Path]:
    value = str(raw_path or "").strip()
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = workspace_root / path
    return path


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Check strict require-real-adapters acceptance path.")
    parser.add_argument(
        "--workspace-root",
        default=str(Path(__file__).resolve().parents[1]),
        help="Workspace root containing src/oled_agent.",
    )
    parser.add_argument(
        "--out-json",
        default="runs/ci/real_no_fallback_gate.json",
        help="Write structured summary JSON to this path (relative to workspace root if not absolute).",
    )
    args = parser.parse_args(argv)
    root = Path(args.workspace_root).resolve()
    out_json_path = _resolve_output_path(workspace_root=root, out_json=str(args.out_json))

    try:
        return _main_impl(root=root, out_json_path=out_json_path)
    except Exception as exc:
        return _finalize(
            payload={
                "status": "failed",
                "check": "require-real-adapters",
                "reason": "unexpected_exception",
                "workspace_root": str(root),
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
            out_json_path=out_json_path,
            exit_code=1,
        )


def _main_impl(*, root: Path, out_json_path: Path) -> int:
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        req = {
            "task_id": "ci_real_no_fallback",
            "request_text": "设计470nm附近且高PLQY分子",
            "mode": "fast_screen",
            "targets": [{"property": "plqy", "objective": "maximize", "target_value": 60.0}],
            "budget": {"max_candidates": 10},
            "model_preferences": {
                "predictor_id": "unimol_lambda_plqy_real_v1",
                "generator_id": "reinvent4_generator_real_v1"
            }
        }
        req_path = tdp / "request.json"
        req_path.write_text(json.dumps(req, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        env = {
            "PYTHONPATH": str(root / "src"),
            "OLED_AGENT_REINVENT4_ADAPTER_MODE": "smoke",
            "OLED_AGENT_UNIMOL_SCORE_MODE": "smoke",
        }

        cp = _run_cli(
            root=root,
            args=[
                sys.executable,
                "-m",
                "oled_agent.cli",
                "agent-run-json",
                "--workspace-root",
                str(root),
                "--catalog",
                str(root / "scripts" / "adapters" / "real_adapters_catalog.json"),
                "--request-json",
                str(req_path),
                "--require-real-adapters",
            ],
            env=env,
        )
        # smoke real-adapters path should pass and not use local fallback.
        if cp.returncode != 0:
            return _finalize(
                payload={
                    "status": "failed",
                    "check": "require-real-adapters",
                    "reason": "full_pipeline_nonzero_exit",
                    "workspace_root": str(root),
                    "full_pipeline_exit_code": int(cp.returncode),
                    "full_pipeline_stdout_excerpt": _snippet(cp.stdout),
                    "full_pipeline_stderr_excerpt": _snippet(cp.stderr),
                },
                out_json_path=out_json_path,
                exit_code=cp.returncode,
            )
        run_payload = _extract_last_json_object(cp.stdout)
        if not isinstance(run_payload, dict):
            return _finalize(
                payload={
                    "status": "failed",
                    "check": "require-real-adapters",
                    "reason": "cannot_parse_agent_run_json_output",
                    "workspace_root": str(root),
                    "full_pipeline_exit_code": int(cp.returncode),
                    "full_pipeline_stdout_excerpt": _snippet(cp.stdout),
                    "full_pipeline_stderr_excerpt": _snippet(cp.stderr),
                },
                out_json_path=out_json_path,
                exit_code=1,
            )

        evaluation_path = _resolve_output_file(
            workspace_root=root,
            raw_path=str(run_payload.get("evaluation_report_path") or run_payload.get("logging_evaluation_report_path") or ""),
        )
        guardrails_path = _resolve_output_file(
            workspace_root=root,
            raw_path=str(run_payload.get("guardrails_report_path") or run_payload.get("logging_guardrails_report_path") or ""),
        )
        if evaluation_path is None or guardrails_path is None:
            return _finalize(
                payload={
                    "status": "failed",
                    "check": "require-real-adapters",
                    "reason": "missing_report_paths",
                    "workspace_root": str(root),
                    "run_payload_keys": sorted(run_payload.keys()),
                    "evaluation_report_path": str(evaluation_path) if evaluation_path else "",
                    "guardrails_report_path": str(guardrails_path) if guardrails_path else "",
                },
                out_json_path=out_json_path,
                exit_code=1,
            )
        if not evaluation_path.exists() or not guardrails_path.exists():
            return _finalize(
                payload={
                    "status": "failed",
                    "check": "require-real-adapters",
                    "reason": "missing_report_files",
                    "workspace_root": str(root),
                    "evaluation_report_path": str(evaluation_path),
                    "guardrails_report_path": str(guardrails_path),
                },
                out_json_path=out_json_path,
                exit_code=1,
            )
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        guardrails = json.loads(guardrails_path.read_text(encoding="utf-8"))
        eval_diag = evaluation.get("failure_diagnostics") if isinstance(evaluation.get("failure_diagnostics"), dict) else {}
        guard_diag = guardrails.get("failure_diagnostics") if isinstance(guardrails.get("failure_diagnostics"), dict) else {}
        if int(eval_diag.get("failed_count") or 0) != 0:
            return _finalize(
                payload={
                    "status": "failed",
                    "check": "require-real-adapters",
                    "reason": "evaluation_failed_count_nonzero",
                    "workspace_root": str(root),
                    "evaluation_diag": eval_diag,
                    "evaluation_report_path": str(evaluation_path),
                    "guardrails_report_path": str(guardrails_path),
                },
                out_json_path=out_json_path,
                exit_code=1,
            )
        if int(guard_diag.get("failed_count") or 0) != 0:
            return _finalize(
                payload={
                    "status": "failed",
                    "check": "require-real-adapters",
                    "reason": "guardrails_failed_count_nonzero",
                    "workspace_root": str(root),
                    "guardrails_diag": guard_diag,
                    "evaluation_report_path": str(evaluation_path),
                    "guardrails_report_path": str(guardrails_path),
                },
                out_json_path=out_json_path,
                exit_code=1,
            )
        if str(guardrails.get("strict_status") or "").strip() != "pass":
            return _finalize(
                payload={
                    "status": "failed",
                    "check": "require-real-adapters",
                    "reason": "guardrails_strict_status_not_pass",
                    "workspace_root": str(root),
                    "strict_status": guardrails.get("strict_status"),
                    "evaluation_report_path": str(evaluation_path),
                    "guardrails_report_path": str(guardrails_path),
                },
                out_json_path=out_json_path,
                exit_code=1,
            )

        step_task_id = "ci_real_no_fallback_step"
        step_task = {
            "version": "2.0",
            "task_id": step_task_id,
            "request_text": "step strict no fallback",
            "execution_mode": "single_step",
            "operation": "score_candidates",
            "property": "plqy",
            "range": "60-100",
            "n_structures": 10,
            "constraints": {},
            "prediction_model": "unimol_lambda_plqy_real_v1",
            "model_preferences": {
                "predictor_id": "unimol_lambda_plqy_real_v1",
                "generator_id": "reinvent4_lambda_em_v2",
            },
            "status": "approved",
        }
        step_task_path = tdp / "task_step.json"
        step_task_path.write_text(json.dumps(step_task, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        in_csv = tdp / "step_input.csv"
        in_csv.write_text("candidate_id,SMILES\nc1,c1ccccc1\n", encoding="utf-8")

        cp_retrieve = _run_cli(
            root=root,
            args=[
                sys.executable,
                "-m",
                "oled_agent.cli",
                "agent-run-step",
                "--workspace-root",
                str(root),
                "--catalog",
                str(root / "scripts" / "adapters" / "real_adapters_catalog.json"),
                "--task-json",
                str(step_task_path),
                "--operation",
                "retrieve_candidate_data",
                "--args-json",
                json.dumps({"candidate_data": str(in_csv)}),
                "--require-real-adapters",
            ],
            env=env,
        )
        if cp_retrieve.returncode != 0:
            return _finalize(
                payload={
                    "status": "failed",
                    "check": "require-real-adapters",
                    "reason": "retrieve_candidate_data_nonzero_exit",
                    "workspace_root": str(root),
                    "retrieve_exit_code": int(cp_retrieve.returncode),
                    "retrieve_stdout_excerpt": _snippet(cp_retrieve.stdout),
                    "retrieve_stderr_excerpt": _snippet(cp_retrieve.stderr),
                    "evaluation_report_path": str(evaluation_path),
                    "guardrails_report_path": str(guardrails_path),
                },
                out_json_path=out_json_path,
                exit_code=cp_retrieve.returncode,
            )

        cp_step = _run_cli(
            root=root,
            args=[
                sys.executable,
                "-m",
                "oled_agent.cli",
                "agent-run-step",
                "--workspace-root",
                str(root),
                "--catalog",
                str(root / "scripts" / "adapters" / "real_adapters_catalog.json"),
                "--task-json",
                str(step_task_path),
                "--operation",
                "score_candidates",
                "--require-real-adapters",
            ],
            env={**env, "OLED_AGENT_SCORE_CMD": f"{sys.executable} -c 'import sys; sys.exit(7)'"},
        )
        if cp_step.returncode != 3:
            return _finalize(
                payload={
                    "status": "failed",
                    "check": "require-real-adapters",
                    "reason": "step_require_real_adapters_not_enforced",
                    "workspace_root": str(root),
                    "single_step_exit_code": int(cp_step.returncode),
                    "single_step_stdout_excerpt": _snippet(cp_step.stdout),
                    "single_step_stderr_excerpt": _snippet(cp_step.stderr),
                    "evaluation_report_path": str(evaluation_path),
                    "guardrails_report_path": str(guardrails_path),
                },
                out_json_path=out_json_path,
                exit_code=1,
            )
        if "require-real-adapters" not in (cp_step.stdout + cp_step.stderr):
            return _finalize(
                payload={
                    "status": "failed",
                    "check": "require-real-adapters",
                    "reason": "step_require_real_adapters_missing_marker",
                    "workspace_root": str(root),
                    "single_step_exit_code": int(cp_step.returncode),
                    "single_step_stdout_excerpt": _snippet(cp_step.stdout),
                    "single_step_stderr_excerpt": _snippet(cp_step.stderr),
                    "evaluation_report_path": str(evaluation_path),
                    "guardrails_report_path": str(guardrails_path),
                },
                out_json_path=out_json_path,
                exit_code=1,
            )

    return _finalize(
        payload={
            "status": "pass",
            "check": "require-real-adapters",
            "workspace_root": str(root),
            "reason": "",
            "full_pipeline_exit_code": int(cp.returncode),
            "single_step_block_exit_code": int(cp_step.returncode),
            "evaluation_failed_count": int(eval_diag.get("failed_count") or 0),
            "guardrails_failed_count": int(guard_diag.get("failed_count") or 0),
            "guardrails_strict_status": str(guardrails.get("strict_status") or ""),
            "evaluation_report_path": str(evaluation_path),
            "guardrails_report_path": str(guardrails_path),
        },
        out_json_path=out_json_path,
        exit_code=0,
    )


if __name__ == "__main__":
    raise SystemExit(main())

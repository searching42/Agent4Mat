#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
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
    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    for line in reversed(lines):
        if not (line.startswith("{") and line.endswith("}")):
            continue
        try:
            payload = json.loads(line)
        except Exception:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def main() -> int:
    root = Path(__file__).resolve().parents[1]
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
            print(cp.stdout)
            print(cp.stderr)
            return cp.returncode
        run_payload = _extract_last_json_object(cp.stdout)
        if not isinstance(run_payload, dict):
            print(cp.stdout)
            print(cp.stderr)
            print(json.dumps({"status": "failed", "reason": "cannot_parse_agent_run_json_output"}, ensure_ascii=False))
            return 1
        evaluation_path = Path(
            str(run_payload.get("evaluation_report_path") or run_payload.get("logging_evaluation_report_path") or "")
        ).resolve()
        guardrails_path = Path(
            str(run_payload.get("guardrails_report_path") or run_payload.get("logging_guardrails_report_path") or "")
        ).resolve()
        if not evaluation_path.exists() or not guardrails_path.exists():
            print(
                json.dumps(
                    {
                        "status": "failed",
                        "reason": "missing_report_paths",
                        "evaluation_report_path": str(evaluation_path),
                        "guardrails_report_path": str(guardrails_path),
                    },
                    ensure_ascii=False,
                )
            )
            return 1
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        guardrails = json.loads(guardrails_path.read_text(encoding="utf-8"))
        eval_diag = evaluation.get("failure_diagnostics") if isinstance(evaluation.get("failure_diagnostics"), dict) else {}
        guard_diag = guardrails.get("failure_diagnostics") if isinstance(guardrails.get("failure_diagnostics"), dict) else {}
        if int(eval_diag.get("failed_count") or 0) != 0:
            print(
                json.dumps(
                    {"status": "failed", "reason": "evaluation_failed_count_nonzero", "evaluation_diag": eval_diag},
                    ensure_ascii=False,
                )
            )
            return 1
        if int(guard_diag.get("failed_count") or 0) != 0:
            print(
                json.dumps(
                    {"status": "failed", "reason": "guardrails_failed_count_nonzero", "guardrails_diag": guard_diag},
                    ensure_ascii=False,
                )
            )
            return 1
        if str(guardrails.get("strict_status") or "").strip() != "pass":
            print(
                json.dumps(
                    {
                        "status": "failed",
                        "reason": "guardrails_strict_status_not_pass",
                        "strict_status": guardrails.get("strict_status"),
                    },
                    ensure_ascii=False,
                )
            )
            return 1

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
            print(cp_retrieve.stdout)
            print(cp_retrieve.stderr)
            return cp_retrieve.returncode

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
            print(cp_step.stdout)
            print(cp_step.stderr)
            print(json.dumps({"status": "failed", "reason": "step_require_real_adapters_not_enforced"}, ensure_ascii=False))
            return 1
        if "require-real-adapters" not in (cp_step.stdout + cp_step.stderr):
            print(cp_step.stdout)
            print(cp_step.stderr)
            print(json.dumps({"status": "failed", "reason": "step_require_real_adapters_missing_marker"}, ensure_ascii=False))
            return 1

    print(
        json.dumps(
            {
                "status": "pass",
                "check": "require-real-adapters",
                "full_pipeline_exit_code": cp.returncode,
                "single_step_block_exit_code": cp_step.returncode,
                "evaluation_failed_count": int(eval_diag.get("failed_count") or 0),
                "guardrails_failed_count": int(guard_diag.get("failed_count") or 0),
                "guardrails_strict_status": str(guardrails.get("strict_status") or ""),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

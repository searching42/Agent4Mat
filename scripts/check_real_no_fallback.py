#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


def _run_cli(*, root: Path, args: list[str], env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )


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
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

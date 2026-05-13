#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        run_tag = tdp.name[-8:]

        def make_task(task_id: str) -> Dict[str, Any]:
            return {
                "version": "2.0",
                "task_id": task_id,
                "request_text": "step mode test",
                "execution_mode": "single_step",
                "operation": "clean_dataset",
                "property": "plqy",
                "range": "60-100",
                "n_structures": 10,
                "constraints": {"mw_min": 10, "mw_max": 1000},
                "train_data": None,
                "candidate_data": None,
                "prediction_model": "unimol_lambda_plqy_v1",
                "model_preferences": {
                    "predictor_id": "unimol_lambda_plqy_v1",
                    "generator_id": "reinvent4_lambda_em_v2",
                },
                "generation_input": {},
                "provenance": {},
                "status": "approved",
                "missing_fields": [],
                "questions": [],
                "compatibility_warnings": [],
            }

        happy_task_id = f"ci_step_mode_happy_{run_tag}"
        score_fail_task_id = f"ci_step_mode_fail_score_{run_tag}"
        train_fail_task_id = f"ci_step_mode_fail_train_{run_tag}"
        happy_json_task_id = f"ci_step_mode_json_happy_{run_tag}"
        score_json_fail_task_id = f"ci_step_mode_json_fail_score_{run_tag}"
        train_json_fail_task_id = f"ci_step_mode_json_fail_train_{run_tag}"

        happy_task_path = tdp / "task_happy.json"
        score_fail_task_path = tdp / "task_score_fail.json"
        train_fail_task_path = tdp / "task_train_fail.json"
        happy_json_task_path = tdp / "task_json_happy.json"
        score_json_fail_task_path = tdp / "task_json_score_fail.json"
        train_json_fail_task_path = tdp / "task_json_train_fail.json"
        happy_task_path.write_text(json.dumps(make_task(happy_task_id), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        score_fail_task_path.write_text(
            json.dumps(make_task(score_fail_task_id), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        train_fail_task_path.write_text(
            json.dumps(make_task(train_fail_task_id), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        happy_json_task_path.write_text(
            json.dumps(make_task(happy_json_task_id), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        score_json_fail_task_path.write_text(
            json.dumps(make_task(score_json_fail_task_id), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        train_json_fail_task_path.write_text(
            json.dumps(make_task(train_json_fail_task_id), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        in_csv = root / "runs" / "ci" / "step_mode_input.csv"
        in_csv.parent.mkdir(parents=True, exist_ok=True)
        in_csv.write_text("candidate_id,SMILES\nc1,c1ccccc1\n", encoding="utf-8")

        def run_step_raw(
            *,
            task_json_path: Path,
            operation: str,
            args_obj: Optional[Dict[str, Any]] = None,
            extra_env: Optional[Dict[str, str]] = None,
        ) -> tuple[subprocess.CompletedProcess[str], Dict[str, Any]]:
            cmd = [
                sys.executable,
                "-m",
                "oled_agent.cli",
                "agent-run-step",
                "--workspace-root",
                str(root),
                "--task-json",
                str(task_json_path),
                "--operation",
                operation,
            ]
            if isinstance(args_obj, dict):
                cmd.extend(["--args-json", json.dumps(args_obj)])
            env = dict(os.environ)
            env["PYTHONPATH"] = str(root / "src")
            if isinstance(extra_env, dict):
                env.update(extra_env)
            cp = subprocess.run(
                cmd,
                cwd=root,
                check=False,
                capture_output=True,
                text=True,
                env=env,
            )
            payload: Dict[str, Any] = {}
            if cp.stdout.strip():
                payload = json.loads(cp.stdout)
            return cp, payload

        def run_step_success(
            *,
            task_json_path: Path,
            operation: str,
            args_obj: Optional[Dict[str, Any]] = None,
            extra_env: Optional[Dict[str, str]] = None,
        ) -> Dict[str, Any]:
            cp, payload = run_step_raw(
                task_json_path=task_json_path,
                operation=operation,
                args_obj=args_obj,
                extra_env=extra_env,
            )
            if cp.returncode != 0:
                print(cp.stdout)
                print(cp.stderr)
                raise SystemExit(cp.returncode)
            if payload.get("status") != "success":
                print(cp.stdout)
                raise SystemExit(1)
            return payload

        def run_step_json_raw(
            *,
            task_json_path: Path,
            operation: str,
            args_obj: Optional[Dict[str, Any]] = None,
            extra_env: Optional[Dict[str, str]] = None,
        ) -> tuple[subprocess.CompletedProcess[str], Dict[str, Any]]:
            task_payload = json.loads(task_json_path.read_text(encoding="utf-8"))
            step_req: Dict[str, Any] = {"task": task_payload, "operation": operation}
            if isinstance(args_obj, dict):
                step_req["args"] = args_obj
            req_path = tdp / f"step_request_{task_payload.get('task_id')}_{operation}.json"
            req_path.write_text(json.dumps(step_req, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

            cmd = [
                sys.executable,
                "-m",
                "oled_agent.cli",
                "agent-run-step-json",
                "--workspace-root",
                str(root),
                "--step-request-json",
                str(req_path),
            ]
            env = dict(os.environ)
            env["PYTHONPATH"] = str(root / "src")
            if isinstance(extra_env, dict):
                env.update(extra_env)
            cp = subprocess.run(
                cmd,
                cwd=root,
                check=False,
                capture_output=True,
                text=True,
                env=env,
            )
            payload: Dict[str, Any] = {}
            if cp.stdout.strip():
                payload = json.loads(cp.stdout)
            return cp, payload

        def run_step_json_success(
            *,
            task_json_path: Path,
            operation: str,
            args_obj: Optional[Dict[str, Any]] = None,
            extra_env: Optional[Dict[str, str]] = None,
        ) -> Dict[str, Any]:
            cp, payload = run_step_json_raw(
                task_json_path=task_json_path,
                operation=operation,
                args_obj=args_obj,
                extra_env=extra_env,
            )
            if cp.returncode != 0:
                print(cp.stdout)
                print(cp.stderr)
                raise SystemExit(cp.returncode)
            if payload.get("status") != "success":
                print(cp.stdout)
                raise SystemExit(1)
            return payload

        clean_out = run_step_success(
            task_json_path=happy_task_path,
            operation="clean_dataset",
            args_obj={"input_csv": str(in_csv)},
        )
        score_out = run_step_success(task_json_path=happy_task_path, operation="score_candidates")
        train_out = run_step_success(task_json_path=happy_task_path, operation="train_predictor")

        score_adapter = str((score_out.get("result") or {}).get("adapter") or "")
        if not score_adapter:
            print(json.dumps({"status": "failed", "reason": "score_adapter_empty"}, ensure_ascii=False))
            return 1

        step_state_path = root / "runs" / "agent" / happy_task_id / "step_tool_state.json"
        if not step_state_path.exists():
            print(json.dumps({"status": "failed", "reason": "step_tool_state_missing"}, ensure_ascii=False))
            return 1
        step_state = json.loads(step_state_path.read_text(encoding="utf-8"))
        if not str(step_state.get("candidate_csv") or "").strip():
            print(json.dumps({"status": "failed", "reason": "candidate_csv_missing_in_state"}, ensure_ascii=False))
            return 1
        if not str(step_state.get("scored_csv") or "").strip():
            print(json.dumps({"status": "failed", "reason": "scored_csv_missing_in_state"}, ensure_ascii=False))
            return 1

        score_fail_cp, score_fail_payload = run_step_raw(
            task_json_path=score_fail_task_path,
            operation="score_candidates",
        )
        if score_fail_cp.returncode == 0:
            print(json.dumps({"status": "failed", "reason": "score_without_candidates_unexpected_success"}, ensure_ascii=False))
            return 1
        if score_fail_payload.get("status") != "failed":
            print(
                json.dumps(
                    {"status": "failed", "reason": "score_without_candidates_missing_failed_payload"},
                    ensure_ascii=False,
                )
            )
            return 1

        train_fail_cp, train_fail_payload = run_step_raw(
            task_json_path=train_fail_task_path,
            operation="train_predictor",
            extra_env={"OLED_AGENT_TRAIN_CMD": "python3 -c \"import sys; sys.exit(7)\""},
        )
        if train_fail_cp.returncode == 0:
            print(json.dumps({"status": "failed", "reason": "train_nonzero_unexpected_success"}, ensure_ascii=False))
            return 1
        if train_fail_payload.get("status") != "failed":
            print(
                json.dumps(
                    {"status": "failed", "reason": "train_nonzero_missing_failed_payload"},
                    ensure_ascii=False,
                )
            )
            return 1

        clean_json_out = run_step_json_success(
            task_json_path=happy_json_task_path,
            operation="clean_dataset",
            args_obj={"input_csv": str(in_csv)},
        )
        score_json_out = run_step_json_success(task_json_path=happy_json_task_path, operation="score_candidates")
        train_json_out = run_step_json_success(task_json_path=happy_json_task_path, operation="train_predictor")

        score_json_adapter = str((score_json_out.get("result") or {}).get("adapter") or "")
        if not score_json_adapter:
            print(json.dumps({"status": "failed", "reason": "score_json_adapter_empty"}, ensure_ascii=False))
            return 1

        step_json_state_path = root / "runs" / "agent" / happy_json_task_id / "step_tool_state.json"
        if not step_json_state_path.exists():
            print(json.dumps({"status": "failed", "reason": "step_json_tool_state_missing"}, ensure_ascii=False))
            return 1

        score_json_fail_cp, score_json_fail_payload = run_step_json_raw(
            task_json_path=score_json_fail_task_path,
            operation="score_candidates",
        )
        if score_json_fail_cp.returncode == 0:
            print(json.dumps({"status": "failed", "reason": "score_json_without_candidates_unexpected_success"}, ensure_ascii=False))
            return 1
        if score_json_fail_payload.get("status") != "failed":
            print(
                json.dumps(
                    {"status": "failed", "reason": "score_json_without_candidates_missing_failed_payload"},
                    ensure_ascii=False,
                )
            )
            return 1

        train_json_fail_cp, train_json_fail_payload = run_step_json_raw(
            task_json_path=train_json_fail_task_path,
            operation="train_predictor",
            extra_env={"OLED_AGENT_TRAIN_CMD": "python3 -c \"import sys; sys.exit(7)\""},
        )
        if train_json_fail_cp.returncode == 0:
            print(json.dumps({"status": "failed", "reason": "train_json_nonzero_unexpected_success"}, ensure_ascii=False))
            return 1
        if train_json_fail_payload.get("status") != "failed":
            print(
                json.dumps(
                    {"status": "failed", "reason": "train_json_nonzero_missing_failed_payload"},
                    ensure_ascii=False,
                )
            )
            return 1

    print(
        json.dumps(
            {
                "status": "pass",
                "check": "agent-run-step + agent-run-step-json happy(clean+score+train)+failure(score_missing_input,train_nonzero)",
                "clean_status": clean_out.get("status"),
                "score_adapter": score_adapter,
                "train_status": str((train_out.get("result") or {}).get("status") or ""),
                "score_missing_input_exit_code": score_fail_cp.returncode,
                "train_nonzero_exit_code": train_fail_cp.returncode,
                "clean_json_status": clean_json_out.get("status"),
                "score_json_adapter": score_json_adapter,
                "train_json_status": str((train_json_out.get("result") or {}).get("status") or ""),
                "score_json_missing_input_exit_code": score_json_fail_cp.returncode,
                "train_json_nonzero_exit_code": train_json_fail_cp.returncode,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

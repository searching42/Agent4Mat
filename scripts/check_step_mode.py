#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        task_id = "ci_step_mode"
        task = {
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
        task_path = tdp / "task.json"
        task_path.write_text(json.dumps(task, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        in_csv = root / "runs" / "ci" / "step_mode_input.csv"
        in_csv.parent.mkdir(parents=True, exist_ok=True)
        in_csv.write_text("candidate_id,SMILES\nc1,c1ccccc1\n", encoding="utf-8")

        env = {"PYTHONPATH": str(root / "src")}

        def run_step(operation: str, args_obj: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
            cmd = [
                sys.executable,
                "-m",
                "oled_agent.cli",
                "agent-run-step",
                "--workspace-root",
                str(root),
                "--task-json",
                str(task_path),
                "--operation",
                operation,
            ]
            if isinstance(args_obj, dict):
                cmd.extend(["--args-json", json.dumps(args_obj)])
            cp = subprocess.run(
                cmd,
                cwd=root,
                check=False,
                capture_output=True,
                text=True,
                env=env,
            )
            if cp.returncode != 0:
                print(cp.stdout)
                print(cp.stderr)
                raise SystemExit(cp.returncode)
            payload = json.loads(cp.stdout)
            if payload.get("status") != "success":
                print(cp.stdout)
                raise SystemExit(1)
            return payload

        clean_out = run_step("clean_dataset", {"input_csv": str(in_csv)})
        score_out = run_step("score_candidates")
        train_out = run_step("train_predictor")

        score_adapter = str((score_out.get("result") or {}).get("adapter") or "")
        if not score_adapter:
            print(json.dumps({"status": "failed", "reason": "score_adapter_empty"}, ensure_ascii=False))
            return 1

        step_state_path = root / "runs" / "agent" / task_id / "step_tool_state.json"
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

    print(
        json.dumps(
            {
                "status": "pass",
                "check": "agent-run-step clean_dataset+score_candidates+train_predictor",
                "clean_status": clean_out.get("status"),
                "score_adapter": score_adapter,
                "train_status": str((train_out.get("result") or {}).get("status") or ""),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

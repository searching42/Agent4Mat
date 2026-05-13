#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        task = {
            "version": "2.0",
            "task_id": "ci_step_mode",
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

        cp = subprocess.run(
            [
                sys.executable,
                "-m",
                "oled_agent.cli",
                "agent-run-step",
                "--workspace-root",
                str(root),
                "--task-json",
                str(task_path),
                "--operation",
                "clean_dataset",
                "--args-json",
                json.dumps({"input_csv": str(in_csv)}),
            ],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
            env={"PYTHONPATH": str(root / "src")},
        )
        if cp.returncode != 0:
            print(cp.stdout)
            print(cp.stderr)
            return cp.returncode
        out = json.loads(cp.stdout)
        if out.get("status") != "success":
            print(cp.stdout)
            return 1

    print(json.dumps({"status": "pass", "check": "agent-run-step clean_dataset"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

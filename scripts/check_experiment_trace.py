#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Tuple


def _run_cmd(cmd: list[str], *, cwd: Path, env: Dict[str, str]) -> Tuple[subprocess.CompletedProcess[str], Dict[str, Any]]:
    cp = subprocess.run(
        cmd,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    payload: Dict[str, Any] = {}
    raw = str(cp.stdout or "").strip()
    if raw:
        payload = json.loads(raw)
    return cp, payload


def _assert_trace_file(path_value: str, *, task_id: str, mode: str) -> Dict[str, Any]:
    p = Path(str(path_value or "")).resolve()
    if not p.exists():
        raise RuntimeError(f"trace file missing: {p}")
    payload = json.loads(p.read_text(encoding="utf-8"))
    if str(payload.get("task_id") or "") != task_id:
        raise RuntimeError(f"trace task_id mismatch: {p}")
    if str(payload.get("execution_mode") or "") != mode:
        raise RuntimeError(f"trace execution_mode mismatch: {p}")
    fingerprints = payload.get("fingerprints")
    if not isinstance(fingerprints, dict):
        raise RuntimeError(f"trace fingerprints missing: {p}")
    execution_sha = str(fingerprints.get("execution_sha256") or "").strip()
    if not execution_sha:
        raise RuntimeError(f"trace execution_sha256 missing: {p}")
    summary = payload.get("execution_summary")
    if not isinstance(summary, dict):
        raise RuntimeError(f"trace execution_summary missing: {p}")
    if int(summary.get("record_count") or 0) <= 0:
        raise RuntimeError(f"trace execution_summary.record_count invalid: {p}")
    core_artifacts = payload.get("core_artifacts")
    if not isinstance(core_artifacts, dict):
        raise RuntimeError(f"trace core_artifacts missing: {p}")
    return payload


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        run_tag = tdp.name[-8:]
        full_task_id = f"ci_exp_trace_full_{run_tag}"
        step_task_id = f"ci_exp_trace_step_{run_tag}"

        request = {
            "task_id": full_task_id,
            "request_text": "设计470nm附近且高PLQY分子",
            "mode": "fast_screen",
            "targets": [{"property": "plqy", "objective": "maximize", "target_value": 60.0}],
            "budget": {"max_candidates": 6},
            "model_preferences": {
                "predictor_id": "unimol_lambda_plqy_v1",
                "generator_id": "reinvent4_lambda_em_v2",
            },
        }
        request_path = tdp / "request.json"
        request_path.write_text(json.dumps(request, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        base_env = dict(os.environ)
        base_env["PYTHONPATH"] = str(root / "src")
        base_env["OLED_AGENT_ENABLE_WEB_EVIDENCE"] = "0"

        full_cmd = [
            sys.executable,
            "-m",
            "oled_agent.cli",
            "agent-run-json",
            "--workspace-root",
            str(root),
            "--request-json",
            str(request_path),
        ]
        cp_full, out_full = _run_cmd(full_cmd, cwd=root, env=base_env)
        if cp_full.returncode != 0:
            print(cp_full.stdout)
            print(cp_full.stderr)
            return cp_full.returncode
        if str(out_full.get("status") or "") != "success":
            print(json.dumps({"status": "fail", "reason": "full_pipeline_status_not_success", "payload": out_full}, ensure_ascii=False))
            return 1

        _assert_trace_file(str(out_full.get("experiment_trace_path") or ""), task_id=full_task_id, mode="full_pipeline")
        _assert_trace_file(str(out_full.get("logging_experiment_trace_path") or ""), task_id=full_task_id, mode="full_pipeline")
        _assert_trace_file(str(out_full.get("result_experiment_trace_path") or ""), task_id=full_task_id, mode="full_pipeline")

        input_csv = tdp / "step_input.csv"
        input_csv.write_text("candidate_id,SMILES\nc1,c1ccccc1\n", encoding="utf-8")
        task = {
            "version": "2.0",
            "task_id": step_task_id,
            "request_text": "clean candidates",
            "execution_mode": "single_step",
            "operation": "clean_dataset",
            "property": "plqy",
            "range": "60-100",
            "n_structures": 8,
            "constraints": {"mw_min": 100, "mw_max": 900},
            "prediction_model": "unimol_lambda_plqy_v1",
        }
        task_path = tdp / "task.json"
        task_path.write_text(json.dumps(task, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        step_cmd = [
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
            json.dumps({"input_csv": str(input_csv)}),
        ]
        cp_step, out_step = _run_cmd(step_cmd, cwd=root, env=base_env)
        if cp_step.returncode != 0:
            print(cp_step.stdout)
            print(cp_step.stderr)
            return cp_step.returncode
        if str(out_step.get("status") or "") != "success":
            print(json.dumps({"status": "fail", "reason": "single_step_status_not_success", "payload": out_step}, ensure_ascii=False))
            return 1

        _assert_trace_file(str(out_step.get("experiment_trace_path") or ""), task_id=step_task_id, mode="single_step")
        _assert_trace_file(str(out_step.get("logging_experiment_trace_path") or ""), task_id=step_task_id, mode="single_step")
        _assert_trace_file(str(out_step.get("result_experiment_trace_path") or ""), task_id=step_task_id, mode="single_step")

    print(
        json.dumps(
            {
                "status": "pass",
                "check": "experiment_trace_guard",
                "full_task_id": full_task_id,
                "step_task_id": step_task_id,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict


def _run_resume(*, workspace_root: Path, task_id: str, planner_provider: str, catalog_path: Path) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str((workspace_root / "src").resolve())
    cmd = [
        sys.executable,
        "-m",
        "oled_agent.cli",
        "agent-resume",
        "--workspace-root",
        str(workspace_root),
        "--task-id",
        task_id,
        "--planner-provider",
        planner_provider,
        "--catalog",
        str(catalog_path),
    ]
    return subprocess.run(
        cmd,
        cwd=workspace_root,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )


def _load_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON payload is not object: {path}")
    return payload


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Check agent-resume idempotence for a completed task")
    p.add_argument("--workspace-root", default=".", help="workspace root")
    p.add_argument("--task-id", default="", help="task id under runs/agent/<task_id>")
    p.add_argument("--result-json", default="", help="optional run result JSON (e.g. quickstart_result.json)")
    p.add_argument("--planner-provider", default="rule_based_v1")
    p.add_argument("--catalog", default="scripts/adapters/quickstart_catalog.json")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    workspace_root = Path(args.workspace_root).resolve()
    catalog_path = Path(args.catalog)
    if not catalog_path.is_absolute():
        catalog_path = (workspace_root / catalog_path).resolve()
    if not catalog_path.exists():
        print(json.dumps({"status": "fail", "reason": "catalog_not_found", "catalog_path": str(catalog_path)}, ensure_ascii=False))
        return 1

    task_id = str(args.task_id or "").strip()
    result_json_path = Path(str(args.result_json or "").strip())
    if str(result_json_path):
        if not result_json_path.is_absolute():
            result_json_path = (workspace_root / result_json_path).resolve()
        if not result_json_path.exists():
            print(json.dumps({"status": "fail", "reason": "result_json_not_found", "result_json": str(result_json_path)}, ensure_ascii=False))
            return 1
        result_payload = _load_json(result_json_path)
        if str(result_payload.get("status") or "").strip() != "success":
            print(
                json.dumps(
                    {
                        "status": "fail",
                        "reason": "result_json_status_not_success",
                        "result_json": str(result_json_path),
                        "result_status": str(result_payload.get("status") or ""),
                    },
                    ensure_ascii=False,
                )
            )
            return 1
        if not task_id:
            task_id = str(result_payload.get("task_id") or "").strip()

    if not task_id:
        print(json.dumps({"status": "fail", "reason": "missing_task_id"}, ensure_ascii=False))
        return 1

    run_dir = (workspace_root / "runs" / "agent" / task_id).resolve()
    if not run_dir.exists():
        print(json.dumps({"status": "fail", "reason": "run_dir_not_found", "run_dir": str(run_dir)}, ensure_ascii=False))
        return 1

    cp = _run_resume(
        workspace_root=workspace_root,
        task_id=task_id,
        planner_provider=str(args.planner_provider),
        catalog_path=catalog_path,
    )
    if cp.returncode != 0:
        print(cp.stdout)
        print(cp.stderr)
        print(json.dumps({"status": "fail", "reason": "agent_resume_nonzero", "returncode": cp.returncode}, ensure_ascii=False))
        return cp.returncode if cp.returncode > 0 else 1

    payload = json.loads(cp.stdout) if cp.stdout.strip() else {}
    if not isinstance(payload, dict):
        print(json.dumps({"status": "fail", "reason": "resume_output_not_object"}, ensure_ascii=False))
        return 1
    if str(payload.get("status") or "").strip() != "success":
        print(json.dumps({"status": "fail", "reason": "resume_status_not_success", "payload": payload}, ensure_ascii=False))
        return 1

    resumed = bool(payload.get("resumed"))
    skipped = int(payload.get("resume_skipped_steps") or 0)
    total = int(payload.get("resume_total_steps") or 0)
    if not resumed:
        print(json.dumps({"status": "fail", "reason": "resume_flag_false", "payload": payload}, ensure_ascii=False))
        return 1
    if total < 1:
        print(json.dumps({"status": "fail", "reason": "resume_total_steps_invalid", "resume_total_steps": total}, ensure_ascii=False))
        return 1
    if skipped != total:
        print(
            json.dumps(
                {
                    "status": "fail",
                    "reason": "resume_not_idempotent_full_skip",
                    "resume_skipped_steps": skipped,
                    "resume_total_steps": total,
                },
                ensure_ascii=False,
            )
        )
        return 1

    execution_path = Path(str(payload.get("execution_path") or "")).resolve()
    decision_path = Path(str(payload.get("decision_summary_path") or "")).resolve()
    task_state_path = Path(str(payload.get("task_state_path") or "")).resolve()
    for key, path in {
        "execution_path": execution_path,
        "decision_summary_path": decision_path,
        "task_state_path": task_state_path,
    }.items():
        if not path.exists():
            print(json.dumps({"status": "fail", "reason": "artifact_missing", "key": key, "path": str(path)}, ensure_ascii=False))
            return 1

    execution = _load_json(execution_path)
    records = execution.get("records") if isinstance(execution.get("records"), list) else []
    if len(records) < 1:
        print(json.dumps({"status": "fail", "reason": "execution_records_empty", "execution_path": str(execution_path)}, ensure_ascii=False))
        return 1

    decision = _load_json(decision_path)
    if not (isinstance(decision.get("inference_step"), dict) or isinstance(decision.get("score_step"), dict)):
        print(
            json.dumps(
                {
                    "status": "fail",
                    "reason": "decision_summary_missing_inference_or_score_step",
                    "decision_summary_path": str(decision_path),
                },
                ensure_ascii=False,
            )
        )
        return 1

    print(
        json.dumps(
            {
                "status": "pass",
                "task_id": task_id,
                "resumed": resumed,
                "resume_skipped_steps": skipped,
                "resume_total_steps": total,
                "execution_records": len(records),
                "execution_path": str(execution_path),
                "decision_summary_path": str(decision_path),
                "task_state_path": str(task_state_path),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

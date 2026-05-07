#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict


def _run_plan_with_mode(*, repo_root: Path, mode: str, task_id: str) -> Dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_root / "src")
    env["OLED_AGENT_LLM_PLANNER_CMD"] = f"{sys.executable} {repo_root / 'scripts' / 'mock_llm_planner.py'}"
    env["MOCK_LLM_MODE"] = mode

    cmd = [
        sys.executable,
        "-m",
        "oled_agent.cli",
        "agent-plan",
        "--workspace-root",
        str(repo_root),
        "--catalog",
        str(repo_root / "configs" / "models" / "catalog.json"),
        "--task-id",
        task_id,
        "--request",
        "设计470nm附近且高PLQY分子",
        "--planner-provider",
        "llm_v1",
    ]
    cp = subprocess.run(
        cmd,
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if cp.returncode != 0:
        raise RuntimeError(
            f"agent-plan failed for mode={mode}, rc={cp.returncode}\nstdout:\n{cp.stdout}\nstderr:\n{cp.stderr}"
        )

    try:
        payload = json.loads(cp.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"agent-plan returned non-JSON stdout for mode={mode}: {exc}") from exc

    md = (((payload or {}).get("design_spec") or {}).get("metadata") or {})
    if not isinstance(md, dict):
        raise RuntimeError(f"missing design_spec.metadata for mode={mode}")
    return {k: str(v) for k, v in md.items()}


def _assert_mode(
    *,
    repo_root: Path,
    mode: str,
    task_id: str,
    expected_planner: str,
    expected_effective: str,
    expected_status: str,
    expected_reason: str = "",
) -> None:
    md = _run_plan_with_mode(repo_root=repo_root, mode=mode, task_id=task_id)
    assert md.get("planner_provider_requested") == "llm_v1", md
    assert md.get("planner_provider_effective") == expected_effective, md
    assert md.get("planner_provider_status") == expected_status, md
    assert md.get("planner") == expected_planner, md

    got_reason = md.get("planner_provider_reason", "")
    if expected_reason:
        assert got_reason == expected_reason, md
    else:
        assert got_reason == "", md

    print(
        f"[PASS] mode={mode} planner={md.get('planner')} "
        f"effective={md.get('planner_provider_effective')} "
        f"status={md.get('planner_provider_status')} "
        f"reason={md.get('planner_provider_reason', '')}"
    )


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]

    _assert_mode(
        repo_root=repo_root,
        mode="active",
        task_id="ci_llm_mode_active",
        expected_planner="llm_v1",
        expected_effective="llm_v1",
        expected_status="active",
    )
    _assert_mode(
        repo_root=repo_root,
        mode="bad_json",
        task_id="ci_llm_mode_bad_json",
        expected_planner="rule_based_v1",
        expected_effective="rule_based_v1",
        expected_status="fallback",
        expected_reason="llm_output_invalid",
    )
    _assert_mode(
        repo_root=repo_root,
        mode="bad_model",
        task_id="ci_llm_mode_bad_model",
        expected_planner="rule_based_v1",
        expected_effective="rule_based_v1",
        expected_status="fallback",
        expected_reason="llm_output_invalid",
    )
    _assert_mode(
        repo_root=repo_root,
        mode="bad_tools",
        task_id="ci_llm_mode_bad_tools",
        expected_planner="rule_based_v1",
        expected_effective="rule_based_v1",
        expected_status="fallback",
        expected_reason="llm_output_invalid",
    )
    _assert_mode(
        repo_root=repo_root,
        mode="exit_nonzero",
        task_id="ci_llm_mode_exit_nonzero",
        expected_planner="rule_based_v1",
        expected_effective="rule_based_v1",
        expected_status="fallback",
        expected_reason="llm_command_failed",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

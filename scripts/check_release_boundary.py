#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Set


ALLOWED_UNTRACKED_PREFIXES = {
    "docs/examples/",
    "scripts/adapters/",
    "schemas/",
    "scripts/validate_",
}

ALLOWED_UNTRACKED_EXACT = {
    "docs/release_boundary.md",
    "docs/script_migration_whitelist.md",
    "docs/real_chain_minimal_acceptance.md",
    "docs/real_chain_acceptance_real.md",
    "docs/ui_prototype.md",
    "docs/script_migration_map.json",
    "scripts/check_release_boundary.py",
    "scripts/build_script_migration_map.py",
    "scripts/run_real_chain_acceptance_minimal.sh",
    "scripts/run_real_chain_acceptance_real.sh",
    "scripts/validate_data_report.py",
    "scripts/validate_model_report.py",
    "scripts/validate_filtering_report.py",
    "scripts/validate_task_state.py",
    "scripts/validate_run_artifacts.py",
    "scripts/validate_memory_context.py",
}

ALLOWED_UNTRACKED_PREFIXES = ALLOWED_UNTRACKED_PREFIXES | {
    "ui/",
}

BLOCKED_PATH_PREFIXES = {
    "logging/",
    "result/",
}


def _run_git_status(repo_root: Path) -> List[str]:
    cp = subprocess.run(
        ["git", "status", "--short"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if cp.returncode != 0:
        raise RuntimeError(cp.stderr.strip() or "git status failed")
    return [line.rstrip("\n") for line in cp.stdout.splitlines() if line.strip()]


def _parse_paths(lines: List[str]) -> Dict[str, List[str]]:
    tracked: List[str] = []
    untracked: List[str] = []
    for line in lines:
        if line.startswith("?? "):
            untracked.append(line[3:].strip())
        else:
            path = line[3:].strip() if len(line) > 3 else ""
            if path:
                tracked.append(path)
    return {"tracked": tracked, "untracked": untracked}


def _is_allowed_untracked(path: str) -> bool:
    if path in ALLOWED_UNTRACKED_EXACT:
        return True
    for prefix in ALLOWED_UNTRACKED_PREFIXES:
        if path.startswith(prefix):
            return True
    return False


def _is_blocked(path: str) -> bool:
    return any(path.startswith(p) for p in BLOCKED_PATH_PREFIXES)


def evaluate_release_boundary(repo_root: Path) -> Dict[str, object]:
    lines = _run_git_status(repo_root)
    parsed = _parse_paths(lines)
    tracked = parsed["tracked"]
    untracked = parsed["untracked"]

    blocked_paths: Set[str] = set()
    disallowed_untracked: List[str] = []

    for p in tracked + untracked:
        if _is_blocked(p):
            blocked_paths.add(p)

    for p in untracked:
        if not _is_allowed_untracked(p):
            disallowed_untracked.append(p)

    status = "pass"
    reasons: List[str] = []
    if blocked_paths:
        status = "fail"
        reasons.append("blocked_runtime_artifacts_present")
    if disallowed_untracked:
        status = "fail"
        reasons.append("disallowed_untracked_files")

    return {
        "status": status,
        "tracked_modified": tracked,
        "untracked": untracked,
        "blocked_paths": sorted(blocked_paths),
        "disallowed_untracked": sorted(disallowed_untracked),
        "reasons": reasons,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Check Agent4Mat release boundary hygiene")
    p.add_argument("--workspace-root", default=".", help="Repo root")
    p.add_argument("--json", action="store_true", help="Emit JSON")
    args = p.parse_args()

    repo_root = Path(args.workspace_root).resolve()
    report = evaluate_release_boundary(repo_root)

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"status={report['status']}")
        if report["blocked_paths"]:
            print("blocked runtime artifacts:")
            for pth in report["blocked_paths"]:
                print(f"  - {pth}")
        if report["disallowed_untracked"]:
            print("disallowed untracked files:")
            for pth in report["disallowed_untracked"]:
                print(f"  - {pth}")

    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def _resolve_workspace_root(path: Path) -> Path:
    # decision summary is typically in: <workspace>/runs/agent/<task_id>/decision_summary.json
    # fallback to script parent for direct invocations.
    try:
        return path.resolve().parents[3]
    except Exception:
        return Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate agent decision_summary.json schema.")
    parser.add_argument("path", help="Path to decision_summary.json")
    args = parser.parse_args()

    p = Path(args.path).resolve()
    if not p.exists():
        print(f"[FAIL] decision summary not found: {p}")
        return 1

    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"[FAIL] invalid JSON: {exc}")
        return 1

    workspace_root = _resolve_workspace_root(p)
    # Ensure package imports inside scripts/ work when called directly.
    repo_src = Path(__file__).resolve().parents[1] / "src"
    if str(repo_src) not in sys.path:
        sys.path.insert(0, str(repo_src))
    from oled_agent.agent.request_contract import RequestValidationError, validate_decision_summary_payload
    try:
        validate_decision_summary_payload(payload=payload, workspace_root=workspace_root)
    except RequestValidationError as exc:
        print(f"[FAIL] decision summary schema invalid: {exc}")
        return 1

    print(f"[PASS] decision summary schema valid: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

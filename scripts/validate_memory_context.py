from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def _resolve_workspace_root(path: Path) -> Path:
    # memory context is typically in: <workspace>/runs/agent/<task_id>/artifacts/memory_context.json
    try:
        return path.resolve().parents[4]
    except Exception:
        return Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate agent memory_context.json schema.")
    parser.add_argument("path", help="Path to memory_context.json")
    args = parser.parse_args()

    p = Path(args.path).resolve()
    if not p.exists():
        print(f"[FAIL] memory context not found: {p}")
        return 1

    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"[FAIL] invalid JSON: {exc}")
        return 1

    workspace_root = _resolve_workspace_root(p)
    repo_src = Path(__file__).resolve().parents[1] / "src"
    if str(repo_src) not in sys.path:
        sys.path.insert(0, str(repo_src))

    from oled_agent.agent.request_contract import RequestValidationError, validate_memory_context_payload

    try:
        validate_memory_context_payload(payload=payload, workspace_root=workspace_root)
    except RequestValidationError as exc:
        print(f"[FAIL] memory context schema invalid: {exc}")
        return 1

    print(f"[PASS] memory context schema valid: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from oled_agent.agent.tool_contracts import build_plan_tool_call_item_schema


def _git_sha() -> str:
    try:
        cp = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return (cp.stdout or "").strip()
    except Exception:
        return ""


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync or check plan.schema.json tool_calls.items")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only check whether schema is in sync; do not write file",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON summary",
    )
    args = parser.parse_args()

    schema_path = REPO_ROOT / "schemas" / "plan.schema.json"
    payload = json.loads(schema_path.read_text(encoding="utf-8"))

    expected_items = build_plan_tool_call_item_schema()
    current_items = payload["properties"]["tool_calls"]["items"]
    git_sha = _git_sha()

    if args.check:
        if current_items != expected_items:
            if args.json:
                print(
                    json.dumps(
                        {
                            "status": "fail",
                            "action": "check",
                            "schema_path": str(schema_path),
                            "git_sha": git_sha,
                            "reason": "out_of_sync",
                            "fix_command": "python3 scripts/sync_plan_tool_schema.py",
                        },
                        ensure_ascii=False,
                    )
                )
            else:
                print(f"[FAIL] out-of-sync tool call schema: {schema_path}")
                print("Run: python3 scripts/sync_plan_tool_schema.py")
            return 1
        if args.json:
            print(
                json.dumps(
                    {
                        "status": "pass",
                        "action": "check",
                        "schema_path": str(schema_path),
                        "git_sha": git_sha,
                    },
                    ensure_ascii=False,
                )
            )
        else:
            print(f"[PASS] tool call schema in sync: {schema_path}")
        return 0

    payload["properties"]["tool_calls"]["items"] = expected_items
    schema_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.json:
        print(
            json.dumps(
                {
                    "status": "pass",
                    "action": "sync",
                    "schema_path": str(schema_path),
                    "git_sha": git_sha,
                },
                ensure_ascii=False,
            )
        )
    else:
        print(f"[PASS] synced tool call schema: {schema_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

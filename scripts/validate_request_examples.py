#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

from oled_agent.agent.request_contract import RequestValidationError, validate_request_payload


def _iter_json_files(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    if not path.exists():
        return []
    return sorted(p for p in path.rglob("*.json") if p.is_file())


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Validate request JSON examples against request.schema.json")
    ap.add_argument(
        "--workspace-root",
        default=".",
        help="Workspace root used to resolve schemas (default: current directory)",
    )
    ap.add_argument(
        "--examples-dir",
        default="configs/request_templates",
        help="Directory containing request JSON examples",
    )
    ap.add_argument("--json", action="store_true", help="Emit machine-readable summary JSON")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    workspace_root = Path(args.workspace_root).resolve()
    examples_dir = (workspace_root / args.examples_dir).resolve()
    files = _iter_json_files(examples_dir)

    summary: Dict[str, object] = {
        "workspace_root": str(workspace_root),
        "examples_dir": str(examples_dir),
        "checked": 0,
        "passed": 0,
        "failed": 0,
        "results": [],
    }

    if not files:
        msg = f"no request examples found under: {examples_dir}"
        if args.json:
            summary["failed"] = 1
            summary["results"] = [{"path": str(examples_dir), "status": "fail", "error": msg}]
            print(json.dumps(summary, ensure_ascii=False, indent=2))
        else:
            print(f"[FAIL] {msg}")
        return 1

    exit_code = 0
    for path in files:
        summary["checked"] = int(summary["checked"]) + 1
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise RequestValidationError("root payload must be object")
            validate_request_payload(payload=payload, workspace_root=workspace_root)
            summary["passed"] = int(summary["passed"]) + 1
            cast_results = list(summary["results"])  # type: ignore[arg-type]
            cast_results.append({"path": str(path), "status": "pass"})
            summary["results"] = cast_results
        except Exception as exc:
            exit_code = 1
            summary["failed"] = int(summary["failed"]) + 1
            cast_results = list(summary["results"])  # type: ignore[arg-type]
            cast_results.append({"path": str(path), "status": "fail", "error": str(exc)})
            summary["results"] = cast_results

    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        for item in summary["results"]:  # type: ignore[index]
            if item["status"] == "pass":
                print(f"[PASS] {item['path']}")
            else:
                print(f"[FAIL] {item['path']}: {item.get('error', '')}")
        print(
            f"REQUEST_EXAMPLES checked={summary['checked']} pass={summary['passed']} fail={summary['failed']}"
        )

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

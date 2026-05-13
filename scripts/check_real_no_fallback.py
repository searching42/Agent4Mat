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
        req = {
            "task_id": "ci_real_no_fallback",
            "request_text": "设计470nm附近且高PLQY分子",
            "mode": "fast_screen",
            "targets": [{"property": "plqy", "objective": "maximize", "target_value": 60.0}],
            "budget": {"max_candidates": 10},
            "model_preferences": {
                "predictor_id": "unimol_lambda_plqy_real_v1",
                "generator_id": "reinvent4_generator_real_v1"
            }
        }
        req_path = tdp / "request.json"
        req_path.write_text(json.dumps(req, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        env = {
            "PYTHONPATH": str(root / "src"),
            "OLED_AGENT_REINVENT4_ADAPTER_MODE": "smoke",
            "OLED_AGENT_UNIMOL_SCORE_MODE": "smoke",
        }

        cp = subprocess.run(
            [
                sys.executable,
                "-m",
                "oled_agent.cli",
                "agent-run-json",
                "--workspace-root",
                str(root),
                "--catalog",
                str(root / "scripts" / "adapters" / "real_adapters_catalog.json"),
                "--request-json",
                str(req_path),
                "--require-real-adapters",
            ],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )
        # smoke real-adapters path should pass and not use local fallback.
        if cp.returncode != 0:
            print(cp.stdout)
            print(cp.stderr)
            return cp.returncode

    print(json.dumps({"status": "pass", "check": "require-real-adapters"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

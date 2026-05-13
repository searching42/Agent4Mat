#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

from oled_agent.agent.request_contract import validate_step_request_payload, validate_task_v2_payload
from oled_agent.agent.task_v2 import infer_task_draft


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    task = infer_task_draft(request_text="设计470nm附近且高PLQY分子", task_id="ci_intake")
    task["property"] = "plqy"
    task["range"] = "60-100"
    task["candidate_data"] = "dummy.csv"
    task["status"] = "approved"

    validate_task_v2_payload(task, root)

    step_req = {
        "task": task,
        "operation": "train_predictor",
        "args": {
            "predictor_id": "unimol_lambda_plqy_v1",
            "targets": ["plqy"],
        },
    }
    validate_step_request_payload(step_req, root)

    print(json.dumps({"status": "pass", "checks": ["task.v2", "step_request"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

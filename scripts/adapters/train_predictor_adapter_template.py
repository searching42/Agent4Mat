#!/usr/bin/env python3
"""JSON-in/JSON-out adapter template for `train_predictor`.

Expected stdin payload keys:
- `predictor_id`, `targets`, `target_specs`, `task_id`, ...
"""
from __future__ import annotations

import json
import sys


def main() -> int:
    payload = json.load(sys.stdin)
    predictor_id = str(payload.get("predictor_id") or "external_predictor")
    targets = payload.get("targets") or []
    print(
        json.dumps(
            {
                "status": "success",
                "adapter": "template_train_cmd",
                "predictor_id": predictor_id,
                "targets": targets,
                "metrics": {
                    "samples": 0,
                    "note": "replace template logic with real training backend",
                },
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

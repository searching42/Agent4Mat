#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

from oled_agent.agent.intake import run_intake


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    with mock.patch("oled_agent.agent.intake.run_duckduckgo_search", return_value=[{"title": "demo", "url": "https://example.com"}]):
        out = run_intake(
            workspace_root=root,
            task_id="ci_web_evidence",
            request_text="设计470nm附近且高PLQY分子",
            enable_web_search=True,
            web_topk=1,
        )
    web_path = Path(out["web_evidence_path"])
    if not web_path.exists():
        print(json.dumps({"status": "fail", "reason": "web_evidence_path_missing"}, ensure_ascii=False))
        return 1
    payload = json.loads(web_path.read_text(encoding="utf-8"))
    if not isinstance(payload.get("evidence"), list):
        print(json.dumps({"status": "fail", "reason": "evidence_not_list"}, ensure_ascii=False))
        return 1
    print(json.dumps({"status": "pass", "evidence_count": len(payload.get("evidence", []))}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

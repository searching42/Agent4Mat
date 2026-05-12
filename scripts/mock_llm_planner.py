#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _load_catalog_defaults(catalog_path: str) -> Tuple[str, str]:
    path = Path(catalog_path)
    if not path.exists():
        return "", ""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return "", ""
    predictor = ""
    generator = ""
    for item in payload.get("models", []) or []:
        if not isinstance(item, dict):
            continue
        kind = str(item.get("kind") or "").strip()
        mid = str(item.get("id") or "").strip()
        if kind == "predictor" and not predictor and mid:
            predictor = mid
        if kind == "generator" and not generator and mid:
            generator = mid
    return predictor, generator


def _read_request() -> Dict[str, Any]:
    raw = sys.stdin.read()
    if not raw.strip():
        raise ValueError("stdin is empty")
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("input must be JSON object")
    return payload


def _ensure_targets(request_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    targets = request_payload.get("targets")
    if isinstance(targets, list) and targets:
        return [t for t in targets if isinstance(t, dict)]
    return [{"property": "plqy", "objective": "maximize", "target_value": 0.6}]


def main() -> int:
    wrapper = _read_request()
    request_payload = wrapper.get("request")
    if not isinstance(request_payload, dict):
        raise ValueError("input.request must be object")

    predictor_default, generator_default = _load_catalog_defaults(str(wrapper.get("catalog_path") or ""))
    prefs = request_payload.get("model_preferences")
    if not isinstance(prefs, dict):
        prefs = {}
    predictor_id = str(prefs.get("predictor_id") or predictor_default or "")
    generator_id = str(prefs.get("generator_id") or generator_default or "")
    targets = _ensure_targets(request_payload)
    mode = str(request_payload.get("mode") or "fast_screen")
    max_candidates = int((request_payload.get("budget") or {}).get("max_candidates") or 500)
    mock_mode = str(os.environ.get("MOCK_LLM_MODE") or "active").strip()

    tool_calls: List[Dict[str, Any]] = [
        {"name": "list_models", "args": {"kind": "predictor"}},
        {"name": "list_models", "args": {"kind": "generator"}},
        {"name": "search_dataset", "args": {"preferences": ["master_database", "subsidiary_database"]}},
    ]
    if mode == "train_then_design":
        tool_calls.append(
            {
                "name": "train_predictor",
                "args": {
                    "predictor_id": predictor_id,
                    "targets": [str(t.get("property") or "") for t in targets],
                },
            }
        )
    tool_calls.extend(
        [
            {
                "name": "generate_candidates",
                "args": {
                    "generator_id": generator_id,
                    "max_candidates": max_candidates,
                    "constraints": request_payload.get("constraints") if isinstance(request_payload.get("constraints"), dict) else {},
                },
            },
            {
                "name": "score_candidates",
                "args": {
                    "predictor_id": predictor_id,
                    "targets": [str(t.get("property") or "") for t in targets],
                },
            },
            {"name": "filter_and_rank", "args": {"topn": 10}},
            {"name": "make_report", "args": {}},
        ]
    )

    if mock_mode == "bad_json":
        print("NOT_JSON")
        return 0
    if mock_mode == "exit_nonzero":
        return 3
    if mock_mode == "bad_tools":
        tool_calls = [{"name": "unsupported_tool", "args": {}}]
    if mock_mode == "alias_load_model_catalog":
        tool_calls = [
            {"name": "load_model_catalog", "args": {"kinds": ["predictor", "generator"]}},
            {"name": "search_dataset", "args": {"preferences": ["master_database", "subsidiary_database"]}},
            {
                "name": "generate_candidates",
                "args": {
                    "generator_id": generator_id,
                    "max_candidates": max_candidates,
                    "constraints": request_payload.get("constraints") if isinstance(request_payload.get("constraints"), dict) else {},
                },
            },
            {
                "name": "score_candidates",
                "args": {
                    "predictor_id": predictor_id,
                    "targets": [str(t.get("property") or "") for t in targets],
                },
            },
            {"name": "filter_and_rank", "args": {"topn": 10}},
            {"name": "make_report", "args": {}},
        ]
    if mock_mode == "active_with_extra_args":
        tool_calls = [
            {"name": "list_models", "args": {"kind": "predictor", "unexpected": True}},
            {"name": "list_models", "args": {"kind": "generator", "note": "extra"}},
            {"name": "search_dataset", "args": {"preferences": ["master_database", "subsidiary_database"]}},
            {
                "name": "generate_candidates",
                "args": {
                    "generator_id": generator_id,
                    "max_candidates": max_candidates,
                    "constraints": request_payload.get("constraints") if isinstance(request_payload.get("constraints"), dict) else {},
                    "mode": mode,
                    "design_bias": "keep-diversity",
                },
            },
            {
                "name": "score_candidates",
                "args": {
                    "predictor_id": predictor_id,
                    "targets": [str(t.get("property") or "") for t in targets],
                    "extra_tag": "x",
                },
            },
            {"name": "filter_and_rank", "args": {"topn": 10, "unused": 1}},
            {"name": "make_report", "args": {"verbose": True}},
        ]
    if mock_mode == "bad_model":
        predictor_id = "not_exists_predictor"
        generator_id = "not_exists_generator"

    out = {
        "summary": "Mock LLM planner output",
        "design_spec": {
            "targets": targets,
            "constraints": request_payload.get("constraints") if isinstance(request_payload.get("constraints"), dict) else {},
            "budget": {"max_candidates": max_candidates},
            "model_choice": {
                "predictor_id": predictor_id,
                "generator_id": generator_id,
            },
        },
        "tool_calls": tool_calls,
    }
    print(json.dumps(out, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

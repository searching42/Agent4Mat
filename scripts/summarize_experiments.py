#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List


def _load_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def _row(trace: Dict[str, Any], trace_path: Path) -> Dict[str, Any]:
    exec_summary = trace.get("execution_summary") if isinstance(trace.get("execution_summary"), dict) else {}
    model_choice = trace.get("model_choice") if isinstance(trace.get("model_choice"), dict) else {}
    return {
        "task_id": str(trace.get("task_id") or ""),
        "run_label": str(trace.get("run_label") or ""),
        "generated_at": str(trace.get("generated_at") or ""),
        "execution_mode": str(trace.get("execution_mode") or ""),
        "status": str(exec_summary.get("status") or ""),
        "record_count": _safe_int(exec_summary.get("record_count")),
        "failed_count": _safe_int(exec_summary.get("failed_count")),
        "predictor_id": str(model_choice.get("predictor_id") or ""),
        "generator_id": str(model_choice.get("generator_id") or ""),
        "trace_path": str(trace_path),
    }


def _summary(rows: List[Dict[str, Any]], *, limit: int, workspace_root: Path) -> Dict[str, Any]:
    by_status = Counter()
    by_mode = Counter()
    by_predictor = Counter()
    by_generator = Counter()
    for r in rows:
        by_status[str(r.get("status") or "unknown")] += 1
        by_mode[str(r.get("execution_mode") or "unknown")] += 1
        by_predictor[str(r.get("predictor_id") or "")] += 1
        by_generator[str(r.get("generator_id") or "")] += 1
    sorted_rows = sorted(rows, key=lambda x: str(x.get("generated_at") or ""), reverse=True)
    return {
        "status": "pass",
        "workspace_root": str(workspace_root),
        "count": len(rows),
        "limit": limit,
        "summary": {
            "by_status": dict(by_status),
            "by_execution_mode": dict(by_mode),
            "by_predictor_id": dict(by_predictor),
            "by_generator_id": dict(by_generator),
        },
        "recent": sorted_rows[:limit],
    }


def _to_markdown(payload: Dict[str, Any]) -> str:
    lines = [
        "# Experiment Summary",
        "",
        f"- workspace_root: {payload.get('workspace_root', '')}",
        f"- count: {payload.get('count', 0)}",
        "",
        "## By Status",
    ]
    by_status = payload.get("summary", {}).get("by_status", {})
    if isinstance(by_status, dict):
        for key, value in sorted(by_status.items(), key=lambda kv: kv[0]):
            lines.append(f"- {key}: {value}")
    lines.extend(["", "## Recent"])
    recent = payload.get("recent", [])
    if isinstance(recent, list):
        for idx, item in enumerate(recent, start=1):
            if not isinstance(item, dict):
                continue
            lines.append(
                f"{idx}. task_id={item.get('task_id','')} run_label={item.get('run_label','')} "
                f"status={item.get('status','')} mode={item.get('execution_mode','')} "
                f"records={item.get('record_count',0)} failed={item.get('failed_count',0)}"
            )
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize experiment_trace artifacts")
    p.add_argument("--workspace-root", default=".")
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--json-out", default="")
    p.add_argument("--md-out", default="")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.workspace_root).resolve()
    runs_root = root / "runs" / "agent"
    rows: List[Dict[str, Any]] = []
    if runs_root.exists():
        for child in runs_root.iterdir():
            if not child.is_dir():
                continue
            trace_path = child / "artifacts" / "experiment_trace.json"
            if not trace_path.exists():
                continue
            try:
                trace = _load_json(trace_path)
                rows.append(_row(trace, trace_path))
            except Exception:
                continue
    payload = _summary(rows, limit=max(1, int(args.limit)), workspace_root=root)
    if str(args.json_out).strip():
        out = Path(args.json_out).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if str(args.md_out).strip():
        out_md = Path(args.md_out).resolve()
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(_to_markdown(payload), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


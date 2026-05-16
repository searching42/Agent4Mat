from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict


def _load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_path(raw: str, cwd: Path) -> Path:
    p = Path(raw)
    if not p.is_absolute():
        p = (cwd / p).resolve()
    return p


def _paths_from_result_json(result_json_path: Path, cwd: Path) -> Dict[str, Path]:
    payload = _load_json(result_json_path)
    required = {
        "decision_summary": "decision_summary_path",
        "task_state": "task_state_path",
        "data_report": "logging_data_report_path",
        "model_report": "logging_model_report_path",
        "filtering_report": "logging_filtering_report_path",
        "evaluation_report": "logging_evaluation_report_path",
        "guardrails_report": "logging_guardrails_report_path",
    }
    out: Dict[str, Path] = {}
    for logical, key in required.items():
        raw = str(payload.get(key) or "").strip()
        if not raw:
            raise ValueError(f"result json missing required key: {key}")
        out[logical] = _resolve_path(raw, cwd)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate Agent4Mat structured run artifacts")
    p.add_argument("--workspace-root", default=".", help="Workspace root containing schemas/")
    p.add_argument("--result-json", default="", help="Path to agent-run output JSON containing artifact paths")
    p.add_argument("--decision-summary", default="", help="Path to decision_summary.json")
    p.add_argument("--task-state", default="", help="Path to task_state.json")
    p.add_argument("--data-report", default="", help="Path to data_report.json")
    p.add_argument("--model-report", default="", help="Path to model_report.json")
    p.add_argument("--filtering-report", default="", help="Path to filtering_report.json")
    p.add_argument("--evaluation-report", default="", help="Path to evaluation_report.json")
    p.add_argument("--guardrails-report", default="", help="Path to guardrails_report.json")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cwd = Path.cwd()
    workspace_root = Path(args.workspace_root).resolve()

    repo_src = Path(__file__).resolve().parents[1] / "src"
    if str(repo_src) not in sys.path:
        sys.path.insert(0, str(repo_src))

    from oled_agent.agent.request_contract import (
        RequestValidationError,
        validate_data_report_payload,
        validate_decision_summary_payload,
        validate_evaluation_report_payload,
        validate_filtering_report_payload,
        validate_guardrails_report_payload,
        validate_model_report_payload,
        validate_task_state_payload,
    )

    try:
        if str(args.result_json).strip():
            result_json_path = _resolve_path(str(args.result_json).strip(), cwd)
            if not result_json_path.exists():
                print(f"[FAIL] result json not found: {result_json_path}")
                return 1
            paths = _paths_from_result_json(result_json_path, cwd)
        else:
            raw_paths = {
                "decision_summary": str(args.decision_summary).strip(),
                "task_state": str(args.task_state).strip(),
                "data_report": str(args.data_report).strip(),
                "model_report": str(args.model_report).strip(),
                "filtering_report": str(args.filtering_report).strip(),
                "evaluation_report": str(args.evaluation_report).strip(),
                "guardrails_report": str(args.guardrails_report).strip(),
            }
            missing = [k for k, v in raw_paths.items() if not v]
            if missing:
                print(
                    "[FAIL] missing required args without --result-json: "
                    + ", ".join(missing)
                )
                return 1
            paths = {k: _resolve_path(v, cwd) for k, v in raw_paths.items()}

        for key, path in paths.items():
            if not path.exists():
                print(f"[FAIL] artifact not found ({key}): {path}")
                return 1

        decision_payload = _load_json(paths["decision_summary"])
        task_state_payload = _load_json(paths["task_state"])
        data_report_payload = _load_json(paths["data_report"])
        model_report_payload = _load_json(paths["model_report"])
        filtering_report_payload = _load_json(paths["filtering_report"])
        evaluation_report_payload = _load_json(paths["evaluation_report"])
        guardrails_report_payload = _load_json(paths["guardrails_report"])

        validate_decision_summary_payload(payload=decision_payload, workspace_root=workspace_root)
        print(f"[PASS] decision summary schema valid: {paths['decision_summary']}")
        validate_task_state_payload(payload=task_state_payload, workspace_root=workspace_root)
        print(f"[PASS] task state schema valid: {paths['task_state']}")
        validate_data_report_payload(payload=data_report_payload, workspace_root=workspace_root)
        print(f"[PASS] data report schema valid: {paths['data_report']}")
        validate_model_report_payload(payload=model_report_payload, workspace_root=workspace_root)
        print(f"[PASS] model report schema valid: {paths['model_report']}")
        validate_filtering_report_payload(payload=filtering_report_payload, workspace_root=workspace_root)
        print(f"[PASS] filtering report schema valid: {paths['filtering_report']}")
        validate_evaluation_report_payload(payload=evaluation_report_payload, workspace_root=workspace_root)
        print(f"[PASS] evaluation report schema valid: {paths['evaluation_report']}")
        validate_guardrails_report_payload(payload=guardrails_report_payload, workspace_root=workspace_root)
        print(f"[PASS] guardrails report schema valid: {paths['guardrails_report']}")
        return 0
    except RequestValidationError as exc:
        print(f"[FAIL] artifact schema invalid: {exc}")
        return 1
    except ValueError as exc:
        print(f"[FAIL] invalid artifact inputs: {exc}")
        return 1
    except json.JSONDecodeError as exc:
        print(f"[FAIL] invalid JSON payload: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

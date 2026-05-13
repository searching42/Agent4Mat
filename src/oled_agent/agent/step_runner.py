from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from oled_agent.agent.request_contract import validate_step_request_payload, validate_task_v2_payload
from oled_agent.agent.task_v2 import task_v2_to_request_payload
from oled_agent.agent.tools import ToolContext, execute_tool

DEFAULT_LOGGING_OUT = Path("logging")
DEFAULT_RESULT_OUT = Path("result")
TASK_STATE_SCHEMA_VERSION = "1.0.0"

_STEP_TOOL_STATE_MAP = {
    "retrieve_candidate_data": "DATA_ACQUISITION",
    "clean_dataset": "PREPROCESSING",
    "prepare_train_data": "PREPROCESSING",
    "train_predictor": "TRAINING_OPTIONAL",
    "generate_candidates": "PREPROCESSING",
    "score_candidates": "INFERENCE",
    "filter_and_rank": "FILTERING",
    "make_report": "REPORTING",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_slug(value: str) -> str:
    out = "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in str(value or "").strip())
    return out or "task"


def _build_run_label(task_id: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{_safe_slug(task_id)}-{ts}"


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _op_to_tool_name(operation: str) -> str:
    op = str(operation or "").strip()
    mapping = {
        "retrieve_candidate_data": "retrieve_candidate_data",
        "clean_dataset": "clean_dataset",
        "prepare_train_data": "prepare_train_data",
        "train_predictor": "train_predictor",
        "generate_candidates": "generate_candidates",
        "score_candidates": "score_candidates",
        "filter_and_rank": "filter_and_rank",
        "make_report": "make_report",
    }
    if op not in mapping:
        raise ValueError(f"Unsupported operation: {op}")
    return mapping[op]


def _build_task_state(*, task_id: str, execution_status: str, tool_name: str, started: str, ended: str) -> Dict[str, Any]:
    mapped_state = _STEP_TOOL_STATE_MAP.get(tool_name, "ROUTING")
    history = [
        {"state": "INIT", "status": "completed", "at": started},
        {"state": "REQUIREMENT_COLLECTION", "status": "completed", "at": started},
        {"state": "VALIDATION", "status": "completed", "at": started},
        {"state": mapped_state, "status": "success" if execution_status == "success" else "failed", "started_at": started, "ended_at": ended},
        {"state": "SAVING", "status": "completed", "at": ended},
        {"state": "QA", "status": "completed" if execution_status == "success" else "failed", "at": ended},
        {"state": "DONE" if execution_status == "success" else "FAILED", "status": execution_status, "at": ended},
    ]
    return {
        "schema_version": TASK_STATE_SCHEMA_VERSION,
        "generated_at": _now_iso(),
        "task_id": task_id,
        "status": execution_status,
        "current_state": history[-1]["state"],
        "history": history,
    }


def _build_decision_summary(*, task_id: str, tool_name: str, result: Dict[str, Any], tool_state: Dict[str, Any], status: str) -> Dict[str, Any]:
    adapter = ""
    fallback_reason = ""
    fallback_error: Dict[str, Any] = {}
    used_fallback = False
    if tool_name == "score_candidates":
        adapter = str(result.get("adapter") or "")
        fallback_reason = str(result.get("fallback_reason") or "")
        fallback_error = result.get("fallback_error") if isinstance(result.get("fallback_error"), dict) else {}
        used_fallback = adapter == "local_deterministic_fallback"
    return {
        "schema_version": "1.0.0",
        "generated_at": _now_iso(),
        "task_id": task_id,
        "status": status,
        "model_choice": {},
        "score_step": {
            "adapter": adapter,
            "used_fallback": used_fallback,
            "fallback_reason": fallback_reason,
            "fallback_code": str(fallback_error.get("code") or ""),
            "fallback_retryable": bool(fallback_error.get("retryable", False)),
            "fallback_details": fallback_error.get("details") if isinstance(fallback_error.get("details"), dict) else {},
        },
        "artifacts": {
            "candidate_csv": str(tool_state.get("candidate_csv") or ""),
            "scored_csv": str(tool_state.get("scored_csv") or ""),
            "final_output": str(tool_state.get("final_output") or ""),
        },
    }


def _copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def _rel_or_abs(path_value: str, workspace_root: Path) -> Path:
    p = Path(path_value)
    if not p.is_absolute():
        p = (workspace_root / p).resolve()
    return p


def _write_logging_and_result(
    *,
    workspace_root: Path,
    run_label: str,
    task_payload: Dict[str, Any],
    execution: Dict[str, Any],
    tool_state: Dict[str, Any],
) -> Dict[str, str]:
    logging_dir = (workspace_root / DEFAULT_LOGGING_OUT / run_label).resolve()
    result_dir = (workspace_root / DEFAULT_RESULT_OUT / run_label).resolve()
    logging_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    logging_task_path = logging_dir / "task.json"
    _write_json(logging_task_path, task_payload)
    (logging_dir / "plan.md").write_text(
        "# Step Plan\n\n"
        f"- task_id: {task_payload.get('task_id', '')}\n"
        f"- operation: {task_payload.get('operation', '')}\n"
        f"- execution_mode: {task_payload.get('execution_mode', '')}\n",
        encoding="utf-8",
    )
    (logging_dir / "execution.log").write_text(json.dumps(execution, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    exec_status = _ensure_status(execution.get("status"))
    data_report = {
        "schema_version": "1.0.0",
        "generated_at": _now_iso(),
        "task_id": str(task_payload.get("task_id") or ""),
        "status": exec_status,
        "dataset_step": {"status": "not_run", "selected": [], "available": []},
        "candidate_step": {
            "status": "not_run",
            "adapter": "",
            "rows": 0,
            "source_csv": "",
            "input_csv": "",
        },
        "artifacts": {
            "candidate_csv": str(tool_state.get("candidate_csv") or ""),
            "scored_csv": str(tool_state.get("scored_csv") or ""),
        },
    }
    model_report = {
        "schema_version": "1.0.0",
        "generated_at": _now_iso(),
        "task_id": str(task_payload.get("task_id") or ""),
        "status": exec_status,
        "model_choice": {"predictor_id": "", "generator_id": ""},
        "training_step": {"ran": False, "status": "not_run", "adapter": "", "result": {}},
        "inference_step": {
            "status": "not_run",
            "adapter": "",
            "used_fallback": False,
            "fallback_error": {},
            "result": {},
        },
    }
    filtering_report = {
        "schema_version": "1.0.0",
        "generated_at": _now_iso(),
        "task_id": str(task_payload.get("task_id") or ""),
        "status": exec_status,
        "filter_step": {"status": "not_run", "topn": 0, "manifest": "", "final_output": ""},
        "report_step": {"status": "not_run", "report": "", "latest_run_dir": ""},
    }
    _write_json(logging_dir / "data_report.json", data_report)
    _write_json(logging_dir / "model_report.json", model_report)
    _write_json(logging_dir / "filtering_report.json", filtering_report)

    target_structures_csv = result_dir / "target_structures.csv"
    copied = False
    for raw in (str(tool_state.get("scored_csv") or ""), str(tool_state.get("candidate_csv") or "")):
        if not raw:
            continue
        src = _rel_or_abs(raw, workspace_root)
        if _copy_if_exists(src, target_structures_csv):
            copied = True
            break

    report_md_out = result_dir / "report.md"
    final_output = str(tool_state.get("final_output") or "")
    if final_output:
        src_report = _rel_or_abs(final_output, workspace_root)
        _copy_if_exists(src_report, report_md_out)

    metadata = {
        "schema_version": "1.0.0",
        "generated_at": _now_iso(),
        "task_id": str(task_payload.get("task_id") or ""),
        "status": str(execution.get("status") or ""),
        "source_artifacts": {
            "candidate_csv": str(tool_state.get("candidate_csv") or ""),
            "scored_csv": str(tool_state.get("scored_csv") or ""),
            "final_output": final_output,
        },
        "outputs": {
            "target_structures_csv": str(target_structures_csv) if copied else "",
            "report_md": str(report_md_out) if report_md_out.exists() else "",
        },
    }
    metadata_path = result_dir / "metadata.json"
    _write_json(metadata_path, metadata)
    return {
        "logging_dir": str(logging_dir),
        "result_dir": str(result_dir),
        "logging_task_path": str(logging_task_path),
        "logging_plan_md_path": str(logging_dir / "plan.md"),
        "logging_execution_log_path": str(logging_dir / "execution.log"),
        "logging_data_report_path": str(logging_dir / "data_report.json"),
        "logging_model_report_path": str(logging_dir / "model_report.json"),
        "logging_filtering_report_path": str(logging_dir / "filtering_report.json"),
        "result_metadata_path": str(metadata_path),
        "result_target_structures_csv_path": str(target_structures_csv) if copied else "",
    }


def _default_args_from_task(task_payload: Dict[str, Any], operation: str) -> Dict[str, Any]:
    req = task_v2_to_request_payload(task_payload)
    prefs = req.get("model_preferences") if isinstance(req.get("model_preferences"), dict) else {}
    predictor_id = str(prefs.get("predictor_id") or "")
    generator_id = str(prefs.get("generator_id") or "")
    constraints = req.get("constraints") if isinstance(req.get("constraints"), dict) else {}
    targets = req.get("targets") if isinstance(req.get("targets"), list) else []
    target_names = [str(t.get("property") or "") for t in targets if isinstance(t, dict) and str(t.get("property") or "").strip()]

    cdata = str(task_payload.get("candidate_data") or "")
    tdata = str(task_payload.get("train_data") or "")

    if operation == "retrieve_candidate_data":
        return {"candidate_data": cdata}
    if operation == "clean_dataset":
        return {"constraints": constraints}
    if operation == "prepare_train_data":
        return {"train_data": tdata}
    if operation == "train_predictor":
        return {"predictor_id": predictor_id, "targets": target_names or ["plqy"]}
    if operation == "generate_candidates":
        args: Dict[str, Any] = {
            "generator_id": generator_id,
            "max_candidates": int(req.get("budget", {}).get("max_candidates", 500)),
            "constraints": constraints,
            "input_csv": cdata,
        }
        gen_in = req.get("generation_input") if isinstance(req.get("generation_input"), dict) else {}
        args.update(gen_in)
        return args
    if operation == "score_candidates":
        return {"predictor_id": predictor_id, "targets": target_names or ["plqy"]}
    if operation == "filter_and_rank":
        return {"topn": min(10, int(task_payload.get("n_structures") or 10))}
    if operation == "make_report":
        return {}
    return {}


def _ensure_status(value: Any) -> str:
    text = str(value or "").strip()
    if text in ("success", "failed"):
        return text
    return "failed"


def run_step(
    *,
    workspace_root: Path,
    task_payload: Dict[str, Any],
    operation: str,
    args_override: Optional[Dict[str, Any]],
    catalog_path: Path,
) -> Dict[str, Any]:
    validate_task_v2_payload(task_payload, workspace_root)
    tool_name = _op_to_tool_name(operation)

    task_id = str(task_payload.get("task_id") or "task_default")
    out_dir = (workspace_root / "runs" / "agent" / task_id).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    state_path = out_dir / "step_tool_state.json"
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except Exception:
            state = {}
    else:
        state = {}

    if not isinstance(state, dict):
        state = {}

    ctx = ToolContext(
        workspace_root=workspace_root.resolve(),
        catalog_path=catalog_path,
        task_id=task_id,
        state=state,
    )

    args = _default_args_from_task(task_payload, operation)
    if isinstance(args_override, dict):
        args.update(args_override)

    started = _now_iso()
    status = "success"
    err = ""
    result: Dict[str, Any] = {}
    try:
        result = execute_tool(ctx, tool_name, args)
    except Exception as exc:
        status = "failed"
        err = str(exc)

    ended = _now_iso()
    execution = {
        "task_id": task_id,
        "status": status,
        "started_at": started,
        "ended_at": ended,
        "records": [
            {
                "name": tool_name,
                "args": args,
                "started_at": started,
                "ended_at": ended,
                "status": status,
                "result": result,
                "error": err,
            }
        ],
    }

    tool_state = dict(ctx.state)
    execution_path = out_dir / "execution.json"
    tool_state_path = out_dir / "tool_state.json"
    task_path = out_dir / "task.step.json"
    decision_summary_path = out_dir / "decision_summary.json"
    task_state_path = out_dir / "task_state.json"

    _write_json(state_path, tool_state)
    _write_json(execution_path, execution)
    _write_json(tool_state_path, tool_state)
    _write_json(task_path, task_payload)
    _write_json(
        decision_summary_path,
        _build_decision_summary(
            task_id=task_id,
            tool_name=tool_name,
            result=result if isinstance(result, dict) else {},
            tool_state=tool_state,
            status=status,
        ),
    )
    _write_json(
        task_state_path,
        _build_task_state(
            task_id=task_id,
            execution_status=status,
            tool_name=tool_name,
            started=started,
            ended=ended,
        ),
    )
    run_label = _build_run_label(task_id)
    mirror = _write_logging_and_result(
        workspace_root=workspace_root.resolve(),
        run_label=run_label,
        task_payload=task_payload,
        execution=execution,
        tool_state=tool_state,
    )

    return {
        "task_id": task_id,
        "status": status,
        "operation": operation,
        "tool_name": tool_name,
        "execution_path": str(execution_path),
        "tool_state_path": str(tool_state_path),
        "decision_summary_path": str(decision_summary_path),
        "task_state_path": str(task_state_path),
        "task_path": str(task_path),
        "run_label": run_label,
        **mirror,
        "result": result,
        "error": err,
    }


def run_step_from_request_payload(
    *,
    workspace_root: Path,
    step_request_payload: Dict[str, Any],
    catalog_path: Path,
) -> Dict[str, Any]:
    validate_step_request_payload(step_request_payload, workspace_root)
    task = step_request_payload.get("task") if isinstance(step_request_payload.get("task"), dict) else {}
    validate_task_v2_payload(task, workspace_root)
    op = str(step_request_payload.get("operation") or "")
    args = step_request_payload.get("args") if isinstance(step_request_payload.get("args"), dict) else {}
    return run_step(
        workspace_root=workspace_root,
        task_payload=task,
        operation=op,
        args_override=args,
        catalog_path=catalog_path,
    )

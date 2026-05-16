from __future__ import annotations

import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from oled_agent.agent.evaluator import build_evaluation_report
from oled_agent.agent.experiment_trace import build_experiment_trace
from oled_agent.agent.guardrails import build_guardrails_report
from oled_agent.agent.memory_context import build_memory_context, update_memory_index
from oled_agent.agent.executor import execute_plan, execute_plan_with_resume, save_execution_result
from oled_agent.agent.planner import (
    DEFAULT_PLANNER_PROVIDER,
    build_plan,
    build_plan_from_request_payload,
)
from oled_agent.agent.request_contract import validate_request_payload
from oled_agent.agent.tools import ToolContext


DEFAULT_CATALOG = Path("configs/models/catalog.json")
DEFAULT_AGENT_OUT = Path("runs/agent")
DECISION_SUMMARY_SCHEMA_VERSION = "1.0.0"
TASK_STATE_SCHEMA_VERSION = "1.0.0"
DEFAULT_LOGGING_OUT = Path("logging")
DEFAULT_RESULT_OUT = Path("result")

_EARLY_STATES = [
    "INIT",
    "REQUIREMENT_COLLECTION",
    "VALIDATION",
    "PLAN_GENERATION",
    "USER_CONFIRMATION",
]
_TOOL_STATE_MAP = {
    "list_models": "ROUTING",
    "search_web_evidence": "DATA_ACQUISITION",
    "search_dataset": "DATA_ACQUISITION",
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
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "").strip())
    return clean or "task"


def _build_run_label(task_id: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{_safe_slug(task_id)}-{ts}"


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _load_json_if_exists(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _copy_if_exists(src: Optional[Path], dst: Path) -> bool:
    if src is None or not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def _build_task_payload_from_plan(plan: Dict[str, Any]) -> Dict[str, Any]:
    design = plan.get("design_spec", {}) if isinstance(plan, dict) else {}
    return {
        "task_id": design.get("task_id", ""),
        "request_text": design.get("user_request", ""),
        "mode": design.get("mode", ""),
        "domain": design.get("domain", ""),
        "targets": design.get("targets", []),
        "constraints": design.get("constraints", {}),
        "model_choice": design.get("model_choice", {}),
        "budget": design.get("budget", {}),
    }


def _build_plan_markdown(plan: Dict[str, Any]) -> str:
    summary = str(plan.get("summary") or "").strip()
    design = plan.get("design_spec", {}) if isinstance(plan.get("design_spec"), dict) else {}
    tool_calls = plan.get("tool_calls", []) if isinstance(plan.get("tool_calls"), list) else []

    lines = []
    lines.append("# Agent Plan")
    if summary:
        lines.append("")
        lines.append(f"- summary: {summary}")
    lines.append(f"- task_id: {design.get('task_id', '')}")
    lines.append(f"- mode: {design.get('mode', '')}")
    model_choice = design.get("model_choice", {}) if isinstance(design.get("model_choice"), dict) else {}
    lines.append(f"- predictor_id: {model_choice.get('predictor_id', '')}")
    lines.append(f"- generator_id: {model_choice.get('generator_id', '')}")

    lines.append("")
    lines.append("## Steps")
    for i, call in enumerate(tool_calls, start=1):
        if not isinstance(call, dict):
            continue
        name = str(call.get("name") or "")
        args = call.get("args", {})
        lines.append(f"{i}. `{name}`")
        if isinstance(args, dict) and args:
            lines.append(f"   - args: `{json.dumps(args, ensure_ascii=False)}`")
    lines.append("")
    return "\n".join(lines)


def _build_task_state(
    *,
    plan: Dict[str, Any],
    execution: Dict[str, Any],
) -> Dict[str, Any]:
    execution_status = "success" if str(execution.get("status") or "").strip() == "success" else "failed"
    history = []
    for state in _EARLY_STATES:
        history.append(
            {
                "state": state,
                "status": "completed",
                "at": _now_iso(),
            }
        )

    records = execution.get("records", []) if isinstance(execution, dict) else []
    for rec in records:
        if not isinstance(rec, dict):
            continue
        tool_name = str(rec.get("name") or "")
        mapped_state = _TOOL_STATE_MAP.get(tool_name, "ROUTING")
        history.append(
            {
                "state": mapped_state,
                "status": rec.get("status", "unknown"),
                "tool": tool_name,
                "started_at": rec.get("started_at", ""),
                "ended_at": rec.get("ended_at", ""),
            }
        )

    history.append(
        {
            "state": "SAVING",
            "status": "completed",
            "at": _now_iso(),
        }
    )
    history.append(
        {
            "state": "QA",
            "status": "completed" if execution_status == "success" else "failed",
            "at": _now_iso(),
        }
    )
    history.append(
        {
            "state": "DONE" if execution_status == "success" else "FAILED",
            "status": execution_status,
            "at": _now_iso(),
        }
    )

    design = plan.get("design_spec", {}) if isinstance(plan, dict) else {}
    return {
        "schema_version": TASK_STATE_SCHEMA_VERSION,
        "generated_at": _now_iso(),
        "task_id": design.get("task_id", ""),
        "status": execution_status,
        "current_state": history[-1]["state"] if history else "UNKNOWN",
        "history": history,
    }


def _first_record(records: list[dict], name: str) -> Dict[str, Any]:
    for rec in records:
        if isinstance(rec, dict) and rec.get("name") == name:
            return rec
    return {}


def _safe_status(value: Any) -> str:
    text = str(value or "").strip()
    return text or "not_run"


def _json_like_equal(left: Any, right: Any) -> bool:
    return json.dumps(left, ensure_ascii=False, sort_keys=True) == json.dumps(right, ensure_ascii=False, sort_keys=True)


def _is_resume_record_compatible(record: Dict[str, Any], planned_call: Dict[str, Any]) -> bool:
    if str(record.get("name") or "") != str(planned_call.get("name") or ""):
        return False
    if str(record.get("status") or "") != "success":
        return False
    planned_args = planned_call.get("args")
    if not isinstance(planned_args, dict):
        planned_args = {}
    record_args = record.get("args")
    if not isinstance(record_args, dict):
        return False
    for key, value in planned_args.items():
        if key not in record_args:
            return False
        if not _json_like_equal(record_args.get(key), value):
            return False
    return True


def _resume_success_prefix(plan_dict: Dict[str, Any], execution_payload: Dict[str, Any]) -> int:
    calls = plan_dict.get("tool_calls") if isinstance(plan_dict.get("tool_calls"), list) else []
    records = execution_payload.get("records") if isinstance(execution_payload.get("records"), list) else []
    matched = 0
    limit = min(len(calls), len(records))
    for idx in range(limit):
        call = calls[idx]
        rec = records[idx]
        if not isinstance(call, dict) or not isinstance(rec, dict):
            break
        if not _is_resume_record_compatible(rec, call):
            break
        matched += 1
    return matched


def _build_structured_reports(
    *,
    plan: Dict[str, Any],
    execution: Dict[str, Any],
    tool_state: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    records = execution.get("records", []) if isinstance(execution, dict) else []
    records = [r for r in records if isinstance(r, dict)]

    search_rec = _first_record(records, "search_dataset")
    gen_rec = _first_record(records, "generate_candidates")
    train_rec = _first_record(records, "train_predictor")
    score_rec = _first_record(records, "score_candidates")
    filter_rec = _first_record(records, "filter_and_rank")
    report_rec = _first_record(records, "make_report")

    design = plan.get("design_spec", {}) if isinstance(plan, dict) else {}
    model_choice = design.get("model_choice", {}) if isinstance(design, dict) else {}

    data_report = {
        "schema_version": "1.0.0",
        "generated_at": _now_iso(),
        "task_id": design.get("task_id", ""),
        "status": execution.get("status", ""),
        "dataset_step": {
            "status": _safe_status(search_rec.get("status", "")),
            "selected": (search_rec.get("result") or {}).get("selected", []),
            "available": (search_rec.get("result") or {}).get("available", []),
        },
        "candidate_step": {
            "status": _safe_status(gen_rec.get("status", "")),
            "adapter": (gen_rec.get("result") or {}).get("adapter", ""),
            "rows": (gen_rec.get("result") or {}).get("rows", 0),
            "source_csv": (gen_rec.get("result") or {}).get("source_csv", ""),
            "input_csv": (gen_rec.get("result") or {}).get("input_csv", ""),
        },
        "artifacts": {
            "candidate_csv": tool_state.get("candidate_csv", ""),
            "scored_csv": tool_state.get("scored_csv", ""),
        },
    }

    score_result = score_rec.get("result") or {}
    model_report = {
        "schema_version": "1.0.0",
        "generated_at": _now_iso(),
        "task_id": design.get("task_id", ""),
        "status": execution.get("status", ""),
        "model_choice": {
            "predictor_id": model_choice.get("predictor_id", ""),
            "generator_id": model_choice.get("generator_id", ""),
        },
        "training_step": {
            "ran": bool(train_rec),
            "status": _safe_status(train_rec.get("status", "")),
            "adapter": (train_rec.get("result") or {}).get("adapter", ""),
            "result": train_rec.get("result") or {},
        },
        "inference_step": {
            "status": _safe_status(score_rec.get("status", "")),
            "adapter": score_result.get("adapter", ""),
            "used_fallback": score_result.get("adapter", "") == "local_deterministic_fallback",
            "fallback_error": score_result.get("fallback_error", {}),
            "result": score_result,
        },
    }

    filter_report = {
        "schema_version": "1.0.0",
        "generated_at": _now_iso(),
        "task_id": design.get("task_id", ""),
        "status": execution.get("status", ""),
        "filter_step": {
            "status": _safe_status(filter_rec.get("status", "")),
            "topn": (filter_rec.get("result") or {}).get("topn", 0),
            "manifest": (filter_rec.get("result") or {}).get("manifest", ""),
            "final_output": (filter_rec.get("result") or {}).get("final_output", ""),
        },
        "report_step": {
            "status": _safe_status(report_rec.get("status", "")),
            "report": (report_rec.get("result") or {}).get("report", ""),
            "latest_run_dir": (report_rec.get("result") or {}).get("latest_run_dir", ""),
        },
    }

    return {
        "data_report": data_report,
        "model_report": model_report,
        "filtering_report": filter_report,
    }


def _mirror_logging_and_result_layout(
    *,
    workspace_root: Path,
    plan: Dict[str, Any],
    execution: Dict[str, Any],
    tool_state: Dict[str, Any],
    run_label: str,
    request_payload: Optional[Dict[str, Any]],
    plan_path: Path,
    execution_path: Path,
    tool_state_path: Path,
    decision_summary_path: Path,
    task_state_path: Path,
    evaluation_report_path: Path,
    guardrails_report_path: Path,
    experiment_trace_path: Optional[Path],
    memory_context_path: Optional[Path],
) -> Dict[str, Any]:
    logging_dir = (workspace_root / DEFAULT_LOGGING_OUT / run_label).resolve()
    result_dir = (workspace_root / DEFAULT_RESULT_OUT / run_label).resolve()
    logging_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    # logging layout
    task_payload = request_payload if request_payload is not None else _build_task_payload_from_plan(plan)
    logging_task_path = logging_dir / "task.json"
    _write_json(logging_task_path, task_payload)

    logging_plan_path = logging_dir / "plan.json"
    _copy_if_exists(plan_path, logging_plan_path)
    logging_execution_path = logging_dir / "execution.json"
    _copy_if_exists(execution_path, logging_execution_path)
    logging_tool_state_path = logging_dir / "tool_state.json"
    _copy_if_exists(tool_state_path, logging_tool_state_path)
    logging_decision_path = logging_dir / "decision_summary.json"
    _copy_if_exists(decision_summary_path, logging_decision_path)
    logging_task_state_path = logging_dir / "task_state.json"
    _copy_if_exists(task_state_path, logging_task_state_path)
    logging_evaluation_report_path = logging_dir / "evaluation_report.json"
    _copy_if_exists(evaluation_report_path, logging_evaluation_report_path)
    logging_guardrails_report_path = logging_dir / "guardrails_report.json"
    _copy_if_exists(guardrails_report_path, logging_guardrails_report_path)
    logging_experiment_trace_path = logging_dir / "experiment_trace.json"
    _copy_if_exists(experiment_trace_path, logging_experiment_trace_path)
    logging_memory_context_path = logging_dir / "memory_context.json"
    _copy_if_exists(memory_context_path, logging_memory_context_path)

    plan_md_path = logging_dir / "plan.md"
    plan_md_path.write_text(_build_plan_markdown(plan), encoding="utf-8")

    exec_log_path = logging_dir / "execution.log"
    exec_log_path.write_text(json.dumps(execution, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    reports = _build_structured_reports(plan=plan, execution=execution, tool_state=tool_state)
    data_report_path = logging_dir / "data_report.json"
    model_report_path = logging_dir / "model_report.json"
    filtering_report_path = logging_dir / "filtering_report.json"
    _write_json(data_report_path, reports["data_report"])
    _write_json(model_report_path, reports["model_report"])
    _write_json(filtering_report_path, reports["filtering_report"])

    # result layout
    candidate_csv = str(tool_state.get("candidate_csv") or "").strip()
    scored_csv = str(tool_state.get("scored_csv") or "").strip()
    final_output = str(tool_state.get("final_output") or "").strip()

    chosen_structures_src: Optional[Path] = None
    for raw in (scored_csv, candidate_csv):
        if not raw:
            continue
        p = Path(raw)
        if not p.is_absolute():
            p = (workspace_root / p).resolve()
        if p.exists():
            chosen_structures_src = p
            break

    target_structures_csv = result_dir / "target_structures.csv"
    copied_target_structures = _copy_if_exists(chosen_structures_src, target_structures_csv)

    report_src: Optional[Path] = None
    if final_output:
        p = Path(final_output)
        if not p.is_absolute():
            p = (workspace_root / p).resolve()
        if p.exists():
            report_src = p
    if report_src is not None:
        _copy_if_exists(report_src, logging_dir / "report.md")
        _copy_if_exists(report_src, result_dir / "report.md")
    result_evaluation_report_path = result_dir / "evaluation_report.json"
    _copy_if_exists(evaluation_report_path, result_evaluation_report_path)
    result_guardrails_report_path = result_dir / "guardrails_report.json"
    _copy_if_exists(guardrails_report_path, result_guardrails_report_path)
    result_experiment_trace_path = result_dir / "experiment_trace.json"
    _copy_if_exists(experiment_trace_path, result_experiment_trace_path)
    result_memory_context_path = result_dir / "memory_context.json"
    _copy_if_exists(memory_context_path, result_memory_context_path)

    result_metadata = {
        "schema_version": "1.0.0",
        "generated_at": _now_iso(),
        "task_id": plan.get("design_spec", {}).get("task_id", ""),
        "status": execution.get("status", ""),
        "source_artifacts": {
            "candidate_csv": candidate_csv,
            "scored_csv": scored_csv,
            "final_output": final_output,
        },
        "outputs": {
            "target_structures_csv": str(target_structures_csv) if copied_target_structures else "",
            "report_md": str(result_dir / "report.md") if (result_dir / "report.md").exists() else "",
            "evaluation_report_json": str(result_evaluation_report_path) if result_evaluation_report_path.exists() else "",
            "guardrails_report_json": str(result_guardrails_report_path) if result_guardrails_report_path.exists() else "",
            "experiment_trace_json": str(result_experiment_trace_path) if result_experiment_trace_path.exists() else "",
            "memory_context_json": str(result_memory_context_path) if result_memory_context_path.exists() else "",
        },
    }
    result_metadata_path = result_dir / "metadata.json"
    _write_json(result_metadata_path, result_metadata)

    return {
        "run_label": run_label,
        "logging_dir": str(logging_dir),
        "result_dir": str(result_dir),
        "logging_task_path": str(logging_task_path),
        "logging_plan_md_path": str(plan_md_path),
        "logging_execution_log_path": str(exec_log_path),
        "logging_data_report_path": str(data_report_path),
        "logging_model_report_path": str(model_report_path),
        "logging_filtering_report_path": str(filtering_report_path),
        "logging_evaluation_report_path": str(logging_evaluation_report_path) if logging_evaluation_report_path.exists() else "",
        "logging_guardrails_report_path": str(logging_guardrails_report_path) if logging_guardrails_report_path.exists() else "",
        "logging_experiment_trace_path": str(logging_experiment_trace_path) if logging_experiment_trace_path.exists() else "",
        "logging_memory_context_path": str(logging_memory_context_path) if logging_memory_context_path.exists() else "",
        "result_metadata_path": str(result_metadata_path),
        "result_target_structures_csv_path": str(target_structures_csv) if copied_target_structures else "",
        "result_evaluation_report_path": str(result_evaluation_report_path) if result_evaluation_report_path.exists() else "",
        "result_guardrails_report_path": str(result_guardrails_report_path) if result_guardrails_report_path.exists() else "",
        "result_experiment_trace_path": str(result_experiment_trace_path) if result_experiment_trace_path.exists() else "",
        "result_memory_context_path": str(result_memory_context_path) if result_memory_context_path.exists() else "",
    }


def _persist_agent_artifacts(
    *,
    workspace_root: Path,
    task_id: str,
    plan_dict: Dict[str, Any],
    result,
    result_dict: Dict[str, Any],
    tool_state: Dict[str, Any],
    request_payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    out_dir = (workspace_root / DEFAULT_AGENT_OUT).resolve() / task_id
    out_dir.mkdir(parents=True, exist_ok=True)

    plan_path = out_dir / "plan.json"
    _write_json(plan_path, plan_dict)

    result_path = out_dir / "execution.json"
    save_execution_result(result, result_path)

    state_path = out_dir / "tool_state.json"
    _write_json(state_path, tool_state)

    decision_summary = _build_decision_summary(
        plan=plan_dict,
        execution=result_dict,
        tool_state=tool_state,
    )
    decision_path = out_dir / "decision_summary.json"
    _write_json(decision_path, decision_summary)

    task_state = _build_task_state(plan=plan_dict, execution=result_dict)
    task_state_path = out_dir / "task_state.json"
    _write_json(task_state_path, task_state)

    artifact_dir = out_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    evaluation_report = build_evaluation_report(
        task_id=task_id,
        execution_mode="full_pipeline",
        execution_payload=result_dict,
        decision_summary=decision_summary,
        task_state=task_state,
        tool_state=tool_state,
        workspace_root=workspace_root.resolve(),
    )
    evaluation_report_path = artifact_dir / "evaluation_report.json"
    _write_json(evaluation_report_path, evaluation_report)
    web_evidence_path = artifact_dir / "web_evidence.json"
    constraints = (
        plan_dict.get("design_spec", {}).get("constraints", {})
        if isinstance(plan_dict.get("design_spec"), dict)
        else {}
    )
    guardrails_report = build_guardrails_report(
        task_id=task_id,
        execution_mode="full_pipeline",
        execution_payload=result_dict,
        tool_state=tool_state,
        workspace_root=workspace_root.resolve(),
        constraints=constraints if isinstance(constraints, dict) else {},
        web_evidence_path=web_evidence_path,
    )
    guardrails_report_path = artifact_dir / "guardrails_report.json"
    _write_json(guardrails_report_path, guardrails_report)

    request_path: Optional[Path] = None
    if request_payload is not None:
        request_path = out_dir / "request.json"
        _write_json(request_path, request_payload)

    run_label = _build_run_label(task_id)
    previous_memory_context = _load_json_if_exists(artifact_dir / "memory_context.json")
    memory_context_payload = build_memory_context(
        task_id=task_id,
        execution_mode="full_pipeline",
        run_label=run_label,
        workspace_root=workspace_root.resolve(),
        execution_payload=result_dict,
        tool_state=tool_state,
        request_payload=request_payload if isinstance(request_payload, dict) else None,
        plan_payload=plan_dict,
        task_payload=request_payload if isinstance(request_payload, dict) else _build_task_payload_from_plan(plan_dict),
        web_evidence_path=web_evidence_path if web_evidence_path.exists() else None,
        previous_memory_context=previous_memory_context,
    )
    memory_context_path = artifact_dir / "memory_context.json"
    _write_json(memory_context_path, memory_context_payload)
    memory_index_path = update_memory_index(
        workspace_root=workspace_root.resolve(),
        memory_context=memory_context_payload,
        memory_context_path=memory_context_path,
    )

    experiment_trace_path = artifact_dir / "experiment_trace.json"
    artifact_paths: Dict[str, Path] = {
        "plan": plan_path,
        "execution": result_path,
        "tool_state": state_path,
        "decision_summary": decision_path,
        "task_state": task_state_path,
        "evaluation_report": evaluation_report_path,
        "guardrails_report": guardrails_report_path,
        "memory_context": memory_context_path,
        "memory_index": memory_index_path,
    }
    if request_path is not None:
        artifact_paths["request"] = request_path
    if web_evidence_path.exists():
        artifact_paths["web_evidence"] = web_evidence_path
    _write_json(
        experiment_trace_path,
        build_experiment_trace(
            task_id=task_id,
            run_label=run_label,
            workspace_root=workspace_root,
            execution_mode="full_pipeline",
            request_payload=request_payload if isinstance(request_payload, dict) else None,
            plan_payload=plan_dict,
            execution_payload=result_dict,
            tool_state=tool_state,
            artifact_paths=artifact_paths,
        ),
    )
    mirror = _mirror_logging_and_result_layout(
        workspace_root=workspace_root,
        plan=plan_dict,
        execution=result_dict,
        tool_state=tool_state,
        run_label=run_label,
        request_payload=request_payload,
        plan_path=plan_path,
        execution_path=result_path,
        tool_state_path=state_path,
        decision_summary_path=decision_path,
        task_state_path=task_state_path,
        evaluation_report_path=evaluation_report_path,
        guardrails_report_path=guardrails_report_path,
        experiment_trace_path=experiment_trace_path,
        memory_context_path=memory_context_path,
    )

    out = {
        "task_id": task_id,
        "status": result.status,
        "plan_path": str(plan_path),
        "execution_path": str(result_path),
        "tool_state_path": str(state_path),
        "decision_summary_path": str(decision_path),
        "task_state_path": str(task_state_path),
        "evaluation_report_path": str(evaluation_report_path),
        "guardrails_report_path": str(guardrails_report_path),
        "memory_context_path": str(memory_context_path),
        "memory_index_path": str(memory_index_path),
        "experiment_trace_path": str(experiment_trace_path),
        **mirror,
    }
    if request_path is not None:
        out["request_path"] = str(request_path)
    return out


def _build_decision_summary(
    *,
    plan: Dict[str, Any],
    execution: Dict[str, Any],
    tool_state: Dict[str, Any],
) -> Dict[str, Any]:
    records = execution.get("records", []) if isinstance(execution, dict) else []
    score_record: Dict[str, Any] = {}
    for rec in records:
        if isinstance(rec, dict) and rec.get("name") == "score_candidates":
            score_record = rec
            break

    score_result = score_record.get("result", {}) if isinstance(score_record, dict) else {}
    adapter = score_result.get("adapter", "")
    fallback_error = score_result.get("fallback_error", {}) if isinstance(score_result, dict) else {}
    used_fallback = adapter == "local_deterministic_fallback"

    return {
        "schema_version": DECISION_SUMMARY_SCHEMA_VERSION,
        "generated_at": _now_iso(),
        "task_id": plan.get("design_spec", {}).get("task_id", ""),
        "status": execution.get("status", ""),
        "model_choice": plan.get("design_spec", {}).get("model_choice", {}),
        "score_step": {
            "adapter": adapter,
            "used_fallback": used_fallback,
            "fallback_reason": score_result.get("fallback_reason", ""),
            "fallback_code": fallback_error.get("code", ""),
            "fallback_retryable": bool(fallback_error.get("retryable", False)),
            "fallback_details": fallback_error.get("details", {}),
        },
        "artifacts": {
            "candidate_csv": tool_state.get("candidate_csv", ""),
            "scored_csv": tool_state.get("scored_csv", ""),
            "final_output": tool_state.get("final_output", ""),
        },
    }


def plan_request(
    *,
    workspace_root: Path,
    user_request: str,
    task_id: str,
    predictor_id: str = "",
    generator_id: str = "",
    mode: str = "fast_screen",
    planner_provider: str = DEFAULT_PLANNER_PROVIDER,
    catalog_path: Optional[Path] = None,
) -> Dict[str, Any]:
    catalog = (catalog_path or (workspace_root / DEFAULT_CATALOG)).resolve()
    plan = build_plan(
        user_request=user_request,
        task_id=task_id,
        catalog_path=catalog,
        predictor_id=predictor_id,
        generator_id=generator_id,
        mode=mode,
        planner_provider=planner_provider,
    )

    return plan.to_dict()


def plan_request_from_payload(
    *,
    workspace_root: Path,
    request_payload: Dict[str, Any],
    planner_provider: str = DEFAULT_PLANNER_PROVIDER,
    catalog_path: Optional[Path] = None,
) -> Dict[str, Any]:
    validate_request_payload(payload=request_payload, workspace_root=workspace_root)
    catalog = (catalog_path or (workspace_root / DEFAULT_CATALOG)).resolve()
    plan = build_plan_from_request_payload(
        request_payload=request_payload,
        catalog_path=catalog,
        planner_provider=planner_provider,
    )
    return plan.to_dict()


def execute_request(
    *,
    workspace_root: Path,
    user_request: str,
    task_id: str,
    predictor_id: str = "",
    generator_id: str = "",
    mode: str = "fast_screen",
    planner_provider: str = DEFAULT_PLANNER_PROVIDER,
    catalog_path: Optional[Path] = None,
) -> Dict[str, Any]:
    catalog = (catalog_path or (workspace_root / DEFAULT_CATALOG)).resolve()

    plan = build_plan(
        user_request=user_request,
        task_id=task_id,
        catalog_path=catalog,
        predictor_id=predictor_id,
        generator_id=generator_id,
        mode=mode,
        planner_provider=planner_provider,
    )

    plan_dict = plan.to_dict()

    tool_ctx = ToolContext(
        workspace_root=workspace_root.resolve(),
        catalog_path=catalog,
        task_id=task_id,
        state={},
    )
    result = execute_plan(plan, tool_ctx)
    result_dict = result.to_dict()
    tool_state = dict(tool_ctx.state)
    return _persist_agent_artifacts(
        workspace_root=workspace_root,
        task_id=task_id,
        plan_dict=plan_dict,
        result=result,
        result_dict=result_dict,
        tool_state=tool_state,
        request_payload=None,
    )


def execute_request_from_payload(
    *,
    workspace_root: Path,
    request_payload: Dict[str, Any],
    planner_provider: str = DEFAULT_PLANNER_PROVIDER,
    catalog_path: Optional[Path] = None,
    resume_existing: bool = False,
) -> Dict[str, Any]:
    validate_request_payload(payload=request_payload, workspace_root=workspace_root)
    catalog = (catalog_path or (workspace_root / DEFAULT_CATALOG)).resolve()
    plan = build_plan_from_request_payload(
        request_payload=request_payload,
        catalog_path=catalog,
        planner_provider=planner_provider,
    )
    plan_dict = plan.to_dict()
    task_id = str(plan.design_spec.task_id)

    resume_from_index = 0
    resume_records = None
    existing_state: Dict[str, Any] = {}
    if bool(resume_existing):
        out_dir = (workspace_root / DEFAULT_AGENT_OUT / task_id).resolve()
        existing_exec = _load_json_if_exists(out_dir / "execution.json")
        existing_state_payload = _load_json_if_exists(out_dir / "tool_state.json")
        if isinstance(existing_state_payload, dict):
            existing_state = dict(existing_state_payload)
        if isinstance(existing_exec, dict):
            resume_from_index = _resume_success_prefix(plan_dict=plan_dict, execution_payload=existing_exec)
            existing_records = existing_exec.get("records")
            if isinstance(existing_records, list) and resume_from_index > 0:
                resume_records = [r for r in existing_records[:resume_from_index] if isinstance(r, dict)]

    tool_ctx = ToolContext(
        workspace_root=workspace_root.resolve(),
        catalog_path=catalog,
        task_id=task_id,
        state=existing_state if bool(resume_existing) else {},
    )
    if bool(resume_existing):
        result = execute_plan_with_resume(
            plan,
            tool_ctx,
            resume_records=resume_records,
            resume_from_index=resume_from_index,
        )
    else:
        result = execute_plan(plan, tool_ctx)
    result_dict = result.to_dict()
    tool_state = dict(tool_ctx.state)
    out = _persist_agent_artifacts(
        workspace_root=workspace_root,
        task_id=task_id,
        plan_dict=plan_dict,
        result=result,
        result_dict=result_dict,
        tool_state=tool_state,
        request_payload=request_payload,
    )
    if bool(resume_existing):
        out["resumed"] = True
        out["resume_skipped_steps"] = int(resume_from_index)
        out["resume_total_steps"] = len(plan_dict.get("tool_calls", []) if isinstance(plan_dict, dict) else [])
    return out

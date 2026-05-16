from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from oled_agent.agent.tool_contracts import ToolContractValidationError, validate_tool_call_args


class RequestValidationError(ValueError):
    """Raised when request JSON does not satisfy the RequestSpec contract."""


@dataclass(frozen=True)
class RequestContractPaths:
    request_schema: Path
    task_v2_schema: Path
    step_request_schema: Path
    decision_summary_schema: Path
    plan_schema: Path
    task_state_schema: Path
    data_report_schema: Path
    model_report_schema: Path
    filtering_report_schema: Path
    evaluation_report_schema: Path
    guardrails_report_schema: Path
    memory_context_schema: Path


def default_contract_paths(workspace_root: Path) -> RequestContractPaths:
    base = workspace_root / "schemas"
    if not base.exists():
        # Fallback for callers that use temp workspace roots in tests or orchestration.
        base = Path(__file__).resolve().parents[3] / "schemas"
    return RequestContractPaths(
        request_schema=base / "request.schema.json",
        task_v2_schema=base / "task.schema.v2.json",
        step_request_schema=base / "step_request.schema.json",
        decision_summary_schema=base / "decision_summary.schema.json",
        plan_schema=base / "plan.schema.json",
        task_state_schema=base / "task_state.schema.json",
        data_report_schema=base / "data_report.schema.json",
        model_report_schema=base / "model_report.schema.json",
        filtering_report_schema=base / "filtering_report.schema.json",
        evaluation_report_schema=base / "evaluation_report.schema.json",
        guardrails_report_schema=base / "guardrails_report.schema.json",
        memory_context_schema=base / "memory_context.schema.json",
    )


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RequestValidationError(f"Schema or payload file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise RequestValidationError(f"Invalid JSON at {path}: {exc}") from exc


def _validate_via_jsonschema(
    instance: Dict[str, Any],
    schema: Dict[str, Any],
    *,
    contract_kind: str,
) -> None:
    try:
        import jsonschema  # type: ignore
    except Exception:
        if contract_kind == "request":
            _validate_request_minimal(instance, schema)
        elif contract_kind == "task_v2":
            _validate_task_v2_minimal(instance, schema)
        elif contract_kind == "step_request":
            _validate_step_request_minimal(instance, schema)
        elif contract_kind == "decision_summary":
            _validate_decision_summary_minimal(instance, schema)
        elif contract_kind == "plan":
            _validate_plan_minimal(instance, schema)
        elif contract_kind == "task_state":
            _validate_task_state_minimal(instance, schema)
        elif contract_kind == "data_report":
            _validate_data_report_minimal(instance, schema)
        elif contract_kind == "model_report":
            _validate_model_report_minimal(instance, schema)
        elif contract_kind == "filtering_report":
            _validate_filtering_report_minimal(instance, schema)
        elif contract_kind == "evaluation_report":
            _validate_evaluation_report_minimal(instance, schema)
        elif contract_kind == "guardrails_report":
            _validate_guardrails_report_minimal(instance, schema)
        elif contract_kind == "memory_context":
            _validate_memory_context_minimal(instance, schema)
        else:
            raise RequestValidationError(f"Unsupported contract kind: {contract_kind}")
        return

    try:
        jsonschema.validate(instance=instance, schema=schema)
    except Exception as exc:
        raise RequestValidationError(f"Request JSON failed schema validation: {exc}") from exc


def _validate_required_and_additional(
    instance: Dict[str, Any],
    schema: Dict[str, Any],
    *,
    path: str = "$",
) -> Dict[str, Any]:
    required = schema.get("required", [])
    for key in required:
        if key not in instance:
            raise RequestValidationError(f"{path}.{key}: missing required field")

    properties = schema.get("properties", {}) if isinstance(schema, dict) else {}
    if schema.get("additionalProperties") is False and isinstance(properties, dict):
        extras = sorted(k for k in instance.keys() if k not in properties)
        if extras:
            raise RequestValidationError(f"{path}: unexpected field(s): {extras}")
    return properties


def _validate_request_minimal(instance: Dict[str, Any], schema: Dict[str, Any]) -> None:
    # Lightweight fallback when jsonschema is not installed.
    properties = _validate_required_and_additional(instance, schema, path="$")

    for key in ("task_id", "request_text", "mode"):
        if key in instance and not isinstance(instance[key], str):
            raise RequestValidationError(f"$.{key}: must be string")

    if "mode" in instance:
        mode_enum = (
            properties.get("mode", {}).get("enum", [])
            if isinstance(properties.get("mode", {}), dict)
            else []
        )
        if mode_enum and instance["mode"] not in mode_enum:
            raise RequestValidationError(f"$.mode: must be one of: {mode_enum}")

    if "targets" in instance and not isinstance(instance["targets"], list):
        raise RequestValidationError("$.targets: must be array")
    if isinstance(instance.get("targets"), list):
        target_items = properties.get("targets", {}).get("items", {}) if isinstance(properties.get("targets", {}), dict) else {}
        target_allowed = set(target_items.get("properties", {}).keys()) if isinstance(target_items, dict) else set()
        target_required = list(target_items.get("required", [])) if isinstance(target_items, dict) else []
        property_enum = target_items.get("properties", {}).get("property", {}).get("enum", []) if isinstance(target_items, dict) else []
        for idx, target in enumerate(instance["targets"], start=1):
            if not isinstance(target, dict):
                raise RequestValidationError(f"$.targets[{idx}] must be object")
            if target_items.get("additionalProperties") is False:
                extras = sorted(k for k in target.keys() if k not in target_allowed)
                if extras:
                    raise RequestValidationError(f"$.targets[{idx}]: unexpected field(s): {extras}")
            for key in target_required:
                if key not in target:
                    raise RequestValidationError(f"$.targets[{idx}].{key}: missing required field")
            if property_enum and target.get("property") not in property_enum:
                raise RequestValidationError(f"$.targets[{idx}].property: must be one of: {property_enum}")

    if "budget" in instance and not isinstance(instance["budget"], dict):
        raise RequestValidationError("$.budget: must be object")

    if "generation_input" in instance:
        generation_input = instance["generation_input"]
        if not isinstance(generation_input, dict):
            raise RequestValidationError("$.generation_input: must be object")
        gen_schema = properties.get("generation_input", {}) if isinstance(properties.get("generation_input", {}), dict) else {}
        gen_props = gen_schema.get("properties", {}) if isinstance(gen_schema.get("properties", {}), dict) else {}
        if gen_schema.get("additionalProperties") is False:
            extras = sorted(k for k in generation_input.keys() if k not in gen_props)
            if extras:
                raise RequestValidationError(f"$.generation_input: unexpected field(s): {extras}")
        for key, value in generation_input.items():
            field_schema = gen_props.get(key, {})
            field_type = field_schema.get("type") if isinstance(field_schema, dict) else None
            if field_type == "string":
                if not isinstance(value, str):
                    raise RequestValidationError(f"$.generation_input.{key}: must be string")
            elif field_type == "array":
                if not isinstance(value, list):
                    raise RequestValidationError(f"$.generation_input.{key}: must be array")
                for idx, item in enumerate(value, start=1):
                    if not isinstance(item, str):
                        raise RequestValidationError(f"$.generation_input.{key}[{idx}]: must be string")


def _validate_task_v2_minimal(instance: Dict[str, Any], schema: Dict[str, Any]) -> None:
    properties = _validate_required_and_additional(instance, schema, path="$")

    def _must_non_empty_str(key: str) -> None:
        if not isinstance(instance.get(key), str) or not str(instance.get(key)).strip():
            raise RequestValidationError(f"$.{key}: must be non-empty string")

    for key in ("version", "task_id", "request_text", "execution_mode", "operation", "property", "range", "prediction_model"):
        _must_non_empty_str(key)

    if instance.get("version") != "2.0":
        raise RequestValidationError("$.version: must be '2.0'")

    em_enum = properties.get("execution_mode", {}).get("enum", []) if isinstance(properties.get("execution_mode"), dict) else []
    if em_enum and instance.get("execution_mode") not in em_enum:
        raise RequestValidationError(f"$.execution_mode: must be one of: {em_enum}")

    op_enum = properties.get("operation", {}).get("enum", []) if isinstance(properties.get("operation"), dict) else []
    if op_enum and instance.get("operation") not in op_enum:
        raise RequestValidationError(f"$.operation: must be one of: {op_enum}")

    prop_enum = properties.get("property", {}).get("enum", []) if isinstance(properties.get("property"), dict) else []
    if prop_enum and instance.get("property") not in prop_enum:
        raise RequestValidationError(f"$.property: must be one of: {prop_enum}")

    n_structures = instance.get("n_structures")
    if not isinstance(n_structures, int) or n_structures < 1:
        raise RequestValidationError("$.n_structures: must be integer >= 1")

    constraints = instance.get("constraints")
    if not isinstance(constraints, dict):
        raise RequestValidationError("$.constraints: must be object")

    for key in ("train_data", "candidate_data"):
        val = instance.get(key)
        if val is not None and not isinstance(val, str):
            raise RequestValidationError(f"$.{key}: must be string|null")

    for key in ("model_preferences", "generation_input", "provenance"):
        if key in instance and not isinstance(instance.get(key), dict):
            raise RequestValidationError(f"$.{key}: must be object")

    for key in ("missing_fields", "questions", "compatibility_warnings"):
        if key in instance and not isinstance(instance.get(key), list):
            raise RequestValidationError(f"$.{key}: must be array")


def _validate_step_request_minimal(instance: Dict[str, Any], schema: Dict[str, Any]) -> None:
    properties = _validate_required_and_additional(instance, schema, path="$")
    task = instance.get("task")
    if not isinstance(task, dict):
        raise RequestValidationError("$.task: must be object")

    operation = instance.get("operation")
    if not isinstance(operation, str) or not operation.strip():
        raise RequestValidationError("$.operation: must be non-empty string")
    op_enum = properties.get("operation", {}).get("enum", []) if isinstance(properties.get("operation"), dict) else []
    if op_enum and operation not in op_enum:
        raise RequestValidationError(f"$.operation: must be one of: {op_enum}")

    args = instance.get("args")
    if args is not None and not isinstance(args, dict):
        raise RequestValidationError("$.args: must be object")


def _validate_decision_summary_minimal(instance: Dict[str, Any], schema: Dict[str, Any]) -> None:
    _validate_required_and_additional(instance, schema, path="$")
    score_step = instance.get("score_step")
    if score_step is not None and not isinstance(score_step, dict):
        raise RequestValidationError("$.score_step: must be object")
    if isinstance(score_step, dict):
        if "fallback_code" in score_step and score_step["fallback_code"] is None:
            raise RequestValidationError("$.score_step.fallback_code: must not be null")
        if "fallback_retryable" in score_step and score_step["fallback_retryable"] is None:
            raise RequestValidationError("$.score_step.fallback_retryable: must not be null")


def _validate_plan_minimal(instance: Dict[str, Any], schema: Dict[str, Any]) -> None:
    _validate_required_and_additional(instance, schema, path="$")
    # Minimal checks for AgentPlan-like payloads.
    if not isinstance(instance.get("summary"), str) or not str(instance.get("summary")).strip():
        raise RequestValidationError("$.summary: must be non-empty string")
    design = instance.get("design_spec")
    if not isinstance(design, dict):
        raise RequestValidationError("$.design_spec: must be object")
    for key in ("task_id", "request_text", "mode", "targets", "budget", "model_choice"):
        if key not in design:
            raise RequestValidationError(f"$.design_spec.{key}: missing required field")
    if not isinstance(design.get("task_id"), str) or not str(design.get("task_id")).strip():
        raise RequestValidationError("$.design_spec.task_id: must be non-empty string")
    if not isinstance(design.get("request_text"), str) or not str(design.get("request_text")).strip():
        raise RequestValidationError("$.design_spec.request_text: must be non-empty string")
    mode = design.get("mode")
    if mode not in ("fast_screen", "train_then_design"):
        raise RequestValidationError("$.design_spec.mode: must be one of: ['fast_screen', 'train_then_design']")
    targets = design.get("targets")
    if not isinstance(targets, list) or not targets:
        raise RequestValidationError("$.design_spec.targets: must be non-empty array")
    for idx, target in enumerate(targets, start=1):
        if not isinstance(target, dict):
            raise RequestValidationError(f"$.design_spec.targets[{idx}] must be object")
        if not isinstance(target.get("name"), str) or not target.get("name", "").strip():
            raise RequestValidationError(f"$.design_spec.targets[{idx}].name: must be non-empty string")
        if target.get("objective") not in ("maximize", "minimize", "target_window"):
            raise RequestValidationError(
                f"$.design_spec.targets[{idx}].objective: must be one of: ['maximize', 'minimize', 'target_window']"
            )
    budget = design.get("budget")
    if not isinstance(budget, dict):
        raise RequestValidationError("$.design_spec.budget: must be object")
    if not isinstance(budget.get("max_candidates"), int) or budget.get("max_candidates", 0) < 1:
        raise RequestValidationError("$.design_spec.budget.max_candidates: must be integer >= 1")
    model_choice = design.get("model_choice")
    if not isinstance(model_choice, dict):
        raise RequestValidationError("$.design_spec.model_choice: must be object")
    for key in ("predictor_id", "generator_id"):
        if not isinstance(model_choice.get(key), str) or not model_choice.get(key, "").strip():
            raise RequestValidationError(f"$.design_spec.model_choice.{key}: must be non-empty string")
    tool_calls = instance.get("tool_calls")
    if not isinstance(tool_calls, list) or not tool_calls:
        raise RequestValidationError("$.tool_calls: must be non-empty array")
    for idx, call in enumerate(tool_calls, start=1):
        if not isinstance(call, dict):
            raise RequestValidationError(f"$.tool_calls[{idx}] must be object")
        if not isinstance(call.get("name"), str) or not call.get("name", "").strip():
            raise RequestValidationError(f"$.tool_calls[{idx}].name: must be non-empty string")
        if not isinstance(call.get("args"), dict):
            raise RequestValidationError(f"$.tool_calls[{idx}].args: must be object")
        try:
            validate_tool_call_args(
                name=str(call.get("name")).strip(),
                args=call.get("args"),
                path=f"$.tool_calls[{idx}]",
            )
        except ToolContractValidationError as exc:
            raise RequestValidationError(str(exc)) from exc


def _validate_task_state_minimal(instance: Dict[str, Any], schema: Dict[str, Any]) -> None:
    _validate_required_and_additional(instance, schema, path="$")
    for key in ("schema_version", "generated_at", "task_id", "status", "current_state"):
        if not isinstance(instance.get(key), str) or not str(instance.get(key)).strip():
            raise RequestValidationError(f"$.{key}: must be non-empty string")
    status_enum = {"success", "failed"}
    if instance.get("status") not in status_enum:
        raise RequestValidationError("$.status: must be one of: ['success', 'failed']")
    state_enum = {
        "INIT",
        "REQUIREMENT_COLLECTION",
        "VALIDATION",
        "PLAN_GENERATION",
        "USER_CONFIRMATION",
        "WAITING_APPROVAL",
        "NEED_INFO",
        "DATA_ACQUISITION",
        "PREPROCESSING",
        "ROUTING",
        "TRAINING_OPTIONAL",
        "INFERENCE",
        "FILTERING",
        "SAVING",
        "REPORTING",
        "PAUSED",
        "QA",
        "DONE",
        "FAILED",
    }
    if instance.get("current_state") not in state_enum:
        raise RequestValidationError("$.current_state: unsupported state")
    history = instance.get("history")
    if not isinstance(history, list) or not history:
        raise RequestValidationError("$.history: must be non-empty array")
    history_status_enum = {"completed", "success", "failed", "unknown"}
    for idx, item in enumerate(history, start=1):
        if not isinstance(item, dict):
            raise RequestValidationError(f"$.history[{idx}]: must be object")
        if not isinstance(item.get("state"), str) or not str(item.get("state")).strip():
            raise RequestValidationError(f"$.history[{idx}].state: must be non-empty string")
        if not isinstance(item.get("status"), str) or not str(item.get("status")).strip():
            raise RequestValidationError(f"$.history[{idx}].status: must be non-empty string")
        if item.get("state") not in state_enum:
            raise RequestValidationError(f"$.history[{idx}].state: unsupported state")
        if item.get("status") not in history_status_enum:
            raise RequestValidationError(
                f"$.history[{idx}].status: must be one of: ['completed', 'success', 'failed', 'unknown']"
            )


def _validate_data_report_minimal(instance: Dict[str, Any], schema: Dict[str, Any]) -> None:
    _validate_required_and_additional(instance, schema, path="$")
    for key in ("schema_version", "generated_at", "task_id", "status"):
        if not isinstance(instance.get(key), str) or not str(instance.get(key)).strip():
            raise RequestValidationError(f"$.{key}: must be non-empty string")
    if instance.get("status") not in ("success", "failed"):
        raise RequestValidationError("$.status: must be one of: ['success', 'failed']")

    dataset = instance.get("dataset_step")
    if not isinstance(dataset, dict):
        raise RequestValidationError("$.dataset_step: must be object")
    if not isinstance(dataset.get("status"), str) or not str(dataset.get("status")).strip():
        raise RequestValidationError("$.dataset_step.status: must be non-empty string")
    for key in ("selected", "available"):
        value = dataset.get(key)
        if not isinstance(value, list):
            raise RequestValidationError(f"$.dataset_step.{key}: must be array")

    candidate = instance.get("candidate_step")
    if not isinstance(candidate, dict):
        raise RequestValidationError("$.candidate_step: must be object")
    if not isinstance(candidate.get("status"), str) or not str(candidate.get("status")).strip():
        raise RequestValidationError("$.candidate_step.status: must be non-empty string")
    if not isinstance(candidate.get("rows"), int) or candidate.get("rows", 0) < 0:
        raise RequestValidationError("$.candidate_step.rows: must be integer >= 0")

    artifacts = instance.get("artifacts")
    if not isinstance(artifacts, dict):
        raise RequestValidationError("$.artifacts: must be object")
    for key in ("candidate_csv", "scored_csv"):
        if not isinstance(artifacts.get(key), str):
            raise RequestValidationError(f"$.artifacts.{key}: must be string")


def _validate_model_report_minimal(instance: Dict[str, Any], schema: Dict[str, Any]) -> None:
    _validate_required_and_additional(instance, schema, path="$")
    for key in ("schema_version", "generated_at", "task_id", "status"):
        if not isinstance(instance.get(key), str) or not str(instance.get(key)).strip():
            raise RequestValidationError(f"$.{key}: must be non-empty string")
    if instance.get("status") not in ("success", "failed"):
        raise RequestValidationError("$.status: must be one of: ['success', 'failed']")

    model_choice = instance.get("model_choice")
    if not isinstance(model_choice, dict):
        raise RequestValidationError("$.model_choice: must be object")
    for key in ("predictor_id", "generator_id"):
        if not isinstance(model_choice.get(key), str) or not str(model_choice.get(key)).strip():
            raise RequestValidationError(f"$.model_choice.{key}: must be non-empty string")

    training = instance.get("training_step")
    if not isinstance(training, dict):
        raise RequestValidationError("$.training_step: must be object")
    if not isinstance(training.get("ran"), bool):
        raise RequestValidationError("$.training_step.ran: must be boolean")
    if not isinstance(training.get("status"), str):
        raise RequestValidationError("$.training_step.status: must be string")

    inference = instance.get("inference_step")
    if not isinstance(inference, dict):
        raise RequestValidationError("$.inference_step: must be object")
    if not isinstance(inference.get("used_fallback"), bool):
        raise RequestValidationError("$.inference_step.used_fallback: must be boolean")
    if not isinstance(inference.get("status"), str) or not str(inference.get("status")).strip():
        raise RequestValidationError("$.inference_step.status: must be non-empty string")


def _validate_filtering_report_minimal(instance: Dict[str, Any], schema: Dict[str, Any]) -> None:
    _validate_required_and_additional(instance, schema, path="$")
    for key in ("schema_version", "generated_at", "task_id", "status"):
        if not isinstance(instance.get(key), str) or not str(instance.get(key)).strip():
            raise RequestValidationError(f"$.{key}: must be non-empty string")
    if instance.get("status") not in ("success", "failed"):
        raise RequestValidationError("$.status: must be one of: ['success', 'failed']")

    filter_step = instance.get("filter_step")
    if not isinstance(filter_step, dict):
        raise RequestValidationError("$.filter_step: must be object")
    if not isinstance(filter_step.get("status"), str) or not str(filter_step.get("status")).strip():
        raise RequestValidationError("$.filter_step.status: must be non-empty string")
    if not isinstance(filter_step.get("topn"), int) or filter_step.get("topn", 0) < 0:
        raise RequestValidationError("$.filter_step.topn: must be integer >= 0")

    report_step = instance.get("report_step")
    if not isinstance(report_step, dict):
        raise RequestValidationError("$.report_step: must be object")
    if not isinstance(report_step.get("status"), str) or not str(report_step.get("status")).strip():
        raise RequestValidationError("$.report_step.status: must be non-empty string")


def _validate_evaluation_report_minimal(instance: Dict[str, Any], schema: Dict[str, Any]) -> None:
    _validate_required_and_additional(instance, schema, path="$")
    for key in ("schema_version", "generated_at", "task_id", "execution_mode", "execution_status", "status"):
        if not isinstance(instance.get(key), str) or not str(instance.get(key)).strip():
            raise RequestValidationError(f"$.{key}: must be non-empty string")
    if instance.get("execution_status") not in ("success", "failed"):
        raise RequestValidationError("$.execution_status: must be one of: ['success', 'failed']")
    if instance.get("status") not in ("pass", "warn", "fail"):
        raise RequestValidationError("$.status: must be one of: ['pass', 'warn', 'fail']")

    summary = instance.get("summary")
    if not isinstance(summary, dict):
        raise RequestValidationError("$.summary: must be object")
    for key in ("checks_total", "pass_count", "warn_count", "fail_count"):
        value = summary.get(key)
        if not isinstance(value, int) or value < 0:
            raise RequestValidationError(f"$.summary.{key}: must be integer >= 0")

    metrics = instance.get("metrics")
    if not isinstance(metrics, dict):
        raise RequestValidationError("$.metrics: must be object")
    for key in ("record_count", "success_count", "failed_count", "fallback_count"):
        value = metrics.get(key)
        if not isinstance(value, int) or value < 0:
            raise RequestValidationError(f"$.metrics.{key}: must be integer >= 0")
    adapters = metrics.get("adapters")
    if not isinstance(adapters, list):
        raise RequestValidationError("$.metrics.adapters: must be array")
    for idx, value in enumerate(adapters, start=1):
        if not isinstance(value, str):
            raise RequestValidationError(f"$.metrics.adapters[{idx}]: must be string")
    duration = metrics.get("duration_seconds")
    if duration is not None and not isinstance(duration, (int, float)):
        raise RequestValidationError("$.metrics.duration_seconds: must be number|null")
    if isinstance(duration, (int, float)) and duration < 0:
        raise RequestValidationError("$.metrics.duration_seconds: must be >= 0")

    checks = instance.get("checks")
    if not isinstance(checks, list) or len(checks) == 0:
        raise RequestValidationError("$.checks: must be non-empty array")
    for idx, item in enumerate(checks, start=1):
        if not isinstance(item, dict):
            raise RequestValidationError(f"$.checks[{idx}]: must be object")
        if not isinstance(item.get("name"), str) or not str(item.get("name")).strip():
            raise RequestValidationError(f"$.checks[{idx}].name: must be non-empty string")
        if not isinstance(item.get("message"), str) or not str(item.get("message")).strip():
            raise RequestValidationError(f"$.checks[{idx}].message: must be non-empty string")
        if item.get("status") not in ("pass", "warn", "fail"):
            raise RequestValidationError(f"$.checks[{idx}].status: must be one of: ['pass', 'warn', 'fail']")


def _validate_guardrails_report_minimal(instance: Dict[str, Any], schema: Dict[str, Any]) -> None:
    _validate_required_and_additional(instance, schema, path="$")
    for key in ("schema_version", "generated_at", "task_id", "execution_mode", "execution_status", "status", "strict_status"):
        if not isinstance(instance.get(key), str) or not str(instance.get(key)).strip():
            raise RequestValidationError(f"$.{key}: must be non-empty string")
    if instance.get("execution_status") not in ("success", "failed"):
        raise RequestValidationError("$.execution_status: must be one of: ['success', 'failed']")
    if instance.get("status") not in ("pass", "warn", "fail"):
        raise RequestValidationError("$.status: must be one of: ['pass', 'warn', 'fail']")
    if instance.get("strict_status") not in ("pass", "fail"):
        raise RequestValidationError("$.strict_status: must be one of: ['pass', 'fail']")

    summary = instance.get("summary")
    if not isinstance(summary, dict):
        raise RequestValidationError("$.summary: must be object")
    for key in ("checks_total", "pass_count", "warn_count", "fail_count", "strict_blocking_count"):
        value = summary.get(key)
        if not isinstance(value, int) or value < 0:
            raise RequestValidationError(f"$.summary.{key}: must be integer >= 0")

    for key in ("blocking_checks", "strict_blocking_checks"):
        rows = instance.get(key)
        if not isinstance(rows, list):
            raise RequestValidationError(f"$.{key}: must be array")
        for idx, value in enumerate(rows, start=1):
            if not isinstance(value, str):
                raise RequestValidationError(f"$.{key}[{idx}]: must be string")

    checks = instance.get("checks")
    if not isinstance(checks, list) or len(checks) == 0:
        raise RequestValidationError("$.checks: must be non-empty array")
    for idx, item in enumerate(checks, start=1):
        if not isinstance(item, dict):
            raise RequestValidationError(f"$.checks[{idx}]: must be object")
        if not isinstance(item.get("name"), str) or not str(item.get("name")).strip():
            raise RequestValidationError(f"$.checks[{idx}].name: must be non-empty string")
        if item.get("status") not in ("pass", "warn", "fail"):
            raise RequestValidationError(f"$.checks[{idx}].status: must be one of: ['pass', 'warn', 'fail']")
        if not isinstance(item.get("strict_blocking"), bool):
            raise RequestValidationError(f"$.checks[{idx}].strict_blocking: must be boolean")
        if not isinstance(item.get("message"), str) or not str(item.get("message")).strip():
            raise RequestValidationError(f"$.checks[{idx}].message: must be non-empty string")


def _validate_memory_context_minimal(instance: Dict[str, Any], schema: Dict[str, Any]) -> None:
    _validate_required_and_additional(instance, schema, path="$")
    for key in ("schema_version", "generated_at", "task_id", "run_label", "execution_mode", "execution_status"):
        if not isinstance(instance.get(key), str) or not str(instance.get(key)).strip():
            raise RequestValidationError(f"$.{key}: must be non-empty string")
    if instance.get("execution_status") not in ("success", "failed"):
        raise RequestValidationError("$.execution_status: must be one of: ['success', 'failed']")

    request_snapshot = instance.get("request_snapshot")
    if not isinstance(request_snapshot, dict):
        raise RequestValidationError("$.request_snapshot: must be object")
    if not isinstance(request_snapshot.get("request_text"), str):
        raise RequestValidationError("$.request_snapshot.request_text: must be string")
    if not isinstance(request_snapshot.get("project_memory_note"), str):
        raise RequestValidationError("$.request_snapshot.project_memory_note: must be string")
    if not isinstance(request_snapshot.get("targets"), list):
        raise RequestValidationError("$.request_snapshot.targets: must be array")
    if not isinstance(request_snapshot.get("constraints"), dict):
        raise RequestValidationError("$.request_snapshot.constraints: must be object")
    if not isinstance(request_snapshot.get("model_choice"), dict):
        raise RequestValidationError("$.request_snapshot.model_choice: must be object")
    for key in ("candidate_data", "train_data", "mode"):
        if not isinstance(request_snapshot.get(key), str):
            raise RequestValidationError(f"$.request_snapshot.{key}: must be string")

    evidence = instance.get("evidence_snapshot")
    if not isinstance(evidence, dict):
        raise RequestValidationError("$.evidence_snapshot: must be object")
    if not isinstance(evidence.get("web_evidence_present"), bool):
        raise RequestValidationError("$.evidence_snapshot.web_evidence_present: must be boolean")
    if not isinstance(evidence.get("web_result_count"), int) or int(evidence.get("web_result_count")) < 0:
        raise RequestValidationError("$.evidence_snapshot.web_result_count: must be integer >= 0")
    if not isinstance(evidence.get("web_host_counts"), dict):
        raise RequestValidationError("$.evidence_snapshot.web_host_counts: must be object")
    for key in ("time_range", "query_effective"):
        if not isinstance(evidence.get(key), str):
            raise RequestValidationError(f"$.evidence_snapshot.{key}: must be string")

    runtime = instance.get("runtime_snapshot")
    if not isinstance(runtime, dict):
        raise RequestValidationError("$.runtime_snapshot: must be object")
    if not isinstance(runtime.get("record_count"), int) or int(runtime.get("record_count")) < 0:
        raise RequestValidationError("$.runtime_snapshot.record_count: must be integer >= 0")
    for key in ("tool_sequence", "failed_tools", "adapters", "selected_datasets"):
        val = runtime.get(key)
        if not isinstance(val, list):
            raise RequestValidationError(f"$.runtime_snapshot.{key}: must be array")
        for idx, item in enumerate(val, start=1):
            if not isinstance(item, str):
                raise RequestValidationError(f"$.runtime_snapshot.{key}[{idx}]: must be string")
    artifacts = runtime.get("artifacts")
    if not isinstance(artifacts, dict):
        raise RequestValidationError("$.runtime_snapshot.artifacts: must be object")
    for key in ("candidate_csv", "scored_csv", "final_output"):
        if not isinstance(artifacts.get(key), str):
            raise RequestValidationError(f"$.runtime_snapshot.artifacts.{key}: must be string")

    key_facts = instance.get("key_facts")
    if not isinstance(key_facts, list):
        raise RequestValidationError("$.key_facts: must be array")
    for idx, item in enumerate(key_facts, start=1):
        if not isinstance(item, str):
            raise RequestValidationError(f"$.key_facts[{idx}]: must be string")

    carry = instance.get("carry_over")
    if not isinstance(carry, dict):
        raise RequestValidationError("$.carry_over: must be object")
    if not isinstance(carry.get("exists"), bool):
        raise RequestValidationError("$.carry_over.exists: must be boolean")
    if not isinstance(carry.get("generated_at"), str):
        raise RequestValidationError("$.carry_over.generated_at: must be string")
    if not isinstance(carry.get("execution_status"), str):
        raise RequestValidationError("$.carry_over.execution_status: must be string")
    if not isinstance(carry.get("key_facts_head"), list):
        raise RequestValidationError("$.carry_over.key_facts_head: must be array")
    for idx, item in enumerate(carry.get("key_facts_head"), start=1):
        if not isinstance(item, str):
            raise RequestValidationError(f"$.carry_over.key_facts_head[{idx}]: must be string")


def load_and_validate_request_json(payload_path: Path, workspace_root: Path) -> Dict[str, Any]:
    payload = _load_json(payload_path)
    validate_request_payload(payload=payload, workspace_root=workspace_root)
    return payload


def validate_request_payload(payload: Dict[str, Any], workspace_root: Path) -> Dict[str, Any]:
    contract = default_contract_paths(workspace_root)
    schema = _load_json(contract.request_schema)
    _validate_via_jsonschema(instance=payload, schema=schema, contract_kind="request")
    return payload


def validate_decision_summary_payload(payload: Dict[str, Any], workspace_root: Path) -> Dict[str, Any]:
    contract = default_contract_paths(workspace_root)
    schema = _load_json(contract.decision_summary_schema)
    _validate_via_jsonschema(instance=payload, schema=schema, contract_kind="decision_summary")
    return payload


def validate_plan_payload(payload: Dict[str, Any], workspace_root: Path) -> Dict[str, Any]:
    contract = default_contract_paths(workspace_root)
    schema = _load_json(contract.plan_schema)
    _validate_via_jsonschema(instance=payload, schema=schema, contract_kind="plan")
    return payload


def validate_task_state_payload(payload: Dict[str, Any], workspace_root: Path) -> Dict[str, Any]:
    contract = default_contract_paths(workspace_root)
    schema = _load_json(contract.task_state_schema)
    _validate_via_jsonschema(instance=payload, schema=schema, contract_kind="task_state")
    return payload


def validate_data_report_payload(payload: Dict[str, Any], workspace_root: Path) -> Dict[str, Any]:
    contract = default_contract_paths(workspace_root)
    schema = _load_json(contract.data_report_schema)
    _validate_via_jsonschema(instance=payload, schema=schema, contract_kind="data_report")
    return payload


def validate_model_report_payload(payload: Dict[str, Any], workspace_root: Path) -> Dict[str, Any]:
    contract = default_contract_paths(workspace_root)
    schema = _load_json(contract.model_report_schema)
    _validate_via_jsonschema(instance=payload, schema=schema, contract_kind="model_report")
    return payload


def validate_filtering_report_payload(payload: Dict[str, Any], workspace_root: Path) -> Dict[str, Any]:
    contract = default_contract_paths(workspace_root)
    schema = _load_json(contract.filtering_report_schema)
    _validate_via_jsonschema(instance=payload, schema=schema, contract_kind="filtering_report")
    return payload


def validate_evaluation_report_payload(payload: Dict[str, Any], workspace_root: Path) -> Dict[str, Any]:
    contract = default_contract_paths(workspace_root)
    schema = _load_json(contract.evaluation_report_schema)
    _validate_via_jsonschema(instance=payload, schema=schema, contract_kind="evaluation_report")
    return payload


def validate_guardrails_report_payload(payload: Dict[str, Any], workspace_root: Path) -> Dict[str, Any]:
    contract = default_contract_paths(workspace_root)
    schema = _load_json(contract.guardrails_report_schema)
    _validate_via_jsonschema(instance=payload, schema=schema, contract_kind="guardrails_report")
    return payload


def validate_memory_context_payload(payload: Dict[str, Any], workspace_root: Path) -> Dict[str, Any]:
    contract = default_contract_paths(workspace_root)
    schema = _load_json(contract.memory_context_schema)
    _validate_via_jsonschema(instance=payload, schema=schema, contract_kind="memory_context")
    return payload


def validate_task_v2_payload(payload: Dict[str, Any], workspace_root: Path) -> Dict[str, Any]:
    contract = default_contract_paths(workspace_root)
    schema = _load_json(contract.task_v2_schema)
    _validate_via_jsonschema(instance=payload, schema=schema, contract_kind="task_v2")
    return payload


def validate_step_request_payload(payload: Dict[str, Any], workspace_root: Path) -> Dict[str, Any]:
    contract = default_contract_paths(workspace_root)
    schema = _load_json(contract.step_request_schema)
    _validate_via_jsonschema(instance=payload, schema=schema, contract_kind="step_request")
    return payload

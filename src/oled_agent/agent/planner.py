from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple

from oled_agent.agent.model_catalog import ModelCatalog
from oled_agent.agent.request_contract import RequestValidationError, validate_plan_payload, validate_request_payload
from oled_agent.agent.specs import (
    AgentPlan,
    BudgetSpec,
    ConstraintSpec,
    DesignSpec,
    ModelChoice,
    PropertyTarget,
    ToolCall,
)
from oled_agent.agent.tool_contracts import supported_tool_names, tool_arg_contracts

DEFAULT_PLANNER_PROVIDER = "rule_based_v1"
LLM_PLANNER_PROVIDER = "llm_v1"


class PlannerValidationError(ValueError):
    """Raised for user-facing planner input/model/contract errors."""


class PlannerProvider(Protocol):
    provider_id: str

    def build_plan(
        self,
        *,
        user_request: str,
        task_id: str,
        catalog_path: Path,
        predictor_id: str = "",
        generator_id: str = "",
        mode: str = "fast_screen",
    ) -> AgentPlan:
        ...

    def build_plan_from_request_payload(
        self,
        *,
        request_payload: Dict[str, Any],
        catalog_path: Path,
    ) -> AgentPlan:
        ...


def _infer_targets_from_text(request: str) -> List[PropertyTarget]:
    text = request.lower()
    targets: List[PropertyTarget] = []

    if "plqy" in text:
        targets.append(
            PropertyTarget(
                name="plqy",
                objective="maximize",
                weight=0.25,
                min_value=30.0,
                target_center=60.0,
                sigma=20.0,
            )
        )
    if "lambda" in text or "发射" in request or "emission" in text:
        center = None
        m = re.search(r"(\d{3})\s*nm", text)
        if m:
            center = float(m.group(1))
        targets.append(
            PropertyTarget(
                name="lambda_em",
                objective="target_window",
                target_center=center or 470.0,
                sigma=12.0,
                weight=0.65,
            )
        )

    if not targets:
        targets = [
            PropertyTarget(name="lambda_em", objective="target_window", target_center=470.0, sigma=12.0, weight=0.7),
            PropertyTarget(name="plqy", objective="maximize", target_center=60.0, sigma=20.0, weight=0.2, min_value=25.0),
        ]

    return targets


def _normalize_plqy_targets_in_payload(payload: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    normalized = dict(payload)
    raw_targets = payload.get("targets")
    if not isinstance(raw_targets, list):
        return normalized, []

    converted_fields: List[str] = []
    out_targets: List[Dict[str, Any]] = []
    for idx, target in enumerate(raw_targets):
        if not isinstance(target, dict):
            out_targets.append(target)  # keep original shape; schema validation handles type errors.
            continue
        item = dict(target)
        prop = str(item.get("property") or "").strip().lower()
        if prop == "plqy":
            for field in ("target_value", "target_min", "target_max"):
                value = item.get(field)
                if isinstance(value, (int, float)) and 0.0 <= float(value) <= 1.0:
                    item[field] = float(value) * 100.0
                    converted_fields.append(f"targets[{idx}].{field}")
        out_targets.append(item)

    normalized["targets"] = out_targets
    return normalized, converted_fields


def _infer_constraints_from_text(request: str) -> ConstraintSpec:
    text = request.lower()
    c = ConstraintSpec(mw_min=150.0, mw_max=700.0, domain_threshold=0.20)

    mw_match = re.search(r"mw\s*[<≤]\s*(\d+)", text)
    if mw_match:
        c.mw_max = float(mw_match.group(1))

    if "严格" in request or "strict" in text:
        c.domain_threshold = 0.30

    return c


def _constraints_to_payload(constraints: ConstraintSpec) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if isinstance(constraints.mw_max, (int, float)):
        out["mw_max"] = float(constraints.mw_max)
    return out


def _build_request_payload_from_text(
    *,
    user_request: str,
    task_id: str,
    mode: str,
    predictor_id: str,
    generator_id: str,
) -> Dict[str, Any]:
    inferred_targets = _infer_targets_from_text(user_request)
    inferred_constraints = _infer_constraints_from_text(user_request)
    targets: List[Dict[str, Any]] = []
    for t in inferred_targets:
        item: Dict[str, Any] = {
            "property": t.name,
            "objective": t.objective,
        }
        if t.objective == "target_window":
            if isinstance(t.target_center, (int, float)) and isinstance(t.sigma, (int, float)):
                item["target_min"] = float(t.target_center - t.sigma)
                item["target_max"] = float(t.target_center + t.sigma)
            elif isinstance(t.target_center, (int, float)):
                item["target_value"] = float(t.target_center)
        else:
            if isinstance(t.target_center, (int, float)):
                item["target_value"] = float(t.target_center)
            if isinstance(t.min_value, (int, float)):
                item["target_min"] = float(t.min_value)
            if isinstance(t.max_value, (int, float)):
                item["target_max"] = float(t.max_value)
        targets.append(item)

    payload: Dict[str, Any] = {
        "task_id": task_id,
        "request_text": user_request,
        "mode": mode,
        "targets": targets,
        "constraints": _constraints_to_payload(inferred_constraints),
        "budget": {"max_candidates": 500},
        "model_preferences": {
            "predictor_id": predictor_id,
            "generator_id": generator_id,
        },
    }
    return payload


def _to_request_payload_from_inputs(
    *,
    user_request: str,
    task_id: str,
    mode: str,
    predictor_id: str,
    generator_id: str,
    request_payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if request_payload is not None:
        return dict(request_payload)
    return _build_request_payload_from_text(
        user_request=user_request,
        task_id=task_id,
        mode=mode,
        predictor_id=predictor_id,
        generator_id=generator_id,
    )


def _pick_default_model_ids(catalog: ModelCatalog) -> ModelChoice:
    predictors = catalog.list(kind="predictor")
    generators = catalog.list(kind="generator")

    pred = predictors[0].id if predictors else ""
    gen = generators[0].id if generators else ""
    return ModelChoice(predictor_id=pred, generator_id=gen)


def _validate_planner_provider(planner_provider: str) -> str:
    provider = str(planner_provider or DEFAULT_PLANNER_PROVIDER).strip()
    if provider not in _PLANNER_PROVIDERS:
        providers = ", ".join(SUPPORTED_PLANNER_PROVIDERS)
        raise PlannerValidationError(f"Unknown planner_provider: {provider}. Supported: {providers}")
    return provider


def _apply_provider_metadata(
    *,
    plan: AgentPlan,
    requested_provider: str,
    effective_provider: str,
    status: str,
    reason: str = "",
) -> AgentPlan:
    metadata = dict(plan.design_spec.metadata or {})
    metadata["planner_provider_requested"] = requested_provider
    metadata["planner_provider_effective"] = effective_provider
    metadata["planner_provider_status"] = status
    metadata.pop("planner_provider_reason", None)
    if reason:
        metadata["planner_provider_reason"] = reason
    plan.design_spec.metadata = metadata
    return plan


def _is_web_evidence_enabled() -> bool:
    raw = str(os.environ.get("OLED_AGENT_ENABLE_WEB_EVIDENCE", "1")).strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _build_rule_based_plan(
    *,
    user_request: str,
    task_id: str,
    catalog_path: Path,
    predictor_id: str = "",
    generator_id: str = "",
    mode: str = "fast_screen",
) -> AgentPlan:
    catalog = ModelCatalog.load(catalog_path)

    default_choice = _pick_default_model_ids(catalog)
    predictor = predictor_id or default_choice.predictor_id
    generator = generator_id or default_choice.generator_id

    errors = catalog.validate_pair(predictor, generator)
    if errors:
        raise PlannerValidationError("; ".join(errors))

    design = DesignSpec(
        task_id=task_id,
        user_request=user_request,
        targets=_infer_targets_from_text(user_request),
        constraints=_infer_constraints_from_text(user_request),
        model_choice=ModelChoice(predictor_id=predictor, generator_id=generator),
        mode=mode,
        dataset_preferences=["master_database", "subsidiary_database"],
        metadata={"planner": "rule_based_v1"},
    )

    calls = [
        ToolCall(name="list_models", args={"kind": "predictor"}),
        ToolCall(name="list_models", args={"kind": "generator"}),
    ]
    if _is_web_evidence_enabled():
        calls.append(ToolCall(name="search_web_evidence", args={"query": user_request, "topk": 5}))
    calls.append(ToolCall(name="search_dataset", args={"preferences": design.dataset_preferences}))

    if mode == "train_then_design":
        calls.append(
            ToolCall(
                name="train_predictor",
                args={
                    "predictor_id": predictor,
                    "targets": [t.name for t in design.targets],
                },
            )
        )

    calls.extend(
        [
            ToolCall(
                name="generate_candidates",
                args={
                    "generator_id": generator,
                    "max_candidates": design.budget.max_candidates,
                    "constraints": design.constraints.to_dict() if hasattr(design.constraints, "to_dict") else {
                        "mw_min": design.constraints.mw_min,
                        "mw_max": design.constraints.mw_max,
                        "domain_threshold": design.constraints.domain_threshold,
                        "banned_alerts": design.constraints.banned_alerts,
                    },
                },
            ),
            ToolCall(
                name="score_candidates",
                args={
                    "predictor_id": predictor,
                    "targets": [t.name for t in design.targets],
                },
            ),
            ToolCall(name="filter_and_rank", args={"topn": 10}),
            ToolCall(name="make_report", args={}),
        ]
    )

    return AgentPlan(
        summary=(
            f"Design {len(design.targets)} target(s) with predictor={predictor}, "
            f"generator={generator}, mode={mode}"
        ),
        design_spec=design,
        tool_calls=calls,
    )


def _targets_from_payload(payload: Dict[str, Any]) -> List[PropertyTarget]:
    out: List[PropertyTarget] = []
    for item in payload.get("targets", []) or []:
        if not isinstance(item, dict):
            continue

        name = str(item.get("property") or "").strip()
        objective = str(item.get("objective") or "").strip()
        target_min = item.get("target_min")
        target_max = item.get("target_max")
        target_value = item.get("target_value")

        center = target_value
        sigma = None
        if objective == "target_window":
            if isinstance(target_min, (int, float)) and isinstance(target_max, (int, float)):
                center = (float(target_min) + float(target_max)) / 2.0
                sigma = max(1e-6, abs(float(target_max) - float(target_min)) / 2.0)
            elif isinstance(target_value, (int, float)):
                center = float(target_value)
                sigma = 12.0 if name == "lambda_em" else 0.2

        out.append(
            PropertyTarget(
                name=name,
                objective=objective,
                target_center=float(center) if isinstance(center, (int, float)) else None,
                sigma=float(sigma) if isinstance(sigma, (int, float)) else None,
                min_value=float(target_min) if isinstance(target_min, (int, float)) else None,
                max_value=float(target_max) if isinstance(target_max, (int, float)) else None,
            )
        )

    return out


def _constraints_from_payload(payload: Dict[str, Any]) -> ConstraintSpec:
    c = payload.get("constraints") or {}
    if not isinstance(c, dict):
        c = {}
    return ConstraintSpec(
        mw_max=float(c["mw_max"]) if isinstance(c.get("mw_max"), (int, float)) else None,
    )


def _budget_from_payload(payload: Dict[str, Any]) -> BudgetSpec:
    b = payload.get("budget") or {}
    if not isinstance(b, dict):
        b = {}
    max_candidates = b.get("max_candidates")
    if not isinstance(max_candidates, int):
        max_candidates = 500
    timeout_sec = b.get("timeout_sec") if isinstance(b.get("timeout_sec"), int) and int(b.get("timeout_sec")) >= 1 else None
    max_tool_calls = (
        b.get("max_tool_calls") if isinstance(b.get("max_tool_calls"), int) and int(b.get("max_tool_calls")) >= 1 else None
    )
    max_external_calls = (
        b.get("max_external_calls")
        if isinstance(b.get("max_external_calls"), int) and int(b.get("max_external_calls")) >= 0
        else None
    )
    on_limit = str(b.get("on_limit") or "fail").strip().lower()
    if on_limit not in {"fail", "need_approval"}:
        on_limit = "fail"
    return BudgetSpec(
        max_candidates=max_candidates,
        timeout_sec=timeout_sec,
        max_tool_calls=max_tool_calls,
        max_external_calls=max_external_calls,
        on_limit=on_limit,
    )


def _collect_generation_inputs(payload: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if not isinstance(payload, dict):
        return out

    generation_input = payload.get("generation_input")
    if isinstance(generation_input, dict):
        for key in ("source_image", "source_images", "source_pdf", "source_pdfs", "input_image", "input_pdf", "paper_path", "image_paths", "pdf_paths"):
            if key in generation_input and generation_input.get(key) is not None:
                out[key] = generation_input.get(key)

    constraints = payload.get("constraints")
    if isinstance(constraints, dict):
        for key in ("source_image", "source_images", "source_pdf", "source_pdfs", "input_image", "input_pdf", "paper_path", "image_paths", "pdf_paths"):
            if key in constraints and constraints.get(key) is not None:
                out[key] = constraints.get(key)

    for key in ("source_image", "source_images", "source_pdf", "source_pdfs", "input_image", "input_pdf", "paper_path", "image_paths", "pdf_paths"):
        if key in payload and payload.get(key) is not None:
            out[key] = payload.get(key)
    return out


def _generation_input_to_tool_args(generation_inputs: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(generation_inputs, dict):
        return {}
    out = dict(generation_inputs)
    return {k: v for k, v in out.items() if v not in (None, "", [], {})}


def _build_rule_based_plan_from_request_payload(
    *,
    request_payload: Dict[str, Any],
    catalog_path: Path,
) -> AgentPlan:
    request_payload, plqy_converted_fields = _normalize_plqy_targets_in_payload(request_payload)
    catalog = ModelCatalog.load(catalog_path)

    default_choice = _pick_default_model_ids(catalog)
    prefs = request_payload.get("model_preferences") or {}
    if not isinstance(prefs, dict):
        prefs = {}
    predictor = str(prefs.get("predictor_id") or default_choice.predictor_id)
    generator = str(prefs.get("generator_id") or default_choice.generator_id)

    errors = catalog.validate_pair(predictor, generator)
    if errors:
        raise PlannerValidationError("; ".join(errors))

    mode = str(request_payload.get("mode") or "fast_screen")
    task_id = str(request_payload.get("task_id") or "")
    request_text = str(request_payload.get("request_text") or "")
    targets = _targets_from_payload(request_payload)
    constraints = _constraints_from_payload(request_payload)
    budget = _budget_from_payload(request_payload)
    generation_inputs = _collect_generation_inputs(request_payload)

    design = DesignSpec(
        task_id=task_id,
        user_request=request_text,
        targets=targets,
        constraints=constraints,
        budget=budget,
        model_choice=ModelChoice(predictor_id=predictor, generator_id=generator),
        mode=mode,
        dataset_preferences=["master_database", "subsidiary_database"],
        metadata={"planner": "request_contract_v1"},
    )
    if plqy_converted_fields:
        design.metadata["plqy_scale"] = "percent_0_100"
        design.metadata["plqy_scale_converted_fields"] = plqy_converted_fields

    calls = [
        ToolCall(name="list_models", args={"kind": "predictor"}),
        ToolCall(name="list_models", args={"kind": "generator"}),
    ]
    if _is_web_evidence_enabled():
        calls.append(ToolCall(name="search_web_evidence", args={"query": request_text, "topk": 5}))
    calls.append(ToolCall(name="search_dataset", args={"preferences": design.dataset_preferences}))

    if mode == "train_then_design":
        calls.append(
            ToolCall(
                name="train_predictor",
                args={
                    "predictor_id": predictor,
                    "targets": [t.name for t in design.targets],
                },
            )
        )

    calls.extend(
        [
            ToolCall(
                name="generate_candidates",
                args={
                    "generator_id": generator,
                    "max_candidates": design.budget.max_candidates,
                    "constraints": design.constraints.to_dict(),
                    **_generation_input_to_tool_args(generation_inputs),
                },
            ),
            ToolCall(
                name="score_candidates",
                args={
                    "predictor_id": predictor,
                    "targets": [t.name for t in design.targets],
                },
            ),
            ToolCall(name="filter_and_rank", args={"topn": 10}),
            ToolCall(name="make_report", args={}),
        ]
    )

    return AgentPlan(
        summary=(
            f"Design {len(design.targets)} target(s) with predictor={predictor}, "
            f"generator={generator}, mode={mode}, source=request_json"
        ),
        design_spec=design,
        tool_calls=calls,
    )


def _validate_tool_calls(tool_calls: List[ToolCall], mode: str) -> None:
    allowed = set(supported_tool_names())
    seen_train = False
    seen_terminal = False
    for i, call in enumerate(tool_calls):
        if not isinstance(call.name, str) or not call.name.strip():
            raise PlannerValidationError(f"Invalid tool call at index {i}: empty name")
        name = call.name.strip()
        if name not in allowed:
            raise PlannerValidationError(f"Unsupported tool call: {name}")
        if name == "train_predictor":
            seen_train = True
        if name == "make_report":
            seen_terminal = True
    if mode == "train_then_design" and not seen_train:
        raise PlannerValidationError("train_then_design mode requires train_predictor in tool calls")
    if not seen_terminal:
        raise PlannerValidationError("Tool calls must include make_report as terminal step")


def _llm_prompt_template() -> str:
    return (
        "You are an OLED molecule design planner. "
        "Output ONLY valid JSON object with keys: summary, design_spec, tool_calls. "
        "Do not add explanations or markdown."
    )


def _llm_cmd_from_env() -> str:
    return str(os.environ.get("OLED_AGENT_LLM_PLANNER_CMD", "")).strip()


def _llm_backend_from_env() -> str:
    return str(os.environ.get("OLED_AGENT_LLM_BACKEND", "")).strip().lower()


def _parse_env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    text = str(raw).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError(f"Invalid {name}: {raw}")


def _llm_debug_error_enabled() -> bool:
    try:
        return _parse_env_bool("OLED_AGENT_LLM_DEBUG_ERROR", False)
    except RuntimeError:
        return False


def _redact_llm_error_detail(exc: Exception) -> str:
    detail = f"{type(exc).__name__}: {exc}"
    api_key = str(os.environ.get("OLED_AGENT_LLM_API_KEY", "") or "")
    if api_key:
        detail = detail.replace(api_key, "***")
    if len(detail) > 800:
        return detail[:800] + "...(truncated)"
    return detail


def _llm_backend_config_from_env(backend: str) -> Dict[str, Any]:
    if backend != "openai_compat":
        raise RuntimeError(f"Unsupported OLED_AGENT_LLM_BACKEND: {backend}")

    model = str(os.environ.get("OLED_AGENT_LLM_MODEL", "")).strip()
    api_key = str(os.environ.get("OLED_AGENT_LLM_API_KEY", "")).strip()
    base_url = str(os.environ.get("OLED_AGENT_LLM_BASE_URL", "https://api.openai.com/v1")).strip()
    chat_path_raw = str(os.environ.get("OLED_AGENT_LLM_CHAT_COMPLETIONS_PATH", "/chat/completions")).strip()
    auth_header = str(os.environ.get("OLED_AGENT_LLM_AUTH_HEADER", "Authorization")).strip()
    auth_scheme = str(os.environ.get("OLED_AGENT_LLM_AUTH_SCHEME", "Bearer")).strip()
    extra_headers_raw = str(os.environ.get("OLED_AGENT_LLM_EXTRA_HEADERS_JSON", "")).strip()
    timeout_raw = str(os.environ.get("OLED_AGENT_LLM_TIMEOUT_SEC", "60")).strip()
    max_retries_raw = str(os.environ.get("OLED_AGENT_LLM_BACKEND_MAX_RETRIES", "0")).strip()
    backoff_raw = str(os.environ.get("OLED_AGENT_LLM_BACKEND_BACKOFF_SEC", "1")).strip()

    if not model:
        raise RuntimeError("Missing OLED_AGENT_LLM_MODEL for openai_compat backend")
    if not api_key:
        raise RuntimeError("Missing OLED_AGENT_LLM_API_KEY for openai_compat backend")

    try:
        timeout_sec = float(timeout_raw)
    except ValueError as exc:
        raise RuntimeError(f"Invalid OLED_AGENT_LLM_TIMEOUT_SEC: {timeout_raw}") from exc
    timeout_sec = max(1.0, timeout_sec)
    try:
        max_retries = int(max_retries_raw)
    except ValueError as exc:
        raise RuntimeError(f"Invalid OLED_AGENT_LLM_BACKEND_MAX_RETRIES: {max_retries_raw}") from exc
    max_retries = max(0, max_retries)
    try:
        backoff_sec = float(backoff_raw)
    except ValueError as exc:
        raise RuntimeError(f"Invalid OLED_AGENT_LLM_BACKEND_BACKOFF_SEC: {backoff_raw}") from exc
    backoff_sec = max(0.0, backoff_sec)
    chat_path = chat_path_raw or "/chat/completions"
    if not chat_path.startswith("/"):
        chat_path = f"/{chat_path}"
    auth_header = auth_header or "Authorization"
    if extra_headers_raw:
        try:
            parsed_headers = json.loads(extra_headers_raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Invalid OLED_AGENT_LLM_EXTRA_HEADERS_JSON: must be JSON object") from exc
        if not isinstance(parsed_headers, dict):
            raise RuntimeError("Invalid OLED_AGENT_LLM_EXTRA_HEADERS_JSON: must be JSON object")
        extra_headers: Dict[str, str] = {}
        for k, v in parsed_headers.items():
            key = str(k).strip()
            if not key:
                raise RuntimeError("Invalid OLED_AGENT_LLM_EXTRA_HEADERS_JSON: header key cannot be empty")
            if isinstance(v, (dict, list)):
                raise RuntimeError("Invalid OLED_AGENT_LLM_EXTRA_HEADERS_JSON: header value must be scalar")
            extra_headers[key] = "" if v is None else str(v)
    else:
        extra_headers = {}
    disable_response_format = _parse_env_bool("OLED_AGENT_LLM_DISABLE_RESPONSE_FORMAT", False)

    return {
        "backend": backend,
        "model": model,
        "api_key": api_key,
        "base_url": base_url.rstrip("/"),
        "chat_completions_path": chat_path,
        "auth_header": auth_header,
        "auth_scheme": auth_scheme,
        "extra_headers": extra_headers,
        "disable_response_format": disable_response_format,
        "timeout_sec": timeout_sec,
        "max_retries": max_retries,
        "backoff_sec": backoff_sec,
    }


def _extract_json_object_text(raw_text: str) -> str:
    text = (raw_text or "").strip()
    if text.startswith("```"):
        m = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
        if m:
            text = m.group(1).strip()
    return text


def _run_openai_compat_planner(
    *,
    config: Dict[str, Any],
    payload: Dict[str, Any],
    catalog_path: Path,
) -> Dict[str, Any]:
    retryable_http_codes = {408, 409, 425, 429, 500, 502, 503, 504}

    request_base = {
        "model": config["model"],
        "temperature": 0,
        "messages": [
            {"role": "system", "content": _llm_prompt_template()},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "request": payload,
                        "catalog_path": str(catalog_path),
                    },
                    ensure_ascii=False,
                ),
            },
        ],
    }
    request_with_json_mode = {
        **request_base,
        "response_format": {"type": "json_object"},
    }
    endpoint = f"{config['base_url']}{config['chat_completions_path']}"

    def _is_retryable_error(exc: Exception) -> bool:
        if isinstance(exc, urllib.error.HTTPError):
            return int(exc.code) in retryable_http_codes
        return isinstance(exc, (urllib.error.URLError, TimeoutError))

    def _do_call(request_obj: Dict[str, Any]) -> str:
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        auth_scheme = str(config.get("auth_scheme", "Bearer"))
        auth_header = str(config.get("auth_header", "Authorization"))
        if auth_scheme:
            headers[auth_header] = f"{auth_scheme} {config['api_key']}"
        else:
            headers[auth_header] = str(config["api_key"])
        extra_headers = config.get("extra_headers") if isinstance(config.get("extra_headers"), dict) else {}
        for k, v in extra_headers.items():
            headers[str(k)] = str(v)
        req = urllib.request.Request(
            endpoint,
            data=json.dumps(request_obj, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=float(config["timeout_sec"])) as resp:
            return resp.read().decode("utf-8", errors="replace")

    def _do_call_with_retry(request_obj: Dict[str, Any]) -> str:
        max_retries = int(config.get("max_retries", 0))
        backoff_sec = float(config.get("backoff_sec", 1.0))
        attempt = 0
        while True:
            try:
                return _do_call(request_obj)
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
                if attempt >= max_retries or not _is_retryable_error(exc):
                    raise
                if backoff_sec > 0:
                    time.sleep(backoff_sec * (2**attempt))
                attempt += 1

    disable_response_format = bool(config.get("disable_response_format", False))
    try:
        if disable_response_format:
            raw = _do_call_with_retry(request_base)
        else:
            raw = _do_call_with_retry(request_with_json_mode)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        # Some OpenAI-compatible backends may not support response_format.
        if (not disable_response_format) and exc.code in (400, 404, 415, 422):
            try:
                raw = _do_call_with_retry(request_base)
            except urllib.error.HTTPError as exc2:
                detail2 = exc2.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"LLM backend HTTP error: {exc2.code} {detail2}") from exc2
            except urllib.error.URLError as exc2:
                raise RuntimeError(f"LLM backend network error: {exc2}") from exc2
            except TimeoutError as exc2:
                raise RuntimeError("LLM backend timeout") from exc2
        else:
            raise RuntimeError(f"LLM backend HTTP error: {exc.code} {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"LLM backend network error: {exc}") from exc
    except TimeoutError as exc:
        raise RuntimeError("LLM backend timeout") from exc

    try:
        body = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise PlannerValidationError(f"LLM backend returned non-JSON envelope: {exc}") from exc

    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise PlannerValidationError("LLM backend response missing choices")
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        raise PlannerValidationError("LLM backend response missing message object")
    content = message.get("content")
    if isinstance(content, str):
        content_text = content
    elif isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        content_text = "".join(parts)
    else:
        raise PlannerValidationError("LLM backend response message.content must be string")

    parsed_text = _extract_json_object_text(content_text)
    try:
        return json.loads(parsed_text)
    except json.JSONDecodeError as exc:
        raise PlannerValidationError(f"LLM backend content is not valid JSON: {exc}") from exc


def _run_llm_backend(
    *,
    backend: str,
    payload: Dict[str, Any],
    catalog_path: Path,
) -> Dict[str, Any]:
    config = _llm_backend_config_from_env(backend)
    if backend == "openai_compat":
        return _run_openai_compat_planner(
            config=config,
            payload=payload,
            catalog_path=catalog_path,
        )
    raise RuntimeError(f"Unsupported LLM backend: {backend}")


def _run_llm_planner_command(
    *,
    cmd: str,
    payload: Dict[str, Any],
    catalog_path: Path,
) -> Dict[str, Any]:
    request = {
        "prompt": _llm_prompt_template(),
        "request": payload,
        "catalog_path": str(catalog_path),
    }
    cp = subprocess.run(
        shlex.split(cmd),
        input=json.dumps(request, ensure_ascii=False),
        text=True,
        capture_output=True,
        check=True,
    )
    raw = (cp.stdout or "").strip()
    if not raw:
        raise PlannerValidationError("LLM planner command returned empty output")
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise PlannerValidationError(f"LLM planner output is not valid JSON: {exc}") from exc


def _targets_to_specs(payload: List[Dict[str, Any]]) -> List[PropertyTarget]:
    return _targets_from_payload({"targets": payload})


def _budget_to_spec(payload: Dict[str, Any]) -> BudgetSpec:
    return _budget_from_payload({"budget": payload})


def _constraints_to_spec(payload: Dict[str, Any]) -> ConstraintSpec:
    return _constraints_from_payload({"constraints": payload})


def _tool_calls_from_payload(payload: List[Dict[str, Any]]) -> List[ToolCall]:
    out: List[ToolCall] = []
    for i, item in enumerate(payload):
        if not isinstance(item, dict):
            raise PlannerValidationError(f"tool_calls[{i}] must be object")
        name = str(item.get("name") or "").strip()
        args = item.get("args") or {}
        if not isinstance(args, dict):
            raise PlannerValidationError(f"tool_calls[{i}].args must be object")
        out.append(ToolCall(name=name, args=dict(args)))
    return out


def _parse_tool_args(value: Any, *, index: int) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise PlannerValidationError(f"tool_calls[{index}].args is not valid JSON object string") from exc
        if not isinstance(parsed, dict):
            raise PlannerValidationError(f"tool_calls[{index}].args JSON string must decode to object")
        return dict(parsed)
    raise PlannerValidationError(f"tool_calls[{index}].args must be object")


def _normalize_model_kind(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    text = value.strip().lower()
    if text in {"predictor", "predictors", "pred", "prediction", "scorer", "score"}:
        return "predictor"
    if text in {"generator", "generators", "gen", "generation"}:
        return "generator"
    return ""


def _expand_load_model_catalog_alias(args: Dict[str, Any]) -> List[Dict[str, Any]]:
    kinds: List[str] = []

    for key in ("kind", "model_kind", "type"):
        kind = _normalize_model_kind(args.get(key))
        if kind:
            kinds.append(kind)

    for key in ("kinds", "model_kinds", "types"):
        value = args.get(key)
        if isinstance(value, list):
            for item in value:
                kind = _normalize_model_kind(item)
                if kind:
                    kinds.append(kind)

    if not kinds:
        kinds = ["predictor", "generator"]

    ordered_unique_kinds: List[str] = []
    for kind in kinds:
        if kind not in ordered_unique_kinds:
            ordered_unique_kinds.append(kind)

    return [{"name": "list_models", "args": {"kind": kind}} for kind in ordered_unique_kinds]


def _canonicalize_tool_call_name_and_args(name: str, args: Dict[str, Any]) -> List[Dict[str, Any]]:
    contracts = tool_arg_contracts()

    def _sanitize(tool_name: str, tool_args: Dict[str, Any]) -> Dict[str, Any]:
        spec = contracts.get(tool_name)
        if not isinstance(spec, dict):
            return dict(tool_args)
        properties = spec.get("properties")
        if not isinstance(properties, dict):
            return dict(tool_args)
        allowed = set(properties.keys())
        return {k: v for k, v in tool_args.items() if k in allowed}

    normalized_name = str(name or "").strip()
    alias = normalized_name.lower()
    if alias in {
        "load_model_catalog",
        "load_models_catalog",
        "list_model_catalog",
        "get_model_catalog",
        "load_models",
    }:
        expanded = _expand_load_model_catalog_alias(args)
        return [{"name": item["name"], "args": _sanitize(item["name"], item["args"])} for item in expanded]
    return [{"name": normalized_name, "args": _sanitize(normalized_name, args)}]


def _normalize_tool_call_items(raw_tool_calls: Any) -> List[Dict[str, Any]]:
    # Accept common OpenAI-compatible variants:
    # - {"name": "...", "args": {...}}
    # - {"tool_name": "...", "arguments": {...}}
    # - {"type": "function", "function": {"name": "...", "arguments": "{\"k\":1}"}}
    if isinstance(raw_tool_calls, str):
        try:
            raw_tool_calls = json.loads(raw_tool_calls)
        except json.JSONDecodeError as exc:
            raise PlannerValidationError("tool_calls string is not valid JSON array") from exc
    if not isinstance(raw_tool_calls, list):
        raise PlannerValidationError("LLM planner result missing tool_calls array")

    normalized: List[Dict[str, Any]] = []
    for i, item in enumerate(raw_tool_calls):
        if not isinstance(item, dict):
            raise PlannerValidationError(f"tool_calls[{i}] must be object")

        name = ""
        for k in ("name", "tool", "tool_name", "function_name"):
            v = item.get(k)
            if isinstance(v, str) and v.strip():
                name = v.strip()
                break

        fn_obj = item.get("function")
        if not name and isinstance(fn_obj, dict):
            v = fn_obj.get("name")
            if isinstance(v, str) and v.strip():
                name = v.strip()

        custom_obj = item.get("custom")
        if not name and isinstance(custom_obj, dict):
            v = custom_obj.get("name")
            if isinstance(v, str) and v.strip():
                name = v.strip()

        args_source: Any = None
        has_args = False
        for k in ("args", "arguments", "params", "parameters", "input"):
            if k in item:
                args_source = item.get(k)
                has_args = True
                break
        if not has_args and isinstance(fn_obj, dict):
            for k in ("arguments", "args", "input"):
                if k in fn_obj:
                    args_source = fn_obj.get(k)
                    has_args = True
                    break
        if not has_args and isinstance(custom_obj, dict):
            if "input" in custom_obj:
                args_source = custom_obj.get("input")
                has_args = True

        args = _parse_tool_args(args_source if has_args else {}, index=i)
        normalized.extend(_canonicalize_tool_call_name_and_args(name, args))

    return normalized


def _normalize_llm_result_payload(
    *,
    llm_result: Dict[str, Any],
    fallback_payload: Dict[str, Any],
) -> Dict[str, Any]:
    if not isinstance(llm_result, dict):
        raise PlannerValidationError("LLM planner result must be object")
    summary = llm_result.get("summary")
    design_spec = llm_result.get("design_spec")
    tool_calls_raw = llm_result.get("tool_calls")
    if tool_calls_raw is None and isinstance(llm_result.get("function_call"), dict):
        tool_calls_raw = [llm_result.get("function_call")]
    if not isinstance(summary, str) or not summary.strip():
        raise PlannerValidationError("LLM planner result missing non-empty summary")
    if not isinstance(design_spec, dict):
        raise PlannerValidationError("LLM planner result missing design_spec object")
    tool_calls = _normalize_tool_call_items(tool_calls_raw)

    # Merge with schema-validated fallback payload for contract stability.
    merged = dict(fallback_payload)
    merged["targets"] = design_spec.get("targets") if isinstance(design_spec.get("targets"), list) else merged.get("targets")
    merged["constraints"] = design_spec.get("constraints") if isinstance(design_spec.get("constraints"), dict) else merged.get("constraints", {})
    merged["budget"] = design_spec.get("budget") if isinstance(design_spec.get("budget"), dict) else merged.get("budget")
    merged_prefs = merged.get("model_preferences") if isinstance(merged.get("model_preferences"), dict) else {}
    if isinstance(merged_prefs, dict):
        predictor_id = (
            design_spec.get("model_choice", {}).get("predictor_id")
            if isinstance(design_spec.get("model_choice"), dict)
            else None
        )
        generator_id = (
            design_spec.get("model_choice", {}).get("generator_id")
            if isinstance(design_spec.get("model_choice"), dict)
            else None
        )
        if isinstance(predictor_id, str) and predictor_id.strip():
            merged_prefs["predictor_id"] = predictor_id.strip()
        if isinstance(generator_id, str) and generator_id.strip():
            merged_prefs["generator_id"] = generator_id.strip()
        merged["model_preferences"] = merged_prefs
    validate_request_payload(payload=merged, workspace_root=Path(__file__).resolve().parents[3])

    plan_like_payload = {
        "summary": summary.strip(),
        "design_spec": {
            "task_id": merged.get("task_id", ""),
            "request_text": merged.get("request_text", ""),
            "mode": merged.get("mode", "fast_screen"),
            "targets": _targets_from_payload({"targets": merged.get("targets") or []}),
            "constraints": _constraints_from_payload({"constraints": merged.get("constraints") or {}}).to_dict(),
            "budget": _budget_from_payload({"budget": merged.get("budget") or {}}).__dict__,
            "model_choice": {
                "predictor_id": str((merged.get("model_preferences") or {}).get("predictor_id") or ""),
                "generator_id": str((merged.get("model_preferences") or {}).get("generator_id") or ""),
            },
        },
        "tool_calls": tool_calls,
    }
    # Use schema contract for plan payload guardrail.
    validate_plan_payload(
        payload={
            "summary": plan_like_payload["summary"],
            "design_spec": {
                "task_id": plan_like_payload["design_spec"]["task_id"],
                "request_text": plan_like_payload["design_spec"]["request_text"],
                "mode": plan_like_payload["design_spec"]["mode"],
                "targets": [
                    {
                        "name": t.name,
                        "objective": t.objective,
                        "target_center": t.target_center,
                        "sigma": t.sigma,
                        "min_value": t.min_value,
                        "max_value": t.max_value,
                        "weight": t.weight,
                    }
                    for t in plan_like_payload["design_spec"]["targets"]
                ],
                "constraints": plan_like_payload["design_spec"]["constraints"],
                "budget": plan_like_payload["design_spec"]["budget"],
                "model_choice": plan_like_payload["design_spec"]["model_choice"],
            },
            "tool_calls": plan_like_payload["tool_calls"],
        },
        workspace_root=Path(__file__).resolve().parents[3],
    )

    return {
        "summary": summary.strip(),
        "request_payload": merged,
        "tool_calls": tool_calls,
    }


def _build_plan_from_normalized_payload(
    *,
    normalized_payload: Dict[str, Any],
    tool_calls_payload: List[Dict[str, Any]],
    metadata_planner: str,
    summary: str,
    catalog_path: Path,
) -> AgentPlan:
    payload, plqy_converted_fields = _normalize_plqy_targets_in_payload(normalized_payload)
    catalog = ModelCatalog.load(catalog_path)

    prefs = payload.get("model_preferences") or {}
    predictor = str(prefs.get("predictor_id") or "")
    generator = str(prefs.get("generator_id") or "")
    errors = catalog.validate_pair(predictor, generator)
    if errors:
        raise PlannerValidationError("; ".join(errors))

    mode = str(payload.get("mode") or "fast_screen")
    generation_inputs = _collect_generation_inputs(payload)
    design = DesignSpec(
        task_id=str(payload.get("task_id") or ""),
        user_request=str(payload.get("request_text") or ""),
        targets=_targets_to_specs(payload.get("targets") or []),
        constraints=_constraints_to_spec(payload.get("constraints") or {}),
        budget=_budget_to_spec(payload.get("budget") or {}),
        model_choice=ModelChoice(predictor_id=predictor, generator_id=generator),
        mode=mode,
        dataset_preferences=["master_database", "subsidiary_database"],
        metadata={"planner": metadata_planner},
    )
    if plqy_converted_fields:
        design.metadata["plqy_scale"] = "percent_0_100"
        design.metadata["plqy_scale_converted_fields"] = plqy_converted_fields
    calls = _tool_calls_from_payload(tool_calls_payload)
    # Merge request-level generation inputs into generate_candidates args for
    # image/pdf-conditioned generators (e.g., MolScribe) without requiring
    # callers to handcraft tool-call args.
    merged_generation_args = _generation_input_to_tool_args(generation_inputs)
    if merged_generation_args:
        for call in calls:
            if call.name == "generate_candidates":
                merged = dict(merged_generation_args)
                merged.update(call.args)
                call.args = merged
    _validate_tool_calls(calls, mode=mode)
    return AgentPlan(
        summary=summary,
        design_spec=design,
        tool_calls=calls,
    )


class RuleBasedPlannerProvider:
    provider_id = DEFAULT_PLANNER_PROVIDER

    def build_plan(
        self,
        *,
        user_request: str,
        task_id: str,
        catalog_path: Path,
        predictor_id: str = "",
        generator_id: str = "",
        mode: str = "fast_screen",
    ) -> AgentPlan:
        return _build_rule_based_plan(
            user_request=user_request,
            task_id=task_id,
            catalog_path=catalog_path,
            predictor_id=predictor_id,
            generator_id=generator_id,
            mode=mode,
        )

    def build_plan_from_request_payload(
        self,
        *,
        request_payload: Dict[str, Any],
        catalog_path: Path,
    ) -> AgentPlan:
        return _build_rule_based_plan_from_request_payload(
            request_payload=request_payload,
            catalog_path=catalog_path,
        )


class LlmPlannerProvider:
    provider_id = LLM_PLANNER_PROVIDER

    def __init__(self, fallback_provider: PlannerProvider):
        self._fallback_provider = fallback_provider

    def _fallback_from_text(
        self,
        *,
        reason: str,
        user_request: str,
        task_id: str,
        catalog_path: Path,
        predictor_id: str,
        generator_id: str,
        mode: str,
        error_detail: str = "",
    ) -> AgentPlan:
        plan = self._fallback_provider.build_plan(
            user_request=user_request,
            task_id=task_id,
            catalog_path=catalog_path,
            predictor_id=predictor_id,
            generator_id=generator_id,
            mode=mode,
        )
        plan = _apply_provider_metadata(
            plan=plan,
            requested_provider=self.provider_id,
            effective_provider=self._fallback_provider.provider_id,
            status="fallback",
            reason=reason,
        )
        if error_detail:
            md = dict(plan.design_spec.metadata or {})
            md["planner_provider_error_detail"] = error_detail
            plan.design_spec.metadata = md
        return plan

    def _fallback_from_payload(
        self,
        *,
        reason: str,
        request_payload: Dict[str, Any],
        catalog_path: Path,
        error_detail: str = "",
    ) -> AgentPlan:
        plan = self._fallback_provider.build_plan_from_request_payload(
            request_payload=request_payload,
            catalog_path=catalog_path,
        )
        plan = _apply_provider_metadata(
            plan=plan,
            requested_provider=self.provider_id,
            effective_provider=self._fallback_provider.provider_id,
            status="fallback",
            reason=reason,
        )
        if error_detail:
            md = dict(plan.design_spec.metadata or {})
            md["planner_provider_error_detail"] = error_detail
            plan.design_spec.metadata = md
        return plan

    def _fallback_reason(self, exc: Exception, source: str) -> str:
        if source == "none":
            return "llm_provider_not_implemented"
        if source == "command":
            if isinstance(exc, subprocess.CalledProcessError):
                return "llm_command_failed"
            if isinstance(exc, FileNotFoundError):
                return "llm_command_failed"
            return "llm_output_invalid"
        if source == "backend":
            if isinstance(exc, (PlannerValidationError, RequestValidationError)):
                return "llm_output_invalid"
            return "llm_backend_failed"
        return "llm_output_invalid"

    def _invoke_llm_source(
        self,
        *,
        payload: Dict[str, Any],
        catalog_path: Path,
    ) -> Tuple[Dict[str, Any], str]:
        cmd = _llm_cmd_from_env()
        if cmd:
            return (
                _run_llm_planner_command(
                    cmd=cmd,
                    payload=payload,
                    catalog_path=catalog_path,
                ),
                "command",
            )
        backend = _llm_backend_from_env()
        if backend:
            return (
                _run_llm_backend(
                    backend=backend,
                    payload=payload,
                    catalog_path=catalog_path,
                ),
                "backend",
            )
        raise PlannerValidationError("LLM planner backend is not configured")

    def build_plan(
        self,
        *,
        user_request: str,
        task_id: str,
        catalog_path: Path,
        predictor_id: str = "",
        generator_id: str = "",
        mode: str = "fast_screen",
    ) -> AgentPlan:
        catalog = ModelCatalog.load(catalog_path)
        defaults = _pick_default_model_ids(catalog)
        resolved_predictor = predictor_id or defaults.predictor_id
        resolved_generator = generator_id or defaults.generator_id
        base_payload = _to_request_payload_from_inputs(
            user_request=user_request,
            task_id=task_id,
            mode=mode,
            predictor_id=resolved_predictor,
            generator_id=resolved_generator,
        )
        source = "none"
        if _llm_cmd_from_env():
            source = "command"
        elif _llm_backend_from_env():
            source = "backend"
        try:
            llm_raw, source = self._invoke_llm_source(
                payload=base_payload,
                catalog_path=catalog_path,
            )
            normalized = _normalize_llm_result_payload(
                llm_result=llm_raw,
                fallback_payload=base_payload,
            )
            plan = _build_plan_from_normalized_payload(
                normalized_payload=normalized["request_payload"],
                tool_calls_payload=normalized["tool_calls"],
                metadata_planner=self.provider_id,
                summary=normalized["summary"],
                catalog_path=catalog_path,
            )
            return _apply_provider_metadata(
                plan=plan,
                requested_provider=self.provider_id,
                effective_provider=self.provider_id,
                status="active",
            )
        except Exception as exc:
            error_detail = _redact_llm_error_detail(exc) if _llm_debug_error_enabled() else ""
            return self._fallback_from_text(
                reason=self._fallback_reason(exc, source),
                user_request=user_request,
                task_id=task_id,
                catalog_path=catalog_path,
                predictor_id=resolved_predictor,
                generator_id=resolved_generator,
                mode=mode,
                error_detail=error_detail,
            )

    def build_plan_from_request_payload(
        self,
        *,
        request_payload: Dict[str, Any],
        catalog_path: Path,
    ) -> AgentPlan:
        base_payload = dict(request_payload)
        source = "none"
        if _llm_cmd_from_env():
            source = "command"
        elif _llm_backend_from_env():
            source = "backend"
        try:
            llm_raw, source = self._invoke_llm_source(
                payload=base_payload,
                catalog_path=catalog_path,
            )
            normalized = _normalize_llm_result_payload(
                llm_result=llm_raw,
                fallback_payload=base_payload,
            )
            plan = _build_plan_from_normalized_payload(
                normalized_payload=normalized["request_payload"],
                tool_calls_payload=normalized["tool_calls"],
                metadata_planner=self.provider_id,
                summary=normalized["summary"],
                catalog_path=catalog_path,
            )
            return _apply_provider_metadata(
                plan=plan,
                requested_provider=self.provider_id,
                effective_provider=self.provider_id,
                status="active",
            )
        except Exception as exc:
            error_detail = _redact_llm_error_detail(exc) if _llm_debug_error_enabled() else ""
            return self._fallback_from_payload(
                reason=self._fallback_reason(exc, source),
                request_payload=base_payload,
                catalog_path=catalog_path,
                error_detail=error_detail,
            )


_RULE_BASED_PROVIDER = RuleBasedPlannerProvider()
_PLANNER_PROVIDERS: Dict[str, PlannerProvider] = {
    _RULE_BASED_PROVIDER.provider_id: _RULE_BASED_PROVIDER,
    LLM_PLANNER_PROVIDER: LlmPlannerProvider(fallback_provider=_RULE_BASED_PROVIDER),
}
SUPPORTED_PLANNER_PROVIDERS = tuple(_PLANNER_PROVIDERS.keys())


def build_plan(
    *,
    user_request: str,
    task_id: str,
    catalog_path: Path,
    predictor_id: str = "",
    generator_id: str = "",
    mode: str = "fast_screen",
    planner_provider: str = DEFAULT_PLANNER_PROVIDER,
) -> AgentPlan:
    provider = _validate_planner_provider(planner_provider)
    plan = _PLANNER_PROVIDERS[provider].build_plan(
        user_request=user_request,
        task_id=task_id,
        catalog_path=catalog_path,
        predictor_id=predictor_id,
        generator_id=generator_id,
        mode=mode,
    )
    if provider == DEFAULT_PLANNER_PROVIDER:
        return _apply_provider_metadata(
            plan=plan,
            requested_provider=provider,
            effective_provider=provider,
            status="active",
        )
    return plan


def build_plan_from_request_payload(
    *,
    request_payload: Dict[str, Any],
    catalog_path: Path,
    planner_provider: str = DEFAULT_PLANNER_PROVIDER,
) -> AgentPlan:
    provider = _validate_planner_provider(planner_provider)
    plan = _PLANNER_PROVIDERS[provider].build_plan_from_request_payload(
        request_payload=request_payload,
        catalog_path=catalog_path,
    )
    if provider == DEFAULT_PLANNER_PROVIDER:
        return _apply_provider_metadata(
            plan=plan,
            requested_provider=provider,
            effective_provider=provider,
            status="active",
        )
    return plan

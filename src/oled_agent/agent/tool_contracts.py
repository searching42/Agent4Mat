from __future__ import annotations

from typing import Any, Dict, List


class ToolContractValidationError(ValueError):
    """Raised when a tool call does not satisfy the shared tool contract."""


_TOOL_ARG_CONTRACTS: Dict[str, Dict[str, Any]] = {
    "list_models": {
        "required": ["kind"],
        "properties": {
            "kind": {"type": "string", "enum": ["predictor", "generator"]},
        },
    },
    "search_dataset": {
        "required": ["preferences"],
        "properties": {
            "preferences": {
                "type": "array",
                "minItems": 1,
                "items": {"type": "string", "minLength": 1},
            },
            "use_web_search": {"type": "boolean"},
            "web_topk": {"type": "integer", "minimum": 1},
        },
    },
    "search_web_evidence": {
        "required": ["query"],
        "properties": {
            "query": {"type": "string", "minLength": 1},
            "topk": {"type": "integer", "minimum": 1},
            "domains": {"type": "array", "items": {"type": "string", "minLength": 1}},
            "time_range": {"type": "string"},
        },
    },
    "retrieve_candidate_data": {
        "required": [],
        "properties": {
            "candidate_data": {"type": "string"},
            "output_csv": {"type": "string"},
        },
    },
    "clean_dataset": {
        "required": [],
        "properties": {
            "input_csv": {"type": "string"},
            "output_csv": {"type": "string"},
            "constraints": {"type": "object"},
        },
    },
    "prepare_train_data": {
        "required": [],
        "properties": {
            "train_data": {"type": "string"},
            "output_csv": {"type": "string"},
        },
    },
    "train_predictor": {
        "required": ["predictor_id", "targets"],
        "properties": {
            "predictor_id": {"type": "string", "minLength": 1},
            "targets": {
                "type": "array",
                "minItems": 1,
                "items": {"type": "string", "minLength": 1},
            },
            "target_specs": {
                "type": "array",
                "items": {"type": "object"},
            },
        },
    },
    "generate_candidates": {
        "required": ["generator_id"],
        "properties": {
            "generator_id": {"type": "string", "minLength": 1},
            "max_candidates": {"type": "integer", "minimum": 1},
            "input_csv": {"type": "string"},
            "constraints": {"type": "object"},
            "source_image": {"type": "string", "minLength": 1},
            "source_images": {"type": "array", "minItems": 1, "items": {"type": "string", "minLength": 1}},
            "source_pdf": {"type": "string", "minLength": 1},
            "source_pdfs": {"type": "array", "minItems": 1, "items": {"type": "string", "minLength": 1}},
            "input_image": {"type": "string", "minLength": 1},
            "input_pdf": {"type": "string", "minLength": 1},
            "paper_path": {"type": "string", "minLength": 1},
            "image_paths": {"type": "array", "minItems": 1, "items": {"type": "string", "minLength": 1}},
            "pdf_paths": {"type": "array", "minItems": 1, "items": {"type": "string", "minLength": 1}},
        },
    },
    "score_candidates": {
        "required": ["predictor_id", "targets"],
        "properties": {
            "predictor_id": {"type": "string", "minLength": 1},
            "targets": {
                "type": "array",
                "minItems": 1,
                "items": {"type": "string", "minLength": 1},
            },
            "target_specs": {
                "type": "array",
                "items": {"type": "object"},
            },
        },
    },
    "filter_and_rank": {
        "required": ["topn"],
        "properties": {
            "topn": {"type": "integer", "minimum": 1},
            "target_specs": {
                "type": "array",
                "items": {"type": "object"},
            },
        },
    },
    "make_report": {
        "required": [],
        "properties": {},
    },
}


def supported_tool_names() -> List[str]:
    return list(_TOOL_ARG_CONTRACTS.keys())


def tool_arg_contracts() -> Dict[str, Dict[str, Any]]:
    # Return copy-shallow to avoid accidental mutation from callers.
    return {
        k: {
            "required": list(v.get("required", [])),
            "properties": dict(v.get("properties", {})),
        }
        for k, v in _TOOL_ARG_CONTRACTS.items()
    }


def _ensure_type(value: Any, expected: str, path: str) -> None:
    if expected == "string":
        if not isinstance(value, str):
            raise ToolContractValidationError(f"{path}: must be string")
        return
    if expected == "integer":
        if not isinstance(value, int):
            raise ToolContractValidationError(f"{path}: must be integer")
        return
    if expected == "number":
        if not isinstance(value, (int, float)):
            raise ToolContractValidationError(f"{path}: must be number")
        return
    if expected == "boolean":
        if not isinstance(value, bool):
            raise ToolContractValidationError(f"{path}: must be boolean")
        return
    if expected == "object":
        if not isinstance(value, dict):
            raise ToolContractValidationError(f"{path}: must be object")
        return
    if expected == "array":
        if not isinstance(value, list):
            raise ToolContractValidationError(f"{path}: must be array")
        return
    raise ToolContractValidationError(f"{path}: unsupported contract type: {expected}")


def _validate_value(value: Any, field_spec: Dict[str, Any], path: str) -> None:
    expected_type = field_spec.get("type")
    if isinstance(expected_type, str):
        _ensure_type(value, expected_type, path)

    enum_values = field_spec.get("enum")
    if isinstance(enum_values, list) and enum_values:
        if value not in enum_values:
            raise ToolContractValidationError(f"{path}: must be one of: {enum_values}")

    if isinstance(value, str):
        min_length = field_spec.get("minLength")
        if isinstance(min_length, int) and len(value) < min_length:
            raise ToolContractValidationError(f"{path}: must have length >= {min_length}")

    if isinstance(value, int):
        minimum = field_spec.get("minimum")
        if isinstance(minimum, int) and value < minimum:
            raise ToolContractValidationError(f"{path}: must be >= {minimum}")

    if isinstance(value, list):
        min_items = field_spec.get("minItems")
        if isinstance(min_items, int) and len(value) < min_items:
            raise ToolContractValidationError(f"{path}: must have at least {min_items} item(s)")
        items_spec = field_spec.get("items")
        if isinstance(items_spec, dict):
            for idx, item in enumerate(value, start=1):
                _validate_value(item, items_spec, f"{path}[{idx}]")


def validate_tool_call_args(*, name: str, args: Any, path: str) -> None:
    """
    Validate a tool call against shared contract.
    `path` should point to the tool-call object path, e.g. '$.tool_calls[2]'.
    """
    tool_name = str(name or "").strip()
    if not tool_name:
        raise ToolContractValidationError(f"{path}.name: must be non-empty string")

    spec = _TOOL_ARG_CONTRACTS.get(tool_name)
    if spec is None:
        raise ToolContractValidationError(f"{path}.name: unsupported tool: {tool_name}")

    if not isinstance(args, dict):
        raise ToolContractValidationError(f"{path}.args: must be object")

    properties = spec.get("properties", {})
    required = spec.get("required", [])
    if not isinstance(properties, dict):
        properties = {}
    if not isinstance(required, list):
        required = []

    extras = sorted(k for k in args.keys() if k not in properties)
    if extras:
        raise ToolContractValidationError(f"{path}.args: unexpected field(s): {extras}")

    for key in required:
        if key not in args:
            raise ToolContractValidationError(f"{path}.args.{key}: missing required field")

    for key, value in args.items():
        field_spec = properties.get(key)
        if isinstance(field_spec, dict):
            _validate_value(value, field_spec, f"{path}.args.{key}")


def build_plan_tool_call_item_schema() -> Dict[str, Any]:
    names = supported_tool_names()
    item_schema: Dict[str, Any] = {
        "type": "object",
        "additionalProperties": False,
        "required": ["name", "args"],
        "properties": {
            "name": {"type": "string", "enum": names},
            "args": {"type": "object"},
        },
        "allOf": [],
    }
    all_of: List[Dict[str, Any]] = item_schema["allOf"]
    for name in names:
        spec = _TOOL_ARG_CONTRACTS[name]
        args_schema = {
            "type": "object",
            "additionalProperties": False,
            "required": list(spec.get("required", [])),
            "properties": dict(spec.get("properties", {})),
        }
        all_of.append(
            {
                "if": {"properties": {"name": {"const": name}}},
                "then": {"properties": {"args": args_schema}},
            }
        )
    return item_schema

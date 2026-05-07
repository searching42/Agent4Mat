from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ModelEntry:
    id: str
    kind: str  # predictor|generator
    backend: str
    task_types: List[str]
    runtime_profile: str  # cpu|gpu|remote
    notes: str = ""
    params: Dict[str, Any] = None

    @staticmethod
    def from_dict(payload: Dict[str, Any]) -> "ModelEntry":
        return ModelEntry(
            id=str(payload["id"]),
            kind=str(payload["kind"]),
            backend=str(payload.get("backend") or ""),
            task_types=list(payload.get("task_types") or []),
            runtime_profile=str(payload.get("runtime_profile") or "cpu"),
            notes=str(payload.get("notes") or ""),
            params=dict(payload.get("params") or {}),
        )

    def adapter_cmd(self, tool_name: str) -> str:
        params = self.params or {}
        if not isinstance(params, dict):
            return ""

        def _extract(mapping: Any) -> str:
            if not isinstance(mapping, dict):
                return ""
            for key in (
                f"{tool_name}_cmd",
                tool_name,
                "cmd",
            ):
                value = mapping.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
            return ""

        cmd = _extract(params.get("adapters"))
        if cmd:
            return cmd

        cmd = _extract(params)
        if cmd:
            return cmd

        if isinstance(params.get("adapter_cmd"), str) and str(params.get("adapter_cmd")).strip():
            return str(params.get("adapter_cmd")).strip()

        return ""


class ModelCatalog:
    def __init__(self, entries: List[ModelEntry]):
        self.entries = entries
        self._index = {e.id: e for e in entries}

    @staticmethod
    def load(path: Path) -> "ModelCatalog":
        payload = json.loads(path.read_text(encoding="utf-8"))
        entries = [ModelEntry.from_dict(item) for item in payload.get("models", [])]
        return ModelCatalog(entries)

    def list(self, kind: Optional[str] = None) -> List[ModelEntry]:
        if kind is None:
            return list(self.entries)
        return [e for e in self.entries if e.kind == kind]

    def get(self, model_id: str) -> Optional[ModelEntry]:
        return self._index.get(model_id)

    def validate_pair(self, predictor_id: str, generator_id: str) -> List[str]:
        errors: List[str] = []
        predictor = self.get(predictor_id)
        generator = self.get(generator_id)
        if predictor is None:
            errors.append(f"Unknown predictor_id: {predictor_id}")
        elif predictor.kind != "predictor":
            errors.append(f"Model {predictor_id} is not a predictor")

        if generator is None:
            errors.append(f"Unknown generator_id: {generator_id}")
        elif generator.kind != "generator":
            errors.append(f"Model {generator_id} is not a generator")

        return errors

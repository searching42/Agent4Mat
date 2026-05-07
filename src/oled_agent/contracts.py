from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class StageSpec:
    name: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineConfig:
    run_tag: str
    description: str = ""
    input_csv: str = ""
    output_root: str = "runs"
    stages: List[StageSpec] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_dict(payload: Dict[str, Any]) -> "PipelineConfig":
        stages_payload = payload.get("stages") or []
        stages = [
            StageSpec(name=str(s["name"]), params=dict(s.get("params") or {}))
            for s in stages_payload
        ]
        return PipelineConfig(
            run_tag=str(payload["run_tag"]),
            description=str(payload.get("description") or ""),
            input_csv=str(payload["input_csv"]),
            output_root=str(payload.get("output_root") or "runs"),
            stages=stages,
            metadata=dict(payload.get("metadata") or {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_tag": self.run_tag,
            "description": self.description,
            "input_csv": self.input_csv,
            "output_root": self.output_root,
            "stages": [asdict(s) for s in self.stages],
            "metadata": self.metadata,
        }


@dataclass
class StageExecutionRecord:
    name: str
    started_at: str
    ended_at: str
    status: str
    output_file: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""


@dataclass
class RunContext:
    workspace_root: Path
    run_root: Path
    run_tag: str
    input_csv: Path
    metadata: Dict[str, Any]


@dataclass
class StageResult:
    output_file: Optional[Path] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""


@dataclass
class RunManifest:
    run_id: str
    run_tag: str
    created_at: str
    workspace_root: str
    run_root: str
    input_csv: str
    config_snapshot: Dict[str, Any]
    stage_records: List[Dict[str, Any]]
    final_output: Optional[str] = None
    status: str = "success"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

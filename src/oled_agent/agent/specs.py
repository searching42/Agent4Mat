from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PropertyTarget:
    name: str
    objective: str  # maximize|minimize|target_window
    target_center: Optional[float] = None
    sigma: Optional[float] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    weight: float = 1.0


@dataclass
class ConstraintSpec:
    mw_min: Optional[float] = None
    mw_max: Optional[float] = None
    domain_threshold: Optional[float] = None
    banned_alerts: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BudgetSpec:
    max_wallclock_hours: float = 4.0
    max_gpu_hours: float = 2.0
    max_candidates: int = 500
    timeout_sec: Optional[int] = None
    max_tool_calls: Optional[int] = None
    max_external_calls: Optional[int] = None
    on_limit: str = "fail"


@dataclass
class ModelChoice:
    predictor_id: str
    generator_id: str


@dataclass
class DesignSpec:
    task_id: str
    user_request: str
    domain: str = "oled_molecule_design"
    targets: List[PropertyTarget] = field(default_factory=list)
    constraints: ConstraintSpec = field(default_factory=ConstraintSpec)
    model_choice: ModelChoice = field(default_factory=lambda: ModelChoice("", ""))
    budget: BudgetSpec = field(default_factory=BudgetSpec)
    mode: str = "fast_screen"  # fast_screen|train_then_design
    dataset_preferences: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ToolCall:
    name: str
    args: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentPlan:
    summary: str
    design_spec: DesignSpec
    tool_calls: List[ToolCall]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": self.summary,
            "design_spec": self.design_spec.to_dict(),
            "tool_calls": [asdict(tc) for tc in self.tool_calls],
        }


@dataclass
class TaskV2:
    version: str
    task_id: str
    request_text: str
    execution_mode: str
    operation: str
    property: str
    range: str
    n_structures: int
    constraints: Dict[str, Any]
    train_data: Optional[str] = None
    candidate_data: Optional[str] = None
    prediction_model: str = ""
    model_preferences: Dict[str, Any] = field(default_factory=dict)
    generation_input: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)
    status: str = "draft"
    missing_fields: List[str] = field(default_factory=list)
    questions: List[str] = field(default_factory=list)
    compatibility_warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

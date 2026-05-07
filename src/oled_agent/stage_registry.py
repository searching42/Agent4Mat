from __future__ import annotations

from oled_agent.stages.base import Stage
from oled_agent.stages.compose_multi_objective import ComposeMultiObjectiveStage
from oled_agent.stages.filter_multi_objective import FilterMultiObjectiveStage
from oled_agent.stages.export_simple_report import ExportSimpleReportStage


def build_stage_registry() -> dict[str, Stage]:
    stages = [
        ComposeMultiObjectiveStage(),
        FilterMultiObjectiveStage(),
        ExportSimpleReportStage(),
    ]
    return {stage.name: stage for stage in stages}

from __future__ import annotations

from abc import ABC, abstractmethod

from oled_agent.contracts import RunContext, StageResult


class Stage(ABC):
    name: str

    @abstractmethod
    def run(self, ctx: RunContext, input_file: str, params: dict) -> StageResult:
        raise NotImplementedError

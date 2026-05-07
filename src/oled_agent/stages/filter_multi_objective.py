from __future__ import annotations

from pathlib import Path

from oled_agent.contracts import RunContext, StageResult
from oled_agent.stages.base import Stage
from oled_agent.stages.io_utils import load_rows, write_rows


class FilterMultiObjectiveStage(Stage):
    name = "filter_multi_objective"

    def run(self, ctx: RunContext, input_file: str, params: dict) -> StageResult:
        rows = load_rows(Path(input_file))

        min_score = float(params.get("min_multi_total_score", 0.0))
        topn = int(params.get("topn", 10))

        passed = []
        for row in rows:
            try:
                score = float(row.get("multi_total_score") or "0")
            except Exception:
                score = 0.0
            if score >= min_score:
                row["multi_filter_pass"] = "1"
                row["multi_filter_reason"] = ""
                passed.append(row)
            else:
                row["multi_filter_pass"] = "0"
                row["multi_filter_reason"] = f"multi_total_score_lt_{min_score:.4f}"

        passed.sort(key=lambda r: float(r.get("multi_total_score") or "0"), reverse=True)
        top = []
        for i, row in enumerate(passed[:topn], start=1):
            row["multi_topn_rank"] = str(i)
            row["multi_selection_reason"] = "rank_by_multi_total_score"
            top.append(row)

        filtered_path = ctx.run_root / "04_filtered_multi_objective.csv"
        topn_path = ctx.run_root / "05_topn_multi_objective.csv"
        write_rows(filtered_path, rows)
        write_rows(topn_path, top)

        return StageResult(
            output_file=topn_path,
            metrics={
                "input_rows": len(rows),
                "passed_rows": len(passed),
                "topn_rows": len(top),
                "min_multi_total_score": min_score,
            },
            notes=f"Filtered and selected top {topn}",
        )

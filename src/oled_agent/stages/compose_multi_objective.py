from __future__ import annotations

from pathlib import Path

from oled_agent.contracts import RunContext, StageResult
from oled_agent.stages.base import Stage
from oled_agent.stages.io_utils import load_rows, write_rows


class ComposeMultiObjectiveStage(Stage):
    name = "compose_multi_objective"

    def run(self, ctx: RunContext, input_file: str, params: dict) -> StageResult:
        rows = load_rows(Path(input_file))
        objectives = params.get("objectives", [])
        common = params.get("common", {})

        for row in rows:
            total = 0.0
            for obj in objectives:
                prop = obj["property_name"]
                weight = float(obj.get("weight", 0.0))
                score_field = f"{prop}_score"
                try:
                    score = float(row.get(score_field) or "0")
                except Exception:
                    score = 0.0
                row[f"{prop}_weight"] = f"{weight:.6f}"
                total += weight * score

            try:
                domain_score = float(row.get("domain_score") or "0")
            except Exception:
                domain_score = 0.0
            try:
                prior_score = float(row.get("common_prior_score") or "0")
            except Exception:
                prior_score = 0.0

            total += float(common.get("domain_weight", 0.0)) * domain_score
            total += float(common.get("prior_weight", 0.0)) * prior_score
            total += float(common.get("diversity_weight", 0.0))
            row["multi_total_score"] = f"{total:.6f}"

        rows.sort(key=lambda r: float(r.get("multi_total_score") or "0"), reverse=True)

        out = ctx.run_root / "03_composed_multi_objective.csv"
        write_rows(out, rows)
        return StageResult(
            output_file=out,
            metrics={"rows": len(rows)},
            notes="Composed weighted multi-objective scores",
        )

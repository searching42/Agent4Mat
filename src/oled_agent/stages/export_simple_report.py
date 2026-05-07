from __future__ import annotations

from pathlib import Path

from oled_agent.contracts import RunContext, StageResult
from oled_agent.stages.base import Stage
from oled_agent.stages.io_utils import load_rows


class ExportSimpleReportStage(Stage):
    name = "export_simple_report"

    def run(self, ctx: RunContext, input_file: str, params: dict) -> StageResult:
        rows = load_rows(Path(input_file))
        report_path = ctx.run_root / "06_report.md"
        topn = int(params.get("topn", len(rows)))

        lines = [
            f"# Run report: {ctx.run_tag}",
            "",
            f"- input_csv: `{ctx.input_csv}`",
            f"- final_rows: {len(rows)}",
            "",
            "## Top candidates",
            "",
            "| Rank | candidate_id | smiles | multi_total_score |",
            "| --- | --- | --- | ---: |",
        ]

        for i, row in enumerate(rows[:topn], start=1):
            lines.append(
                f"| {i} | {row.get('candidate_id', '-')} | {row.get('smiles', '-')} | {row.get('multi_total_score', '-')} |"
            )

        report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return StageResult(
            output_file=report_path,
            metrics={"report_rows": min(topn, len(rows))},
            notes="Generated markdown summary report",
        )

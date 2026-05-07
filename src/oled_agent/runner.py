from __future__ import annotations

import json
import traceback
from dataclasses import asdict
from pathlib import Path

from oled_agent.contracts import (
    PipelineConfig,
    RunContext,
    RunManifest,
    StageExecutionRecord,
    utc_now_iso,
)
from oled_agent.stage_registry import build_stage_registry


def _workspace_rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except Exception:
        return str(path.resolve())


def load_pipeline_config(config_path: Path) -> PipelineConfig:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    return PipelineConfig.from_dict(payload)


def run_pipeline(config_path: Path, workspace_root: Path) -> Path:
    cfg = load_pipeline_config(config_path)

    run_id = f"{cfg.run_tag}_{utc_now_iso().replace(':', '').replace('-', '')}"
    output_root = (workspace_root / cfg.output_root).resolve()
    run_root = (output_root / run_id).resolve()
    run_root.mkdir(parents=True, exist_ok=True)

    input_csv = (workspace_root / cfg.input_csv).resolve()
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    ctx = RunContext(
        workspace_root=workspace_root.resolve(),
        run_root=run_root,
        run_tag=cfg.run_tag,
        input_csv=input_csv,
        metadata=dict(cfg.metadata),
    )

    registry = build_stage_registry()
    stage_records: list[StageExecutionRecord] = []
    current_file = str(input_csv)
    final_output: str | None = None
    status = "success"

    try:
        for stage_spec in cfg.stages:
            if stage_spec.name not in registry:
                raise RuntimeError(f"Unknown stage: {stage_spec.name}")

            stage = registry[stage_spec.name]
            started = utc_now_iso()
            result = stage.run(ctx, current_file, stage_spec.params)
            ended = utc_now_iso()

            if result.output_file is not None:
                current_file = str(result.output_file)
                final_output = _workspace_rel(result.output_file, workspace_root)

            stage_records.append(
                StageExecutionRecord(
                    name=stage_spec.name,
                    started_at=started,
                    ended_at=ended,
                    status="success",
                    output_file=_workspace_rel(result.output_file, workspace_root)
                    if result.output_file
                    else None,
                    metrics=result.metrics,
                    notes=result.notes,
                )
            )

    except Exception:
        status = "failed"
        stage_records.append(
            StageExecutionRecord(
                name="pipeline_exception",
                started_at=utc_now_iso(),
                ended_at=utc_now_iso(),
                status="failed",
                notes=traceback.format_exc(),
            )
        )

    manifest = RunManifest(
        run_id=run_id,
        run_tag=cfg.run_tag,
        created_at=utc_now_iso(),
        workspace_root=str(workspace_root.resolve()),
        run_root=_workspace_rel(run_root, workspace_root),
        input_csv=_workspace_rel(input_csv, workspace_root),
        config_snapshot=cfg.to_dict(),
        stage_records=[asdict(r) for r in stage_records],
        final_output=final_output,
        status=status,
    )

    manifest_path = run_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    if status != "success":
        raise RuntimeError(f"Pipeline failed. See manifest: {manifest_path}")

    return manifest_path

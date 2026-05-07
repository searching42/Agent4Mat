from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from oled_agent.runner import run_pipeline


def run_smoke(*, workspace_root: Path, config_path: Optional[Path] = None) -> Dict[str, Any]:
    cfg = config_path or (workspace_root / "configs" / "pipelines" / "demo.json")
    manifest_path = run_pipeline(config_path=cfg.resolve(), workspace_root=workspace_root.resolve())

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    status = payload.get("status", "unknown")
    final_output = payload.get("final_output")

    final_exists = False
    if isinstance(final_output, str) and final_output:
        final_path = (workspace_root / final_output).resolve()
        final_exists = final_path.exists()

    return {
        "config": str(cfg.resolve()),
        "manifest": str(manifest_path.resolve()),
        "status": status,
        "final_output": final_output,
        "final_output_exists": final_exists,
    }

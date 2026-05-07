#!/usr/bin/env python3
"""MinerU generate_candidates adapter skeleton with explicit preflight errors.

This adapter is intentionally strict/fail-fast until the project wires
an agreed MinerU generation command contract.
"""
from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Dict, List

from runtime_helpers import AdapterFailure, emit_failure, emit_success, read_payload, to_int


def _write_smoke_output(output_csv: Path, rows: int) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    seed: List[Dict[str, str]] = []
    for i in range(max(1, rows)):
        seed.append(
            {
                "candidate_id": f"cand_{i+1:06d}",
                "smiles": "c1ccc2ccccc2c1",
                "source": "mineru_adapter_smoke",
            }
        )
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(seed[0].keys()))
        writer.writeheader()
        writer.writerows(seed)


def main() -> int:
    try:
        payload = read_payload()
        output_csv_raw = str(payload.get("output_csv") or "").strip()
        if not output_csv_raw:
            raise AdapterFailure(code="missing_output_csv", message="output_csv is required")
        output_csv = Path(output_csv_raw)
        if not output_csv.is_absolute():
            workspace_root = Path(str(payload.get("workspace_root") or ".")).resolve()
            output_csv = (workspace_root / output_csv).resolve()

        smoke_mode = (os.environ.get("OLED_AGENT_MINERU_ADAPTER_MODE") or "preflight").strip().lower()
        max_candidates = to_int(
            payload.get("max_candidates"),
            default=5,
            min_value=1,
            name="max_candidates",
        )

        # Default behavior: explicit preflight failure with actionable codes.
        if smoke_mode not in ("smoke", "preflight"):
            raise AdapterFailure(
                code="invalid_env_config",
                message="OLED_AGENT_MINERU_ADAPTER_MODE must be preflight or smoke",
                details={"value": smoke_mode},
            )
        if smoke_mode == "preflight":
            raise AdapterFailure(
                code="mineru_not_configured",
                message=(
                    "MinerU adapter not configured for real generation. "
                    "Set OLED_AGENT_MINERU_ADAPTER_MODE=smoke for dry-run output, "
                    "or wire a project-specific MinerU command."
                ),
                details={"mode": smoke_mode},
            )

        _write_smoke_output(output_csv, rows=max_candidates)
        return emit_success(
            {
                "adapter": "mineru_generate_adapter_v1",
                "output_csv": str(output_csv),
                "rows": max_candidates,
                "mode": smoke_mode,
            }
        )
    except AdapterFailure as exc:
        return emit_failure(code=exc.code, message=exc.message, details=exc.details)
    except Exception as exc:
        return emit_failure(code="unexpected_adapter_error", message=str(exc))


if __name__ == "__main__":
    raise SystemExit(main())

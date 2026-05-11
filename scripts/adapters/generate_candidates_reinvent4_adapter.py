#!/usr/bin/env python3
"""REINVENT4-backed generate_candidates adapter.

Modes (env: OLED_AGENT_REINVENT4_ADAPTER_MODE):
- preflight: fail fast with actionable config error
- smoke: deterministic local CSV output
- real: run external REINVENT4 pipeline, normalize rankready CSV to adapter output
"""
from __future__ import annotations

import csv
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

from runtime_helpers import AdapterFailure, emit_failure, emit_success, read_payload, run_argv_cmd, to_int


_SMILES_KEYS = ["smiles", "SMILES", "Smiles", "canonical_smiles", "CANONICAL_SMILES", "sampled_smiles", "molecule"]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _external_workspace_root() -> Path:
    # current repo: <workspace>/oled-agent
    # external scripts/data live at: <workspace>/scripts and <workspace>/artifacts
    return _repo_root().parent


def _resolve_path(path_like: str, *, workspace_root: Path) -> Path:
    p = Path(path_like)
    if p.is_absolute():
        return p
    return (workspace_root / p).resolve()


def _pick_latest_sampling_csv(ext_root: Path) -> Optional[Path]:
    run_dir = ext_root / "artifacts" / "server_sync" / "reinvent4_runs"
    if not run_dir.exists():
        return None
    files = sorted(
        run_dir.glob("openclaw_sampling_project_v1_*.csv"),
        key=lambda x: x.stat().st_mtime,
        reverse=True,
    )
    return files[0] if files else None


def _extract_smiles(row: Dict[str, str]) -> str:
    for key in _SMILES_KEYS:
        value = (row.get(key) or "").strip()
        if value:
            return value
    return ""


def _load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _normalize_rows(rows: List[Dict[str, str]], *, source_label: str, max_rows: int) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    seen = set()
    for i, row in enumerate(rows, start=1):
        smiles = _extract_smiles(row)
        if not smiles:
            continue
        if smiles in seen:
            continue
        seen.add(smiles)

        cid = (row.get("candidate_id") or "").strip()
        if not cid:
            cid = f"cand_{len(out)+1:06d}"

        out.append(
            {
                "candidate_id": cid,
                "smiles": smiles,
                "source": source_label,
                "source_row": str(i),
            }
        )
        if len(out) >= max_rows:
            break
    return out


def _write_rows(path: Path, rows: List[Dict[str, str]]) -> None:
    if not rows:
        raise AdapterFailure(code="empty_candidate_set", message="no candidate rows after normalization")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_smoke_output(output_csv: Path, rows: int) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    seed = [
        "c1ccc2ccccc2c1",
        "c1ncccc1",
        "CCOC(=O)N1CCN(CC1)C",
        "CN1C=NC2=CC=CC=C21",
        "c1ccc(cc1)N(c2ccccc2)c3ccccc3",
    ]
    out: List[Dict[str, str]] = []
    for i in range(max(1, rows)):
        out.append(
            {
                "candidate_id": f"cand_{i+1:06d}",
                "smiles": seed[i % len(seed)],
                "source": "reinvent4_adapter_smoke",
                "source_row": str(i + 1),
            }
        )
    _write_rows(output_csv, out)


def _resolve_source_csv(payload: Dict[str, object], *, workspace_root: Path, ext_root: Path) -> Optional[Path]:
    payload_input = str(payload.get("input_csv") or "").strip()
    if payload_input:
        p = _resolve_path(payload_input, workspace_root=workspace_root)
        if p.exists():
            return p

    env_input = (os.environ.get("OLED_AGENT_REINVENT4_SOURCE_CSV") or "").strip()
    if env_input:
        p = Path(env_input)
        if not p.is_absolute():
            p = (ext_root / p).resolve()
        if p.exists():
            return p

    return _pick_latest_sampling_csv(ext_root)


def _run_reinvent4_pipeline(*, ext_root: Path, source_csv: Path, task_id: str, timeout_sec: int) -> Path:
    pipeline_script = (os.environ.get("OLED_AGENT_REINVENT4_PIPELINE_SCRIPT") or "").strip()
    if pipeline_script:
        script = Path(pipeline_script)
        if not script.is_absolute():
            script = (ext_root / script).resolve()
    else:
        script = (ext_root / "scripts" / "run_reinvent4_lambda_em_v2_pipeline.sh").resolve()

    if not script.exists():
        raise AdapterFailure(
            code="reinvent4_pipeline_missing",
            message="REINVENT4 pipeline script not found",
            details={"script": str(script)},
        )

    run_tag = f"agent4mat_{task_id}_{int(time.time())}"
    run_argv_cmd(
        argv=["bash", str(script), str(source_csv), run_tag],
        cwd=ext_root,
        timeout_sec=timeout_sec,
    )

    rankready_env = (os.environ.get("OLED_AGENT_REINVENT4_RANKREADY_CSV") or "").strip()
    if rankready_env:
        rankready = Path(rankready_env)
        if not rankready.is_absolute():
            rankready = (ext_root / rankready).resolve()
    else:
        rankready = (ext_root / "reports" / "end2end" / f"{run_tag}_rankready.csv").resolve()

    if not rankready.exists():
        raise AdapterFailure(
            code="reinvent4_rankready_missing",
            message="REINVENT4 pipeline finished but rankready CSV is missing",
            details={
                "run_tag": run_tag,
                "expected_rankready_csv": str(rankready),
            },
        )
    return rankready


def main() -> int:
    try:
        payload = read_payload()
        workspace_root = Path(str(payload.get("workspace_root") or ".")).resolve()
        ext_root = _external_workspace_root()

        output_csv_raw = str(payload.get("output_csv") or "").strip()
        if not output_csv_raw:
            raise AdapterFailure(code="missing_output_csv", message="output_csv is required")
        output_csv = _resolve_path(output_csv_raw, workspace_root=workspace_root)

        max_candidates = to_int(
            payload.get("max_candidates"),
            default=50,
            min_value=1,
            name="max_candidates",
        )
        mode = (os.environ.get("OLED_AGENT_REINVENT4_ADAPTER_MODE") or "preflight").strip().lower()
        if mode not in ("preflight", "smoke", "real"):
            raise AdapterFailure(
                code="invalid_env_config",
                message="OLED_AGENT_REINVENT4_ADAPTER_MODE must be preflight|smoke|real",
                details={"value": mode},
            )

        if mode == "smoke":
            _write_smoke_output(output_csv, rows=max_candidates)
            return emit_success(
                {
                    "adapter": "reinvent4_generate_adapter_v1",
                    "output_csv": str(output_csv),
                    "rows": max_candidates,
                    "mode": mode,
                }
            )

        source_csv = _resolve_source_csv(payload, workspace_root=workspace_root, ext_root=ext_root)
        pipeline_default = (ext_root / "scripts" / "run_reinvent4_lambda_em_v2_pipeline.sh").resolve()
        if mode == "preflight":
            if source_csv is None:
                raise AdapterFailure(
                    code="reinvent4_source_missing",
                    message=(
                        "No REINVENT4 source CSV found. Provide payload input_csv, "
                        "set OLED_AGENT_REINVENT4_SOURCE_CSV, or ensure artifacts/server_sync/reinvent4_runs exists."
                    ),
                )
            if not pipeline_default.exists() and not (os.environ.get("OLED_AGENT_REINVENT4_PIPELINE_SCRIPT") or "").strip():
                raise AdapterFailure(
                    code="reinvent4_pipeline_missing",
                    message="REINVENT4 pipeline script is not configured",
                    details={"expected": str(pipeline_default)},
                )
            raise AdapterFailure(
                code="reinvent4_not_enabled",
                message=(
                    "REINVENT4 adapter preflight is healthy but real execution is disabled. "
                    "Set OLED_AGENT_REINVENT4_ADAPTER_MODE=real to execute pipeline."
                ),
                details={"source_csv": str(source_csv)},
            )

        if source_csv is None:
            raise AdapterFailure(
                code="reinvent4_source_missing",
                message="real mode requires a source CSV (input_csv/env/latest artifact)",
            )

        timeout_sec = to_int(
            os.environ.get("OLED_AGENT_REINVENT4_ADAPTER_TIMEOUT_SEC", ""),
            default=7200,
            min_value=1,
            name="OLED_AGENT_REINVENT4_ADAPTER_TIMEOUT_SEC",
        )
        rankready = _run_reinvent4_pipeline(
            ext_root=ext_root,
            source_csv=source_csv,
            task_id=str(payload.get("task_id") or "task"),
            timeout_sec=timeout_sec,
        )

        normalized = _normalize_rows(
            _load_rows(rankready),
            source_label="reinvent4_adapter_real",
            max_rows=max_candidates,
        )
        _write_rows(output_csv, normalized)

        return emit_success(
            {
                "adapter": "reinvent4_generate_adapter_v1",
                "output_csv": str(output_csv),
                "rows": len(normalized),
                "mode": mode,
                "source_csv": str(source_csv),
                "rankready_csv": str(rankready),
            }
        )
    except AdapterFailure as exc:
        return emit_failure(code=exc.code, message=exc.message, details=exc.details)
    except Exception as exc:
        return emit_failure(code="unexpected_adapter_error", message=str(exc))


if __name__ == "__main__":
    raise SystemExit(main())

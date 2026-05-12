#!/usr/bin/env python3
"""Uni-Mol score_candidates adapter (structured shell around workspace scorer script)."""
from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any, Dict, List

from runtime_helpers import (
    AdapterFailure,
    emit_failure,
    emit_success,
    read_payload,
    repo_root_from_script,
    resolve_path,
    run_argv_cmd,
    to_int,
)


def _load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _normalize_smiles_and_candidate_id(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for i, row in enumerate(rows):
        r = dict(row)
        smiles = ""
        for key in ("smiles", "SMILES", "Smiles", "canonical_smiles", "CANONICAL_SMILES"):
            value = (r.get(key) or "").strip()
            if value:
                smiles = value
                break
        if not smiles:
            raise AdapterFailure(
                code="missing_smiles_column",
                message=f"missing smiles/SMILES in input row {i + 1}",
                details={"row_index": i + 1},
            )
        r["smiles"] = smiles

        cid = (r.get("candidate_id") or "").strip()
        if not cid:
            cid = f"cand_{i+1:06d}"
        r["candidate_id"] = cid
        out.append(r)
    return out


def _write_rows(path: Path, rows: List[Dict[str, str]]) -> None:
    if not rows:
        raise AdapterFailure(code="empty_candidate_set", message="no rows to score")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _model_dir_for_property(prop: str, explicit_model_dir: str) -> str:
    if explicit_model_dir:
        return explicit_model_dir
    env_key = f"UNIMOL_REMOTE_MODEL_DIR_{str(prop or '').strip().upper()}"
    prop_model_dir = (os.environ.get(env_key) or "").strip()
    if prop_model_dir:
        return prop_model_dir
    fallback = (os.environ.get("UNIMOL_REMOTE_MODEL_DIR") or "").strip()
    return fallback


def _merge_csvs(base_csv: Path, addon_csv: Path, key: str = "candidate_id") -> None:
    base_rows = _load_rows(base_csv)
    addon_rows = _load_rows(addon_csv)
    addon_map = {}
    for i, row in enumerate(addon_rows, start=1):
        k = (row.get(key) or "").strip()
        if not k:
            raise AdapterFailure(
                code="missing_candidate_id",
                message=f"addon score CSV missing candidate_id at row {i}",
                details={"row_index": i, "key": key},
            )
        addon_map[k] = row

    merged: List[Dict[str, str]] = []
    for row in base_rows:
        cid = (row.get(key) or "").strip()
        if not cid:
            raise AdapterFailure(
                code="missing_candidate_id",
                message="base score CSV missing candidate_id",
                details={"key": key},
            )
        out = dict(row)
        extra = addon_map.get(cid, {})
        for k, v in extra.items():
            if k == key:
                continue
            out[k] = v
        merged.append(out)

    fieldnames: List[str] = []
    seen = set()
    for row in merged + addon_rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)

    with base_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(merged)


def main() -> int:
    try:
        payload = read_payload()
        workspace_root = Path(str(payload.get("workspace_root") or ".")).resolve()
        input_csv_raw = str(payload.get("input_csv") or "").strip()
        output_csv_raw = str(payload.get("output_csv") or "").strip()
        targets = payload.get("targets") or []
        target_specs = payload.get("target_specs") or []
        if not input_csv_raw:
            raise AdapterFailure(code="missing_input_csv", message="input_csv is required")
        if not output_csv_raw:
            raise AdapterFailure(code="missing_output_csv", message="output_csv is required")
        if not isinstance(target_specs, list):
            raise AdapterFailure(
                code="invalid_target_specs",
                message="target_specs must be an array",
                details={"json_type": type(target_specs).__name__},
            )
        mode = (os.environ.get("OLED_AGENT_UNIMOL_SCORE_MODE") or "preflight").strip().lower()
        if mode not in ("preflight", "smoke", "real"):
            raise AdapterFailure(
                code="invalid_env_config",
                message="OLED_AGENT_UNIMOL_SCORE_MODE must be preflight|smoke|real",
                details={"value": mode},
            )

        input_csv = resolve_path(input_csv_raw, workspace_root=workspace_root)
        output_csv = resolve_path(output_csv_raw, workspace_root=workspace_root)
        if not input_csv.exists():
            raise AdapterFailure(
                code="missing_input_csv",
                message="input_csv does not exist",
                details={"path": str(input_csv)},
            )

        rows = _normalize_smiles_and_candidate_id(_load_rows(input_csv))
        if not rows:
            raise AdapterFailure(code="empty_candidate_set", message="input_csv has zero rows")
        _write_rows(output_csv, rows)
        if mode in ("preflight", "smoke"):
            for row in rows:
                row.setdefault("domain_score", "0.500000")
                row.setdefault("common_prior_score", "0.500000")
                row.setdefault("plqy_pred", "0.600000")
                row.setdefault("plqy_score", "0.600000")
            _write_rows(output_csv, rows)
            return emit_success(
                {
                    "adapter": "unimol_score_adapter_v1",
                    "output_csv": str(output_csv),
                    "rows": len(rows),
                    "mode": mode,
                    "note": "preflight/smoke mode wrote deterministic stub scores",
                }
            )

        repo_root = repo_root_from_script()
        scorer_override = (os.environ.get("OLED_AGENT_UNIMOL_SCORE_SCRIPT") or "").strip()
        if scorer_override:
            scorer = resolve_path(scorer_override, workspace_root=workspace_root)
        else:
            scorer = (repo_root.parent / "scripts" / "score_unimol_property_candidates.py").resolve()
        if not scorer.exists():
            raise AdapterFailure(
                code="external_scorer_script_missing",
                message="workspace scorer script is missing",
                details={"expected": str(scorer)},
            )

        timeout_sec = to_int(
            os.environ.get("OLED_AGENT_UNIMOL_ADAPTER_TIMEOUT_SEC", ""),
            default=900,
            min_value=1,
            name="OLED_AGENT_UNIMOL_ADAPTER_TIMEOUT_SEC",
        )
        # Best-effort pass-through to scorer runtime; scorer itself enforces strict env completeness.
        env_host = (os.environ.get("UNIMOL_REMOTE_HOST") or "").strip()
        env_py = (os.environ.get("UNIMOL_REMOTE_PY") or "").strip()
        env_tmp = (os.environ.get("UNIMOL_REMOTE_TMP_BASE") or "").strip()
        allow_default = (os.environ.get("ALLOW_DEFAULT_UNIMOL_REMOTE") or "0").strip()
        if any([env_host, env_py, env_tmp]) and not all([env_host, env_py, env_tmp]):
            raise AdapterFailure(
                code="external_runtime_config_incomplete",
                message="UNIMOL_REMOTE_* must be set together",
                details={
                    "present": [k for k, v in {
                        "UNIMOL_REMOTE_HOST": env_host,
                        "UNIMOL_REMOTE_PY": env_py,
                        "UNIMOL_REMOTE_TMP_BASE": env_tmp,
                    }.items() if v],
                },
            )

        specs = [s for s in target_specs if isinstance(s, dict)]
        if not specs:
            specs = [{"name": str(t), "objective": "target_window", "target_center": 470.0, "sigma": 12.0} for t in targets]
        if not specs:
            specs = [{"name": "plqy", "objective": "maximize", "target_center": 60.0, "sigma": 20.0}]

        # Keep normalized working CSV as merge base.
        for spec in specs:
            prop = str(spec.get("name") or "").strip()
            if not prop:
                continue
            objective = str(spec.get("objective") or "target_window")
            target_center = float(spec.get("target_center") or (470.0 if prop == "lambda_em" else 50.0))
            sigma = float(spec.get("sigma") or (12.0 if prop == "lambda_em" else 20.0))
            model_dir = _model_dir_for_property(prop, str(spec.get("model_dir") or "").strip())

            per_output = output_csv.with_name(f"{output_csv.stem}_{prop}.csv")
            cmd_parts = [
                "python3",
                str(scorer),
                str(output_csv),
                str(per_output),
                "--property-name",
                prop,
                "--objective-type",
                objective,
                "--target-center",
                str(target_center),
                "--sigma",
                str(sigma),
            ]
            if model_dir:
                cmd_parts.extend(["--model-dir", model_dir])

            run_argv_cmd(
                argv=cmd_parts,
                cwd=workspace_root,
                timeout_sec=timeout_sec,
            )
            _merge_csvs(output_csv, per_output, key="candidate_id")

        return emit_success(
            {
                "adapter": "unimol_score_adapter_v1",
                "output_csv": str(output_csv),
                "rows": len(rows),
                "mode": mode,
                "remote_runtime": {
                    "configured": bool(env_host and env_py and env_tmp),
                    "allow_default": allow_default == "1",
                },
            }
        )
    except AdapterFailure as exc:
        return emit_failure(code=exc.code, message=exc.message, details=exc.details)
    except Exception as exc:
        return emit_failure(code="unexpected_adapter_error", message=str(exc))


if __name__ == "__main__":
    raise SystemExit(main())

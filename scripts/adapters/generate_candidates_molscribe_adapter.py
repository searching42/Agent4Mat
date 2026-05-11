#!/usr/bin/env python3
"""MolScribe-backed generate_candidates adapter.

This adapter formalizes image/PDF-to-SMILES ingestion into the standard
`generate_candidates` tool contract.

Modes (env: OLED_AGENT_MOLSCRIBE_ADAPTER_MODE):
- preflight: verify input pointers and runtime hooks, then fail with actionable code
- smoke: deterministic local output (contract-only)
- real: execute configured extractor command and normalize its CSV output
"""
from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Dict, List, Optional

from runtime_helpers import AdapterFailure, emit_failure, emit_success, parse_cmd, read_payload, run_argv_cmd, to_int


_SMILES_KEYS = ["smiles", "SMILES", "Smiles", "canonical_smiles", "CANONICAL_SMILES", "mol_smiles"]
_SOURCE_KEYS = [
    "source_image",
    "source_images",
    "source_pdf",
    "source_pdfs",
    "input_image",
    "input_pdf",
    "paper_path",
    "image_paths",
]
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
_PDF_SUFFIXES = {".pdf"}


def _resolve_path(path_like: str, *, workspace_root: Path) -> Path:
    p = Path(path_like)
    if p.is_absolute():
        return p
    return (workspace_root / p).resolve()


def _extract_smiles(row: Dict[str, str]) -> str:
    for key in _SMILES_KEYS:
        value = (row.get(key) or "").strip()
        if value:
            return value
    return ""


def _load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_rows(path: Path, rows: List[Dict[str, str]]) -> None:
    if not rows:
        raise AdapterFailure(code="empty_candidate_set", message="no candidate rows to write")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _normalize_rows(rows: List[Dict[str, str]], *, max_rows: int, source: str) -> List[Dict[str, str]]:
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

        out_row = dict(row)
        out_row["candidate_id"] = cid
        out_row["smiles"] = smiles
        out_row.setdefault("source", source)
        out.append(out_row)
        if len(out) >= max_rows:
            break
    return out


def _seed_rows(max_rows: int) -> List[Dict[str, str]]:
    seed = [
        "c1ccc2ccccc2c1",
        "c1ncccc1",
        "CCOC(=O)N1CCN(CC1)C",
        "CN1C=NC2=CC=CC=C21",
        "c1ccc(cc1)N(c2ccccc2)c3ccccc3",
    ]
    out: List[Dict[str, str]] = []
    for i in range(max(1, max_rows)):
        out.append(
            {
                "candidate_id": f"cand_{i+1:06d}",
                "smiles": seed[i % len(seed)],
                "source": "molscribe_adapter_smoke",
            }
        )
    return out


def _source_hints(payload: Dict[str, object], constraints: Dict[str, object]) -> List[str]:
    hints: List[str] = []
    for key in _SOURCE_KEYS:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            hints.append(value.strip())
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, str) and item.strip():
                    hints.append(item.strip())

        cval = constraints.get(key)
        if isinstance(cval, str) and cval.strip():
            hints.append(cval.strip())
        elif isinstance(cval, list):
            for item in cval:
                if isinstance(item, str) and item.strip():
                    hints.append(item.strip())

    dedup: List[str] = []
    seen = set()
    for h in hints:
        if h not in seen:
            seen.add(h)
            dedup.append(h)
    return dedup


def _resolve_existing_sources(hints: List[str], *, workspace_root: Path) -> List[Path]:
    out: List[Path] = []
    for h in hints:
        p = _resolve_path(h, workspace_root=workspace_root)
        if p.exists():
            out.append(p)
    return out


def _run_real_extractor(
    *,
    cmd: str,
    workspace_root: Path,
    output_csv: Path,
    source_files: List[Path],
    timeout_sec: int,
) -> Path:
    argv = parse_cmd(cmd)
    # Contract for external extractor:
    #   --output-csv <path>
    #   --input <path> (repeatable)
    argv = list(argv)
    argv.extend(["--output-csv", str(output_csv)])
    for src in source_files:
        argv.extend(["--input", str(src)])

    run_argv_cmd(argv=argv, cwd=workspace_root, timeout_sec=timeout_sec)

    if not output_csv.exists():
        raise AdapterFailure(
            code="molscribe_output_missing",
            message="MolScribe extractor finished but output CSV is missing",
            details={"output_csv": str(output_csv)},
        )
    return output_csv


def _run_pdf_extract_cmd(*, cmd_template: str, workspace_root: Path, pdf_files: List[Path], out_dir: Path, timeout_sec: int) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    extracted: List[Path] = []
    for idx, pdf in enumerate(pdf_files, start=1):
        cmd = cmd_template.format(input_pdf=str(pdf), output_dir=str(out_dir), index=idx)
        argv = parse_cmd(cmd)
        run_argv_cmd(argv=argv, cwd=workspace_root, timeout_sec=timeout_sec)

    for p in sorted(out_dir.glob("*")):
        if p.suffix.lower() in _IMAGE_SUFFIXES and p.is_file():
            extracted.append(p)
    return extracted


def _predict_with_native_molscribe(*, image_files: List[Path], max_candidates: int) -> List[Dict[str, str]]:
    ckpt = (os.environ.get("OLED_AGENT_MOLSCRIBE_CHECKPOINT") or "").strip()
    if not ckpt:
        raise AdapterFailure(
            code="molscribe_checkpoint_missing",
            message="OLED_AGENT_MOLSCRIBE_CHECKPOINT is required for native MolScribe mode",
        )
    ckpt_path = Path(ckpt)
    if not ckpt_path.is_absolute():
        ckpt_path = (Path.cwd() / ckpt_path).resolve()
    if not ckpt_path.exists():
        raise AdapterFailure(
            code="molscribe_checkpoint_missing",
            message="MolScribe checkpoint does not exist",
            details={"checkpoint": str(ckpt_path)},
        )

    device_name = (os.environ.get("OLED_AGENT_MOLSCRIBE_DEVICE") or "cpu").strip()
    try:
        import torch  # type: ignore
        from molscribe import MolScribe  # type: ignore
    except Exception as exc:
        raise AdapterFailure(
            code="molscribe_runtime_missing",
            message="MolScribe runtime missing; install MolScribe and torch first",
            details={"error": str(exc)},
        ) from exc

    model = MolScribe(str(ckpt_path), device=torch.device(device_name))
    rows: List[Dict[str, str]] = []
    for i, image_path in enumerate(image_files, start=1):
        try:
            pred = model.predict_image_file(
                str(image_path),
                return_atoms_bonds=False,
                return_confidence=True,
            )
        except TypeError:
            # Older interfaces use compute_confidence/get_atoms_bonds flags.
            pred = model.predict_image_file(
                str(image_path),
                compute_confidence=True,
                get_atoms_bonds=False,
            )
        if not isinstance(pred, dict):
            continue
        smiles = str(pred.get("smiles") or "").strip()
        if not smiles:
            continue
        row: Dict[str, str] = {
            "candidate_id": f"cand_{i:06d}",
            "smiles": smiles,
            "source": "molscribe_native",
            "input_image": str(image_path),
        }
        if pred.get("confidence") is not None:
            try:
                row["molscribe_confidence"] = f"{float(pred.get('confidence')):.6f}"
            except Exception:
                row["molscribe_confidence"] = str(pred.get("confidence"))
        rows.append(row)
        if len(rows) >= max_candidates:
            break
    return rows


def main() -> int:
    try:
        payload = read_payload()
        workspace_root = Path(str(payload.get("workspace_root") or ".")).resolve()
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
        mode = (os.environ.get("OLED_AGENT_MOLSCRIBE_ADAPTER_MODE") or "preflight").strip().lower()
        if mode not in ("preflight", "smoke", "real"):
            raise AdapterFailure(
                code="invalid_env_config",
                message="OLED_AGENT_MOLSCRIBE_ADAPTER_MODE must be preflight|smoke|real",
                details={"value": mode},
            )

        constraints = payload.get("constraints") if isinstance(payload.get("constraints"), dict) else {}
        source_hints = _source_hints(payload, constraints)
        source_files = _resolve_existing_sources(source_hints, workspace_root=workspace_root)

        if mode == "smoke":
            rows = _seed_rows(max_candidates)
            _write_rows(output_csv, rows)
            return emit_success(
                {
                    "adapter": "molscribe_generate_adapter_v1",
                    "output_csv": str(output_csv),
                    "rows": len(rows),
                    "mode": mode,
                }
            )

        extractor_cmd = (os.environ.get("OLED_AGENT_MOLSCRIBE_CMD") or "").strip()
        timeout_sec = to_int(
            os.environ.get("OLED_AGENT_MOLSCRIBE_ADAPTER_TIMEOUT_SEC", ""),
            default=1800,
            min_value=1,
            name="OLED_AGENT_MOLSCRIBE_ADAPTER_TIMEOUT_SEC",
        )

        images = [p for p in source_files if p.suffix.lower() in _IMAGE_SUFFIXES]
        pdfs = [p for p in source_files if p.suffix.lower() in _PDF_SUFFIXES]

        if pdfs:
            pdf_cmd = (os.environ.get("OLED_AGENT_MOLSCRIBE_PDF_EXTRACT_CMD") or "").strip()
            if not pdf_cmd and mode == "real" and not extractor_cmd:
                raise AdapterFailure(
                    code="molscribe_pdf_not_supported",
                    message=(
                        "PDF input requires OLED_AGENT_MOLSCRIBE_PDF_EXTRACT_CMD "
                        "or OLED_AGENT_MOLSCRIBE_CMD."
                    ),
                    details={"pdf_inputs": [str(p) for p in pdfs]},
                )
            if pdf_cmd:
                tmp_dir = output_csv.parent / f"{output_csv.stem}_molscribe_images"
                extracted = _run_pdf_extract_cmd(
                    cmd_template=pdf_cmd,
                    workspace_root=workspace_root,
                    pdf_files=pdfs,
                    out_dir=tmp_dir,
                    timeout_sec=timeout_sec,
                )
                images.extend(extracted)
        if mode == "preflight":
            if not source_hints:
                raise AdapterFailure(
                    code="molscribe_input_missing",
                    message=(
                        "MolScribe input is missing. Provide image/pdf path via "
                        "payload or constraints (source_image/source_pdf/image_paths)."
                    ),
                )
            if not source_files:
                raise AdapterFailure(
                    code="molscribe_input_not_found",
                    message="MolScribe input path(s) do not exist",
                    details={"hints": source_hints},
                )
            if not extractor_cmd and not images:
                raise AdapterFailure(
                    code="molscribe_input_not_found",
                    message="no usable image inputs found for MolScribe",
                    details={"inputs": [str(p) for p in source_files]},
                )
            if not extractor_cmd and not (os.environ.get("OLED_AGENT_MOLSCRIBE_CHECKPOINT") or "").strip():
                raise AdapterFailure(
                    code="molscribe_not_configured",
                    message=(
                        "Set OLED_AGENT_MOLSCRIBE_CMD, or configure native mode with "
                        "OLED_AGENT_MOLSCRIBE_CHECKPOINT."
                    ),
                )
            raise AdapterFailure(
                code="molscribe_not_enabled",
                message=(
                    "MolScribe preflight passed, real execution is disabled. "
                    "Set OLED_AGENT_MOLSCRIBE_ADAPTER_MODE=real to run extractor."
                ),
                details={"inputs": [str(p) for p in source_files]},
            )

        if not source_files:
            raise AdapterFailure(
                code="molscribe_input_not_found",
                message="real mode requires existing source image/pdf path(s)",
                details={"hints": source_hints},
            )
        if extractor_cmd:
            raw_output = _run_real_extractor(
                cmd=extractor_cmd,
                workspace_root=workspace_root,
                output_csv=output_csv,
                source_files=source_files,
                timeout_sec=timeout_sec,
            )
            normalized = _normalize_rows(
                _load_rows(raw_output),
                max_rows=max_candidates,
                source="molscribe_adapter_real",
            )
            _write_rows(output_csv, normalized)
        else:
            if not images:
                raise AdapterFailure(
                    code="molscribe_input_not_found",
                    message="real mode native MolScribe requires at least one image input",
                    details={"inputs": [str(p) for p in source_files]},
                )
            native_rows = _predict_with_native_molscribe(image_files=images, max_candidates=max_candidates)
            if not native_rows:
                raise AdapterFailure(
                    code="molscribe_empty_output",
                    message="MolScribe produced no valid smiles from provided images",
                    details={"images": [str(p) for p in images]},
                )
            normalized = _normalize_rows(native_rows, max_rows=max_candidates, source="molscribe_native")
            _write_rows(output_csv, normalized)

        return emit_success(
            {
                "adapter": "molscribe_generate_adapter_v1",
                "output_csv": str(output_csv),
                "rows": len(normalized),
                "mode": mode,
                "inputs": [str(p) for p in source_files],
            }
        )
    except AdapterFailure as exc:
        return emit_failure(code=exc.code, message=exc.message, details=exc.details)
    except Exception as exc:
        return emit_failure(code="unexpected_adapter_error", message=str(exc))


if __name__ == "__main__":
    raise SystemExit(main())

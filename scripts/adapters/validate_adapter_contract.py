#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple


class ContractValidationError(RuntimeError):
    def __init__(self, *, code: str, message: str, details: Dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.details = details or {}


def _split_cmd(cmd: str) -> List[str]:
    try:
        argv = shlex.split(cmd)
    except ValueError as exc:
        raise ContractValidationError(
            code="invalid_cmd",
            message=f"failed to parse command: {exc}",
            details={"cmd": cmd},
        ) from exc
    if not argv:
        raise ContractValidationError(code="invalid_cmd", message="empty command", details={"cmd": cmd})
    return argv


def _run_json_adapter(
    *,
    cmd: str,
    payload: Dict[str, Any],
    workspace_root: Path,
    timeout_sec: int,
) -> Dict[str, Any]:
    argv = _split_cmd(cmd)
    try:
        cp = subprocess.run(
            argv,
            input=json.dumps(payload, ensure_ascii=False),
            cwd=str(workspace_root),
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired as exc:
        raise ContractValidationError(
            code="adapter_timeout",
            message=f"adapter timed out after {timeout_sec}s",
            details={"cmd": cmd},
        ) from exc

    if cp.returncode != 0:
        raise ContractValidationError(
            code="adapter_nonzero_exit",
            message=f"adapter exited with non-zero status: {cp.returncode}",
            details={
                "cmd": cmd,
                "returncode": cp.returncode,
                "stderr_tail": (cp.stderr or "")[-1000:],
                "stdout_tail": (cp.stdout or "")[-1000:],
            },
        )

    raw = (cp.stdout or "").strip()
    if not raw:
        raise ContractValidationError(
            code="empty_stdout",
            message="adapter returned empty stdout; expected one JSON object",
            details={"cmd": cmd},
        )
    try:
        out = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ContractValidationError(
            code="invalid_json_stdout",
            message=f"adapter stdout is not valid JSON: {exc}",
            details={"cmd": cmd, "stdout_tail": raw[-1000:]},
        ) from exc
    if not isinstance(out, dict):
        raise ContractValidationError(
            code="invalid_json_type",
            message="adapter stdout JSON must be an object",
            details={"cmd": cmd, "json_type": type(out).__name__},
        )
    return out


def _resolve_output_csv(result: Dict[str, Any], payload: Dict[str, Any], workspace_root: Path) -> Path:
    output = str(result.get("output_csv") or result.get("output") or payload.get("output_csv") or "").strip()
    if not output:
        raise ContractValidationError(
            code="missing_output_csv",
            message="adapter result missing output_csv/output",
            details={"result_keys": sorted(result.keys())},
        )
    path = Path(output)
    if not path.is_absolute():
        path = (workspace_root / path).resolve()
    return path


def _load_csv(path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    if not path.exists():
        raise ContractValidationError(
            code="missing_output_file",
            message=f"output CSV not found: {path}",
            details={"path": str(path)},
        )
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    if not fieldnames:
        raise ContractValidationError(
            code="empty_csv_header",
            message=f"output CSV missing header: {path}",
            details={"path": str(path)},
        )
    return rows, fieldnames


def _require_status_string(result: Dict[str, Any]) -> None:
    status = result.get("status")
    if not isinstance(status, str) or not status.strip():
        raise ContractValidationError(
            code="missing_status",
            message="adapter result must include non-empty string status",
            details={"result_keys": sorted(result.keys())},
        )


def _validate_train(*, cmd: str, workspace_root: Path, timeout_sec: int) -> Dict[str, Any]:
    payload = {
        "workspace_root": str(workspace_root),
        "task_id": "contract_train",
        "predictor_id": "pred_contract_v1",
        "targets": ["plqy"],
        "target_specs": [{"name": "plqy", "objective": "maximize"}],
        "state": {},
    }
    result = _run_json_adapter(cmd=cmd, payload=payload, workspace_root=workspace_root, timeout_sec=timeout_sec)
    _require_status_string(result)
    return {
        "tool": "train_predictor",
        "status": "pass",
        "checks": [
            "stdout is JSON object",
            "status field is non-empty string",
        ],
        "result_preview": {k: result.get(k) for k in ("status", "adapter", "predictor_id")},
    }


def _validate_generate(*, cmd: str, workspace_root: Path, timeout_sec: int) -> Dict[str, Any]:
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        output_csv = td_path / "generated.csv"
        payload = {
            "workspace_root": str(workspace_root),
            "task_id": "contract_generate",
            "generator_id": "gen_contract_v1",
            "max_candidates": 5,
            "constraints": {"mw_max": 700},
            "output_csv": str(output_csv),
            "state": {},
        }
        result = _run_json_adapter(cmd=cmd, payload=payload, workspace_root=workspace_root, timeout_sec=timeout_sec)
        _require_status_string(result)

        produced = _resolve_output_csv(result, payload, workspace_root)
        rows, fieldnames = _load_csv(produced)
        if not rows:
            raise ContractValidationError(
                code="empty_output_rows",
                message="generated CSV contains zero rows",
                details={"output_csv": str(produced)},
            )
        if "smiles" not in fieldnames and "SMILES" not in fieldnames:
            raise ContractValidationError(
                code="missing_smiles_column",
                message="generated CSV must include smiles or SMILES column",
                details={"fieldnames": fieldnames},
            )

        return {
            "tool": "generate_candidates",
            "status": "pass",
            "checks": [
                "stdout is JSON object",
                "status field is non-empty string",
                "output CSV exists",
                "output CSV has rows",
                "output CSV includes smiles/SMILES",
            ],
            "result_preview": {
                "status": result.get("status"),
                "adapter": result.get("adapter"),
                "output_csv": str(produced),
                "rows": len(rows),
            },
        }


def _validate_score(*, cmd: str, workspace_root: Path, timeout_sec: int) -> Dict[str, Any]:
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        input_csv = td_path / "input.csv"
        output_csv = td_path / "scored.csv"

        with input_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["candidate_id", "smiles"])
            writer.writeheader()
            writer.writerow({"candidate_id": "cand_000001", "smiles": "c1ccccc1"})

        payload = {
            "workspace_root": str(workspace_root),
            "task_id": "contract_score",
            "predictor_id": "pred_contract_v1",
            "targets": ["plqy"],
            "target_specs": [{"name": "plqy", "objective": "maximize", "target_center": 0.6, "sigma": 0.2}],
            "input_csv": str(input_csv),
            "output_csv": str(output_csv),
            "state": {},
        }
        result = _run_json_adapter(cmd=cmd, payload=payload, workspace_root=workspace_root, timeout_sec=timeout_sec)
        _require_status_string(result)

        produced = _resolve_output_csv(result, payload, workspace_root)
        rows, fieldnames = _load_csv(produced)
        if not rows:
            raise ContractValidationError(
                code="empty_output_rows",
                message="scored CSV contains zero rows",
                details={"output_csv": str(produced)},
            )
        if "candidate_id" not in fieldnames:
            raise ContractValidationError(
                code="missing_candidate_id",
                message="scored CSV must include candidate_id",
                details={"fieldnames": fieldnames},
            )
        if "smiles" not in fieldnames and "SMILES" not in fieldnames:
            raise ContractValidationError(
                code="missing_smiles_column",
                message="scored CSV must include smiles or SMILES column",
                details={"fieldnames": fieldnames},
            )
        if "plqy_score" not in fieldnames:
            raise ContractValidationError(
                code="missing_score_column",
                message="scored CSV must include plqy_score for this validation payload",
                details={"fieldnames": fieldnames},
            )
        return {
            "tool": "score_candidates",
            "status": "pass",
            "checks": [
                "stdout is JSON object",
                "status field is non-empty string",
                "output CSV exists",
                "output CSV has rows",
                "output CSV includes candidate_id",
                "output CSV includes smiles/SMILES",
                "output CSV includes plqy_score",
            ],
            "result_preview": {
                "status": result.get("status"),
                "adapter": result.get("adapter"),
                "output_csv": str(produced),
                "rows": len(rows),
            },
        }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate oled-agent adapter command contract")
    p.add_argument("--tool", required=True, choices=["train_predictor", "generate_candidates", "score_candidates"])
    p.add_argument("--cmd", required=True, help="Adapter command, for example: 'python3 scripts/adapters/score_candidates_adapter_template.py'")
    p.add_argument("--workspace-root", default=".", help="Working directory used to execute adapter command")
    p.add_argument("--timeout-sec", type=int, default=60, help="Adapter command timeout in seconds")
    p.add_argument("--json", action="store_true", help="Emit machine-readable JSON result")
    return p.parse_args()


def _emit(payload: Dict[str, Any], as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    if payload.get("status") == "pass":
        print(f"[PASS] tool={payload.get('tool')}")
        for c in payload.get("checks", []):
            print(f"- {c}")
    else:
        print(f"[FAIL] tool={payload.get('tool')} code={payload.get('error', {}).get('code')}")
        print(payload.get("error", {}).get("message", "unknown error"))


def main() -> int:
    args = parse_args()
    workspace_root = Path(args.workspace_root).resolve()
    timeout_sec = max(1, int(args.timeout_sec))

    try:
        if args.tool == "train_predictor":
            report = _validate_train(cmd=args.cmd, workspace_root=workspace_root, timeout_sec=timeout_sec)
        elif args.tool == "generate_candidates":
            report = _validate_generate(cmd=args.cmd, workspace_root=workspace_root, timeout_sec=timeout_sec)
        else:
            report = _validate_score(cmd=args.cmd, workspace_root=workspace_root, timeout_sec=timeout_sec)
    except ContractValidationError as exc:
        payload = {
            "tool": args.tool,
            "status": "fail",
            "error": {
                "code": exc.code,
                "message": str(exc),
                "details": exc.details,
            },
        }
        _emit(payload, args.json)
        return 2

    _emit(report, args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

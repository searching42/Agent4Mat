#!/usr/bin/env python3
from __future__ import annotations

import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class AdapterFailure(Exception):
    code: str
    message: str
    details: Dict[str, Any]

    def __init__(self, *, code: str, message: str, details: Dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[2]


def to_int(value: object, *, default: int, min_value: int = 1, name: str = "value") -> int:
    if value in (None, ""):
        return default
    try:
        parsed = int(value)
    except Exception as exc:
        raise AdapterFailure(
            code="invalid_env_config",
            message=f"{name} must be an integer",
            details={"name": name, "value": value},
        ) from exc
    if parsed < min_value:
        raise AdapterFailure(
            code="invalid_env_config",
            message=f"{name} must be >= {min_value}",
            details={"name": name, "value": parsed, "min_value": min_value},
        )
    return parsed


def read_payload() -> Dict[str, Any]:
    raw = sys.stdin.read()
    if not raw.strip():
        raise AdapterFailure(code="empty_stdin", message="stdin is empty; expected one JSON object")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise AdapterFailure(
            code="invalid_json_stdin",
            message=f"stdin is not valid JSON: {exc}",
        ) from exc
    if not isinstance(payload, dict):
        raise AdapterFailure(
            code="invalid_payload_type",
            message="stdin JSON must be an object",
            details={"json_type": type(payload).__name__},
        )
    return payload


def resolve_path(path_like: str, *, workspace_root: Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return (workspace_root / path).resolve()


def parse_cmd(cmd: str) -> List[str]:
    try:
        argv = shlex.split(cmd)
    except ValueError as exc:
        raise AdapterFailure(
            code="invalid_cmd",
            message=f"failed to parse command: {exc}",
            details={"cmd": cmd},
        ) from exc
    if not argv:
        raise AdapterFailure(code="invalid_cmd", message="empty command", details={"cmd": cmd})
    return argv


def run_json_cmd(*, cmd: str, payload: Dict[str, Any], cwd: Path, timeout_sec: int) -> Dict[str, Any]:
    argv = parse_cmd(cmd)
    try:
        cp = subprocess.run(
            argv,
            input=json.dumps(payload, ensure_ascii=False),
            cwd=str(cwd),
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired as exc:
        raise AdapterFailure(
            code="adapter_timeout",
            message=f"adapter command timed out after {timeout_sec}s",
            details={"cmd": cmd},
        ) from exc

    if cp.returncode != 0:
        raise AdapterFailure(
            code="adapter_nonzero_exit",
            message=f"adapter command exited with non-zero status: {cp.returncode}",
            details={
                "cmd": cmd,
                "returncode": cp.returncode,
                "stderr_tail": (cp.stderr or "")[-1000:],
                "stdout_tail": (cp.stdout or "")[-1000:],
            },
        )

    raw = (cp.stdout or "").strip()
    if not raw:
        raise AdapterFailure(
            code="empty_stdout",
            message="adapter command returned empty stdout; expected one JSON object",
            details={"cmd": cmd},
        )
    try:
        result = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise AdapterFailure(
            code="invalid_json_stdout",
            message=f"adapter command stdout is not valid JSON: {exc}",
            details={"cmd": cmd, "stdout_tail": raw[-1000:]},
        ) from exc
    if not isinstance(result, dict):
        raise AdapterFailure(
            code="invalid_json_type",
            message="adapter command stdout JSON must be an object",
            details={"json_type": type(result).__name__},
        )
    return result


def run_argv_cmd(*, argv: List[str], cwd: Path, timeout_sec: int) -> Dict[str, Any]:
    if not argv:
        raise AdapterFailure(code="invalid_cmd", message="empty argv")
    try:
        cp = subprocess.run(
            argv,
            cwd=str(cwd),
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired as exc:
        raise AdapterFailure(
            code="adapter_timeout",
            message=f"command timed out after {timeout_sec}s",
            details={"argv": argv},
        ) from exc
    if cp.returncode != 0:
        raise AdapterFailure(
            code="adapter_nonzero_exit",
            message=f"command exited with non-zero status: {cp.returncode}",
            details={
                "argv": argv,
                "returncode": cp.returncode,
                "stderr_tail": (cp.stderr or "")[-1000:],
                "stdout_tail": (cp.stdout or "")[-1000:],
            },
        )
    return {
        "stdout": cp.stdout or "",
        "stderr": cp.stderr or "",
    }


def emit_success(payload: Dict[str, Any]) -> int:
    out = dict(payload)
    out.setdefault("status", "success")
    print(json.dumps(out, ensure_ascii=False))
    return 0


def emit_failure(*, code: str, message: str, details: Dict[str, Any] | None = None, exit_code: int = 2) -> int:
    print(
        json.dumps(
            {
                "status": "failed",
                "error": {
                    "code": code,
                    "message": message,
                    "details": details or {},
                },
            },
            ensure_ascii=False,
        )
    )
    return exit_code

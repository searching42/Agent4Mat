from __future__ import annotations

import importlib.util
import json
import os
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_UNIMOL_REMOTE_HOST = "lbh@211.86.155.63"
DEFAULT_UNIMOL_REMOTE_PY = "/home/lbh/miniconda3/envs/unimol/bin/python"
DEFAULT_UNIMOL_REMOTE_TMP_BASE = "/home/lbh/work/wk1/openclaw_sync"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _mk_result(
    *,
    name: str,
    status: str,
    message: str,
    details: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "name": name,
        "status": status,
        "message": message,
        "details": details or {},
    }


def _check_python_version(min_major: int = 3, min_minor: int = 9) -> Dict[str, Any]:
    major = sys.version_info.major
    minor = sys.version_info.minor
    version_str = f"{major}.{minor}.{sys.version_info.micro}"
    ok = (major, minor) >= (min_major, min_minor)
    if ok:
        return _mk_result(
            name="python_version",
            status="pass",
            message=f"Python {version_str} is supported",
            details={"version": version_str, "minimum": f"{min_major}.{min_minor}"},
        )
    return _mk_result(
        name="python_version",
        status="fail",
        message=f"Python {version_str} is below required {min_major}.{min_minor}",
        details={"version": version_str, "minimum": f"{min_major}.{min_minor}"},
    )


def _check_workspace_writable(workspace_root: Path) -> Dict[str, Any]:
    workspace_root.mkdir(parents=True, exist_ok=True)
    probe = workspace_root / ".doctor_write_probe"
    try:
        probe.write_text("ok\n", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return _mk_result(
            name="workspace_writable",
            status="pass",
            message=f"Workspace is writable: {workspace_root}",
        )
    except Exception as exc:
        return _mk_result(
            name="workspace_writable",
            status="fail",
            message=f"Workspace is not writable: {workspace_root}",
            details={"error": str(exc)},
        )


def _check_command(cmd: str, required: bool = False) -> Dict[str, Any]:
    path = shutil.which(cmd)
    if path:
        return _mk_result(
            name=f"command:{cmd}",
            status="pass",
            message=f"Found command '{cmd}'",
            details={"path": path},
        )
    return _mk_result(
        name=f"command:{cmd}",
        status="fail" if required else "warn",
        message=f"Command '{cmd}' not found",
    )


def _check_module(module_name: str, required: bool = False) -> Dict[str, Any]:
    spec = importlib.util.find_spec(module_name)
    if spec is not None:
        return _mk_result(
            name=f"module:{module_name}",
            status="pass",
            message=f"Python module '{module_name}' is importable",
        )
    return _mk_result(
        name=f"module:{module_name}",
        status="fail" if required else "warn",
        message=f"Python module '{module_name}' is not importable",
    )


def _check_nvidia_smi() -> Dict[str, Any]:
    which = shutil.which("nvidia-smi")
    if not which:
        return _mk_result(
            name="gpu:nvidia_smi",
            status="warn",
            message="nvidia-smi not found (GPU profile may be unavailable)",
        )
    try:
        cp = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception as exc:
        return _mk_result(
            name="gpu:nvidia_smi",
            status="warn",
            message="Failed to execute nvidia-smi",
            details={"error": str(exc)},
        )

    if cp.returncode != 0:
        return _mk_result(
            name="gpu:nvidia_smi",
            status="warn",
            message="nvidia-smi returned non-zero exit code",
            details={"returncode": cp.returncode, "stderr": cp.stderr.strip()},
        )

    lines = [line.strip() for line in cp.stdout.splitlines() if line.strip()]
    return _mk_result(
        name="gpu:nvidia_smi",
        status="pass",
        message="GPU detected via nvidia-smi",
        details={"gpus": lines},
    )


def _check_docker_compose() -> Dict[str, Any]:
    docker = shutil.which("docker")
    if not docker:
        return _mk_result(
            name="docker:compose",
            status="warn",
            message="docker not found",
        )

    cp = subprocess.run(
        ["docker", "compose", "version"],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if cp.returncode == 0:
        return _mk_result(
            name="docker:compose",
            status="pass",
            message="docker compose is available",
            details={"version": cp.stdout.strip()},
        )

    return _mk_result(
        name="docker:compose",
        status="warn",
        message="docker compose command failed",
        details={"stderr": cp.stderr.strip(), "returncode": cp.returncode},
    )


def _check_env_var(name: str) -> Dict[str, Any]:
    value = os.environ.get(name, "").strip()
    if value:
        return _mk_result(
            name=f"env:{name}",
            status="pass",
            message=f"Environment variable {name} is set",
        )
    return _mk_result(
        name=f"env:{name}",
        status="warn",
        message=f"Environment variable {name} is not set",
    )


def _tail(text: str, n: int = 500) -> str:
    return (text or "")[-n:]


def _resolve_unimol_remote_runtime() -> Dict[str, Any]:
    host = os.environ.get("UNIMOL_REMOTE_HOST", "").strip()
    remote_py = os.environ.get("UNIMOL_REMOTE_PY", "").strip()
    tmp_base = os.environ.get("UNIMOL_REMOTE_TMP_BASE", "").strip()
    allow_default = os.environ.get("ALLOW_DEFAULT_UNIMOL_REMOTE", "0").strip() == "1"

    values = {
        "UNIMOL_REMOTE_HOST": host,
        "UNIMOL_REMOTE_PY": remote_py,
        "UNIMOL_REMOTE_TMP_BASE": tmp_base,
    }
    present = [k for k, v in values.items() if v]

    if present and len(present) < len(values):
        missing = [k for k in values if k not in present]
        return {
            "status": "invalid",
            "source": "partial",
            "present": present,
            "missing": missing,
            "allow_default": allow_default,
        }

    if len(present) == len(values):
        return {
            "status": "ok",
            "source": "env",
            "host": host,
            "remote_py": remote_py,
            "tmp_base": tmp_base,
            "allow_default": allow_default,
        }

    if allow_default:
        return {
            "status": "ok",
            "source": "default",
            "host": DEFAULT_UNIMOL_REMOTE_HOST,
            "remote_py": DEFAULT_UNIMOL_REMOTE_PY,
            "tmp_base": DEFAULT_UNIMOL_REMOTE_TMP_BASE,
            "allow_default": allow_default,
        }

    return {
        "status": "missing",
        "source": "none",
        "present": present,
        "allow_default": allow_default,
    }


def _check_external_runtime_config() -> Dict[str, Any]:
    cfg = _resolve_unimol_remote_runtime()
    if cfg["status"] == "invalid":
        return _mk_result(
            name="external:runtime_config",
            status="fail",
            message="Incomplete UNIMOL remote configuration",
            details={
                "present": cfg.get("present", []),
                "missing": cfg.get("missing", []),
                "required": ["UNIMOL_REMOTE_HOST", "UNIMOL_REMOTE_PY", "UNIMOL_REMOTE_TMP_BASE"],
            },
        )

    if cfg["status"] == "missing":
        return _mk_result(
            name="external:runtime_config",
            status="fail",
            message="Remote runtime is not configured",
            details={
                "required": ["UNIMOL_REMOTE_HOST", "UNIMOL_REMOTE_PY", "UNIMOL_REMOTE_TMP_BASE"],
                "hint": "Set all UNIMOL_REMOTE_* vars or ALLOW_DEFAULT_UNIMOL_REMOTE=1 for legacy defaults.",
            },
        )

    source = cfg.get("source")
    status = "pass" if source == "env" else "warn"
    message = "Remote runtime configured via UNIMOL_REMOTE_* env vars"
    if source == "default":
        message = "Using legacy default remote runtime (ALLOW_DEFAULT_UNIMOL_REMOTE=1)"
    return _mk_result(
        name="external:runtime_config",
        status=status,
        message=message,
        details={
            "source": source,
            "host": cfg.get("host", ""),
            "remote_py": cfg.get("remote_py", ""),
            "tmp_base": cfg.get("tmp_base", ""),
        },
    )


def _classify_ssh_failure(stderr: str) -> str:
    text = (stderr or "").lower()
    if "permission denied" in text or "publickey" in text or "authentication failed" in text:
        return "auth_failed"
    if "host key verification failed" in text or "remote host identification has changed" in text:
        return "host_key_issue"
    if "no route to host" in text or "network is unreachable" in text:
        return "host_unreachable"
    if "connection refused" in text:
        return "connection_refused"
    if "connection timed out" in text or "operation timed out" in text:
        return "connection_timeout"
    if "could not resolve hostname" in text or "name or service not known" in text:
        return "dns_resolution_failed"
    if "temporary failure in name resolution" in text:
        return "dns_resolution_failed"
    return "unknown"


def _run_ssh_probe(host: str, remote_cmd: str, *, timeout: int = 12) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "ConnectTimeout=8",
            host,
            remote_cmd,
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _check_external_ssh_connectivity(host: str) -> Dict[str, Any]:
    try:
        cp = _run_ssh_probe(host, "echo OPENCLAW_REMOTE_OK", timeout=12)
    except Exception as exc:
        return _mk_result(
            name="external:ssh_connectivity",
            status="fail",
            message="SSH probe could not be executed",
            details={"host": host, "error": str(exc)},
        )

    if cp.returncode == 0:
        return _mk_result(
            name="external:ssh_connectivity",
            status="pass",
            message="SSH connectivity probe passed",
            details={"host": host},
        )

    stderr_tail = _tail(cp.stderr)
    return _mk_result(
        name="external:ssh_connectivity",
        status="fail",
        message=f"SSH connectivity probe failed ({_classify_ssh_failure(stderr_tail)})",
        details={
            "host": host,
            "returncode": cp.returncode,
            "stderr_tail": stderr_tail,
            "classification": _classify_ssh_failure(stderr_tail),
        },
    )


def _check_external_remote_python(host: str, remote_py: str) -> Dict[str, Any]:
    remote_cmd = f"test -x {shlex.quote(remote_py)} && echo OPENCLAW_PY_OK"
    try:
        cp = _run_ssh_probe(host, remote_cmd, timeout=12)
    except Exception as exc:
        return _mk_result(
            name="external:remote_python",
            status="fail",
            message="Remote python probe could not be executed",
            details={"host": host, "remote_py": remote_py, "error": str(exc)},
        )

    if cp.returncode == 0:
        return _mk_result(
            name="external:remote_python",
            status="pass",
            message="Remote python path is executable",
            details={"host": host, "remote_py": remote_py},
        )

    stderr_tail = _tail(cp.stderr)
    return _mk_result(
        name="external:remote_python",
        status="fail",
        message="Remote python path is not executable or host is unreachable",
        details={
            "host": host,
            "remote_py": remote_py,
            "returncode": cp.returncode,
            "stderr_tail": stderr_tail,
            "classification": _classify_ssh_failure(stderr_tail),
        },
    )


def _check_external_remote_tmp_base(host: str, tmp_base: str) -> Dict[str, Any]:
    remote_cmd = (
        f"mkdir -p {shlex.quote(tmp_base)} "
        f"&& test -w {shlex.quote(tmp_base)} "
        f"&& echo OPENCLAW_TMP_OK"
    )
    try:
        cp = _run_ssh_probe(host, remote_cmd, timeout=15)
    except Exception as exc:
        return _mk_result(
            name="external:remote_tmp_base",
            status="fail",
            message="Remote tmp_base probe could not be executed",
            details={"host": host, "tmp_base": tmp_base, "error": str(exc)},
        )

    if cp.returncode == 0:
        return _mk_result(
            name="external:remote_tmp_base",
            status="pass",
            message="Remote tmp_base is writable",
            details={"host": host, "tmp_base": tmp_base},
        )

    stderr_tail = _tail(cp.stderr)
    classification = _classify_ssh_failure(stderr_tail)
    if classification == "unknown":
        classification = "remote_path_or_permission_issue"
    return _mk_result(
        name="external:remote_tmp_base",
        status="fail",
        message="Remote tmp_base is not writable or host is unreachable",
        details={
            "host": host,
            "tmp_base": tmp_base,
            "returncode": cp.returncode,
            "stderr_tail": stderr_tail,
            "classification": classification,
        },
    )


def _resolve_external_workspace_root(workspace_root: Path) -> Optional[Path]:
    candidates = [workspace_root, workspace_root.parent, workspace_root.parent.parent]
    scorer_rel = Path("scripts") / "score_unimol_property_candidates.py"

    for c in candidates:
        if (c / scorer_rel).exists():
            return c

    for c in candidates:
        if (c / "scripts").exists():
            return c
    return None


def _check_external_scorer_chain(workspace_root: Path) -> Dict[str, Any]:
    ext_root = _resolve_external_workspace_root(workspace_root)
    if ext_root is None:
        return _mk_result(
            name="external:scorer_chain",
            status="warn",
            message="external workspace root with scripts/ not found",
        )

    scorer = ext_root / "scripts" / "score_unimol_property_candidates.py"
    if not scorer.exists():
        return _mk_result(
            name="external:scorer_chain",
            status="warn",
            message="external scorer script missing",
            details={"scorer": str(scorer)},
        )

    enabled = os.environ.get("OLED_AGENT_USE_EXTERNAL_SCORER", "0").strip() == "1"
    if not enabled:
        return _mk_result(
            name="external:scorer_chain",
            status="warn",
            message="external scorer disabled (set OLED_AGENT_USE_EXTERNAL_SCORER=1 to enable)",
            details={"scorer": str(scorer), "workspace": str(ext_root)},
        )

    probe = subprocess.run(
        ["python3", str(scorer), "--help"],
        cwd=str(ext_root),
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
    if probe.returncode != 0:
        return _mk_result(
            name="external:scorer_chain",
            status="fail",
            message="external scorer preflight failed",
            details={
                "scorer": str(scorer),
                "returncode": probe.returncode,
                "stderr_tail": (probe.stderr or "")[-500:],
            },
        )

    return _mk_result(
        name="external:scorer_chain",
        status="pass",
        message="external scorer chain is available",
        details={"scorer": str(scorer), "workspace": str(ext_root)},
    )


def build_doctor_report(workspace_root: Path) -> Dict[str, Any]:
    checks: List[Dict[str, Any]] = []

    checks.append(_check_python_version(3, 9))
    checks.append(_check_workspace_writable(workspace_root))

    checks.append(_check_command("python3", required=True))
    checks.append(_check_command("git", required=False))
    checks.append(_check_command("bash", required=False))

    checks.append(_check_docker_compose())
    checks.append(_check_nvidia_smi())

    checks.append(_check_module("torch", required=False))
    checks.append(_check_module("rdkit", required=False))
    checks.append(_check_module("unimol_tools", required=False))
    # MinerU package name on PyPI is magic-pdf; import path is magic_pdf
    checks.append(_check_module("magic_pdf", required=False))

    checks.append(_check_env_var("HF_ENDPOINT"))
    checks.append(_check_env_var("UNIMOL_WEIGHT_DIR"))
    checks.append(_check_external_scorer_chain(workspace_root))

    summary = {"pass": 0, "warn": 0, "fail": 0}
    for check in checks:
        status = check["status"]
        if status in summary:
            summary[status] += 1

    overall = "pass"
    if summary["fail"] > 0:
        overall = "fail"
    elif summary["warn"] > 0:
        overall = "warn"

    return {
        "generated_at": _now_iso(),
        "workspace_root": str(workspace_root.resolve()),
        "overall": overall,
        "summary": summary,
        "checks": checks,
    }


def run_external_preflight(*, workspace_root: Path) -> Dict[str, Any]:
    checks: List[Dict[str, Any]] = []
    chain = _check_external_scorer_chain(workspace_root.resolve())
    checks.append(chain)

    # Basic tool requirements for remote transport.
    checks.append(_check_command("ssh", required=True))
    checks.append(_check_command("scp", required=True))

    # If local script/enablement failed, return immediately with actionable status.
    if chain["status"] != "pass":
        summary = {"pass": 0, "warn": 0, "fail": 0}
        for c in checks:
            summary[c["status"]] += 1
        overall = "fail" if summary["fail"] > 0 else ("warn" if summary["warn"] > 0 else "pass")
        return {
            "generated_at": _now_iso(),
            "workspace_root": str(workspace_root.resolve()),
            "overall": overall,
            "summary": summary,
            "checks": checks,
            "exit_code": 0 if overall == "pass" else 1,
        }

    runtime = _check_external_runtime_config()
    checks.append(runtime)

    if runtime["status"] in ("pass", "warn"):
        cfg = _resolve_unimol_remote_runtime()
        host = str(cfg.get("host", ""))
        remote_py = str(cfg.get("remote_py", ""))
        tmp_base = str(cfg.get("tmp_base", ""))
        if host:
            checks.append(_check_external_ssh_connectivity(host))
            checks.append(_check_external_remote_python(host, remote_py))
            checks.append(_check_external_remote_tmp_base(host, tmp_base))

    summary = {"pass": 0, "warn": 0, "fail": 0}
    for c in checks:
        summary[c["status"]] += 1
    overall = "pass"
    if summary["fail"] > 0:
        overall = "fail"
    elif summary["warn"] > 0:
        overall = "warn"
    return {
        "generated_at": _now_iso(),
        "workspace_root": str(workspace_root.resolve()),
        "overall": overall,
        "summary": summary,
        "checks": checks,
        "exit_code": 0 if overall == "pass" else 1,
    }


def _build_external_connectivity_summary(report: Dict[str, Any]) -> Dict[str, Any]:
    checks = report.get("checks", [])
    by_name = {c.get("name", ""): c for c in checks}
    blocking_checks = [c.get("name", "") for c in checks if c.get("status") == "fail"]

    failure_classes: List[str] = []
    for c in checks:
        details = c.get("details") or {}
        classification = details.get("classification")
        if classification and classification not in failure_classes:
            failure_classes.append(classification)

    runtime_source = ""
    runtime_check = by_name.get("external:runtime_config")
    if runtime_check:
        runtime_source = str((runtime_check.get("details") or {}).get("source") or "")

    return {
        "chain_ready": report.get("overall") == "pass",
        "runtime_source": runtime_source or "unknown",
        "blocking_checks": blocking_checks,
        "failure_classes": failure_classes,
        "check_status": {c.get("name", ""): c.get("status", "") for c in checks},
    }


def run_external_connectivity_debug(
    *,
    workspace_root: Path,
    json_out: Optional[Path] = None,
) -> Dict[str, Any]:
    report = run_external_preflight(workspace_root=workspace_root.resolve())
    report["report_type"] = "external_connectivity_debug_v1"
    report["connectivity"] = _build_external_connectivity_summary(report)

    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return report


def run_doctor(
    *,
    workspace_root: Path,
    json_out: Optional[Path] = None,
    strict: bool = False,
) -> Dict[str, Any]:
    report = build_doctor_report(workspace_root)

    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if strict:
        if report["summary"]["fail"] > 0 or report["summary"]["warn"] > 0:
            report["exit_code"] = 1
            return report

    report["exit_code"] = 1 if report["summary"]["fail"] > 0 else 0
    return report

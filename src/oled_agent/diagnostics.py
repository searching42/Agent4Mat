from __future__ import annotations

import importlib.util
import json
import os
import shlex
import shutil
import socket
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Legacy defaults are intentionally non-sensitive placeholders and can be
# overridden explicitly via env for local compatibility.
DEFAULT_UNIMOL_REMOTE_HOST = os.environ.get("OLED_AGENT_DEFAULT_UNIMOL_REMOTE_HOST", "<user@host>")
DEFAULT_UNIMOL_REMOTE_PY = os.environ.get("OLED_AGENT_DEFAULT_UNIMOL_REMOTE_PY", "<remote_python_path>")
DEFAULT_UNIMOL_REMOTE_TMP_BASE = os.environ.get(
    "OLED_AGENT_DEFAULT_UNIMOL_REMOTE_TMP_BASE",
    "<remote_writable_tmp_dir>",
)


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


def _build_summary_from_checks(checks: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary = {"pass": 0, "warn": 0, "fail": 0}
    for c in checks:
        status = str(c.get("status") or "")
        if status in summary:
            summary[status] += 1
    return summary


def _overall_from_summary(summary: Dict[str, int]) -> str:
    if int(summary.get("fail", 0)) > 0:
        return "fail"
    if int(summary.get("warn", 0)) > 0:
        return "warn"
    return "pass"


def _redact_sensitive_text(text: str) -> str:
    out = str(text or "")
    api_key = str(os.environ.get("OLED_AGENT_LLM_API_KEY", "") or "")
    if api_key:
        out = out.replace(api_key, "***")
    # Lightweight token masking after Bearer prefix.
    marker = "Bearer "
    pos = out.find(marker)
    while pos >= 0:
        start = pos + len(marker)
        end = start
        while end < len(out) and out[end] not in " \t\r\n\",;":
            end += 1
        if end > start:
            out = out[:start] + "***" + out[end:]
            pos = out.find(marker, start + 3)
        else:
            pos = out.find(marker, end)
    return out


def _build_minimal_probe_body(config: Dict[str, Any]) -> Dict[str, Any]:
    body: Dict[str, Any] = {
        "model": config["model"],
        "messages": [
            {"role": "system", "content": "You are a connectivity probe."},
            {"role": "user", "content": "Reply with short text: ok"},
        ],
    }
    extra_raw = str(os.environ.get("OLED_AGENT_LLM_CONNECTIVITY_PROBE_EXTRA_BODY_JSON", "") or "").strip()
    if extra_raw:
        try:
            extra_obj = json.loads(extra_raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "Invalid OLED_AGENT_LLM_CONNECTIVITY_PROBE_EXTRA_BODY_JSON: must be JSON object"
            ) from exc
        if not isinstance(extra_obj, dict):
            raise RuntimeError(
                "Invalid OLED_AGENT_LLM_CONNECTIVITY_PROBE_EXTRA_BODY_JSON: must be JSON object"
            )
        body.update(extra_obj)
    return body


def _is_timeout_url_error(exc: urllib.error.URLError) -> bool:
    reason = getattr(exc, "reason", None)
    if isinstance(reason, (socket.timeout, TimeoutError)):
        return True
    text = str(reason or exc).lower()
    return "timed out" in text or "timeout" in text


def _probe_openai_compat_connectivity(config: Dict[str, Any]) -> Dict[str, Any]:
    endpoint = f"{config['base_url']}{config['chat_completions_path']}"
    headers: Dict[str, str] = {"Content-Type": "application/json"}
    auth_scheme = str(config.get("auth_scheme", "Bearer"))
    auth_header = str(config.get("auth_header", "Authorization"))
    if auth_scheme:
        headers[auth_header] = f"{auth_scheme} {config['api_key']}"
    else:
        headers[auth_header] = str(config["api_key"])
    extra_headers = config.get("extra_headers")
    if isinstance(extra_headers, dict):
        for k, v in extra_headers.items():
            headers[str(k)] = str(v)

    body = _build_minimal_probe_body(config)
    req = urllib.request.Request(
        endpoint,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=float(config["timeout_sec"])) as resp:
        raw = resp.read().decode("utf-8", errors="replace")

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"connectivity probe response is not JSON: {exc}") from exc

    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("connectivity probe response missing choices")

    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        raise RuntimeError("connectivity probe response missing message object")

    return {
        "endpoint": endpoint,
        "response_id": str(payload.get("id") or ""),
        "model": str(config.get("model") or ""),
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


def run_llm_connectivity(
    *,
    workspace_root: Path,
    catalog_path: Optional[Path] = None,
    json_out: Optional[Path] = None,
) -> Dict[str, Any]:
    from oled_agent.agent.planner import (
        _llm_backend_config_from_env,
        _llm_backend_from_env,
        _llm_cmd_from_env,
        _run_llm_planner_command,
    )

    ws_root = workspace_root.resolve()
    cat_path = catalog_path.resolve() if catalog_path is not None else (ws_root / "configs" / "models" / "catalog.json")
    checks: List[Dict[str, Any]] = []
    source = "none"

    cmd = _llm_cmd_from_env()
    backend = _llm_backend_from_env()
    if cmd:
        source = "command"
    elif backend:
        source = "backend"

    checks.append(
        _mk_result(
            name="llm:source",
            status="pass" if source != "none" else "fail",
            message="LLM source unresolved (none)" if source == "none" else f"LLM source resolved to {source}",
            details={"source": source, "catalog_path": str(cat_path)},
        )
    )

    if source == "none":
        checks.append(
            _mk_result(
                name="llm:config",
                status="fail",
                message="LLM required config missing (set OLED_AGENT_LLM_PLANNER_CMD or OLED_AGENT_LLM_BACKEND)",
            )
        )
    elif source == "command":
        probe_payload = {
            "task_id": "llm_connectivity_probe",
            "request_text": "LLM connectivity probe",
            "mode": "fast_screen",
            "targets": [{"property": "plqy", "objective": "maximize", "target_value": 0.6}],
            "budget": {"max_candidates": 3},
        }
        try:
            result = _run_llm_planner_command(
                cmd=cmd,
                payload=probe_payload,
                catalog_path=cat_path,
            )
            if not isinstance(result, dict):
                raise RuntimeError("planner command output is not JSON object")
            checks.append(
                _mk_result(
                    name="llm:command_probe",
                    status="pass",
                    message="LLM planner command responded with JSON",
                    details={
                        "keys": sorted(result.keys()),
                    },
                )
            )
        except Exception as exc:
            checks.append(
                _mk_result(
                    name="llm:command_probe",
                    status="fail",
                    message="LLM planner command probe failed",
                    details={"error": _redact_sensitive_text(str(exc))},
                )
            )
    else:
        try:
            config = _llm_backend_config_from_env(backend)
            checks.append(
                _mk_result(
                    name="llm:backend_config",
                    status="pass",
                    message=f"LLM backend config parsed ({backend})",
                    details={
                        "backend": backend,
                        "base_url": str(config.get("base_url") or ""),
                        "chat_completions_path": str(config.get("chat_completions_path") or ""),
                        "model": str(config.get("model") or ""),
                        "auth_header": str(config.get("auth_header") or ""),
                    },
                )
            )
        except Exception as exc:
            checks.append(
                _mk_result(
                    name="llm:backend_config",
                    status="fail",
                    message="LLM backend config invalid",
                    details={"backend": backend, "error": _redact_sensitive_text(str(exc))},
                )
            )
            config = None

        if config is not None:
            try:
                details = _probe_openai_compat_connectivity(config)
                checks.append(
                    _mk_result(
                        name="llm:backend_probe",
                        status="pass",
                        message="LLM backend connectivity probe succeeded",
                        details=details,
                    )
                )
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                details: Dict[str, Any] = {"code": int(exc.code)}
                try:
                    from oled_agent.agent.planner import _llm_debug_error_enabled

                    debug_error = bool(_llm_debug_error_enabled())
                except Exception:
                    debug_error = False
                if debug_error:
                    details["body_tail"] = _tail(_redact_sensitive_text(body), 500)
                checks.append(
                    _mk_result(
                        name="llm:backend_probe",
                        status="fail",
                        message=f"LLM backend HTTP error: {exc.code}",
                        details=details,
                    )
                )
            except urllib.error.URLError as exc:
                if _is_timeout_url_error(exc):
                    checks.append(
                        _mk_result(
                            name="llm:backend_probe",
                            status="fail",
                            message="LLM backend timeout",
                            details={"error": _redact_sensitive_text(str(exc))},
                        )
                    )
                else:
                    checks.append(
                        _mk_result(
                            name="llm:backend_probe",
                            status="fail",
                            message="LLM backend network error",
                            details={"error": _redact_sensitive_text(str(exc))},
                        )
                    )
            except socket.timeout as exc:
                checks.append(
                    _mk_result(
                        name="llm:backend_probe",
                        status="fail",
                        message="LLM backend timeout",
                        details={"error": _redact_sensitive_text(str(exc))},
                    )
                )
            except TimeoutError as exc:
                checks.append(
                    _mk_result(
                        name="llm:backend_probe",
                        status="fail",
                        message="LLM backend timeout",
                        details={"error": _redact_sensitive_text(str(exc))},
                    )
                )
            except Exception as exc:
                checks.append(
                    _mk_result(
                        name="llm:backend_probe",
                        status="fail",
                        message="LLM backend probe failed",
                        details={"error": _redact_sensitive_text(str(exc))},
                    )
                )

    summary = _build_summary_from_checks(checks)
    overall = _overall_from_summary(summary)
    report: Dict[str, Any] = {
        "report_type": "llm_connectivity_v1",
        "generated_at": _now_iso(),
        "workspace_root": str(ws_root),
        "overall": overall,
        "summary": summary,
        "checks": checks,
        "source": source,
        "exit_code": 1 if int(summary.get("fail", 0)) > 0 else 0,
    }
    report["connectivity"] = {
        "source": source,
        "is_ready": overall == "pass",
        "check_status": {c.get("name", ""): c.get("status", "") for c in checks},
        "blocking_checks": [c.get("name", "") for c in checks if c.get("status") == "fail"],
    }

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

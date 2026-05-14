from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_sha256(payload: Any) -> str:
    try:
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    except Exception:
        return ""
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _path_snapshot(path: Optional[Path]) -> Dict[str, Any]:
    if path is None:
        return {"path": "", "exists": False}
    out: Dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        return out
    try:
        st = path.stat()
        out["is_file"] = path.is_file()
        out["is_dir"] = path.is_dir()
        out["size_bytes"] = int(st.st_size)
        out["mtime"] = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat()
        if path.is_file():
            out["sha256"] = _sha256_file(path)
    except Exception as exc:
        out["snapshot_error"] = f"{type(exc).__name__}: {exc}"
    return out


def _resolve_optional_path(raw: Any, workspace_root: Path) -> Optional[Path]:
    text = str(raw or "").strip()
    if not text:
        return None
    p = Path(text)
    if not p.is_absolute():
        p = (workspace_root / p).resolve()
    else:
        p = p.resolve()
    return p


def _execution_summary(execution_payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    execution = execution_payload if isinstance(execution_payload, dict) else {}
    records = execution.get("records", []) if isinstance(execution.get("records"), list) else []
    success_n = 0
    failed_n = 0
    failed_steps: list[str] = []
    adapters: set[str] = set()
    for rec in records:
        if not isinstance(rec, dict):
            continue
        name = str(rec.get("name") or "")
        status = str(rec.get("status") or "")
        if status == "success":
            success_n += 1
        else:
            failed_n += 1
            if name:
                failed_steps.append(name)
        result = rec.get("result")
        if isinstance(result, dict):
            adapter = str(result.get("adapter") or "").strip()
            if adapter:
                adapters.add(adapter)
    return {
        "status": str(execution.get("status") or ""),
        "started_at": execution.get("started_at", ""),
        "ended_at": execution.get("ended_at", ""),
        "record_count": len(records),
        "success_count": success_n,
        "failed_count": failed_n,
        "failed_steps": failed_steps,
        "adapters": sorted(adapters),
    }


def _artifact_snapshots(artifact_paths: Optional[Dict[str, Path]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    if not isinstance(artifact_paths, dict):
        return out
    for key, value in artifact_paths.items():
        if not isinstance(key, str):
            continue
        if not isinstance(value, Path):
            continue
        out[key] = _path_snapshot(value)
    return out


def _pick_model_choice(
    *,
    model_choice: Optional[Dict[str, Any]],
    plan_payload: Optional[Dict[str, Any]],
    task_payload: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    if isinstance(model_choice, dict):
        return dict(model_choice)
    if isinstance(plan_payload, dict):
        design = plan_payload.get("design_spec")
        if isinstance(design, dict) and isinstance(design.get("model_choice"), dict):
            return dict(design.get("model_choice"))
    if isinstance(task_payload, dict):
        if isinstance(task_payload.get("model_choice"), dict):
            return dict(task_payload.get("model_choice"))
        return {
            "predictor_id": str(task_payload.get("prediction_model") or ""),
            "generator_id": "",
        }
    return {}


def _source_artifact_snapshots(tool_state: Optional[Dict[str, Any]], workspace_root: Path) -> Dict[str, Dict[str, Any]]:
    state = tool_state if isinstance(tool_state, dict) else {}
    return {
        "candidate_csv": _path_snapshot(_resolve_optional_path(state.get("candidate_csv"), workspace_root)),
        "scored_csv": _path_snapshot(_resolve_optional_path(state.get("scored_csv"), workspace_root)),
        "final_output": _path_snapshot(_resolve_optional_path(state.get("final_output"), workspace_root)),
    }


def build_experiment_trace(
    *,
    task_id: str,
    run_label: str,
    workspace_root: Path,
    execution_mode: str,
    request_payload: Optional[Dict[str, Any]] = None,
    plan_payload: Optional[Dict[str, Any]] = None,
    task_payload: Optional[Dict[str, Any]] = None,
    execution_payload: Optional[Dict[str, Any]] = None,
    tool_state: Optional[Dict[str, Any]] = None,
    model_choice: Optional[Dict[str, Any]] = None,
    artifact_paths: Optional[Dict[str, Path]] = None,
) -> Dict[str, Any]:
    return {
        "schema_version": "1.0.0",
        "generated_at": _now_iso(),
        "task_id": str(task_id or ""),
        "run_label": str(run_label or ""),
        "execution_mode": str(execution_mode or ""),
        "model_choice": _pick_model_choice(
            model_choice=model_choice,
            plan_payload=plan_payload,
            task_payload=task_payload,
        ),
        "fingerprints": {
            "request_sha256": _json_sha256(request_payload) if isinstance(request_payload, dict) else "",
            "plan_sha256": _json_sha256(plan_payload) if isinstance(plan_payload, dict) else "",
            "task_sha256": _json_sha256(task_payload) if isinstance(task_payload, dict) else "",
            "execution_sha256": _json_sha256(execution_payload) if isinstance(execution_payload, dict) else "",
            "tool_state_sha256": _json_sha256(tool_state) if isinstance(tool_state, dict) else "",
        },
        "execution_summary": _execution_summary(execution_payload),
        "source_artifacts": _source_artifact_snapshots(tool_state, workspace_root),
        "core_artifacts": _artifact_snapshots(artifact_paths),
    }

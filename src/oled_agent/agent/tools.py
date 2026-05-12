from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from oled_agent.agent.model_catalog import ModelCatalog
from oled_agent.runner import run_pipeline


@dataclass
class ToolContext:
    workspace_root: Path
    catalog_path: Path
    task_id: str = "task_default"
    state: Dict[str, Any] = field(default_factory=dict)


class ToolError(RuntimeError):
    pass


class ExternalScorerError(ToolError):
    def __init__(
        self,
        *,
        code: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.details = details or {}
        self.retryable = retryable


def _external_error_payload(exc: Exception) -> Dict[str, Any]:
    if isinstance(exc, ExternalScorerError):
        return {
            "code": exc.code,
            "message": str(exc),
            "details": exc.details,
            "retryable": exc.retryable,
        }
    return {
        "code": "unexpected_external_error",
        "message": str(exc),
        "details": {},
        "retryable": False,
    }


def _workspace_rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except Exception:
        return str(path.resolve())


def _agent_artifact_dir(ctx: ToolContext) -> Path:
    out = ctx.workspace_root / "runs" / "agent" / ctx.task_id / "artifacts"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _tool_runtime_from_env(prefix: str) -> Dict[str, str]:
    return {
        "cmd": os.environ.get(f"{prefix}_CMD", "").strip(),
        "mode": os.environ.get(f"{prefix}_MODE", "").strip().lower(),
        "script": os.environ.get(f"{prefix}_SCRIPT", "").strip(),
        "timeout_sec": os.environ.get(f"{prefix}_TIMEOUT_SEC", "").strip(),
    }


def _run_json_tool_command(
    *,
    cmd: str,
    payload: Dict[str, Any],
    cwd: Path,
    timeout_sec: int = 900,
    env_overrides: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)
    cp = subprocess.run(
        shlex.split(cmd),
        input=json.dumps(payload, ensure_ascii=False),
        cwd=str(cwd),
        env=env,
        text=True,
        capture_output=True,
        check=True,
        timeout=timeout_sec,
    )
    raw = (cp.stdout or "").strip()
    if not raw:
        raise ToolError("Tool command returned empty output")
    try:
        result = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ToolError(f"Tool command output is not valid JSON: {exc}") from exc
    if not isinstance(result, dict):
        raise ToolError("Tool command output must be a JSON object")
    return result


def _tool_timeout(name: str, default: int = 900) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return max(1, int(raw))
    except ValueError as exc:
        raise ToolError(f"{name} must be an integer, got {raw!r}") from exc


def _resolve_tool_adapter_cmd(ctx: ToolContext, *, env_prefix: str, model_id: str, tool_name: str) -> str:
    runtime = _tool_runtime_from_env(env_prefix)
    if runtime["cmd"]:
        return runtime["cmd"]

    if not model_id:
        return ""

    try:
        catalog = ModelCatalog.load(ctx.catalog_path)
    except Exception:
        return ""

    model = catalog.get(model_id)
    if model is None:
        return ""
    return model.adapter_cmd(tool_name)


def _resolve_bundled_unimol_score_adapter_cmd(ctx: ToolContext, *, predictor_id: str) -> str:
    """Return bundled Uni-Mol score adapter command when catalog does not provide one."""
    if os.environ.get("OLED_AGENT_DISABLE_BUNDLED_UNIMOL_SCORE_ADAPTER", "0").strip() == "1":
        return ""

    try:
        catalog = ModelCatalog.load(ctx.catalog_path)
    except Exception:
        return ""

    model = catalog.get(predictor_id)
    if model is None or model.backend != "unimol_tools":
        return ""

    script = Path(__file__).resolve().parents[3] / "scripts" / "adapters" / "score_candidates_unimol_adapter.py"
    if not script.exists():
        return ""

    return f"python3 {script}"


def _resolve_bundled_reinvent4_generate_adapter_cmd(ctx: ToolContext, *, generator_id: str) -> str:
    """Return bundled REINVENT4 generate adapter command when catalog does not provide one."""
    if os.environ.get("OLED_AGENT_DISABLE_BUNDLED_REINVENT4_GENERATE_ADAPTER", "0").strip() == "1":
        return ""

    try:
        catalog = ModelCatalog.load(ctx.catalog_path)
    except Exception:
        return ""

    model = catalog.get(generator_id)
    if model is None or model.backend != "reinvent4":
        return ""

    script = Path(__file__).resolve().parents[3] / "scripts" / "adapters" / "generate_candidates_reinvent4_adapter.py"
    if not script.exists():
        return ""

    return f"python3 {script}"


def _default_adapter_mode_overrides(*, cmd: str, mode_env: str, default_mode: str) -> Dict[str, str]:
    """Return env override for adapter mode only when user has not set it explicitly."""
    if not cmd:
        return {}
    if os.environ.get(mode_env, "").strip():
        return {}
    return {mode_env: default_mode}


def _generate_candidates_local_fallback(
    *,
    ctx: ToolContext,
    generator_id: str,
    max_candidates: int,
    input_csv: str,
    constraints: Optional[Dict[str, Any]],
    out_csv: Path,
) -> Dict[str, Any]:
    if input_csv:
        explicit = Path(input_csv)
        if not explicit.is_absolute():
            explicit = (ctx.workspace_root / explicit).resolve()
        if not explicit.exists():
            raise ToolError(f"input_csv does not exist: {explicit}")
        n = _copy_head_rows(explicit, out_csv, max_candidates)
        ctx.state["candidate_csv"] = str(out_csv)
        return {
            "generator_id": generator_id,
            "status": "success",
            "adapter": "explicit_input_csv",
            "input_csv": str(explicit),
            "output": str(out_csv),
            "rows": n,
            "constraints": constraints or {},
        }

    ext_root = _resolve_external_workspace_root(ctx)
    if ext_root is not None:
        latest = _pick_latest_reinvent_csv(ext_root)
        if latest is not None:
            n = _copy_head_rows(latest, out_csv, max_candidates)
            ctx.state["candidate_csv"] = str(out_csv)
            return {
                "generator_id": generator_id,
                "status": "success",
                "adapter": "reuse_latest_reinvent_artifact",
                "source_csv": str(latest),
                "output": str(out_csv),
                "rows": n,
                "constraints": constraints or {},
            }

    n = _write_stub_candidates(out_csv, max_candidates)
    ctx.state["candidate_csv"] = str(out_csv)
    return {
        "generator_id": generator_id,
        "status": "success",
        "adapter": "stub_generator",
        "output": str(out_csv),
        "rows": n,
        "constraints": constraints or {},
    }


def _resolve_external_workspace_root(ctx: ToolContext) -> Optional[Path]:
    candidates = [ctx.workspace_root, ctx.workspace_root.parent, ctx.workspace_root.parent.parent]
    scorer_rel = Path("scripts") / "score_unimol_property_candidates.py"

    # Prefer the workspace root that actually contains the external scorer.
    for c in candidates:
        if (c / scorer_rel).exists():
            return c

    # Fallback to first root that has scripts/ for compatibility.
    for c in candidates:
        if (c / "scripts").exists():
            return c
    return None


def _pick_latest_reinvent_csv(ext_root: Path) -> Optional[Path]:
    run_dir = ext_root / "artifacts" / "server_sync" / "reinvent4_runs"
    if not run_dir.exists():
        return None
    files = sorted(run_dir.glob("openclaw_sampling_project_v1_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def _load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _extract_smiles(row: Dict[str, str]) -> str:
    for key in ("smiles", "SMILES", "Smiles", "canonical_smiles", "CANONICAL_SMILES"):
        value = (row.get(key) or "").strip()
        if value:
            return value
    return ""


def _normalize_candidate_rows(rows: List[Dict[str, str]], *, require_smiles: bool = False) -> List[Dict[str, str]]:
    normalized: List[Dict[str, str]] = []
    for i, row in enumerate(rows):
        out = dict(row)

        # Normalize SMILES field naming from heterogeneous upstream CSVs.
        smiles = _extract_smiles(out)
        if smiles:
            out["smiles"] = smiles
        elif require_smiles:
            raise ToolError(
                f"Missing smiles/SMILES in candidate row {i + 1}. "
                "Cannot proceed with scoring."
            )

        if not (out.get("candidate_id") or "").strip():
            out["candidate_id"] = f"cand_{i+1:06d}"

        normalized.append(out)
    return normalized


def _normalize_candidate_csv(path: Path, *, require_smiles: bool = False) -> int:
    rows = _load_rows(path)
    rows = _normalize_candidate_rows(rows, require_smiles=require_smiles)
    _write_rows(path, rows)
    return len(rows)


def _env_int(name: str, default: int, *, min_value: int = 0) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ExternalScorerError(
            code="invalid_env_config",
            message=f"{name} must be an integer, got '{raw}'",
            details={"name": name, "value": raw},
        ) from exc
    if value < min_value:
        raise ExternalScorerError(
            code="invalid_env_config",
            message=f"{name} must be >= {min_value}, got {value}",
            details={"name": name, "value": value, "min_value": min_value},
        )
    return value


def _env_float(name: str, default: float, *, min_value: float = 0.0) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ExternalScorerError(
            code="invalid_env_config",
            message=f"{name} must be a float, got '{raw}'",
            details={"name": name, "value": raw},
        ) from exc
    if value < min_value:
        raise ExternalScorerError(
            code="invalid_env_config",
            message=f"{name} must be >= {min_value}, got {value}",
            details={"name": name, "value": value, "min_value": min_value},
        )
    return value


def _run_command_with_retry(
    *,
    cmd: List[str],
    cwd: Path,
    timeout_sec: int,
    retries: int,
    backoff_sec: float,
    run_fn: Callable[..., Any] = subprocess.run,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> Dict[str, Any]:
    max_attempts = retries + 1
    error_history: List[Dict[str, Any]] = []

    for attempt in range(1, max_attempts + 1):
        try:
            cp = run_fn(
                cmd,
                cwd=str(cwd),
                check=True,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            )
            return {
                "attempts": attempt,
                "stdout_tail": (cp.stdout or "")[-2000:],
                "stderr_tail": (cp.stderr or "")[-2000:],
            }
        except subprocess.TimeoutExpired as exc:
            error_history.append(
                {
                    "attempt": attempt,
                    "type": "timeout",
                    "timeout_sec": timeout_sec,
                    "stdout_tail": ((exc.stdout or "") if isinstance(exc.stdout, str) else "")[-500:],
                    "stderr_tail": ((exc.stderr or "") if isinstance(exc.stderr, str) else "")[-500:],
                }
            )
            if attempt < max_attempts:
                sleep_fn(backoff_sec * attempt)
                continue
            raise ExternalScorerError(
                code="external_timeout",
                message=f"External scorer timed out after {max_attempts} attempt(s)",
                details={
                    "cmd": cmd,
                    "timeout_sec": timeout_sec,
                    "attempts": max_attempts,
                    "errors": error_history,
                },
                retryable=True,
            ) from exc
        except subprocess.CalledProcessError as exc:
            stderr_tail = (exc.stderr or "")[-500:]
            retryable = exc.returncode in (255, 124)
            if "timed out" in stderr_tail.lower() or "connection timed out" in stderr_tail.lower():
                retryable = True
            if "network is unreachable" in stderr_tail.lower() or "no route to host" in stderr_tail.lower():
                retryable = True
            if "connection refused" in stderr_tail.lower():
                retryable = True
            if "temporary failure in name resolution" in stderr_tail.lower():
                retryable = True
            if "exit status 255" in stderr_tail.lower():
                retryable = True
            if "scp" in stderr_tail.lower() and "calledprocesserror" in stderr_tail.lower():
                retryable = True

            error_history.append(
                {
                    "attempt": attempt,
                    "type": "nonzero_exit",
                    "returncode": exc.returncode,
                    "stdout_tail": (exc.stdout or "")[-500:],
                    "stderr_tail": stderr_tail,
                }
            )
            if retryable and attempt < max_attempts:
                sleep_fn(backoff_sec * attempt)
                continue
            raise ExternalScorerError(
                code="external_command_failed",
                message=f"External scorer command failed after {max_attempts} attempt(s)",
                details={
                    "cmd": cmd,
                    "attempts": max_attempts,
                    "errors": error_history,
                },
                retryable=retryable,
            ) from exc


def _write_rows(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def _copy_head_rows(src: Path, dst: Path, limit: int) -> int:
    rows = _load_rows(src)
    rows = rows[: max(1, limit)]
    rows = _normalize_candidate_rows(rows, require_smiles=False)
    _write_rows(dst, rows)
    return len(rows)


def _merge_csvs(base_csv: Path, addon_csv: Path, key: str = "candidate_id") -> None:
    base_rows = _load_rows(base_csv)
    addon_rows = _load_rows(addon_csv)

    base_rows = _normalize_candidate_rows(base_rows, require_smiles=False)

    # External scorer outputs must retain candidate_id for safe key-based merge.
    missing_key_rows = [i for i, r in enumerate(addon_rows, start=1) if not (r.get(key) or "").strip()]
    if missing_key_rows:
        raise ToolError(
            f"Cannot merge score file without '{key}' in addon rows: first missing rows={missing_key_rows[:5]}"
        )

    addon_map = {r.get(key, ""): r for r in addon_rows}

    for row in base_rows:
        extra = addon_map.get(row[key], {})
        for k, v in extra.items():
            if k == key:
                continue
            row[k] = v

    fieldnames = []
    seen = set()
    for r in base_rows + addon_rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)

    with base_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(base_rows)


def _ensure_candidate_id(rows: List[Dict[str, str]]) -> None:
    for i, row in enumerate(rows):
        if not (row.get("candidate_id") or "").strip():
            row["candidate_id"] = f"cand_{i+1:06d}"


def _stable_rand(smiles: str, salt: str) -> float:
    h = hashlib.sha256(f"{salt}:{smiles}".encode("utf-8")).hexdigest()
    v = int(h[:12], 16)
    return (v % 10_000_000) / 10_000_000.0


def _deterministic_target_pred(smiles: str, target_name: str) -> float:
    r = _stable_rand(smiles, f"pred:{target_name}")
    if target_name == "lambda_em":
        return 360.0 + 260.0 * r
    if target_name == "plqy":
        return max(0.0, min(100.0, 15.0 + 75.0 * r))
    return r


def _objective_score(pred: float, objective: str, target_center: float, sigma: float) -> float:
    if objective == "target_window":
        return math.exp(-abs(pred - target_center) / max(sigma, 1e-6))
    if objective == "maximize":
        return pred
    if objective == "minimize":
        return -pred
    return pred


def _write_stub_candidates(dst: Path, n: int) -> int:
    seed_smiles = [
        "c1ccc2ccccc2c1",
        "c1ncccc1",
        "CCOC(=O)N1CCN(CC1)C",
        "CN1C=NC2=CC=CC=C21",
        "c1ccc(cc1)N(c2ccccc2)c3ccccc3",
        "COc1ccc2ncccc2c1",
        "CC1=CC(=O)N(c2ccccc2)C=C1",
        "c1ccc2nc(cc2c1)N3CCN(CC3)C",
        "CCN1C=NC2=CC=CC=C21",
        "O=C(Nc1ccccc1)c2ccccc2",
    ]
    rows: List[Dict[str, str]] = []
    for i in range(max(1, n)):
        smi = seed_smiles[i % len(seed_smiles)]
        rows.append(
            {
                "candidate_id": f"cand_{i+1:06d}",
                "smiles": smi,
                "source": "generator_stub_v1",
            }
        )
    _write_rows(dst, rows)
    return len(rows)


def list_models(ctx: ToolContext, *, kind: str = "") -> Dict[str, Any]:
    catalog = ModelCatalog.load(ctx.catalog_path)
    entries = catalog.list(kind=kind or None)
    return {
        "count": len(entries),
        "models": [
            {
                "id": e.id,
                "kind": e.kind,
                "backend": e.backend,
                "task_types": e.task_types,
                "runtime_profile": e.runtime_profile,
            }
            for e in entries
        ],
    }


def search_dataset(ctx: ToolContext, *, preferences: List[str]) -> Dict[str, Any]:
    ext_root = _resolve_external_workspace_root(ctx)

    available = []
    if ext_root is not None:
        if (ext_root / "db" / "oled_tadf.sqlite").exists():
            available.append("master_database")
        if (ext_root / "reports" / "training_ready_subset_v1_2026-03-28.csv").exists():
            available.append("training_ready_subset_v1")
        if (ext_root / "reports" / "unimol_lambda_em_dataset_v1_2026-03-28.csv").exists():
            available.append("unimol_lambda_em_dataset_v1")
        if (ext_root / "artifacts" / "server_sync" / "reinvent4_runs").exists():
            available.append("reinvent4_runs")

    # keep compatibility labels
    available.extend([x for x in ["subsidiary_database"] if x not in available])

    matched = [x for x in preferences if x in available]
    if not matched:
        matched = [available[0]] if available else ["training_ready_subset_v1"]

    ctx.state["selected_datasets"] = matched
    return {"selected": matched, "available": available}


def train_predictor(ctx: ToolContext, *, predictor_id: str, targets: List[str], target_specs: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    adapter_cmd = _resolve_tool_adapter_cmd(
        ctx,
        env_prefix="OLED_AGENT_TRAIN",
        model_id=predictor_id,
        tool_name="train_predictor",
    )
    if adapter_cmd:
        timeout_sec = _tool_timeout("OLED_AGENT_TRAIN_TIMEOUT_SEC", default=3600)
        payload = {
            "workspace_root": str(ctx.workspace_root),
            "task_id": ctx.task_id,
            "predictor_id": predictor_id,
            "targets": targets,
            "target_specs": target_specs or [],
            "state": dict(ctx.state),
        }
        result = _run_json_tool_command(
            cmd=adapter_cmd,
            payload=payload,
            cwd=ctx.workspace_root,
            timeout_sec=timeout_sec,
        )
        out = dict(result)
        out.setdefault("status", "success")
        out.setdefault("adapter", "external_train_cmd")
        return out

    return {
        "predictor_id": predictor_id,
        "targets": targets,
        "target_specs": target_specs or [],
        "status": "stubbed",
        "note": "Training orchestration placeholder; connect to actual training backend adapter.",
    }


def generate_candidates(
    ctx: ToolContext,
    *,
    generator_id: str,
    max_candidates: int = 500,
    input_csv: str = "",
    constraints: Optional[Dict[str, Any]] = None,
    **generation_inputs: Any,
) -> Dict[str, Any]:
    artifact_dir = _agent_artifact_dir(ctx)
    out_csv = artifact_dir / "generated_candidates.csv"

    adapter_cmd = _resolve_tool_adapter_cmd(
        ctx,
        env_prefix="OLED_AGENT_GENERATE",
        model_id=generator_id,
        tool_name="generate_candidates",
    )
    explicit_env_cmd = bool(os.environ.get("OLED_AGENT_GENERATE_CMD", "").strip())
    bundled_reinvent4 = False
    default_reinvent4_adapter = False
    if not adapter_cmd:
        adapter_cmd = _resolve_bundled_reinvent4_generate_adapter_cmd(ctx, generator_id=generator_id)
    if adapter_cmd and not explicit_env_cmd:
        bundled_reinvent4 = "generate_candidates_reinvent4_adapter.py" in adapter_cmd
        if bundled_reinvent4:
            default_reinvent4_adapter = True
    adapter_error: Optional[Exception] = None
    if adapter_cmd:
        env_overrides: Dict[str, str] = {}
        # Keep default path runnable on machines without real REINVENT4 runtime.
        if bundled_reinvent4:
            env_overrides.update(
                _default_adapter_mode_overrides(
                    cmd=adapter_cmd,
                    mode_env="OLED_AGENT_REINVENT4_ADAPTER_MODE",
                    default_mode="smoke",
                )
            )
        timeout_sec = _tool_timeout("OLED_AGENT_GENERATE_TIMEOUT_SEC", default=3600)
        payload = {
            "workspace_root": str(ctx.workspace_root),
            "task_id": ctx.task_id,
            "generator_id": generator_id,
            "max_candidates": max_candidates,
            "input_csv": input_csv,
            "constraints": constraints or {},
            "output_csv": str(out_csv),
            "state": dict(ctx.state),
        }
        if generation_inputs:
            payload.update(generation_inputs)
        try:
            result = _run_json_tool_command(
                cmd=adapter_cmd,
                payload=payload,
                cwd=ctx.workspace_root,
                timeout_sec=timeout_sec,
                env_overrides=env_overrides,
            )
            result_out = dict(result)
            produced = str(result_out.get("output_csv") or result_out.get("output") or out_csv)
            produced_path = Path(produced)
            if not produced_path.is_absolute():
                produced_path = (ctx.workspace_root / produced_path).resolve()
            if not produced_path.exists():
                raise ToolError(f"generate command output csv not found: {produced_path}")
            count = _normalize_candidate_csv(produced_path, require_smiles=False)
            ctx.state["candidate_csv"] = str(produced_path)
            result_out.setdefault("status", "success")
            result_out.setdefault("adapter", "external_generate_cmd")
            result_out.setdefault("rows", count)
            result_out.setdefault("output", str(produced_path))
            return result_out
        except Exception as exc:
            if not default_reinvent4_adapter:
                raise
            adapter_error = exc
    local = _generate_candidates_local_fallback(
        ctx=ctx,
        generator_id=generator_id,
        max_candidates=max_candidates,
        input_csv=input_csv,
        constraints=constraints,
        out_csv=out_csv,
    )
    if adapter_error is not None:
        local["fallback_reason"] = "REINVENT4 adapter failed; fell back to local generator path"
        local["fallback_error"] = {
            "code": "reinvent4_generate_cmd_failed",
            "message": "REINVENT4 adapter failed; fell back to local generator path",
            "details": {"error": str(adapter_error)},
            "retryable": True,
        }
    return local


def _try_external_unimol_scoring(
    *,
    ctx: ToolContext,
    predictor_id: str,
    input_csv: Path,
    target_specs: List[Dict[str, Any]],
    scored_csv: Path,
) -> Dict[str, Any]:
    """Try legacy external scorer script; fallback is handled by caller."""
    ext_root = _resolve_external_workspace_root(ctx)
    if ext_root is None:
        raise ExternalScorerError(
            code="external_workspace_missing",
            message="External workspace root with scripts/ not found",
            retryable=False,
        )

    scorer = ext_root / "scripts" / "score_unimol_property_candidates.py"
    if not scorer.exists():
        raise ExternalScorerError(
            code="external_scorer_script_missing",
            message=f"External scorer not found: {scorer}",
            details={"scorer": str(scorer)},
            retryable=False,
        )

    # Avoid accidental remote calls unless explicitly enabled.
    if os.environ.get("OLED_AGENT_USE_EXTERNAL_SCORER", "0").strip() != "1":
        raise ExternalScorerError(
            code="external_scorer_disabled",
            message="External scorer is disabled. Set OLED_AGENT_USE_EXTERNAL_SCORER=1 to enable.",
            retryable=False,
        )

    timeout_sec = _env_int("OLED_AGENT_EXTERNAL_SCORER_TIMEOUT_SEC", 900, min_value=1)
    retries = _env_int("OLED_AGENT_EXTERNAL_SCORER_RETRIES", 1, min_value=0)
    backoff_sec = _env_float("OLED_AGENT_EXTERNAL_SCORER_BACKOFF_SEC", 1.5, min_value=0.0)

    # Normalize schema before external scoring to guarantee stable merge keys.
    shutil.copy2(input_csv, scored_csv)
    candidate_count = _normalize_candidate_csv(scored_csv, require_smiles=True)
    if candidate_count <= 0:
        raise ExternalScorerError(
            code="empty_candidate_set",
            message="No candidates available after schema normalization",
            details={"input_csv": str(input_csv)},
            retryable=False,
        )

    catalog = ModelCatalog.load(ctx.catalog_path)
    model = catalog.get(predictor_id)
    model_dirs = {}
    if model and isinstance(model.params, dict):
        md = model.params.get("model_dirs")
        if isinstance(md, dict):
            model_dirs = md

    spec_outputs: List[Dict[str, Any]] = []
    for spec in target_specs:
        prop = str(spec.get("name") or "").strip()
        if not prop:
            continue

        out_csv = scored_csv.parent / f"score_external_{prop}.csv"
        objective = str(spec.get("objective") or "target_window")
        target_center = float(spec.get("target_center") or (470.0 if prop == "lambda_em" else 50.0))
        sigma = float(spec.get("sigma") or (12.0 if prop == "lambda_em" else 20.0))
        model_dir = str(model_dirs.get(prop) or model_dirs.get("default") or "")

        cmd = [
            "python3",
            str(scorer),
            str(scored_csv),
            str(out_csv),
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
            cmd.extend(["--model-dir", model_dir])

        run_meta = _run_command_with_retry(
            cmd=cmd,
            cwd=ext_root,
            timeout_sec=timeout_sec,
            retries=retries,
            backoff_sec=backoff_sec,
        )
        _merge_csvs(scored_csv, out_csv, key="candidate_id")
        spec_outputs.append(
            {
                "property": prop,
                "output_csv": str(out_csv),
                "attempts": run_meta["attempts"],
            }
        )

    return {
        "adapter": "external_unimol_script",
        "scored_csv": str(scored_csv),
        "schema": {"required_columns": ["candidate_id", "smiles"], "rows": candidate_count},
        "retry_policy": {
            "timeout_sec": timeout_sec,
            "retries": retries,
            "backoff_sec": backoff_sec,
        },
        "spec_outputs": spec_outputs,
    }


def _local_fallback_scoring(
    *,
    input_csv: Path,
    scored_csv: Path,
    target_specs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    rows = _load_rows(input_csv)
    rows = _normalize_candidate_rows(rows, require_smiles=False)
    _ensure_candidate_id(rows)

    for row in rows:
        smiles = _extract_smiles(row)
        if not smiles:
            smiles = f"DUMMY_SMILES_{row['candidate_id']}"
            row["smiles"] = smiles

        if not row.get("domain_score"):
            row["domain_score"] = f"{0.15 + 0.75 * _stable_rand(smiles, 'domain'):.6f}"
        if not row.get("common_prior_score"):
            row["common_prior_score"] = f"{0.10 + 0.80 * _stable_rand(smiles, 'prior'):.6f}"

        for spec in target_specs:
            name = str(spec.get("name") or "").strip()
            if not name:
                continue
            objective = str(spec.get("objective") or "target_window")
            center = float(spec.get("target_center") or (470.0 if name == "lambda_em" else 50.0))
            sigma = float(spec.get("sigma") or (12.0 if name == "lambda_em" else 20.0))
            pred = _deterministic_target_pred(smiles, name)
            score = _objective_score(pred, objective, center, sigma)
            row[f"{name}_pred"] = f"{pred:.6f}"
            row[f"{name}_score"] = f"{score:.6f}"

    _write_rows(scored_csv, rows)
    return {
        "adapter": "local_deterministic_fallback",
        "scored_csv": str(scored_csv),
        "rows": len(rows),
    }


def score_candidates(
    ctx: ToolContext,
    *,
    predictor_id: str,
    targets: List[str],
    target_specs: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    candidate_csv = ctx.state.get("candidate_csv")
    if not candidate_csv:
        raise ToolError("No candidate_csv in context state. Call generate_candidates first.")

    input_csv = Path(candidate_csv)
    if not input_csv.exists():
        raise ToolError(f"candidate_csv does not exist: {input_csv}")

    artifact_dir = _agent_artifact_dir(ctx)
    scored_csv = artifact_dir / "scored_candidates.csv"

    specs = target_specs or [{"name": t, "objective": "target_window", "target_center": 470.0, "sigma": 12.0} for t in targets]

    adapter_cmd = _resolve_tool_adapter_cmd(
        ctx,
        env_prefix="OLED_AGENT_SCORE",
        model_id=predictor_id,
        tool_name="score_candidates",
    )
    if not adapter_cmd:
        adapter_cmd = _resolve_bundled_unimol_score_adapter_cmd(
            ctx,
            predictor_id=predictor_id,
        )
    if adapter_cmd:
        env_overrides: Dict[str, str] = {}
        # Keep default path runnable on machines without real Uni-Mol runtime.
        if "score_candidates_unimol_adapter.py" in adapter_cmd:
            env_overrides.update(
                _default_adapter_mode_overrides(
                    cmd=adapter_cmd,
                    mode_env="OLED_AGENT_UNIMOL_SCORE_MODE",
                    default_mode="smoke",
                )
            )
        timeout_sec = _tool_timeout("OLED_AGENT_SCORE_TIMEOUT_SEC", default=3600)
        payload = {
            "workspace_root": str(ctx.workspace_root),
            "task_id": ctx.task_id,
            "predictor_id": predictor_id,
            "targets": targets,
            "target_specs": specs,
            "input_csv": str(input_csv),
            "output_csv": str(scored_csv),
            "state": dict(ctx.state),
        }
        try:
            result = _run_json_tool_command(
                cmd=adapter_cmd,
                payload=payload,
                cwd=ctx.workspace_root,
                timeout_sec=timeout_sec,
                env_overrides=env_overrides,
            )
            result_out = dict(result)
            produced = str(result_out.get("output_csv") or result_out.get("output") or scored_csv)
            produced_path = Path(produced)
            if not produced_path.is_absolute():
                produced_path = (ctx.workspace_root / produced_path).resolve()
            if not produced_path.exists():
                raise ToolError(f"score command output csv not found: {produced_path}")
            _normalize_candidate_csv(produced_path, require_smiles=True)
            ctx.state["scored_csv"] = str(produced_path)
            return {
                "predictor_id": predictor_id,
                "targets": targets,
                "target_specs": specs,
                "status": "success",
                "adapter": str(result_out.get("adapter") or "external_score_cmd"),
                "output": str(produced_path),
                **result_out,
            }
        except Exception as exc:
            local = _local_fallback_scoring(input_csv=input_csv, scored_csv=scored_csv, target_specs=specs)
            local["fallback_reason"] = str(exc)
            local["fallback_error"] = {
                "code": "external_score_cmd_failed",
                "message": str(exc),
                "details": {},
                "retryable": False,
            }
            ctx.state["scored_csv"] = str(scored_csv)
            return {
                "predictor_id": predictor_id,
                "targets": targets,
                "target_specs": specs,
                "status": "success",
                "adapter": local["adapter"],
                "output": str(scored_csv),
                **local,
            }

    try:
        external = _try_external_unimol_scoring(
            ctx=ctx,
            predictor_id=predictor_id,
            input_csv=input_csv,
            target_specs=specs,
            scored_csv=scored_csv,
        )
        adapter = external["adapter"]
    except Exception as exc:
        local = _local_fallback_scoring(input_csv=input_csv, scored_csv=scored_csv, target_specs=specs)
        adapter = local["adapter"]
        local["fallback_reason"] = str(exc)
        local["fallback_error"] = _external_error_payload(exc)
        result_payload = local
    else:
        result_payload = external

    ctx.state["scored_csv"] = str(scored_csv)
    return {
        "predictor_id": predictor_id,
        "targets": targets,
        "target_specs": specs,
        "status": "success",
        "adapter": adapter,
        "output": str(scored_csv),
        **result_payload,
    }


def _build_rank_config(
    *,
    ctx: ToolContext,
    scored_csv: Path,
    topn: int,
    target_specs: List[Dict[str, Any]],
) -> Path:
    run_tag = f"agent_rank_{ctx.task_id}"
    objectives = []
    for spec in target_specs:
        name = str(spec.get("name") or "").strip()
        if not name:
            continue
        objectives.append(
            {
                "property_name": name,
                "weight": float(spec.get("weight") or 0.5),
            }
        )

    if not objectives:
        objectives = [{"property_name": "lambda_em", "weight": 0.65}, {"property_name": "plqy", "weight": 0.25}]

    cfg = {
        "run_tag": run_tag,
        "description": "agent dynamic filter/rank pipeline",
        "input_csv": _workspace_rel(scored_csv, ctx.workspace_root),
        "output_root": "runs",
        "metadata": {"source": "agent.filter_and_rank", "task_id": ctx.task_id},
        "stages": [
            {
                "name": "compose_multi_objective",
                "params": {
                    "objectives": objectives,
                    "common": {
                        "domain_weight": 0.20,
                        "prior_weight": 0.05,
                        "diversity_weight": 0.00,
                    },
                },
            },
            {
                "name": "filter_multi_objective",
                "params": {
                    "min_multi_total_score": 0.20,
                    "topn": topn,
                },
            },
            {
                "name": "export_simple_report",
                "params": {"topn": topn},
            },
        ],
    }

    cfg_path = _agent_artifact_dir(ctx) / "dynamic_rank_config.json"
    cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return cfg_path


def filter_and_rank(
    ctx: ToolContext,
    *,
    topn: int = 10,
    target_specs: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    scored_csv = ctx.state.get("scored_csv")

    if scored_csv:
        scored_path = Path(scored_csv)
        if not scored_path.exists():
            raise ToolError(f"scored_csv not found: {scored_path}")
        cfg = _build_rank_config(ctx=ctx, scored_csv=scored_path, topn=topn, target_specs=target_specs or [])
    else:
        cfg = ctx.workspace_root / "configs" / "pipelines" / "demo.json"

    manifest = run_pipeline(config_path=cfg.resolve(), workspace_root=ctx.workspace_root)
    payload = json.loads(manifest.read_text(encoding="utf-8"))

    final_output = payload.get("final_output")
    if isinstance(final_output, str) and final_output:
        ctx.state["final_output"] = final_output
    ctx.state["latest_manifest"] = str(manifest)

    return {
        "status": payload.get("status"),
        "manifest": str(manifest),
        "topn": topn,
        "final_output": final_output,
        "config": str(cfg),
    }


def make_report(ctx: ToolContext) -> Dict[str, Any]:
    final_output = ctx.state.get("final_output")
    if final_output:
        report = (ctx.workspace_root / final_output).resolve()
        if report.exists():
            return {
                "latest_run_dir": str(report.parent),
                "report": str(report),
                "source": "context_state",
            }

    runs_root = ctx.workspace_root / "runs"
    run_dirs = sorted([p for p in runs_root.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)
    if not run_dirs:
        raise ToolError("No run directories found under runs/")
    latest = run_dirs[0]
    report = latest / "06_report.md"
    if not report.exists():
        raise ToolError(f"Expected report file not found: {report}")
    return {"latest_run_dir": str(latest), "report": str(report), "source": "latest_run_scan"}


TOOL_REGISTRY = {
    "list_models": list_models,
    "search_dataset": search_dataset,
    "train_predictor": train_predictor,
    "generate_candidates": generate_candidates,
    "score_candidates": score_candidates,
    "filter_and_rank": filter_and_rank,
    "make_report": make_report,
}


def execute_tool(ctx: ToolContext, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    fn = TOOL_REGISTRY.get(name)
    if fn is None:
        raise ToolError(f"Unknown tool: {name}")
    return fn(ctx, **args)

#!/usr/bin/env python3
"""Uni-Mol train_predictor adapter with strict env checks and structured errors."""
from __future__ import annotations

import os
from pathlib import Path

from runtime_helpers import AdapterFailure, emit_failure, emit_success, read_payload, repo_root_from_script, run_argv_cmd, to_int


def _require_env(name: str) -> str:
    value = (os.environ.get(name) or "").strip()
    if not value:
        raise AdapterFailure(
            code="missing_required_env",
            message=f"missing required env var: {name}",
            details={"name": name},
        )
    return value


def main() -> int:
    try:
        payload = read_payload()
        predictor_id = str(payload.get("predictor_id") or "").strip()
        if not predictor_id:
            raise AdapterFailure(code="missing_predictor_id", message="predictor_id is required")

        mode = (os.environ.get("OLED_AGENT_UNIMOL_TRAIN_MODE") or "preflight").strip().lower()
        if mode not in ("preflight", "smoke", "real"):
            raise AdapterFailure(
                code="invalid_env_config",
                message="OLED_AGENT_UNIMOL_TRAIN_MODE must be preflight|smoke|real",
                details={"value": mode},
            )
        if mode in ("preflight", "smoke"):
            return emit_success(
                {
                    "adapter": "unimol_train_adapter_v1",
                    "predictor_id": predictor_id,
                    "mode": mode,
                    "metrics": {
                        "note": "train adapter preflight/smoke success; set OLED_AGENT_UNIMOL_TRAIN_MODE=real for real execution",
                    },
                }
            )

        repo_root = repo_root_from_script()
        train_script = (repo_root.parent / "scripts" / "train_unimol_end2end_plqy_v2_candidate_remote.py").resolve()
        if not train_script.exists():
            raise AdapterFailure(
                code="training_script_missing",
                message="workspace training script is missing",
                details={"expected": str(train_script)},
            )

        # Preflight env guards for a predictable failure mode.
        _require_env("UNIMOL_REMOTE_HOST")
        _require_env("UNIMOL_REMOTE_PY")
        _require_env("UNIMOL_REMOTE_TMP_BASE")

        timeout_sec = to_int(
            os.environ.get("OLED_AGENT_UNIMOL_TRAIN_TIMEOUT_SEC", ""),
            default=3600,
            min_value=1,
            name="OLED_AGENT_UNIMOL_TRAIN_TIMEOUT_SEC",
        )
        workspace_root = Path(str(payload.get("workspace_root") or ".")).resolve()
        run_argv_cmd(
            argv=["python3", str(train_script)],
            cwd=workspace_root,
            timeout_sec=timeout_sec,
        )

        return emit_success(
            {
                "adapter": "unimol_train_adapter_v1",
                "predictor_id": predictor_id,
                "mode": mode,
                "metrics": {
                    "note": "training command executed; inspect external logs/artifacts for detailed metrics",
                },
            }
        )
    except AdapterFailure as exc:
        return emit_failure(code=exc.code, message=exc.message, details=exc.details)
    except Exception as exc:
        return emit_failure(code="unexpected_adapter_error", message=str(exc))


if __name__ == "__main__":
    raise SystemExit(main())

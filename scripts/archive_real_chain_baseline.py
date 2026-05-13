#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _git_sha(workspace_root: Path) -> str:
    try:
        cp = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=workspace_root,
            check=False,
            capture_output=True,
            text=True,
        )
        if cp.returncode == 0:
            return (cp.stdout or "").strip()
    except Exception:
        pass
    return ""


def _resolve_path(raw: str, workspace_root: Path) -> Path:
    p = Path(str(raw or "").strip())
    if p.is_absolute():
        return p.resolve()
    return (workspace_root / p).resolve()


def _dest_path_for_source(src: Path, *, workspace_root: Path, out_dir: Path) -> Path:
    try:
        rel = src.relative_to(workspace_root)
        return (out_dir / "files" / rel).resolve()
    except ValueError:
        sanitized = src.as_posix().replace("/", "__").replace(":", "_")
        return (out_dir / "external" / sanitized).resolve()


def _collect_entries(*, baseline: Dict[str, Any], workspace_root: Path, baseline_summary_path: Path) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = [
        {
            "label": "baseline_summary",
            "source": str(baseline_summary_path),
            "required": True,
        }
    ]
    runs = baseline.get("runs", [])
    if not isinstance(runs, list):
        return entries

    for idx, item in enumerate(runs, start=1):
        if not isinstance(item, dict):
            continue
        task_id = str(item.get("task_id") or f"run_{idx}")
        for key in ("strict_summary", "result_json", "release_evidence_json"):
            raw = str(item.get(key) or "").strip()
            entries.append(
                {
                    "label": f"{task_id}:{key}",
                    "source": str(_resolve_path(raw, workspace_root)) if raw else "",
                    "required": True,
                }
            )
        release_json_raw = str(item.get("release_evidence_json") or "").strip()
        if release_json_raw:
            release_json_path = _resolve_path(release_json_raw, workspace_root)
            entries.append(
                {
                    "label": f"{task_id}:release_evidence_md",
                    "source": str(release_json_path.with_suffix(".md")),
                    "required": False,
                }
            )

        result_raw = str(item.get("result_json") or "").strip()
        if not result_raw:
            continue
        result_path = _resolve_path(result_raw, workspace_root)
        if not result_path.exists():
            continue
        try:
            result_payload = _load_json(result_path)
        except Exception:
            continue
        for r_key in ("plan_path", "execution_path", "decision_summary_path", "task_state_path", "tool_state_path"):
            r_raw = str(result_payload.get(r_key) or "").strip()
            if not r_raw:
                continue
            entries.append(
                {
                    "label": f"{task_id}:{r_key}",
                    "source": str(_resolve_path(r_raw, workspace_root)),
                    "required": False,
                }
            )
    return entries


def _write_manifest_md(manifest: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Baseline Archive Manifest")
    lines.append("")
    lines.append(f"- generated_at: `{manifest.get('generated_at')}`")
    lines.append(f"- base_task_id: `{manifest.get('base_task_id')}`")
    lines.append(f"- status: `{manifest.get('status')}`")
    lines.append(f"- git_sha: `{manifest.get('git_sha')}`")
    lines.append(f"- copied_count: `{manifest.get('copied_count', 0)}`")
    lines.append(f"- missing_required_count: `{manifest.get('missing_required_count', 0)}`")
    tar_gz_path = str(manifest.get("tar_gz_path") or "").strip()
    if tar_gz_path:
        lines.append(f"- tar_gz_path: `{tar_gz_path}`")
    lines.append("")
    lines.append("## Required Missing")
    missing = manifest.get("missing_required", [])
    if isinstance(missing, list) and missing:
        for item in missing:
            if isinstance(item, dict):
                lines.append(f"- `{item.get('label')}`: `{item.get('source')}`")
    else:
        lines.append("- none")
    lines.append("")
    lines.append("## Copied Files")
    copied = manifest.get("copied", [])
    if isinstance(copied, list):
        for item in copied:
            if isinstance(item, dict):
                lines.append(f"- `{item.get('label')}` -> `{item.get('dest')}`")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Archive strict real-chain baseline artifacts into one bundle")
    p.add_argument("--workspace-root", default=".", help="Workspace root")
    p.add_argument("--base-task-id", required=True, help="Baseline task id (without _r1/_r2/_r3)")
    p.add_argument(
        "--baseline-summary",
        default="",
        help="Path to baseline_summary.json (default: runs/agent/<base_task_id>/baseline_summary.json)",
    )
    p.add_argument(
        "--out-dir",
        default="",
        help="Output directory for archive bundle (default: runs/archive/<base_task_id>)",
    )
    p.add_argument(
        "--tar-gz",
        action="store_true",
        help="Also write a .tar.gz package from out-dir",
    )
    p.add_argument(
        "--tar-gz-path",
        default="",
        help="Output path for .tar.gz package (default: <out-dir>.tar.gz)",
    )
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing out-dir")
    p.add_argument("--allow-nonpass", action="store_true", help="Do not fail when baseline summary status is not pass")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    workspace_root = Path(args.workspace_root).resolve()
    base_task_id = str(args.base_task_id or "").strip()
    if not base_task_id:
        print("[FAIL] base-task-id is required")
        return 2

    baseline_summary_path = (
        _resolve_path(args.baseline_summary, workspace_root)
        if str(args.baseline_summary).strip()
        else (workspace_root / "runs" / "agent" / base_task_id / "baseline_summary.json").resolve()
    )
    if not baseline_summary_path.exists():
        print(f"[FAIL] baseline summary not found: {baseline_summary_path}")
        return 1

    baseline = _load_json(baseline_summary_path)
    status = str(baseline.get("status") or "")
    if not args.allow_nonpass and status != "pass":
        print(f"[FAIL] baseline summary status is not pass: {status}")
        return 1

    out_dir = (
        _resolve_path(args.out_dir, workspace_root)
        if str(args.out_dir).strip()
        else (workspace_root / "runs" / "archive" / base_task_id).resolve()
    )
    if out_dir.exists():
        if args.overwrite:
            shutil.rmtree(out_dir)
        else:
            print(f"[FAIL] output directory already exists: {out_dir} (use --overwrite)")
            return 1
    out_dir.mkdir(parents=True, exist_ok=True)

    entries = _collect_entries(
        baseline=baseline,
        workspace_root=workspace_root,
        baseline_summary_path=baseline_summary_path,
    )

    copied: List[Dict[str, Any]] = []
    missing_required: List[Dict[str, Any]] = []
    optional_missing: List[Dict[str, Any]] = []

    for item in entries:
        label = str(item.get("label") or "")
        required = bool(item.get("required", False))
        src_raw = str(item.get("source") or "").strip()
        if not src_raw:
            miss = {"label": label, "source": src_raw}
            if required:
                missing_required.append(miss)
            else:
                optional_missing.append(miss)
            continue
        src = Path(src_raw)
        if not src.exists():
            miss = {"label": label, "source": str(src)}
            if required:
                missing_required.append(miss)
            else:
                optional_missing.append(miss)
            continue
        dest = _dest_path_for_source(src, workspace_root=workspace_root, out_dir=out_dir)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        copied.append({"label": label, "source": str(src), "dest": str(dest)})

    manifest = {
        "generated_at": _now_iso(),
        "base_task_id": base_task_id,
        "status": "pass" if not missing_required else "fail",
        "git_sha": _git_sha(workspace_root),
        "workspace_root": str(workspace_root),
        "baseline_summary_path": str(baseline_summary_path),
        "out_dir": str(out_dir),
        "copied_count": len(copied),
        "missing_required_count": len(missing_required),
        "optional_missing_count": len(optional_missing),
        "copied": copied,
        "missing_required": missing_required,
        "optional_missing": optional_missing,
    }
    manifest_json = out_dir / "archive_manifest.json"
    manifest_md = out_dir / "archive_manifest.md"
    manifest_json.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest_md.write_text(_write_manifest_md(manifest), encoding="utf-8")

    tar_gz_path = None
    if bool(args.tar_gz):
        tar_gz_path = (
            _resolve_path(args.tar_gz_path, workspace_root)
            if str(args.tar_gz_path).strip()
            else out_dir.with_suffix(".tar.gz")
        )
        tar_gz_path.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tar_gz_path, mode="w:gz") as tf:
            tf.add(out_dir, arcname=out_dir.name)
        manifest["tar_gz_path"] = str(tar_gz_path)
        manifest_json.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        manifest_md.write_text(_write_manifest_md(manifest), encoding="utf-8")

    print(f"ARCHIVE_DIR={out_dir}")
    print(f"ARCHIVE_MANIFEST_JSON={manifest_json}")
    print(f"ARCHIVE_MANIFEST_MD={manifest_md}")
    if tar_gz_path is not None:
        print(f"ARCHIVE_TAR_GZ={tar_gz_path}")
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "base_task_id": base_task_id,
                "copied_count": len(copied),
                "missing_required_count": len(missing_required),
                "tar_gz_path": str(tar_gz_path) if tar_gz_path is not None else "",
            },
            ensure_ascii=False,
        )
    )
    return 0 if not missing_required else 1


if __name__ == "__main__":
    raise SystemExit(main())

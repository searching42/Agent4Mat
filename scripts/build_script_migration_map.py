#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


def _classify_script(name: str) -> Dict[str, str]:
    n = name.lower()

    if "train_unimol" in n:
        return {
            "module": "train_predictor",
            "adapter": "scripts/adapters/train_predictor_unimol_adapter.py",
            "integration_status": "integrated",
        }
    if "score_unimol" in n or "score_candidates_unimol" in n:
        return {
            "module": "score_candidates",
            "adapter": "scripts/adapters/score_candidates_unimol_adapter.py",
            "integration_status": "integrated",
        }
    if "run_reinvent4" in n or "filter_reinvent4" in n or "rank_reinvent4" in n:
        return {
            "module": "generate_candidates",
            "adapter": "scripts/adapters/generate_candidates_reinvent4_adapter.py",
            "integration_status": "partially_integrated",
        }
    if "mineru" in n:
        return {
            "module": "generate_candidates",
            "adapter": "scripts/adapters/generate_candidates_mineru_adapter.py",
            "integration_status": "partially_integrated",
        }
    if "molscribe" in n:
        return {
            "module": "generate_candidates",
            "adapter": "scripts/adapters/generate_candidates_molscribe_adapter.py",
            "integration_status": "partially_integrated",
        }
    return {
        "module": "research_or_analysis",
        "adapter": "",
        "integration_status": "not_in_scope",
    }


def build_map(workspace_scripts_root: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for path in sorted(workspace_scripts_root.glob("*.py")):
        cls = _classify_script(path.name)
        rows.append(
            {
                "script": f"workspace/scripts/{path.name}",
                "script_name": path.name,
                "module": cls["module"],
                "adapter": cls["adapter"],
                "integration_status": cls["integration_status"],
            }
        )
    return rows


def main() -> int:
    p = argparse.ArgumentParser(description="Build script migration map from workspace/scripts")
    p.add_argument("--workspace-scripts-root", default="../scripts", help="Path to workspace/scripts")
    p.add_argument("--out", default="docs/script_migration_map.json", help="Output JSON path")
    args = p.parse_args()

    scripts_root = Path(args.workspace_scripts_root).resolve()
    out_path = Path(args.out).resolve()

    if not scripts_root.exists():
        raise SystemExit(f"workspace scripts root not found: {scripts_root}")

    rows = build_map(scripts_root)
    payload = {
        "workspace_scripts_root": "workspace/scripts",
        "total": len(rows),
        "rows": rows,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({"status": "pass", "out": str(out_path), "total": len(rows)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""JSON-in/JSON-out adapter template for `generate_candidates`.

Expected stdin payload keys:
- `output_csv` (required)
- `max_candidates` (optional)
- `generator_id`, `constraints`, `task_id`, ...

Expected stdout payload keys:
- `status`: "success"
- `adapter`: adapter name
- `output_csv`: produced CSV path
- `rows`: number of rows
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

_SEED_SMILES = [
    "c1ccc2ccccc2c1",
    "c1ncccc1",
    "CCOC(=O)N1CCN(CC1)C",
    "CN1C=NC2=CC=CC=C21",
    "c1ccc(cc1)N(c2ccccc2)c3ccccc3",
]


def _int_or_default(value: object, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def main() -> int:
    payload = json.load(sys.stdin)
    output_csv = str(payload.get("output_csv") or "").strip()
    if not output_csv:
        print(json.dumps({"status": "failed", "error": "missing output_csv"}, ensure_ascii=False))
        return 2

    max_candidates = max(1, _int_or_default(payload.get("max_candidates"), 5))
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for i in range(max_candidates):
        rows.append(
            {
                "candidate_id": f"cand_{i+1:06d}",
                "smiles": _SEED_SMILES[i % len(_SEED_SMILES)],
                "source": "adapter_template_generate",
            }
        )

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(
        json.dumps(
            {
                "status": "success",
                "adapter": "template_generate_cmd",
                "output_csv": str(out_path),
                "rows": len(rows),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

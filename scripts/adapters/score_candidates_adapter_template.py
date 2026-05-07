#!/usr/bin/env python3
"""JSON-in/JSON-out adapter template for `score_candidates`.

Expected stdin payload keys:
- `input_csv` (required)
- `output_csv` (required)
- `predictor_id`, `targets`, `target_specs`, ...
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


def main() -> int:
    payload = json.load(sys.stdin)
    input_csv = str(payload.get("input_csv") or "").strip()
    output_csv = str(payload.get("output_csv") or "").strip()
    if not input_csv or not output_csv:
        print(json.dumps({"status": "failed", "error": "missing input_csv/output_csv"}, ensure_ascii=False))
        return 2

    in_path = Path(input_csv)
    out_path = Path(output_csv)
    rows = list(csv.DictReader(in_path.open("r", encoding="utf-8")))
    for row in rows:
        row.setdefault("smiles", row.get("SMILES", ""))
        row["domain_score"] = row.get("domain_score") or "0.500000"
        row["common_prior_score"] = row.get("common_prior_score") or "0.500000"
        row["plqy_pred"] = row.get("plqy_pred") or "0.600000"
        row["plqy_score"] = row.get("plqy_score") or "0.600000"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["candidate_id", "smiles"])
        writer.writeheader()
        writer.writerows(rows)

    print(
        json.dumps(
            {
                "status": "success",
                "adapter": "template_score_cmd",
                "output_csv": str(out_path),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

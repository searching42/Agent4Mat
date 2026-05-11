#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description="Stub Uni-Mol scorer for adapter real-mode contract checks")
    ap.add_argument("input_csv")
    ap.add_argument("output_csv")
    ap.add_argument("--property-name", default="plqy")
    ap.add_argument("--objective-type", default="maximize")
    ap.add_argument("--target-center", default="0.6")
    ap.add_argument("--sigma", default="0.2")
    ap.add_argument("--model-dir", default="")
    args = ap.parse_args()

    in_path = Path(args.input_csv)
    out_path = Path(args.output_csv)
    if not in_path.exists():
        raise SystemExit(f"missing input csv: {in_path}")

    with in_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit("input csv has no rows")

    for row in rows:
        row.setdefault("domain_score", "0.500000")
        row.setdefault("common_prior_score", "0.500000")
        row[f"{args.property_name}_pred"] = "0.610000"
        row[f"{args.property_name}_score"] = "0.610000"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

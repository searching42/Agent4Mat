from __future__ import annotations

import argparse
from pathlib import Path


def _validate_profile_file(path: Path) -> list[str]:
    issues: list[str] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for idx, line in enumerate(lines, start=1):
        s = line.strip()
        if not s or s.startswith("#") or s.startswith("-r "):
            continue
        if "==" not in s:
            issues.append(f"{path.name}:{idx}: dependency must be pinned with '==': {s}")
    return issues


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate pinned requirements profiles.")
    parser.add_argument("--requirements-dir", default="requirements", help="Directory containing *.in profile files")
    args = parser.parse_args()

    req_dir = Path(args.requirements_dir).resolve()
    targets = ["base.in", "cpu.in", "gpu.in", "dev.in"]
    issues: list[str] = []

    for name in targets:
        p = req_dir / name
        if not p.exists():
            issues.append(f"missing file: {p}")
            continue
        issues.extend(_validate_profile_file(p))

    if issues:
        for issue in issues:
            print(f"[FAIL] {issue}")
        return 1

    print("[PASS] requirements profiles are pinned and complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


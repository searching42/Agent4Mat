from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict

from setuptools import setup


def _load_project_metadata(pyproject_path: Path) -> Dict[str, str]:
    try:
        import tomllib  # type: ignore[attr-defined]
    except ModuleNotFoundError:
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ModuleNotFoundError:
            return _fallback_parse_project(pyproject_path.read_text(encoding="utf-8"))

    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    project = data.get("project", {})
    if not isinstance(project, dict):
        return {}
    return {
        "name": str(project.get("name", "")),
        "version": str(project.get("version", "")),
    }


def _fallback_parse_project(pyproject_text: str) -> Dict[str, str]:
    # Minimal legacy fallback: parse only name/version from [project].
    section = re.search(r"(?ms)^\[project\]\s*(.*?)^\[", pyproject_text + "\n[__end__]\n")
    if not section:
        return {}
    body = section.group(1)
    out: Dict[str, str] = {}
    name = re.search(r'(?m)^\s*name\s*=\s*"([^"]+)"\s*$', body)
    version = re.search(r'(?m)^\s*version\s*=\s*"([^"]+)"\s*$', body)
    if name:
        out["name"] = name.group(1).strip()
    if version:
        out["version"] = version.group(1).strip()
    return out


def _handle_legacy_metadata_query(argv: list[str], metadata: Dict[str, str]) -> bool:
    # Keep compatibility for `python setup.py --name/--version` without
    # re-declaring project metadata in setup() arguments.
    query_map = {"--name": "name", "--version": "version"}
    queried = [query_map[arg] for arg in argv if arg in query_map]
    if not queried:
        return False
    for key in queried:
        print(metadata.get(key, ""))
    return True


if __name__ == "__main__":
    project_metadata = _load_project_metadata(Path(__file__).with_name("pyproject.toml"))
    if _handle_legacy_metadata_query(sys.argv[1:], project_metadata):
        raise SystemExit(0)
    # Compatibility shim:
    # Metadata source of truth stays in pyproject.toml.
    # Keep setup.py as a minimal legacy entrypoint only.
    setup()

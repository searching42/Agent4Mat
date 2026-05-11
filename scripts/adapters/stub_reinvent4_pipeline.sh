#!/usr/bin/env bash
set -euo pipefail

SOURCE_CSV="${1:-}"
RUN_TAG="${2:-stub_run}"

if [ -z "$SOURCE_CSV" ]; then
  echo "missing source csv" >&2
  exit 2
fi

if [ ! -f "$SOURCE_CSV" ]; then
  echo "source csv not found: $SOURCE_CSV" >&2
  exit 2
fi

OUT="${OLED_AGENT_REINVENT4_RANKREADY_CSV:?OLED_AGENT_REINVENT4_RANKREADY_CSV is required}"
mkdir -p "$(dirname "$OUT")"
cp "$SOURCE_CSV" "$OUT"
echo "stub pipeline completed: $RUN_TAG" >&2

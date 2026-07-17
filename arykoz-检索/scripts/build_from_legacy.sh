#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LEGACY_INDEX="${1:?usage: build_from_legacy.sh LEGACY_INDEX [OUTPUT_INDEX]}"
OUTPUT_INDEX="${2:-$ROOT/b_index_v2.pkl}"
python "$ROOT/migrate_legacy_index_v2.py" "$LEGACY_INDEX" --output "$OUTPUT_INDEX" --export-dir "$ROOT/index_v2_export"
python "$ROOT/fact_index_builder.py" "$OUTPUT_INDEX" --output "$OUTPUT_INDEX.fact_index.pkl" --force

#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUTPUT_INDEX="${AFAC_INDEX_V2_PKL:-$ROOT/b_index_v2.pkl}"
python "$ROOT/build_index_v2.py" \
  --extracted-dir "${AFAC_EXTRACTED_DIR:?set AFAC_EXTRACTED_DIR}" \
  --extracted-dir "${AFAC_REGULATORY_V2_DIR:?set AFAC_REGULATORY_V2_DIR}" \
  --output "$OUTPUT_INDEX" --export-dir "$ROOT/index_v2_export"
python "$ROOT/fact_index_builder.py" "$OUTPUT_INDEX" --output "$OUTPUT_INDEX.fact_index.pkl" --force

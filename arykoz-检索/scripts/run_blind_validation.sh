#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
QUESTIONS="${1:?usage: run_blind_validation.sh A_QUESTIONS_JSONL}"
INDEX="${AFAC_INDEX_V2_PKL:-$ROOT/b_index_v2.pkl}"
mkdir -p "$ROOT/validation"
for domain in financial_contracts financial_reports insurance regulatory research; do
  for start in 0 5 10 15; do
    python "$ROOT/evaluation/evaluate_blind_retrieval.py" "$QUESTIONS" --index "$INDEX" --domain "$domain" --start "$start" --limit 5 --output "$ROOT/validation/blind_${domain}_${start}_5.json"
  done
done
python "$ROOT/evaluation/aggregate_blind_reports.py" --validation-dir "$ROOT/validation"

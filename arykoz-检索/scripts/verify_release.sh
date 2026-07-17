#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
python -m compileall -q .
python -m unittest discover -s tests -v
python -m unittest -v integration.test_reasoning_engine
python evaluation/verify_index_v2.py "${AFAC_INDEX_V2_PKL:-$ROOT/b_index_v2.pkl}"
if grep -R -E '(fc|fin|ins|reg|res)_a_[0-9]+' --include='*.py' . | grep -v '/tests/'; then
  echo 'qid hardcode detected' >&2; exit 1
fi
if grep -R -E '100分答案|ground_truth|answer_100|04_100分答案' --include='*.py' .; then
  echo 'answer leakage token detected' >&2; exit 1
fi

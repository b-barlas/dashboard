#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "[1/4] Python runtime check"
python --version
python - <<'PY'
import sys
if sys.version_info < (3, 12):
    raise SystemExit("Python 3.12+ is required.")
if sys.version_info >= (3, 14):
    raise SystemExit("Python 3.14+ is not validated yet. Use 3.12.x or 3.13.x.")
print("Runtime gate OK (3.12.x-3.13.x).")
PY

echo "[2/4] Compile gate"
python -m compileall -q core tabs ui crypto_meta2.py

echo "[3/4] Smoke/contract gate"
python -m pytest -q \
  tests/test_market_decision_contract.py \
  tests/test_market_table_price_delta_contract.py \
  tests/test_market_coin_column_contract.py \
  tests/test_scalp_gate_contract.py \
  tests/test_guide_content_contract.py \
  tests/test_tabs_ctx_access_contract.py

echo "[4/4] Optional full suite"
if [[ "${1:-}" == "--full" ]]; then
  python -m pytest -q
else
  echo "Skipped full suite (run './scripts/quality_gate.sh --full' before release)."
fi

echo "✅ Quality gate passed."

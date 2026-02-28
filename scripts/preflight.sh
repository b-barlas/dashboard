#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "[1/4] Python version"
python --version
python - << 'PY'
import sys
if sys.version_info < (3, 12):
    raise SystemExit("Python 3.12+ is required.")
if sys.version_info >= (3, 14):
    raise SystemExit("Python 3.14+ is not validated yet. Use Python 3.12.x or 3.13.x.")
print("Python runtime check: OK (supported range: 3.12.x - 3.13.x)")
PY

echo "[2/4] Compile check"
python -m compileall -q core tabs ui crypto_meta2.py

echo "[3/4] Test suite"
python -m pytest -q

echo "[4/4] Dependency snapshot"
python - << 'PY'
import importlib.metadata as m
pkgs = ['streamlit','pandas','numpy','requests','plotly','ccxt','ta','scikit-learn']
for p in pkgs:
    try:
        print(f"{p}=={m.version(p)}")
    except Exception:
        print(f"{p}=<not-installed>")
PY

echo "✅ Preflight passed. Ready to deploy."

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "[1/2] Quality gate (smoke + full)"
"$ROOT_DIR/scripts/quality_gate.sh" --full

echo "[2/2] Dependency snapshot"
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

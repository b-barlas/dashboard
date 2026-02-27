# Crypto Market Intelligence Hub

A Streamlit-based cryptocurrency analytics dashboard with modular architecture (`core/`, `tabs/`, `ui/`).

## Requirements

- Python 3.9+
- Internet access for market data APIs

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run (Streamlit)

```bash
streamlit run crypto_meta2.py
```

By default, Streamlit serves the app locally (typically `http://localhost:8501`).

## Deploy Readiness

Run preflight before every deploy:

```bash
./scripts/preflight.sh
```

This runs:
- compile checks
- full test suite
- dependency snapshot output

## Production Smoke Test (5 minutes)

After deploy/reboot, validate:

1. `Market` tab: scan loads with non-empty rows.
2. `Market` tab: timeframe switch still returns candidates and action-ready rows.
3. `Position` tab: changing coin updates current/entry context correctly.
4. `Backtest` tab: at least one symbol+timeframe can produce trades with a sensible threshold.
5. Cache warnings: if any tab shows cached snapshot warning, do not execute directly without fresh live confirmation.
## UK Exchange Data Policy

The exchange fallback list is intentionally restricted to:

- Kraken
- Coinbase
- Bitstamp

This is configured in `core/services.py` and is designed to avoid relying on commonly restricted exchange sources.

## CI Quality Gates

GitHub Actions runs on `push` and `pull_request` to `main`:

- `compileall` syntax check
- `ruff` critical checks (`E9,F63,F7,F82`)
- `mypy` on core engine contracts
- full `pytest` suite

## Tests (local)

Run the test suite:

```bash
python -m pytest -q
```

# Crypto Command Center

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

## UK Exchange Data Policy

The exchange fallback list is intentionally restricted to:

- Kraken
- Coinbase
- Bitstamp

This is configured in `core/services.py` and is designed to avoid relying on commonly restricted exchange sources.

## Tests

Run the test suite:

```bash
python3 -m unittest tests/test_ui_styles.py tests/test_ui_helpers.py tests/test_tabs_contract.py tests/test_core_engines.py
```

Some tests may be skipped in environments where optional dependencies are not installed.

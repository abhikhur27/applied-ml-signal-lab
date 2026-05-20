# Applied ML Signal Lab

Applied machine learning sandbox for market-regime classification with reproducible training runs.

## What this project does

- Builds a labeled time-series dataset from either:
- synthetic regime-switching prices
- or user-provided OHLCV CSV data
- Engineers features commonly used in quant workflows:
- log returns
- rolling volatility
- momentum windows
- moving-average spread
- RSI proxy
- Trains a `RandomForestClassifier` using a strict chronological split (no random shuffle leakage).
- Produces evaluation artifacts:
- classification report
- confusion matrix (CSV)
- feature importance table
- markdown run report
- serialized model (`joblib`)

## Why this is useful

This repo is meant to be a practical base for:

- walk-forward strategy research
- signal diagnostics before backtesting
- translating raw market data into ML-ready features without notebook sprawl

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m src.train --use-synthetic
```

Artifacts are written to `artifacts/`.

## Use your own data

Input CSV should include:

- `date`
- `close`

Optional columns (`open`, `high`, `low`, `volume`) are accepted but not required by the baseline pipeline.

Run:

```bash
python -m src.train --csv path/to/ohlcv.csv
```

## Next steps

- add walk-forward evaluation windows
- add probability calibration and threshold tuning
- compare tree models against temporal neural baselines

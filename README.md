# Applied ML Signal Lab

Applied machine learning sandbox for market-regime classification with reproducible training runs.

## What this project does

- Builds a labeled time-series dataset from either:
  - synthetic regime-switching prices
  - user-provided OHLCV CSV data
- Input validation now hard-fails on duplicate dates, non-numeric closes, or non-positive close values before feature engineering.
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
  - holdout predictions with target/prediction labels
  - model summary JSON for downstream scripting
  - markdown run report
  - walk-forward metrics + markdown summary by default
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

Key outputs now include:

- `test_predictions.csv`: holdout rows with actual vs predicted regime labels
- `test_predictions.csv` now also includes per-class probabilities, confidence, and margin to runner-up
- `model_summary.json`: compact machine-readable accuracy + feature summary
- `model_summary.json` now includes confidence posture and prediction mix
- `walk_forward_metrics.csv`: expanding-window accuracy by evaluation slice

If you want the fastest baseline-only pass, skip walk-forward explicitly:

```bash
python -m src.train --use-synthetic --skip-walk-forward
```

To make synthetic experiments reproducible across runs and tune dataset size:

```bash
python -m src.train --use-synthetic --synthetic-seed 7 --synthetic-points 3000 --model-seed 42
```

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

- add probability calibration and threshold tuning
- compare tree models against temporal neural baselines

## Portfolio Repro Checklist

Use this sequence before publishing a run artifact:

1. Baseline synthetic run:
`python -m src.train --use-synthetic --artifacts artifacts/baseline`
2. Baseline + walk-forward:
`python -m src.train --use-synthetic --artifacts artifacts/walkforward`
3. Confirm both directories include:
- `model_summary.json`
- `test_predictions.csv`
- `walk_forward_metrics.csv` (for non-skip runs)
4. In notes, distinguish single holdout accuracy from walk-forward average accuracy.

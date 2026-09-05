# Applied ML Signal Lab

Applied machine learning workflow for market-regime classification with reproducible training runs.

## What this project does

- Builds a labeled time-series dataset from synthetic regime-switching prices or user-provided OHLCV CSV data.
- Input validation now hard-fails on duplicate dates, non-numeric closes, or non-positive close values before feature engineering.
- Engineers features commonly used in quant workflows:
  - log returns
  - rolling volatility
  - momentum windows
  - moving-average spread
  - RSI proxy
- Supports auditable one- or multi-session forward-return labels with explicit label end dates.
- Trains a `RandomForestClassifier` using purge-aware chronological splits so training labels never cross validation, calibration, holdout, or walk-forward boundaries.
- Fits a standardized, class-balanced logistic challenger on the same leakage-safe rows so the forest is compared with a simpler model family, not only naive labels.
- Produces evaluation artifacts:
  - classification report
  - confusion matrix (CSV)
  - feature importance table
  - confidence bucket and calibration tables
  - benchmark comparison versus regularized logistic, horizon-aware persistence, and majority-class baselines using accuracy, balanced accuracy, and macro-F1
  - holdout predictions with target/prediction labels, forward returns, and label end dates
  - model summary JSON for downstream scripting
  - markdown run report
- Runs walk-forward evaluation by default, with the same horizon purge and observable-label rules.
- Supports optional threshold sweeps, feature ablation, chronological model search, and audited probability calibration.
- Promotes a searched model configuration only if it beats both naive baselines on validation.
- Applies probability calibration only when it improves multiclass Brier score on a separate chronological audit slice without collapsing prediction diversity.
- Serializes the fitted model with `joblib`.

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

- `test_predictions.csv`: auditable holdout rows with label dates, forward returns, actual/predicted regimes, per-class probabilities, confidence, margin, and raw-vs-calibrated confidence columns
- `model_summary.json`: compact machine-readable accuracy + feature summary
- `benchmark_accuracy.csv`: accuracy, balanced accuracy, macro-F1, prediction mix, and forest-relative deltas for the forest, regularized-logistic challenger, persistence, and majority baselines
- `model_summary.json` now includes confidence posture and prediction mix
- `confidence_buckets.csv`: row counts and accuracy by confidence band so weak predictions are easier to triage
- `confidence_calibration.csv`: decile-style confidence calibration table with empirical accuracy and confidence gap
- `probability_comparison.csv`: per-class Brier comparison between raw forest probabilities and calibrated probabilities
- `class_balance.csv`: dataset label mix so class skew is visible before you trust the accuracy
- `feature_drift.csv`: train-vs-test feature shift table so covariate drift is visible before you trust a holdout win
- `walk_forward_metrics.csv`: expanding-window accuracy by evaluation slice
- `walk_forward_metrics.csv` includes all three metrics for the forest, regularized-logistic challenger, persistence, and majority baselines so each window exposes both overall accuracy and class coverage

If you want the fastest baseline-only pass, skip walk-forward explicitly:

```bash
python -m src.train --use-synthetic --skip-walk-forward
```

To make synthetic experiments reproducible across runs and tune dataset size:

```bash
python -m src.train --use-synthetic --synthetic-seed 7 --synthetic-points 3000 --model-seed 42
```

If you want probability calibration on top of the raw forest output, enable it on the trailing slice of the chronological training window:

```bash
python -m src.train --use-synthetic --calibrate-probabilities
```

You can also swap calibration method when you have enough rows for a less parametric fit:

```bash
python -m src.train --use-synthetic --calibrate-probabilities --calibration-method isotonic
```

To compare multiple symmetric bull/bear label thresholds before settling on one:

```bash
python -m src.train --use-synthetic --threshold-sweep --threshold-sweep-values 0.002,0.003,0.004,0.005
```

That writes `threshold_sweep.csv` and `threshold_sweep_report.md` alongside the normal training artifacts.

To measure which engineered features the holdout result depends on most:

```bash
python -m src.train --use-synthetic --feature-ablation
```

That writes `feature_ablation.csv` and `feature_ablation_report.md` beside the normal training artifacts.

To justify the random-forest choice instead of relying on one fixed config, run a chronological search across several maintained candidates:

```bash
python -m src.train --use-synthetic --model-search
```

That writes `model_search.csv` and `model_search_report.md`, then carries the selected configuration into the final holdout evaluation and optional probability calibration.

## Frozen real-data benchmark

The repo includes three 3,327-row European Central Bank reference-rate fixtures covering 2012 through 2024: EUR/USD, EUR/GBP, and EUR/JPY. Each real financial series has checked-in provenance, a checksum frozen in the benchmark contract, and an official-data updater—not synthetic price generation presented as market evidence.

Run the full credibility gate with:

```bash
python -m src.benchmark --artifacts artifacts/ecb-benchmark
```

The contract checks fixture integrity, chronological holdout behavior, naive baseline advantages, calibration safety, and six walk-forward windows per instrument. It also applies a model-family promotion gate: the regularized-logistic challenger must improve both balanced accuracy and macro-F1 across at least two-thirds of holdouts and walk-forward regimes, show positive mean gains on at least two instruments, clear one-point aggregate gains in both metrics, preserve all three prediction classes on every instrument, and avoid a per-instrument mean regression worse than one point. A single favorable holdout cannot replace the maintained model.

The current frozen result retains the random forest. Logistic wins both balanced holdout metrics on all three pairs, but wins both metrics in only 3 of 18 walk-forward regimes; its cross-regime mean deltas are -0.0098 balanced accuracy and -0.0239 macro-F1.

Suite artifacts include `benchmark_suite_results.json`, `benchmark_suite_report.md`, and `model_family_promotion.csv`, plus the normal training and walk-forward artifacts under one directory per currency pair. This benchmark is a regression contract for honest behavior, not evidence of a tradable signal. See [`data/README.md`](data/README.md) for source and reuse details and [`benchmarks/ecb_fx_contract.json`](benchmarks/ecb_fx_contract.json) for the fixed expectations and promotion policy.

## Leakage-safe forecast horizons

The default target remains the next-session return. For a less noise-dominated regime benchmark, define both the forecast horizon and thresholds explicitly:

```bash
python -m src.train --use-synthetic --synthetic-points 3000 --label-horizon 10 --bull-threshold 0.012 --bear-threshold -0.012 --walk-forward-windows 4 --walk-forward-test-size 180
```

For a horizon of `N`, the pipeline removes the final `N` rows that do not have a realized label, purges `N` rows before every evaluation boundary, and lags the persistence baseline by `N`. This prevents future prices from leaking into training or into a deceptively strong one-row-lag baseline.

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

- test whether economically motivated features improve the cross-instrument promotion evidence without loosening the frozen gate
- add a second independent, redistribution-safe asset family before making any predictive-performance claim

## Portfolio Repro Checklist

Use this sequence before publishing a run artifact:

1. Baseline synthetic run:
`python -m src.train --use-synthetic --artifacts artifacts/baseline`
2. Multi-session benchmark + walk-forward:
`python -m src.train --use-synthetic --synthetic-points 3000 --label-horizon 10 --bull-threshold 0.012 --bear-threshold -0.012 --artifacts artifacts/walkforward`
3. Confirm both directories include:
- `model_summary.json`
- `test_predictions.csv`
- `walk_forward_metrics.csv` (for non-skip runs)
4. In notes, distinguish single holdout accuracy from walk-forward average accuracy.

## Portfolio Positioning

- Project type: Python CLI/ML workflow
- Verification path: python -m src.train --help

## Artifact reading guide

- `model_summary.json`: quickest machine-readable snapshot of accuracy, class mix, and confidence posture
- `benchmark_accuracy.csv`: first sanity check for whether the forest beats the linear challenger or trivial baselines, including class-sensitive metrics
- `test_predictions.csv`: holdout prediction ledger for false-positive / false-negative review
- `probability_comparison.csv`: whether calibration improved per-class Brier score or merely shifted confidence
- `walk_forward_metrics.csv`: better read on temporal robustness than a single holdout score
- `walk_forward_summary.json`: includes mean metrics and how often the forest beat logistic, persistence, and majority baselines across windows
- `benchmark_suite_results.json`: cross-instrument contract status plus the model-family promotion decision and every gate requirement
- `model_family_promotion.csv`: compact holdout and walk-forward balanced-metric deltas for each ECB currency pair
- `feature_drift.csv`: quick read on whether the holdout slice has drifted materially away from the training regime
- `model_search.csv`: chronological validation scoreboard, including whether each forest candidate was eligible to replace the maintained default
- `run_report.md`: human-facing summary worth linking in notes or portfolio discussion

## Label Tuning

You can tighten or relax the bull/bear labeling cutoffs without editing code:

```bash
python -m src.train --csv path/to/ohlcv.csv --bull-threshold 0.004 --bear-threshold -0.004
```


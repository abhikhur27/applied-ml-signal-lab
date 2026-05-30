from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix

FEATURE_COLS = [
    "log_ret_1",
    "ret_5",
    "ret_10",
    "vol_10",
    "vol_20",
    "ma_spread",
    "rsi_14",
]


def generate_synthetic_prices(points: int = 2200, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    states = np.zeros(points, dtype=int)
    prices = np.zeros(points, dtype=float)
    prices[0] = 100.0

    # 0 bull, 1 neutral, 2 bear
    transition = np.array(
        [
            [0.93, 0.05, 0.02],
            [0.08, 0.84, 0.08],
            [0.03, 0.07, 0.90],
        ]
    )
    drift = np.array([0.0009, 0.0001, -0.0012])
    vol = np.array([0.007, 0.004, 0.011])

    states[0] = 1
    for i in range(1, points):
        prev = states[i - 1]
        states[i] = rng.choice([0, 1, 2], p=transition[prev])
        ret = drift[states[i]] + rng.normal(0, vol[states[i]])
        prices[i] = prices[i - 1] * (1 + ret)

    dates = pd.date_range("2017-01-01", periods=points, freq="B")
    return pd.DataFrame({"date": dates, "close": prices})


def build_features(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()
    df = df.sort_values("date").reset_index(drop=True)
    df["log_ret_1"] = np.log(df["close"]).diff()
    df["ret_5"] = df["close"].pct_change(5)
    df["ret_10"] = df["close"].pct_change(10)
    df["vol_10"] = df["log_ret_1"].rolling(10).std()
    df["vol_20"] = df["log_ret_1"].rolling(20).std()
    df["ma_5"] = df["close"].rolling(5).mean()
    df["ma_20"] = df["close"].rolling(20).mean()
    df["ma_spread"] = (df["ma_5"] - df["ma_20"]) / df["ma_20"]

    delta = df["close"].diff()
    up = delta.clip(lower=0).rolling(14).mean()
    down = (-delta.clip(upper=0)).rolling(14).mean() + 1e-12
    rs = up / down
    df["rsi_14"] = 100 - (100 / (1 + rs))

    future_ret = df["close"].shift(-1) / df["close"] - 1
    df["target"] = np.select(
        [future_ret > 0.003, future_ret < -0.003],
        [2, 0],
        default=1,
    )
    return df.dropna().reset_index(drop=True)


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "date" not in df.columns or "close" not in df.columns:
        raise ValueError("CSV must contain `date` and `close` columns.")
    df["date"] = pd.to_datetime(df["date"])
    return validate_price_series(df)


def validate_price_series(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df[["date", "close"]].copy()
    cleaned["close"] = pd.to_numeric(cleaned["close"], errors="coerce")

    invalid_close = cleaned["close"].isna().sum()
    if invalid_close:
        raise ValueError(f"CSV contains {invalid_close} non-numeric close values.")

    non_positive = int((cleaned["close"] <= 0).sum())
    if non_positive:
        raise ValueError("Close prices must be strictly positive for log-return feature engineering.")

    duplicate_dates = int(cleaned["date"].duplicated().sum())
    if duplicate_dates:
        raise ValueError(f"CSV contains {duplicate_dates} duplicate date rows; deduplicate before training.")

    return cleaned.sort_values("date").reset_index(drop=True)


def run_training(df: pd.DataFrame, artifacts_dir: Path, model_seed: int) -> None:
    split = int(len(df) * 0.8)
    train_df = df.iloc[:split]
    test_df = df.iloc[split:]

    x_train = train_df[FEATURE_COLS]
    y_train = train_df["target"]
    x_test = test_df[FEATURE_COLS]
    y_test = test_df["target"]

    model = build_model(model_seed)
    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    probs = model.predict_proba(x_test)

    label_names = {0: "bear", 1: "neutral", 2: "bull"}
    report = classification_report(
        y_test,
        preds,
        target_names=[label_names[0], label_names[1], label_names[2]],
        digits=4,
    )
    matrix = confusion_matrix(y_test, preds, labels=[0, 1, 2])
    matrix_df = pd.DataFrame(
        matrix,
        index=["actual_bear", "actual_neutral", "actual_bull"],
        columns=["pred_bear", "pred_neutral", "pred_bull"],
    )

    importance = (
        pd.DataFrame({"feature": FEATURE_COLS, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    predictions_df = test_df[["date", "close", "target"]].copy()
    predictions_df["prediction"] = preds
    predictions_df["target_label"] = predictions_df["target"].map(label_names)
    predictions_df["prediction_label"] = predictions_df["prediction"].map(label_names)
    predictions_df["prob_bear"] = probs[:, 0]
    predictions_df["prob_neutral"] = probs[:, 1]
    predictions_df["prob_bull"] = probs[:, 2]
    predictions_df["confidence"] = probs.max(axis=1)
    sorted_probs = np.sort(probs, axis=1)
    predictions_df["margin_to_runner_up"] = sorted_probs[:, -1] - sorted_probs[:, -2]
    predictions_df["correct"] = predictions_df["target"] == predictions_df["prediction"]
    confidence_by_label = (
        predictions_df.groupby("prediction_label")["confidence"]
        .mean()
        .round(4)
        .to_dict()
    )
    low_confidence_rate = round(float((predictions_df["confidence"] < 0.5).mean()), 4)
    model_summary = {
        "samples": len(df),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "test_accuracy": round(float(accuracy_score(y_test, preds)), 4),
        "mean_confidence": round(float(predictions_df["confidence"].mean()), 4),
        "low_confidence_rate": low_confidence_rate,
        "mean_confidence_by_prediction": confidence_by_label,
        "prediction_mix": predictions_df["prediction_label"].value_counts(normalize=True).round(4).to_dict(),
        "top_features": importance.head(5).to_dict(orient="records"),
        "class_labels": label_names,
    }

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, artifacts_dir / "regime_classifier.joblib")
    matrix_df.to_csv(artifacts_dir / "confusion_matrix.csv", index=True)
    importance.to_csv(artifacts_dir / "feature_importance.csv", index=False)
    predictions_df.to_csv(artifacts_dir / "test_predictions.csv", index=False)
    (artifacts_dir / "model_summary.json").write_text(json.dumps(model_summary, indent=2), encoding="utf-8")

    report_md = [
        "# Training Run Report",
        "",
        f"- Samples: {len(df)}",
        f"- Train rows: {len(train_df)}",
        f"- Test rows: {len(test_df)}",
        f"- Mean confidence: {predictions_df['confidence'].mean():.4f}",
        f"- Low-confidence share (<0.50): {low_confidence_rate:.4f}",
        "",
        "## Classification report",
        "```",
        report,
        "```",
        "## Confidence summary",
        "```",
        predictions_df.groupby("prediction_label")[["confidence", "margin_to_runner_up"]].mean().round(4).to_string(),
        "```",
        "## Top feature importance",
        "```",
        importance.head(7).to_string(index=False),
        "```",
    ]
    (artifacts_dir / "run_report.md").write_text("\n".join(report_md), encoding="utf-8")

    print(report)
    print("\nSaved artifacts:")
    print(f"- {artifacts_dir / 'regime_classifier.joblib'}")
    print(f"- {artifacts_dir / 'confusion_matrix.csv'}")
    print(f"- {artifacts_dir / 'feature_importance.csv'}")
    print(f"- {artifacts_dir / 'test_predictions.csv'}")
    print(f"- {artifacts_dir / 'model_summary.json'}")
    print(f"- {artifacts_dir / 'run_report.md'}")


def build_model(model_seed: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=350,
        min_samples_leaf=6,
        max_depth=8,
        random_state=model_seed,
        class_weight="balanced_subsample",
    )


def run_walk_forward(df: pd.DataFrame, artifacts_dir: Path, windows: int, test_size: int, model_seed: int) -> None:
    minimum_train_size = max(160, len(df) // 3)
    rows = []

    for window_index in range(windows):
        train_end = minimum_train_size + window_index * test_size
        test_end = train_end + test_size
        if test_end > len(df):
            break

        train_df = df.iloc[:train_end]
        test_df = df.iloc[train_end:test_end]
        if len(test_df) < max(12, test_size // 2):
            break

        model = build_model(model_seed)
        model.fit(train_df[FEATURE_COLS], train_df["target"])
        preds = model.predict(test_df[FEATURE_COLS])

        rows.append(
            {
                "window": window_index + 1,
                "train_rows": len(train_df),
                "test_rows": len(test_df),
                "start_date": test_df["date"].iloc[0].date().isoformat(),
                "end_date": test_df["date"].iloc[-1].date().isoformat(),
                "accuracy": round(accuracy_score(test_df["target"], preds), 4),
                "bull_share": round(float((preds == 2).mean()), 4),
                "bear_share": round(float((preds == 0).mean()), 4),
            }
        )

    artifacts_dir.mkdir(parents=True, exist_ok=True)

    if not rows:
        (artifacts_dir / "walk_forward_report.md").write_text(
            "# Walk-Forward Evaluation\n\nNo valid windows were generated for the current dataset.\n",
            encoding="utf-8",
        )
        print("Walk-forward evaluation skipped: not enough rows for the requested window settings.")
        return

    walk_forward_df = pd.DataFrame(rows)
    walk_forward_df.to_csv(artifacts_dir / "walk_forward_metrics.csv", index=False)

    report_lines = [
        "# Walk-Forward Evaluation",
        "",
        f"- Windows completed: {len(walk_forward_df)}",
        f"- Mean accuracy: {walk_forward_df['accuracy'].mean():.4f}",
        f"- Best window accuracy: {walk_forward_df['accuracy'].max():.4f}",
        f"- Worst window accuracy: {walk_forward_df['accuracy'].min():.4f}",
        "",
        "## Window metrics",
        "```",
        walk_forward_df.to_string(index=False),
        "```",
    ]
    (artifacts_dir / "walk_forward_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    print("\nWalk-forward evaluation complete:")
    print(f"- {artifacts_dir / 'walk_forward_metrics.csv'}")
    print(f"- {artifacts_dir / 'walk_forward_report.md'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a market regime classifier.")
    parser.add_argument("--csv", type=Path, help="Path to CSV with date and close columns.")
    parser.add_argument("--use-synthetic", action="store_true", help="Use synthetic regime data.")
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"), help="Output artifacts directory.")
    parser.add_argument("--skip-walk-forward", action="store_true", help="Skip expanding-window walk-forward evaluation.")
    parser.add_argument("--walk-forward-windows", type=int, default=4, help="Number of walk-forward windows to evaluate.")
    parser.add_argument("--walk-forward-test-size", type=int, default=120, help="Rows per walk-forward test window.")
    parser.add_argument("--synthetic-points", type=int, default=2200, help="Number of synthetic price rows when --use-synthetic is enabled.")
    parser.add_argument("--synthetic-seed", type=int, default=7, help="Random seed for synthetic price generation.")
    parser.add_argument("--model-seed", type=int, default=42, help="Random seed for model training.")
    args = parser.parse_args()

    if not args.use_synthetic and args.csv is None:
        raise SystemExit("Provide --csv path/to/file.csv or use --use-synthetic.")

    if args.csv is not None:
        raw = load_csv(args.csv)
    else:
        raw = generate_synthetic_prices(points=max(300, args.synthetic_points), seed=args.synthetic_seed)
        raw = validate_price_series(raw)

    processed = build_features(raw)
    run_training(processed, args.artifacts, model_seed=args.model_seed)
    if not args.skip_walk_forward:
        run_walk_forward(
            processed,
            args.artifacts,
            windows=max(1, args.walk_forward_windows),
            test_size=max(20, args.walk_forward_test_size),
            model_seed=args.model_seed,
        )


if __name__ == "__main__":
    main()

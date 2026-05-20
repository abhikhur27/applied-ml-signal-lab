from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix


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
    return df


def run_training(df: pd.DataFrame, artifacts_dir: Path) -> None:
    feature_cols = [
        "log_ret_1",
        "ret_5",
        "ret_10",
        "vol_10",
        "vol_20",
        "ma_spread",
        "rsi_14",
    ]
    split = int(len(df) * 0.8)
    train_df = df.iloc[:split]
    test_df = df.iloc[split:]

    x_train = train_df[feature_cols]
    y_train = train_df["target"]
    x_test = test_df[feature_cols]
    y_test = test_df["target"]

    model = RandomForestClassifier(
        n_estimators=350,
        min_samples_leaf=6,
        max_depth=8,
        random_state=42,
        class_weight="balanced_subsample",
    )
    model.fit(x_train, y_train)
    preds = model.predict(x_test)

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
        pd.DataFrame({"feature": feature_cols, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, artifacts_dir / "regime_classifier.joblib")
    matrix_df.to_csv(artifacts_dir / "confusion_matrix.csv", index=True)
    importance.to_csv(artifacts_dir / "feature_importance.csv", index=False)

    report_md = [
        "# Training Run Report",
        "",
        f"- Samples: {len(df)}",
        f"- Train rows: {len(train_df)}",
        f"- Test rows: {len(test_df)}",
        "",
        "## Classification report",
        "```",
        report,
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
    print(f"- {artifacts_dir / 'run_report.md'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a market regime classifier.")
    parser.add_argument("--csv", type=Path, help="Path to CSV with date and close columns.")
    parser.add_argument("--use-synthetic", action="store_true", help="Use synthetic regime data.")
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"), help="Output artifacts directory.")
    args = parser.parse_args()

    if not args.use_synthetic and args.csv is None:
        raise SystemExit("Provide --csv path/to/file.csv or use --use-synthetic.")

    if args.csv is not None:
        raw = load_csv(args.csv)
    else:
        raw = generate_synthetic_prices()

    processed = build_features(raw)
    run_training(processed, args.artifacts)


if __name__ == "__main__":
    main()

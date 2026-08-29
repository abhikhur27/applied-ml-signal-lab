from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.metrics import brier_score_loss
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURE_COLS = [
    "log_ret_1",
    "ret_5",
    "ret_10",
    "vol_10",
    "vol_20",
    "ma_spread",
    "rsi_14",
]

DEFAULT_MODEL_CONFIG = {
    "name": "rf_balanced_depth8_leaf6_estimators350",
    "n_estimators": 350,
    "min_samples_leaf": 6,
    "max_depth": 8,
}

MODEL_SEARCH_CANDIDATES = [
    DEFAULT_MODEL_CONFIG,
    {
        "name": "rf_balanced_depth6_leaf8_estimators250",
        "n_estimators": 250,
        "min_samples_leaf": 8,
        "max_depth": 6,
    },
    {
        "name": "rf_balanced_depth10_leaf4_estimators450",
        "n_estimators": 450,
        "min_samples_leaf": 4,
        "max_depth": 10,
    },
    {
        "name": "rf_balanced_depth12_leaf3_estimators550",
        "n_estimators": 550,
        "min_samples_leaf": 3,
        "max_depth": 12,
    },
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


def build_features(
    data: pd.DataFrame,
    bull_threshold: float = 0.003,
    bear_threshold: float = -0.003,
    label_horizon: int = 1,
) -> pd.DataFrame:
    if bull_threshold <= 0:
        raise ValueError("bull_threshold must be positive.")
    if bear_threshold >= 0:
        raise ValueError("bear_threshold must be negative.")
    if bull_threshold <= abs(bear_threshold) / 10:
        raise ValueError("bull_threshold is unrealistically tight relative to the bear threshold.")
    if label_horizon < 1:
        raise ValueError("label_horizon must be at least 1.")

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

    df["label_end_date"] = df["date"].shift(-label_horizon)
    df["forward_return"] = df["close"].shift(-label_horizon) / df["close"] - 1
    df = df.dropna(subset=FEATURE_COLS + ["label_end_date", "forward_return"]).copy()
    df["target"] = np.select(
        [df["forward_return"] > bull_threshold, df["forward_return"] < bear_threshold],
        [2, 0],
        default=1,
    )
    return df.reset_index(drop=True)


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


def chronological_holdout(
    df: pd.DataFrame,
    label_horizon: int,
    test_fraction: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    if label_horizon < 1:
        raise ValueError("label_horizon must be at least 1.")
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1.")
    split = int(len(df) * (1 - test_fraction))
    train_end = split - label_horizon
    if train_end < 1 or split >= len(df):
        raise ValueError("Dataset is too small for the requested chronological holdout and label horizon.")

    train_df = df.iloc[:train_end].copy()
    test_df = df.iloc[split:].copy()
    observable_target_history = df.iloc[:split]["target"].copy()
    return train_df, test_df, observable_target_history


def build_baseline_predictions(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_horizon: int = 1,
    observable_target_history: pd.Series | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    majority_label = int(train_df["target"].mode().iloc[0])
    majority_preds = np.full(len(test_df), majority_label, dtype=int)

    history = train_df["target"] if observable_target_history is None else observable_target_history
    if len(history) < label_horizon:
        raise ValueError("Not enough observable target history for the persistence baseline.")
    lagged_targets = pd.concat(
        [
            history.iloc[-label_horizon:],
            test_df["target"].iloc[:-label_horizon],
        ],
        ignore_index=True,
    )
    persistence_preds = lagged_targets.to_numpy(dtype=int)
    return majority_preds, persistence_preds


def classification_metrics(y_true: pd.Series | np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": round(float(accuracy_score(y_true, predictions)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, predictions)), 4),
        "macro_f1": round(float(f1_score(y_true, predictions, average="macro", zero_division=0)), 4),
    }


def build_benchmark_summary(
    y_true: pd.Series,
    rf_preds: np.ndarray,
    linear_preds: np.ndarray,
    majority_preds: np.ndarray,
    persistence_preds: np.ndarray,
) -> pd.DataFrame:
    y_array = y_true.to_numpy()
    rows = []
    benchmark_map = {
        "random_forest": rf_preds,
        "regularized_logistic": linear_preds,
        "persistence": persistence_preds,
        "majority_class": majority_preds,
    }
    for name, preds in benchmark_map.items():
        metrics = classification_metrics(y_array, preds)
        rows.append(
            {
                "model": name,
                **metrics,
                "bull_share": round(float((preds == 2).mean()), 4),
                "neutral_share": round(float((preds == 1).mean()), 4),
                "bear_share": round(float((preds == 0).mean()), 4),
            }
        )

    benchmark_df = pd.DataFrame(rows)
    rf_row = benchmark_df.loc[benchmark_df["model"] == "random_forest"].iloc[0]
    for metric in ["accuracy", "balanced_accuracy", "macro_f1"]:
        benchmark_df[f"{metric}_delta_vs_random_forest"] = benchmark_df[metric].apply(
            lambda value, baseline=float(rf_row[metric]): round(float(value - baseline), 4)
        )
    return benchmark_df


def select_model_config(
    train_df: pd.DataFrame,
    model_seed: int,
    enable_model_search: bool,
    label_horizon: int = 1,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    if not enable_model_search:
        return DEFAULT_MODEL_CONFIG, pd.DataFrame(), {
            "enabled": False,
            "applied": False,
            "selected_config": DEFAULT_MODEL_CONFIG["name"],
            "reason": "disabled_by_flag",
            "validation_rows": 0,
            "search_rows": len(train_df),
        }

    validation_rows = max(90, int(len(train_df) * 0.2))
    validation_rows = min(validation_rows, max(0, len(train_df) - 160 - label_horizon))
    if validation_rows < 60:
        return DEFAULT_MODEL_CONFIG, pd.DataFrame(), {
            "enabled": True,
            "applied": False,
            "selected_config": DEFAULT_MODEL_CONFIG["name"],
            "reason": "not_enough_rows_for_validation_slice",
            "validation_rows": validation_rows,
            "search_rows": len(train_df),
        }

    fit_end = len(train_df) - validation_rows - label_horizon
    fit_df = train_df.iloc[:fit_end]
    validation_df = train_df.iloc[-validation_rows:]
    all_labels = {0, 1, 2}
    fit_labels = set(int(value) for value in fit_df["target"].unique())
    validation_labels = set(int(value) for value in validation_df["target"].unique())
    if fit_df.empty or validation_df.empty or fit_labels != all_labels or validation_labels != all_labels:
        return DEFAULT_MODEL_CONFIG, pd.DataFrame(), {
            "enabled": True,
            "applied": False,
            "selected_config": DEFAULT_MODEL_CONFIG["name"],
            "reason": "missing_class_in_validation_slice",
            "validation_rows": validation_rows,
            "search_rows": len(train_df),
        }

    x_fit = fit_df[FEATURE_COLS]
    y_fit = fit_df["target"]
    x_validation = validation_df[FEATURE_COLS]
    y_validation = validation_df["target"]
    majority_preds, persistence_preds = build_baseline_predictions(
        fit_df,
        validation_df,
        label_horizon=label_horizon,
        observable_target_history=train_df.iloc[:-validation_rows]["target"],
    )
    majority_accuracy = float(accuracy_score(y_validation, majority_preds))
    persistence_accuracy = float(accuracy_score(y_validation, persistence_preds))

    rows = []
    for candidate in MODEL_SEARCH_CANDIDATES:
        model = build_model(model_seed, candidate)
        model.fit(x_fit, y_fit)
        probs = model.predict_proba(x_validation)
        preds = model.predict(x_validation)
        accuracy = float(accuracy_score(y_validation, preds))
        rows.append(
            {
                "config_name": candidate["name"],
                "n_estimators": candidate["n_estimators"],
                "max_depth": candidate["max_depth"],
                "min_samples_leaf": candidate["min_samples_leaf"],
                "validation_accuracy": round(accuracy, 4),
                "accuracy_vs_persistence": round(accuracy - persistence_accuracy, 4),
                "accuracy_vs_majority": round(accuracy - majority_accuracy, 4),
                "eligible_for_selection": accuracy > persistence_accuracy and accuracy > majority_accuracy,
                "mean_confidence": round(float(probs.max(axis=1).mean()), 4),
                "low_confidence_rate": round(float((probs.max(axis=1) < 0.5).mean()), 4),
                "bull_share": round(float((preds == 2).mean()), 4),
                "neutral_share": round(float((preds == 1).mean()), 4),
                "bear_share": round(float((preds == 0).mean()), 4),
            }
        )

    search_df = pd.DataFrame(rows).sort_values(
        ["eligible_for_selection", "validation_accuracy", "accuracy_vs_persistence", "mean_confidence"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    top_candidate = search_df.iloc[0]
    selection_applied = bool(top_candidate["eligible_for_selection"])
    selected_name = str(top_candidate["config_name"]) if selection_applied else str(DEFAULT_MODEL_CONFIG["name"])
    selected_config = next(candidate for candidate in MODEL_SEARCH_CANDIDATES if candidate["name"] == selected_name)
    return selected_config, search_df, {
        "enabled": True,
        "applied": selection_applied,
        "selected_config": selected_name,
        "reason": "best_baseline_beating_validation_accuracy" if selection_applied else "no_candidate_beat_both_baselines",
        "validation_rows": len(validation_df),
        "search_rows": len(fit_df),
        "purged_rows": label_horizon,
        "validation_start_date": validation_df["date"].iloc[0].date().isoformat(),
        "validation_end_date": validation_df["date"].iloc[-1].date().isoformat(),
        "validation_baselines": {
            "persistence_accuracy": round(persistence_accuracy, 4),
            "majority_accuracy": round(majority_accuracy, 4),
        },
    }


def run_training(
    df: pd.DataFrame,
    artifacts_dir: Path,
    model_seed: int,
    bull_threshold: float,
    bear_threshold: float,
    label_horizon: int,
    calibrate_probabilities: bool,
    calibration_fraction: float,
    calibration_method: str,
    model_search: bool,
) -> None:
    train_df, test_df, observable_target_history = chronological_holdout(
        df,
        label_horizon=label_horizon,
    )

    x_test = test_df[FEATURE_COLS]
    y_test = test_df["target"]
    selected_config, model_search_df, model_search_summary = select_model_config(
        train_df=train_df,
        model_seed=model_seed,
        enable_model_search=model_search,
        label_horizon=label_horizon,
    )
    fitted_model, calibration_summary = fit_model_with_optional_calibration(
        train_df=train_df,
        model_seed=model_seed,
        calibrate_probabilities=calibrate_probabilities,
        calibration_fraction=calibration_fraction,
        calibration_method=calibration_method,
        model_config=selected_config,
        label_horizon=label_horizon,
    )
    feature_drift_df = build_feature_drift_report(
        calibration_summary["fit_features"],
        x_test,
    )
    baseline_probs = calibration_summary["baseline_probabilities"](x_test)
    probs = fitted_model.predict_proba(x_test)
    preds = fitted_model.predict(x_test)
    linear_model = build_linear_challenger(model_seed)
    linear_model.fit(train_df[FEATURE_COLS], train_df["target"])
    linear_preds = linear_model.predict(x_test)
    majority_preds, persistence_preds = build_baseline_predictions(
        train_df,
        test_df,
        label_horizon=label_horizon,
        observable_target_history=observable_target_history,
    )
    benchmark_df = build_benchmark_summary(
        y_true=y_test,
        rf_preds=preds,
        linear_preds=linear_preds,
        majority_preds=majority_preds,
        persistence_preds=persistence_preds,
    )

    label_names = {0: "bear", 1: "neutral", 2: "bull"}
    report = classification_report(
        y_test,
        preds,
        target_names=[label_names[0], label_names[1], label_names[2]],
        digits=4,
        zero_division=0,
    )
    matrix = confusion_matrix(y_test, preds, labels=[0, 1, 2])
    matrix_df = pd.DataFrame(
        matrix,
        index=["actual_bear", "actual_neutral", "actual_bull"],
        columns=["pred_bear", "pred_neutral", "pred_bull"],
    )

    importance = (
        pd.DataFrame({"feature": FEATURE_COLS, "importance": calibration_summary["importance_model"].feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    predictions_df = test_df[["date", "label_end_date", "close", "forward_return", "target"]].copy()
    predictions_df["prediction"] = preds
    predictions_df["target_label"] = predictions_df["target"].map(label_names)
    predictions_df["prediction_label"] = predictions_df["prediction"].map(label_names)
    predictions_df["majority_prediction"] = majority_preds
    predictions_df["majority_prediction_label"] = pd.Series(majority_preds).map(label_names)
    predictions_df["persistence_prediction"] = persistence_preds
    predictions_df["persistence_prediction_label"] = pd.Series(persistence_preds).map(label_names)
    predictions_df["linear_prediction"] = linear_preds
    predictions_df["linear_prediction_label"] = pd.Series(linear_preds).map(label_names)
    predictions_df["baseline_prob_bear"] = baseline_probs[:, 0]
    predictions_df["baseline_prob_neutral"] = baseline_probs[:, 1]
    predictions_df["baseline_prob_bull"] = baseline_probs[:, 2]
    predictions_df["prob_bear"] = probs[:, 0]
    predictions_df["prob_neutral"] = probs[:, 1]
    predictions_df["prob_bull"] = probs[:, 2]
    predictions_df["baseline_confidence"] = baseline_probs.max(axis=1)
    predictions_df["confidence"] = probs.max(axis=1)
    predictions_df["confidence_delta"] = predictions_df["confidence"] - predictions_df["baseline_confidence"]
    sorted_probs = np.sort(probs, axis=1)
    predictions_df["margin_to_runner_up"] = sorted_probs[:, -1] - sorted_probs[:, -2]
    predictions_df["correct"] = predictions_df["target"] == predictions_df["prediction"]
    confidence_by_label = (
        predictions_df.groupby("prediction_label")["confidence"]
        .mean()
        .round(4)
        .to_dict()
    )
    confidence_bins = pd.cut(
        predictions_df["confidence"],
        bins=[0.0, 0.45, 0.60, 0.75, 1.000001],
        labels=["very_low", "guarded", "usable", "high"],
        include_lowest=True,
        right=False,
    )
    confidence_bucket_df = (
        predictions_df.assign(confidence_bucket=confidence_bins)
        .groupby("confidence_bucket", observed=False)
        .agg(
            rows=("confidence", "size"),
            mean_confidence=("confidence", "mean"),
            accuracy=("correct", "mean"),
            mean_margin=("margin_to_runner_up", "mean"),
        )
        .reset_index()
    )
    confidence_bucket_df["rows"] = confidence_bucket_df["rows"].fillna(0).astype(int)
    confidence_bucket_df["share"] = (confidence_bucket_df["rows"] / len(predictions_df)).round(4)
    for column in ["mean_confidence", "accuracy", "mean_margin"]:
        confidence_bucket_df[column] = confidence_bucket_df[column].fillna(0.0).round(4)
    calibration_df = (
        predictions_df.assign(
            confidence_band=pd.cut(
                predictions_df["confidence"],
                bins=np.linspace(0.0, 1.0, 11),
                labels=[f"{start/10:.1f}-{(start+1)/10:.1f}" for start in range(10)],
                include_lowest=True,
                right=True,
            )
        )
        .groupby("confidence_band", observed=False)
        .agg(
            rows=("confidence", "size"),
            mean_confidence=("confidence", "mean"),
            empirical_accuracy=("correct", "mean"),
            mean_margin=("margin_to_runner_up", "mean"),
        )
        .reset_index()
    )
    calibration_df["rows"] = calibration_df["rows"].fillna(0).astype(int)
    calibration_df["share"] = (calibration_df["rows"] / len(predictions_df)).round(4)
    calibration_df["confidence_gap"] = (
        calibration_df["mean_confidence"].fillna(0.0) - calibration_df["empirical_accuracy"].fillna(0.0)
    ).round(4)
    for column in ["mean_confidence", "empirical_accuracy", "mean_margin"]:
        calibration_df[column] = calibration_df[column].fillna(0.0).round(4)
    low_confidence_rate = round(float((predictions_df["confidence"] < 0.5).mean()), 4)
    baseline_low_confidence_rate = round(float((predictions_df["baseline_confidence"] < 0.5).mean()), 4)
    top_class_brier = round(
        float(brier_score_loss(predictions_df["correct"].astype(int), predictions_df["confidence"])),
        4,
    )
    baseline_top_class_brier = round(
        float(brier_score_loss(predictions_df["correct"].astype(int), predictions_df["baseline_confidence"])),
        4,
    )
    probability_comparison_df = build_probability_comparison_report(
        y_test=y_test.to_numpy(),
        baseline_probs=baseline_probs,
        calibrated_probs=probs,
        label_names=label_names,
    )
    model_summary = {
        "samples": len(df),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "label_horizon": label_horizon,
        "purged_holdout_rows": label_horizon,
        "bull_threshold": bull_threshold,
        "bear_threshold": bear_threshold,
        "probability_calibration": calibration_summary["metadata"],
        "model_selection": model_search_summary,
        "test_accuracy": round(float(accuracy_score(y_test, preds)), 4),
        "test_balanced_accuracy": round(float(balanced_accuracy_score(y_test, preds)), 4),
        "test_macro_f1": round(float(f1_score(y_test, preds, average="macro", zero_division=0)), 4),
        "benchmark_accuracy": benchmark_df.to_dict(orient="records"),
        "mean_confidence": round(float(predictions_df["confidence"].mean()), 4),
        "baseline_mean_confidence": round(float(predictions_df["baseline_confidence"].mean()), 4),
        "low_confidence_rate": low_confidence_rate,
        "baseline_low_confidence_rate": baseline_low_confidence_rate,
        "top_class_brier_score": top_class_brier,
        "baseline_top_class_brier_score": baseline_top_class_brier,
        "mean_confidence_by_prediction": confidence_by_label,
        "confidence_buckets": confidence_bucket_df.to_dict(orient="records"),
        "confidence_calibration": calibration_df.to_dict(orient="records"),
        "probability_comparison": probability_comparison_df.to_dict(orient="records"),
        "prediction_mix": predictions_df["prediction_label"].value_counts(normalize=True).round(4).to_dict(),
        "dataset_target_mix": df["target"].map(label_names).value_counts(normalize=True).round(4).to_dict(),
        "top_features": importance.head(5).to_dict(orient="records"),
        "feature_drift": feature_drift_df.to_dict(orient="records"),
        "class_labels": label_names,
    }
    class_balance = (
        df["target"]
        .map(label_names)
        .value_counts()
        .rename_axis("label")
        .reset_index(name="rows")
    )
    class_balance["share"] = (class_balance["rows"] / len(df)).round(4)

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted_model, artifacts_dir / "regime_classifier.joblib")
    joblib.dump(linear_model, artifacts_dir / "regularized_logistic_challenger.joblib")
    matrix_df.to_csv(artifacts_dir / "confusion_matrix.csv", index=True)
    importance.to_csv(artifacts_dir / "feature_importance.csv", index=False)
    class_balance.to_csv(artifacts_dir / "class_balance.csv", index=False)
    feature_drift_df.to_csv(artifacts_dir / "feature_drift.csv", index=False)
    confidence_bucket_df.to_csv(artifacts_dir / "confidence_buckets.csv", index=False)
    calibration_df.to_csv(artifacts_dir / "confidence_calibration.csv", index=False)
    benchmark_df.to_csv(artifacts_dir / "benchmark_accuracy.csv", index=False)
    probability_comparison_df.to_csv(artifacts_dir / "probability_comparison.csv", index=False)
    predictions_df.to_csv(artifacts_dir / "test_predictions.csv", index=False)
    (artifacts_dir / "model_summary.json").write_text(json.dumps(model_summary, indent=2), encoding="utf-8")
    if model_search_summary["enabled"]:
        if model_search_df.empty:
            search_lines = [
                "# Model Search",
                "",
                f"- Status: requested but skipped ({model_search_summary['reason']})",
                f"- Selected config fallback: {model_search_summary['selected_config']}",
            ]
        else:
            model_search_df.to_csv(artifacts_dir / "model_search.csv", index=False)
            search_lines = [
                "# Model Search",
                "",
                f"- Selected config: {model_search_summary['selected_config']}",
                f"- Decision: {model_search_summary['reason']}",
                f"- Search rows: {model_search_summary['search_rows']}",
                f"- Validation rows: {model_search_summary['validation_rows']}",
                f"- Validation date range: {model_search_summary['validation_start_date']} to {model_search_summary['validation_end_date']}",
                f"- Validation persistence accuracy: {model_search_summary['validation_baselines']['persistence_accuracy']:.4f}",
                f"- Validation majority accuracy: {model_search_summary['validation_baselines']['majority_accuracy']:.4f}",
                "",
                "## Candidate comparison",
                "```",
                model_search_df.to_string(index=False),
                "```",
            ]
        (artifacts_dir / "model_search_report.md").write_text("\n".join(search_lines), encoding="utf-8")

    report_md = [
        "# Training Run Report",
        "",
        f"- Samples: {len(df)}",
        f"- Train rows: {len(train_df)}",
        f"- Test rows: {len(test_df)}",
        f"- Label horizon: {label_horizon} trading row(s)",
        f"- Purged rows before holdout: {label_horizon}",
        f"- Bull threshold: {bull_threshold:.4f}",
        f"- Bear threshold: {bear_threshold:.4f}",
        f"- Selected model config: {model_search_summary['selected_config']}",
        f"- Model search: {format_model_search_status(model_search_summary)}",
        f"- Probability calibration: {format_calibration_status(calibration_summary['metadata'])}",
        f"- Random-forest balanced accuracy: {benchmark_df.loc[benchmark_df['model'] == 'random_forest', 'balanced_accuracy'].iloc[0]:.4f}",
        f"- Random-forest macro-F1: {benchmark_df.loc[benchmark_df['model'] == 'random_forest', 'macro_f1'].iloc[0]:.4f}",
        f"- Regularized-logistic accuracy: {benchmark_df.loc[benchmark_df['model'] == 'regularized_logistic', 'accuracy'].iloc[0]:.4f}",
        f"- Regularized-logistic balanced accuracy: {benchmark_df.loc[benchmark_df['model'] == 'regularized_logistic', 'balanced_accuracy'].iloc[0]:.4f}",
        f"- Regularized-logistic macro-F1: {benchmark_df.loc[benchmark_df['model'] == 'regularized_logistic', 'macro_f1'].iloc[0]:.4f}",
        f"- Persistence benchmark accuracy: {benchmark_df.loc[benchmark_df['model'] == 'persistence', 'accuracy'].iloc[0]:.4f}",
        f"- Majority-class benchmark accuracy: {benchmark_df.loc[benchmark_df['model'] == 'majority_class', 'accuracy'].iloc[0]:.4f}",
        f"- Mean confidence: {predictions_df['confidence'].mean():.4f}",
        f"- Baseline mean confidence: {predictions_df['baseline_confidence'].mean():.4f}",
        f"- Low-confidence share (<0.50): {low_confidence_rate:.4f}",
        f"- Baseline low-confidence share (<0.50): {baseline_low_confidence_rate:.4f}",
        f"- Top-class Brier score: {top_class_brier:.4f}",
        f"- Baseline top-class Brier score: {baseline_top_class_brier:.4f}",
        "",
        "## Classification report",
        "```",
        report,
        "```",
        "## Confidence summary",
        "```",
        predictions_df.groupby("prediction_label")[["confidence", "margin_to_runner_up"]].mean().round(4).to_string(),
        "```",
        "## Benchmark comparison",
        "```",
        benchmark_df.to_string(index=False),
        "```",
        "## Top feature importance",
        "```",
        importance.head(7).to_string(index=False),
        "```",
        "## Dataset class balance",
        "```",
        class_balance.to_string(index=False),
        "```",
        "## Feature drift",
        "```",
        feature_drift_df.to_string(index=False),
        "```",
        "## Confidence buckets",
        "```",
        confidence_bucket_df.to_string(index=False),
        "```",
        "## Confidence calibration",
        "```",
        calibration_df.to_string(index=False),
        "```",
        "## Probability comparison",
        "```",
        probability_comparison_df.to_string(index=False),
        "```",
    ]
    (artifacts_dir / "run_report.md").write_text("\n".join(report_md), encoding="utf-8")

    print(report)
    print("\nSaved artifacts:")
    print(f"- {artifacts_dir / 'regime_classifier.joblib'}")
    print(f"- {artifacts_dir / 'regularized_logistic_challenger.joblib'}")
    print(f"- {artifacts_dir / 'confusion_matrix.csv'}")
    print(f"- {artifacts_dir / 'feature_importance.csv'}")
    print(f"- {artifacts_dir / 'class_balance.csv'}")
    print(f"- {artifacts_dir / 'feature_drift.csv'}")
    print(f"- {artifacts_dir / 'confidence_buckets.csv'}")
    print(f"- {artifacts_dir / 'confidence_calibration.csv'}")
    print(f"- {artifacts_dir / 'benchmark_accuracy.csv'}")
    print(f"- {artifacts_dir / 'probability_comparison.csv'}")
    print(f"- {artifacts_dir / 'test_predictions.csv'}")
    print(f"- {artifacts_dir / 'model_summary.json'}")
    print(f"- {artifacts_dir / 'run_report.md'}")
    if model_search_summary["enabled"]:
        if not model_search_df.empty:
            print(f"- {artifacts_dir / 'model_search.csv'}")
        print(f"- {artifacts_dir / 'model_search_report.md'}")


def build_model(model_seed: int, config: dict[str, Any] | None = None) -> RandomForestClassifier:
    resolved = DEFAULT_MODEL_CONFIG if config is None else config
    return RandomForestClassifier(
        n_estimators=int(resolved["n_estimators"]),
        min_samples_leaf=int(resolved["min_samples_leaf"]),
        max_depth=int(resolved["max_depth"]),
        random_state=model_seed,
        class_weight="balanced_subsample",
    )


def build_linear_challenger(model_seed: int) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=model_seed,
                ),
            ),
        ]
    )


def multiclass_brier_score(y_true: np.ndarray, probabilities: np.ndarray, classes: np.ndarray) -> float:
    return float(
        np.mean(
            [
                brier_score_loss((y_true == label).astype(int), probabilities[:, index])
                for index, label in enumerate(classes)
            ]
        )
    )


def fit_model_with_optional_calibration(
    train_df: pd.DataFrame,
    model_seed: int,
    calibrate_probabilities: bool,
    calibration_fraction: float,
    calibration_method: str,
    model_config: dict[str, Any],
    label_horizon: int = 1,
) -> tuple[object, dict[str, object]]:
    x_train = train_df[FEATURE_COLS]
    y_train = train_df["target"]
    base_model = build_model(model_seed, model_config)

    if not calibrate_probabilities:
        base_model.fit(x_train, y_train)
        return base_model, {
            "fit_features": x_train,
            "importance_model": base_model,
            "baseline_probabilities": base_model.predict_proba,
            "metadata": {
                "enabled": False,
                "applied": False,
                "method": calibration_method,
                "reason": "disabled_by_flag",
                "fit_rows": len(train_df),
                "calibration_rows": 0,
                "model_config": model_config["name"],
            },
        }

    calibration_rows = max(120, int(len(train_df) * calibration_fraction))
    calibration_rows = min(calibration_rows, max(0, len(train_df) - 120 - (2 * label_horizon)))
    if calibration_rows < 120:
        base_model.fit(x_train, y_train)
        return base_model, {
            "fit_features": x_train,
            "importance_model": base_model,
            "baseline_probabilities": base_model.predict_proba,
            "metadata": {
                "enabled": True,
                "applied": False,
                "method": calibration_method,
                "reason": "not_enough_rows_for_calibration",
                "fit_rows": len(train_df),
                "calibration_rows": calibration_rows,
                "model_config": model_config["name"],
            },
        }

    audit_rows = max(60, int(calibration_rows * 0.3))
    calibration_start = len(train_df) - calibration_rows
    fit_end = calibration_start - label_horizon
    calibration_end = len(train_df) - audit_rows - label_horizon
    fit_df = train_df.iloc[:fit_end]
    calibration_df = train_df.iloc[calibration_start:calibration_end]
    audit_df = train_df.iloc[-audit_rows:]
    fit_labels = set(int(value) for value in fit_df["target"].unique())
    calibration_labels = set(int(value) for value in calibration_df["target"].unique())
    audit_labels = set(int(value) for value in audit_df["target"].unique())
    all_labels = {0, 1, 2}
    if (
        fit_df.empty
        or calibration_df.empty
        or audit_df.empty
        or fit_labels != all_labels
        or calibration_labels != all_labels
        or audit_labels != all_labels
    ):
        base_model.fit(x_train, y_train)
        return base_model, {
            "fit_features": x_train,
            "importance_model": base_model,
            "baseline_probabilities": base_model.predict_proba,
            "metadata": {
                "enabled": True,
                "applied": False,
                "method": calibration_method,
                "reason": "missing_class_in_fit_calibration_or_audit_slice",
                "fit_rows": len(train_df),
                "calibration_rows": len(calibration_df),
                "audit_rows": len(audit_df),
                "model_config": model_config["name"],
            },
        }

    x_fit = fit_df[FEATURE_COLS]
    y_fit = fit_df["target"]
    x_calibration = calibration_df[FEATURE_COLS]
    y_calibration = calibration_df["target"]
    x_audit = audit_df[FEATURE_COLS]
    y_audit = audit_df["target"].to_numpy()
    base_model.fit(x_fit, y_fit)
    calibrated_model = CalibratedClassifierCV(
        FrozenEstimator(base_model),
        method=calibration_method,
    )
    calibrated_model.fit(x_calibration, y_calibration)

    baseline_audit_probs = base_model.predict_proba(x_audit)
    calibrated_audit_probs = calibrated_model.predict_proba(x_audit)
    baseline_audit_brier = multiclass_brier_score(y_audit, baseline_audit_probs, base_model.classes_)
    calibrated_audit_brier = multiclass_brier_score(y_audit, calibrated_audit_probs, calibrated_model.classes_)
    audit_brier_delta = calibrated_audit_brier - baseline_audit_brier
    baseline_prediction_classes = int(np.unique(base_model.predict(x_audit)).size)
    calibrated_prediction_classes = int(np.unique(calibrated_model.predict(x_audit)).size)
    collapsed_predictions = calibrated_prediction_classes < min(2, len(audit_labels))
    audit_metadata = {
        "audit_rows": len(audit_df),
        "audit_start_date": audit_df["date"].iloc[0].date().isoformat(),
        "audit_end_date": audit_df["date"].iloc[-1].date().isoformat(),
        "baseline_multiclass_brier": round(baseline_audit_brier, 4),
        "calibrated_multiclass_brier": round(calibrated_audit_brier, 4),
        "multiclass_brier_delta": round(audit_brier_delta, 4),
        "baseline_prediction_classes": baseline_prediction_classes,
        "calibrated_prediction_classes": calibrated_prediction_classes,
    }

    if collapsed_predictions or audit_brier_delta >= -0.001:
        fallback_model = build_model(model_seed, model_config)
        fallback_model.fit(x_train, y_train)
        reason = "audit_rejected_prediction_collapse" if collapsed_predictions else "audit_rejected_no_brier_improvement"
        return fallback_model, {
            "fit_features": x_train,
            "importance_model": fallback_model,
            "baseline_probabilities": fallback_model.predict_proba,
            "metadata": {
                "enabled": True,
                "applied": False,
                "method": calibration_method,
                "reason": reason,
                "fit_rows": len(train_df),
                "calibration_rows": len(calibration_df),
                "purged_rows": 2 * label_horizon,
                "model_config": model_config["name"],
                "calibration_start_date": calibration_df["date"].iloc[0].date().isoformat(),
                "calibration_end_date": calibration_df["date"].iloc[-1].date().isoformat(),
                **audit_metadata,
            },
        }

    return calibrated_model, {
        "fit_features": x_fit,
        "importance_model": base_model,
        "baseline_probabilities": base_model.predict_proba,
        "metadata": {
            "enabled": True,
            "applied": True,
            "method": calibration_method,
            "reason": "audit_passed",
            "fit_rows": len(fit_df),
            "calibration_rows": len(calibration_df),
            "purged_rows": 2 * label_horizon,
            "model_config": model_config["name"],
            "calibration_start_date": calibration_df["date"].iloc[0].date().isoformat(),
            "calibration_end_date": calibration_df["date"].iloc[-1].date().isoformat(),
            **audit_metadata,
        },
    }


def build_probability_comparison_report(
    y_test: np.ndarray,
    baseline_probs: np.ndarray,
    calibrated_probs: np.ndarray,
    label_names: dict[int, str],
) -> pd.DataFrame:
    rows = []
    for label_index, label_name in label_names.items():
        one_vs_rest = (y_test == label_index).astype(int)
        rows.append(
            {
                "label": label_name,
                "baseline_brier": round(float(brier_score_loss(one_vs_rest, baseline_probs[:, label_index])), 4),
                "calibrated_brier": round(float(brier_score_loss(one_vs_rest, calibrated_probs[:, label_index])), 4),
                "brier_delta": round(
                    float(
                        brier_score_loss(one_vs_rest, calibrated_probs[:, label_index])
                        - brier_score_loss(one_vs_rest, baseline_probs[:, label_index])
                    ),
                    4,
                ),
                "baseline_mean_probability": round(float(baseline_probs[:, label_index].mean()), 4),
                "calibrated_mean_probability": round(float(calibrated_probs[:, label_index].mean()), 4),
            }
        )

    baseline_top_conf = baseline_probs.max(axis=1)
    calibrated_top_conf = calibrated_probs.max(axis=1)
    rows.append(
        {
            "label": "top_class_confidence",
            "baseline_brier": round(float(brier_score_loss((baseline_probs.argmax(axis=1) == y_test).astype(int), baseline_top_conf)), 4),
            "calibrated_brier": round(float(brier_score_loss((calibrated_probs.argmax(axis=1) == y_test).astype(int), calibrated_top_conf)), 4),
            "brier_delta": round(
                float(
                    brier_score_loss((calibrated_probs.argmax(axis=1) == y_test).astype(int), calibrated_top_conf)
                    - brier_score_loss((baseline_probs.argmax(axis=1) == y_test).astype(int), baseline_top_conf)
                ),
                4,
            ),
            "baseline_mean_probability": round(float(baseline_top_conf.mean()), 4),
            "calibrated_mean_probability": round(float(calibrated_top_conf.mean()), 4),
        }
    )
    return pd.DataFrame(rows)


def format_calibration_status(metadata: dict[str, object]) -> str:
    if metadata["applied"]:
        return f"enabled ({metadata['method']}, {metadata['calibration_rows']} calibration rows)"
    if metadata["enabled"]:
        return f"requested but skipped ({metadata['reason']})"
    return "disabled"


def format_model_search_status(metadata: dict[str, Any]) -> str:
    if metadata["applied"]:
        return f"enabled ({metadata['validation_rows']} validation rows)"
    if metadata["reason"] == "no_candidate_beat_both_baselines":
        return "evaluated; retained default because no candidate beat both baselines"
    if metadata["enabled"]:
        return f"requested but skipped ({metadata['reason']})"
    return "disabled"


def build_feature_drift_report(train_features: pd.DataFrame, test_features: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature in FEATURE_COLS:
        train_series = train_features[feature]
        test_series = test_features[feature]
        train_std = float(train_series.std(ddof=0))
        test_std = float(test_series.std(ddof=0))
        mean_shift_sigma = 0.0 if train_std == 0 else float((test_series.mean() - train_series.mean()) / train_std)
        std_ratio = 0.0 if train_std == 0 else float(test_std / train_std)
        q50_shift_sigma = 0.0 if train_std == 0 else float((test_series.median() - train_series.median()) / train_std)
        rows.append(
            {
                "feature": feature,
                "train_mean": round(float(train_series.mean()), 6),
                "test_mean": round(float(test_series.mean()), 6),
                "train_std": round(train_std, 6),
                "test_std": round(test_std, 6),
                "mean_shift_sigma": round(mean_shift_sigma, 4),
                "median_shift_sigma": round(q50_shift_sigma, 4),
                "std_ratio": round(std_ratio, 4),
            }
        )

    drift_df = pd.DataFrame(rows)
    drift_df["abs_mean_shift_sigma"] = drift_df["mean_shift_sigma"].abs().round(4)
    return drift_df.sort_values(["abs_mean_shift_sigma", "std_ratio"], ascending=[False, False]).reset_index(drop=True)


def parse_threshold_values(raw: str) -> list[float]:
    values = []
    for token in raw.split(","):
        stripped = token.strip()
        if not stripped:
            continue
        value = float(stripped)
        if value <= 0:
            raise ValueError("Threshold sweep values must be positive.")
        values.append(value)

    unique_values = sorted(set(values))
    if not unique_values:
        raise ValueError("Provide at least one positive threshold sweep value.")
    return unique_values


def run_threshold_sweep(
    raw: pd.DataFrame,
    artifacts_dir: Path,
    model_seed: int,
    threshold_values: list[float],
    label_horizon: int,
) -> None:
    rows = []

    for threshold in threshold_values:
        processed = build_features(
            raw,
            bull_threshold=threshold,
            bear_threshold=-threshold,
            label_horizon=label_horizon,
        )
        train_df, test_df, _ = chronological_holdout(processed, label_horizon=label_horizon)
        if train_df.empty or test_df.empty:
            continue

        model = build_model(model_seed)
        model.fit(train_df[FEATURE_COLS], train_df["target"])
        preds = model.predict(test_df[FEATURE_COLS])
        probs = model.predict_proba(test_df[FEATURE_COLS])

        rows.append(
            {
                "threshold": round(threshold, 4),
                "samples": len(processed),
                "test_accuracy": round(float(accuracy_score(test_df["target"], preds)), 4),
                "mean_confidence": round(float(probs.max(axis=1).mean()), 4),
                "low_confidence_rate": round(float((probs.max(axis=1) < 0.5).mean()), 4),
                "bull_share": round(float((processed["target"] == 2).mean()), 4),
                "neutral_share": round(float((processed["target"] == 1).mean()), 4),
                "bear_share": round(float((processed["target"] == 0).mean()), 4),
            }
        )

    artifacts_dir.mkdir(parents=True, exist_ok=True)

    if not rows:
        (artifacts_dir / "threshold_sweep_report.md").write_text(
            "# Threshold Sweep\n\nNo valid threshold sweep rows were generated.\n",
            encoding="utf-8",
        )
        print("Threshold sweep skipped: no valid threshold combinations were generated.")
        return

    sweep_df = pd.DataFrame(rows).sort_values(["test_accuracy", "mean_confidence"], ascending=[False, False]).reset_index(drop=True)
    sweep_df.to_csv(artifacts_dir / "threshold_sweep.csv", index=False)

    best = sweep_df.iloc[0]
    report_lines = [
        "# Threshold Sweep",
        "",
        f"- Thresholds tested: {', '.join(f'{value:.4f}' for value in threshold_values)}",
        f"- Best threshold by holdout accuracy: {best['threshold']:.4f}",
        f"- Best holdout accuracy: {best['test_accuracy']:.4f}",
        f"- Best mean confidence: {best['mean_confidence']:.4f}",
        "",
        "## Threshold comparison",
        "```",
        sweep_df.to_string(index=False),
        "```",
    ]
    (artifacts_dir / "threshold_sweep_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    print("\nThreshold sweep complete:")
    print(f"- {artifacts_dir / 'threshold_sweep.csv'}")
    print(f"- {artifacts_dir / 'threshold_sweep_report.md'}")


def run_feature_ablation(
    df: pd.DataFrame,
    artifacts_dir: Path,
    model_seed: int,
    label_horizon: int,
) -> None:
    train_df, test_df, _ = chronological_holdout(df, label_horizon=label_horizon)

    if train_df.empty or test_df.empty:
        (artifacts_dir / "feature_ablation_report.md").write_text(
            "# Feature Ablation\n\nDataset split did not leave enough rows for ablation.\n",
            encoding="utf-8",
        )
        print("Feature ablation skipped: not enough rows after the chronological split.")
        return

    rows = []
    baseline_model = build_model(model_seed)
    baseline_model.fit(train_df[FEATURE_COLS], train_df["target"])
    baseline_accuracy = float(accuracy_score(test_df["target"], baseline_model.predict(test_df[FEATURE_COLS])))

    rows.append(
        {
            "feature_removed": "none",
            "feature_count": len(FEATURE_COLS),
            "test_accuracy": round(baseline_accuracy, 4),
            "accuracy_delta": 0.0,
        }
    )

    for feature in FEATURE_COLS:
        ablated_features = [column for column in FEATURE_COLS if column != feature]
        model = build_model(model_seed)
        model.fit(train_df[ablated_features], train_df["target"])
        accuracy = float(accuracy_score(test_df["target"], model.predict(test_df[ablated_features])))
        rows.append(
            {
                "feature_removed": feature,
                "feature_count": len(ablated_features),
                "test_accuracy": round(accuracy, 4),
                "accuracy_delta": round(accuracy - baseline_accuracy, 4),
            }
        )

    ablation_df = pd.DataFrame(rows).sort_values(
        ["feature_removed", "test_accuracy"],
        ascending=[True, False],
    )
    ablation_df = pd.concat(
        [
            ablation_df[ablation_df["feature_removed"] == "none"],
            ablation_df[ablation_df["feature_removed"] != "none"].sort_values("accuracy_delta"),
        ],
        ignore_index=True,
    )
    ablation_df.to_csv(artifacts_dir / "feature_ablation.csv", index=False)

    most_helpful = ablation_df[ablation_df["feature_removed"] != "none"].iloc[0]
    report_lines = [
        "# Feature Ablation",
        "",
        f"- Baseline holdout accuracy: {baseline_accuracy:.4f}",
        f"- Largest accuracy drop: removing `{most_helpful['feature_removed']}` changed accuracy by {most_helpful['accuracy_delta']:.4f}",
        "",
        "## Ablation table",
        "```",
        ablation_df.to_string(index=False),
        "```",
    ]
    (artifacts_dir / "feature_ablation_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    print("\nFeature ablation complete:")
    print(f"- {artifacts_dir / 'feature_ablation.csv'}")
    print(f"- {artifacts_dir / 'feature_ablation_report.md'}")


def run_walk_forward(
    df: pd.DataFrame,
    artifacts_dir: Path,
    windows: int,
    test_size: int,
    model_seed: int,
    label_horizon: int,
) -> None:
    minimum_train_size = max(160, len(df) // 3)
    rows = []

    for window_index in range(windows):
        train_end = minimum_train_size + window_index * test_size
        test_end = train_end + test_size
        if test_end > len(df):
            break

        model_train_end = train_end - label_horizon
        if model_train_end < 1:
            break
        train_df = df.iloc[:model_train_end]
        test_df = df.iloc[train_end:test_end]
        if len(test_df) < max(12, test_size // 2):
            break

        model = build_model(model_seed)
        linear_model = build_linear_challenger(model_seed)
        model.fit(train_df[FEATURE_COLS], train_df["target"])
        linear_model.fit(train_df[FEATURE_COLS], train_df["target"])
        preds = model.predict(test_df[FEATURE_COLS])
        linear_preds = linear_model.predict(test_df[FEATURE_COLS])
        majority_preds, persistence_preds = build_baseline_predictions(
            train_df,
            test_df,
            label_horizon=label_horizon,
            observable_target_history=df.iloc[:train_end]["target"],
        )
        rf_metrics = classification_metrics(test_df["target"], preds)
        linear_metrics = classification_metrics(test_df["target"], linear_preds)
        majority_metrics = classification_metrics(test_df["target"], majority_preds)
        persistence_metrics = classification_metrics(test_df["target"], persistence_preds)

        rows.append(
            {
                "window": window_index + 1,
                "train_rows": len(train_df),
                "purged_rows": label_horizon,
                "test_rows": len(test_df),
                "start_date": test_df["date"].iloc[0].date().isoformat(),
                "end_date": test_df["date"].iloc[-1].date().isoformat(),
                "accuracy": rf_metrics["accuracy"],
                "balanced_accuracy": rf_metrics["balanced_accuracy"],
                "macro_f1": rf_metrics["macro_f1"],
                "linear_accuracy": linear_metrics["accuracy"],
                "linear_balanced_accuracy": linear_metrics["balanced_accuracy"],
                "linear_macro_f1": linear_metrics["macro_f1"],
                "persistence_accuracy": persistence_metrics["accuracy"],
                "persistence_balanced_accuracy": persistence_metrics["balanced_accuracy"],
                "persistence_macro_f1": persistence_metrics["macro_f1"],
                "majority_accuracy": majority_metrics["accuracy"],
                "majority_balanced_accuracy": majority_metrics["balanced_accuracy"],
                "majority_macro_f1": majority_metrics["macro_f1"],
                "accuracy_vs_linear": round(rf_metrics["accuracy"] - linear_metrics["accuracy"], 4),
                "balanced_accuracy_vs_linear": round(
                    rf_metrics["balanced_accuracy"] - linear_metrics["balanced_accuracy"], 4
                ),
                "macro_f1_vs_linear": round(rf_metrics["macro_f1"] - linear_metrics["macro_f1"], 4),
                "accuracy_vs_persistence": round(rf_metrics["accuracy"] - persistence_metrics["accuracy"], 4),
                "accuracy_vs_majority": round(rf_metrics["accuracy"] - majority_metrics["accuracy"], 4),
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
    summary_payload = {
        "windows_completed": int(len(walk_forward_df)),
        "label_horizon": label_horizon,
        "purged_rows_per_window": label_horizon,
        "mean_accuracy": round(float(walk_forward_df["accuracy"].mean()), 4),
        "mean_balanced_accuracy": round(float(walk_forward_df["balanced_accuracy"].mean()), 4),
        "mean_macro_f1": round(float(walk_forward_df["macro_f1"].mean()), 4),
        "best_window_accuracy": round(float(walk_forward_df["accuracy"].max()), 4),
        "worst_window_accuracy": round(float(walk_forward_df["accuracy"].min()), 4),
        "mean_persistence_accuracy": round(float(walk_forward_df["persistence_accuracy"].mean()), 4),
        "mean_majority_accuracy": round(float(walk_forward_df["majority_accuracy"].mean()), 4),
        "mean_linear_accuracy": round(float(walk_forward_df["linear_accuracy"].mean()), 4),
        "mean_linear_balanced_accuracy": round(float(walk_forward_df["linear_balanced_accuracy"].mean()), 4),
        "mean_linear_macro_f1": round(float(walk_forward_df["linear_macro_f1"].mean()), 4),
        "mean_accuracy_vs_persistence": round(float(walk_forward_df["accuracy_vs_persistence"].mean()), 4),
        "mean_accuracy_vs_majority": round(float(walk_forward_df["accuracy_vs_majority"].mean()), 4),
        "windows_beating_persistence": int((walk_forward_df["accuracy"] > walk_forward_df["persistence_accuracy"]).sum()),
        "windows_beating_majority": int((walk_forward_df["accuracy"] > walk_forward_df["majority_accuracy"]).sum()),
        "windows_beating_linear_accuracy": int((walk_forward_df["accuracy"] > walk_forward_df["linear_accuracy"]).sum()),
        "windows_beating_linear_balanced_accuracy": int(
            (walk_forward_df["balanced_accuracy"] > walk_forward_df["linear_balanced_accuracy"]).sum()
        ),
        "windows_beating_linear_macro_f1": int((walk_forward_df["macro_f1"] > walk_forward_df["linear_macro_f1"]).sum()),
        "mean_bull_share": round(float(walk_forward_df["bull_share"].mean()), 4),
        "mean_bear_share": round(float(walk_forward_df["bear_share"].mean()), 4),
    }
    (artifacts_dir / "walk_forward_summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    report_lines = [
        "# Walk-Forward Evaluation",
        "",
        f"- Windows completed: {len(walk_forward_df)}",
        f"- Label horizon: {label_horizon} trading row(s)",
        f"- Purged rows per window: {label_horizon}",
        f"- Mean accuracy: {walk_forward_df['accuracy'].mean():.4f}",
        f"- Mean balanced accuracy: {walk_forward_df['balanced_accuracy'].mean():.4f}",
        f"- Mean macro-F1: {walk_forward_df['macro_f1'].mean():.4f}",
        f"- Regularized-logistic mean accuracy: {walk_forward_df['linear_accuracy'].mean():.4f}",
        f"- Regularized-logistic mean balanced accuracy: {walk_forward_df['linear_balanced_accuracy'].mean():.4f}",
        f"- Regularized-logistic mean macro-F1: {walk_forward_df['linear_macro_f1'].mean():.4f}",
        f"- Best window accuracy: {walk_forward_df['accuracy'].max():.4f}",
        f"- Worst window accuracy: {walk_forward_df['accuracy'].min():.4f}",
        f"- Mean persistence accuracy: {walk_forward_df['persistence_accuracy'].mean():.4f}",
        f"- Mean majority accuracy: {walk_forward_df['majority_accuracy'].mean():.4f}",
        f"- Windows beating persistence: {(walk_forward_df['accuracy'] > walk_forward_df['persistence_accuracy']).sum()} / {len(walk_forward_df)}",
        f"- Windows beating majority: {(walk_forward_df['accuracy'] > walk_forward_df['majority_accuracy']).sum()} / {len(walk_forward_df)}",
        f"- Windows beating regularized logistic (accuracy): {(walk_forward_df['accuracy'] > walk_forward_df['linear_accuracy']).sum()} / {len(walk_forward_df)}",
        f"- Windows beating regularized logistic (balanced accuracy): {(walk_forward_df['balanced_accuracy'] > walk_forward_df['linear_balanced_accuracy']).sum()} / {len(walk_forward_df)}",
        f"- Windows beating regularized logistic (macro-F1): {(walk_forward_df['macro_f1'] > walk_forward_df['linear_macro_f1']).sum()} / {len(walk_forward_df)}",
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
    print(f"- {artifacts_dir / 'walk_forward_summary.json'}")


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
    parser.add_argument("--bull-threshold", type=float, default=0.003, help="Future-return cutoff for the bull label.")
    parser.add_argument("--bear-threshold", type=float, default=-0.003, help="Future-return cutoff for the bear label.")
    parser.add_argument(
        "--label-horizon",
        type=int,
        default=1,
        help="Trading rows ahead used to calculate the forward-return label; chronological splits purge the same number of rows.",
    )
    parser.add_argument("--threshold-sweep", action="store_true", help="Evaluate multiple symmetric bull/bear label thresholds.")
    parser.add_argument("--feature-ablation", action="store_true", help="Measure holdout accuracy after dropping one engineered feature at a time.")
    parser.add_argument(
        "--calibrate-probabilities",
        action="store_true",
        help="Apply chronological probability calibration on a trailing training slice before evaluating holdout confidence.",
    )
    parser.add_argument(
        "--calibration-fraction",
        type=float,
        default=0.2,
        help="Fraction of the chronological training window reserved for probability calibration.",
    )
    parser.add_argument(
        "--calibration-method",
        choices=["sigmoid", "isotonic"],
        default="sigmoid",
        help="Calibration method used on the trailing training slice.",
    )
    parser.add_argument(
        "--threshold-sweep-values",
        default="0.002,0.003,0.004,0.005",
        help="Comma-separated positive threshold values used when --threshold-sweep is enabled.",
    )
    parser.add_argument(
        "--model-search",
        action="store_true",
        help="Score multiple random-forest configurations on a chronological validation slice before the final fit.",
    )
    args = parser.parse_args()

    if not args.use_synthetic and args.csv is None:
        raise SystemExit("Provide --csv path/to/file.csv or use --use-synthetic.")
    if args.label_horizon < 1:
        parser.error("--label-horizon must be at least 1.")

    if args.csv is not None:
        raw = load_csv(args.csv)
    else:
        raw = generate_synthetic_prices(points=max(300, args.synthetic_points), seed=args.synthetic_seed)
        raw = validate_price_series(raw)

    label_horizon = args.label_horizon
    processed = build_features(
        raw,
        bull_threshold=args.bull_threshold,
        bear_threshold=args.bear_threshold,
        label_horizon=label_horizon,
    )
    run_training(
        processed,
        args.artifacts,
        model_seed=args.model_seed,
        bull_threshold=args.bull_threshold,
        bear_threshold=args.bear_threshold,
        label_horizon=label_horizon,
        calibrate_probabilities=args.calibrate_probabilities,
        calibration_fraction=min(max(args.calibration_fraction, 0.05), 0.4),
        calibration_method=args.calibration_method,
        model_search=args.model_search,
    )
    if not args.skip_walk_forward:
        run_walk_forward(
            processed,
            args.artifacts,
            windows=max(1, args.walk_forward_windows),
            test_size=max(20, args.walk_forward_test_size),
            model_seed=args.model_seed,
            label_horizon=label_horizon,
        )
    if args.threshold_sweep:
        run_threshold_sweep(
            raw,
            args.artifacts,
            model_seed=args.model_seed,
            threshold_values=parse_threshold_values(args.threshold_sweep_values),
            label_horizon=label_horizon,
        )
    if args.feature_ablation:
        run_feature_ablation(
            processed,
            args.artifacts,
            model_seed=args.model_seed,
            label_horizon=label_horizon,
        )


if __name__ == "__main__":
    main()

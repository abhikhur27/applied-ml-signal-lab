from pathlib import Path

import pandas as pd

from src.train import (
    DEFAULT_MODEL_CONFIG,
    FEATURE_COLS,
    build_baseline_predictions,
    build_benchmark_summary,
    build_features,
    build_linear_challenger,
    chronological_holdout,
    fit_model_with_optional_calibration,
    generate_synthetic_prices,
    load_csv,
    multiclass_brier_score,
    parse_threshold_values,
    select_model_config,
)


def test_parse_threshold_values_sorts_and_deduplicates() -> None:
    assert parse_threshold_values("0.004, 0.002, 0.004,0.003") == [0.002, 0.003, 0.004]


def test_parse_threshold_values_rejects_non_positive_values() -> None:
    try:
        parse_threshold_values("0.003,0")
    except ValueError as exc:
        assert "positive" in str(exc)
        return
    raise AssertionError("Expected parse_threshold_values to reject non-positive values.")


def test_select_model_config_returns_fallback_when_disabled() -> None:
    raw = generate_synthetic_prices(points=600, seed=7)
    processed = build_features(raw)

    selected_config, search_df, summary = select_model_config(
        train_df=processed.iloc[:400].reset_index(drop=True),
        model_seed=42,
        enable_model_search=False,
    )

    assert selected_config["name"] == DEFAULT_MODEL_CONFIG["name"]
    assert search_df.empty
    assert summary["reason"] == "disabled_by_flag"


def test_select_model_config_rejects_candidates_that_miss_baselines() -> None:
    raw = generate_synthetic_prices(points=2200, seed=7)
    processed = build_features(raw)

    selected_config, search_df, summary = select_model_config(
        train_df=processed.iloc[:1400].reset_index(drop=True),
        model_seed=42,
        enable_model_search=True,
    )

    assert summary["validation_rows"] >= 90
    assert len(search_df) >= 4
    top_row = search_df.iloc[0]
    assert not bool(top_row["eligible_for_selection"])
    assert selected_config["name"] == DEFAULT_MODEL_CONFIG["name"]
    assert summary["applied"] is False
    assert summary["reason"] == "no_candidate_beat_both_baselines"
    assert all(
        (
            (top_row["validation_accuracy"] > row["validation_accuracy"])
            or (
                top_row["validation_accuracy"] == row["validation_accuracy"]
                and top_row["accuracy_vs_persistence"] >= row["accuracy_vs_persistence"]
            )
        )
        for _, row in search_df.iterrows()
    )


def test_build_features_aligns_and_drops_multi_day_labels() -> None:
    raw = generate_synthetic_prices(points=160, seed=7)
    processed = build_features(raw, bull_threshold=0.01, bear_threshold=-0.01, label_horizon=5)

    last = processed.iloc[-1]
    assert last["label_end_date"] == raw.iloc[-1]["date"]
    expected_return = raw.iloc[-1]["close"] / last["close"] - 1
    assert abs(last["forward_return"] - expected_return) < 1e-12
    assert processed["label_end_date"].notna().all()
    assert processed["forward_return"].notna().all()


def test_chronological_holdout_purges_overlapping_labels() -> None:
    processed = build_features(generate_synthetic_prices(points=300, seed=7), label_horizon=5)
    train_df, test_df, history = chronological_holdout(processed, label_horizon=5)

    assert len(history) == len(train_df) + 5
    assert train_df.iloc[-1]["label_end_date"] < test_df.iloc[0]["date"]
    assert processed.iloc[len(history) - 1]["label_end_date"] >= test_df.iloc[0]["date"]


def test_persistence_baseline_uses_observable_horizon_lag() -> None:
    train_df = pd.DataFrame({"target": [0, 0, 1, 1]})
    test_df = pd.DataFrame({"target": [0, 2, 1, 2, 0]})
    observable_history = pd.Series([0, 1, 2, 0, 1])

    majority, persistence = build_baseline_predictions(
        train_df,
        test_df,
        label_horizon=3,
        observable_target_history=observable_history,
    )

    assert majority.tolist() == [0, 0, 0, 0, 0]
    assert persistence.tolist() == [2, 0, 1, 0, 2]


def test_uncalibrated_training_path_fits_without_name_error() -> None:
    processed = build_features(generate_synthetic_prices(points=500, seed=7))
    train_df, _, _ = chronological_holdout(processed, label_horizon=1)
    small_config = {
        "name": "test_forest",
        "n_estimators": 10,
        "min_samples_leaf": 4,
        "max_depth": 5,
    }

    model, details = fit_model_with_optional_calibration(
        train_df=train_df,
        model_seed=42,
        calibrate_probabilities=False,
        calibration_fraction=0.2,
        calibration_method="sigmoid",
        model_config=small_config,
    )

    assert details["metadata"]["applied"] is False
    assert len(model.predict(train_df.iloc[-5:][list(model.feature_names_in_)])) == 5


def test_multiclass_brier_score_averages_one_vs_rest_scores() -> None:
    y_true = pd.Series([0, 1, 2]).to_numpy()
    probabilities = pd.DataFrame(
        [
            [0.8, 0.1, 0.1],
            [0.2, 0.6, 0.2],
            [0.1, 0.2, 0.7],
        ]
    ).to_numpy()

    assert abs(multiclass_brier_score(y_true, probabilities, pd.Series([0, 1, 2]).to_numpy()) - 0.0488888889) < 1e-9


def test_benchmark_summary_exposes_class_coverage_beyond_accuracy() -> None:
    y_true = pd.Series([0, 1, 2, 0, 1, 2])
    summary = build_benchmark_summary(
        y_true=y_true,
        rf_preds=pd.Series([0, 0, 2, 0, 1, 1]).to_numpy(),
        linear_preds=y_true.to_numpy(),
        majority_preds=pd.Series([1, 1, 1, 1, 1, 1]).to_numpy(),
        persistence_preds=pd.Series([2, 0, 1, 2, 0, 1]).to_numpy(),
    ).set_index("model")

    assert summary.loc["regularized_logistic", "macro_f1"] == 1.0
    assert summary.loc["majority_class", "accuracy"] == 0.3333
    assert summary.loc["majority_class", "balanced_accuracy"] == 0.3333
    assert summary.loc["majority_class", "macro_f1"] == 0.1667


def test_regularized_linear_challenger_scales_features_and_covers_real_fixture_classes() -> None:
    fixture = load_csv(Path("data/ecb_eur_usd_2012_2024.csv"))
    processed = build_features(fixture, bull_threshold=0.008, bear_threshold=-0.008, label_horizon=5)
    train_df, test_df, _ = chronological_holdout(processed, label_horizon=5)
    model = build_linear_challenger(model_seed=42)

    model.fit(train_df[FEATURE_COLS], train_df["target"])
    predictions = model.predict(test_df[FEATURE_COLS])

    assert list(model.named_steps) == ["scale", "classifier"]
    assert set(predictions) == {0, 1, 2}


def test_real_fixture_rejects_calibration_prediction_collapse() -> None:
    fixture = load_csv(Path("data/ecb_eur_usd_2012_2024.csv"))
    processed = build_features(fixture, bull_threshold=0.008, bear_threshold=-0.008, label_horizon=5)
    train_df, _, _ = chronological_holdout(processed, label_horizon=5)
    test_config = {
        "name": "test_forest",
        "n_estimators": 40,
        "min_samples_leaf": 6,
        "max_depth": 8,
    }

    _, details = fit_model_with_optional_calibration(
        train_df=train_df,
        model_seed=42,
        calibrate_probabilities=True,
        calibration_fraction=0.2,
        calibration_method="sigmoid",
        model_config=test_config,
        label_horizon=5,
    )

    metadata = details["metadata"]
    assert metadata["applied"] is False
    assert metadata["reason"] == "audit_rejected_prediction_collapse"
    assert metadata["baseline_prediction_classes"] == 3
    assert metadata["calibrated_prediction_classes"] == 1

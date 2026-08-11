from src.train import (
    DEFAULT_MODEL_CONFIG,
    build_features,
    generate_synthetic_prices,
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


def test_select_model_config_builds_ranked_search_report() -> None:
    raw = generate_synthetic_prices(points=2200, seed=7)
    processed = build_features(raw)

    selected_config, search_df, summary = select_model_config(
        train_df=processed.iloc[:1400].reset_index(drop=True),
        model_seed=42,
        enable_model_search=True,
    )

    assert summary["applied"] is True
    assert summary["validation_rows"] >= 90
    assert len(search_df) >= 4
    assert search_df.iloc[0]["config_name"] == selected_config["name"]
    top_row = search_df.iloc[0]
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

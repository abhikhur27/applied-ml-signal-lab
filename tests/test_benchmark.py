from src.benchmark import evaluate_promotion_gate


def make_instrument(
    *,
    holdout_joint_win: bool = True,
    balanced_delta: float = 0.02,
    macro_f1_delta: float = 0.02,
    joint_wins: int = 5,
    windows: int = 6,
    asset_family: str = "foreign_exchange",
) -> dict[str, object]:
    return {
        "asset_family": asset_family,
        "holdout": {
            "linear_joint_balanced_metric_win": holdout_joint_win,
            "linear_prediction_classes": 3,
        },
        "walk_forward": {
            "windows_completed": windows,
            "linear_joint_balanced_metric_wins": joint_wins,
            "linear_positive_balanced_metric_means": balanced_delta > 0 and macro_f1_delta > 0,
            "linear_mean_deltas": {
                "balanced_accuracy": balanced_delta,
                "macro_f1": macro_f1_delta,
            },
        },
    }


def gate_config() -> dict[str, object]:
    return {
        "incumbent": "random_forest",
        "challenger": "regularized_logistic",
        "minimum_instruments": 3,
        "minimum_asset_families": 2,
        "minimum_asset_family_holdout_joint_win_rate": 0.5,
        "minimum_asset_families_with_holdout_joint_wins": 2,
        "minimum_asset_family_walk_forward_joint_win_rate": 0.3,
        "minimum_asset_families_with_positive_walk_forward_means": 2,
        "maximum_asset_family_mean_metric_regression": -0.01,
        "minimum_holdout_joint_win_rate": 0.66,
        "minimum_walk_forward_joint_win_rate": 0.66,
        "minimum_instruments_with_positive_walk_forward_means": 2,
        "minimum_overall_mean_balanced_accuracy_delta": 0.01,
        "minimum_overall_mean_macro_f1_delta": 0.01,
        "maximum_instrument_mean_metric_regression": -0.01,
        "minimum_prediction_classes_per_instrument": 3,
    }


def test_promotion_gate_requires_broad_balanced_metric_wins() -> None:
    instruments = [
        make_instrument(),
        make_instrument(),
        make_instrument(joint_wins=2, asset_family="interest_rates"),
    ]

    result = evaluate_promotion_gate(instruments, gate_config())

    assert result["eligible_for_promotion"] is True
    assert result["decision"] == "promote_regularized_logistic"
    assert result["holdout_joint_wins"] == 3
    assert result["walk_forward_joint_wins"] == 12
    assert result["asset_family_count"] == 2


def test_promotion_gate_blocks_material_instrument_regression() -> None:
    instruments = [
        make_instrument(),
        make_instrument(),
        make_instrument(
            balanced_delta=-0.02,
            macro_f1_delta=0.01,
            joint_wins=4,
            asset_family="interest_rates",
        ),
    ]

    result = evaluate_promotion_gate(instruments, gate_config())

    assert result["eligible_for_promotion"] is False
    assert result["decision"] == "retain_random_forest"
    regression_check = next(
        requirement
        for requirement in result["requirements"]
        if requirement["name"] == "worst instrument mean balanced-metric regression"
    )
    assert regression_check["passed"] is False


def test_promotion_gate_blocks_challenger_class_collapse() -> None:
    instruments = [
        make_instrument(),
        make_instrument(),
        make_instrument(asset_family="interest_rates"),
    ]
    instruments[1]["holdout"]["linear_prediction_classes"] = 2

    result = evaluate_promotion_gate(instruments, gate_config())

    assert result["eligible_for_promotion"] is False
    assert result["minimum_prediction_classes_per_instrument"] == 2


def test_promotion_gate_requires_independent_asset_family_evidence() -> None:
    instruments = [make_instrument(), make_instrument(), make_instrument()]

    result = evaluate_promotion_gate(instruments, gate_config())

    assert result["eligible_for_promotion"] is False
    family_check = next(
        requirement
        for requirement in result["requirements"]
        if requirement["name"] == "asset-family breadth"
    )
    assert family_check["passed"] is False

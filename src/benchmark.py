from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.train import build_features, load_csv, run_training, run_walk_forward


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = ROOT / "benchmarks" / "ecb_fx_contract.json"
BALANCED_METRICS = ("balanced_accuracy", "macro_f1")


def add_check(checks: list[dict[str, Any]], name: str, passed: bool, observed: Any, expected: str) -> None:
    checks.append(
        {
            "name": name,
            "passed": bool(passed),
            "observed": observed,
            "expected": expected,
        }
    )


def benchmark_rows(model_summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["model"]: row for row in model_summary["benchmark_accuracy"]}


def summarize_instrument(
    instrument: dict[str, Any],
    metadata: dict[str, Any],
    model_summary: dict[str, Any],
    walk_forward: dict[str, Any],
    walk_forward_metrics: pd.DataFrame,
) -> dict[str, Any]:
    rows = benchmark_rows(model_summary)
    forest = rows["random_forest"]
    linear = rows["regularized_logistic"]
    holdout_deltas = {
        metric: round(float(linear[metric] - forest[metric]), 4)
        for metric in ("accuracy", *BALANCED_METRICS)
    }

    window_deltas = pd.DataFrame(
        {
            "balanced_accuracy": (
                walk_forward_metrics["linear_balanced_accuracy"] - walk_forward_metrics["balanced_accuracy"]
            ),
            "macro_f1": walk_forward_metrics["linear_macro_f1"] - walk_forward_metrics["macro_f1"],
        }
    )
    joint_wins = (window_deltas["balanced_accuracy"] > 0) & (window_deltas["macro_f1"] > 0)
    mean_deltas = {
        metric: round(float(window_deltas[metric].mean()), 4)
        for metric in BALANCED_METRICS
    }
    linear_prediction_classes = sum(linear[f"{label}_share"] > 0 for label in ("bear", "neutral", "bull"))

    return {
        "pair": instrument["pair"],
        "slug": instrument["slug"],
        "fixture": instrument["fixture"],
        "fixture_rows": metadata["rows"],
        "holdout": {
            "random_forest": {metric: forest[metric] for metric in ("accuracy", *BALANCED_METRICS)},
            "regularized_logistic": {metric: linear[metric] for metric in ("accuracy", *BALANCED_METRICS)},
            "linear_deltas": holdout_deltas,
            "linear_joint_balanced_metric_win": all(holdout_deltas[metric] > 0 for metric in BALANCED_METRICS),
            "linear_prediction_classes": linear_prediction_classes,
        },
        "walk_forward": {
            **walk_forward,
            "linear_mean_deltas": mean_deltas,
            "linear_joint_balanced_metric_wins": int(joint_wins.sum()),
            "linear_joint_balanced_metric_win_rate": round(float(joint_wins.mean()), 4),
            "linear_positive_balanced_metric_means": all(mean_deltas[metric] > 0 for metric in BALANCED_METRICS),
        },
        "calibration": model_summary["probability_calibration"],
    }


def evaluate_reference_expectations(
    checks: list[dict[str, Any]],
    instrument: dict[str, Any],
    model_summary: dict[str, Any],
    walk_forward: dict[str, Any],
) -> None:
    expected = instrument.get("expectations")
    if not expected:
        return

    prefix = instrument["pair"]
    rows = benchmark_rows(model_summary)
    benchmarks = {name: row["accuracy"] for name, row in rows.items()}
    forest = rows["random_forest"]
    linear = rows["regularized_logistic"]
    calibration = model_summary["probability_calibration"]

    add_check(
        checks,
        f"{prefix} majority baseline advantage",
        benchmarks["majority_class"] - benchmarks["random_forest"] >= expected["minimum_majority_advantage"],
        round(benchmarks["majority_class"] - benchmarks["random_forest"], 4),
        f">= {expected['minimum_majority_advantage']}",
    )
    add_check(
        checks,
        f"{prefix} persistence baseline advantage",
        benchmarks["persistence"] - benchmarks["random_forest"] >= expected["minimum_persistence_advantage"],
        round(benchmarks["persistence"] - benchmarks["random_forest"], 4),
        f">= {expected['minimum_persistence_advantage']}",
    )
    for metric, expectation_key in (
        ("accuracy", "minimum_linear_accuracy_advantage"),
        ("balanced_accuracy", "minimum_linear_balanced_accuracy_advantage"),
        ("macro_f1", "minimum_linear_macro_f1_advantage"),
    ):
        advantage = round(float(linear[metric] - forest[metric]), 4)
        add_check(
            checks,
            f"{prefix} regularized-linear holdout {metric.replace('_', '-')} advantage",
            advantage >= expected[expectation_key],
            advantage,
            f">= {expected[expectation_key]}",
        )

    linear_prediction_classes = sum(linear[f"{label}_share"] > 0 for label in ("bear", "neutral", "bull"))
    add_check(
        checks,
        f"{prefix} regularized-linear class coverage",
        linear_prediction_classes == expected["linear_prediction_classes"],
        linear_prediction_classes,
        str(expected["linear_prediction_classes"]),
    )
    add_check(
        checks,
        f"{prefix} calibration collapse guard",
        not calibration["applied"] and calibration["reason"] == expected["calibration_rejection_reason"],
        {"applied": calibration["applied"], "reason": calibration["reason"]},
        f"rejected with {expected['calibration_rejection_reason']}",
    )
    add_check(
        checks,
        f"{prefix} calibration audit class diversity",
        calibration["calibrated_prediction_classes"] == expected["calibrated_audit_prediction_classes"],
        calibration["calibrated_prediction_classes"],
        str(expected["calibrated_audit_prediction_classes"]),
    )

    accuracy_range = round(walk_forward["best_window_accuracy"] - walk_forward["worst_window_accuracy"], 4)
    add_check(
        checks,
        f"{prefix} walk-forward regime spread",
        accuracy_range >= expected["minimum_walk_forward_accuracy_range"],
        accuracy_range,
        f">= {expected['minimum_walk_forward_accuracy_range']}",
    )
    persistence_wins = walk_forward["windows_beating_persistence"]
    add_check(
        checks,
        f"{prefix} regime-dependent persistence comparison",
        expected["minimum_windows_beating_persistence"]
        <= persistence_wins
        <= expected["maximum_windows_beating_persistence"],
        persistence_wins,
        f"{expected['minimum_windows_beating_persistence']}..{expected['maximum_windows_beating_persistence']}",
    )
    add_check(
        checks,
        f"{prefix} majority baseline remains unbeaten",
        walk_forward["windows_beating_majority"] <= expected["maximum_windows_beating_majority"],
        walk_forward["windows_beating_majority"],
        f"<= {expected['maximum_windows_beating_majority']}",
    )
    linear_mean_advantage = round(walk_forward["mean_linear_accuracy"] - walk_forward["mean_accuracy"], 4)
    add_check(
        checks,
        f"{prefix} regularized-linear walk-forward mean advantage",
        linear_mean_advantage >= expected["minimum_linear_walk_forward_mean_advantage"],
        linear_mean_advantage,
        f">= {expected['minimum_linear_walk_forward_mean_advantage']}",
    )
    add_check(
        checks,
        f"{prefix} forest retains class-sensitive regime wins",
        walk_forward["windows_beating_linear_macro_f1"]
        >= expected["minimum_forest_windows_beating_linear_macro_f1"],
        walk_forward["windows_beating_linear_macro_f1"],
        f">= {expected['minimum_forest_windows_beating_linear_macro_f1']}",
    )


def evaluate_promotion_gate(
    instruments: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    instrument_count = len(instruments)
    holdout_joint_wins = sum(
        int(instrument["holdout"]["linear_joint_balanced_metric_win"])
        for instrument in instruments
    )
    holdout_joint_win_rate = holdout_joint_wins / instrument_count if instrument_count else 0.0
    total_windows = sum(instrument["walk_forward"]["windows_completed"] for instrument in instruments)
    walk_forward_joint_wins = sum(
        instrument["walk_forward"]["linear_joint_balanced_metric_wins"]
        for instrument in instruments
    )
    walk_forward_joint_win_rate = walk_forward_joint_wins / total_windows if total_windows else 0.0
    instruments_with_positive_means = sum(
        int(instrument["walk_forward"]["linear_positive_balanced_metric_means"])
        for instrument in instruments
    )
    minimum_prediction_classes = min(
        (instrument["holdout"]["linear_prediction_classes"] for instrument in instruments),
        default=0,
    )

    window_weighted_deltas = {}
    for metric in BALANCED_METRICS:
        numerator = sum(
            instrument["walk_forward"]["linear_mean_deltas"][metric]
            * instrument["walk_forward"]["windows_completed"]
            for instrument in instruments
        )
        window_weighted_deltas[metric] = round(numerator / total_windows, 4) if total_windows else 0.0
    worst_instrument_mean_delta = min(
        (
            instrument["walk_forward"]["linear_mean_deltas"][metric]
            for instrument in instruments
            for metric in BALANCED_METRICS
        ),
        default=0.0,
    )

    requirements = [
        {
            "name": "instrument breadth",
            "passed": instrument_count >= config["minimum_instruments"],
            "observed": instrument_count,
            "expected": f">= {config['minimum_instruments']}",
        },
        {
            "name": "holdout joint balanced-metric win rate",
            "passed": holdout_joint_win_rate >= config["minimum_holdout_joint_win_rate"],
            "observed": round(holdout_joint_win_rate, 4),
            "expected": f">= {config['minimum_holdout_joint_win_rate']}",
        },
        {
            "name": "walk-forward joint balanced-metric win rate",
            "passed": walk_forward_joint_win_rate >= config["minimum_walk_forward_joint_win_rate"],
            "observed": round(walk_forward_joint_win_rate, 4),
            "expected": f">= {config['minimum_walk_forward_joint_win_rate']}",
        },
        {
            "name": "instruments with positive walk-forward balanced-metric means",
            "passed": instruments_with_positive_means
            >= config["minimum_instruments_with_positive_walk_forward_means"],
            "observed": instruments_with_positive_means,
            "expected": f">= {config['minimum_instruments_with_positive_walk_forward_means']}",
        },
        {
            "name": "overall mean balanced-accuracy delta",
            "passed": window_weighted_deltas["balanced_accuracy"]
            >= config["minimum_overall_mean_balanced_accuracy_delta"],
            "observed": window_weighted_deltas["balanced_accuracy"],
            "expected": f">= {config['minimum_overall_mean_balanced_accuracy_delta']}",
        },
        {
            "name": "overall mean macro-F1 delta",
            "passed": window_weighted_deltas["macro_f1"] >= config["minimum_overall_mean_macro_f1_delta"],
            "observed": window_weighted_deltas["macro_f1"],
            "expected": f">= {config['minimum_overall_mean_macro_f1_delta']}",
        },
        {
            "name": "worst instrument mean balanced-metric regression",
            "passed": worst_instrument_mean_delta >= config["maximum_instrument_mean_metric_regression"],
            "observed": round(worst_instrument_mean_delta, 4),
            "expected": f">= {config['maximum_instrument_mean_metric_regression']}",
        },
        {
            "name": "challenger class coverage per instrument",
            "passed": minimum_prediction_classes >= config["minimum_prediction_classes_per_instrument"],
            "observed": minimum_prediction_classes,
            "expected": f">= {config['minimum_prediction_classes_per_instrument']}",
        },
    ]
    passed = all(requirement["passed"] for requirement in requirements)
    return {
        "incumbent": config["incumbent"],
        "challenger": config["challenger"],
        "decision": f"promote_{config['challenger']}" if passed else f"retain_{config['incumbent']}",
        "eligible_for_promotion": passed,
        "instrument_count": instrument_count,
        "holdout_joint_wins": holdout_joint_wins,
        "holdout_joint_win_rate": round(holdout_joint_win_rate, 4),
        "walk_forward_windows": total_windows,
        "walk_forward_joint_wins": walk_forward_joint_wins,
        "walk_forward_joint_win_rate": round(walk_forward_joint_win_rate, 4),
        "instruments_with_positive_walk_forward_means": instruments_with_positive_means,
        "minimum_prediction_classes_per_instrument": minimum_prediction_classes,
        "overall_mean_deltas": window_weighted_deltas,
        "worst_instrument_mean_metric_delta": round(worst_instrument_mean_delta, 4),
        "requirements": requirements,
    }


def validate_instrument(
    checks: list[dict[str, Any]],
    instrument: dict[str, Any],
    metadata: dict[str, Any],
    fixture_bytes: bytes,
    walk_forward: dict[str, Any],
    expected_windows: int,
) -> None:
    prefix = instrument["pair"]
    fixture_hash = hashlib.sha256(fixture_bytes).hexdigest()
    add_check(
        checks,
        f"{prefix} fixture row count",
        metadata["rows"] == instrument["fixture_rows"],
        metadata["rows"],
        str(instrument["fixture_rows"]),
    )
    add_check(
        checks,
        f"{prefix} fixture checksum",
        fixture_hash == instrument["fixture_sha256"] == metadata["fixture_sha256"],
        fixture_hash,
        instrument["fixture_sha256"],
    )
    add_check(
        checks,
        f"{prefix} metadata identity",
        metadata.get("pair") == instrument["pair"],
        metadata.get("pair"),
        instrument["pair"],
    )
    add_check(
        checks,
        f"{prefix} walk-forward window count",
        walk_forward["windows_completed"] == expected_windows,
        walk_forward["windows_completed"],
        str(expected_windows),
    )


def promotion_evidence_frame(instruments: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for instrument in instruments:
        holdout = instrument["holdout"]
        walk_forward = instrument["walk_forward"]
        rows.append(
            {
                "pair": instrument["pair"],
                "holdout_balanced_accuracy_delta": holdout["linear_deltas"]["balanced_accuracy"],
                "holdout_macro_f1_delta": holdout["linear_deltas"]["macro_f1"],
                "holdout_joint_win": holdout["linear_joint_balanced_metric_win"],
                "walk_forward_balanced_accuracy_delta": walk_forward["linear_mean_deltas"]["balanced_accuracy"],
                "walk_forward_macro_f1_delta": walk_forward["linear_mean_deltas"]["macro_f1"],
                "walk_forward_joint_wins": walk_forward["linear_joint_balanced_metric_wins"],
                "walk_forward_windows": walk_forward["windows_completed"],
                "walk_forward_joint_win_rate": walk_forward["linear_joint_balanced_metric_win_rate"],
            }
        )
    return pd.DataFrame(rows)


def render_suite_report(result: dict[str, Any]) -> str:
    promotion = result["promotion_gate"]
    lines = [
        "# ECB multi-instrument benchmark contract",
        "",
        f"- Contract status: {result['status'].upper()}",
        f"- Model-family decision: `{promotion['decision']}`",
        f"- Instruments: {promotion['instrument_count']}",
        f"- Walk-forward regimes: {promotion['walk_forward_windows']}",
        (
            "- Challenger joint balanced-metric wins: "
            f"{promotion['walk_forward_joint_wins']} / {promotion['walk_forward_windows']} "
            f"({promotion['walk_forward_joint_win_rate']:.4f})"
        ),
        "",
        "## Promotion evidence",
        "",
    ]
    for instrument in result["instruments"]:
        holdout = instrument["holdout"]["linear_deltas"]
        walk_forward = instrument["walk_forward"]
        lines.append(
            f"- {instrument['pair']}: holdout Δ balanced accuracy {holdout['balanced_accuracy']:+.4f}, "
            f"Δ macro-F1 {holdout['macro_f1']:+.4f}; walk-forward mean Δ balanced accuracy "
            f"{walk_forward['linear_mean_deltas']['balanced_accuracy']:+.4f}, Δ macro-F1 "
            f"{walk_forward['linear_mean_deltas']['macro_f1']:+.4f}; joint wins "
            f"{walk_forward['linear_joint_balanced_metric_wins']}/{walk_forward['windows_completed']}"
        )
    lines.extend(["", "## Promotion requirements", ""])
    lines.extend(
        f"- {'PASS' if requirement['passed'] else 'FAIL'} — {requirement['name']}: "
        f"observed `{requirement['observed']}`; expected {requirement['expected']}"
        for requirement in promotion["requirements"]
    )
    lines.extend(["", "## Contract checks", ""])
    lines.extend(
        f"- {'PASS' if check['passed'] else 'FAIL'} — {check['name']}: "
        f"observed `{check['observed']}`; expected {check['expected']}"
        for check in result["checks"]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the frozen ECB multi-instrument benchmark contract.")
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts/ecb-benchmark"))
    args = parser.parse_args()

    contract_path = args.contract.resolve()
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    parameters = contract["parameters"]
    checks: list[dict[str, Any]] = []
    instrument_results = []

    for instrument in contract["instruments"]:
        fixture_path = ROOT / instrument["fixture"]
        metadata_path = ROOT / instrument["metadata"]
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        instrument_artifacts = args.artifacts / instrument["slug"]
        raw = load_csv(fixture_path)
        processed = build_features(
            raw,
            bull_threshold=parameters["bull_threshold"],
            bear_threshold=parameters["bear_threshold"],
            label_horizon=parameters["label_horizon"],
        )
        run_training(
            processed,
            instrument_artifacts,
            model_seed=parameters["model_seed"],
            bull_threshold=parameters["bull_threshold"],
            bear_threshold=parameters["bear_threshold"],
            label_horizon=parameters["label_horizon"],
            calibrate_probabilities=True,
            calibration_fraction=parameters["calibration_fraction"],
            calibration_method=parameters["calibration_method"],
            model_search=False,
        )
        run_walk_forward(
            processed,
            instrument_artifacts,
            windows=parameters["walk_forward_windows"],
            test_size=parameters["walk_forward_test_size"],
            model_seed=parameters["model_seed"],
            label_horizon=parameters["label_horizon"],
        )

        model_summary = json.loads((instrument_artifacts / "model_summary.json").read_text(encoding="utf-8"))
        walk_forward = json.loads(
            (instrument_artifacts / "walk_forward_summary.json").read_text(encoding="utf-8")
        )
        walk_forward_metrics = pd.read_csv(instrument_artifacts / "walk_forward_metrics.csv")
        validate_instrument(
            checks,
            instrument,
            metadata,
            fixture_path.read_bytes(),
            walk_forward,
            parameters["walk_forward_windows"],
        )
        evaluate_reference_expectations(checks, instrument, model_summary, walk_forward)
        instrument_results.append(
            summarize_instrument(instrument, metadata, model_summary, walk_forward, walk_forward_metrics)
        )

    promotion = evaluate_promotion_gate(instrument_results, contract["promotion_gate"])
    add_check(
        checks,
        "model-family promotion decision",
        promotion["decision"] == contract["promotion_gate"]["expected_decision"],
        promotion["decision"],
        contract["promotion_gate"]["expected_decision"],
    )
    result = {
        "status": "pass" if all(check["passed"] for check in checks) else "fail",
        "contract": contract_path.relative_to(ROOT).as_posix(),
        "parameters": parameters,
        "instruments": instrument_results,
        "promotion_gate": promotion,
        "checks": checks,
    }

    args.artifacts.mkdir(parents=True, exist_ok=True)
    (args.artifacts / "benchmark_suite_results.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.artifacts / "benchmark_suite_report.md").write_text(render_suite_report(result), encoding="utf-8")
    promotion_evidence_frame(instrument_results).to_csv(
        args.artifacts / "model_family_promotion.csv",
        index=False,
    )

    print("\nECB multi-instrument contract checks:")
    for check in checks:
        print(f"{'PASS' if check['passed'] else 'FAIL'}: {check['name']} ({check['observed']})")
    print(f"Model-family decision: {promotion['decision']}")
    if result["status"] != "pass":
        raise SystemExit("ECB benchmark contract failed; inspect benchmark_suite_results.json.")


if __name__ == "__main__":
    main()

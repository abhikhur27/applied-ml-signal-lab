from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from src.train import build_features, load_csv, run_training, run_walk_forward


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = ROOT / "benchmarks" / "ecb_eur_usd_contract.json"


def add_check(checks: list[dict[str, Any]], name: str, passed: bool, observed: Any, expected: str) -> None:
    checks.append(
        {
            "name": name,
            "passed": bool(passed),
            "observed": observed,
            "expected": expected,
        }
    )


def evaluate_contract(
    contract: dict[str, Any],
    metadata: dict[str, Any],
    fixture_bytes: bytes,
    model_summary: dict[str, Any],
    walk_forward: dict[str, Any],
) -> list[dict[str, Any]]:
    expected = contract["expectations"]
    benchmarks = {row["model"]: row["accuracy"] for row in model_summary["benchmark_accuracy"]}
    calibration = model_summary["probability_calibration"]
    checks: list[dict[str, Any]] = []

    fixture_hash = hashlib.sha256(fixture_bytes).hexdigest()
    add_check(checks, "fixture row count", metadata["rows"] == expected["fixture_rows"], metadata["rows"], str(expected["fixture_rows"]))
    add_check(checks, "fixture checksum", fixture_hash == metadata["fixture_sha256"], fixture_hash, metadata["fixture_sha256"])
    add_check(
        checks,
        "majority baseline advantage",
        benchmarks["majority_class"] - benchmarks["random_forest"] >= expected["minimum_majority_advantage"],
        round(benchmarks["majority_class"] - benchmarks["random_forest"], 4),
        f">= {expected['minimum_majority_advantage']}",
    )
    add_check(
        checks,
        "persistence baseline advantage",
        benchmarks["persistence"] - benchmarks["random_forest"] >= expected["minimum_persistence_advantage"],
        round(benchmarks["persistence"] - benchmarks["random_forest"], 4),
        f">= {expected['minimum_persistence_advantage']}",
    )
    add_check(
        checks,
        "calibration collapse guard",
        not calibration["applied"] and calibration["reason"] == expected["calibration_rejection_reason"],
        {"applied": calibration["applied"], "reason": calibration["reason"]},
        f"rejected with {expected['calibration_rejection_reason']}",
    )
    add_check(
        checks,
        "calibration audit class diversity",
        calibration["calibrated_prediction_classes"] == expected["calibrated_audit_prediction_classes"],
        calibration["calibrated_prediction_classes"],
        str(expected["calibrated_audit_prediction_classes"]),
    )
    add_check(
        checks,
        "walk-forward window count",
        walk_forward["windows_completed"] == expected["walk_forward_windows"],
        walk_forward["windows_completed"],
        str(expected["walk_forward_windows"]),
    )
    accuracy_range = round(walk_forward["best_window_accuracy"] - walk_forward["worst_window_accuracy"], 4)
    add_check(
        checks,
        "walk-forward regime spread",
        accuracy_range >= expected["minimum_walk_forward_accuracy_range"],
        accuracy_range,
        f">= {expected['minimum_walk_forward_accuracy_range']}",
    )
    persistence_wins = walk_forward["windows_beating_persistence"]
    add_check(
        checks,
        "regime-dependent persistence comparison",
        expected["minimum_windows_beating_persistence"] <= persistence_wins <= expected["maximum_windows_beating_persistence"],
        persistence_wins,
        f"{expected['minimum_windows_beating_persistence']}..{expected['maximum_windows_beating_persistence']}",
    )
    add_check(
        checks,
        "majority baseline remains unbeaten",
        walk_forward["windows_beating_majority"] <= expected["maximum_windows_beating_majority"],
        walk_forward["windows_beating_majority"],
        f"<= {expected['maximum_windows_beating_majority']}",
    )
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the frozen ECB real-data benchmark contract.")
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts/ecb-benchmark"))
    args = parser.parse_args()

    contract_path = args.contract.resolve()
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    fixture_path = ROOT / contract["fixture"]
    metadata_path = ROOT / contract["metadata"]
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    parameters = contract["parameters"]

    raw = load_csv(fixture_path)
    processed = build_features(
        raw,
        bull_threshold=parameters["bull_threshold"],
        bear_threshold=parameters["bear_threshold"],
        label_horizon=parameters["label_horizon"],
    )
    run_training(
        processed,
        args.artifacts,
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
        args.artifacts,
        windows=parameters["walk_forward_windows"],
        test_size=parameters["walk_forward_test_size"],
        model_seed=parameters["model_seed"],
        label_horizon=parameters["label_horizon"],
    )

    model_summary = json.loads((args.artifacts / "model_summary.json").read_text(encoding="utf-8"))
    walk_forward = json.loads((args.artifacts / "walk_forward_summary.json").read_text(encoding="utf-8"))
    checks = evaluate_contract(contract, metadata, fixture_path.read_bytes(), model_summary, walk_forward)
    result = {
        "status": "pass" if all(check["passed"] for check in checks) else "fail",
        "contract": contract_path.relative_to(ROOT).as_posix(),
        "fixture": contract["fixture"],
        "parameters": parameters,
        "observations": {
            "holdout_accuracy": model_summary["test_accuracy"],
            "benchmark_accuracy": model_summary["benchmark_accuracy"],
            "calibration": model_summary["probability_calibration"],
            "walk_forward": walk_forward,
        },
        "checks": checks,
    }
    args.artifacts.mkdir(parents=True, exist_ok=True)
    (args.artifacts / "benchmark_contract_results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    report = [
        "# ECB EUR/USD benchmark contract",
        "",
        f"- Status: {result['status'].upper()}",
        f"- Holdout accuracy: {model_summary['test_accuracy']:.4f}",
        f"- Walk-forward mean accuracy: {walk_forward['mean_accuracy']:.4f}",
        f"- Calibration decision: {model_summary['probability_calibration']['reason']}",
        "",
        "## Checks",
        "",
    ]
    report.extend(
        f"- {'PASS' if check['passed'] else 'FAIL'} — {check['name']}: observed `{check['observed']}`; expected {check['expected']}"
        for check in checks
    )
    (args.artifacts / "benchmark_contract_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    for check in checks:
        print(f"{'PASS' if check['passed'] else 'FAIL'}: {check['name']} ({check['observed']})")
    if result["status"] != "pass":
        raise SystemExit("ECB benchmark contract failed; inspect benchmark_contract_results.json.")


if __name__ == "__main__":
    main()

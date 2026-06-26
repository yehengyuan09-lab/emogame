"""Run an ML calibration loop for score-vs-sales gaps."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.skin_repository import DEFAULT_DB_PATH  # noqa: E402
from models.sales_calibration import collect_calibration_samples, fit_until_gap_probability  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate sales-blind score to public sales evidence.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH, help="Path to skins.sqlite3.")
    parser.add_argument("--limit", type=int, help="Limit number of samples.")
    parser.add_argument("--gap-threshold", type=int, default=10, help="Failure threshold for absolute gap.")
    parser.add_argument("--target-probability", type=float, default=0.10, help="Target P(abs(gap)>threshold).")
    parser.add_argument("--alpha", type=float, default=0.10, help="One-sided binomial test alpha.")
    parser.add_argument("--max-iterations", type=int, default=500)
    parser.add_argument("--write-model", type=Path, help="Write calibrated model JSON.")
    parser.add_argument("--write-report", type=Path, help="Write full calibration report JSON.")
    parser.add_argument(
        "--include-non-official",
        action="store_true",
        help="Include non-official public evidence. Default is official-only.",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    return parser.parse_args()


def serializable_report(result: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in result.items()
        if key != "model"
    }


def print_text(report: dict[str, Any]) -> None:
    print("sales score calibration")
    print("=" * 44)
    print(f"evidence scope: {report.get('evidence_scope', 'official_only')}")
    print(f"samples:       {report['calibrated']['n']}")
    print(f"iterations:    {report['iterations']}")
    print(f"minimum n:     {report['minimum_samples_for_zero_failures']} for zero failures")
    print(
        "baseline:      "
        f"{report['baseline']['exceedances']}/{report['baseline']['n']} failures, "
        f"MAE {report['baseline']['mae']}, p={report['baseline']['binomial_p_value_at_target_probability']}"
    )
    print(
        "calibrated:    "
        f"{report['calibrated']['exceedances']}/{report['calibrated']['n']} failures, "
        f"MAE {report['calibrated']['mae']}, p={report['calibrated']['binomial_p_value_at_target_probability']}, "
        f"passed={report['calibrated']['passed']}"
    )
    loo = report["leave_one_out"]
    if loo.get("available"):
        print(
            "leave-one-out: "
            f"{loo['exceedances']}/{loo['n']} failures, MAE {loo['mae']}, "
            f"p={loo['binomial_p_value_at_target_probability']}, passed={loo['passed']}"
        )
    print(
        "model:         "
        f"gamma={report['model_config']['gamma']}, "
        f"regularization={report['model_config']['regularization']}"
    )

    print("\ncalibrated gaps")
    print("-" * 92)
    print(f"{'source_key':18s} {'skin':24s} {'base':>5s} {'cal':>5s} {'sales':>5s} {'gap':>5s}")
    for sample in report["samples"]:
        skin = f"{sample['hero_name']}/{sample['skin_name']}"
        print(
            f"{sample['source_key']:18s} {skin[:24]:24s} "
            f"{sample['base_score']:5d} {sample['calibrated_score']:5d} "
            f"{sample['sales_score']:5d} {sample['calibrated_gap']:5d}"
        )


def main() -> int:
    args = parse_args()
    if not args.db.exists():
        print(f"database not found: {args.db}", file=sys.stderr)
        return 1

    samples = collect_calibration_samples(
        args.db,
        limit=args.limit,
        official_only=not args.include_non_official,
    )
    try:
        result = fit_until_gap_probability(
            samples,
            gap_threshold=args.gap_threshold,
            target_probability=args.target_probability,
            alpha=args.alpha,
            max_iterations=args.max_iterations,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    report = serializable_report(result)
    report["evidence_scope"] = "all_public_evidence" if args.include_non_official else "official_only"
    if args.write_model:
        args.write_model.parent.mkdir(parents=True, exist_ok=True)
        args.write_model.write_text(
            json.dumps(result["model"].to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    if args.write_report:
        args.write_report.parent.mkdir(parents=True, exist_ok=True)
        args.write_report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print_text(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

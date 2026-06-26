"""Compare skin evaluation scores with available public sales evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.market_signal_repository import MarketSignalRepository  # noqa: E402
from data.skin_repository import DEFAULT_DB_PATH, SkinRepository  # noqa: E402
from feature_engineering.features import MarketValidationSignals  # noqa: E402
from feature_engineering.pipeline import FeatureBuilder  # noqa: E402
from models.rule_engine import RuleEngine  # noqa: E402
from models.sales_calibration import RbfSalesCalibrator, calibration_features  # noqa: E402
from models.sales_deviation import compare_score_to_sales, sales_blind_signals  # noqa: E402
from scripts.evaluate_skin import load_signal_payload, resolve_source_key  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare evaluation score against sales evidence.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH, help="Path to skins.sqlite3.")
    parser.add_argument("--source-key", help="Internal source key, e.g. 107-08.")
    parser.add_argument("--search", help="Search by hero, skin, or skin id and use the first match.")
    parser.add_argument("--signals-json", type=Path, help="Optional market validation signals JSON.")
    parser.add_argument("--ignore-db-signals", action="store_true", help="Do not load signals from SQLite.")
    parser.add_argument("--all-with-sales", action="store_true", help="Compare all skins with sales evidence.")
    parser.add_argument("--limit", type=int, default=50, help="Limit for --all-with-sales.")
    parser.add_argument("--calibration-model", type=Path, help="Optional sales calibration model JSON.")
    parser.add_argument(
        "--include-non-official",
        action="store_true",
        help="Include non-official public evidence. Default is official-only.",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    return parser.parse_args()


def build_comparison(
    db_path: Path,
    source_key: str,
    *,
    signals_json: Path | None = None,
    ignore_db_signals: bool = False,
    calibration_model: RbfSalesCalibrator | None = None,
    official_only: bool = True,
) -> dict[str, Any]:
    repo = SkinRepository(db_path)
    market_repo = MarketSignalRepository(db_path)
    if signals_json:
        signals = MarketValidationSignals.from_dict(load_signal_payload(signals_json, source_key))
    elif ignore_db_signals:
        signals = MarketValidationSignals()
    else:
        signals = market_repo.get_signals(source_key)

    sales_features = FeatureBuilder(repo).build(source_key, sales_blind_signals(signals) if official_only else signals)
    score_features = FeatureBuilder(repo).build(source_key, sales_blind_signals(signals))
    evaluation = RuleEngine().evaluate(score_features)
    evidence = market_repo.list_evidence(source_key, official_only=official_only)
    gap = compare_score_to_sales(sales_features, evaluation, evidence)
    if calibration_model is not None and gap["sales_score"] is not None:
        calibrated_score = calibration_model.predict(calibration_features(score_features, evaluation))
        calibrated_gap = calibrated_score - int(gap["sales_score"])
        gap["base_score"] = gap["score"]
        gap["base_gap"] = gap["gap"]
        gap["calibrated_score"] = calibrated_score
        gap["calibrated_gap"] = calibrated_gap
        gap["calibrated_gap_direction"] = (
            "aligned"
            if abs(calibrated_gap) <= 8
            else "score_above_sales"
            if calibrated_gap > 0
            else "sales_above_score"
        )
        gap["score"] = calibrated_score
        gap["score_basis"] = "calibrated_sales_score"
        gap["gap"] = calibrated_gap
        gap["absolute_gap"] = abs(calibrated_gap)
        gap["gap_direction"] = gap["calibrated_gap_direction"]
    return {
        "evaluation": evaluation.to_dict(),
        "sales_gap": gap,
    }


def build_all_comparisons(
    args: argparse.Namespace,
    calibration_model: RbfSalesCalibrator | None = None,
) -> list[dict[str, Any]]:
    market_repo = MarketSignalRepository(args.db)
    official_only = not args.include_non_official
    source_keys = market_repo.list_source_keys_with_sales_evidence(official_only=official_only)[: max(0, args.limit)]
    return [
        build_comparison(
            args.db,
            source_key,
            signals_json=args.signals_json,
            ignore_db_signals=args.ignore_db_signals,
            calibration_model=calibration_model,
            official_only=official_only,
        )
        for source_key in source_keys
    ]


def print_text(payload: dict[str, Any] | list[dict[str, Any]]) -> None:
    if isinstance(payload, list):
        has_calibration = any("calibrated_score" in item["sales_gap"] for item in payload)
        if has_calibration:
            print(f"{'source_key':18s} {'skin':24s} {'base':>5s} {'cal':>5s} {'sales':>5s} {'gap':>5s} direction")
            print("-" * 90)
        else:
            print(f"{'source_key':18s} {'skin':24s} {'score':>5s} {'sales':>5s} {'gap':>5s} direction")
            print("-" * 78)
        for item in payload:
            gap = item["sales_gap"]
            skin = f"{gap['hero_name']}/{gap['skin_name']}"
            if has_calibration:
                print(
                    f"{gap['source_key']:18s} {skin[:24]:24s} "
                    f"{_fmt(gap.get('base_score', gap['score'])):>5s} {_fmt(gap['score']):>5s} "
                    f"{_fmt(gap['sales_score']):>5s} {_fmt(gap['gap']):>5s} {gap['gap_direction']}"
                )
            else:
                print(
                    f"{gap['source_key']:18s} {skin[:24]:24s} "
                    f"{_fmt(gap['score']):>5s} {_fmt(gap['sales_score']):>5s} "
                    f"{_fmt(gap['gap']):>5s} {gap['gap_direction']}"
                )
        return

    evaluation = payload["evaluation"]
    gap = payload["sales_gap"]
    print(f"{gap['hero_name']} / {gap['skin_name']}")
    print("=" * 40)
    print(f"score:          {gap['score']}/100 ({gap['score_basis']})")
    if "base_score" in gap:
        print(f"base score:     {gap['base_score']}/100")
    print(f"sales score:    {_fmt(gap['sales_score'])}/100 ({gap['sales_basis'] or 'N/A'})")
    print(f"gap:            {_fmt(gap['gap'])}")
    print(f"direction:      {gap['gap_direction']}")
    print(f"confidence:     {gap['confidence']:.2f}")
    print(f"validation:     {evaluation['validation_status']}")
    print(f"basis:          {gap['interpretation']}")
    if gap["sales_evidence"]:
        evidence = gap["sales_evidence"]
        if evidence.get("volume") is not None:
            print(f"sales volume:   {evidence['volume']} ({evidence.get('volume_relation')})")
        if evidence.get("rank") is not None:
            print(f"sales rank:     {evidence['rank']} / {evidence['rank_size']}")
        if evidence.get("source_title"):
            print(f"source:         {evidence['source_title']}")
        if evidence.get("source_url"):
            print(f"url:            {evidence['source_url']}")
    print("warnings:")
    for warning in gap["warnings"] or ["none"]:
        print(f"- {warning}")


def _fmt(value: Any) -> str:
    return "N/A" if value is None else str(value)


def load_calibration_model(path: Path | None) -> RbfSalesCalibrator | None:
    if path is None:
        return None
    return RbfSalesCalibrator.from_dict(json.loads(path.read_text(encoding="utf-8-sig")))


def main() -> int:
    args = parse_args()
    if not args.db.exists():
        print(f"database not found: {args.db}", file=sys.stderr)
        return 1

    try:
        calibration_model = load_calibration_model(args.calibration_model)
        if args.all_with_sales:
            payload: dict[str, Any] | list[dict[str, Any]] = build_all_comparisons(args, calibration_model)
        else:
            repo = SkinRepository(args.db)
            source_key = resolve_source_key(repo, args.source_key, args.search)
            payload = build_comparison(
                args.db,
                source_key,
                signals_json=args.signals_json,
                ignore_db_signals=args.ignore_db_signals,
                calibration_model=calibration_model,
                official_only=not args.include_non_official,
            )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print_text(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

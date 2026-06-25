"""Generate a sales action report for one skin."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from business.sales_advisor import SalesAdvisor  # noqa: E402
from data.market_signal_repository import MarketSignalRepository  # noqa: E402
from data.skin_repository import DEFAULT_DB_PATH, SkinRepository  # noqa: E402
from feature_engineering.features import MarketValidationSignals  # noqa: E402
from feature_engineering.pipeline import FeatureBuilder  # noqa: E402
from models.rule_engine import RuleEngine  # noqa: E402
from scripts.evaluate_skin import load_signal_payload, resolve_source_key  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a sales action report for one WZRY skin.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH, help="Path to skins.sqlite3.")
    parser.add_argument("--source-key", help="Internal source key, e.g. 105-02.")
    parser.add_argument("--search", help="Search by hero, skin, or skin id and use the first match.")
    parser.add_argument("--signals-json", type=Path, help="Optional market validation signals JSON.")
    parser.add_argument("--ignore-db-signals", action="store_true", help="Do not load signals from SQLite.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    return parser.parse_args()


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    if not args.db.exists():
        raise ValueError(f"database not found: {args.db}")

    repo = SkinRepository(args.db)
    source_key = resolve_source_key(repo, args.source_key, args.search)
    if args.signals_json:
        signals = MarketValidationSignals.from_dict(load_signal_payload(args.signals_json, source_key))
    elif args.ignore_db_signals:
        signals = MarketValidationSignals()
    else:
        signals = MarketSignalRepository(args.db).get_signals(source_key)

    features = FeatureBuilder(repo).build(source_key, signals)
    evaluation = RuleEngine().evaluate(features)
    sales_report = SalesAdvisor().advise(features, evaluation)
    return {
        "evaluation": evaluation.to_dict(),
        "sales_report": sales_report.to_dict(),
    }


def print_text(payload: dict[str, Any]) -> None:
    evaluation = payload["evaluation"]
    report = payload["sales_report"]
    print(f"{report['hero_name']} / {report['skin_name']}")
    print("=" * 36)
    print(f"decision:        {report['decision']}")
    print(f"reason:          {report['decision_reason']}")
    print(f"sales readiness: {report['sales_readiness']}/100")
    print(f"evidence score:  {evaluation['evaluation_score'] if evaluation['evaluation_score'] is not None else 'N/A'}")
    print(f"confidence:      {evaluation['confidence']:.2f}")
    print(f"validation:      {evaluation['validation_status']}")

    print("\npurchase drivers")
    print("-" * 36)
    for item in report["purchase_drivers"]:
        print(f"- {item}")

    print("\nconversion blockers")
    print("-" * 36)
    for item in report["conversion_blockers"] or ["none"]:
        print(f"- {item}")

    print("\nrecommended actions")
    print("-" * 36)
    for item in report["recommended_actions"]:
        print(f"- {item['action']}: {item['detail']}")

    print("\npricing guidance")
    print("-" * 36)
    pricing = report["pricing_guidance"]
    print(f"posture: {pricing['posture']}")
    print(f"reason:  {pricing['rationale']}")
    if pricing.get("official_price_text"):
        print(f"official price: {pricing['official_price_text']}")

    print("\nevidence gaps")
    print("-" * 36)
    for item in report["evidence_gaps"] or ["none"]:
        print(f"- {item}")


def main() -> int:
    args = parse_args()
    try:
        payload = build_report(args)
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

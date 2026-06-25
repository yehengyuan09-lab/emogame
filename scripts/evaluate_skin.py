"""Evaluate one WZRY skin with the MVP emotional premium rule engine."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.skin_repository import DEFAULT_DB_PATH, SkinRepository  # noqa: E402
from data.market_signal_repository import MarketSignalRepository  # noqa: E402
from feature_engineering.features import MarketValidationSignals  # noqa: E402
from feature_engineering.pipeline import FeatureBuilder  # noqa: E402
from models.rule_engine import RuleEngine  # noqa: E402


def load_signal_payload(path: Path | None, source_key: str | None) -> dict[str, Any]:
    if not path:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if source_key and source_key in payload and isinstance(payload[source_key], dict):
        return payload[source_key]
    return payload


def resolve_source_key(repo: SkinRepository, source_key: str | None, search: str | None) -> str:
    if source_key:
        if not repo.get_skin(source_key):
            raise ValueError(f"skin not found: {source_key}")
        return source_key

    if not search:
        raise ValueError("provide --source-key or --search")

    rows = repo.search_skins(search, limit=5)
    if not rows:
        raise ValueError(f"no skin matched search: {search}")
    if len(rows) > 1:
        print("matched skins:", file=sys.stderr)
        for row in rows:
            print(f"  {row['source_key']}: {row['hero_name']} / {row['skin_name']}", file=sys.stderr)
        print(f"using first match: {rows[0]['source_key']}", file=sys.stderr)
    return rows[0]["source_key"]


def print_text(result: dict[str, Any]) -> None:
    print(f"{result['hero_name']} / {result['skin_name']}")
    print("=" * 32)
    score = result["evaluation_score"]
    print(f"evidence score:    {score if score is not None else 'N/A'}")
    print(f"official prior:    {result['official_prior_score']}/100")
    print(f"confidence:        {result['confidence']:.2f}")
    print(f"validation:        {result['validation_status']}")
    print(f"evidence coverage: {result['evidence_coverage']:.2f}")

    print("\naspect scores")
    print("-" * 32)
    for name, aspect_score in result["aspect_scores"].items():
        value = f"{aspect_score:3d}" if aspect_score is not None else "N/A"
        print(f"{name:24s} {value}")

    print("\nwarnings")
    print("-" * 32)
    for warning in result["warnings"] or ["none"]:
        print(warning)

    print("\nevidence")
    print("-" * 32)
    for key, value in result["evidence"].items():
        print(f"{key}: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a WZRY skin emotional premium score.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH, help="Path to skins.sqlite3.")
    parser.add_argument("--source-key", help="Internal source key, e.g. 105-02.")
    parser.add_argument("--search", help="Search by hero, skin, or skin id and evaluate the first match.")
    parser.add_argument("--signals-json", type=Path, help="Optional market validation signals JSON.")
    parser.add_argument("--ignore-db-signals", action="store_true", help="Do not load signals from SQLite.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.db.exists():
        print(f"database not found: {args.db}", file=sys.stderr)
        print("run: python crawlers/wzry_skin_crawler.py --skip-images", file=sys.stderr)
        return 1

    repo = SkinRepository(args.db)
    try:
        source_key = resolve_source_key(repo, args.source_key, args.search)
        if args.signals_json:
            signals = MarketValidationSignals.from_dict(load_signal_payload(args.signals_json, source_key))
        elif args.ignore_db_signals:
            signals = MarketValidationSignals()
        else:
            signals = MarketSignalRepository(args.db).get_signals(source_key)
        features = FeatureBuilder(repo).build(source_key, signals)
        result = RuleEngine().evaluate(features).to_dict()
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print_text(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

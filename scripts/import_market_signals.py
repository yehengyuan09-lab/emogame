"""Import manually curated market/opinion signals and evidence JSON."""

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


def normalize_payload(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if "source_key" in payload:
        return {payload["source_key"]: payload}
    return payload


def import_payload(db_path: Path, payload: dict[str, Any], *, strict: bool = True) -> dict[str, int]:
    skin_repo = SkinRepository(db_path)
    market_repo = MarketSignalRepository(db_path)
    market_repo.ensure_schema()
    imported_signals = 0
    imported_evidence = 0

    for source_key, item in normalize_payload(payload).items():
        if strict and not skin_repo.get_skin(source_key):
            raise ValueError(f"skin not found: {source_key}")

        signals_data = item.get("signals") if isinstance(item.get("signals"), dict) else item
        signals = MarketValidationSignals.from_dict(signals_data)
        if signals.present_fields():
            market_repo.upsert_signals(source_key, signals, signal_source=item.get("signal_source", "manual"), raw_json=item)
            imported_signals += 1

        for evidence in item.get("evidence", []) or []:
            market_repo.add_evidence(
                source_key,
                platform=evidence.get("platform", "manual"),
                external_id=evidence.get("external_id"),
                url=evidence.get("url"),
                title=evidence.get("title"),
                author=evidence.get("author"),
                published_at=evidence.get("published_at"),
                text=evidence.get("text"),
                metrics=evidence.get("metrics"),
                aspect_tags=evidence.get("aspect_tags"),
                raw_json=evidence,
            )
            imported_evidence += 1
        if item.get("evidence"):
            market_repo.aggregate_evidence_signals(source_key)

    return {"signals": imported_signals, "evidence": imported_evidence}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import market signal JSON into local SQLite.")
    parser.add_argument("json_file", type=Path)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--no-strict", action="store_true", help="Allow source_key values not present in skins table.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = json.loads(args.json_file.read_text(encoding="utf-8-sig"))
    try:
        result = import_payload(args.db, payload, strict=not args.no_strict)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

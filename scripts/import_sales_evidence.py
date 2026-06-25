"""Import curated public sales evidence JSON.

This is a thin validator around the generic market signal importer. It keeps
sales evidence explicit, instead of hiding rank or estimate quality in text.
"""

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
from scripts.import_market_signals import import_payload, normalize_payload  # noqa: E402


SALES_METRIC_KEYS = {
    "sales_volume",
    "estimated_sales_volume",
    "sales_volume_estimate",
    "sales_volume_upper_bound",
    "sales_volume_lower_bound",
    "units_sold",
    "sales",
    "sales_rank",
    "hot_sales_rank",
    "rank",
    "ownership_rate",
    "avg_spend_to_obtain",
}


def validate_sales_payload(payload: dict[str, Any]) -> None:
    normalized = normalize_payload(payload)
    for source_key, item in normalized.items():
        evidence_items = item.get("evidence") or []
        if not evidence_items:
            signals = item.get("signals") if isinstance(item.get("signals"), dict) else item
            if not any(key in signals for key in SALES_METRIC_KEYS):
                raise ValueError(f"{source_key}: missing sales evidence or sales signal fields")
            continue

        for index, evidence in enumerate(evidence_items, start=1):
            metrics = evidence.get("metrics") or {}
            if not any(key in metrics for key in SALES_METRIC_KEYS):
                raise ValueError(f"{source_key} evidence #{index}: missing sales metric")
            evidence.setdefault("platform", "sales_public")
            if not evidence.get("external_id"):
                evidence["external_id"] = stable_external_id(source_key, evidence, index)


def stable_external_id(source_key: str, evidence: dict[str, Any], index: int) -> str:
    if evidence.get("url"):
        return str(evidence["url"])
    if evidence.get("title"):
        return f"{source_key}:{evidence['title']}"
    return f"{source_key}:sales:{index}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import public sales evidence JSON into local SQLite.")
    parser.add_argument("json_file", type=Path)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--no-strict", action="store_true", help="Allow source_key values not present in skins table.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = json.loads(args.json_file.read_text(encoding="utf-8-sig"))
    try:
        validate_sales_payload(payload)
        result = import_payload(args.db, payload, strict=not args.no_strict)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

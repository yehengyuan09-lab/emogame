"""Fetch Bilibili video metrics for a known BVID/URL and store as evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crawlers.bilibili_evidence import fetch_bilibili_video  # noqa: E402
from data.market_signal_repository import MarketSignalRepository  # noqa: E402
from data.skin_repository import DEFAULT_DB_PATH, SkinRepository  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch one Bilibili video as opinion evidence.")
    parser.add_argument("--source-key", required=True, help="Skin source key, e.g. 105-02.")
    parser.add_argument("--video", required=True, help="Bilibili BVID or video URL.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument(
        "--aspect-tags",
        default="",
        help="Comma-separated aspect tags, e.g. visual,feel,craftsmanship.",
    )
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    skin_repo = SkinRepository(args.db)
    if not skin_repo.get_skin(args.source_key):
        print(f"skin not found: {args.source_key}", file=sys.stderr)
        return 1

    try:
        evidence = fetch_bilibili_video(args.video)
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"failed to fetch Bilibili evidence: {exc}", file=sys.stderr)
        return 1

    market_repo = MarketSignalRepository(args.db)
    aspect_tags = [item.strip() for item in args.aspect_tags.split(",") if item.strip()]
    market_repo.add_evidence(
        args.source_key,
        platform="bilibili",
        external_id=evidence.bvid,
        url=evidence.url,
        title=evidence.title,
        author=evidence.author,
        published_at=evidence.published_at,
        text=evidence.text,
        metrics=evidence.metrics,
        aspect_tags=aspect_tags,
        raw_json=evidence.raw_json,
    )
    aggregate = market_repo.aggregate_evidence_signals(args.source_key)

    result = {
        "source_key": args.source_key,
        "platform": "bilibili",
        "bvid": evidence.bvid,
        "title": evidence.title,
        "metrics": evidence.metrics,
        "aggregate_signals": aggregate.present_fields(),
    }
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"stored Bilibili evidence: {evidence.bvid} {evidence.title}")
        print(f"metrics={evidence.metrics}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

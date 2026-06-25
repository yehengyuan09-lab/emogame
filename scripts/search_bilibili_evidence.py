"""Search Bilibili videos for one skin and store them as market evidence."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crawlers.bilibili_evidence import fetch_bilibili_video, search_bilibili_videos  # noqa: E402
from data.market_signal_repository import MarketSignalRepository  # noqa: E402
from data.skin_repository import DEFAULT_DB_PATH, SkinRepository  # noqa: E402


DEFAULT_ASPECT_TAGS = ["visual", "feel", "craftsmanship", "purchase_intent"]


def build_skin_query(skin: dict[str, object]) -> str:
    hero_name = str(skin.get("hero_name") or "").strip()
    skin_name = str(skin.get("skin_name") or "").strip()
    return " ".join(part for part in ["王者荣耀", hero_name, skin_name, "皮肤"] if part)


def is_relevant(text: str, terms: list[str]) -> bool:
    normalized = text.lower()
    return any(term and term.lower() in normalized for term in terms)


def skin_name_terms(skin_name: str) -> list[str]:
    return [
        term
        for term in re.split(r"[\s·・\-_/|●]+", skin_name.strip())
        if len(term) >= 2
    ]


def is_relevant_to_skin(text: str, *, hero_name: str, skin_name: str) -> bool:
    terms = skin_name_terms(skin_name)
    if terms:
        return is_relevant(text, terms)
    return is_relevant(text, [hero_name])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search Bilibili and import matching videos as evidence.")
    parser.add_argument("--source-key", required=True, help="Skin source key, e.g. 105-02.")
    parser.add_argument("--query", default="", help="Override the query. Default: 王者荣耀 + hero + skin + 皮肤.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--page", type=int, default=1)
    parser.add_argument(
        "--aspect-tags",
        default=",".join(DEFAULT_ASPECT_TAGS),
        help="Comma-separated aspect tags for imported videos.",
    )
    parser.add_argument(
        "--no-filter",
        action="store_true",
        help="Do not require hero/skin terms to appear in title or description.",
    )
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    skin_repo = SkinRepository(args.db)
    skin = skin_repo.get_skin(args.source_key)
    if not skin:
        print(f"skin not found: {args.source_key}", file=sys.stderr)
        return 1

    query = args.query.strip() or build_skin_query(skin)
    hero_name = str(skin.get("hero_name") or "")
    skin_name = str(skin.get("skin_name") or "")
    aspect_tags = [item.strip() for item in args.aspect_tags.split(",") if item.strip()]

    try:
        search_results = search_bilibili_videos(query, limit=max(args.limit * 3, args.limit), page=args.page)
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"failed to search Bilibili evidence: {exc}", file=sys.stderr)
        return 1

    market_repo = MarketSignalRepository(args.db)
    imported: list[dict[str, object]] = []
    skipped: list[dict[str, str]] = []

    for search_result in search_results:
        if len(imported) >= args.limit:
            break
        try:
            evidence = fetch_bilibili_video(search_result.bvid)
        except Exception as exc:  # noqa: BLE001 - keep one failed video from aborting whole batch
            skipped.append({"bvid": search_result.bvid, "reason": str(exc)})
            continue

        relevance_text = "\n".join([search_result.title, search_result.description, evidence.text])
        if not args.no_filter and not is_relevant_to_skin(
            relevance_text,
            hero_name=hero_name,
            skin_name=skin_name,
        ):
            skipped.append({"bvid": search_result.bvid, "reason": "missing skin term"})
            continue

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
            raw_json={
                "query": query,
                "search_result": search_result.raw_json,
                "video": evidence.raw_json,
            },
        )
        imported.append(
            {
                "bvid": evidence.bvid,
                "title": evidence.title,
                "metrics": evidence.metrics,
            }
        )

    aggregate = market_repo.aggregate_evidence_signals(args.source_key)
    result = {
        "source_key": args.source_key,
        "query": query,
        "imported": len(imported),
        "skipped": skipped,
        "videos": imported,
        "aggregate_signals": {
            name: getattr(aggregate, name)
            for name in aggregate.present_fields()
        },
    }
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"query={query}")
        print(f"imported={len(imported)} skipped={len(skipped)}")
        for item in imported:
            print(f"- {item['bvid']} {item['title']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

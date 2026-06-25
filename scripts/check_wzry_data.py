"""Print a WZRY data quality report from the local SQLite database."""

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


def compact_skin(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_key": row["source_key"],
        "hero_name": row["hero_name"],
        "skin_index": row["skin_index"],
        "skin_name": row["skin_name"],
        "quality": row.get("quality") or "",
        "catalog_source": row["catalog_source"],
        "has_detail_record": bool(row["has_detail_record"]),
        "primary_asset_status": row.get("primary_asset_status"),
    }


def build_report(repo: SkinRepository, sample_limit: int) -> dict[str, Any]:
    return {
        "stats": repo.stats(),
        "missing_detail_samples": [
            compact_skin(row) for row in repo.missing_detail_skins(limit=sample_limit)
        ],
        "missing_primary_asset_samples": [
            compact_skin(row) for row in repo.missing_primary_asset_skins(limit=sample_limit)
        ],
        "failed_asset_samples": repo.failed_assets(limit=sample_limit),
    }


def print_text_report(report: dict[str, Any]) -> None:
    stats = report["stats"]
    print("WZRY data quality report")
    print("=" * 32)
    print(f"heroes:                {stats['heroes']}")
    print(f"skins:                 {stats['skins']}")
    print(f"assets:                {stats['assets']}")
    print(f"skins with detail:     {stats['with_detail']}")
    print(f"skins missing detail:  {stats['missing_detail']}")
    print(f"detail-only skins:     {stats['detail_only']}")
    print(f"skins missing primary: {stats['missing_primary_asset']}")
    print(f"failed assets:         {stats['failed_assets']}")

    print("\nMissing detail samples")
    print("-" * 32)
    for row in report["missing_detail_samples"]:
        print(f"{row['source_key']}: {row['hero_name']} / {row['skin_name']}")
    if not report["missing_detail_samples"]:
        print("none")

    print("\nMissing primary asset samples")
    print("-" * 32)
    for row in report["missing_primary_asset_samples"]:
        print(f"{row['source_key']}: {row['hero_name']} / {row['skin_name']}")
    if not report["missing_primary_asset_samples"]:
        print("none")

    print("\nFailed asset samples")
    print("-" * 32)
    for row in report["failed_asset_samples"]:
        print(f"{row['asset_id']}: {row['error']}")
    if not report["failed_asset_samples"]:
        print("none")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check local WZRY crawler data quality.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH, help="Path to skins.sqlite3.")
    parser.add_argument("--samples", type=int, default=10, help="Number of sample rows per issue type.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.db.exists():
        print(f"database not found: {args.db}", file=sys.stderr)
        print("run: python crawlers/wzry_skin_crawler.py --skip-images", file=sys.stderr)
        return 1

    repo = SkinRepository(args.db)
    report = build_report(repo, max(0, args.samples))
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print_text_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

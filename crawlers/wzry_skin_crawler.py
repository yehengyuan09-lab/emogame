"""Collect Honor of Kings hero/skin catalog data into SQLite.

The crawler uses two official pvp.qq.com JSON sources:

- ``herolist.json`` is treated as the catalog baseline. It has the full hero list
  and skin names.
- ``heroskinlist.json`` enriches catalog rows with skin ids, quality labels,
  release dates, detail links, videos, and image URLs.

Only standard-library modules are used so this script can run before the full
ML/VLM dependency stack is installed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


HERO_LIST_URL = "https://pvp.qq.com/web201605/js/herolist.json"
SKIN_LIST_URL = "https://pvp.qq.com/zlkdatasys/heroskinlist.json"
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125 Safari/537.36"
)
HERO_TYPE_MAP = {
    1: "战士",
    2: "法师",
    3: "坦克",
    4: "刺客",
    5: "射手",
    6: "辅助",
}
PRICE_PATTERN = re.compile(r"\d+(?:\.\d+)?\s*(?:点券|荣耀积分|皮肤碎片)")


@dataclass(slots=True)
class HeroRecord:
    hero_id: str
    hero_name: str
    id_name: str
    title: str
    hero_type_code: int | None
    hero_type: str
    skin_names: list[str]
    raw_json: dict[str, Any]


@dataclass(slots=True)
class SkinRecord:
    source_key: str
    source_index: int
    hero_id: str
    hero_name: str
    skin_index: int
    skin_id: str
    skin_name: str
    quality: str
    online_date: str
    intro: str
    acquire_method: str
    price_text: str | None
    image_url: str
    detail_url: str
    mobile_url: str
    video_id: str
    catalog_source: str
    detail_source: str | None
    has_detail_record: bool
    raw_catalog_json: dict[str, Any] = field(default_factory=dict)
    raw_detail_json: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AssetRecord:
    source_key: str
    asset_type: str
    remote_url: str
    local_path: str | None = None
    content_hash: str | None = None
    download_status: str = "pending"
    error: str | None = None


def request_bytes(url: str, timeout: float = 30.0) -> bytes:
    if url.startswith("//"):
        url = "https:" + url
    req = Request(url, headers={"User-Agent": DEFAULT_USER_AGENT})
    with urlopen(req, timeout=timeout) as resp:
        return resp.read()


def load_json(url: str) -> Any:
    return json.loads(request_bytes(url).decode("utf-8"))


def safe_filename(value: str, max_length: int = 140) -> str:
    value = re.sub(r'[\\/:*?"<>|\r\n\t]+', "_", value).strip(" .")
    value = re.sub(r"\s+", " ", value)
    return (value or "unknown")[:max_length]


def normalize_text(value: Any) -> str:
    return str(value or "").strip()


def normalize_date(value: Any) -> str:
    text = normalize_text(value)
    if re.fullmatch(r"\d{8}", text):
        return f"{text[:4]}-{text[4:6]}-{text[6:8]}"
    return text


def image_extension(url: str) -> str:
    path = urlparse("https:" + url if url.startswith("//") else url).path
    suffix = Path(path).suffix.lower()
    return suffix if suffix in {".jpg", ".jpeg", ".png", ".gif", ".webp"} else ".jpg"


def parse_price_text(acquire_method: str) -> str | None:
    """Extract explicit price strings such as ``60点券`` from source text."""
    matches = PRICE_PATTERN.findall(acquire_method or "")
    return " / ".join(matches) if matches else None


def split_skin_names(value: Any) -> list[str]:
    return [item.strip() for item in normalize_text(value).split("|") if item.strip()]


def normalize_hero(raw: dict[str, Any]) -> HeroRecord:
    hero_type_code = raw.get("hero_type")
    try:
        hero_type_code = int(hero_type_code)
    except (TypeError, ValueError):
        hero_type_code = None
    return HeroRecord(
        hero_id=normalize_text(raw.get("ename")),
        hero_name=normalize_text(raw.get("cname")),
        id_name=normalize_text(raw.get("id_name")),
        title=normalize_text(raw.get("title")),
        hero_type_code=hero_type_code,
        hero_type=HERO_TYPE_MAP.get(hero_type_code or 0, ""),
        skin_names=split_skin_names(raw.get("skin_name")),
        raw_json=raw,
    )


def normalize_detail(raw: dict[str, Any], source_index: int) -> SkinRecord:
    acquire_method = normalize_text(raw.get("hqfs_8609"))
    image_url = (
        normalize_text(raw.get("fmlb_4536"))
        or normalize_text(raw.get("fmb1lb_5300"))
        or normalize_text(raw.get("yxtxlb_8443"))
    )
    skin_id = normalize_text(raw.get("pfidlb_3934"))
    hero_name = normalize_text(raw.get("yxmclb_9965"))
    skin_name = normalize_text(raw.get("pfmclb_7523"))
    return SkinRecord(
        source_key=f"detail-{source_index:04d}-{skin_id or safe_filename(hero_name + '-' + skin_name, 80)}",
        source_index=source_index,
        hero_id=f"detail-{safe_filename(hero_name, 80)}",
        hero_name=hero_name,
        skin_index=0,
        skin_id=skin_id,
        skin_name=skin_name,
        quality=normalize_text(raw.get("pfpzlb_3289")),
        online_date=normalize_date(raw.get("sxsjlb_1516")),
        intro=normalize_text(raw.get("yjhjsl_5003")),
        acquire_method=acquire_method,
        price_text=parse_price_text(acquire_method),
        image_url=image_url,
        detail_url=normalize_text(raw.get("pcljlb_9272")),
        mobile_url=normalize_text(raw.get("mdljlb_1924")),
        video_id=normalize_text(raw.get("spvidl_6663")),
        catalog_source="detail_only",
        detail_source="heroskinlist",
        has_detail_record=True,
        raw_detail_json=raw,
    )


def detail_match_key(hero_name: str, skin_name: str) -> tuple[str, str]:
    return (normalize_text(hero_name), normalize_text(skin_name))


def merge_catalog(
    raw_heroes: Iterable[dict[str, Any]],
    raw_details: Iterable[dict[str, Any]],
    limit: int = 0,
) -> tuple[list[HeroRecord], list[SkinRecord]]:
    heroes = [normalize_hero(item) for item in raw_heroes]

    details_by_key: dict[tuple[str, str], list[SkinRecord]] = {}
    for index, raw in enumerate(raw_details, start=1):
        detail = normalize_detail(raw, index)
        details_by_key.setdefault(detail_match_key(detail.hero_name, detail.skin_name), []).append(detail)

    skins: list[SkinRecord] = []
    source_index = 0
    for hero in heroes:
        for skin_index, skin_name in enumerate(hero.skin_names, start=1):
            source_index += 1
            matched = None
            detail_list = details_by_key.get(detail_match_key(hero.hero_name, skin_name))
            if detail_list:
                matched = detail_list.pop(0)

            skin = SkinRecord(
                source_key=f"{hero.hero_id}-{skin_index:02d}",
                source_index=source_index,
                hero_id=hero.hero_id,
                hero_name=hero.hero_name,
                skin_index=skin_index,
                skin_id=matched.skin_id if matched else "",
                skin_name=skin_name,
                quality=matched.quality if matched else "",
                online_date=matched.online_date if matched else "",
                intro=matched.intro if matched else "",
                acquire_method=matched.acquire_method if matched else "",
                price_text=matched.price_text if matched else None,
                image_url=matched.image_url if matched else "",
                detail_url=matched.detail_url if matched else "",
                mobile_url=matched.mobile_url if matched else "",
                video_id=matched.video_id if matched else "",
                catalog_source="herolist",
                detail_source=matched.detail_source if matched else None,
                has_detail_record=matched is not None,
                raw_catalog_json={"hero": hero.raw_json, "skin_index": skin_index, "skin_name": skin_name},
                raw_detail_json=matched.raw_detail_json if matched else {},
            )
            skins.append(skin)

            if limit and len(skins) >= limit:
                return heroes, skins

    remaining = [detail for values in details_by_key.values() for detail in values]
    for detail in remaining:
        if limit and len(skins) >= limit:
            break
        detail.source_index = len(skins) + 1
        skins.append(detail)

    return heroes, skins


def asset_candidates(skin: SkinRecord) -> list[AssetRecord]:
    raw = skin.raw_detail_json or {}
    candidates = [
        ("skin_primary", skin.image_url),
        ("skin_secondary", normalize_text(raw.get("fmb1lb_5300"))),
        ("hero_icon", normalize_text(raw.get("yxtxlb_8443"))),
    ]
    seen: set[str] = set()
    assets: list[AssetRecord] = []
    for asset_type, url in candidates:
        if not url or url in seen:
            continue
        seen.add(url)
        assets.append(AssetRecord(source_key=skin.source_key, asset_type=asset_type, remote_url=url))
    return assets


def ensure_schema(conn: sqlite3.Connection) -> None:
    current_skin_columns = {
        row[1] for row in conn.execute("PRAGMA table_info(skins)").fetchall()
    }
    required_skin_columns = {
        "source_key",
        "hero_id",
        "skin_index",
        "catalog_source",
        "has_detail_record",
        "raw_catalog_json",
        "raw_detail_json",
    }
    if current_skin_columns and not required_skin_columns.issubset(current_skin_columns):
        conn.executescript(
            """
            DROP TABLE IF EXISTS skin_assets;
            DROP TABLE IF EXISTS skins;
            DROP TABLE IF EXISTS heroes;
            DROP TABLE IF EXISTS crawl_runs;
            """
        )

    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS crawl_runs (
            run_id INTEGER PRIMARY KEY AUTOINCREMENT,
            started_at TEXT NOT NULL,
            finished_at TEXT,
            hero_list_url TEXT NOT NULL,
            skin_list_url TEXT NOT NULL,
            hero_count INTEGER NOT NULL DEFAULT 0,
            skin_count INTEGER NOT NULL DEFAULT 0,
            asset_count INTEGER NOT NULL DEFAULT 0,
            downloaded_count INTEGER NOT NULL DEFAULT 0,
            image_failed_count INTEGER NOT NULL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS heroes (
            hero_id TEXT PRIMARY KEY,
            hero_name TEXT NOT NULL,
            id_name TEXT,
            title TEXT,
            hero_type_code INTEGER,
            hero_type TEXT,
            skin_count INTEGER NOT NULL DEFAULT 0,
            raw_json TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS skins (
            source_key TEXT PRIMARY KEY,
            source_index INTEGER NOT NULL,
            hero_id TEXT NOT NULL,
            hero_name TEXT NOT NULL,
            skin_index INTEGER NOT NULL,
            skin_id TEXT,
            skin_name TEXT NOT NULL,
            quality TEXT,
            online_date TEXT,
            acquire_method TEXT,
            price_text TEXT,
            image_url TEXT,
            image_path TEXT,
            detail_url TEXT,
            mobile_url TEXT,
            video_id TEXT,
            intro TEXT,
            catalog_source TEXT NOT NULL,
            detail_source TEXT,
            has_detail_record INTEGER NOT NULL,
            raw_catalog_json TEXT NOT NULL,
            raw_detail_json TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS skin_assets (
            asset_id TEXT PRIMARY KEY,
            source_key TEXT NOT NULL,
            asset_type TEXT NOT NULL,
            remote_url TEXT NOT NULL,
            local_path TEXT,
            content_hash TEXT,
            download_status TEXT NOT NULL,
            error TEXT,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(source_key) REFERENCES skins(source_key)
        );

        CREATE INDEX IF NOT EXISTS idx_heroes_hero_name ON heroes(hero_name);
        CREATE INDEX IF NOT EXISTS idx_skins_hero_name ON skins(hero_name);
        CREATE INDEX IF NOT EXISTS idx_skins_skin_name ON skins(skin_name);
        CREATE INDEX IF NOT EXISTS idx_skins_skin_id ON skins(skin_id);
        CREATE INDEX IF NOT EXISTS idx_skins_has_detail_record ON skins(has_detail_record);
        CREATE INDEX IF NOT EXISTS idx_skin_assets_source_key ON skin_assets(source_key);
        CREATE UNIQUE INDEX IF NOT EXISTS idx_skins_hero_skin_index ON skins(hero_id, skin_index)
            WHERE catalog_source = 'herolist';
        CREATE UNIQUE INDEX IF NOT EXISTS idx_skin_assets_source_type ON skin_assets(source_key, asset_type);
        """
    )
    conn.executescript(
        """
        DELETE FROM skin_assets;
        DELETE FROM skins;
        DELETE FROM heroes;
        """
    )


def save_run_start(conn: sqlite3.Connection, hero_list_url: str, skin_list_url: str) -> int:
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    cur = conn.execute(
        """
        INSERT INTO crawl_runs(started_at, hero_list_url, skin_list_url)
        VALUES(?, ?, ?)
        """,
        (now, hero_list_url, skin_list_url),
    )
    return int(cur.lastrowid)


def save_run_finish(
    conn: sqlite3.Connection,
    run_id: int,
    hero_count: int,
    skin_count: int,
    asset_count: int,
    downloaded_count: int,
    failed_count: int,
) -> None:
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    conn.execute(
        """
        UPDATE crawl_runs
        SET finished_at = ?, hero_count = ?, skin_count = ?, asset_count = ?,
            downloaded_count = ?, image_failed_count = ?
        WHERE run_id = ?
        """,
        (now, hero_count, skin_count, asset_count, downloaded_count, failed_count, run_id),
    )


def save_hero(conn: sqlite3.Connection, hero: HeroRecord) -> None:
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    conn.execute(
        """
        INSERT INTO heroes(
            hero_id, hero_name, id_name, title, hero_type_code, hero_type,
            skin_count, raw_json, updated_at
        )
        VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            hero.hero_id,
            hero.hero_name,
            hero.id_name,
            hero.title,
            hero.hero_type_code,
            hero.hero_type,
            len(hero.skin_names),
            json.dumps(hero.raw_json, ensure_ascii=False, separators=(",", ":")),
            now,
        ),
    )


def save_skin(conn: sqlite3.Connection, skin: SkinRecord, image_path: Path | None) -> None:
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    conn.execute(
        """
        INSERT INTO skins(
            source_key, source_index, hero_id, hero_name, skin_index, skin_id, skin_name,
            quality, online_date, acquire_method, price_text, image_url, image_path,
            detail_url, mobile_url, video_id, intro, catalog_source, detail_source,
            has_detail_record, raw_catalog_json, raw_detail_json, updated_at
        )
        VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            skin.source_key,
            skin.source_index,
            skin.hero_id,
            skin.hero_name,
            skin.skin_index,
            skin.skin_id,
            skin.skin_name,
            skin.quality,
            skin.online_date,
            skin.acquire_method,
            skin.price_text,
            skin.image_url,
            str(image_path) if image_path else None,
            skin.detail_url,
            skin.mobile_url,
            skin.video_id,
            skin.intro,
            skin.catalog_source,
            skin.detail_source,
            1 if skin.has_detail_record else 0,
            json.dumps(skin.raw_catalog_json, ensure_ascii=False, separators=(",", ":")),
            json.dumps(skin.raw_detail_json, ensure_ascii=False, separators=(",", ":")),
            now,
        ),
    )


def save_asset(conn: sqlite3.Connection, asset: AssetRecord) -> None:
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    asset_id = f"{asset.source_key}:{asset.asset_type}"
    conn.execute(
        """
        INSERT INTO skin_assets(
            asset_id, source_key, asset_type, remote_url, local_path, content_hash,
            download_status, error, updated_at
        )
        VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            asset_id,
            asset.source_key,
            asset.asset_type,
            asset.remote_url,
            asset.local_path,
            asset.content_hash,
            asset.download_status,
            asset.error,
            now,
        ),
    )


def content_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_asset(skin: SkinRecord, asset: AssetRecord, image_dir: Path, overwrite: bool = False) -> AssetRecord:
    image_dir.mkdir(parents=True, exist_ok=True)
    filename = safe_filename(
        f"{skin.source_index:04d}-{skin.hero_name}-{skin.skin_index:02d}-{skin.skin_id or 'noid'}-{skin.skin_name}-{asset.asset_type}"
    )
    path = image_dir / f"{filename}{image_extension(asset.remote_url)}"
    try:
        if not path.exists() or overwrite:
            path.write_bytes(request_bytes(asset.remote_url))
            asset.download_status = "downloaded"
        else:
            asset.download_status = "existing"
        asset.local_path = str(path)
        asset.content_hash = content_hash(path)
        asset.error = None
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        asset.download_status = "failed"
        asset.error = str(exc)
    return asset


def selected_asset_types(value: str) -> set[str]:
    if value.strip().lower() == "all":
        return {"skin_primary", "skin_secondary", "hero_icon"}
    return {item.strip() for item in value.split(",") if item.strip()}


def crawl(args: argparse.Namespace) -> None:
    raw_heroes = load_json(args.hero_list_url)
    raw_skin_data = load_json(args.skin_list_url)
    raw_details = raw_skin_data.get("pflb20_3469", [])
    heroes, skins = merge_catalog(raw_heroes, raw_details, limit=args.limit)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.db.parent.mkdir(parents=True, exist_ok=True)
    requested_asset_types = selected_asset_types(args.download_assets)

    conn = sqlite3.connect(args.db)
    downloaded = 0
    failed = 0
    asset_count = 0
    try:
        ensure_schema(conn)
        run_id = save_run_start(conn, args.hero_list_url, args.skin_list_url)
        for hero in heroes:
            save_hero(conn, hero)

        for skin in skins:
            image_path = None
            assets = asset_candidates(skin)
            processed_assets: list[AssetRecord] = []
            for asset in assets:
                should_download = not args.skip_images and asset.asset_type in requested_asset_types
                if should_download:
                    asset = download_asset(skin, asset, args.output_dir, overwrite=args.overwrite)
                    if asset.asset_type == "skin_primary" and asset.local_path:
                        image_path = Path(asset.local_path)
                    if asset.download_status in {"downloaded", "existing"}:
                        downloaded += 1
                    elif asset.download_status == "failed":
                        failed += 1
                        print(f"[warn] image failed: {skin.hero_name} / {skin.skin_name}: {asset.error}")
                    if args.sleep:
                        time.sleep(args.sleep)
                else:
                    asset.download_status = "skipped"
                processed_assets.append(asset)
                asset_count += 1
            save_skin(conn, skin, image_path)
            for asset in processed_assets:
                save_asset(conn, asset)

        save_run_finish(conn, run_id, len(heroes), len(skins), asset_count, downloaded, failed)
        conn.commit()
    finally:
        conn.close()

    with_detail = sum(1 for skin in skins if skin.has_detail_record)
    print(
        f"heroes={len(heroes)} skins={len(skins)} with_detail={with_detail} "
        f"assets={asset_count} downloaded_or_existing={downloaded} image_failed={failed}"
    )
    print(f"db={args.db}")
    print(f"images={args.output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch WZRY hero/skin catalog data and images.")
    parser.add_argument("--hero-list-url", default=HERO_LIST_URL)
    parser.add_argument("--skin-list-url", "--data-url", dest="skin_list_url", default=SKIN_LIST_URL)
    parser.add_argument("--output-dir", type=Path, default=Path("data/wzry_skins/images"))
    parser.add_argument("--db", type=Path, default=Path("data/wzry_skins/skins.sqlite3"))
    parser.add_argument("--limit", type=int, default=0, help="Only collect first N catalog skins; 0 means all.")
    parser.add_argument("--sleep", type=float, default=0.05, help="Delay between downloaded image requests.")
    parser.add_argument("--overwrite", action="store_true", help="Redownload existing images.")
    parser.add_argument("--skip-images", action="store_true", help="Collect metadata and asset URLs without downloading images.")
    parser.add_argument(
        "--download-assets",
        default="skin_primary",
        help="Comma-separated asset types to download, or 'all'. Default: skin_primary.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    crawl(parse_args())

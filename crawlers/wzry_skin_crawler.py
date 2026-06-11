"""Fetch Honor of Kings skin metadata/images and store them in SQLite.

Data source used by https://pvp.qq.com/coming/v2/index.shtml:
https://pvp.qq.com/zlkdatasys/heroskinlist.json
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


DATA_URL = "https://pvp.qq.com/zlkdatasys/heroskinlist.json"
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125 Safari/537.36"
)


def request_bytes(url: str, timeout: float = 30.0) -> bytes:
    if url.startswith("//"):
        url = "https:" + url
    req = Request(url, headers={"User-Agent": DEFAULT_USER_AGENT})
    with urlopen(req, timeout=timeout) as resp:
        return resp.read()


def load_json(url: str) -> dict[str, Any]:
    return json.loads(request_bytes(url).decode("utf-8"))


def safe_filename(value: str, max_length: int = 140) -> str:
    value = re.sub(r'[\\/:*?"<>|\r\n\t]+', "_", value).strip(" .")
    value = re.sub(r"\s+", " ", value)
    return (value or "unknown")[:max_length]


def image_extension(url: str) -> str:
    path = urlparse("https:" + url if url.startswith("//") else url).path
    suffix = Path(path).suffix.lower()
    return suffix if suffix in {".jpg", ".jpeg", ".png", ".gif", ".webp"} else ".jpg"


def parse_price_text(acquire_method: str) -> str | None:
    """Extract explicit price strings such as '60点券' from the source text."""
    matches = re.findall(r"\d+(?:\.\d+)?\s*(?:点券|荣耀积分|皮肤碎片)", acquire_method or "")
    return " / ".join(matches) if matches else None


def normalize_skin(raw: dict[str, Any], source_index: int) -> dict[str, Any]:
    acquire_method = raw.get("hqfs_8609", "")
    image_url = raw.get("fmlb_4536") or raw.get("fmb1lb_5300") or raw.get("yxtxlb_8443") or ""
    skin_id = raw.get("pfidlb_3934", "")
    return {
        "source_key": f"{source_index:04d}-{skin_id}",
        "source_index": source_index,
        "skin_id": skin_id,
        "skin_name": raw.get("pfmclb_7523", ""),
        "hero_name": raw.get("yxmclb_9965", ""),
        "online_date": raw.get("sxsjlb_1516", ""),
        "intro": raw.get("yjhjsl_5003", ""),
        "quality": raw.get("pfpzlb_3289", ""),
        "acquire_method": acquire_method,
        "price_text": parse_price_text(acquire_method),
        "image_url": image_url,
        "detail_url": raw.get("pcljlb_9272", ""),
        "mobile_url": raw.get("mdljlb_1924", ""),
        "video_id": raw.get("spvidl_6663", ""),
        "raw_json": json.dumps(raw, ensure_ascii=False, separators=(",", ":")),
    }


def ensure_schema(conn: sqlite3.Connection) -> None:
    if conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='skins'").fetchone():
        columns = {row[1] for row in conn.execute("PRAGMA table_info(skins)")}
        if "source_key" not in columns:
            conn.execute("DROP TABLE skins")

    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS heroes (
            hero_name TEXT PRIMARY KEY,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS skins (
            source_key TEXT PRIMARY KEY,
            source_index INTEGER NOT NULL,
            skin_id TEXT NOT NULL,
            hero_name TEXT NOT NULL,
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
            raw_json TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(hero_name) REFERENCES heroes(hero_name)
        );

        CREATE INDEX IF NOT EXISTS idx_skins_hero_name ON skins(hero_name);
        CREATE INDEX IF NOT EXISTS idx_skins_skin_name ON skins(skin_name);
        """
    )


def save_skin(conn: sqlite3.Connection, skin: dict[str, Any], image_path: Path | None) -> None:
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    conn.execute(
        """
        INSERT INTO heroes(hero_name, updated_at)
        VALUES(?, ?)
        ON CONFLICT(hero_name) DO UPDATE SET updated_at = excluded.updated_at
        """,
        (skin["hero_name"], now),
    )
    conn.execute(
        """
        INSERT INTO skins(
            source_key, source_index, skin_id, hero_name, skin_name, quality, online_date, acquire_method,
            price_text, image_url, image_path, detail_url, mobile_url, video_id,
            intro, raw_json, updated_at
        )
        VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(source_key) DO UPDATE SET
            source_index = excluded.source_index,
            skin_id = excluded.skin_id,
            hero_name = excluded.hero_name,
            skin_name = excluded.skin_name,
            quality = excluded.quality,
            online_date = excluded.online_date,
            acquire_method = excluded.acquire_method,
            price_text = excluded.price_text,
            image_url = excluded.image_url,
            image_path = excluded.image_path,
            detail_url = excluded.detail_url,
            mobile_url = excluded.mobile_url,
            video_id = excluded.video_id,
            intro = excluded.intro,
            raw_json = excluded.raw_json,
            updated_at = excluded.updated_at
        """,
        (
            skin["source_key"],
            skin["source_index"],
            skin["skin_id"],
            skin["hero_name"],
            skin["skin_name"],
            skin["quality"],
            skin["online_date"],
            skin["acquire_method"],
            skin["price_text"],
            skin["image_url"],
            str(image_path) if image_path else None,
            skin["detail_url"],
            skin["mobile_url"],
            skin["video_id"],
            skin["intro"],
            skin["raw_json"],
            now,
        ),
    )


def download_image(skin: dict[str, Any], image_dir: Path, overwrite: bool = False) -> Path | None:
    url = skin["image_url"]
    if not url:
        return None

    image_dir.mkdir(parents=True, exist_ok=True)
    filename = safe_filename(
        f"{skin['source_index']:04d}-{skin['hero_name']}-{skin['skin_id']}-{skin['skin_name']}"
    )
    path = image_dir / f"{filename}{image_extension(url)}"
    if path.exists() and not overwrite:
        return path

    path.write_bytes(request_bytes(url))
    return path


def crawl(args: argparse.Namespace) -> None:
    data = load_json(args.data_url)
    raw_skins = data.get("pflb20_3469", [])
    skins = [normalize_skin(item, index + 1) for index, item in enumerate(raw_skins)]
    if args.limit:
        skins = skins[: args.limit]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(args.db)
    try:
        ensure_schema(conn)
        downloaded = 0
        failed = 0
        for skin in skins:
            image_path = None
            try:
                image_path = download_image(skin, args.output_dir, overwrite=args.overwrite)
                downloaded += 1 if image_path else 0
            except (HTTPError, URLError, TimeoutError, OSError) as exc:
                failed += 1
                print(f"[warn] image failed: {skin['hero_name']} / {skin['skin_name']}: {exc}")

            save_skin(conn, skin, image_path)
            if args.sleep:
                time.sleep(args.sleep)
        conn.commit()
    finally:
        conn.close()

    print(f"skins={len(skins)} downloaded_or_existing={downloaded} image_failed={failed}")
    print(f"db={args.db}")
    print(f"images={args.output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch WZRY skin images and metadata.")
    parser.add_argument("--data-url", default=DATA_URL)
    parser.add_argument("--output-dir", type=Path, default=Path("data/wzry_skins/images"))
    parser.add_argument("--db", type=Path, default=Path("data/wzry_skins/skins.sqlite3"))
    parser.add_argument("--limit", type=int, default=0, help="Only fetch first N skins; 0 means all.")
    parser.add_argument("--sleep", type=float, default=0.05, help="Delay between image requests.")
    parser.add_argument("--overwrite", action="store_true", help="Redownload existing images.")
    return parser.parse_args()


if __name__ == "__main__":
    crawl(parse_args())

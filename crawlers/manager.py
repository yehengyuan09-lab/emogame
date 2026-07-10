"""Phase 1 crawler coordination helpers.

The long-term docs describe official, community, and market crawlers. The
current repo has a working WZRY official skin database and Weibo comment
ingestion, so this manager exposes those concrete sources behind a stable
local API for the feature and UI layers.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SKIN_DB = PROJECT_ROOT / "data" / "wzry_skins" / "skins.sqlite3"
DEFAULT_WEIBO_DIR = PROJECT_ROOT / "data" / "weibo_comments"


@dataclass(frozen=True)
class SkinSummary:
    """A single official skin row."""

    source_key: str
    skin_id: str
    hero_name: str
    skin_name: str
    quality: str
    online_date: str
    price_text: str | None
    image_path: str | None

    @property
    def display_name(self) -> str:
        return f"{self.hero_name} - {self.skin_name}"


class CrawlerManager:
    """Facade over the available Phase 1 data-ingestion outputs."""

    def __init__(
        self,
        skin_db: str | Path = DEFAULT_SKIN_DB,
        weibo_dir: str | Path = DEFAULT_WEIBO_DIR,
    ) -> None:
        self.skin_db = Path(skin_db)
        self.weibo_dir = Path(weibo_dir)

    def list_heroes(self) -> list[str]:
        """Return hero names from the official skin database."""
        if not self.skin_db.exists():
            return []
        with self._connect_skin_db() as conn:
            rows = conn.execute(
                "SELECT hero_name FROM heroes ORDER BY hero_name"
            ).fetchall()
        return [str(row["hero_name"]) for row in rows]

    def list_skins(
        self,
        hero_name: str | None = None,
        limit: int = 200,
    ) -> list[SkinSummary]:
        """Return official skins, optionally filtered by hero."""
        if not self.skin_db.exists():
            return []
        query = """
            SELECT source_key, skin_id, hero_name, skin_name, quality,
                   online_date, price_text, image_path
            FROM skins
        """
        params: list[Any] = []
        if hero_name:
            query += " WHERE hero_name = ?"
            params.append(hero_name)
        query += " ORDER BY source_index LIMIT ?"
        params.append(limit)
        with self._connect_skin_db() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._skin_from_row(row) for row in rows]

    def get_skin(self, source_key: str) -> SkinSummary | None:
        """Return one skin by crawler source key."""
        if not self.skin_db.exists():
            return None
        with self._connect_skin_db() as conn:
            row = conn.execute(
                """
                SELECT source_key, skin_id, hero_name, skin_name, quality,
                       online_date, price_text, image_path
                FROM skins
                WHERE source_key = ?
                """,
                (source_key,),
            ).fetchone()
        return self._skin_from_row(row) if row else None

    def search_skins(self, text: str, limit: int = 50) -> list[SkinSummary]:
        """Search by hero or skin name."""
        if not text.strip() or not self.skin_db.exists():
            return []
        pattern = f"%{text.strip()}%"
        with self._connect_skin_db() as conn:
            rows = conn.execute(
                """
                SELECT source_key, skin_id, hero_name, skin_name, quality,
                       online_date, price_text, image_path
                FROM skins
                WHERE hero_name LIKE ? OR skin_name LIKE ?
                ORDER BY source_index
                LIMIT ?
                """,
                (pattern, pattern, limit),
            ).fetchall()
        return [self._skin_from_row(row) for row in rows]

    def load_weibo_comment_sets(self) -> list[dict[str, Any]]:
        """Return summary rows for saved Weibo comment crawl outputs."""
        rows: list[dict[str, Any]] = []
        for path in sorted(self.weibo_dir.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(data, list):
                continue
            comments = sum(len(item.get("comments", [])) for item in data if isinstance(item, dict))
            meaningful = sum(int(item.get("meaningful", 0)) for item in data if isinstance(item, dict))
            rows.append(
                {
                    "path": str(path),
                    "name": path.name,
                    "posts": len(data),
                    "comments": comments,
                    "meaningful": meaningful,
                }
            )
        return rows

    @contextmanager
    def _connect_skin_db(self):
        conn = sqlite3.connect(str(self.skin_db))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    @staticmethod
    def _skin_from_row(row: sqlite3.Row) -> SkinSummary:
        return SkinSummary(
            source_key=str(row["source_key"]),
            skin_id=str(row["skin_id"]),
            hero_name=str(row["hero_name"]),
            skin_name=str(row["skin_name"]),
            quality=str(row["quality"] or ""),
            online_date=str(row["online_date"] or ""),
            price_text=row["price_text"],
            image_path=row["image_path"],
        )

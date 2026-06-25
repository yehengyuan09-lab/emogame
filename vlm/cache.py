"""SQLite-based result cache for the VLM pipeline.

Caches L1 / L2 / L3 results keyed by ``(image_path_hash, level)`` with a
configurable TTL (default 30 days).  Content-hash change detection
automatically invalidates entries when the underlying image file changes.

Uses Python's stdlib ``sqlite3`` — no extra dependencies.
"""

from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from vlm.config import get_settings
from vlm.utils import image_path_hash


class CacheManager:
    """SQLite-backed cache for VLM pipeline results."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.db_path = Path(self.settings.cache_dir) / "vlm_cache.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ── schema ──

    def _init_db(self) -> None:
        with self._get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS vlm_cache (
                    cache_key   TEXT    NOT NULL,
                    level       TEXT    NOT NULL,   -- 'l1' | 'l2' | 'l3'
                    content_hash TEXT,              -- SHA-256 of image bytes
                    result      TEXT    NOT NULL,   -- JSON string
                    created_at  REAL    NOT NULL,   -- Unix timestamp
                    PRIMARY KEY (cache_key, level)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_created
                ON vlm_cache(created_at)
            """)

    @contextmanager
    def _get_conn(self):
        conn = sqlite3.connect(str(self.db_path))
        try:
            yield conn
        finally:
            conn.close()

    # ── TTL logic ──

    def _is_expired(self, created_at: float) -> bool:
        age_days = (time.time() - created_at) / 86400.0
        return age_days > self.settings.cache_ttl_days

    # ── public API ──

    def get(
        self, image_path: Path, level: str, content_hash: str | None = None
    ) -> dict | None:
        """Retrieve a cached result.

        Returns ``None`` when the entry is missing, expired, or the image
        content has changed since it was cached.
        """
        cache_key = image_path_hash(image_path)
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT result, created_at, content_hash "
                "FROM vlm_cache WHERE cache_key = ? AND level = ?",
                (cache_key, level),
            ).fetchone()

        if row is None:
            return None

        result_str, created_at, stored_hash = row

        if self._is_expired(created_at):
            self._delete(cache_key, level)
            return None

        if content_hash is not None and stored_hash is not None and stored_hash != content_hash:
            self._delete(cache_key, level)
            return None

        return json.loads(result_str)

    def set(
        self,
        image_path: Path,
        level: str,
        result: dict,
        content_hash: str | None = None,
    ) -> None:
        """Store a result in the cache (upsert)."""
        cache_key = image_path_hash(image_path)
        with self._get_conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO vlm_cache "
                "(cache_key, level, content_hash, result, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    cache_key,
                    level,
                    content_hash,
                    json.dumps(result, ensure_ascii=False),
                    time.time(),
                ),
            )
            conn.commit()

    def invalidate(self, image_path: Path) -> None:
        """Remove all cache entries for an image."""
        cache_key = image_path_hash(image_path)
        with self._get_conn() as conn:
            conn.execute(
                "DELETE FROM vlm_cache WHERE cache_key = ?", (cache_key,)
            )
            conn.commit()

    def cleanup(self) -> int:
        """Purge all expired entries.  Returns the count of removed rows."""
        cutoff = time.time() - self.settings.cache_ttl_days * 86400
        with self._get_conn() as conn:
            cursor = conn.execute(
                "DELETE FROM vlm_cache WHERE created_at < ?", (cutoff,)
            )
            conn.commit()
            return cursor.rowcount

    # ── internal ──

    def _delete(self, cache_key: str, level: str) -> None:
        with self._get_conn() as conn:
            conn.execute(
                "DELETE FROM vlm_cache WHERE cache_key = ? AND level = ?",
                (cache_key, level),
            )
            conn.commit()

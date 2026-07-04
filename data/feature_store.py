"""SQLite feature-store primitives for Phase 1.

This module is the small local storage layer that the roadmap calls
"SQLite schema + CRUD API". It stores JSON feature vectors by skin key and
keeps the schema narrow so the Phase 2 feature pipeline can evolve without
database churn.
"""

from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_DB_PATH = Path("data/emogame.db")


@dataclass(frozen=True)
class FeatureRecord:
    """A feature vector row returned by ``FeatureStore``."""

    skin_key: str
    payload: dict[str, Any]
    source: str
    created_at: float
    updated_at: float

    def is_stale(self, ttl_days: int = 30) -> bool:
        """Return True when the record is older than ``ttl_days``."""
        return (time.time() - self.updated_at) > ttl_days * 86400


class FeatureStore:
    """Small SQLite-backed CRUD API for skin feature vectors."""

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS skin_features (
                    skin_key   TEXT PRIMARY KEY,
                    payload    TEXT NOT NULL,
                    source     TEXT NOT NULL DEFAULT 'manual',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_skin_features_updated_at
                ON skin_features(updated_at)
                """
            )
            conn.commit()

    def put(
        self,
        skin_key: str,
        payload: dict[str, Any],
        source: str = "manual",
    ) -> FeatureRecord:
        """Create or replace the feature vector for one skin."""
        now = time.time()
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        with self._connect() as conn:
            existing = conn.execute(
                "SELECT created_at FROM skin_features WHERE skin_key = ?",
                (skin_key,),
            ).fetchone()
            created_at = float(existing["created_at"]) if existing else now
            conn.execute(
                """
                INSERT INTO skin_features(skin_key, payload, source, created_at, updated_at)
                VALUES(?, ?, ?, ?, ?)
                ON CONFLICT(skin_key) DO UPDATE SET
                    payload = excluded.payload,
                    source = excluded.source,
                    updated_at = excluded.updated_at
                """,
                (skin_key, encoded, source, created_at, now),
            )
            conn.commit()
        return FeatureRecord(skin_key, dict(payload), source, created_at, now)

    def get(self, skin_key: str) -> FeatureRecord | None:
        """Return one feature record, or ``None`` when absent."""
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT skin_key, payload, source, created_at, updated_at
                FROM skin_features
                WHERE skin_key = ?
                """,
                (skin_key,),
            ).fetchone()
        return self._row_to_record(row) if row else None

    def delete(self, skin_key: str) -> bool:
        """Delete one feature record and return whether anything changed."""
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM skin_features WHERE skin_key = ?",
                (skin_key,),
            )
            conn.commit()
            return cursor.rowcount > 0

    def list(
        self,
        limit: int = 100,
        offset: int = 0,
    ) -> list[FeatureRecord]:
        """Return feature records ordered by most recently updated."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT skin_key, payload, source, created_at, updated_at
                FROM skin_features
                ORDER BY updated_at DESC
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            ).fetchall()
        return [self._row_to_record(row) for row in rows]

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> FeatureRecord:
        return FeatureRecord(
            skin_key=str(row["skin_key"]),
            payload=json.loads(row["payload"]),
            source=str(row["source"]),
            created_at=float(row["created_at"]),
            updated_at=float(row["updated_at"]),
        )

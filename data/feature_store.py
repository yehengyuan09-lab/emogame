"""SQLite feature-store primitives.

Stores canonical feature vectors by skin key.  Phase 2 extends the schema
with ``schema_version``, ``image_hash``, ``validation_status``, and
``validation_errors`` columns while keeping Phase 1 records readable.
"""

from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
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
    schema_version: int = 1
    image_hash: str | None = None
    validation_status: str = "unvalidated"
    validation_errors: list[str] = field(default_factory=list)

    def is_stale(self, ttl_days: int = 30) -> bool:
        """Return True when the record is older than ``ttl_days``."""
        return (time.time() - self.updated_at) > ttl_days * 86400

    def is_phase2(self) -> bool:
        """Return True when this record uses the Phase 2 schema (v2+)."""
        return self.schema_version >= 2


class FeatureStore:
    """SQLite-backed CRUD API for skin feature vectors."""

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
        # Phase 2 migration — add new columns if they don't exist yet.
        self._migrate_v2()

    def _migrate_v2(self) -> None:
        """Add Phase 2 columns (schema_version, image_hash, validation_status,
        validation_errors).  ``ALTER TABLE ADD COLUMN`` is a no-op in SQLite
        when the column already exists, so this is safe to call repeatedly."""
        with self._connect() as conn:
            for col_ddl in [
                "ALTER TABLE skin_features ADD COLUMN schema_version INTEGER DEFAULT 1",
                "ALTER TABLE skin_features ADD COLUMN image_hash TEXT",
                "ALTER TABLE skin_features ADD COLUMN validation_status TEXT DEFAULT 'unvalidated'",
                "ALTER TABLE skin_features ADD COLUMN validation_errors TEXT DEFAULT '[]'",
            ]:
                try:
                    conn.execute(col_ddl)
                except sqlite3.OperationalError:
                    # Column already exists — skip.
                    pass
            conn.commit()

    def put(
        self,
        skin_key: str,
        payload: dict[str, Any],
        source: str = "manual",
        schema_version: int = 2,
        image_hash: str | None = None,
        validation_status: str = "unvalidated",
        validation_errors: list[str] | None = None,
    ) -> FeatureRecord:
        """Create or replace the feature vector for one skin."""
        now = time.time()
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        errors_json = json.dumps(
            validation_errors or [], ensure_ascii=False
        )
        with self._connect() as conn:
            existing = conn.execute(
                "SELECT created_at FROM skin_features WHERE skin_key = ?",
                (skin_key,),
            ).fetchone()
            created_at = float(existing["created_at"]) if existing else now
            conn.execute(
                """
                INSERT INTO skin_features(
                    skin_key, payload, source, created_at, updated_at,
                    schema_version, image_hash, validation_status, validation_errors
                )
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(skin_key) DO UPDATE SET
                    payload = excluded.payload,
                    source = excluded.source,
                    updated_at = excluded.updated_at,
                    schema_version = excluded.schema_version,
                    image_hash = excluded.image_hash,
                    validation_status = excluded.validation_status,
                    validation_errors = excluded.validation_errors
                """,
                (
                    skin_key, encoded, source, created_at, now,
                    schema_version, image_hash, validation_status, errors_json,
                ),
            )
            conn.commit()
        return FeatureRecord(
            skin_key, dict(payload), source, created_at, now,
            schema_version=schema_version,
            image_hash=image_hash,
            validation_status=validation_status,
            validation_errors=validation_errors or [],
        )

    def get(self, skin_key: str) -> FeatureRecord | None:
        """Return one feature record, or ``None`` when absent."""
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT skin_key, payload, source, created_at, updated_at,
                       schema_version, image_hash, validation_status, validation_errors
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
                SELECT skin_key, payload, source, created_at, updated_at,
                       schema_version, image_hash, validation_status, validation_errors
                FROM skin_features
                ORDER BY updated_at DESC
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def list_all(self) -> list[FeatureRecord]:
        """Return every feature record.  Use sparingly on large datasets."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT skin_key, payload, source, created_at, updated_at,
                       schema_version, image_hash, validation_status, validation_errors
                FROM skin_features
                ORDER BY skin_key
                """
            ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def count(self) -> int:
        """Return total number of rows in the store."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS cnt FROM skin_features"
            ).fetchone()
            return int(row["cnt"]) if row else 0

    def count_by_status(self, status: str) -> int:
        """Count records with a specific ``validation_status``."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS cnt FROM skin_features WHERE validation_status = ?",
                (status,),
            ).fetchone()
            return int(row["cnt"]) if row else 0

    def get_phase2_records(self) -> list[FeatureRecord]:
        """Return all records with ``schema_version >= 2``."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT skin_key, payload, source, created_at, updated_at,
                       schema_version, image_hash, validation_status, validation_errors
                FROM skin_features
                WHERE schema_version >= 2
                ORDER BY skin_key
                """
            ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def get_by_status(self, status: str) -> list[FeatureRecord]:
        """Return records with the given pipeline/validation status."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT skin_key, payload, source, created_at, updated_at,
                       schema_version, image_hash, validation_status, validation_errors
                FROM skin_features
                WHERE validation_status = ?
                ORDER BY skin_key
                """,
                (status,),
            ).fetchall()
        return [self._row_to_record(row) for row in rows]

    # ── internal helpers ──

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> FeatureRecord:
        errors_raw = row["validation_errors"] if "validation_errors" in row.keys() else "[]"
        try:
            validation_errors: list[str] = json.loads(errors_raw) if errors_raw else []
        except (json.JSONDecodeError, TypeError):
            validation_errors = []

        return FeatureRecord(
            skin_key=str(row["skin_key"]),
            payload=json.loads(row["payload"]),
            source=str(row["source"]),
            created_at=float(row["created_at"]),
            updated_at=float(row["updated_at"]),
            schema_version=int(row["schema_version"]) if row["schema_version"] is not None else 1,
            image_hash=str(row["image_hash"]) if row["image_hash"] is not None else None,
            validation_status=str(row["validation_status"]) if row["validation_status"] is not None else "unvalidated",
            validation_errors=validation_errors,
        )

"""Persistence for public-opinion and market evidence signals."""

from __future__ import annotations

import json
import sqlite3
import time
from contextlib import closing
from dataclasses import asdict
from pathlib import Path
from typing import Any

from data.skin_repository import DEFAULT_DB_PATH
from feature_engineering.features import MarketValidationSignals


SIGNAL_FIELDS = (
    "visual_score",
    "feel_score",
    "craftsmanship_score",
    "collection_score",
    "value_score",
    "purchase_intent_score",
    "sentiment_score",
    "discussion_count",
    "video_views",
    "marketing_volume",
    "sales_volume",
    "avg_spend_to_obtain",
    "ownership_rate",
)


class MarketSignalRepository:
    """Store and retrieve market/opinion signals for skins."""

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def _rows(cursor: sqlite3.Cursor) -> list[dict[str, Any]]:
        return [dict(row) for row in cursor.fetchall()]

    def ensure_schema(self) -> None:
        with closing(self._connect()) as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS market_signal_records (
                    source_key TEXT PRIMARY KEY,
                    visual_score REAL,
                    feel_score REAL,
                    craftsmanship_score REAL,
                    collection_score REAL,
                    value_score REAL,
                    purchase_intent_score REAL,
                    sentiment_score REAL,
                    discussion_count INTEGER,
                    video_views INTEGER,
                    marketing_volume INTEGER,
                    sales_volume INTEGER,
                    avg_spend_to_obtain REAL,
                    ownership_rate REAL,
                    signal_source TEXT,
                    evidence_count INTEGER NOT NULL DEFAULT 0,
                    raw_json TEXT NOT NULL DEFAULT '{}',
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS opinion_evidence_items (
                    evidence_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_key TEXT NOT NULL,
                    platform TEXT NOT NULL,
                    external_id TEXT,
                    url TEXT,
                    title TEXT,
                    author TEXT,
                    published_at TEXT,
                    text TEXT,
                    metrics_json TEXT NOT NULL DEFAULT '{}',
                    aspect_tags TEXT NOT NULL DEFAULT '[]',
                    raw_json TEXT NOT NULL DEFAULT '{}',
                    collected_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_opinion_evidence_source_key
                    ON opinion_evidence_items(source_key);
                CREATE INDEX IF NOT EXISTS idx_opinion_evidence_platform
                    ON opinion_evidence_items(platform);
                CREATE UNIQUE INDEX IF NOT EXISTS idx_opinion_evidence_unique_external
                    ON opinion_evidence_items(source_key, platform, external_id)
                    WHERE external_id IS NOT NULL;
                """
            )
            conn.commit()

    def upsert_signals(
        self,
        source_key: str,
        signals: MarketValidationSignals | dict[str, Any],
        *,
        signal_source: str = "manual",
        raw_json: dict[str, Any] | None = None,
        merge: bool = True,
    ) -> None:
        """Insert or update aggregate signals.

        When ``merge`` is true, missing incoming fields keep the existing value.
        """
        self.ensure_schema()
        signal_obj = signals if isinstance(signals, MarketValidationSignals) else MarketValidationSignals.from_dict(signals)
        values = asdict(signal_obj)
        now = time.strftime("%Y-%m-%d %H:%M:%S")

        with closing(self._connect()) as conn:
            existing = conn.execute(
                "SELECT * FROM market_signal_records WHERE source_key = ?",
                (source_key,),
            ).fetchone()
            if existing and merge:
                merged = dict(existing)
                for field in SIGNAL_FIELDS:
                    if values.get(field) is not None:
                        merged[field] = values[field]
                evidence_count = merged.get("evidence_count", 0) or 0
            else:
                merged = {field: values.get(field) for field in SIGNAL_FIELDS}
                evidence_count = 0

            conn.execute(
                """
                INSERT INTO market_signal_records(
                    source_key, visual_score, feel_score, craftsmanship_score,
                    collection_score, value_score, purchase_intent_score,
                    sentiment_score, discussion_count, video_views, marketing_volume,
                    sales_volume, avg_spend_to_obtain, ownership_rate, signal_source,
                    evidence_count, raw_json, updated_at
                )
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_key) DO UPDATE SET
                    visual_score = excluded.visual_score,
                    feel_score = excluded.feel_score,
                    craftsmanship_score = excluded.craftsmanship_score,
                    collection_score = excluded.collection_score,
                    value_score = excluded.value_score,
                    purchase_intent_score = excluded.purchase_intent_score,
                    sentiment_score = excluded.sentiment_score,
                    discussion_count = excluded.discussion_count,
                    video_views = excluded.video_views,
                    marketing_volume = excluded.marketing_volume,
                    sales_volume = excluded.sales_volume,
                    avg_spend_to_obtain = excluded.avg_spend_to_obtain,
                    ownership_rate = excluded.ownership_rate,
                    signal_source = excluded.signal_source,
                    evidence_count = excluded.evidence_count,
                    raw_json = excluded.raw_json,
                    updated_at = excluded.updated_at
                """,
                (
                    source_key,
                    *(merged.get(field) for field in SIGNAL_FIELDS),
                    signal_source,
                    evidence_count,
                    json.dumps(raw_json or {}, ensure_ascii=False, separators=(",", ":")),
                    now,
                ),
            )
            conn.commit()

    def add_evidence(
        self,
        source_key: str,
        *,
        platform: str,
        external_id: str | None = None,
        url: str | None = None,
        title: str | None = None,
        author: str | None = None,
        published_at: str | None = None,
        text: str | None = None,
        metrics: dict[str, Any] | None = None,
        aspect_tags: list[str] | None = None,
        raw_json: dict[str, Any] | None = None,
    ) -> None:
        self.ensure_schema()
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        with closing(self._connect()) as conn:
            conn.execute(
                """
                INSERT INTO opinion_evidence_items(
                    source_key, platform, external_id, url, title, author,
                    published_at, text, metrics_json, aspect_tags, raw_json, collected_at
                )
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_key, platform, external_id) WHERE external_id IS NOT NULL
                DO UPDATE SET
                    url = excluded.url,
                    title = excluded.title,
                    author = excluded.author,
                    published_at = excluded.published_at,
                    text = excluded.text,
                    metrics_json = excluded.metrics_json,
                    aspect_tags = excluded.aspect_tags,
                    raw_json = excluded.raw_json,
                    collected_at = excluded.collected_at
                """,
                (
                    source_key,
                    platform,
                    external_id,
                    url,
                    title,
                    author,
                    published_at,
                    text,
                    json.dumps(metrics or {}, ensure_ascii=False, separators=(",", ":")),
                    json.dumps(aspect_tags or [], ensure_ascii=False, separators=(",", ":")),
                    json.dumps(raw_json or {}, ensure_ascii=False, separators=(",", ":")),
                    now,
                ),
            )
            conn.commit()

    def get_signals(self, source_key: str) -> MarketValidationSignals:
        self.ensure_schema()
        with closing(self._connect()) as conn:
            row = conn.execute(
                """
                SELECT visual_score, feel_score, craftsmanship_score, collection_score,
                       value_score, purchase_intent_score, sentiment_score,
                       discussion_count, video_views, marketing_volume, sales_volume,
                       avg_spend_to_obtain, ownership_rate
                FROM market_signal_records
                WHERE source_key = ?
                """,
                (source_key,),
            ).fetchone()
        return MarketValidationSignals.from_dict(dict(row) if row else {})

    def list_evidence(self, source_key: str) -> list[dict[str, Any]]:
        self.ensure_schema()
        with closing(self._connect()) as conn:
            rows = self._rows(
                conn.execute(
                    """
                    SELECT evidence_id, source_key, platform, external_id, url, title,
                           author, published_at, text, metrics_json, aspect_tags,
                           raw_json, collected_at
                    FROM opinion_evidence_items
                    WHERE source_key = ?
                    ORDER BY collected_at DESC, evidence_id DESC
                    """,
                    (source_key,),
                )
            )
        for row in rows:
            row["metrics"] = json.loads(row.pop("metrics_json") or "{}")
            row["aspect_tags"] = json.loads(row["aspect_tags"] or "[]")
            row["raw"] = json.loads(row.pop("raw_json") or "{}")
        return rows

    def aggregate_evidence_signals(self, source_key: str) -> MarketValidationSignals:
        """Aggregate numeric evidence into coarse validation signals."""
        evidence = self.list_evidence(source_key)
        video_views = 0
        discussion_count = 0
        marketing_volume = 0
        evidence_count = 0
        for item in evidence:
            metrics = item.get("metrics") or {}
            evidence_count += 1
            video_views += int(metrics.get("view") or metrics.get("video_views") or 0)
            discussion_count += sum(
                int(metrics.get(name) or 0)
                for name in ("reply", "danmaku", "comment_count", "reply_count", "total_number")
            )
            marketing_volume += sum(
                int(metrics.get(name) or 0)
                for name in (
                    "favorite",
                    "coin",
                    "share",
                    "like",
                    "like_count",
                    "reposts_count",
                    "attitudes_count",
                )
            )

        signals = MarketValidationSignals(
            discussion_count=discussion_count or None,
            video_views=video_views or None,
            marketing_volume=marketing_volume or None,
        )
        self.upsert_signals(
            source_key,
            signals,
            signal_source="evidence_aggregate",
            raw_json={"evidence_count": evidence_count},
            merge=True,
        )
        self._set_evidence_count(source_key, evidence_count)
        return signals

    def _set_evidence_count(self, source_key: str, evidence_count: int) -> None:
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        with closing(self._connect()) as conn:
            conn.execute(
                """
                UPDATE market_signal_records
                SET evidence_count = ?, updated_at = ?
                WHERE source_key = ?
                """,
                (evidence_count, now, source_key),
            )
            conn.commit()

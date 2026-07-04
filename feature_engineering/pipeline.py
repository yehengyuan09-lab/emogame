"""Minimal Phase 1 feature pipeline.

Phase 2 will fill in the complete 33-dimensional feature vector. For Phase 1,
this wrapper stores whatever feature payload has been produced for a skin and
hydrates basic official metadata from the crawler manager when available.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from crawlers.manager import CrawlerManager
from data.feature_store import FeatureRecord, FeatureStore


class FeaturePipeline:
    """Bridge crawler metadata, VLM output, and the SQLite feature store."""

    def __init__(
        self,
        feature_store: FeatureStore | None = None,
        crawler_manager: CrawlerManager | None = None,
    ) -> None:
        self.store = feature_store or FeatureStore()
        self.crawlers = crawler_manager or CrawlerManager()

    def get(self, skin_key: str) -> FeatureRecord | None:
        """Return cached features for one skin."""
        return self.store.get(skin_key)

    def save_manual_features(
        self,
        skin_key: str,
        features: dict[str, Any],
        source: str = "manual",
    ) -> FeatureRecord:
        """Persist a feature payload exactly as supplied."""
        return self.store.put(skin_key, features, source=source)

    def seed_from_official_skin(self, source_key: str) -> FeatureRecord | None:
        """Create a baseline feature record from official crawler metadata."""
        skin = self.crawlers.get_skin(source_key)
        if skin is None:
            return None
        payload = {
            "skin_key": skin.source_key,
            "skin_id": skin.skin_id,
            "hero_name": skin.hero_name,
            "skin_name": skin.skin_name,
            "quality": skin.quality,
            "online_date": skin.online_date,
            "price_text": skin.price_text,
            "image_path": skin.image_path,
            "official": asdict(skin),
        }
        return self.store.put(skin.source_key, payload, source="official_seed")

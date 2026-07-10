"""Feature-engineering pipeline — bridges crawler metadata, VLM output, and the feature store.

This module exposes two orchestrators:

* ``FeatureBuilder`` — converts a ``SkinRepository`` row plus optional market
  signals into a single ``SkinFeatureVector`` (used by the sales workbench /
  rule-engine / calibration code paths).
* ``FeaturePipeline`` — the Phase-2 async pipeline that runs the VLM pipeline,
  extracts official features, merges them into a 33-dimensional
  ``SkinFeatureVector``, validates, and persists to the feature store.

Both share the same unified ``SkinFeatureVector`` model defined in
``feature_engineering.features``.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Any

from data.skin_repository import SkinRepository
from feature_engineering.features import (
    FEATURE_FIELD_NAMES,
    MarketValidationSignals,
    SkinFeatureVector,
)
from loguru import logger


QUALITY_TIER_MAP = {
    "": 0,
    "伴生": 0,
    "勇者": 1,
    "勇者限定": 2,
    "史诗": 2,
    "史诗限定": 3,
    "珍品史诗": 3,
    "传说": 4,
    "传说限定": 4,
    "珍品传说": 4,
    "无双": 5,
    "无双限定": 5,
    "荣耀典藏": 5,
}


class FeatureBuilder:
    """Convert repository rows plus optional market signals into features."""

    def __init__(self, repo: SkinRepository, reference_date: date | None = None):
        self.repo = repo
        self.reference_date = reference_date or date.today()

    def build(
        self,
        source_key: str,
        market_signals: MarketValidationSignals | dict[str, Any] | None = None,
    ) -> SkinFeatureVector:
        skin = self.repo.get_skin(source_key)
        if not skin:
            raise ValueError(f"skin not found: {source_key}")

        if isinstance(market_signals, MarketValidationSignals):
            signals = market_signals
        else:
            signals = MarketValidationSignals.from_dict(market_signals)

        hero_skin_count = len(self.repo.list_skins(hero_id=skin["hero_id"], limit=None))
        quality = skin.get("quality") or ""
        acquire_method = skin.get("acquire_method") or ""
        official_tier = quality_to_tier(quality)

        features = SkinFeatureVector(
            skin_key=skin["source_key"],
            source_key=skin["source_key"],
            hero_id=skin["hero_id"],
            hero_name=skin["hero_name"],
            skin_name=skin["skin_name"],
            skin_id=skin.get("skin_id") or "",
            quality=quality,
            acquire_method=acquire_method,
            price_text=skin.get("price_text"),
            official_tier=official_tier,
            quality_score=official_tier / 5,
            is_limited=contains_any(quality + acquire_method, ["限定", "限时", "返场"]),
            is_gacha=contains_any(acquire_method + quality, ["抽奖", "祈愿", "夺宝", "荣耀水晶", "无双"]),
            is_direct_sale=contains_any(acquire_method, ["商城直售", "直售"]),
            is_event=contains_any(acquire_method, ["活动", "福利", "任务"]),
            is_battle_pass=contains_any(acquire_method, ["战令", "礼册"]),
            is_shard_exchange=contains_any(acquire_method, ["碎片", "碎片商城", "碎片商店"]),
            has_detail_record=bool(skin.get("has_detail_record")),
            has_primary_asset=bool(skin.get("primary_asset_url")),
            skin_age_days=parse_age_days(skin.get("online_date"), self.reference_date),
            hero_skin_count=hero_skin_count,
            market_signals=signals,
            provenance={"online_date": skin.get("online_date") or ""},
        )
        return features


def quality_to_tier(quality: str) -> int:
    if quality in QUALITY_TIER_MAP:
        return QUALITY_TIER_MAP[quality]
    if "荣耀典藏" in quality or "无双" in quality:
        return 5
    if "传说" in quality:
        return 4
    if "史诗" in quality:
        return 3 if "限定" in quality else 2
    if "勇者" in quality:
        return 2 if "限定" in quality else 1
    return 0


def contains_any(value: str, needles: list[str]) -> bool:
    return any(needle in value for needle in needles)


def parse_age_days(value: str | None, reference_date: date) -> int | None:
    if not value:
        return None
    for fmt in ("%Y-%m-%d", "%Y%m%d"):
        try:
            parsed = datetime.strptime(value, fmt).date()
            return max(0, (reference_date - parsed).days)
        except ValueError:
            continue
    return None


# ═══════════════════════════════════════════════════════════════════════════
# Phase 2 async pipeline (FeaturePipeline)
# ═══════════════════════════════════════════════════════════════════════════

import time

from crawlers.manager import CrawlerManager, SkinSummary
from data.feature_store import FeatureRecord, FeatureStore
from feature_engineering.official_extractor import OfficialFeatureMapper


class FeaturePipeline:
    """Orchestrates feature extraction for one skin.

    Dependencies are injected so callers can supply mocks or custom
    instances.  When a dependency is ``None`` a sensible default is created
    lazily.
    """

    def __init__(
        self,
        feature_store: FeatureStore | None = None,
        crawler_manager: CrawlerManager | None = None,
        vlm_pipeline: Any | None = None,
        official_mapper: OfficialFeatureMapper | None = None,
    ) -> None:
        self.store = feature_store or FeatureStore()
        self.crawlers = crawler_manager or CrawlerManager()
        self._vlm = vlm_pipeline
        self._mapper = official_mapper or OfficialFeatureMapper()

    # ── Phase 1 API (preserved) ────────────────────────────────────────

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
            "official": {
                "source_key": skin.source_key,
                "skin_id": skin.skin_id,
                "hero_name": skin.hero_name,
                "skin_name": skin.skin_name,
                "quality": skin.quality,
                "online_date": skin.online_date,
                "price_text": skin.price_text,
                "image_path": skin.image_path,
            },
        }
        return self.store.put(skin.source_key, payload, source="official_seed")

    # ── Phase 2 API ────────────────────────────────────────────────────

    async def extract_features(
        self,
        skin_key: str,
        run_l3: bool = False,
        force: bool = False,
        acquire_method: str = "",
        intro: str = "",
    ) -> SkinFeatureVector:
        """Produce and persist one complete 33-dimensional feature vector.

        Args:
            skin_key: Crawler source key for the skin (e.g. ``"0001-56304"``).
            run_l3: If ``True``, run L3 semantic analysis in addition to L1+L2.
            force: If ``True``, reprocess even when a valid Phase-2 record
                already exists in the feature store.
            acquire_method: Raw acquisition text from the SQLite skins table
                (not in ``SkinSummary`` — callers should query it).
            intro: Raw intro/description text from the SQLite skins table.

        Returns:
            A ``SkinFeatureVector`` with ``pipeline_status`` set to
            ``"complete"``, ``"partial"``, or ``"error"``.
        """
        t_start = time.perf_counter()

        # ── 1. Check feature-store cache ───────────────────────────────
        if not force:
            cached = self.store.get(skin_key)
            if cached is not None and cached.is_phase2():
                logger.debug(f"Cache hit for {skin_key} (schema_v={cached.schema_version})")
                cached_vector = SkinFeatureVector(**cached.payload)
                cached_vector.pipeline_status = "cached"
                return cached_vector

        # ── 2. Look up skin metadata ───────────────────────────────────
        skin = self.crawlers.get_skin(skin_key)
        if skin is None:
            logger.error(f"Skin not found in crawler DB: {skin_key}")
            return SkinFeatureVector(
                skin_key=skin_key,
                pipeline_status="error",
                validation_errors=[f"Skin not found: {skin_key}"],
            )

        # ── 3. Run VLM pipeline ────────────────────────────────────────
        vlm_features: dict[str, Any] = {}
        vlm_raw: dict[str, Any] = {}
        pipeline_status = "complete"
        image_hash: str | None = None

        image_path = self._resolve_image_path(skin)
        if image_path is not None:
            try:
                mode_str = "l1_l2" if not run_l3 else "full"
                vlm = self._get_vlm()
                # Import ExecutionMode lazily to avoid hard dependency at class-load
                from vlm.pipeline import ExecutionMode
                mode = ExecutionMode.L1_L2 if not run_l3 else ExecutionMode.FULL
                vlm_features = await vlm.analyze(image_path, mode=mode)

                # Compute image hash
                from vlm.utils import image_content_hash
                image_hash = image_content_hash(image_path)

                # Preserve raw VLM outputs for analysis
                vlm_raw = dict(vlm_features)

                # Adjust status based on VLM result
                vlm_status = vlm_features.get("status", "ok")
                if vlm_status == "error":
                    pipeline_status = "error"
                elif vlm_status in ("degraded", "partial"):
                    pipeline_status = "partial"

                logger.info(
                    f"VLM done for {skin_key}: mode={mode_str} "
                    f"status={vlm_status} elapsed={vlm_features.get('pipeline_elapsed', 0)}s"
                )
            except Exception as exc:
                logger.error(f"VLM pipeline failed for {skin_key}: {exc}")
                pipeline_status = "error"
                vlm_features = {}
        else:
            logger.warning(f"No valid image for {skin_key}, skipping VLM")

        # ── 4. Extract official features ────────────────────────────────
        official_features, provenance = self._mapper.extract_all(
            quality=skin.quality,
            online_date=skin.online_date,
            acquire_method=acquire_method,
            price_text=skin.price_text,
            intro=intro,
            hero_name=skin.hero_name,
            skin_name=skin.skin_name,
        )

        # ── 5. Merge into SkinFeatureVector ────────────────────────────
        vector = SkinFeatureVector.from_dicts(
            skin_key=skin_key,
            skin_id=skin.skin_id,
            hero_name=skin.hero_name,
            skin_name=skin.skin_name,
            vlm_features=vlm_features,
            official_features=official_features,
            image_hash=image_hash,
            pipeline_status=pipeline_status,
            vlm_raw=vlm_raw,
        )

        # Augment provenance with official-mapper output.
        for name, prov in provenance.items():
            if name in FEATURE_FIELD_NAMES and name not in vector.provenance:
                vector.provenance[name] = prov
            elif name in FEATURE_FIELD_NAMES and vector.provenance.get(name) in (None, "", "unavailable"):
                vector.provenance[name] = prov

        # ── 6. Validate ─────────────────────────────────────────────────
        validation_errors = vector.validate_ranges()
        vector.validation_errors = validation_errors
        vector.recompute_availability()

        # ── 7. Persist to feature store ─────────────────────────────────
        el = round(time.perf_counter() - t_start, 2)
        logger.info(
            f"Feature extraction complete for {skin_key}: "
            f"available={sum(vector.availability_array())}/33 "
            f"errors={len(validation_errors)} elapsed={el}s"
        )

        try:
            self.store.put(
                skin_key=skin_key,
                payload=vector.model_dump(exclude_none=False),
                source="phase2_pipeline",
                schema_version=2,
                image_hash=image_hash,
                validation_status="valid" if not validation_errors else "invalid",
                validation_errors=validation_errors,
            )
        except Exception as exc:
            logger.error(f"Failed to persist {skin_key}: {exc}")
            vector.validation_errors.append(f"store_write_failed: {exc}")

        return vector

    # ── internal helpers ───────────────────────────────────────────────

    def _resolve_image_path(self, skin: SkinSummary) -> Path | None:
        """Return the absolute path to a skin's wallpaper image, or ``None``."""
        if not skin.image_path:
            return None
        path = Path(skin.image_path)
        if not path.is_absolute():
            # Relative paths are relative to the project root.
            from crawlers.manager import PROJECT_ROOT
            path = PROJECT_ROOT / path
        return path if path.exists() else None

    def _get_vlm(self) -> Any:
        """Return the VlmPipeline singleton (lazy init)."""
        if self._vlm is None:
            from vlm.pipeline import VlmPipeline
            self._vlm = VlmPipeline()
        return self._vlm

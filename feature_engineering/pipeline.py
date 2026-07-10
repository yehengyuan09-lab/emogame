"""Feature-engineering pipeline — bridges crawler metadata, VLM output, and the feature store.

Phase 2 adds ``extract_features()`` which orchestrates the full flow:
crawler lookup → VLM analysis → official-feature mapping → merge → validate → store.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from loguru import logger

from crawlers.manager import CrawlerManager, SkinSummary
from data.feature_store import FeatureRecord, FeatureStore
from feature_engineering.features import FEATURE_FIELD_NAMES, SkinFeatureVector
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
                return SkinFeatureVector(
                    skin_key=skin_key,
                    pipeline_status="cached",
                    **cached.payload,
                )

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

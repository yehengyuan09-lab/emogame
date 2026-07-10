"""VLM Pipeline orchestrator — the main entry point.

Sequences L1 → L2 → L3, merges results into a flat ``VLMFeatureVector``,
and provides the public ``analyze(image_path)`` method consumed by the
feature-engineering layer.
"""

from __future__ import annotations

import time
from enum import Enum
from pathlib import Path
from typing import Any


class ExecutionMode(str, Enum):
    """Execution mode for the VLM pipeline."""

    L1_L2 = "l1_l2"   # Run only L1 + L2, skip L3
    FULL = "full"      # Run all 3 tiers (default)

from loguru import logger

from vlm.cache import CacheManager
from vlm.config import get_settings
from vlm.degradation import DegradationHandler
from vlm.l1_classifier import L1Classifier
from vlm.l2_analyzer import L2Analyzer
from vlm.l3_semantic import L3Semantic
from vlm.ollama_client import OllamaClient
from vlm.preprocess import Preprocessor
from vlm.schemas import VLMFeatureVector
from vlm.utils import image_content_hash


class VlmPipeline:
    """Orchestrates the 3-tier VLM analysis pipeline for a single skin image.

    Usage::

        pipeline = VlmPipeline()
        result = await pipeline.analyze("path/to/wallpaper.jpg")
        # result is a flat dict with vlm_art_quality, vlm_effect_score, etc.

    Or synchronously::

        result = pipeline.analyze_sync("path/to/wallpaper.jpg")
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.ollama = OllamaClient()
        self.cache = CacheManager()
        self.preprocessor = Preprocessor()
        self.degradation = DegradationHandler(self.ollama)
        self.l1 = L1Classifier(self.ollama, self.cache, self.degradation)
        self.l2 = L2Analyzer(self.ollama, self.cache, self.degradation)
        self.l3 = L3Semantic(self.cache, self.degradation)

    # ── public API ──

    async def analyze(
        self,
        image_path: str | Path,
        mode: ExecutionMode = ExecutionMode.FULL,
    ) -> dict[str, Any]:
        """Run the VLM pipeline on a skin wallpaper image.

        Args:
            image_path: Path to a wallpaper-bigskin image (1920×882).
            mode: ``ExecutionMode.L1_L2`` to skip L3, ``ExecutionMode.FULL``
                (default) to run all three tiers.

        Returns:
            Flat dict with keys matching ``VLMFeatureVector``.  The 4
            required fields are ``vlm_art_quality``, ``vlm_effect_score``,
            ``vlm_color_harmony``, and ``vlm_composition``.
        """
        image_path = Path(image_path)
        if not image_path.exists():
            logger.error(f"Image not found: {image_path}")
            return VLMFeatureVector(status="error").to_dict()

        started = time.perf_counter()
        content_hash = image_content_hash(image_path)

        # ── Preprocessing ──
        _, preprocess_result = self.preprocessor.process(image_path)

        # ── L1: Classification ──
        l1_cache_hit = self.cache.get(image_path, "l1", content_hash) is not None
        l1_result = await self.l1.classify(image_path, preprocess_result)
        logger.debug(f"L1 complete: source={l1_result.get('_source')}")

        # ── L2: Aesthetic Analysis ──
        l2_cache_hit = self.cache.get(image_path, "l2", content_hash) is not None
        l2_result = await self.l2.analyze(image_path)
        needs_l2_fallback = l2_result.get("_source") == "deferred_to_l3"
        logger.debug(f"L2 complete: source={l2_result.get('_source')}")

        # ── L3: Semantic Analysis ──
        l3_cache_hit = False
        l3_result: dict[str, Any] = {"_source": "skipped"}
        if mode == ExecutionMode.L1_L2:
            logger.debug("L3 skipped (mode=l1_l2)")
        else:
            l3_cache_hit = (
                self.cache.get(
                    image_path,
                    "l3_combined" if needs_l2_fallback else "l3",
                    content_hash,
                )
                is not None
            )
            l3_result = await self.l3.analyze(
                image_path,
                l1_result,
                l2_result if not needs_l2_fallback else None,
                needs_l2_fallback=needs_l2_fallback,
            )
            logger.debug(f"L3 complete: source={l3_result.get('_source')}")

        # ── Extract L2 scores from combined response if deferred ──
        if needs_l2_fallback and l3_result.get("_source") in (
            "autodl_combined",
            "cache",
        ):
            l2_fields = {
                "model_detail",
                "effect_quality",
                "color_scheme",
                "composition",
                "uniqueness",
                "costume_design",
                "background_quality",
                "ui_elements",
            }
            l2_result = {k: v for k, v in l3_result.items() if k in l2_fields}
            l2_result["_source"] = "autodl_fallback"

        # ── Determine pipeline status ──
        status = "ok"
        if (
            l1_result.get("_source", "").startswith("cv_fallback")
            or l2_result.get("_source", "")
            in ("autodl_fallback", "deferred_to_l3")
            or l3_result.get("_source", "") == "autodl_combined"
        ):
            status = "degraded"
        if (
            l1_result.get("_source") == "error"
            or l2_result.get("_source") == "error"
            or l3_result.get("_source") == "error"
        ):
            status = "partial"

        # ── Merge into VLMFeatureVector ──
        elapsed = round(time.perf_counter() - started, 2)

        feature_vector = VLMFeatureVector(
            # Required mapped fields
            vlm_art_quality=float(l2_result.get("model_detail", 0)),
            vlm_effect_score=float(l2_result.get("effect_quality", 0)),
            vlm_color_harmony=float(l2_result.get("color_scheme", 0)) / 10.0,
            vlm_composition=float(l2_result.get("composition", 0)),
            # L1 raw
            rarity_tier=str(l1_result.get("rarity_tier", "")),
            dominant_colors=list(l1_result.get("dominant_colors", [])),
            scene_type=str(l1_result.get("scene_type", "")),
            character_ratio=float(l1_result.get("character_ratio", 0)),
            effect_density=str(l1_result.get("effect_density", "")),
            confidence=float(l1_result.get("confidence", 0)),
            # L2 raw
            model_detail=float(l2_result.get("model_detail", 0)),
            effect_quality=float(l2_result.get("effect_quality", 0)),
            color_scheme=float(l2_result.get("color_scheme", 0)),
            uniqueness=float(l2_result.get("uniqueness", 0)),
            costume_design=float(l2_result.get("costume_design", 0)),
            background_quality=float(l2_result.get("background_quality", 0)),
            ui_elements=(
                l2_result.get("ui_elements", {})
                if isinstance(l2_result.get("ui_elements"), dict)
                else {}
            ),
            # L3 raw
            design_style=str(l3_result.get("design_style", "")),
            cultural_references=list(l3_result.get("cultural_references", [])),
            target_audience=list(l3_result.get("target_audience", [])),
            similar_skins=list(l3_result.get("similar_skins", [])),
            differentiation=str(l3_result.get("differentiation", "")),
            # Preprocessing
            edge_density=float(preprocess_result.edge_density),
            sharpness=float(preprocess_result.sharpness),
            dominant_colors_cv=list(preprocess_result.dominant_colors_cv),
            # Metadata
            status=status,
            pipeline_elapsed=elapsed,
            cache_hit_l1=l1_cache_hit,
            cache_hit_l2=l2_cache_hit,
            cache_hit_l3=l3_cache_hit,
            execution_mode=mode.value,
        )

        logger.info(
            f"VLM pipeline complete: {image_path.name} "
            f"status={status} elapsed={elapsed}s "
            f"art={feature_vector.vlm_art_quality:.1f} "
            f"effect={feature_vector.vlm_effect_score:.1f}"
        )
        return feature_vector.to_dict()

    def analyze_sync(
        self,
        image_path: str | Path,
        mode: ExecutionMode = ExecutionMode.FULL,
    ) -> dict[str, Any]:
        """Synchronous wrapper for callers that don't use asyncio."""
        import asyncio

        return asyncio.run(self.analyze(image_path, mode=mode))

"""L1: Qwen2.5VL-3B fast skin classification via Ollama.

Cached, with automatic CV fallback when Ollama is unavailable or errors.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from vlm.cache import CacheManager
from vlm.config import get_settings
from vlm.degradation import DegradationHandler, PipelineTier
from vlm.ollama_client import OllamaClient
from vlm.preprocess import PreprocessResult, Preprocessor
from vlm.prompts import L1_PROMPT
from vlm.schemas import L1Output
from vlm.utils import encode_image


class L1Classifier:
    """L1: fast classification (rarity, colours, scene, effect density)."""

    def __init__(
        self,
        ollama: OllamaClient,
        cache: CacheManager,
        degradation: DegradationHandler,
    ) -> None:
        self.settings = get_settings()
        self.ollama = ollama
        self.cache = cache
        self.degradation = degradation
        self.preprocessor = Preprocessor()

    async def classify(
        self, image_path: Path, preprocess_result: PreprocessResult | None = None
    ) -> dict[str, Any]:
        """Run L1 classification.

        Returns a dict matching ``L1Output`` schema plus ``_source`` metadata.
        """
        tier = await self.degradation.detect_tier()

        # ── CV fallback path ──
        if tier in (PipelineTier.DEGRADED_CV_L1, PipelineTier.API_ONLY):
            if preprocess_result is None:
                _, preprocess_result = self.preprocessor.process(image_path)
            result = await self.degradation.run_l1_cv_fallback(
                preprocess_result, str(image_path)
            )
            result["_source"] = "cv_fallback"
            self.cache.set(image_path, "l1", result)
            return result

        # ── Cache hit ──
        cached = self.cache.get(image_path, "l1")
        if cached is not None:
            cached["_source"] = "cache"
            return cached

        # ── Select model ──
        model = self.settings.l1_model
        try:
            installed = await self.ollama.list_models()
        except Exception:
            installed = set()

        if model not in installed and self.settings.l1_fallback_model and self.settings.l1_fallback_model in installed:
            model = self.settings.l1_fallback_model

        # ── Ollama call ──
        image_b64 = encode_image(image_path)
        try:
            response = await self._call_with_retry(model, image_b64)
        except Exception as exc:
            logger.error(f"L1 classification failed: {exc}")
            if preprocess_result is None:
                _, preprocess_result = self.preprocessor.process(image_path)
            result = await self.degradation.run_l1_cv_fallback(
                preprocess_result, str(image_path)
            )
            result["_source"] = "cv_fallback_after_error"
            return result

        parsed = response.get("parsed_json", {})
        try:
            validated = L1Output(**parsed).model_dump()
        except Exception:
            logger.warning(f"L1 output validation failed, using raw: {parsed}")
            validated = parsed

        validated["_source"] = "ollama"
        validated["_elapsed"] = response.get("elapsed_seconds", 0)
        self.cache.set(image_path, "l1", validated)
        return validated

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def _call_with_retry(self, model: str, image_b64: str) -> dict:
        return await self.ollama.chat(
            model, image_b64, L1_PROMPT, timeout=self.settings.l1_timeout
        )

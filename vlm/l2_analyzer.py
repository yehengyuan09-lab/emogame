"""L2: Qwen2.5VL-3B fine aesthetic analysis via Ollama.

Cached, with deferral to AutoDL when Ollama L2 is unavailable.

Phase 1 intentionally uses the same local model family as L1 because the
audit in ``progress/2026-06-20.md`` showed that it satisfies the JSON
contract, while ``llama3.2-vision:11b`` needs prompt and token-budget work
before it can be promoted.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger

from vlm.cache import CacheManager
from vlm.config import get_settings
from vlm.degradation import DegradationHandler, PipelineTier
from vlm.ollama_client import OllamaClient
from vlm.prompts import L2_PROMPT
from vlm.schemas import L2Output
from vlm.utils import encode_image


class L2Analyzer:
    """L2: fine aesthetic analysis (8 dimensions, 1-10 scoring)."""

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

    async def analyze(self, image_path: Path) -> dict[str, Any]:
        """Run L2 aesthetic analysis.

        Returns a dict matching ``L2Output`` schema plus ``_source`` metadata.
        When Ollama L2 is unavailable, returns ``{"_source": "deferred_to_l3"}``
        so the pipeline can produce L2 scores via AutoDL GPT-5.4-mini instead.
        """
        tier = await self.degradation.detect_tier()

        # ── Defer to AutoDL GPT-5.4-mini ──
        if tier in (
            PipelineTier.API_ONLY,
            PipelineTier.DEGRADED_API_L2,
            PipelineTier.DEGRADED_CV_L1,
        ):
            return {"_source": "deferred_to_l3"}

        # ── Cache hit ──
        cached = self.cache.get(image_path, "l2")
        if cached is not None:
            cached["_source"] = "cache"
            return cached

        # ── Ollama call ──
        image_b64 = encode_image(image_path)
        try:
            response = await self.ollama.chat(
                self.settings.l2_model,
                image_b64,
                L2_PROMPT,
                timeout=self.settings.l2_timeout,
            )
        except Exception as exc:
            logger.error(f"L2 analysis failed: {exc}")
            return {"_source": "error", "_error": str(exc)}

        parsed = response.get("parsed_json", {})
        try:
            validated = L2Output(**parsed).model_dump()
        except Exception:
            logger.warning(f"L2 output validation failed, using raw: {parsed}")
            validated = parsed

        validated["_source"] = "ollama"
        validated["_elapsed"] = response.get("elapsed_seconds", 0)
        self.cache.set(image_path, "l2", validated)
        return validated

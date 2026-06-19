"""L3: AutoDL GPT-5.4-mini semantic / cultural analysis via OpenAI-compatible API.

Supports two modes:
1. Standard L3 — semantic analysis only (design style, cultural references, etc.)
2. Combined L2+L3 — when Ollama L2 is unavailable, AutoDL handles both
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger
from openai import AsyncOpenAI

from vlm.cache import CacheManager
from vlm.config import get_settings
from vlm.degradation import DegradationHandler
from vlm.prompts import L3_SYSTEM_PROMPT, L3_USER_PROMPT_TEMPLATE
from vlm.schemas import L3Output
from vlm.utils import encode_image, parse_jsonish


class L3Semantic:
    """L3: AutoDL GPT-5.4-mini semantic understanding."""

    def __init__(
        self,
        cache: CacheManager,
        degradation: DegradationHandler,
    ) -> None:
        self.settings = get_settings()
        self.cache = cache
        self.degradation = degradation
        self._client: AsyncOpenAI | None = None

    def _get_client(self) -> AsyncOpenAI | None:
        """Return an OpenAI-compatible client pointed at AutoDL, or ``None``
        when no token is configured."""
        if self._client is not None:
            return self._client
        if not self.settings.autodl_token:
            logger.warning("No AUTODL_TOKEN set — L3 semantic analysis disabled")
            return None
        kwargs: dict[str, Any] = {
            "api_key": self.settings.autodl_token,
            "base_url": self.settings.autodl_base_url,
        }
        self._client = AsyncOpenAI(**kwargs)
        return self._client

    async def analyze(
        self,
        image_path: Path,
        l1_result: dict[str, Any],
        l2_result: dict[str, Any] | None = None,
        needs_l2_fallback: bool = False,
    ) -> dict[str, Any]:
        """Run L3 semantic analysis.

        When ``needs_l2_fallback`` is True, AutoDL produces both L2
        scores and L3 semantics in a single combined call.

        Returns an empty dict with ``_source: "no_autodl_token"`` when no
        AutoDL token is configured.
        """
        client = self._get_client()
        if client is None:
            return {"_source": "no_autodl_token"}

        # ── Cache hit ──
        cache_level = "l3_combined" if needs_l2_fallback else "l3"
        cached = self.cache.get(image_path, cache_level)
        if cached is not None:
            cached["_source"] = "cache"
            return cached

        if needs_l2_fallback:
            return await self._combined_l2_l3(image_path, l1_result)

        return await self._standard_l3(image_path, l1_result, l2_result)

    async def _standard_l3(
        self,
        image_path: Path,
        l1_result: dict[str, Any],
        l2_result: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Standard L3: semantic analysis only."""
        image_b64 = encode_image(image_path)
        client = self._get_client()
        if client is None:
            return {"_source": "no_autodl_token"}

        user_prompt = L3_USER_PROMPT_TEMPLATE.format(
            rarity_tier=l1_result.get("rarity_tier", "未知"),
            dominant_colors=", ".join(l1_result.get("dominant_colors", [])),
            scene_type=l1_result.get("scene_type", "未知"),
            effect_density=l1_result.get("effect_density", "未知"),
            model_detail=l2_result.get("model_detail", "N/A") if l2_result else "N/A",
            color_scheme=l2_result.get("color_scheme", "N/A") if l2_result else "N/A",
        )

        try:
            response = await client.chat.completions.create(
                model=self.settings.l3_model,
                messages=[
                    {"role": "system", "content": L3_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_b64}"
                                },
                            },
                        ],
                    },
                ],
                temperature=0.0,
                max_tokens=1000,
                timeout=self.settings.l3_timeout,
            )
            content = response.choices[0].message.content or ""
        except Exception as exc:
            logger.error(f"L3 analysis failed: {exc}")
            return {"_source": "error", "_error": str(exc)}

        parsed = parse_jsonish(content) or {}
        try:
            validated = L3Output(**parsed).model_dump()
        except Exception:
            logger.warning(f"L3 output validation failed, using raw: {parsed}")
            validated = parsed

        validated["_source"] = "autodl"
        self.cache.set(image_path, "l3", validated)
        return validated

    async def _combined_l2_l3(
        self,
        image_path: Path,
        l1_result: dict[str, Any],
    ) -> dict[str, Any]:
        """Combined L2+L3: delegate to ``DegradationHandler``."""
        client = self._get_client()
        if client is None:
            return {"_source": "no_autodl_token"}
        result = await self.degradation.run_combined_l2_l3_api(
            image_path, l1_result, client
        )
        # Cache the combined result
        if result.get("_source") != "error":
            self.cache.set(image_path, "l3_combined", result)
        return result

"""Degradation / fallback strategy for the VLM pipeline.

When Ollama GPU models are unavailable the pipeline degrades gracefully:
- L1 falls back to traditional-CV heuristics (edge density + colour histogram)
- L2 + L3 fall back to a combined AutoDL API call
"""

from __future__ import annotations

from enum import Enum, auto
from pathlib import Path
from typing import Any

from loguru import logger

from vlm.config import get_settings
from vlm.ollama_client import OllamaClient
from vlm.preprocess import PreprocessResult
from vlm.prompts import L2_PROMPT, L3_USER_PROMPT_TEMPLATE
from vlm.schemas import L2Output, L3Output
from vlm.utils import encode_image, parse_jsonish


class PipelineTier(Enum):
    """Available capability tiers for the VLM pipeline."""

    FULL = auto()  # All 3 tiers available via Ollama
    DEGRADED_CV_L1 = auto()  # L1 via CV heuristics, L2+L3 via AutoDL
    DEGRADED_API_L2 = auto()  # L1 via Ollama, L2+L3 via AutoDL
    API_ONLY = auto()  # Everything via AutoDL where possible
    UNAVAILABLE = auto()  # Nothing works — return empty results


class DegradationHandler:
    """Detects available models and provides fallback implementations."""

    def __init__(self, ollama_client: OllamaClient) -> None:
        self.settings = get_settings()
        self.ollama = ollama_client
        self._cached_tier: PipelineTier | None = None

    # ── tier detection ──

    async def detect_tier(self) -> PipelineTier:
        """Detect which models are available and return the capability tier.

        The result is cached in-process so we only query Ollama once.
        """
        if self._cached_tier is not None:
            return self._cached_tier

        if not self.settings.use_degradation:
            self._cached_tier = PipelineTier.FULL
            return self._cached_tier

        try:
            installed = await self.ollama.list_models()
        except Exception:
            logger.warning("Ollama unreachable — switching to API-only mode")
            self._cached_tier = PipelineTier.API_ONLY
            return self._cached_tier

        l1_ok = self.settings.l1_model in installed
        l2_ok = self.settings.l2_model in installed
        l1_fallback_ok = self.settings.l1_fallback_model in installed

        # A fallback model makes L1 "effectively ok" for tier purposes
        l1_effective = l1_ok or l1_fallback_ok

        if l1_effective and l2_ok:
            self._cached_tier = PipelineTier.FULL
        elif l1_effective and not l2_ok:
            self._cached_tier = PipelineTier.DEGRADED_API_L2
        elif not l1_effective and l2_ok:
            self._cached_tier = PipelineTier.DEGRADED_CV_L1
        else:
            # Neither L1 nor L2 available — everything via API
            self._cached_tier = PipelineTier.API_ONLY

        return self._cached_tier

    # ── L1 CV fallback ──

    async def run_l1_cv_fallback(
        self, preprocess_result: PreprocessResult, image_path: str
    ) -> dict[str, Any]:
        """Traditional-CV fallback for L1 classification.

        Uses edge density and colour histogram to produce a coarse
        rarity / scene / effect estimate.
        """
        density = preprocess_result.edge_density
        colors = preprocess_result.dominant_colors_cv

        # Heuristic rarity ↔ edge-density mapping
        if density > 0.25:
            effect_density = "extreme"
            rarity = "荣耀典藏"
        elif density > 0.18:
            effect_density = "high"
            rarity = "传说"
        elif density > 0.12:
            effect_density = "mid"
            rarity = "史诗"
        else:
            effect_density = "low"
            rarity = "勇者"

        # Scene-type heuristic based on warm/cool colour balance
        warm_count = sum(1 for c in colors if self._is_warm(c))
        if warm_count >= 4:
            scene_type = "战场"
        elif warm_count <= 1:
            scene_type = "异界"
        elif 2 <= warm_count <= 3:
            scene_type = "自然"
        else:
            scene_type = "抽象"

        return {
            "rarity_tier": rarity,
            "dominant_colors": colors[:5],
            "scene_type": scene_type,
            "character_ratio": 0.6,  # default assumption for skin wallpapers
            "effect_density": effect_density,
            "confidence": 0.5,  # lower confidence for heuristic
        }

    @staticmethod
    def _is_warm(hex_color: str) -> bool:
        """Heuristic: is a hex colour warm-toned?"""
        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)
        return r > b and r > g * 0.8

    # ── L2+L3 combined API fallback ──

    async def run_combined_l2_l3_api(
        self,
        image_path: Path,
        l1_result: dict[str, Any],
        api_client,
    ) -> dict[str, Any]:
        """When Ollama L2 is unavailable, AutoDL handles both L2 scoring
        and L3 semantics in a single API call.

        Returns a flat dict with both L2 and L3 fields merged.
        """
        image_b64 = encode_image(image_path)

        user_text = L3_USER_PROMPT_TEMPLATE.format(
            rarity_tier=l1_result.get("rarity_tier", "未知"),
            dominant_colors=", ".join(l1_result.get("dominant_colors", [])),
            scene_type=l1_result.get("scene_type", "未知"),
            effect_density=l1_result.get("effect_density", "未知"),
            model_detail="N/A",
            color_scheme="N/A",
        )

        combined_prompt = f"""{L2_PROMPT}

Additionally, after the aesthetic scoring, provide semantic analysis:

{user_text}

Output a single JSON object with BOTH the 8 L2 scoring dimensions (as top-level keys) AND the L3 semantic fields (also as top-level keys).
"""

        try:
            response = await api_client.chat.completions.create(
                model=self.settings.l3_model,
                messages=[
                    {
                        "role": "system",
                        "content": "你是游戏美术与文化分析专家。输出严格 JSON 格式。",
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": combined_prompt},
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
                max_tokens=1500,
                timeout=self.settings.l3_timeout,
            )
            content = response.choices[0].message.content or ""
        except Exception as exc:
            logger.error(f"Combined L2+L3 via AutoDL failed: {exc}")
            return {"_source": "error", "_error": str(exc)}

        parsed = parse_jsonish(content) or {}

        # Validate as much as we can
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
        l2_part = {k: v for k, v in parsed.items() if k in l2_fields}
        l3_part = {k: v for k, v in parsed.items() if k not in l2_fields}

        try:
            l2_validated = L2Output(**l2_part).model_dump() if l2_part else {}
        except Exception:
            l2_validated = l2_part
        try:
            l3_validated = L3Output(**l3_part).model_dump() if l3_part else {}
        except Exception:
            l3_validated = l3_part

        combined = {**l2_validated, **l3_validated, "_source": "autodl_combined"}
        return combined

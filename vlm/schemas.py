"""Pydantic models for VLM pipeline structured outputs.

Defines the output schema for each tier (L1 / L2 / L3), the preprocessing
result, and the merged ``VLMFeatureVector`` that the pipeline returns.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


# ── Shared literals ──

RARITY_TIER = Literal["勇者", "史诗", "传说", "无双", "荣耀典藏"]
SCENE_TYPE = Literal["战场", "主城", "异界", "抽象", "自然"]
EFFECT_DENSITY = Literal["low", "mid", "high", "extreme"]


# ── L1: Fast classification (Qwen2.5VL-3B) ──

class L1Output(BaseModel):
    """Output schema for L1 fast skin classification."""

    rarity_tier: str = ""  # RARITY_TIER
    dominant_colors: list[str] = Field(default_factory=list, min_length=3, max_length=5)
    scene_type: str = ""  # SCENE_TYPE
    character_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    effect_density: str = ""  # EFFECT_DENSITY
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    @field_validator("dominant_colors")
    @classmethod
    def _check_hex(cls, v: list[str]) -> list[str]:
        for c in v:
            if not c.startswith("#"):
                raise ValueError(f"Invalid hex color (missing #): {c}")
            if len(c) not in (4, 7):
                raise ValueError(f"Invalid hex color length: {c}")
        return v


# ── L2: Fine aesthetic analysis (Qwen2.5VL-3B) ──

class UIElements(BaseModel):
    """UI markers detected on the skin card."""

    has_limited_tag: bool = False
    has_discount_tag: bool = False
    has_gacha_tag: bool = False
    special_border: bool = False
    tier_label: str = ""


class L2Output(BaseModel):
    """Output schema for L2 fine aesthetic analysis (8 dimensions, 1-10)."""

    model_detail: float = Field(default=0.0, ge=0.0, le=10.0)
    effect_quality: float = Field(default=0.0, ge=0.0, le=10.0)
    color_scheme: float = Field(default=0.0, ge=0.0, le=10.0)
    composition: float = Field(default=0.0, ge=0.0, le=10.0)
    uniqueness: float = Field(default=0.0, ge=0.0, le=10.0)
    costume_design: float = Field(default=0.0, ge=0.0, le=10.0)
    background_quality: float = Field(default=0.0, ge=0.0, le=10.0)
    ui_elements: UIElements = Field(default_factory=UIElements)


# ── L3: Semantic understanding (AutoDL GPT-5.4-mini) ──

class SimilarSkin(BaseModel):
    """A skin that is stylistically similar to the analyzed one."""

    name: str = ""
    similarity_reason: str = ""


class L3Output(BaseModel):
    """Output schema for L3 semantic / cultural analysis."""

    design_style: str = ""
    cultural_references: list[str] = Field(default_factory=list)
    target_audience: list[str] = Field(default_factory=list)
    similar_skins: list[SimilarSkin] = Field(default_factory=list)
    differentiation: str = ""


# ── Preprocessing result ──

class PreprocessResult(BaseModel):
    """Traditional-CV features extracted during image preprocessing."""

    edge_density: float = Field(default=0.0, ge=0.0, le=1.0)
    sharpness: float = Field(default=0.0, ge=0.0)  # Laplacian variance
    dominant_colors_cv: list[str] = Field(default_factory=list)
    has_face: bool = False


# ── Merged VLM Feature Vector ──
#
# This is the **flat output contract** that ``VlmPipeline.analyze()`` returns.
# The feature-engineering layer (``SkinFeatureVector``) expects these exact
# field names at the top level when it does ``**vlm_features`` unpacking.
#
# Required fields consumed by the feature-engineering & model layers:
#   vlm_art_quality   float [0, 10]  ← L2.model_detail
#   vlm_effect_score  float [0, 10]  ← L2.effect_quality
#   vlm_color_harmony float [0, 1]   ← L2.color_scheme / 10
#   vlm_composition   float [0, 10]  ← L2.composition

class VLMFeatureVector(BaseModel):
    """Flat feature dict returned by the VLM pipeline.

    Contains the 4 required mapped fields plus all raw L1/L2/L3 outputs
    and preprocessing features for downstream consumers.
    """

    # ── Required mapped fields (consumed by feature-engineering layer) ──
    vlm_art_quality: float = Field(default=0.0, ge=0.0, le=10.0)
    vlm_effect_score: float = Field(default=0.0, ge=0.0, le=10.0)
    vlm_color_harmony: float = Field(default=0.0, ge=0.0, le=1.0)
    vlm_composition: float = Field(default=0.0, ge=0.0, le=10.0)

    # ── L1 raw fields ──
    rarity_tier: str = ""
    dominant_colors: list[str] = Field(default_factory=list)
    scene_type: str = ""
    character_ratio: float = 0.0
    effect_density: str = ""
    confidence: float = 0.0

    # ── L2 raw fields ──
    model_detail: float = 0.0
    effect_quality: float = 0.0
    color_scheme: float = 0.0
    # vlm_composition above doubles as composition
    uniqueness: float = 0.0
    costume_design: float = 0.0
    background_quality: float = 0.0
    ui_elements: dict = Field(default_factory=dict)

    # ── L3 raw fields ──
    design_style: str = ""
    cultural_references: list[str] = Field(default_factory=list)
    target_audience: list[str] = Field(default_factory=list)
    similar_skins: list[dict] = Field(default_factory=list)
    differentiation: str = ""

    # ── Preprocessing features ──
    edge_density: float = 0.0
    sharpness: float = 0.0
    dominant_colors_cv: list[str] = Field(default_factory=list)

    # ── Metadata ──
    status: str = "ok"  # "ok" | "degraded" | "partial" | "error"
    pipeline_elapsed: float = 0.0
    cache_hit_l1: bool = False
    cache_hit_l2: bool = False
    cache_hit_l3: bool = False

    def to_dict(self) -> dict:
        """Serialize to a plain dict (compatible with ``**vlm_features`` usage)."""
        return self.model_dump()

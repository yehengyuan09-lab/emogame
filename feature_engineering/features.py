"""SkinFeatureVector — canonical 33-dimensional feature vector for skin emotional premium.

Defines the validated Pydantic model consumed by the feature-engineering layer
and the downstream XGBoost / rule-engine model layer.  The 33 fields are
organised into the five emotional-premium groups defined in the architecture
docs:

    Aesthetic   (8)  w = 0.30   — VLM visual scores, official rarity, effects
    Belonging   (8)  w = 0.20   — hero popularity, IP strength, community buzz
    Showing-off (6)  w = 0.25   — in-game visibility, kill broadcast, gifting
    Collection  (7)  w = 0.15   — series, limited status, rerun history
    Surprise    (4)  w = 0.10   — acquisition method, gacha pity, drop rate

All 33 fields default to ``None`` — missing inputs are never silently imputed
to zero.  Metadata (availability, provenance, image hash, pipeline status,
validation errors) lives *outside* the 33 model dimensions.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator


# ═══════════════════════════════════════════════════════════════════════════
# Enums
# ═══════════════════════════════════════════════════════════════════════════


class AcquisitionMethod(IntEnum):
    DIRECT = 0       # 直购
    GACHA = 1        # 抽奖
    BATTLE_PASS = 2  # 战令
    SHARD = 3        # 碎片兑换
    EVENT = 4        # 活动赠送


class IPSourceType(IntEnum):
    ORIGINAL = 0     # 原创
    ANIME = 1        # 动漫联动
    FILM_TV = 2      # 影视联动
    CULTURAL_IP = 3  # 文化IP


class LimitedType(IntEnum):
    ANNIVERSARY = 0  # 周年庆
    FESTIVAL = 1     # 节日限定
    COLLAB = 2       # 联动限定
    BATTLE_PASS = 3  # 战令限定
    SEASON = 4       # 赛季限定


class VisibilityLevel(IntEnum):
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3


# ═══════════════════════════════════════════════════════════════════════════
# Group metadata (for documentation, reporting, and to_array() ordering)
# ═══════════════════════════════════════════════════════════════════════════

FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "aesthetic": (
        "vlm_art_quality",
        "vlm_effect_score",
        "vlm_color_harmony",
        "vlm_composition",
        "official_tier",
        "has_voice_pack",
        "has_custom_anim",
        "skin_age_days",
    ),
    "belonging": (
        "baidu_index_7d",
        "weibo_followers",
        "ip_source_type",
        "ip_popularity",
        "character_usage_rate",
        "character_win_rate",
        "community_post_count",
        "bilibili_video_views",
    ),
    "showing_off": (
        "lobby_display_level",
        "battle_effect_visibility",
        "kill_broadcast_type",
        "has_leaderboard_bonus",
        "is_giftable",
        "gift_popularity_rank",
    ),
    "collection": (
        "series_membership",
        "series_completion_bonus",
        "is_limited",
        "limited_type",
        "rerun_count",
        "days_since_last_rerun",
        "ownership_rate",
    ),
    "surprise": (
        "acquisition_method",
        "gacha_pity_amount",
        "drop_rate_percentile",
        "avg_spend_to_obtain",
    ),
}

# Ordered list of all 33 field names (flattened from FEATURE_GROUPS).
FEATURE_FIELD_NAMES: tuple[str, ...] = tuple(
    name
    for group in ("aesthetic", "belonging", "showing_off", "collection", "surprise")
    for name in FEATURE_GROUPS[group]
)

assert len(FEATURE_FIELD_NAMES) == 33, (
    f"Expected 33 feature fields, got {len(FEATURE_FIELD_NAMES)}"
)

# Per-field range constraints for validation.
FIELD_RANGES: dict[str, tuple[float | None, float | None]] = {
    "vlm_art_quality": (0.0, 10.0),
    "vlm_effect_score": (0.0, 10.0),
    "vlm_color_harmony": (0.0, 1.0),
    "vlm_composition": (0.0, 10.0),
    "official_tier": (0, 5),
    "has_voice_pack": (None, None),        # bool — no numeric range
    "has_custom_anim": (None, None),
    "skin_age_days": (0, None),
    "baidu_index_7d": (0, None),
    "weibo_followers": (0, None),
    "ip_source_type": (0, 3),
    "ip_popularity": (0.0, 1.0),
    "character_usage_rate": (0.0, 1.0),
    "character_win_rate": (0.0, 1.0),
    "community_post_count": (0, None),
    "bilibili_video_views": (0, None),
    "lobby_display_level": (0, 3),
    "battle_effect_visibility": (0, 3),
    "kill_broadcast_type": (0, 3),
    "has_leaderboard_bonus": (None, None),
    "is_giftable": (None, None),
    "gift_popularity_rank": (0, None),
    "series_membership": (0, None),
    "series_completion_bonus": (None, None),
    "is_limited": (None, None),
    "limited_type": (None, None),
    "rerun_count": (0, None),
    "days_since_last_rerun": (0, None),
    "ownership_rate": (0.0, 1.0),
    "acquisition_method": (0, 4),
    "gacha_pity_amount": (0.0, None),
    "drop_rate_percentile": (0.0, 1.0),
    "avg_spend_to_obtain": (0.0, None),
}


# ═══════════════════════════════════════════════════════════════════════════
# SkinFeatureVector
# ═══════════════════════════════════════════════════════════════════════════


class SkinFeatureVector(BaseModel):
    """Canonical 33-dimensional feature vector for one skin.

    All 33 model features are nullable by default.  Call ``to_array()`` to
    get a fixed-order list of values for the model layer, and
    ``availability_array()`` for the corresponding binary flags.

    Metadata fields (schema_version, image_hash, pipeline_status, provenance,
    validation_errors) live outside the 33 dimensions and are NOT included in
    ``to_array()`` output.
    """

    # ── Identity ──
    skin_key: str
    skin_id: str = ""
    hero_name: str = ""
    skin_name: str = ""

    # ── Aesthetic (8) ─────────────────────────────────────────────────────

    vlm_art_quality: float | None = Field(default=None, ge=0.0, le=10.0)
    vlm_effect_score: float | None = Field(default=None, ge=0.0, le=10.0)
    vlm_color_harmony: float | None = Field(default=None, ge=0.0, le=1.0)
    vlm_composition: float | None = Field(default=None, ge=0.0, le=10.0)
    official_tier: int | None = Field(default=None, ge=0, le=5)
    has_voice_pack: bool | None = None
    has_custom_anim: bool | None = None
    skin_age_days: int | None = Field(default=None, ge=0)

    # ── Belonging (8) ─────────────────────────────────────────────────────

    baidu_index_7d: int | None = Field(default=None, ge=0)
    weibo_followers: int | None = Field(default=None, ge=0)
    ip_source_type: int | None = Field(default=None, ge=0, le=3)
    ip_popularity: float | None = Field(default=None, ge=0.0, le=1.0)
    character_usage_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    character_win_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    community_post_count: int | None = Field(default=None, ge=0)
    bilibili_video_views: int | None = Field(default=None, ge=0)

    # ── Showing-off (6) ───────────────────────────────────────────────────

    lobby_display_level: int | None = Field(default=None, ge=0, le=3)
    battle_effect_visibility: int | None = Field(default=None, ge=0, le=3)
    kill_broadcast_type: int | None = Field(default=None, ge=0, le=3)
    has_leaderboard_bonus: bool | None = None
    is_giftable: bool | None = None
    gift_popularity_rank: int | None = Field(default=None, ge=0)

    # ── Collection (7) ────────────────────────────────────────────────────

    series_membership: int | None = Field(default=None, ge=0)
    series_completion_bonus: bool | None = None
    is_limited: bool | None = None
    limited_type: int | None = None
    rerun_count: int | None = Field(default=None, ge=0)
    days_since_last_rerun: int | None = Field(default=None, ge=0)
    ownership_rate: float | None = Field(default=None, ge=0.0, le=1.0)

    # ── Surprise (4) ──────────────────────────────────────────────────────

    acquisition_method: int | None = Field(default=None, ge=0, le=4)
    gacha_pity_amount: float | None = Field(default=None, ge=0.0)
    drop_rate_percentile: float | None = Field(default=None, ge=0.0, le=1.0)
    avg_spend_to_obtain: float | None = Field(default=None, ge=0.0)

    # ── Metadata (outside the 33 model dimensions) ────────────────────────

    schema_version: int = 2
    availability: dict[str, bool] = Field(default_factory=dict)
    provenance: dict[str, str] = Field(default_factory=dict)
    image_hash: str | None = None
    pipeline_status: str = "pending"
    validation_errors: list[str] = Field(default_factory=list)

    # Extra storage for raw VLM outputs (retained for analysis, not part of
    # the 33 model dimensions).
    vlm_raw: dict[str, Any] = Field(default_factory=dict)

    # ── Public methods ────────────────────────────────────────────────────

    def to_array(
        self, missing_policy: str = "null"
    ) -> list[float | int | None]:
        """Return 33 values in the documented fixed order.

        Args:
            missing_policy: ``"null"`` (default) leaves ``None`` values as-is.
                ``"zero"`` replaces ``None`` with ``0`` (use only at model
                training time, and only after imputation).

        Returns:
            List of 33 values, one per feature dimension.
        """
        if missing_policy not in ("null", "zero"):
            raise ValueError(
                f"missing_policy must be 'null' or 'zero', got {missing_policy!r}"
            )

        values: list[float | int | None] = []
        for name in FEATURE_FIELD_NAMES:
            v = getattr(self, name)
            if v is None and missing_policy == "zero":
                values.append(0)
            else:
                values.append(v)
        return values

    def availability_array(self) -> list[int]:
        """Return 33 binary flags (1 = present, 0 = null).

        Used by the model layer to apply availability-aware loss masking.
        """
        return [
            1 if getattr(self, name) is not None else 0
            for name in FEATURE_FIELD_NAMES
        ]

    def validate_ranges(self) -> list[str]:
        """Check all 33 fields against documented ranges.

        Returns:
            List of human-readable violation strings.  Empty list means
            all present values are within bounds.
        """
        errors: list[str] = []
        for name in FEATURE_FIELD_NAMES:
            value = getattr(self, name)
            if value is None:
                continue
            lo, hi = FIELD_RANGES.get(name, (None, None))
            if isinstance(value, bool):
                continue  # booleans have no numeric range check
            if lo is not None and value < lo:
                errors.append(
                    f"{name}={value} below minimum {lo}"
                )
            if hi is not None and value > hi:
                errors.append(
                    f"{name}={value} above maximum {hi}"
                )
            # Type-check: if the range expects int, reject float
            if lo is not None and hi is not None and isinstance(lo, int) and isinstance(hi, int):
                if isinstance(value, float) and value != int(value):
                    errors.append(
                        f"{name}={value} should be int, got float"
                    )
        return errors

    def recompute_availability(self) -> None:
        """Rebuild the ``availability`` dict from the current 33 field values."""
        self.availability = {
            name: getattr(self, name) is not None
            for name in FEATURE_FIELD_NAMES
        }

    def group_coverage(self) -> dict[str, float]:
        """Return per-group fraction of non-null features (0.0–1.0)."""
        coverage: dict[str, float] = {}
        for group, names in FEATURE_GROUPS.items():
            available = sum(
                1 for n in names if getattr(self, n) is not None
            )
            coverage[group] = available / len(names)
        return coverage

    # ── Class methods ─────────────────────────────────────────────────────

    @classmethod
    def from_dicts(
        cls,
        skin_key: str,
        skin_id: str = "",
        hero_name: str = "",
        skin_name: str = "",
        vlm_features: dict[str, Any] | None = None,
        official_features: dict[str, Any] | None = None,
        provenance: dict[str, str] | None = None,
        image_hash: str | None = None,
        pipeline_status: str = "complete",
        vlm_raw: dict[str, Any] | None = None,
    ) -> SkinFeatureVector:
        """Factory: build a vector from separate VLM and official-feature dicts.

        VLM keys are expected to use the same field names as the 4 mapped
        VLM dimensions (``vlm_art_quality``, ``vlm_effect_score``,
        ``vlm_color_harmony``, ``vlm_composition``).
        """
        vlm = vlm_features or {}
        official = official_features or {}

        # Collect values for the 33 fields from the two dicts.
        # VLM dict takes precedence over official for the 4 VLM-mapped fields.
        vlm_field_names = {
            "vlm_art_quality", "vlm_effect_score",
            "vlm_color_harmony", "vlm_composition",
        }

        field_values: dict[str, Any] = {
            "skin_key": skin_key,
            "skin_id": skin_id,
            "hero_name": hero_name,
            "skin_name": skin_name,
        }

        for name in FEATURE_FIELD_NAMES:
            if name in vlm_field_names and name in vlm:
                field_values[name] = vlm[name]
            elif name in official:
                field_values[name] = official[name]
            else:
                field_values[name] = None

        # Build provenance
        prov: dict[str, str] = {}
        if provenance:
            prov.update(provenance)
        for name in FEATURE_FIELD_NAMES:
            if name not in prov:
                if name in vlm_field_names and name in vlm:
                    prov[name] = "vlm_l2"
                elif name in official:
                    prov[name] = official.get(f"_provenance_{name}", "official")
                else:
                    prov[name] = "unavailable"

        # Build availability
        availability = {
            name: field_values.get(name) is not None
            for name in FEATURE_FIELD_NAMES
        }

        return cls(
            **field_values,
            availability=availability,
            provenance=prov,
            image_hash=image_hash,
            pipeline_status=pipeline_status,
            vlm_raw=vlm_raw or {},
        )

    @classmethod
    def field_names(cls) -> tuple[str, ...]:
        """Return the ordered tuple of 33 feature field names."""
        return FEATURE_FIELD_NAMES

    @classmethod
    def field_count(cls) -> int:
        """Return 33 — the number of model dimensions."""
        return len(FEATURE_FIELD_NAMES)


# ═══════════════════════════════════════════════════════════════════════════
# Provenance constants
# ═══════════════════════════════════════════════════════════════════════════

class Provenance:
    """Standard provenance labels for feature sources."""

    VLM_L2 = "vlm_l2"
    VLM_L3 = "vlm_l3"
    OFFICIAL = "official"
    OFFICIAL_QUALITY_MAP = "official:quality_map"
    OFFICIAL_DATE = "official:date"
    OFFICIAL_ACQUIRE = "official:acquire_text"
    OFFICIAL_INTRO = "official:intro_text"
    UNAVAILABLE = "unavailable"
    IMAGE_MISSING = "image_missing"
    AMBIGUOUS = "source:ambiguous"

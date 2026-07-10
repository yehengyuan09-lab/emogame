"""SkinFeatureVector — canonical feature vector for skin emotional premium.

This module unifies two previously-conflicting designs:

* The **33-dimensional Phase-2 model** (from the feature-engineering pipeline):
  Pydantic ``SkinFeatureVector`` with ``vlm_*`` / ``official_*`` dimensions,
  ``to_array()``, ``availability_array()``, ``validate_ranges()``,
  ``from_dicts()``, and the ``FEATURE_GROUPS`` / ``FEATURE_FIELD_NAMES`` /
  ``FIELD_RANGES`` metadata used by the model layer.
* The **sales-workbench fields** (from the evaluation / rule-engine / sales
  calibration code): identity columns (``source_key``, ``hero_id``,
  ``quality``, ``acquire_method`` …), boolean acquisition flags (``is_gacha``,
  ``is_direct_sale`` …), data-quality flags (``has_detail_record``,
  ``has_primary_asset``), and the nested ``MarketValidationSignals`` struct.

All 33 *model* dimensions default to ``None`` — missing inputs are never
silently imputed to zero.  Metadata (availability, provenance, image hash,
pipeline status, validation errors) and the sales-workbench fields live
*outside* the 33 model dimensions.

The five emotional-premium groups (see ``FEATURE_GROUPS``):

    Aesthetic   (8)  w = 0.30   — VLM visual scores, official rarity, effects
    Belonging   (8)  w = 0.20   — hero popularity, IP strength, community buzz
    Showing-off (6)  w = 0.25   — in-game visibility, kill broadcast, gifting
    Collection  (7)  w = 0.15   — series, limited status, rerun history
    Surprise    (4)  w = 0.10   — acquisition method, gacha pity, drop rate
"""

from __future__ import annotations

import hashlib
from enum import IntEnum
from typing import Any

from pydantic import BaseModel, Field


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
# MarketValidationSignals — external calibration signals (sales workbench)
# ═══════════════════════════════════════════════════════════════════════════


class MarketValidationSignals(BaseModel):
    """External signals used to research and calibrate skin evaluation.

    All fields are optional because these sources will be collected in later
    phases. Missing values lower confidence instead of being backfilled with
    arbitrary assumptions.
    """

    visual_score: float | None = None
    feel_score: float | None = None
    craftsmanship_score: float | None = None
    collection_score: float | None = None
    value_score: float | None = None
    purchase_intent_score: float | None = None
    sentiment_score: float | None = None
    discussion_count: int | None = None
    video_views: int | None = None
    marketing_volume: int | None = None
    sales_volume: int | None = None
    avg_spend_to_obtain: float | None = None
    ownership_rate: float | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "MarketValidationSignals":
        data = data or {}
        return cls(
            visual_score=_optional_float(data.get("visual_score")),
            feel_score=_optional_float(data.get("feel_score")),
            craftsmanship_score=_optional_float(data.get("craftsmanship_score")),
            collection_score=_optional_float(data.get("collection_score")),
            value_score=_optional_float(data.get("value_score")),
            purchase_intent_score=_optional_float(data.get("purchase_intent_score")),
            sentiment_score=_optional_float(data.get("sentiment_score")),
            discussion_count=_optional_int(data.get("discussion_count")),
            video_views=_optional_int(data.get("video_views")),
            marketing_volume=_optional_int(data.get("marketing_volume")),
            sales_volume=_optional_int(data.get("sales_volume")),
            avg_spend_to_obtain=_optional_float(data.get("avg_spend_to_obtain")),
            ownership_rate=_optional_float(data.get("ownership_rate")),
        )

    def present_fields(self) -> list[str]:
        return [
            name
            for name in (
                "visual_score",
                "feel_score",
                "craftsmanship_score",
                "collection_score",
                "value_score",
                "purchase_intent_score",
                "sentiment_score",
                "discussion_count",
                "video_views",
                "marketing_volume",
                "sales_volume",
                "avg_spend_to_obtain",
                "ownership_rate",
            )
            if getattr(self, name) is not None
        ]

    def coverage(self) -> float:
        return len(self.present_fields()) / 13

    def aspect_coverage(self) -> float:
        aspect_fields = (
            "visual_score",
            "feel_score",
            "craftsmanship_score",
            "collection_score",
            "value_score",
            "purchase_intent_score",
        )
        return sum(1 for name in aspect_fields if getattr(self, name) is not None) / len(aspect_fields)


# ═══════════════════════════════════════════════════════════════════════════
# SkinFeatureVector
# ═══════════════════════════════════════════════════════════════════════════


class SkinFeatureVector(BaseModel):
    """Canonical feature vector for one skin.

    Combines the 33-dimensional Phase-2 model features with the
    sales-workbench identity / acquisition / data-quality fields.

    All 33 model features are nullable by default.  Call ``to_array()`` to
    get a fixed-order list of values for the model layer, and
    ``availability_array()`` for the corresponding binary flags.

    Metadata fields (schema_version, image_hash, pipeline_status, provenance,
    validation_errors) and sales-workbench fields (source_key, hero_id,
    quality, market_signals, …) live *outside* the 33 dimensions and are NOT
    included in ``to_array()`` output.
    """

    # ── Identity ──
    skin_key: str = ""
    source_key: str = ""          # alias used by the sales workbench
    skin_id: str = ""
    hero_id: str = ""
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

    # ── Sales-workbench fields (outside the 33 model dimensions) ──────────

    quality: str = ""
    acquire_method: str = ""
    price_text: str | None = None
    quality_score: float = 0.0
    is_gacha: bool = False
    is_direct_sale: bool = False
    is_event: bool = False
    is_battle_pass: bool = False
    is_shard_exchange: bool = False
    has_detail_record: bool = False
    has_primary_asset: bool = False
    hero_skin_count: int = 0
    market_signals: MarketValidationSignals = Field(
        default_factory=MarketValidationSignals
    )

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

    def to_dict(self) -> dict[str, Any]:
        """Return the sales-workbench dict shape (used by ``app.py`` / ``RuleEngine``)."""
        return {
            "source_key": self.source_key or self.skin_key,
            "hero_id": self.hero_id,
            "hero_name": self.hero_name,
            "skin_name": self.skin_name,
            "skin_id": self.skin_id,
            "quality": self.quality,
            "online_date": self.provenance.get("online_date", ""),
            "acquire_method": self.acquire_method,
            "price_text": self.price_text,
            "official_tier": self.official_tier if self.official_tier is not None else 0,
            "quality_score": self.quality_score,
            "is_limited": bool(self.is_limited),
            "is_gacha": self.is_gacha,
            "is_direct_sale": self.is_direct_sale,
            "is_event": self.is_event,
            "is_battle_pass": self.is_battle_pass,
            "is_shard_exchange": self.is_shard_exchange,
            "has_detail_record": self.has_detail_record,
            "has_primary_asset": self.has_primary_asset,
            "skin_age_days": self.skin_age_days,
            "hero_skin_count": self.hero_skin_count,
            "market_signals": self.market_signals.model_dump(),
        }

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
            "source_key": skin_key,
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


# ═══════════════════════════════════════════════════════════════════════════
# Optional-value helpers
# ═══════════════════════════════════════════════════════════════════════════


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None

"""Official metadata extractor — deterministic mappings from crawler fields to features.

All functions are pure (no I/O) so they are easy to unit-test.  Each
extractor returns a ``(value, provenance)`` tuple.  When the source text is
ambiguous or unparseable the value is ``None`` and the provenance explains
why (e.g. ``"source:ambiguous_quality"``).
"""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any

from loguru import logger

from feature_engineering.features import Provenance


# ═══════════════════════════════════════════════════════════════════════════
# Quality → official_tier
# ═══════════════════════════════════════════════════════════════════════════

# Ordered from most-specific to least-specific to handle substring matches.
QUALITY_TO_TIER: dict[str, int] = {
    "荣耀典藏": 5,
    "无双限定": 4,
    "无双": 4,
    "珍品传说": 4,
    "传说限定": 3,
    "传说": 3,
    "史诗限定": 2,
    "史诗": 2,
    "勇者限定": 1,
    "勇者": 1,
    "伴生": 0,
}

# Patterns that indicate a limited skin.
LIMITED_PATTERNS: list[str] = [
    "限定", "限时", "周年庆", "赛季", "情人节", "春节", "五五",
    "圣诞", "万圣", "夏日", "战令",
]

# Pattern for limited type classification.
LIMITED_TYPE_PATTERNS: dict[str, int] = {
    "周年庆": 0,
    "节日限定|情人节|春节|圣诞|万圣|夏日|五五": 1,
    "联动": 2,
    "战令": 3,
    "赛季": 4,
}

# Acquisition method keyword → enum value.
ACQUISITION_KEYWORDS: dict[str, int] = {
    "直购": 0,
    "商城直售": 0,
    "限时直售": 0,
    "限时上架": 0,
    "秒杀": 0,
    "点券": 0,
    "抽奖": 1,
    "夺宝": 1,
    "水晶": 1,
    "荣耀水晶": 1,
    "战令": 2,
    "进阶": 2,
    "荣耀战令": 2,
    "碎片": 3,
    "兑换": 3,
    "皮肤碎片": 3,
    "活动": 4,
    "赠送": 4,
    "累充": 4,
    "任务": 4,
    "免费": 4,
}

# Order for acquisition-method keyword matching (longest-first to avoid
# partial matches like "战令" eating "荣耀战令").
ACQUISITION_ORDERED = sorted(
    ACQUISITION_KEYWORDS.keys(), key=len, reverse=True
)

# Patterns for boolean feature extraction from intro text.
VOICE_PACK_KEYWORDS: list[str] = ["语音", "配音", "声优", "CV", "语音包"]
CUSTOM_ANIM_KEYWORDS: list[str] = [
    "动作", "动画", "特效", "回城特效", "出场动画",
    "待机动作", "移动动作", "展示动作", "随机动作",
]
GIFTABLE_KEYWORDS: list[str] = ["赠送", "可赠送", "索要", "礼物"]


# ═══════════════════════════════════════════════════════════════════════════
# Extractor functions
# ═══════════════════════════════════════════════════════════════════════════


def extract_official_tier(quality: str) -> tuple[int | None, str]:
    """Map a quality string to ``official_tier`` (0–5).

    Returns:
        ``(tier, provenance)`` where *tier* is ``None`` for unknown inputs.
    """
    if not quality or not quality.strip():
        return None, Provenance.AMBIGUOUS
    q = quality.strip()
    if q in QUALITY_TO_TIER:
        return QUALITY_TO_TIER[q], Provenance.OFFICIAL_QUALITY_MAP
    # Try substring match (longest first)
    for key, tier in sorted(QUALITY_TO_TIER.items(), key=lambda x: -len(x[0])):
        if key in q:
            return tier, Provenance.OFFICIAL_QUALITY_MAP
    logger.debug(f"Unknown quality string: {q!r}")
    return None, Provenance.AMBIGUOUS


def extract_skin_age(online_date: str) -> tuple[int | None, str]:
    """Compute days since release from an ``online_date`` string.

    Expected format: ``YYYYMMDD`` (e.g. ``20240101``).
    Returns ``(days, provenance)`` or ``(None, …)`` on parse failure.
    """
    if not online_date or not online_date.strip():
        return None, Provenance.AMBIGUOUS
    raw = online_date.strip()
    for fmt in ("%Y%m%d", "%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d"):
        try:
            dt = datetime.strptime(raw, fmt).date()
            return (date.today() - dt).days, Provenance.OFFICIAL_DATE
        except ValueError:
            continue
    logger.debug(f"Unparseable online_date: {raw!r}")
    return None, Provenance.AMBIGUOUS


def extract_acquisition(
    acquire_method: str,
    price_text: str | None = None,
    quality: str = "",
) -> dict[str, Any]:
    """Parse acquisition text into structured features.

    Returns a dict with keys:
        * ``acquisition_method`` — int [0,4] or None
        * ``is_giftable`` — bool or None
        * ``is_limited`` — bool or None
        * ``limited_type`` — int or None
        * ``avg_spend_to_obtain`` — float or None
    """
    result: dict[str, Any] = {
        "acquisition_method": None,
        "is_giftable": None,
        "is_limited": None,
        "limited_type": None,
        "avg_spend_to_obtain": None,
    }

    if not acquire_method or not acquire_method.strip():
        text = ""
    else:
        text = acquire_method.strip()

    # ── acquisition_method ──
    if text:
        for keyword in ACQUISITION_ORDERED:
            if keyword in text:
                result["acquisition_method"] = ACQUISITION_KEYWORDS[keyword]
                break

    # ── is_limited (from quality + acquire text) ──
    combined = (quality + " " + text).strip()
    for pattern in LIMITED_PATTERNS:
        if pattern in combined:
            result["is_limited"] = True
            break
    if result["is_limited"] is None:
        result["is_limited"] = False  # Explicitly mark as known non-limited

    # ── limited_type ──
    if result["is_limited"]:
        for pattern_str, lt_val in LIMITED_TYPE_PATTERNS.items():
            for sub_pat in pattern_str.split("|"):
                if sub_pat in combined:
                    result["limited_type"] = lt_val
                    break
            if result["limited_type"] is not None:
                break

    # ── is_giftable ──
    if text:
        for kw in GIFTABLE_KEYWORDS:
            if kw in text:
                result["is_giftable"] = True
                break

    # ── avg_spend_to_obtain ──
    if price_text:
        result["avg_spend_to_obtain"] = _parse_price(price_text)

    return result


def extract_from_intro(intro: str) -> dict[str, Any]:
    """Extract boolean features from the skin-intro / description text.

    Returns a dict with keys:
        * ``has_voice_pack`` — bool or None
        * ``has_custom_anim`` — bool or None
        * ``series_membership`` — int or None (estimated from keywords)
    """
    result: dict[str, Any] = {
        "has_voice_pack": None,
        "has_custom_anim": None,
        "series_membership": None,
    }

    if not intro or not intro.strip():
        return result

    text = intro.strip()

    for kw in VOICE_PACK_KEYWORDS:
        if kw in text:
            result["has_voice_pack"] = True
            break

    for kw in CUSTOM_ANIM_KEYWORDS:
        if kw in text:
            result["has_custom_anim"] = True
            break

    # Series membership — look for "系列" mentions
    series_match = re.search(r"(.{1,6})系列", text)
    if series_match:
        # Placeholder: we mark it as 1 (part of a series) but don't know the
        # exact count from intro text alone.
        result["series_membership"] = 1

    return result


def determine_ip_source_type(
    hero_name: str, skin_name: str, intro: str = ""
) -> tuple[int | None, str]:
    """Detect IP source type from hero/skin name and intro.

    Returns ``(ip_source_type, provenance)`` — None when uncertain.
    """
    combined = f"{hero_name} {skin_name} {intro}"
    # Collaboration patterns
    collab_patterns: dict[str, int] = {
        "联动": 2,
        "合作": 2,
        "动漫": 1,
        "动画": 1,
        "电影": 2,
        "影视": 2,
        "文化": 3,
        "传统": 3,
        "文创": 3,
        "非遗": 3,
        "圣斗士": 1,
        "SNK": 1,
        "故宫": 3,
        "敦煌": 3,
    }
    for keyword, iptype in sorted(collab_patterns.items(), key=lambda x: -len(x[0])):
        if keyword in combined:
            return iptype, Provenance.OFFICIAL_INTRO
    return 0, Provenance.OFFICIAL  # default: original


# ═══════════════════════════════════════════════════════════════════════════
# Main extractor
# ═══════════════════════════════════════════════════════════════════════════


class OfficialFeatureMapper:
    """Deterministic mapping from crawler fields to official feature dimensions.

    All methods are pure functions — no I/O, no side effects.  Each one
    returns ``(value, provenance)`` or a dict of such pairs.

    Usage::

        mapper = OfficialFeatureMapper()
        features, provenance = mapper.extract_all(
            quality="传说",
            online_date="20240101",
            acquire_method="1688点券 限时上架",
            price_text="1688点券",
            intro="全新传说皮肤，水墨风格，含独立语音包",
            hero_name="李白",
            skin_name="诗剑行",
        )
    """

    def extract_all(
        self,
        quality: str = "",
        online_date: str = "",
        acquire_method: str = "",
        price_text: str | None = None,
        intro: str = "",
        hero_name: str = "",
        skin_name: str = "",
    ) -> tuple[dict[str, Any], dict[str, str]]:
        """Run all extractors and return ``(features_dict, provenance_dict)``.

        Feature keys match the ``SkinFeatureVector`` field names.
        """
        features: dict[str, Any] = {}
        prov: dict[str, str] = {}

        # ── official_tier ──
        tier, tier_prov = extract_official_tier(quality)
        features["official_tier"] = tier
        prov["official_tier"] = tier_prov

        # ── skin_age_days ──
        age, age_prov = extract_skin_age(online_date)
        features["skin_age_days"] = age
        prov["skin_age_days"] = age_prov

        # ── acquisition group ──
        acq = extract_acquisition(acquire_method, price_text, quality)
        for key in (
            "acquisition_method", "is_limited", "limited_type",
            "is_giftable", "avg_spend_to_obtain",
        ):
            features[key] = acq.get(key)
            prov[key] = Provenance.OFFICIAL_ACQUIRE if acq.get(key) is not None else Provenance.AMBIGUOUS

        # ── intro-based features ──
        intro_f = extract_from_intro(intro)
        for key in ("has_voice_pack", "has_custom_anim", "series_membership"):
            features[key] = intro_f.get(key)
            prov[key] = Provenance.OFFICIAL_INTRO if intro_f.get(key) is not None else Provenance.UNAVAILABLE

        # ── ip_source_type ──
        ip_type, ip_prov = determine_ip_source_type(hero_name, skin_name, intro)
        features["ip_source_type"] = ip_type
        prov["ip_source_type"] = ip_prov

        # ── series_completion_bonus ──
        # We can't determine this from official text alone; leave as None.
        features["series_completion_bonus"] = None
        prov["series_completion_bonus"] = Provenance.UNAVAILABLE

        # ── Fields that are always unavailable from official data ──
        always_unavailable = [
            "baidu_index_7d",
            "weibo_followers",
            "ip_popularity",
            "character_usage_rate",
            "character_win_rate",
            "community_post_count",
            "bilibili_video_views",
            "lobby_display_level",
            "battle_effect_visibility",
            "kill_broadcast_type",
            "has_leaderboard_bonus",
            "gift_popularity_rank",
            "rerun_count",
            "days_since_last_rerun",
            "ownership_rate",
            "gacha_pity_amount",
            "drop_rate_percentile",
        ]
        for name in always_unavailable:
            if name not in features:
                features[name] = None
                prov[name] = Provenance.UNAVAILABLE

        return features, prov


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════


def _parse_price(price_text: str) -> float | None:
    """Extract a numeric price from text like ``"1688点券"`` or ``"888"``.

    Returns the float value, or ``None`` on failure.
    """
    if not price_text:
        return None
    # Find the first number (possibly with a decimal point)
    m = re.search(r"(\d+(?:\.\d+)?)", price_text.strip())
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None

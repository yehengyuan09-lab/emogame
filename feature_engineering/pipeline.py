"""Build MVP evaluation features from the local skin repository."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from data.skin_repository import SkinRepository
from feature_engineering.features import MarketValidationSignals, SkinFeatureVector


QUALITY_TIER_MAP = {
    "": 0,
    "伴生": 0,
    "勇者": 1,
    "勇者限定": 2,
    "史诗": 2,
    "史诗限定": 3,
    "珍品史诗": 3,
    "传说": 4,
    "传说限定": 4,
    "珍品传说": 4,
    "无双": 5,
    "无双限定": 5,
    "荣耀典藏": 5,
}


class FeatureBuilder:
    """Convert repository rows plus optional market signals into features."""

    def __init__(self, repo: SkinRepository, reference_date: date | None = None):
        self.repo = repo
        self.reference_date = reference_date or date.today()

    def build(
        self,
        source_key: str,
        market_signals: MarketValidationSignals | dict[str, Any] | None = None,
    ) -> SkinFeatureVector:
        skin = self.repo.get_skin(source_key)
        if not skin:
            raise ValueError(f"skin not found: {source_key}")

        if isinstance(market_signals, MarketValidationSignals):
            signals = market_signals
        else:
            signals = MarketValidationSignals.from_dict(market_signals)

        hero_skin_count = len(self.repo.list_skins(hero_id=skin["hero_id"], limit=None))
        quality = skin.get("quality") or ""
        acquire_method = skin.get("acquire_method") or ""
        official_tier = quality_to_tier(quality)

        return SkinFeatureVector(
            source_key=skin["source_key"],
            hero_id=skin["hero_id"],
            hero_name=skin["hero_name"],
            skin_name=skin["skin_name"],
            skin_id=skin.get("skin_id") or "",
            quality=quality,
            online_date=skin.get("online_date") or "",
            acquire_method=acquire_method,
            price_text=skin.get("price_text"),
            official_tier=official_tier,
            quality_score=official_tier / 5,
            is_limited=contains_any(quality + acquire_method, ["限定", "限时", "返场"]),
            is_gacha=contains_any(acquire_method + quality, ["抽奖", "祈愿", "夺宝", "荣耀水晶", "无双"]),
            is_direct_sale=contains_any(acquire_method, ["商城直售", "直售"]),
            is_event=contains_any(acquire_method, ["活动", "福利", "任务"]),
            is_battle_pass=contains_any(acquire_method, ["战令", "礼册"]),
            is_shard_exchange=contains_any(acquire_method, ["碎片", "碎片商城", "碎片商店"]),
            has_detail_record=bool(skin.get("has_detail_record")),
            has_primary_asset=bool(skin.get("primary_asset_url")),
            skin_age_days=parse_age_days(skin.get("online_date"), self.reference_date),
            hero_skin_count=hero_skin_count,
            market_signals=signals,
        )


def quality_to_tier(quality: str) -> int:
    if quality in QUALITY_TIER_MAP:
        return QUALITY_TIER_MAP[quality]
    if "荣耀典藏" in quality or "无双" in quality:
        return 5
    if "传说" in quality:
        return 4
    if "史诗" in quality:
        return 3 if "限定" in quality else 2
    if "勇者" in quality:
        return 2 if "限定" in quality else 1
    return 0


def contains_any(value: str, needles: list[str]) -> bool:
    return any(needle in value for needle in needles)


def parse_age_days(value: str | None, reference_date: date) -> int | None:
    if not value:
        return None
    for fmt in ("%Y-%m-%d", "%Y%m%d"):
        try:
            parsed = datetime.strptime(value, fmt).date()
            return max(0, (reference_date - parsed).days)
        except ValueError:
            continue
    return None

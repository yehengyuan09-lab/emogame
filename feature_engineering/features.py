"""Feature data structures for emotional premium evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class MarketValidationSignals:
    """External signals used to validate and calibrate the MVP score.

    All fields are optional because these sources will be collected in later
    phases. Missing values lower confidence instead of blocking evaluation.
    """

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
        return len(self.present_fields()) / 7


@dataclass(slots=True)
class SkinFeatureVector:
    source_key: str
    hero_id: str
    hero_name: str
    skin_name: str
    skin_id: str
    quality: str
    online_date: str
    acquire_method: str
    price_text: str | None
    official_tier: int
    quality_score: float
    is_limited: bool
    is_gacha: bool
    is_direct_sale: bool
    is_event: bool
    is_battle_pass: bool
    is_shard_exchange: bool
    has_detail_record: bool
    has_primary_asset: bool
    skin_age_days: int | None
    hero_skin_count: int
    market_signals: MarketValidationSignals = field(default_factory=MarketValidationSignals)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_key": self.source_key,
            "hero_id": self.hero_id,
            "hero_name": self.hero_name,
            "skin_name": self.skin_name,
            "skin_id": self.skin_id,
            "quality": self.quality,
            "online_date": self.online_date,
            "acquire_method": self.acquire_method,
            "price_text": self.price_text,
            "official_tier": self.official_tier,
            "quality_score": self.quality_score,
            "is_limited": self.is_limited,
            "is_gacha": self.is_gacha,
            "is_direct_sale": self.is_direct_sale,
            "is_event": self.is_event,
            "is_battle_pass": self.is_battle_pass,
            "is_shard_exchange": self.is_shard_exchange,
            "has_detail_record": self.has_detail_record,
            "has_primary_asset": self.has_primary_asset,
            "skin_age_days": self.skin_age_days,
            "hero_skin_count": self.hero_skin_count,
            "market_signals": {
                name: getattr(self.market_signals, name)
                for name in (
                    "sentiment_score",
                    "discussion_count",
                    "video_views",
                    "marketing_volume",
                    "sales_volume",
                    "avg_spend_to_obtain",
                    "ownership_rate",
                )
            },
        }


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

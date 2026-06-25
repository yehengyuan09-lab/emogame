"""Rule-based MVP emotional premium evaluator."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from feature_engineering.features import SkinFeatureVector


DIMENSION_WEIGHTS = {
    "aesthetic": 0.30,
    "belonging": 0.20,
    "showing_off": 0.25,
    "collection": 0.15,
    "surprise": 0.10,
}


@dataclass(slots=True)
class EvaluationResult:
    source_key: str
    hero_name: str
    skin_name: str
    total_premium: int
    sub_scores: dict[str, int]
    confidence: float
    validation_status: str
    market_signal_score: int | None
    signal_coverage: float
    warnings: list[str]
    evidence: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_key": self.source_key,
            "hero_name": self.hero_name,
            "skin_name": self.skin_name,
            "total_premium": self.total_premium,
            "sub_scores": self.sub_scores,
            "confidence": self.confidence,
            "validation_status": self.validation_status,
            "market_signal_score": self.market_signal_score,
            "signal_coverage": self.signal_coverage,
            "warnings": self.warnings,
            "evidence": self.evidence,
        }


class RuleEngine:
    """Cold-start scorer until labeled market data is available."""

    def evaluate(self, features: SkinFeatureVector) -> EvaluationResult:
        sub_scores = {
            "aesthetic": round_score(self._aesthetic(features)),
            "belonging": round_score(self._belonging(features)),
            "showing_off": round_score(self._showing_off(features)),
            "collection": round_score(self._collection(features)),
            "surprise": round_score(self._surprise(features)),
        }
        total = round_score(
            sum(score * DIMENSION_WEIGHTS[name] for name, score in sub_scores.items())
        )
        market_signal_score = self._market_signal_score(features)
        signal_coverage = features.market_signals.coverage()
        confidence = self._confidence(features, signal_coverage)
        warnings = self._warnings(features, signal_coverage)

        return EvaluationResult(
            source_key=features.source_key,
            hero_name=features.hero_name,
            skin_name=features.skin_name,
            total_premium=total,
            sub_scores=sub_scores,
            confidence=confidence,
            validation_status="market_validated" if signal_coverage >= 0.5 else "needs_market_validation",
            market_signal_score=market_signal_score,
            signal_coverage=round(signal_coverage, 2),
            warnings=warnings,
            evidence=self._evidence(features),
        )

    def _aesthetic(self, f: SkinFeatureVector) -> float:
        recent_bonus = 0.5 if f.skin_age_days is not None and f.skin_age_days <= 180 else 0.25
        return 100 * (
            0.45 * f.quality_score
            + 0.20 * bool_score(f.has_primary_asset)
            + 0.20 * bool_score(f.has_detail_record)
            + 0.15 * recent_bonus
        )

    def _belonging(self, f: SkinFeatureVector) -> float:
        s = f.market_signals
        sentiment = clamp01(s.sentiment_score if s.sentiment_score is not None else 0.5)
        discussion = log_score(s.discussion_count, 10000)
        views = log_score(s.video_views, 5_000_000)
        marketing = log_score(s.marketing_volume, 10000)
        sales = log_score(s.sales_volume, 500_000)
        return 100 * (0.30 * sentiment + 0.20 * discussion + 0.20 * views + 0.15 * marketing + 0.15 * sales)

    def _showing_off(self, f: SkinFeatureVector) -> float:
        rare_method = f.is_gacha or f.official_tier >= 4
        return 100 * (
            0.45 * f.quality_score
            + 0.20 * bool_score(f.is_limited)
            + 0.20 * bool_score(rare_method)
            + 0.15 * bool_score(f.has_primary_asset)
        )

    def _collection(self, f: SkinFeatureVector) -> float:
        ownership = f.market_signals.ownership_rate
        scarcity = 0.5 if ownership is None else 1 - clamp01(ownership)
        series_score = clamp01(f.hero_skin_count / 12)
        rare_quality = bool_score(f.official_tier >= 4)
        return 100 * (
            0.35 * bool_score(f.is_limited)
            + 0.25 * rare_quality
            + 0.20 * scarcity
            + 0.20 * series_score
        )

    def _surprise(self, f: SkinFeatureVector) -> float:
        acquire_score = acquisition_score(f)
        spend_score = log_score(f.market_signals.avg_spend_to_obtain, 2000)
        promo_bonus = bool_score("秒杀" in f.acquire_method or "福利" in f.acquire_method)
        return 100 * (0.40 * acquire_score + 0.25 * bool_score(f.is_gacha) + 0.20 * spend_score + 0.15 * promo_bonus)

    def _market_signal_score(self, f: SkinFeatureVector) -> int | None:
        if not f.market_signals.present_fields():
            return None
        return round_score(self._belonging(f))

    def _confidence(self, f: SkinFeatureVector, signal_coverage: float) -> float:
        base = 0.35
        source_quality = 0.15 * bool_score(f.has_detail_record) + 0.10 * bool_score(f.has_primary_asset)
        market_quality = 0.40 * signal_coverage
        return round(clamp01(base + source_quality + market_quality), 2)

    def _warnings(self, f: SkinFeatureVector, signal_coverage: float) -> list[str]:
        warnings: list[str] = []
        if not f.has_detail_record:
            warnings.append("missing_official_detail")
        if not f.has_primary_asset:
            warnings.append("missing_primary_asset")
        if signal_coverage < 0.5:
            warnings.append("market_validation_incomplete")
        if f.market_signals.sentiment_score is None:
            warnings.append("missing_sentiment_score")
        if f.market_signals.sales_volume is None and f.market_signals.marketing_volume is None:
            warnings.append("missing_sales_or_marketing_volume")
        return warnings

    def _evidence(self, f: SkinFeatureVector) -> dict[str, Any]:
        return {
            "quality": f.quality,
            "official_tier": f.official_tier,
            "acquire_method": f.acquire_method,
            "is_limited": f.is_limited,
            "is_gacha": f.is_gacha,
            "has_detail_record": f.has_detail_record,
            "has_primary_asset": f.has_primary_asset,
            "skin_age_days": f.skin_age_days,
            "hero_skin_count": f.hero_skin_count,
            "market_signal_fields": f.market_signals.present_fields(),
        }


def acquisition_score(f: SkinFeatureVector) -> float:
    if f.is_gacha:
        return 0.90
    if f.is_battle_pass:
        return 0.65
    if f.is_event:
        return 0.60
    if f.is_shard_exchange:
        return 0.50
    if f.is_direct_sale:
        return 0.30
    return 0.20


def bool_score(value: bool) -> float:
    return 1.0 if value else 0.0


def clamp01(value: float) -> float:
    return min(max(value, 0.0), 1.0)


def log_score(value: int | float | None, max_value: int | float) -> float:
    if value is None or value <= 0:
        return 0.0
    return clamp01(math.log(value + 1) / math.log(max_value + 1))


def round_score(value: float) -> int:
    return min(max(round(value), 0), 100)

"""Evidence-first MVP skin evaluator.

This module does not claim to produce a final valuation from official metadata.
It separates:

- official priors: weak clues from crawler data, used for context only;
- market/aspect evidence: public opinion and marketing/sales signals that can
  validate a skin evaluation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from feature_engineering.features import SkinFeatureVector


ASPECT_WEIGHTS = {
    "visual_appeal": 0.18,
    "in_game_feel": 0.18,
    "craftsmanship_quality": 0.18,
    "collection_value": 0.16,
    "value_for_money": 0.15,
    "purchase_intent": 0.10,
    "market_heat": 0.05,
}


@dataclass(slots=True)
class EvaluationResult:
    source_key: str
    hero_name: str
    skin_name: str
    evaluation_score: int | None
    official_prior_score: int
    aspect_scores: dict[str, int | None]
    confidence: float
    validation_status: str
    evidence_coverage: float
    warnings: list[str]
    evidence: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_key": self.source_key,
            "hero_name": self.hero_name,
            "skin_name": self.skin_name,
            "evaluation_score": self.evaluation_score,
            "official_prior_score": self.official_prior_score,
            "aspect_scores": self.aspect_scores,
            "confidence": self.confidence,
            "validation_status": self.validation_status,
            "evidence_coverage": self.evidence_coverage,
            "warnings": self.warnings,
            "evidence": self.evidence,
        }


class RuleEngine:
    """Aspect research evaluator with explicit evidence coverage."""

    def evaluate(self, features: SkinFeatureVector) -> EvaluationResult:
        aspect_scores = self._aspect_scores(features)
        evidence_coverage = self._evidence_coverage(features, aspect_scores)
        official_prior_score = self._official_prior(features)
        evaluation_score = self._weighted_score(aspect_scores)
        confidence = self._confidence(features, evidence_coverage)
        validation_status = (
            "evidence_validated" if evidence_coverage >= 0.5 else "insufficient_market_evidence"
        )

        return EvaluationResult(
            source_key=features.source_key,
            hero_name=features.hero_name,
            skin_name=features.skin_name,
            evaluation_score=evaluation_score,
            official_prior_score=official_prior_score,
            aspect_scores=aspect_scores,
            confidence=confidence,
            validation_status=validation_status,
            evidence_coverage=round(evidence_coverage, 2),
            warnings=self._warnings(features, evidence_coverage),
            evidence=self._evidence(features),
        )

    def _aspect_scores(self, f: SkinFeatureVector) -> dict[str, int | None]:
        s = f.market_signals
        return {
            "visual_appeal": optional_score(s.visual_score),
            "in_game_feel": optional_score(s.feel_score),
            "craftsmanship_quality": optional_score(s.craftsmanship_score),
            "collection_value": optional_score(s.collection_score),
            "value_for_money": optional_score(s.value_score),
            "purchase_intent": optional_score(s.purchase_intent_score),
            "market_heat": self._market_heat_score(f),
        }

    def _market_heat_score(self, f: SkinFeatureVector) -> int | None:
        s = f.market_signals
        heat_parts = [
            log_score(s.discussion_count, 10000),
            log_score(s.video_views, 5_000_000),
            log_score(s.marketing_volume, 10000),
            log_score(s.sales_volume, 500_000),
        ]
        available = [part for part in heat_parts if part is not None]
        if not available:
            return None
        return round_score(sum(available) / len(available) * 100)

    def _weighted_score(self, aspect_scores: dict[str, int | None]) -> int | None:
        weighted_sum = 0.0
        used_weight = 0.0
        for name, score in aspect_scores.items():
            if score is None:
                continue
            weight = ASPECT_WEIGHTS[name]
            weighted_sum += score * weight
            used_weight += weight
        if used_weight <= 0:
            return None
        return round_score(weighted_sum / used_weight)

    def _official_prior(self, f: SkinFeatureVector) -> int:
        scarcity = 30 * bool_score(f.is_limited or f.is_gacha)
        quality = 45 * f.quality_score
        data_quality = 15 * bool_score(f.has_detail_record) + 10 * bool_score(f.has_primary_asset)
        return round_score(scarcity + quality + data_quality)

    def _evidence_coverage(self, f: SkinFeatureVector, aspect_scores: dict[str, int | None]) -> float:
        aspect_coverage = sum(1 for score in aspect_scores.values() if score is not None) / len(aspect_scores)
        data_quality = 0.5 * bool_score(f.has_detail_record) + 0.5 * bool_score(f.has_primary_asset)
        return clamp01(0.8 * aspect_coverage + 0.2 * data_quality)

    def _confidence(self, f: SkinFeatureVector, evidence_coverage: float) -> float:
        source_quality = 0.5 * bool_score(f.has_detail_record) + 0.5 * bool_score(f.has_primary_asset)
        return round(clamp01(0.15 + 0.70 * evidence_coverage + 0.15 * source_quality), 2)

    def _warnings(self, f: SkinFeatureVector, evidence_coverage: float) -> list[str]:
        warnings: list[str] = []
        if not f.has_detail_record:
            warnings.append("missing_official_detail")
        if not f.has_primary_asset:
            warnings.append("missing_primary_asset")
        if evidence_coverage < 0.5:
            warnings.append("insufficient_public_opinion_evidence")
        missing_aspects = []
        for name, value in {
            "visual_score": f.market_signals.visual_score,
            "feel_score": f.market_signals.feel_score,
            "craftsmanship_score": f.market_signals.craftsmanship_score,
            "collection_score": f.market_signals.collection_score,
            "value_score": f.market_signals.value_score,
            "purchase_intent_score": f.market_signals.purchase_intent_score,
        }.items():
            if value is None:
                missing_aspects.append(name)
        if missing_aspects:
            warnings.append("missing_aspect_scores:" + ",".join(missing_aspects))
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


def optional_score(value: float | None) -> int | None:
    if value is None:
        return None
    if 0 <= value <= 1:
        return round_score(value * 100)
    return round_score(value)


def bool_score(value: bool) -> float:
    return 1.0 if value else 0.0


def clamp01(value: float) -> float:
    return min(max(value, 0.0), 1.0)


def log_score(value: int | float | None, max_value: int | float) -> float | None:
    if value is None:
        return None
    if value <= 0:
        return 0.0
    return clamp01(math.log(value + 1) / math.log(max_value + 1))


def round_score(value: float) -> int:
    return min(max(round(value), 0), 100)

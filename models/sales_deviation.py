"""Compare evaluation scores with public sales evidence.

The comparison is intentionally evidence-aware. Exact volumes, estimated
volumes, public hot-sales ranks, and upper/lower-bound claims are different
inputs and should not be collapsed into a fake precise sales number.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from feature_engineering.features import MarketValidationSignals, SkinFeatureVector
from models.rule_engine import EvaluationResult, log_score, round_score


SALES_VOLUME_KEYS = (
    "sales_volume",
    "estimated_sales_volume",
    "sales_volume_estimate",
    "units_sold",
    "sales",
)
SALES_UPPER_BOUND_KEYS = ("sales_volume_upper_bound", "sales_upper_bound")
SALES_LOWER_BOUND_KEYS = ("sales_volume_lower_bound", "sales_lower_bound")
SALES_RANK_KEYS = ("sales_rank", "hot_sales_rank", "rank")
DEFAULT_MAX_SALES_VOLUME = 10_000_000
DEFAULT_RANK_SIZE = 10
DEFAULT_RANK_FLOOR_SCORE = 60.0
ALIGNED_GAP = 8


@dataclass(slots=True)
class SalesCandidate:
    sales_score: int
    basis: str
    confidence: float
    source_title: str | None = None
    source_url: str | None = None
    volume: int | None = None
    volume_relation: str | None = None
    rank: int | None = None
    rank_size: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "sales_score": self.sales_score,
            "basis": self.basis,
            "confidence": self.confidence,
            "source_title": self.source_title,
            "source_url": self.source_url,
            "volume": self.volume,
            "volume_relation": self.volume_relation,
            "rank": self.rank,
            "rank_size": self.rank_size,
        }


def compare_score_to_sales(
    features: SkinFeatureVector,
    evaluation: EvaluationResult,
    evidence_items: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return the score-vs-sales gap for one evaluated skin."""
    score = evaluation.evaluation_score
    score_basis = "evaluation_score"
    warnings: list[str] = []
    if score is None:
        score = evaluation.official_prior_score
        score_basis = "official_prior_score"
        warnings.append("using_official_prior_score")

    candidate = select_sales_candidate(features, evidence_items or [])
    if candidate is None:
        return {
            "source_key": features.source_key,
            "hero_name": features.hero_name,
            "skin_name": features.skin_name,
            "score": score,
            "score_basis": score_basis,
            "sales_score": None,
            "sales_basis": None,
            "gap": None,
            "absolute_gap": None,
            "gap_direction": "insufficient_sales_data",
            "confidence": 0.0,
            "warnings": warnings + ["missing_sales_evidence"],
            "sales_evidence": None,
            "evidence_count": len(evidence_items or []),
            "interpretation": "No public sales evidence was available for comparison.",
        }

    gap = int(score) - candidate.sales_score
    direction = gap_direction(gap)
    if candidate.basis in {"sales_rank", "rank_proxy"}:
        warnings.append("sales_score_uses_rank_proxy")
    if candidate.volume_relation in {"upper_bound", "lower_bound"}:
        warnings.append(f"sales_volume_is_{candidate.volume_relation}")
    if candidate.volume_relation == "estimated":
        warnings.append("sales_volume_is_estimated")

    return {
        "source_key": features.source_key,
        "hero_name": features.hero_name,
        "skin_name": features.skin_name,
        "score": int(score),
        "score_basis": score_basis,
        "sales_score": candidate.sales_score,
        "sales_basis": candidate.basis,
        "gap": gap,
        "absolute_gap": abs(gap),
        "gap_direction": direction,
        "confidence": round(candidate.confidence, 2),
        "warnings": warnings,
        "sales_evidence": candidate.to_dict(),
        "evidence_count": len(evidence_items or []),
        "interpretation": interpretation(direction, candidate),
    }


def sales_blind_signals(signals: MarketValidationSignals) -> MarketValidationSignals:
    """Remove direct sales/ownership fields before computing a comparison score."""
    return MarketValidationSignals(
        visual_score=signals.visual_score,
        feel_score=signals.feel_score,
        craftsmanship_score=signals.craftsmanship_score,
        collection_score=signals.collection_score,
        value_score=signals.value_score,
        purchase_intent_score=signals.purchase_intent_score,
        sentiment_score=signals.sentiment_score,
        discussion_count=signals.discussion_count,
        video_views=signals.video_views,
        marketing_volume=signals.marketing_volume,
    )


def select_sales_candidate(
    features: SkinFeatureVector,
    evidence_items: list[dict[str, Any]],
) -> SalesCandidate | None:
    candidates: list[tuple[int, SalesCandidate]] = []

    for item in evidence_items:
        metrics = item.get("metrics") or {}
        source_title = item.get("title")
        source_url = item.get("url")
        confidence = metric_float(metrics, "source_confidence")

        exact_volume = first_metric(metrics, SALES_VOLUME_KEYS)
        if exact_volume is not None:
            relation = str(metrics.get("sales_volume_relation") or "exact").lower()
            basis = "estimated_sales_volume" if relation == "estimated" else "sales_volume"
            default_confidence = 0.55 if relation == "estimated" else 0.85
            candidates.append(
                (
                    40 if relation == "estimated" else 50,
                    volume_candidate(
                        exact_volume,
                        basis=basis,
                        relation=relation,
                        confidence=confidence or default_confidence,
                        source_title=source_title,
                        source_url=source_url,
                    ),
                )
            )

        lower_bound = first_metric(metrics, SALES_LOWER_BOUND_KEYS)
        if lower_bound is not None:
            candidates.append(
                (
                    30,
                    volume_candidate(
                        lower_bound,
                        basis="sales_volume_lower_bound",
                        relation="lower_bound",
                        confidence=confidence or 0.45,
                        source_title=source_title,
                        source_url=source_url,
                    ),
                )
            )

        upper_bound = first_metric(metrics, SALES_UPPER_BOUND_KEYS)
        if upper_bound is not None:
            candidates.append(
                (
                    20,
                    volume_candidate(
                        upper_bound,
                        basis="sales_volume_upper_bound",
                        relation="upper_bound",
                        confidence=confidence or 0.35,
                        source_title=source_title,
                        source_url=source_url,
                    ),
                )
            )

        rank = first_metric(metrics, SALES_RANK_KEYS)
        if rank is not None:
            rank_size = int(first_metric(metrics, ("rank_size", "sales_rank_size")) or DEFAULT_RANK_SIZE)
            rank_floor = float(
                first_metric(metrics, ("rank_floor_score",)) or DEFAULT_RANK_FLOOR_SCORE
            )
            candidates.append(
                (
                    35,
                    rank_candidate(
                        rank,
                        rank_size=rank_size,
                        rank_floor_score=rank_floor,
                        confidence=confidence or 0.60,
                        source_title=source_title,
                        source_url=source_url,
                    ),
                )
            )

    if features.market_signals.sales_volume is not None:
        candidates.append(
            (
                25,
                volume_candidate(
                    features.market_signals.sales_volume,
                    basis="aggregate_sales_volume",
                    relation="aggregate",
                    confidence=0.70,
                ),
            )
        )

    if not candidates:
        return None

    candidates.sort(key=lambda pair: (pair[0], pair[1].confidence, pair[1].sales_score), reverse=True)
    return candidates[0][1]


def volume_candidate(
    volume: int | float,
    *,
    basis: str,
    relation: str,
    confidence: float,
    source_title: str | None = None,
    source_url: str | None = None,
) -> SalesCandidate:
    value = max(0, int(volume))
    return SalesCandidate(
        sales_score=round_score(log_score(value, DEFAULT_MAX_SALES_VOLUME) * 100),
        basis=basis,
        confidence=confidence,
        source_title=source_title,
        source_url=source_url,
        volume=value,
        volume_relation=relation,
    )


def rank_candidate(
    rank: int | float,
    *,
    rank_size: int,
    rank_floor_score: float,
    confidence: float,
    source_title: str | None = None,
    source_url: str | None = None,
) -> SalesCandidate:
    safe_size = max(1, int(rank_size))
    safe_rank = min(max(1, int(rank)), safe_size)
    if safe_size == 1:
        score = 100
    else:
        percentile_score = (safe_size - safe_rank) / (safe_size - 1)
        score = rank_floor_score + (100 - rank_floor_score) * percentile_score
    return SalesCandidate(
        sales_score=round_score(score),
        basis="sales_rank",
        confidence=confidence,
        source_title=source_title,
        source_url=source_url,
        rank=safe_rank,
        rank_size=safe_size,
    )


def gap_direction(gap: int) -> str:
    if abs(gap) <= ALIGNED_GAP:
        return "aligned"
    if gap > 0:
        return "score_above_sales"
    return "sales_above_score"


def interpretation(direction: str, candidate: SalesCandidate) -> str:
    if direction == "aligned":
        return "Evaluation score is close to the available sales signal."
    if direction == "score_above_sales":
        return "Evaluation score is higher than the sales signal; conversion or pricing may be the blocker."
    if direction == "sales_above_score":
        return "Sales signal is stronger than the evaluation score; hero/IP/channel demand may be underweighted."
    return f"Compared against {candidate.basis}."


def first_metric(metrics: dict[str, Any], names: tuple[str, ...]) -> float | None:
    for name in names:
        value = metric_float(metrics, name)
        if value is not None:
            return value
    return None


def metric_float(metrics: dict[str, Any], name: str) -> float | None:
    if name not in metrics:
        return None
    return coerce_number(metrics.get(name))


def coerce_number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        text = value.strip().lower().replace(",", "")
        multiplier = 1
        if "万" in text or text.endswith("w"):
            multiplier = 10_000
        match = re.search(r"-?\d+(?:\.\d+)?", text)
        if not match:
            return None
        try:
            return float(match.group(0)) * multiplier
        except ValueError:
            return None
    return None

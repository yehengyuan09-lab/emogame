"""ML-style calibration for score-vs-sales gaps.

The calibrator keeps the original evaluation score intact and produces a
separate sales-aligned score. Inputs stay sales-blind: direct sales volume,
ownership rate, and spend-to-obtain are not used as model features.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from data.market_signal_repository import MarketSignalRepository
from data.skin_repository import SkinRepository
from feature_engineering.features import SkinFeatureVector
from feature_engineering.pipeline import FeatureBuilder
from models.rule_engine import EvaluationResult, RuleEngine
from models.sales_deviation import compare_score_to_sales, gap_direction, sales_blind_signals


DEFAULT_FEATURE_NAMES = (
    "base_score",
    "official_prior_score",
    "confidence",
    "evidence_coverage",
    "official_tier",
    "quality_score",
    "hero_skin_count",
    "skin_age_years",
    "is_limited",
    "is_gacha",
    "is_direct_sale",
    "is_event",
    "is_battle_pass",
    "is_shard_exchange",
    "has_detail_record",
    "has_primary_asset",
    "aspect_coverage",
    "visual_appeal",
    "in_game_feel",
    "craftsmanship_quality",
    "collection_value",
    "value_for_money",
    "purchase_intent",
    "market_heat",
)


@dataclass(slots=True)
class CalibrationSample:
    source_key: str
    hero_name: str
    skin_name: str
    features: dict[str, float]
    base_score: int
    target_sales_score: int
    sales_basis: str
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_key": self.source_key,
            "hero_name": self.hero_name,
            "skin_name": self.skin_name,
            "features": self.features,
            "base_score": self.base_score,
            "target_sales_score": self.target_sales_score,
            "sales_basis": self.sales_basis,
            "confidence": self.confidence,
        }


@dataclass(slots=True)
class RbfSalesCalibrator:
    feature_names: list[str]
    mean: list[float]
    scale: list[float]
    train_vectors: list[list[float]]
    dual_coef: list[float]
    gamma: float
    regularization: float

    def predict(self, features: dict[str, float]) -> int:
        vector = normalize_vector(vectorize_features(features, self.feature_names), self.mean, self.scale)
        train = np.asarray(self.train_vectors, dtype=float)
        diff = train - np.asarray(vector, dtype=float)
        kernel = np.exp(-self.gamma * np.sum(diff * diff, axis=1))
        value = float(kernel @ np.asarray(self.dual_coef, dtype=float))
        return clamp_score(value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_type": "rbf_kernel_ridge",
            "feature_names": self.feature_names,
            "mean": self.mean,
            "scale": self.scale,
            "train_vectors": self.train_vectors,
            "dual_coef": self.dual_coef,
            "gamma": self.gamma,
            "regularization": self.regularization,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RbfSalesCalibrator":
        return cls(
            feature_names=list(payload["feature_names"]),
            mean=[float(v) for v in payload["mean"]],
            scale=[float(v) for v in payload["scale"]],
            train_vectors=[[float(v) for v in row] for row in payload["train_vectors"]],
            dual_coef=[float(v) for v in payload["dual_coef"]],
            gamma=float(payload["gamma"]),
            regularization=float(payload["regularization"]),
        )


def collect_calibration_samples(
    db_path: str | Path,
    *,
    limit: int | None = None,
) -> list[CalibrationSample]:
    repo = SkinRepository(db_path)
    market_repo = MarketSignalRepository(db_path)
    source_keys = market_repo.list_source_keys_with_sales_evidence()
    if limit is not None:
        source_keys = source_keys[: max(0, limit)]

    samples: list[CalibrationSample] = []
    builder = FeatureBuilder(repo)
    engine = RuleEngine()
    for source_key in source_keys:
        signals = market_repo.get_signals(source_key)
        sales_features = builder.build(source_key, signals)
        score_features = builder.build(source_key, sales_blind_signals(signals))
        evaluation = engine.evaluate(score_features)
        evidence = market_repo.list_evidence(source_key)
        gap = compare_score_to_sales(sales_features, evaluation, evidence)
        if gap["sales_score"] is None:
            continue
        samples.append(
            CalibrationSample(
                source_key=source_key,
                hero_name=score_features.hero_name,
                skin_name=score_features.skin_name,
                features=calibration_features(score_features, evaluation),
                base_score=int(gap["score"]),
                target_sales_score=int(gap["sales_score"]),
                sales_basis=str(gap["sales_basis"]),
                confidence=float(gap["confidence"]),
            )
        )
    return samples


def calibration_features(features: SkinFeatureVector, evaluation: EvaluationResult) -> dict[str, float]:
    score = evaluation.evaluation_score
    base_score = float(score if score is not None else evaluation.official_prior_score)
    aspect_scores = evaluation.aspect_scores

    return {
        "base_score": base_score,
        "official_prior_score": float(evaluation.official_prior_score),
        "confidence": 100.0 * evaluation.confidence,
        "evidence_coverage": 100.0 * evaluation.evidence_coverage,
        "official_tier": float(features.official_tier),
        "quality_score": 100.0 * features.quality_score,
        "hero_skin_count": float(features.hero_skin_count),
        "skin_age_years": float((features.skin_age_days or 0) / 365.0),
        "is_limited": float(features.is_limited),
        "is_gacha": float(features.is_gacha),
        "is_direct_sale": float(features.is_direct_sale),
        "is_event": float(features.is_event),
        "is_battle_pass": float(features.is_battle_pass),
        "is_shard_exchange": float(features.is_shard_exchange),
        "has_detail_record": float(features.has_detail_record),
        "has_primary_asset": float(features.has_primary_asset),
        "aspect_coverage": 100.0 * features.market_signals.aspect_coverage(),
        "visual_appeal": aspect_value(aspect_scores.get("visual_appeal")),
        "in_game_feel": aspect_value(aspect_scores.get("in_game_feel")),
        "craftsmanship_quality": aspect_value(aspect_scores.get("craftsmanship_quality")),
        "collection_value": aspect_value(aspect_scores.get("collection_value")),
        "value_for_money": aspect_value(aspect_scores.get("value_for_money")),
        "purchase_intent": aspect_value(aspect_scores.get("purchase_intent")),
        "market_heat": aspect_value(aspect_scores.get("market_heat")),
    }


def fit_until_gap_probability(
    samples: list[CalibrationSample],
    *,
    gap_threshold: int = 10,
    target_probability: float = 0.10,
    alpha: float = 0.10,
    max_iterations: int = 500,
) -> dict[str, Any]:
    if not samples:
        raise ValueError("no calibration samples")

    baseline_predictions = [sample.base_score for sample in samples]
    targets = [sample.target_sales_score for sample in samples]
    baseline_stats = gap_statistics(baseline_predictions, targets, gap_threshold, target_probability, alpha)

    gammas = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
    regularizations = [1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1]
    best: tuple[tuple[float, float, float, float], RbfSalesCalibrator, list[int], dict[str, Any]] | None = None
    iterations = 0

    for gamma in gammas:
        for regularization in regularizations:
            iterations += 1
            model = fit_rbf(samples, gamma=gamma, regularization=regularization)
            predictions = [model.predict(sample.features) for sample in samples]
            stats = gap_statistics(predictions, targets, gap_threshold, target_probability, alpha)
            objective = (
                float(stats["exceedances"]),
                float(stats["mae"]),
                float(stats["max_abs_gap"]),
                regularization,
            )
            if best is None or objective < best[0]:
                best = (objective, model, predictions, stats)
            if stats["passed"]:
                best = (objective, model, predictions, stats)
                break
            if iterations >= max_iterations:
                break
        if best and best[3]["passed"]:
            break
        if iterations >= max_iterations:
            break

    assert best is not None
    _, model, predictions, calibrated_stats = best
    loo_stats = leave_one_out_stats(
        samples,
        gamma=model.gamma,
        regularization=model.regularization,
        gap_threshold=gap_threshold,
        target_probability=target_probability,
        alpha=alpha,
    )

    return {
        "model": model,
        "model_config": {
            "model_type": "rbf_kernel_ridge",
            "gamma": model.gamma,
            "regularization": model.regularization,
            "feature_names": model.feature_names,
        },
        "iterations": iterations,
        "minimum_samples_for_zero_failures": minimum_samples_for_zero_failures(target_probability, alpha),
        "baseline": baseline_stats,
        "calibrated": calibrated_stats,
        "leave_one_out": loo_stats,
        "samples": [
            sample_report(sample, prediction, gap_threshold)
            for sample, prediction in zip(samples, predictions, strict=True)
        ],
    }


def fit_rbf(
    samples: list[CalibrationSample],
    *,
    gamma: float,
    regularization: float,
) -> RbfSalesCalibrator:
    feature_names = list(DEFAULT_FEATURE_NAMES)
    matrix = np.asarray([vectorize_features(sample.features, feature_names) for sample in samples], dtype=float)
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale[scale < 1e-9] = 1.0
    normalized = (matrix - mean) / scale
    diff = normalized[:, None, :] - normalized[None, :, :]
    kernel = np.exp(-gamma * np.sum(diff * diff, axis=2))
    targets = np.asarray([sample.target_sales_score for sample in samples], dtype=float)
    system = kernel + regularization * np.eye(len(samples))
    try:
        dual = np.linalg.solve(system, targets)
    except np.linalg.LinAlgError:
        dual = np.linalg.lstsq(system, targets, rcond=None)[0]
    return RbfSalesCalibrator(
        feature_names=feature_names,
        mean=mean.tolist(),
        scale=scale.tolist(),
        train_vectors=normalized.tolist(),
        dual_coef=dual.tolist(),
        gamma=gamma,
        regularization=regularization,
    )


def leave_one_out_stats(
    samples: list[CalibrationSample],
    *,
    gamma: float,
    regularization: float,
    gap_threshold: int,
    target_probability: float,
    alpha: float,
) -> dict[str, Any]:
    if len(samples) < 3:
        return {"available": False, "reason": "need_at_least_3_samples"}

    predictions: list[int] = []
    targets: list[int] = []
    for index, held_out in enumerate(samples):
        train = samples[:index] + samples[index + 1 :]
        model = fit_rbf(train, gamma=gamma, regularization=regularization)
        predictions.append(model.predict(held_out.features))
        targets.append(held_out.target_sales_score)
    stats = gap_statistics(predictions, targets, gap_threshold, target_probability, alpha)
    stats["available"] = True
    return stats


def gap_statistics(
    predictions: list[int],
    targets: list[int],
    gap_threshold: int,
    target_probability: float,
    alpha: float,
) -> dict[str, Any]:
    gaps = [int(prediction) - int(target) for prediction, target in zip(predictions, targets, strict=True)]
    abs_gaps = [abs(gap) for gap in gaps]
    exceedances = sum(1 for value in abs_gaps if value > gap_threshold)
    n = len(gaps)
    p_value = binomial_lower_tail(exceedances, n, target_probability)
    return {
        "n": n,
        "gap_threshold": gap_threshold,
        "target_probability": target_probability,
        "alpha": alpha,
        "exceedances": exceedances,
        "empirical_probability": round(exceedances / n, 4) if n else None,
        "binomial_p_value_at_target_probability": round(p_value, 6),
        "passed": bool(n and exceedances / n <= target_probability and p_value <= alpha),
        "mae": round(sum(abs_gaps) / n, 3) if n else None,
        "max_abs_gap": max(abs_gaps) if abs_gaps else None,
    }


def binomial_lower_tail(k: int, n: int, p: float) -> float:
    if n <= 0:
        return 1.0
    return sum(math.comb(n, i) * (p**i) * ((1 - p) ** (n - i)) for i in range(k + 1))


def minimum_samples_for_zero_failures(target_probability: float, alpha: float) -> int:
    if not 0 < target_probability < 1:
        raise ValueError("target_probability must be between 0 and 1")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    return math.ceil(math.log(alpha) / math.log(1 - target_probability))


def vectorize_features(features: dict[str, float], feature_names: list[str]) -> list[float]:
    return [float(features.get(name, 0.0)) for name in feature_names]


def normalize_vector(vector: list[float], mean: list[float], scale: list[float]) -> list[float]:
    return [(value - center) / spread for value, center, spread in zip(vector, mean, scale, strict=True)]


def sample_report(sample: CalibrationSample, prediction: int, gap_threshold: int) -> dict[str, Any]:
    gap = prediction - sample.target_sales_score
    return {
        "source_key": sample.source_key,
        "hero_name": sample.hero_name,
        "skin_name": sample.skin_name,
        "base_score": sample.base_score,
        "calibrated_score": prediction,
        "sales_score": sample.target_sales_score,
        "base_gap": sample.base_score - sample.target_sales_score,
        "calibrated_gap": gap,
        "gap_direction": gap_direction(gap),
        "exceeds_threshold": abs(gap) > gap_threshold,
        "sales_basis": sample.sales_basis,
        "confidence": sample.confidence,
    }


def aspect_value(value: int | None) -> float:
    return -1.0 if value is None else float(value)


def clamp_score(value: float) -> int:
    return min(max(round(value), 0), 100)

"""Sales-oriented action report for one evaluated skin."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from feature_engineering.features import SkinFeatureVector
from models.rule_engine import EvaluationResult


@dataclass(slots=True)
class SalesActionReport:
    source_key: str
    hero_name: str
    skin_name: str
    decision: str
    decision_reason: str
    sales_readiness: int
    purchase_drivers: list[str] = field(default_factory=list)
    conversion_blockers: list[str] = field(default_factory=list)
    recommended_actions: list[dict[str, str]] = field(default_factory=list)
    pricing_guidance: dict[str, Any] = field(default_factory=dict)
    evidence_gaps: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_key": self.source_key,
            "hero_name": self.hero_name,
            "skin_name": self.skin_name,
            "decision": self.decision,
            "decision_reason": self.decision_reason,
            "sales_readiness": self.sales_readiness,
            "purchase_drivers": self.purchase_drivers,
            "conversion_blockers": self.conversion_blockers,
            "recommended_actions": self.recommended_actions,
            "pricing_guidance": self.pricing_guidance,
            "evidence_gaps": self.evidence_gaps,
        }


class SalesAdvisor:
    """Convert evidence-first evaluation output into sales actions."""

    def advise(self, features: SkinFeatureVector, evaluation: EvaluationResult) -> SalesActionReport:
        readiness = self._sales_readiness(features, evaluation)
        decision, reason = self._decision(features, evaluation, readiness)
        drivers = self._purchase_drivers(features, evaluation)
        blockers = self._conversion_blockers(features, evaluation)
        gaps = self._evidence_gaps(features, evaluation)
        actions = self._recommended_actions(features, evaluation, drivers, blockers, gaps)

        return SalesActionReport(
            source_key=features.source_key,
            hero_name=features.hero_name,
            skin_name=features.skin_name,
            decision=decision,
            decision_reason=reason,
            sales_readiness=readiness,
            purchase_drivers=drivers,
            conversion_blockers=blockers,
            recommended_actions=actions,
            pricing_guidance=self._pricing_guidance(features, evaluation),
            evidence_gaps=gaps,
        )

    def _sales_readiness(self, f: SkinFeatureVector, e: EvaluationResult) -> int:
        score = e.evaluation_score if e.evaluation_score is not None else e.official_prior_score * 0.35
        score += e.confidence * 25
        s = f.market_signals
        if s.purchase_intent_score is not None:
            score += normalized(s.purchase_intent_score) * 15
        if s.value_score is not None and normalized(s.value_score) < 0.45:
            score -= 10
        if e.evidence_coverage < 0.5:
            score -= 20
        return clamp_score(score)

    def _decision(
        self,
        f: SkinFeatureVector,
        e: EvaluationResult,
        readiness: int,
    ) -> tuple[str, str]:
        s = f.market_signals
        if e.validation_status != "evidence_validated":
            return (
                "collect_more_evidence",
                "外部舆论、购买意愿或传播证据不足，不能作为放量销售依据。",
            )
        if s.value_score is not None and normalized(s.value_score) < 0.45:
            return (
                "fix_price_or_bundle",
                "用户价值感偏弱，直接放量可能带来价格争议，需要先处理价格/权益感知。",
            )
        if readiness >= 80 and normalized(s.purchase_intent_score) >= 0.7:
            return (
                "scale_marketing",
                "购买意愿、舆论证据和销售准备度都足够，可以进入更强投放或预热。",
            )
        if readiness >= 65:
            return (
                "controlled_launch",
                "证据基本支持上线，但仍需控制节奏并持续监测阻力点。",
            )
        return (
            "hold_and_research",
            "当前证据虽可评估，但销售准备度不足，需要补齐转化证据或优化卖点。",
        )

    def _purchase_drivers(self, f: SkinFeatureVector, e: EvaluationResult) -> list[str]:
        s = f.market_signals
        drivers: list[str] = []
        if high(s.visual_score):
            drivers.append("观感/原画/特效反馈较强，可作为首屏视觉卖点。")
        if high(s.feel_score):
            drivers.append("手感或局内反馈较好，适合用实战测评内容促进转化。")
        if high(s.craftsmanship_score):
            drivers.append("品质感被市场认可，可强化传说/史诗/限定等品质锚点。")
        if high(s.collection_score) or f.is_limited or f.is_gacha:
            drivers.append("收藏与稀缺属性可用，但需要避免过度制造焦虑。")
        if high(s.purchase_intent_score):
            drivers.append("评论或视频证据中存在明确购买意愿。")
        if e.aspect_scores.get("market_heat") is not None and e.aspect_scores["market_heat"] >= 70:
            drivers.append("传播热度较高，适合承接预约、抽奖、福利或首周转化。")
        return drivers or ["暂未发现足够稳定的购买驱动力。"]

    def _conversion_blockers(self, f: SkinFeatureVector, e: EvaluationResult) -> list[str]:
        s = f.market_signals
        blockers: list[str] = []
        if low(s.value_score):
            blockers.append("性价比/价格接受度偏弱，可能压低付费转化。")
        if low(s.purchase_intent_score):
            blockers.append("购买意愿偏弱，热度可能停留在围观而非成交。")
        if low(s.visual_score):
            blockers.append("观感反馈偏弱，首屏素材和视觉卖点需要重做验证。")
        if low(s.feel_score):
            blockers.append("手感或局内体验反馈偏弱，容易影响核心玩家口碑。")
        if e.evidence_coverage < 0.5:
            blockers.append("证据覆盖不足，当前结论不适合指导大额投放。")
        if f.market_signals.sales_volume is None:
            blockers.append("缺少真实销量或拥有率信号，无法校准评分和转化关系。")
        return blockers

    def _recommended_actions(
        self,
        f: SkinFeatureVector,
        e: EvaluationResult,
        drivers: list[str],
        blockers: list[str],
        gaps: list[str],
    ) -> list[dict[str, str]]:
        actions: list[dict[str, str]] = []
        if e.validation_status != "evidence_validated":
            actions.append(action("补证据", "先采集 B站搜索、微博评论、销量/拥有率，再输出销售判断。"))
        if any("价格" in item or "性价比" in item for item in blockers):
            actions.append(action("处理价格阻力", "测试首周折扣、礼包绑定或福利返利文案，降低用户的贵感。"))
        if any("观感" in item for item in drivers):
            actions.append(action("放大视觉卖点", "短视频首帧和商店首屏突出原画、特效、建模差异。"))
        if any("手感" in item for item in drivers):
            actions.append(action("补实战内容", "优先投放技能连招、局内手感和音效对比类测评。"))
        if any("收藏" in item or "稀缺" in item for item in drivers):
            actions.append(action("控制稀缺叙事", "强调档期和纪念价值，同时避免过度透支限定信任。"))
        if "missing_sales_volume" in gaps:
            actions.append(action("接销量验证", "上线后按日回填销量、拥有率或支付转化，用来校准评分和真实销售。"))
        if not actions:
            actions.append(action("小流量验证", "用低预算素材测试点击、评论和购买意愿，再决定是否放量。"))
        return actions

    def _pricing_guidance(self, f: SkinFeatureVector, e: EvaluationResult) -> dict[str, Any]:
        s = f.market_signals
        posture = "hold"
        rationale = "证据不足以建议改变官方价格。"
        if e.validation_status != "evidence_validated":
            posture = "do_not_change_price"
            rationale = "缺少市场验证，先不要用当前评分做定价决策。"
        elif s.value_score is not None and normalized(s.value_score) < 0.45:
            posture = "discount_or_bundle"
            rationale = "价值感偏弱，应先用权益包、折扣或福利降低价格阻力。"
        elif high(s.purchase_intent_score) and high(s.collection_score):
            posture = "protect_premium"
            rationale = "购买意愿与收藏价值较强，可维持高品质锚点，但仍需监控负面价格反馈。"

        return {
            "posture": posture,
            "rationale": rationale,
            "official_price_text": f.price_text,
            "official_tier": f.official_tier,
        }

    def _evidence_gaps(self, f: SkinFeatureVector, e: EvaluationResult) -> list[str]:
        gaps: list[str] = []
        s = f.market_signals
        if s.purchase_intent_score is None:
            gaps.append("missing_purchase_intent")
        if s.value_score is None:
            gaps.append("missing_value_score")
        if s.feel_score is None:
            gaps.append("missing_feel_score")
        if s.sales_volume is None:
            gaps.append("missing_sales_volume")
        if s.ownership_rate is None:
            gaps.append("missing_ownership_rate")
        if e.evidence_coverage < 0.5:
            gaps.append("low_evidence_coverage")
        return gaps


def action(title: str, detail: str) -> dict[str, str]:
    return {"action": title, "detail": detail}


def normalized(value: float | None) -> float:
    if value is None:
        return 0.0
    return value / 100 if value > 1 else value


def high(value: float | None, threshold: float = 0.7) -> bool:
    return value is not None and normalized(value) >= threshold


def low(value: float | None, threshold: float = 0.45) -> bool:
    return value is not None and normalized(value) < threshold


def clamp_score(value: float) -> int:
    return min(max(round(value), 0), 100)

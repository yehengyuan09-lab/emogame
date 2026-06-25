# 07 — 商业价值分析模块

## 定价引擎

将情绪溢价指数映射到具体人民币定价区间。

```python
@dataclass
class PricingRecommendation:
    suggested_price_min: float   # 建议最低价 (¥)
    suggested_price_max: float   # 建议最高价 (¥)
    confidence: float            # 置信度 (0-1)
    price_per_emotion_point: float
    comparable_skins: list[dict]


class PricingEngine:
    # 王者荣耀皮肤价格锚点
    PRICE_ANCHORS = {
        "MOBA": {
            0:   (0, 0),        # 活动赠送
            20:  (6, 28),       # 伴生皮
            40:  (28, 48),      # 勇者
            60:  (48, 88),      # 史诗
            80:  (88, 178),     # 传说
            95:  (178, 288),    # 无双/荣耀典藏
        }
    }

    def recommend_price(self, premium_score: float, official_tier: int,
                        game_genre: str = "MOBA") -> PricingRecommendation:
        anchors = self.PRICE_ANCHORS.get(game_genre, self.PRICE_ANCHORS["MOBA"])
        anchor_points = sorted(anchors.keys())

        # 最近锚点插值
        closest_idx = min(range(len(anchor_points)),
                          key=lambda i: abs(anchor_points[i] - premium_score))
        base_min, base_max = anchors[anchor_points[closest_idx]]

        # 溢价加成 (premium_score 越高，加成越大)
        price_scale = 1.0 + 0.5 * (premium_score / 100)

        return PricingRecommendation(
            suggested_price_min=round(base_min * price_scale),
            suggested_price_max=round(base_max * price_scale),
            confidence=0.85 if official_tier >= 3 else 0.70,
            price_per_emotion_point=round(base_max * price_scale / premium_score, 1),
            comparable_skins=self._find_comparables(premium_score, official_tier),
        )
```

### 定价映射逻辑

| 情绪溢价 | 锚点区间 | 典型皮肤类型 | 建议价格 (¥) |
|----------|----------|-------------|-------------|
| 0-20 | 伴生/活动 | 原皮、活动赠送 | 0-28 |
| 20-40 | 勇者 | 换色皮、基础皮 | 28-48 |
| 40-60 | 史诗 | 主流皮肤 (含特效) | 48-88 |
| 60-80 | 传说 | 高品质皮肤 (全特效) | 88-178 |
| 80-100 | 无双/典藏 | 天花板皮肤 (抽奖获取) | 178-288 |

## 风险分析器

```python
class RiskAnalyzer:
    """三类核心风险检测"""

    def analyze(self, features: dict, premium: dict,
                pricing: PricingRecommendation) -> list[dict]:
        risks = []

        # 风险 1: 定价偏离情绪价值
        official_price = features.get("official_price", 0)
        if official_price > pricing.suggested_price_max * 1.3:
            risks.append({
                "level": "high",
                "type": "overpricing",
                "message": f"官方定价 ¥{official_price} 远超情绪价值上限 ¥{pricing.suggested_price_max}",
                "action": "建议增加特效层级或降低首发价格"
            })
        elif official_price < pricing.suggested_price_min * 0.7:
            risks.append({
                "level": "medium",
                "type": "underpricing",
                "message": f"定价 ¥{official_price} 低于情绪价值下限 ¥{pricing.suggested_price_min}",
                "action": "可考虑小幅提价或增加捆绑销售方案"
            })

        # 风险 2: 限定皮肤返场过频 → 稀缺性稀释
        if features.get("is_limited") and features.get("rerun_count", 0) >= 3:
            risks.append({
                "level": "high",
                "type": "rerun_dilution",
                "message": f"限定皮肤已返场 {features['rerun_count']} 次，稀缺性被严重稀释",
                "action": "建议停止返场或转为常驻皮肤"
            })

        # 风险 3: 审美短板 (高溢价但低审美的 "炒作型" 皮肤)
        if premium["sub_scores"]["aesthetic"] < 40 and premium["total_premium"] > 60:
            risks.append({
                "level": "medium",
                "type": "aesthetic_deficit",
                "message": "溢价主要来自稀缺性和IP热度，审美质量不足",
                "action": "建议在下次皮肤迭代中提升模型和特效质量"
            })

        return risks
```

## 竞品对比分析

```python
class CompetitorAnalyzer:
    async def find_competitors(self, skin_features: dict, game_genre: str,
                               top_k: int = 5) -> list[dict]:
        # 1. 从特征存储检索同稀有度、同价位段皮肤
        similar = await self.feature_store.query_similar(
            skin_features,
            filters={
                "game_genre": game_genre,
                "tier_range": (skin_features["official_tier"] - 1,
                               skin_features["official_tier"] + 1),
            },
            top_k=top_k * 2,
        )

        # 2. 余弦相似度排序
        scored = [{**s, "similarity": self._cosine_similarity(
            self._to_vector(skin_features), self._to_vector(s)
        )} for s in similar]

        return sorted(scored, key=lambda x: -x["similarity"])[:top_k]

    def generate_comparison_report(self, target: dict,
                                   competitors: list[dict]) -> str:
        lines = []
        for c in competitors:
            diff = target["total_premium"] - c["total_premium"]
            lines.append(
                f"- {c['game_name']} {c['hero_name']}-{c['skin_name']}: "
                f"情绪溢价 {c['total_premium']} "
                f"({'↑' if diff > 0 else '↓'}{abs(diff)} vs 当前皮肤)"
            )
        return "\n".join(lines)
```

## 案例库

```python
CASE_LIBRARY = [
    {
        "name": "孙悟空-至尊宝",
        "game": "王者荣耀", "genre": "MOBA",
        "total_premium": 85,
        "sub_scores": {
            "aesthetic": 78, "belonging": 92,
            "showing_off": 85, "collection": 90, "surprise": 75
        },
        "official_tier": 3,
        "price": 88.8,
        "resale_premium": 120,  # 二级市场溢价 120%
        "key_insight": "大话西游IP联动 + 限定返场稀缺性 = 极高情绪溢价",
    },
    {
        "name": "小乔-天鹅之梦",
        "game": "王者荣耀", "genre": "MOBA",
        "total_premium": 92,
        "sub_scores": {
            "aesthetic": 95, "belonging": 88,
            "showing_off": 93, "collection": 85, "surprise": 90
        },
        "official_tier": 5,
        "price": 288,  # 积分夺宝保底约 ¥2000
        "resale_premium": 0,
        "key_insight": "荣耀典藏天花板，视觉质量 + 获取难度双重溢价",
    },
    # ... 10+ 案例
]
```

## 下一步

- → [08 — 用户工作流与前端](08-user-workflow.md)

## 当前落地入口

仓库已经提供一个轻量销售动作报告 CLI：

```bash
python scripts/generate_sales_report.py --source-key 105-02
python scripts/generate_sales_report.py --search 龙胆 --json
```

该入口复用 `RuleEngine` 的证据优先评估结果，输出：

- `decision`：是否补证据、控制上线、放量投放或先处理价格阻力。
- `sales_readiness`：销售准备度，不等同于销量预测。
- `purchase_drivers`：可用于素材和卖点的购买驱动力。
- `conversion_blockers`：可能压低转化的阻力。
- `recommended_actions`：运营可执行动作。
- `evidence_gaps`：还需要补采的舆论、销量或拥有率信号。

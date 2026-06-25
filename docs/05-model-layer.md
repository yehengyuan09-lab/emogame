# 05 — 情绪溢价模型层

## 模型架构

```
                     Input: SkinFeatureVector (33 dims)
                              │
              ┌───────────────┼───────────────┐
              │               │               │
              ▼               ▼               ▼
    ┌──────────────┐  ┌─────────────┐  ┌──────────────┐
    │ 规则引擎      │  │ XGBoost     │  │ Ensemble     │
    │ (冷启动 MVP)  │  │ Regressor   │  │ (Blending)   │
    │              │  │             │  │              │
    │ 可解释、透明  │  │ 非线性拟合  │  │ 规则+ML混合  │
    └──────┬───────┘  └──────┬──────┘  └──────┬───────┘
           │                 │                │
           └─────────────────┼────────────────┘
                             │
                             ▼
              Total Premium Index (0-100)
                             │
            ┌────────────────┼────────────────┐
            │                │                │
            ▼                ▼                ▼
    ┌──────────────┐  ┌─────────────┐  ┌──────────────┐
    │ Pricing Range │  │ 5-Dim Radar │  │ Risk Alerts  │
    │ (¥)          │  │ Breakdown   │  │              │
    └──────────────┘  └─────────────┘  └──────────────┘
```

## 规则引擎（MVP 冷启动方案）

在标注数据不足（<50 条）时，使用加权评分卡：

当前已实现第一版研究型 MVP。它不会把官方元数据硬转成最终分数，而是区分：

- `official_prior_score`：官方品质、限定、获取方式、图片/详情完整度形成的弱先验。
- `evaluation_score`：来自舆论、营销、销量等外部证据的维度研究分。证据不足时为 `null`。
- `confidence` / `validation_status`：明确提示当前是否已被外部市场信号验证。

```bash
python scripts/evaluate_skin.py --search 地狱岩魂
python scripts/evaluate_skin.py --source-key 105-02 --json
```

实现文件：

| 文件 | 说明 |
|------|------|
| `feature_engineering/features.py` | 评分特征与市场验证信号数据结构 |
| `feature_engineering/pipeline.py` | 从 SQLite 皮肤数据构建评分特征 |
| `models/rule_engine.py` | 五维度规则评分与置信度计算 |
| `scripts/evaluate_skin.py` | 单皮肤评估 CLI |

### 市场验证信号

官方数据只能支撑冷启动分数，最终需要舆论和营销量验证。CLI 支持传入 JSON：

```json
{
  "visual_score": 0.78,
  "feel_score": 0.82,
  "craftsmanship_score": 0.75,
  "collection_score": 0.68,
  "value_score": 0.70,
  "purchase_intent_score": 0.77,
  "sentiment_score": 0.82,
  "discussion_count": 5000,
  "video_views": 1200000,
  "marketing_volume": 3000,
  "sales_volume": 100000,
  "avg_spend_to_obtain": 180,
  "ownership_rate": 0.25
}
```

也可以按 `source_key` 组织多皮肤信号：

```json
{
  "105-02": {
    "visual_score": 0.78,
    "feel_score": 0.82,
    "craftsmanship_score": 0.75,
    "collection_score": 0.68,
    "value_score": 0.70,
    "purchase_intent_score": 0.77,
    "sentiment_score": 0.82,
    "discussion_count": 5000,
    "video_views": 1200000,
    "marketing_volume": 3000,
    "sales_volume": 100000
  }
}
```

运行：

```bash
python scripts/evaluate_skin.py --source-key 105-02 --signals-json market_signals.json
```

也可以先把证据导入 SQLite，再直接评估：

```bash
python scripts/import_market_signals.py market_signals.json
python scripts/evaluate_skin.py --source-key 105-02
```

B 站第一阶段不做搜索爬虫，只支持已知视频 URL/BVID 的证据采集：

```bash
python scripts/fetch_bilibili_evidence.py \
  --source-key 105-02 \
  --video https://www.bilibili.com/video/BVxxxxxxxxxx \
  --aspect-tags visual,feel,craftsmanship
```

该脚本会存储视频标题、作者、URL、播放、弹幕、评论、收藏、投币、分享、点赞等指标，并聚合成 `video_views`、`discussion_count` 和传播互动量。观感、手感、品质、收藏价值、性价比、购买意愿仍需人工标注或后续 NLP 归因后写入。

没有市场信号时，系统只输出 `official_prior_score`，`evaluation_score = null`，`validation_status = insufficient_market_evidence`；证据覆盖足够时会变成 `evidence_validated`。

```python
class RuleEngine:
    """基于领域知识的情绪溢价规则引擎"""

    # 五维度 sub-score 计算规则
    RULES = {
        "aesthetic": {
            "vlm_art_quality":     {"weight": 0.25, "normalize": "minmax"},
            "vlm_effect_score":    {"weight": 0.25, "normalize": "minmax"},
            "vlm_color_harmony":   {"weight": 0.15, "normalize": "minmax"},
            "vlm_composition":     {"weight": 0.10, "normalize": "minmax"},
            "official_tier":       {"weight": 0.20, "normalize": "max_divide", "max": 5},
            "has_voice_pack":      {"weight": 0.03, "binary": True},
            "has_custom_anim":     {"weight": 0.02, "binary": True},
        },
        "belonging": {
            "baidu_index_7d":       {"weight": 0.30, "normalize": "log"},
            "community_post_count": {"weight": 0.25, "normalize": "log"},
            "ip_popularity":        {"weight": 0.20, "normalize": "minmax"},
            "character_usage_rate": {"weight": 0.15, "normalize": "minmax"},
            "bilibili_video_views": {"weight": 0.10, "normalize": "log"},
        },
        "showing_off": {
            "battle_effect_visibility": {"weight": 0.35, "normalize": "minmax"},
            "lobby_display_level":     {"weight": 0.25, "normalize": "minmax"},
            "kill_broadcast_type":     {"weight": 0.20, "normalize": "minmax"},
            "is_giftable":             {"weight": 0.10, "binary": True},
            "gift_popularity_rank":    {"weight": 0.10, "normalize": "invert_rank"},
        },
        "collection": {
            "is_limited":             {"weight": 0.30, "binary": True},
            "series_completion_bonus": {"weight": 0.20, "binary": True},
            "ownership_rate":          {"weight": 0.25, "normalize": "invert_minmax"},
            "rerun_count":             {"weight": 0.10, "normalize": "invert_rank"},
            "days_since_last_rerun":   {"weight": 0.15, "normalize": "log_clip", "max": 730},
        },
        "surprise": {
            "acquisition_method_score": {"weight": 0.35, "mapped": {
                "direct": 0.2, "battle_pass": 0.4, "shard": 0.5,
                "gacha": 0.8, "event": 0.6
            }},
            "gacha_pity_amount":    {"weight": 0.25, "normalize": "log"},
            "drop_rate_percentile": {"weight": 0.25, "normalize": "invert_minmax"},
            "avg_spend_to_obtain":  {"weight": 0.15, "normalize": "log"},
        },
    }

    DIMENSION_WEIGHTS = {
        "aesthetic": 0.30, "belonging": 0.20,
        "showing_off": 0.25, "collection": 0.15, "surprise": 0.10,
    }

    def compute_total_premium(self, features: dict) -> dict:
        sub_scores = {}
        for dim, rules in self.RULES.items():
            score = sum(
                self._normalize(features.get(feat, 0), cfg) * cfg["weight"]
                for feat, cfg in rules.items()
            )
            sub_scores[dim] = min(max(round(score * 100), 0), 100)

        total = sum(sub_scores[d] * self.DIMENSION_WEIGHTS[d] for d in self.DIMENSION_WEIGHTS)
        return {"total_premium": round(total), "sub_scores": sub_scores}
```

## XGBoost 回归模型

```python
import xgboost as xgb
from sklearn.model_selection import cross_val_score, KFold

def train_xgboost_model(X: np.ndarray, y: np.ndarray, feature_names: list[str]):
    """
    训练 XGBoost 情绪溢价预测模型
    X: (n_samples, 33) 特征矩阵
    y: 目标变量（市场价格/官方定价/二级市场溢价）
    """

    params = {
        "objective": "reg:squarederror",
        "max_depth": 5,
        "learning_rate": 0.05,
        "n_estimators": 200,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
    }

    model = xgb.XGBRegressor(**params)

    # 5-Fold CV, 目标 MAPE < 25%
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=kf, scoring="neg_mean_absolute_percentage_error")
    print(f"CV MAPE: {-cv_scores.mean():.2%} ± {cv_scores.std():.2%}")

    model.fit(X, y)

    # 特征重要性
    for name, imp in sorted(zip(feature_names, model.feature_importances_), key=lambda x: -x[1]):
        print(f"  {name}: {imp:.4f}")

    return model
```

### 训练数据构建

1. **初始目标**：官方皮肤定价（点券 → 人民币）作为 y
2. **抽奖皮肤**：使用社区统计的平均获取花费
3. **二级市场溢价**：`resale_premium_ratio` 校正目标值
4. **最终目标**：`y = price_normalized / max_price_in_game`（归一化到 [0,1]）

## Ensemble 融合策略

```python
class EnsemblePredictor:
    """规则引擎 + XGBoost 混合预测"""

    def __init__(self, rule_engine, xgb_model, alpha=0.3):
        self.rule_engine = rule_engine
        self.xgb_model = xgb_model
        self.alpha = alpha  # 规则引擎权重

    def predict(self, features: dict) -> dict:
        rule_result = self.rule_engine.compute_total_premium(features)
        xgb_pred = self.xgb_model.predict(self._to_array(features).reshape(1, -1))[0]

        blended = self.alpha * rule_result["total_premium"] + (1 - self.alpha) * (xgb_pred * 100)

        return {
            "total_premium": round(blended),
            "rule_based": rule_result["total_premium"],
            "ml_based": round(xgb_pred * 100),
            "sub_scores": rule_result["sub_scores"],
        }

    def update_alpha(self, data_size: int):
        """数据越多，ML 权重越高: sigmoid 1.0→0.2"""
        self.alpha = 0.2 + 0.8 / (1 + data_size / 100)
```

### 动态 Alpha 曲线

```
data_size=0   → alpha=1.00 (纯规则引擎)
data_size=50  → alpha=0.73
data_size=100 → alpha=0.60
data_size=200 → alpha=0.47
data_size=500 → alpha=0.33 (几乎纯 ML)
```

## 模型评估指标

| 指标 | 目标值 | 说明 |
|------|--------|------|
| MAPE | < 25% | 平均绝对百分比误差 |
| R² | > 0.7 | 可解释方差比例 |
| MAE | < 10 (满分 100) | 平均绝对误差 |
| 维度一致性 | > 0.8 | 五维度排序与人工标注的 Spearman 相关系数 |

## 下一步

- → [06 — 智能体执行架构](06-agent-architecture.md)

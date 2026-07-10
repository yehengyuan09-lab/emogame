# 04 — 特征工程系统

## 五维情绪溢价特征树

```
Total Emotional Premium = Σ(Dimension_i × Weight_i)

                    ┌──────────┐
                    │ Aesthetic │───────► VLM视觉评分、官方稀有度、特效层级
                    │  审美     │        w = 0.30
                    └──────────┘
                    ┌──────────┐
                    │ Belonging │───────► 英雄人气、IP强度、声优阵容、社区话题量
                    │  归属     │        w = 0.20
                    └──────────┘
                    ┌──────────┐
      溢价 ←────────┤Showing-off│───────► 局内可见度、击杀播报、回城特效、社交礼物
                    │  炫耀     │        w = 0.25
                    └──────────┘
                    ┌──────────┐
                    │Collection │───────► 系列完整度、限定标识、返场间隔、拥有率
                    │  收集     │        w = 0.15
                    └──────────┘
                    ┌──────────┐
                    │ Surprise  │───────► 获取方式(直购/抽奖/战令)、保底金额、掉率
                    │  惊喜     │        w = 0.10
                    └──────────┘
```

## 完整特征定义（33 维）

### 审美维度 (Aesthetic, w=0.30)

| # | 特征名 | 类型 | 来源 | 说明 |
|---|--------|------|------|------|
| 1 | `vlm_art_quality` | float [0,10] | VLM L2 | 模型精细度综合评分 (L2.model_detail) |
| 2 | `vlm_effect_score` | float [0,10] | VLM L2 | 特效质量评分 (L2.effect_quality) |
| 3 | `vlm_color_harmony` | float [0,1] | VLM L2 | 色彩和谐度 (L2.color_scheme / 10) |
| 4 | `vlm_composition` | float [0,10] | VLM L2 | 构图评分 (L2.composition) |
| 5 | `official_tier` | int [0,5] | 爬虫 | 官方稀有度 (伴生=0...荣耀典藏=5) |
| 6 | `has_voice_pack` | bool | 爬虫 | 是否含独立语音包 |
| 7 | `has_custom_anim` | bool | 爬虫 | 是否有自定义动作 |
| 8 | `skin_age_days` | int | 爬虫 | 皮肤已发布天数 |

### 归属维度 (Belonging, w=0.20)

| # | 特征名 | 类型 | 来源 | 说明 |
|---|--------|------|------|------|
| 9 | `baidu_index_7d` | int | 百度指数 | 近 7 日搜索指数均值 |
| 10 | `weibo_followers` | int | 爬虫 | 英雄超话粉丝数 |
| 11 | `ip_source_type` | int [0,3] | 爬虫 | 原创=0, 动漫联动=1, 影视联动=2, 文化IP=3 |
| 12 | `ip_popularity` | float [0,1] | 手动+爬虫 | IP 热度标准化值 |
| 13 | `character_usage_rate` | float [0,1] | 官方数据 | 英雄排位出场率 |
| 14 | `character_win_rate` | float [0,1] | 官方数据 | 英雄排位胜率 |
| 15 | `community_post_count` | int | 爬虫 | 贴吧/NGA 相关帖数 |
| 16 | `bilibili_video_views` | int | B站 API | 皮肤视频总播放量 |

### 炫耀维度 (Showing-off, w=0.25)

| # | 特征名 | 类型 | 来源 | 说明 |
|---|--------|------|------|------|
| 17 | `lobby_display_level` | int [0,3] | 游戏机制 | 大厅展示效果 |
| 18 | `battle_effect_visibility` | int [0,3] | 游戏机制 | 局内特效可见度 |
| 19 | `kill_broadcast_type` | int [0,3] | 游戏机制 | 击杀播报类型 |
| 20 | `has_leaderboard_bonus` | bool | 游戏机制 | 排行榜加成 |
| 21 | `is_giftable` | bool | 爬虫 | 是否可赠送 |
| 22 | `gift_popularity_rank` | int | 爬虫 | 赠送排行 (越小越热门) |

### 收集维度 (Collection, w=0.15)

| # | 特征名 | 类型 | 来源 | 说明 |
|---|--------|------|------|------|
| 23 | `series_membership` | int | 爬虫 | 所属系列皮肤数 |
| 24 | `series_completion_bonus` | bool | 游戏机制 | 集齐额外奖励 |
| 25 | `is_limited` | bool | 爬虫 | 是否限定皮肤 |
| 26 | `limited_type` | int | 爬虫 | 周年庆=0...赛季=4 |
| 27 | `rerun_count` | int | 爬虫 | 已返场次数 |
| 28 | `days_since_last_rerun` | int | 爬虫 | 距上次返场天数 |
| 29 | `ownership_rate` | float [0,1] | 社区采样 | 玩家拥有率 |

### 惊喜维度 (Surprise, w=0.10)

| # | 特征名 | 类型 | 来源 | 说明 |
|---|--------|------|------|------|
| 30 | `acquisition_method` | int [0,4] | 爬虫 | 直购=0...活动=4 |
| 31 | `gacha_pity_amount` | float | 游戏机制 | 抽奖保底金额 (元) |
| 32 | `drop_rate_percentile` | float [0,1] | 游戏机制 | 掉率分位数 |
| 33 | `avg_spend_to_obtain` | float | 社区采样 | 玩家平均花费 (元) |

## 归一化策略

| 策略 | 公式 | 适用场景 |
|------|------|----------|
| minmax | `(x - min) / (max - min)` | 有明确边界的连续值（率、评分） |
| log | `log(x + 1) / log(max + 1)` | 长尾分布（搜索指数、播放量） |
| invert_minmax | `1 - (x - min) / (max - min)` | 越小越好（拥有率、返场次数） |
| invert_rank | `1 - (rank / max_rank)` | 排名类（赠送排行） |
| binary | `{0, 1}` | 布尔值 |
| mapped | 枚举 → 连续映射表 | 离散枚举（获取方式） |
| max_divide | `x / max` | 有上限的离散值（稀有度 0-5） |
| log_clip | `log(clip(x, 0, max) + 1) / log(max + 1)` | 有上限的长尾值 |

## SkinFeatureVector 数据结构

```python
from dataclasses import dataclass
from enum import IntEnum

class AcquisitionMethod(IntEnum):
    DIRECT = 0       # 直购
    GACHA = 1        # 抽奖
    BATTLE_PASS = 2  # 战令
    SHARD = 3        # 碎片兑换
    EVENT = 4        # 活动赠送

@dataclass
class SkinFeatureVector:
    skin_id: str
    hero_name: str

    # 审美 (8)
    vlm_art_quality: float
    vlm_effect_score: float
    vlm_color_harmony: float
    vlm_composition: float
    official_tier: int
    has_voice_pack: bool
    has_custom_anim: bool
    skin_age_days: int

    # 归属 (8)
    baidu_index_7d: int
    weibo_followers: int
    ip_source_type: int
    ip_popularity: float
    character_usage_rate: float
    character_win_rate: float
    community_post_count: int
    bilibili_video_views: int

    # 炫耀 (6)
    lobby_display_level: int
    battle_effect_visibility: int
    kill_broadcast_type: int
    has_leaderboard_bonus: bool
    is_giftable: bool
    gift_popularity_rank: int

    # 收集 (7)
    series_membership: int
    series_completion_bonus: bool
    is_limited: bool
    limited_type: int
    rerun_count: int
    days_since_last_rerun: int
    ownership_rate: float

    # 惊喜 (4)
    acquisition_method: int
    gacha_pity_amount: float
    drop_rate_percentile: float
    avg_spend_to_obtain: float

    # 总计 33 维
    def to_array(self) -> np.ndarray: ...
    def to_emotional_scores(self) -> dict[str, float]: ...
```

## 特征工程 Pipeline

```python
class FeaturePipeline:
    def __init__(self, vlm_client, feature_store, crawler_manager):
        self.vlm = vlm_client
        self.store = feature_store
        self.crawler = crawler_manager

    async def extract_features(self, skin_id: str) -> SkinFeatureVector:
        # 1. 缓存检查
        cached = await self.store.get(skin_id)
        if cached and not cached.is_stale(ttl_days=30):
            return cached

        # 2. 并行采集
        vlm_features = await self.vlm.analyze(skin_id)
        market_features = await self.crawler.fetch_market(skin_id)
        community_features = await self.crawler.fetch_community(skin_id)

        # 3. 特征融合 + 写入存储
        vector = SkinFeatureVector(skin_id=skin_id, **vlm_features, **market_features, **community_features)
        await self.store.put(skin_id, vector)
        return vector
```

## 初始权重（AHP 层次分析法）

```
审美 (0.30) ████████████████████████████████
炫耀 (0.25) ██████████████████████████████
归属 (0.20) ████████████████████████
收集 (0.15) ██████████████████
惊喜 (0.10) ████████████
```

初始权重通过 AHP 两两比较矩阵确定，后续通过 XGBoost 特征重要性自动校正。

## 下一步

- → [05 — 情绪溢价模型层](05-model-layer.md)

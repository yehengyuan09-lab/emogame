# EmoGame 技术实现方案（完整单页版）

> 游戏虚拟商品情绪溢价评估智能体 — 完整技术解决方案
>
> **📌 本文档为单页完整版。按主题查阅请参见 [结构化文档索引](README.md)。**

---

## 目录

1. [总体架构](#1-总体架构)
2. [底层视觉语言模型 VLM](#2-底层视觉语言模型-vlm)
3. [数据采集与爬取系统](#3-数据采集与爬取系统)
4. [特征工程系统](#4-特征工程系统)
5. [情绪溢价模型层](#5-情绪溢价模型层)
6. [智能体执行架构](#6-智能体执行架构)
7. [商业价值分析模块](#7-商业价值分析模块)
8. [用户工作流](#8-用户工作流)
9. [技术选型与依赖清单](#9-技术选型与依赖清单)
10. [部署架构](#10-部署架构)
11. [开发路线图](#11-开发路线图)

---

## 1. 总体架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                      Presentation Layer (Streamlit)                   │
│   仪表盘 │ 皮肤录入 │ 溢价报告 │ 竞品对比 │ 风险预警 │ 案例库       │
├─────────────────────────────────────────────────────────────────────┤
│                      Agent Orchestration Layer                       │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐   │
│  │ Collector │  │  Analyzer │  │  Inferer  │  │ Report Generator │   │
│  │  Agent    │  │  Agent    │  │  Agent    │  │  Agent (LLM)      │   │
│  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘  └────────┬─────────┘   │
│        │              │              │                   │            │
│  ┌─────┴──────────────┴──────────────┴───────────────────┴─────────┐ │
│  │                    Message Bus (Redis / In-memory Queue)          │ │
│  └──────────────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────────────┤
│                        Model & Feature Layer                         │
│  ┌───────────────┐  ┌──────────────┐  ┌──────────────────────────┐  │
│  │ VLM Pipeline  │  │ Feature Eng.  │  │  Regression / XGBoost    │  │
│  │ (qwen2.5vl)   │  │ Pipeline      │  │  Model Server            │  │
│  └───────┬───────┘  └──────┬───────┘  └───────────┬──────────────┘  │
│          │                 │                      │                  │
│  ┌───────┴─────────────────┴──────────────────────┴──────────────┐  │
│  │                    Feature Store (SQLite → PostgreSQL)          │  │
│  └────────────────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────────┤
│                          Data Layer                                  │
│  ┌──────────┐  ┌───────────┐  ┌──────────┐  ┌──────────────────┐   │
│  │ Official │  │ Community │  │  Social   │  │  hero-skin-image │   │
│  │ Store    │  │ (Tieba,   │  │  Media    │  │  (local images)  │   │
│  │ Crawler  │  │  NGA)     │  │  (Weibo)  │  │                  │   │
│  └──────────┘  └───────────┘  └──────────┘  └──────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

**分层职责：**

| 层 | 职责 | 核心组件 |
|----|------|----------|
| 数据层 | 多源数据采集、清洗、存储 | Scrapy/httpx 爬虫, 数据管道 |
| 模型层 | VLM 视觉理解、特征工程、溢价回归 | qwen2.5vl, AutoDL GPT5.4-mini, XGBoost, 特征存储 |
| 智能体层 | 多 Agent 协作、任务调度、LLM 报告 | LangGraph, GPT5.4-mini |
| 展示层 | Web 交互、可视化、报告导出 | Streamlit, ECharts |

---

## 2. 底层视觉语言模型 VLM

### 2.1 VLM 选型

对于王者荣耀皮肤图片的视觉理解任务，需要模型能够识别：
- 皮肤精细度（模型面数、材质质感）
- 特效层级（粒子效果、光影、动态范围）
- 色彩方案（主色调、配色协调度、主题一致性）
- 角色姿态与构图（动感、视觉冲击力）
- UI 标识元素（边框装饰、稀有度标签、限定标识）

**推荐方案（Phase 1 本地默认 + API 语义层）：**

| 层级 | 模型 | 部署方式 | 用途 | 延迟 |
|------|------|----------|------|------|
| L1 快速提取 | qwen2.5vl:3b | 本地 Ollama | 批量皮肤分类、色彩提取、构图分析 | ~0.5-2s/img |
| L2 精细分析 | qwen2.5vl:3b | 本地 Ollama | 审美维度评分、特效层级判别、缺陷检测 | ~1-3s/img |
| L3 深度理解 | GPT5.4-mini | AutoDL API | 设计语言解读、文化符号识别、竞品对比描述 | ~3s/img |

**为什么选择 qwen2.5vl:3b？**
- 开源可本地部署，无数据隐私顾虑
- Ollama 原生支持，部署门槛低
- 当前中文 L1/L2 prompt 可以稳定返回可解析 JSON
- 3B 版本适合 Phase 1 批量处理和快速联调

### 2.2 VLM 处理管线

```
                    ┌──────────────────────────────────┐
                    │         VLM Processing Pipeline    │
                    └──────────────────────────────────┘

  Skin Image (1920×882)
        │
        ├─[1]─► 图像预处理
        │       │  - 格式标准化 (JPEG → RGB Tensor)
        │       │  - 多尺度裁剪 (full/center/face)
        │       │  - 色彩直方图提取
        │       │  - Canny 边缘密度检测
        │       │
        │       ▼
        ├─[2]─► qwen2.5vl:3b: 快速分类
        │       │  - 皮肤稀有度分类 (勇者/史诗/传说/无双/荣耀典藏)
        │       │  - 主色调识别 (5-dominant-colors)
        │       │  - 场景类型 (战场/主城/异界/抽象)
        │       │  - 人物占比估算 (0-1)
        │       │  - 特效密集度 (low/mid/high/extreme)
        │       │
        │       ▼
        ├─[3]─► qwen2.5vl:3b: 精细分析
        │       │  - 面部与服饰细节评价 (1-10)
        │       │  - 特效质量判别 (粒子数量、光影层次、动态暗示)
        │       │  - 构图评分 (三分法、视觉引导线、留白比例)
        │       │  - UI 标识识别 (限定标签、首周折扣、抽奖专属)
        │       │  - 与同系列皮肤的一致性判断
        │       │
        │       ▼
        ├─[4]─► GPT5.4-mini: 语义理解
        │       │  - 设计风格解读 (eg. "水墨国风"、"赛博朋克"）
        │       │  - 文化符号锚定 (eg. "山海经异兽"、"敦煌飞天")
        │       │  - 受众群体画像 (核心向/泛用户/女性向/收藏党)
        │       │  - 与同类皮肤差异化描述
        │       │
        │       ▼
        └─[5]─► 输出: VLMFeatureVector
                {
                  "aesthetic": {
                    "art_quality": 8.5,
                    "effect_tier": 3,        // 0-4
                    "color_harmony": 0.82,
                    "pose_dynamism": 7.2,
                    "detail_density": 0.76
                  },
                  "rarity_signals": {
                    "tier_label": "传说",
                    "has_special_border": true,
                    "has_limited_tag": false,
                    "effect_richness": "high"
                  },
                  "style_semantics": {
                    "design_style": "水墨国风",
                    "cultural_ref": "山海经",
                    "target_audience": ["核心玩家", "收藏党"],
                    "similar_skins": ["李白-诗剑行", "上官婉儿-万象笔"]
                  }
                }
```

### 2.3 VLM Prompt 工程

为每个分析阶段设计结构化 prompt，保证输出一致性：

**L1 快速分类 Prompt 模板：**
```
Analyze this game skin image and output JSON:
{
  "rarity_tier": "史诗|传说|无双|荣耀典藏",
  "dominant_colors": ["#HEX1", "#HEX2", "#HEX3", "#HEX4", "#HEX5"],
  "scene_type": "战场|主城|异界|抽象|自然",
  "character_ratio": 0.0-1.0,
  "effect_density": "low|mid|high|extreme"
}
```

**L2 精细分析 Prompt 模板：**
```
[System] 你是游戏皮肤视觉评估专家，需要为下方皮肤图片的审美维度打分。

请从以下角度评估（1-10分），并生成JSON：
- model_detail: 模型精细度（面数、材质、光影反射）
- effect_quality: 特效质量（粒子效果、动态光效、色彩过渡）
- color_scheme: 配色方案（和谐度、辨识度、主题一致性）
- composition: 构图（视觉焦点、动感、层次感）
- uniqueness: 独创性（与同英雄其他皮肤的差异化程度）
- costume_design: 服饰设计（纹理细节、风格统一性）
- background_quality: 背景质量（与主体的融合度、氛围营造）
- ui_elements: 识别所有UI标记（限定标签、价格标签、活动标记）

返回JSON格式。
```

### 2.4 VLM 批量推理优化

对于 130 英雄 × avg 7 皮肤 ≈ 900 张图片的批量分析：

```python
# 策略 1: 分级缓存
# L1 和 L2 结果写入特征存储，仅变更时重新推理

# 策略 2: 批处理 + 动态批大小
# qwen2.5vl:3b 可用于本地批量推理（实际 batch size 按显存调节）
# 总计: 900/8 × 0.5s ≈ 1 分钟完成全量 L1 分析

# 策略 3: 增量更新
# 仅对新皮肤/重做皮肤触发 VLM 推理
```

---

## 3. 数据采集与爬取系统

### 3.1 数据源全景

```
┌─────────────────────────────────────────────────────────────────┐
│                      Data Source Matrix                          │
├──────────────┬──────────────┬────────────┬──────────────────────┤
│ 数据源        │ 采集内容      │ 频率        │ 用途                  │
├──────────────┼──────────────┼────────────┼──────────────────────┤
│ pvp.qq.com   │ 英雄详情、    │ 每日        │ 基础属性、定价、       │
│ (官网)       │ 皮肤列表、    │            │ 官方稀有度标签         │
│              │ 价格、发布时间 │            │                       │
├──────────────┼──────────────┼────────────┼──────────────────────┤
│ hero-skin-   │ 898张皮肤图片 │ 跟随上游    │ VLM 视觉分析输入       │
│ image (本地) │ (5种分辨率)   │ 仓库更新    │                       │
├──────────────┼──────────────┼────────────┼──────────────────────┤
│ Tieba/贴吧   │ 皮肤讨论帖、   │ 每周        │ 玩家情感、口碑、       │
│              │ 购买意愿、口碑 │            │ 热度指数               │
├──────────────┼──────────────┼────────────┼──────────────────────┤
│ NGA 论坛     │ 皮肤评测帖、   │ 每周        │ 硬核玩家评价、         │
│              │ 性价比讨论    │            │ 皮肤质量排名           │
├──────────────┼──────────────┼────────────┼──────────────────────┤
│ Bilibili     │ 皮肤展示视频   │ 每周        │ 视频播放量(热度代理)、  │
│              │ (播放量、弹幕)│            │ 弹幕情感分析           │
├──────────────┼──────────────┼────────────┼──────────────────────┤
│ 百度指数     │ 英雄名搜索热度 │ 每日        │ 英雄人气代理指标       │
├──────────────┼──────────────┼────────────┼──────────────────────┤
│ 闲鱼/5173    │ 账号/皮肤     │ 每周        │ 二级市场溢价、         │
│              │ 二手交易价格  │            │ 稀缺性验证             │
├──────────────┼──────────────┼────────────┼──────────────────────┤
│ 官方公告     │ 皮肤发布日历、 │ 实时        │ 限定/联动/返场         │
│              │ 活动排期      │            │ 时间戳数据            │
└──────────────┴──────────────┴────────────┴──────────────────────┘
```

### 3.2 爬虫架构

```
                        ┌──────────────────┐
                        │  Crawler Manager  │
                        │  (APScheduler)    │
                        └────────┬─────────┘
                                 │
          ┌──────────────────────┼──────────────────────┐
          │                      │                      │
    ┌─────┴─────┐         ┌──────┴──────┐        ┌──────┴──────┐
    │  WebDriver │         │ HTTP Client │        │  API Client │
    │  (Selenium)│         │ (httpx)     │        │ (aiohttp)   │
    └─────┬─────┘         └──────┬──────┘        └──────┬──────┘
          │                      │                      │
    ┌─────┴─────┐         ┌──────┴──────┐        ┌──────┴──────┐
    │ JS渲染页面 │         │ REST API    │        │ 百度指数API │
    │ (Tieba)   │         │ (pvp.qq)    │        │ Bilibili API│
    └───────────┘         └─────────────┘        └─────────────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                        ┌────────┴─────────┐
                        │   Data Pipeline   │
                        │  ┌─────────────┐  │
                        │  │ Cleaner     │  │
                        │  │ Deduplicator│  │
                        │  │ Normalizer  │  │
                        │  └─────────────┘  │
                        └────────┬─────────┘
                                 │
                        ┌────────┴─────────┐
                        │  Feature Store    │
                        │  (SQLite → PG)    │
                        └──────────────────┘
```

### 3.3 核心爬虫实现

```python
# crawlers/official_store.py — 官方商城爬虫
class OfficialStoreCrawler:
    """爬取王者荣耀官网英雄/皮肤数据"""

    BASE_URL = "https://pvp.qq.com/web201605/"
    HERO_LIST_URL = f"{BASE_URL}herolist.shtml"

    async def crawl_hero_list(self) -> list[dict]:
        """获取全英雄列表及基础信息"""
        # 官网 /web201605/js/herolist.json 提供英雄数据
        ...

    async def crawl_hero_detail(self, hero_id: str) -> dict:
        """获取单个英雄的皮肤列表、皮肤名称、皮肤URL"""
        # 访问 herodetail/{hero_name}.shtml
        ...

    async def crawl_skin_pricing(self) -> list[dict]:
        """获取皮肤价格（点券/人民币）、上线时间、限定类型"""
        ...


# crawlers/community.py — 社区情感爬虫
class CommunityCrawler:
    """爬取贴吧、NGA 社区数据"""

    async def crawl_tieba_posts(self, hero_name: str, limit: int = 50):
        """爬取贴吧英雄讨论帖，提取皮肤相关讨论"""
        ...

    async def crawl_nga_reviews(self, skin_name: str) -> list[dict]:
        """爬取 NGA 皮肤评测帖，提取评分和关键评价"""
        ...

    async def crawl_bilibili_videos(self, skin_name: str) -> dict:
        """获取 B 站皮肤展示视频的播放量、弹幕数、点赞数"""
        ...


# crawlers/market.py — 二级市场爬虫
class SecondaryMarketCrawler:
    """爬取闲鱼、5173 等二级市场交易数据"""

    async def crawl_xianyu_listings(self, keyword: str) -> list[dict]:
        """搜索闲鱼上限定皮肤的账号/CDK交易价格"""
        ...

    async def compute_premium_index(self, official_price: float, resale_price: float) -> float:
        """计算二级市场溢价率 = (二手价 - 原价) / 原价"""
        ...
```

### 3.4 反爬策略

| 目标站点 | 反爬难度 | 策略 |
|----------|----------|------|
| pvp.qq.com | 低 | 直接请求 JSON API，无需渲染 |
| 贴吧 | 中 | Selenium + 随机延迟 + Cookie 池 |
| NGA | 中 | httpx + 登录态 + 请求频率限制 (3 req/min) |
| Bilibili | 中 | 开放 API + User-Agent 轮换 |
| 闲鱼 | 高 | 手机 APP 抓包 + token 刷新策略 |
| 百度指数 | 高 | 官方付费 API 或替代（微信指数） |

---

## 4. 特征工程系统

### 4.1 五维情绪溢价特征树

```
Total Emotional Premium = Σ(Dimension_i × Weight_i)

                    ┌──────────┐
                    │ Aesthetic │───────► VLM视觉评分、官方稀有度、特效层级
                    │  审美     │
                    └──────────┘
                    ┌──────────┐
                    │ Belonging │───────► 英雄人气、IP强度、声优阵容、社区话题量
                    │  归属     │
                    └──────────┘
                    ┌──────────┐
      溢价 ←────────┤Showing-off│───────► 局内可见度、击杀播报、回城特效、社交礼物
                    │  炫耀     │
                    └──────────┘
                    ┌──────────┐
                    │Collection │───────► 系列完整度、限定标识、返场间隔、拥有率
                    │  收集     │
                    └──────────┘
                    ┌──────────┐
                    │ Surprise  │───────► 获取方式(直购/抽奖/战令)、保底金额、掉率
                    │  惊喜     │
                    └──────────┘
```

### 4.2 特征定义表

#### 审美维度 (Aesthetic, w=0.30)
| 特征名 | 类型 | 来源 | 说明 |
|--------|------|------|------|
| `vlm_art_quality` | float (0-10) | VLM L1 | 模型精细度综合评分 |
| `vlm_effect_score` | float (0-10) | VLM L1 | 特效质量评分 |
| `vlm_color_harmony` | float (0-1) | VLM L1 | 色彩和谐度 |
| `vlm_composition` | float (0-10) | VLM L2 | 构图评分 |
| `official_tier` | int (0-5) | 爬虫 | 官方稀有度标签映射 (勇者=1,...荣耀典藏=5) |
| `has_voice_pack` | bool | 爬虫 | 是否含独立语音包 |
| `has_custom_anim` | bool | 爬虫 | 是否有自定义动作/待机动画 |
| `skin_age_days` | int | 爬虫 | 皮肤已发布天数 |

#### 归属维度 (Belonging, w=0.20)
| 特征名 | 类型 | 来源 | 说明 |
|--------|------|------|------|
| `baidu_index_7d` | int | 百度指数 | 近7日英雄名搜索指数均值 |
| `weibo_followers` | int | 爬虫 | 英雄相关超话粉丝数 |
| `ip_source_type` | enum | 爬虫 | IP来源 (原创=0, 联动动漫=1, 联动影视=2, 文化IP=3) |
| `ip_popularity` | float (0-1) | 手动+爬虫 | IP热度标准化值 |
| `character_usage_rate` | float | 官方数据 | 英雄出场率(%) |
| `character_win_rate` | float | 官方数据 | 英雄胜率(%) |
| `community_post_count` | int | 爬虫 | 贴吧/NGA 皮肤讨论帖数量 |
| `bilibili_video_views` | int | B站API | 皮肤相关视频总播放量 |

#### 炫耀维度 (Showing-off, w=0.25)
| 特征名 | 类型 | 来源 | 说明 |
|--------|------|------|------|
| `lobby_display_level` | int (0-3) | 游戏机制 | 大厅展示效果(头像框/名片/3D模型) |
| `battle_effect_visibility` | int (0-3) | 游戏机制 | 局内特效可见度(普攻/技能/击杀/回城) |
| `kill_broadcast_type` | enum | 游戏机制 | 击杀播报类型(无/普通/全屏/动态) |
| `has_leaderboard_bonus` | bool | 游戏机制 | 是否带排行榜加成 |
| `is_giftable` | bool | 爬虫 | 是否可赠送 |
| `gift_popularity_rank` | int | 爬虫 | 赠送排行榜排名 |

#### 收集维度 (Collection, w=0.15)
| 特征名 | 类型 | 来源 | 说明 |
|--------|------|------|------|
| `series_membership` | int | 爬虫 | 所属系列皮肤数量 |
| `series_completion_bonus` | bool | 游戏机制 | 集齐是否有额外奖励 |
| `is_limited` | bool | 爬虫 | 是否限定皮肤 |
| `limited_type` | enum | 爬虫 | 限定类型(周年庆/情人节/年限/战令/赛季) |
| `rerun_count` | int | 爬虫 | 已返场次数 |
| `days_since_last_rerun` | int | 爬虫 | 距上次返场天数 |
| `ownership_rate` | float | 社区采样 | 拥有率估计值 |

#### 惊喜维度 (Surprise, w=0.10)
| 特征名 | 类型 | 来源 | 说明 |
|--------|------|------|------|
| `acquisition_method` | enum | 爬虫 | 获取方式(直购/抽奖/战令/碎片兑换/活动赠送) |
| `gacha_pity_amount` | float | 游戏机制 | 抽奖保底金额(元) |
| `drop_rate_percentile` | float | 游戏机制 | 掉率在全部物品中的分位数 |
| `pity_mechanism_type` | enum | 游戏机制 | 保底类型(无/累计/硬保底/软保底) |
| `avg_spend_to_obtain` | float | 社区采样 | 玩家平均花费(元) |

### 4.3 特征工程 Pipeline

```python
# feature_engineering/pipeline.py

from dataclasses import dataclass
from enum import Enum
import numpy as np

class AcquisitionMethod(Enum):
    DIRECT = "direct"       # 直购
    GACHA = "gacha"         # 抽奖
    BATTLE_PASS = "battle_pass"  # 战令
    SHARD = "shard"         # 碎片兑换
    EVENT = "event"         # 活动赠送

@dataclass
class SkinFeatureVector:
    """单个皮肤的全量特征向量"""
    skin_id: str
    hero_name: str

    # === 审美 ===
    vlm_art_quality: float
    vlm_effect_score: float
    vlm_color_harmony: float
    vlm_composition: float
    official_tier: int
    has_voice_pack: bool
    has_custom_anim: bool
    skin_age_days: int

    # === 归属 ===
    baidu_index_7d: int
    weibo_followers: int
    ip_source_type: int
    ip_popularity: float
    character_usage_rate: float
    community_post_count: int
    bilibili_video_views: int

    # === 炫耀 ===
    lobby_display_level: int
    battle_effect_visibility: int
    kill_broadcast_type: int
    has_leaderboard_bonus: bool
    is_giftable: bool
    gift_popularity_rank: int

    # === 收集 ===
    series_membership: int
    series_completion_bonus: bool
    is_limited: bool
    limited_type: str
    rerun_count: int
    days_since_last_rerun: int
    ownership_rate: float

    # === 惊喜 ===
    acquisition_method: AcquisitionMethod
    gacha_pity_amount: float
    drop_rate_percentile: float
    avg_spend_to_obtain: float

    def to_array(self) -> np.ndarray:
        """转为模型输入数组 (31维特征)"""
        ...

    def to_emotional_scores(self) -> dict[str, float]:
        """按五维度分组，计算各维度原始分"""
        ...


class FeaturePipeline:
    """特征工程主管线"""

    def __init__(self, vlm_client, feature_store, crawler_manager):
        self.vlm = vlm_client
        self.store = feature_store
        self.crawler = crawler_manager

    async def extract_features(self, skin_id: str) -> SkinFeatureVector:
        """端到端特征提取"""
        # 1. 从特征存储查询已有特征
        cached = await self.store.get(skin_id)
        if cached and not cached.is_stale():
            return cached

        # 2. 并行采集特征
        vlm_features = await self.vlm.analyze(skin_id)      # VLM 管线
        market_features = await self.crawler.fetch_market(skin_id)  # 爬虫管线
        community_features = await self.crawler.fetch_community(skin_id)

        # 3. 特征融合
        vector = SkinFeatureVector(
            skin_id=skin_id,
            **vlm_features,
            **market_features,
            **community_features,
        )

        # 4. 写入特征存储
        await self.store.put(skin_id, vector)
        return vector
```

### 4.4 特征重要性预分析

基于领域知识（王者荣耀皮肤定价机制），预期权重分布：

```
审美 (0.30) ████████████████████████████████
炫耀 (0.25) ██████████████████████████████
归属 (0.20) ████████████████████████
收集 (0.15) ██████████████████
惊喜 (0.10) ████████████
```

初始权重通过 AHP 层次分析法确定，后续通过 XGBoost 特征重要性自动校正。

---

## 5. 情绪溢价模型层

### 5.1 模型架构

```
                     Input: SkinFeatureVector (31 dims)
                              │
              ┌───────────────┼───────────────┐
              │               │               │
              ▼               ▼               ▼
    ┌──────────────┐  ┌─────────────┐  ┌──────────────┐
    │ 规则引擎      │  │ XGBoost     │  │ Ensemble     │
    │ (冷启动)      │  │ Regressor   │  │ (Blending)   │
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

### 5.2 规则引擎（MVP 冷启动方案）

在数据量不足时使用加权评分卡：

```python
# models/rule_engine.py

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
            "baidu_index_7d":      {"weight": 0.30, "normalize": "log"},
            "community_post_count": {"weight": 0.25, "normalize": "log"},
            "ip_popularity":       {"weight": 0.20, "normalize": "minmax"},
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
            "gacha_pity_amount":        {"weight": 0.25, "normalize": "log"},
            "drop_rate_percentile":     {"weight": 0.25, "normalize": "invert_minmax"},
            "avg_spend_to_obtain":      {"weight": 0.15, "normalize": "log"},
        },
    }

    # 维度间权重（AHP 初始化，后续 ML 校正）
    DIMENSION_WEIGHTS = {
        "aesthetic": 0.30,
        "belonging": 0.20,
        "showing_off": 0.25,
        "collection": 0.15,
        "surprise": 0.10,
    }

    def compute_total_premium(self, features: dict) -> dict:
        sub_scores = {}
        for dim, rules in self.RULES.items():
            score = 0
            for feat_name, cfg in rules.items():
                raw = features.get(feat_name, 0)
                score += self._normalize(raw, cfg) * cfg["weight"]
            # 缩放到 0-100
            sub_scores[dim] = min(max(round(score * 100), 0), 100)

        total = sum(
            sub_scores[dim] * self.DIMENSION_WEIGHTS[dim]
            for dim in self.DIMENSION_WEIGHTS
        )
        return {
            "total_premium": round(total),
            "sub_scores": sub_scores,
            "dimension_weights": self.DIMENSION_WEIGHTS,
        }
```

### 5.3 XGBoost 回归模型

训练流程：

```python
# models/train.py

import xgboost as xgb
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_absolute_percentage_error

def train_xgboost_model(
    X: np.ndarray,      # 特征矩阵 (n_samples, 31)
    y: np.ndarray,      # 目标变量: 市场价格 / 官方最大定价 / 二级市场溢价
    feature_names: list[str],
):
    """训练 XGBoost 情绪溢价预测模型"""

    # 超参数（通过 Optuna 搜索优化）
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

    # 5-Fold 交叉验证
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=kf, scoring="neg_mean_absolute_percentage_error")

    print(f"CV MAPE: {-cv_scores.mean():.2%} ± {cv_scores.std():.2%}")
    # 目标: MAPE < 25%

    # 全量训练
    model.fit(X, y)

    # 特征重要性分析
    importance = model.feature_importances_
    for name, imp in sorted(zip(feature_names, importance), key=lambda x: -x[1]):
        print(f"  {name}: {imp:.4f}")

    return model
```

**训练数据构建策略：**
1. 以官方皮肤定价（点券 → 人民币）作为初始目标变量
2. 对于限定/抽奖皮肤，使用社区统计的平均花费
3. 对于二级市场有溢价的皮肤，使用 `resale_premium_ratio` 校正
4. 最终目标 `y = price_normalized / max_price_in_game`

### 5.4 模型融合

```python
# models/ensemble.py

class EnsemblePredictor:
    """规则引擎 + XGBoost 混合预测器"""

    def __init__(self, rule_engine, xgb_model, blend_alpha=0.3):
        self.rule_engine = rule_engine
        self.xgb_model = xgb_model
        self.alpha = blend_alpha  # 规则引擎权重（数据少时更高）

    def predict(self, features: dict) -> dict:
        # 规则引擎预测
        rule_result = self.rule_engine.compute_total_premium(features)

        # XGBoost 预测
        X = self._features_to_array(features)
        xgb_pred = self.xgb_model.predict(X.reshape(1, -1))[0]

        # Blending
        blended_total = self.alpha * rule_result["total_premium"] + \
                        (1 - self.alpha) * (xgb_pred * 100)

        return {
            "total_premium": round(blended_total),
            "rule_based": rule_result["total_premium"],
            "ml_based": round(xgb_pred * 100),
            "sub_scores": rule_result["sub_scores"],
        }

    def update_alpha(self, data_size: int):
        """数据量越大，ML 权重越高"""
        # sigmoid 从 1.0 (0条数据) 降到 0.2 (500+条数据)
        self.alpha = 0.2 + 0.8 / (1 + data_size / 100)
```

---

## 6. 智能体执行架构

### 6.1 多智能体协作系统

采用 **LangGraph** 构建有状态的多智能体工作流：

```
                        ┌──────────────────────┐
                        │   Orchestrator Agent  │
                        │   (主管智能体)          │
                        │   - 任务分解与调度       │
                        │   - 状态管理            │
                        │   - 异常处理            │
                        └──────────┬───────────┘
                                   │
           ┌───────────────────────┼───────────────────────┐
           │                       │                       │
    ┌──────┴──────┐         ┌──────┴──────┐         ┌──────┴──────┐
    │ Collector   │         │  Analyzer   │         │  Reporter   │
    │ Agent       │         │  Agent      │         │  Agent      │
    │             │         │             │         │             │
    │ - 数据采集   │         │ - VLM 触发  │         │ - LLM 报告  │
    │ - 爬虫调度   │         │ - 特征工程  │         │ - 策略建议  │
    │ - 数据清洗   │         │ - 模型推理  │         │ - 可视化数据│
    └──────┬──────┘         └──────┬──────┘         └──────┬──────┘
           │                       │                       │
           └───────────────────────┼───────────────────────┘
                                   │
                        ┌──────────┴───────────┐
                        │   Shared State        │
                        │   (Pydantic Models)    │
                        │   - TaskContext        │
                        │   - AnalysisResult     │
                        │   - ReportArtifacts    │
                        └──────────────────────┘
```

### 6.2 LangGraph 工作流定义

```python
# agents/workflow.py

from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator

class AgentState(TypedDict):
    """智能体共享状态"""
    # 输入
    skin_id: str
    hero_name: str
    game_genre: str  # MOBA / FPS / RPG / Gacha

    # 中间产物
    raw_data: dict          # Collector Agent 采集的原始数据
    vlm_features: dict      # VLM 分析结果
    feature_vector: dict    # 特征工程输出
    premium_result: dict    # 模型推理结果
    business_analysis: dict # 商业分析结果

    # 控制
    errors: Annotated[list, operator.add]
    current_stage: str


def create_evaluation_graph() -> StateGraph:
    """构建情绪溢价评估工作流图"""
    workflow = StateGraph(AgentState)

    # 添加节点
    workflow.add_node("collect_data", collector_node)
    workflow.add_node("vlm_analyze", vlm_node)
    workflow.add_node("feature_engineer", feature_engineer_node)
    workflow.add_node("model_inference", model_inference_node)
    workflow.add_node("business_analyze", business_analysis_node)
    workflow.add_node("generate_report", report_generation_node)
    workflow.add_node("error_handler", error_handler_node)

    # 定义边
    workflow.set_entry_point("collect_data")
    workflow.add_edge("collect_data", "vlm_analyze")
    workflow.add_edge("vlm_analyze", "feature_engineer")
    workflow.add_edge("feature_engineer", "model_inference")
    workflow.add_edge("model_inference", "business_analyze")
    workflow.add_edge("business_analyze", "generate_report")
    workflow.add_edge("generate_report", END)

    # 条件边：错误时跳转到 error_handler
    workflow.add_conditional_edges(
        "vlm_analyze",
        lambda s: "error_handler" if s["errors"] else "feature_engineer",
    )

    return workflow.compile()


# ─── Agent 节点实现 ───

async def collector_node(state: AgentState) -> AgentState:
    """数据采集智能体：调度爬虫/API 获取多源数据"""
    # 调用爬虫管理器
    crawler = get_crawler_manager()
    raw_data = await crawler.collect_all(
        skin_id=state["skin_id"],
        hero_name=state["hero_name"],
    )
    state["raw_data"] = raw_data
    state["current_stage"] = "data_collected"
    return state


async def vlm_node(state: AgentState) -> AgentState:
    """VLM 分析智能体：触发视觉模型管线"""
    try:
        vlm = get_vlm_pipeline()
        image_path = state["raw_data"]["image_paths"]["wallpaper_big"]
        vlm_features = await vlm.analyze(image_path)
        state["vlm_features"] = vlm_features
    except Exception as e:
        state["errors"].append(f"VLM error: {e}")
    state["current_stage"] = "vlm_done"
    return state


async def feature_engineer_node(state: AgentState) -> AgentState:
    """特征工程智能体：融合多源数据构建特征向量"""
    fe = get_feature_pipeline()
    feature_vector = await fe.build_vector(
        vlm_features=state["vlm_features"],
        market_data=state["raw_data"]["market"],
        community_data=state["raw_data"]["community"],
    )
    state["feature_vector"] = feature_vector.to_dict()
    state["current_stage"] = "features_ready"
    return state


async def model_inference_node(state: AgentState) -> AgentState:
    """模型推理智能体：运行规则引擎 + XGBoost"""
    predictor = get_ensemble_predictor()
    result = predictor.predict(state["feature_vector"])
    state["premium_result"] = result
    state["current_stage"] = "inference_done"
    return state


async def business_analysis_node(state: AgentState) -> AgentState:
    """商业分析智能体：定价建议、风险、竞品"""
    analyzer = get_business_analyzer()
    analysis = await analyzer.analyze(
        premium_result=state["premium_result"],
        feature_vector=state["feature_vector"],
        hero_name=state["hero_name"],
        game_genre=state["game_genre"],
    )
    state["business_analysis"] = analysis
    return state


async def report_generation_node(state: AgentState) -> AgentState:
    """报告生成智能体：调用 LLM 生成自然语言报告"""
    llm = get_llm_client()
    report = await llm.generate_report(
        premium=state["premium_result"],
        business=state["business_analysis"],
        skin_id=state["skin_id"],
    )
    state["report"] = report
    state["current_stage"] = "completed"
    return state
```

### 6.3 LLM 报告生成 Prompt

```python
REPORT_SYSTEM_PROMPT = """你是游戏商业化分析师，专精于MOBA游戏虚拟物品定价策略。

你收到一份情绪溢价评估数据，需要生成一份面向游戏运营人员的策略报告。

报告结构：
1. **情绪溢价总览** (2-3句总结)
2. **五维度雷达解读** (每个维度1-2句分析)
3. **定价区间建议** (给出具体人民币区间及理由)
4. **风险提示** (列出2-3条业务风险)
5. **竞品参照** (与同类皮肤对比)
6. **运营策略建议** (2-3条可执行建议)

要求：
- 数据驱动，引用具体得分
- 语气专业但不失亲和
- 每条建议可落地执行
- 使用中文输出
"""

REPORT_USER_PROMPT_TEMPLATE = """
请基于以下数据生成评估报告：

**皮肤信息**
- 英雄：{hero_name}
- 皮肤：{skin_name}
- 游戏类型：{game_genre}

**情绪溢价指数**: {total_premium}/100

**五维度得分**:
- 审美: {aesthetic}/100
- 归属: {belonging}/100
- 炫耀: {showing_off}/100
- 收集: {collection}/100
- 惊喜: {surprise}/100

**关键特征**:
- 官方稀有度: {official_tier}
- 获取方式: {acquisition_method}
- 是否限定: {is_limited}
- IP来源: {ip_source}
- VLM视觉评分: {vlm_art_quality}/10
- 英雄搜索热度: {baidu_index}
- 社区讨论量: {community_posts}

**市场数据**:
- 官方定价: ¥{official_price}
- 二级市场溢价率: {resale_premium}%
- 玩家平均获取花费: ¥{avg_spend}

请生成完整的策略报告。
"""
```

### 6.4 工具定义（Agent Tools）

```python
# agents/tools.py

from langchain.tools import tool

@tool
def search_hero_skins(hero_name: str) -> list[dict]:
    """查询指定英雄的所有皮肤列表"""
    ...

@tool
def query_skin_price(skin_name: str) -> dict:
    """查询指定皮肤的历史定价和当前价格"""
    ...

@tool
def get_vlm_analysis(skin_id: str) -> dict:
    """触发 VLM 对皮肤图片进行视觉分析"""
    ...

@tool
def fetch_community_sentiment(hero_name: str, days: int = 30) -> dict:
    """获取社区情感数据（近N天讨论量、正面/负面比例）"""
    ...

@tool
def compare_similar_skins(skin_id: str, top_k: int = 5) -> list[dict]:
    """查找同类皮肤（同稀有度/同英雄/同系列）并返回对比数据"""
    ...

@tool
def calculate_premium(features: dict) -> dict:
    """输入特征向量，返回情绪溢价指数和五维度分"""
    ...

@tool
def get_market_trends(game_genre: str) -> dict:
    """获取特定游戏品类的市场趋势数据"""
    ...
```

---

## 7. 商业价值分析模块

### 7.1 分析引擎

```python
# business/pricing.py

import numpy as np
from dataclasses import dataclass
from typing import Optional

@dataclass
class PricingRecommendation:
    """定价建议"""
    suggested_price_min: float  # 建议最低价 (¥)
    suggested_price_max: float  # 建议最高价 (¥)
    confidence: float           # 置信度 (0-1)
    price_per_emotion_point: float  # 单位情绪溢价对应的价格
    comparable_skins: list[dict]    # 可比皮肤列表


class PricingEngine:
    """定价引擎：将情绪溢价指数映射到价格区间"""

    # 王者荣耀皮肤价格锚点 (2024 年市场数据)
    PRICE_ANCHORS = {
        "MOBA": {
            0:   (0, 0),      # 活动赠送
            20:  (6, 28),     # 伴生皮
            40:  (28, 48),    # 勇者
            60:  (48, 88),    # 史诗
            80:  (88, 178),   # 传说
            95:  (178, 288),  # 无双/典藏
        }
    }

    def recommend_price(
        self,
        premium_score: float,
        official_tier: int,
        game_genre: str = "MOBA",
    ) -> PricingRecommendation:
        """基于情绪溢价和官方稀有度推荐定价区间"""

        anchors = self.PRICE_ANCHORS.get(game_genre, self.PRICE_ANCHORS["MOBA"])

        # 找到最接近的锚点
        anchor_points = sorted(anchors.keys())
        closest_idx = min(range(len(anchor_points)),
                          key=lambda i: abs(anchor_points[i] - premium_score))

        base_min, base_max = anchors[anchor_points[closest_idx]]

        # 根据溢价在相邻锚点间插值
        if premium_score <= anchor_points[0]:
            factor = premium_score / anchor_points[0]
        elif premium_score >= anchor_points[-1]:
            factor = 1.0
        else:
            lower = anchor_points[max(0, closest_idx - 1)]
            upper = anchor_points[min(len(anchor_points) - 1, closest_idx + 1)]
            factor = (premium_score - lower) / (upper - lower + 1e-6)

        price_scale = 1.0 + 0.5 * (premium_score / 100)  # 溢价加成
        suggested_min = round(base_min * price_scale)
        suggested_max = round(base_max * price_scale)

        return PricingRecommendation(
            suggested_price_min=suggested_min,
            suggested_price_max=suggested_max,
            confidence=0.85 if official_tier >= 3 else 0.70,
            price_per_emotion_point=round(suggested_max / premium_score, 1) if premium_score > 0 else 0,
            comparable_skins=self._find_comparables(premium_score, official_tier),
        )


class RiskAnalyzer:
    """风险分析器"""

    def analyze(self, features: dict, premium: dict, pricing: PricingRecommendation) -> list[dict]:
        risks = []

        # 风险 1: 定价偏离情绪价值
        official_price = features.get("official_price", 0)
        if official_price > pricing.suggested_price_max * 1.3:
            risks.append({
                "level": "high",
                "type": "overpricing",
                "message": f"官方定价 ¥{official_price} 远超情绪价值定价上限 ¥{pricing.suggested_price_max}，建议控制价格或提升皮肤质量",
                "action": "建议增加特效层级或降低首发价格"
            })
        elif official_price < pricing.suggested_price_min * 0.7:
            risks.append({
                "level": "medium",
                "type": "underpricing",
                "message": f"当前定价 ¥{official_price} 低于情绪价值支撑的价格下限 ¥{pricing.suggested_price_min}，存在利润损失",
                "action": "可考虑小幅提价或增加捆绑销售方案"
            })

        # 风险 2: 限定皮肤返场过频
        if features.get("is_limited") and features.get("rerun_count", 0) >= 3:
            risks.append({
                "level": "high",
                "type": "rerun_dilution",
                "message": f"限定皮肤已返场 {features['rerun_count']} 次，稀缺性被严重稀释",
                "action": "建议停止返场或转为常驻皮肤"
            })

        # 风险 3: 低审美高分皮肤
        if premium["sub_scores"]["aesthetic"] < 40 and premium["total_premium"] > 60:
            risks.append({
                "level": "medium",
                "type": "aesthetic_deficit",
                "message": "溢价主要来自稀缺性和IP热度，审美质量不足，长期可能影响口碑",
                "action": "建议在下次皮肤迭代中提升模型和特效质量"
            })

        return risks
```

### 7.2 竞品对比分析

```python
# business/competitor.py

class CompetitorAnalyzer:
    """竞品对比分析器"""

    async def find_competitors(
        self,
        skin_features: dict,
        game_genre: str,
        top_k: int = 5,
    ) -> list[dict]:
        """在同类游戏中寻找可比皮肤"""
        # 1. 从特征存储中检索同稀有度、同价位段皮肤
        similar = await self.feature_store.query_similar(
            skin_features,
            filters={
                "game_genre": game_genre,
                "tier_range": (skin_features["official_tier"] - 1, skin_features["official_tier"] + 1),
            },
            top_k=top_k * 2,
        )

        # 2. 计算多维相似度
        scored = []
        for s in similar:
            score = self._cosine_similarity(
                self._to_vector(skin_features),
                self._to_vector(s),
            )
            scored.append({**s, "similarity": score})

        # 3. 按相似度排序，返回 top_k
        scored.sort(key=lambda x: -x["similarity"])
        return scored[:top_k]

    def generate_comparison_report(self, target: dict, competitors: list[dict]) -> str:
        """生成竞品对比自然语言报告"""
        comparisons = []
        for c in competitors:
            diff = target["total_premium"] - c["total_premium"]
            comparisons.append(
                f"- {c['game_name']} {c['hero_name']}-{c['skin_name']}: "
                f"情绪溢价 {c['total_premium']} ({'+' if diff > 0 else ''}{diff} vs 当前皮肤)"
            )
        return "\n".join(comparisons)
```

### 7.3 案例库构建

```python
# business/case_library.py

CASE_LIBRARY = [
    {
        "name": "孙悟空-至尊宝",
        "game": "王者荣耀",
        "genre": "MOBA",
        "total_premium": 85,
        "sub_scores": {"aesthetic": 78, "belonging": 92, "showing_off": 85, "collection": 90, "surprise": 75},
        "official_tier": 3,  # 史诗 (但限定返场)
        "price": 88.8,
        "resale_premium": 120,  # 二级市场溢价 120%
        "key_insight": "大话西游IP联动 + 限定返场稀缺性 = 极高情绪溢价",
    },
    {
        "name": "小乔-天鹅之梦",
        "game": "王者荣耀",
        "genre": "MOBA",
        "total_premium": 92,
        "sub_scores": {"aesthetic": 95, "belonging": 88, "showing_off": 93, "collection": 85, "surprise": 90},
        "official_tier": 5,  # 荣耀典藏
        "price": 288,  # 保底约 ¥2000（积分夺宝）
        "resale_premium": 0,
        "key_insight": "荣耀典藏天花板，视觉质量 + 获取难度双重溢价",
    },
    # ... 10+ 案例
]
```

---

## 8. 用户工作流

### 8.1 完整用户旅程

```
┌─────────────────────────────────────────────────────────────────────┐
│                    User Workflow (端到端)                             │
└─────────────────────────────────────────────────────────────────────┘

  [用户]                          [系统]                         [输出]
    │                               │                              │
    │  ① 输入皮肤信息                │                              │
    │  ┌─────────────────────┐      │                              │
    │  │ 游戏: 王者荣耀       │      │                              │
    │  │ 英雄: 孙悟空         │      │                              │
    │  │ 皮肤: 至尊宝         │      │                              │
    │  │ 上传皮肤图片(可选)    │      │                              │
    │  └─────────────────────┘      │                              │
    │──────────────────────────────►│                              │
    │                               │                              │
    │                               │  ② 触发数据采集流水线         │
    │                               │  ┌──────────────────────┐    │
    │                               │  │ Collector Agent      │    │
    │                               │  │ - 查询官方数据库      │    │
    │                               │  │ - 爬取社区讨论        │    │
    │                               │  │ - 获取百度指数        │    │
    │                               │  └──────────────────────┘    │
    │                               │                              │
    │                               │  ③ VLM 视觉分析              │
    │                               │  ┌──────────────────────┐    │
    │                               │  │ qwen2.5vl L1/L2      │    │
    │                               │  │ → GPT5.4-mini        │    │
    │                               │  └──────────────────────┘    │
    │                               │                              │
    │                               │  ④ 特征工程 + 模型推理        │
    │                               │  ┌──────────────────────┐    │
    │                               │  │ Rule Engine          │    │
    │                               │  │ XGBoost Predictor    │    │
    │                               │  │ Ensemble Blending    │    │
    │                               │  └──────────────────────┘    │
    │                               │                              │
    │                               │  ⑤ 商业分析                   │
    │                               │  ┌──────────────────────┐    │
    │                               │  │ 定价建议              │    │
    │                               │  │ 风险预警              │    │
    │                               │  │ 竞品对比              │    │
    │                               │  └──────────────────────┘    │
    │                               │                              │
    │                               │  ⑥ LLM 生成报告              │
    │                               │  ┌──────────────────────┐    │
    │                               │  │ GPT5.4-mini          │    │
    │                               │  │ 策略建议 + 自然语言   │    │
    │                               │  └──────────────────────┘    │
    │                               │                              │
    │  ⑦ 返回结构化报告              │                              │
    │◄──────────────────────────────│                              │
    │                               │                              │
    │  ┌─────────────────────┐      │                              │
    │  │ 📊 情绪溢价指数: 85  │      │                              │
    │  │ ╭─────────────────╮ │      │                              │
    │  │ │  审美  ████░░ 70  │ │      │                              │
    │  │ │  归属  █████░ 92  │ │      │    雷达图                   │
    │  │ │  炫耀  ████░  80  │ │      │    PNG/HTML                 │
    │  │ │  收集  █████  90  │ │      │                              │
    │  │ │  惊喜  ███░░  60  │ │      │                              │
    │  │ ╰─────────────────╯ │      │                              │
    │  │                      │      │                              │
    │  │ 💰 建议定价: ¥88-178 │      │                              │
    │  │ ⚠️ 风险: 返场过频     │      │                              │
    │  │ 📋 竞品: 李白-凤求凰  │      │                              │
    │  │ 📝 策略建议: ...      │      │                              │
    │  └─────────────────────┘      │                              │
    │                               │                              │
    │  ⑧ 交互操作                    │                              │
    │  - 调整特征参数（what-if）      │                              │
    │  - 导出 PDF/JSON 报告          │                              │
    │  - 分享报告链接                │                              │
    │  - 保存到案例库                │                              │
    │                               │                              │
```

### 8.2 Streamlit 前端页面结构

```python
# app.py — Streamlit 主入口

import streamlit as st
from agents.workflow import create_evaluation_graph

st.set_page_config(
    page_title="EmoGame - 情绪溢价评估",
    page_icon="🎮",
    layout="wide",
)

# 侧边栏：输入区
with st.sidebar:
    st.title("📋 皮肤信息录入")

    game = st.selectbox("游戏", ["王者荣耀", "英雄联盟手游", "原神", "崩坏：星穹铁道"])
    hero_name = st.text_input("英雄/角色名")
    skin_name = st.text_input("皮肤名称")

    # 高级选项
    with st.expander("高级特征输入"):
        official_tier = st.selectbox("官方稀有度", ["勇者", "史诗", "传说", "无双", "荣耀典藏"])
        acquisition = st.selectbox("获取方式", ["直购", "抽奖", "战令", "碎片兑换", "活动赠送"])
        is_limited = st.checkbox("限定皮肤")
        official_price = st.number_input("官方定价 (¥)", min_value=0.0, step=1.0)

    uploaded_image = st.file_uploader("上传皮肤图片 (可选)", type=["jpg", "png", "webp"])

    submitted = st.button("🔍 开始评估", type="primary", use_container_width=True)

# 主区域
st.title("🎮 EmoGame 情绪溢价评估")

if submitted:
    with st.spinner("正在采集数据..."):
        # 运行智能体工作流
        workflow = create_evaluation_graph()
        result = workflow.invoke({
            "skin_id": f"{hero_name}-{skin_name}",
            "hero_name": hero_name,
            "game_genre": "MOBA",
        })

    # 结果展示
    col1, col2 = st.columns([2, 1])

    with col1:
        # 情绪溢价指数大数字
        st.metric(
            label="情绪溢价指数",
            value=f"{result['premium_result']['total_premium']}/100",
            delta=None,
        )

        # 五维度雷达图
        st.subheader("五维度雷达图")
        radar_fig = plot_radar(result["premium_result"]["sub_scores"])
        st.plotly_chart(radar_fig, use_container_width=True)

        # LLM 生成的策略报告
        st.subheader("📝 策略报告")
        st.markdown(result.get("report", "报告生成中..."))

    with col2:
        # 定价建议
        st.subheader("💰 定价建议")
        pricing = result["business_analysis"]["pricing"]
        st.metric("建议价格区间", f"¥{pricing['min']} - ¥{pricing['max']}")

        # 风险预警
        st.subheader("⚠️ 风险预警")
        for risk in result["business_analysis"]["risks"]:
            level_color = {"high": "red", "medium": "orange", "low": "green"}
            st.warning(f"**{risk['type']}**: {risk['message']}")

        # 竞品对比
        st.subheader("📋 竞品对比")
        for comp in result["business_analysis"]["competitors"]:
            st.info(f"**{comp['name']}**: 溢价 {comp['premium']}/100")

    # 导出功能
    st.divider()
    col_export1, col_export2, col_export3 = st.columns(3)
    with col_export1:
        st.download_button("📄 导出 PDF 报告", data=generate_pdf(result), file_name="report.pdf")
    with col_export2:
        st.download_button("📊 导出 JSON 数据", data=generate_json(result), file_name="data.json")
    with col_export3:
        if st.button("💾 保存到案例库"):
            save_to_case_library(result)
            st.success("已保存!")
```

---

## 9. 技术选型与依赖清单

### 9.1 完整技术栈

| 层级 | 技术 | 版本 | 用途 |
|------|------|------|------|
| **语言** | Python | 3.12 | 全栈后端 |
| **Web框架** | FastAPI | 0.111+ | REST API |
| **前端** | Streamlit | 1.35+ | 原型仪表盘 |
| **VLM** | qwen2.5vl | 3B | L1 视觉分类 + L2 精细视觉分析 |
| | Pillow / OpenCV | — | 图像预处理 |
| **LLM** | AutoDL GPT5.4-mini | — | 语义分析 / 报告生成 |
| | LangChain / LangGraph | 0.2+ | 智能体编排 |
| **ML** | XGBoost | 2.1+ | 溢价回归 |
| | scikit-learn | 1.5+ | 预处理、交叉验证 |
| | Optuna | 3.6+ | 超参数搜索 |
| **数据采集** | httpx | 0.27+ | 异步 HTTP 客户端 |
| | Selenium | 4.22+ | JS 渲染页面 |
| | BeautifulSoup4 | 4.12+ | HTML 解析 |
| **数据存储** | SQLite → PostgreSQL | — | 特征存储 / 案例库 |
| | Redis | 7.2+ | 消息队列 / 缓存 |
| **可视化** | Plotly | 5.22+ | 雷达图/折线图 |
| | ECharts (via streamlit-echarts) | — | 仪表盘图表 |
| **部署** | Docker | — | 容器化 |
| | nginx | — | 反向代理 |
| | Uvicorn | 0.30+ | ASGI Server |

### 9.2 Python 依赖

```txt
# requirements.txt — 核心依赖
fastapi==0.111.1
uvicorn[standard]==0.30.1
streamlit==1.35.0
httpx==0.27.0
selenium==4.22.0
beautifulsoup4==4.12.3

# ML / Data
xgboost==2.1.0
scikit-learn==1.5.0
optuna==3.6.1
numpy==1.26.4
pandas==2.2.2

# VLM / Vision
torch==2.3.1
torchvision==0.18.1
transformers==4.41.0
pillow==10.3.0
opencv-python-headless==4.10.0

# LLM / Agent
openai==1.35.0
langchain==0.2.5
langgraph==0.1.0
langchain-openai==0.1.9

# Visualization
plotly==5.22.0
streamlit-echarts==1.4.0

# Utils
pydantic==2.7.2
pydantic-settings==2.3.0
python-dotenv==1.0.1
tenacity==8.3.0
loguru==0.7.2
```

---

## 10. 部署架构

```
                         ┌──────────────────┐
                         │    User Browser   │
                         └────────┬─────────┘
                                  │ HTTP
                         ┌────────┴─────────┐
                         │  nginx (:443)     │
                         │  Reverse Proxy    │
                         └────────┬─────────┘
                                  │
                    ┌─────────────┼─────────────┐
                    │             │             │
           ┌────────┴──────┐ ┌───┴──────┐ ┌───┴──────────┐
           │ Streamlit     │ │ FastAPI  │ │ Static Files  │
           │ (:8501)       │ │ (:8000)  │ │ (images/cases)│
           └────────┬──────┘ └───┬──────┘ └──────────────┘
                    │             │
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
     ┌────────┴──────┐ ┌───┴──────┐ ┌──┴───────────┐
     │ VLM Server    │ │ Redis    │ │ PostgreSQL    │
     │ (GPU: T4/A10) │ │ (:6379)  │ │ (:5432)       │
     └───────────────┘ └──────────┘ └───────────────┘
```

**GPU 需求：**
- 开发环境：本地 Ollama 运行 qwen2.5vl:3b
- 生产环境：A10 24GB 可并行运行 qwen2.5vl:3b，并使用 AutoDL GPT5.4-mini 作为 L3
- 若 GPU 不可用：退化为传统 CV + AutoDL API + CPU XGBoost

---

## 11. 开发路线图

```
Phase 1 (Week 1-2): Foundation
├── ✅ 环境搭建: .venv + pip install
├── ✅ 数据加载: hero-skin-image JSON 解析
├── ✅ VLM 集成: qwen2.5vl:3b L1/L2 + AutoDL L3 + 缓存
├── ✅ 爬虫框架: WZRY 官方皮肤数据 + Weibo 社交评论入口
├── ✅ 特征存储: SQLite schema + CRUD API
└── ✅ 基础 Streamlit: 英雄选择 + 皮肤浏览

Phase 2 (Week 3-4): Feature Engineering
├── ⬜ 31 维特征向量完整实现
├── ⬜ VLM 三级管线联调
├── ⬜ 社区数据爬虫 (贴吧 + NGA + Bilibili)
├── ⬜ 特征归一化 + 缺失值处理
└── ⬜ 初始数据集标注 (50 条)

Phase 3 (Week 5-6): Model
├── ⬜ 规则引擎完整实现
├── ⬜ XGBoost 训练 + CV 验证
├── ⬜ Ensemble blending
├── ⬜ 初始权重 AHP 校准
└── ⬜ 模型序列化 + 版本管理

Phase 4 (Week 7-8): Agent & Integration
├── ⬜ LangGraph 智能体工作流
├── ⬜ LLM 报告生成 (GPT5.4-mini)
├── ⬜ FastAPI 端点
├── ⬜ Streamlit 完整仪表盘
└── ⬜ 案例库 (10+ 条)

Phase 5 (Week 9-10): Polish & Deploy
├── ⬜ 竞品对比分析模块
├── ⬜ 风险预警完善
├── ⬜ PDF/JSON 导出
├── ⬜ Docker 容器化
├── ⬜ 性能优化 (缓存策略)
└── ⬜ 用户文档 + 演示视频素材
```

---

## 附录 A: 项目目录结构

```
emogame/
├── app.py                          # Streamlit 主入口
├── api/
│   ├── main.py                     # FastAPI app
│   └── routes/
│       ├── evaluation.py           # /api/evaluate
│       ├── cases.py                # /api/cases
│       └── export.py               # /api/export
├── agents/
│   ├── __init__.py
│   ├── workflow.py                 # LangGraph 智能体工作流
│   ├── tools.py                    # Agent 工具定义
│   └── prompts.py                  # LLM Prompt 模板
├── vlm/
│   ├── __init__.py
│   ├── pipeline.py                 # VLM 三级管线
│   ├── l1_classifier.py            # qwen2.5vl L1 分类
│   ├── l2_analyzer.py              # qwen2.5vl L2 审美分析
│   ├── l3_semantic.py              # AutoDL GPT5.4-mini 语义分析
│   └── preprocess.py               # 图像预处理
├── crawlers/
│   ├── __init__.py
│   ├── manager.py                  # 爬虫管理器
│   ├── official_store.py           # 官网爬虫
│   ├── community.py                # 社区爬虫
│   └── market.py                   # 二级市场爬虫
├── feature_engineering/
│   ├── __init__.py
│   ├── pipeline.py                 # 特征工程主管线
│   ├── features.py                 # SkinFeatureVector 定义
│   └── normalizers.py              # 归一化策略
├── models/
│   ├── __init__.py
│   ├── rule_engine.py              # 规则引擎
│   ├── xgboost_model.py            # XGBoost 模型
│   ├── ensemble.py                 # 模型融合
│   └── train.py                    # 训练脚本
├── business/
│   ├── __init__.py
│   ├── pricing.py                  # 定价引擎
│   ├── risk.py                     # 风险分析
│   ├── competitor.py               # 竞品分析
│   └── case_library.py             # 案例库
├── data/
│   ├── feature_store.py            # 特征存储服务
│   ├── schemas.py                  # 数据库 Schema
│   └── migrations/                 # 数据库迁移
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_vlm_analysis.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_training.ipynb
│   └── 05_validation.ipynb
├── tests/
│   ├── test_vlm_pipeline.py
│   ├── test_feature_engineering.py
│   ├── test_rule_engine.py
│   └── test_workflow.py
├── hero-skin-image/                # Git submodule
├── docs/
│   └── technical-implementation-plan.md  # 本文档
├── .venv/
├── CLAUDE.md
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

---

*文档版本: v1.0 | 最后更新: 2026-06-05*

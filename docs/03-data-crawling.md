# 03 — 数据采集与爬取系统

## 数据源全景

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

## 爬虫架构

```
                        ┌──────────────────┐
                        │  Crawler Manager  │
                        │  (APScheduler)    │
                        │  - 定时调度        │
                        │  - 并发控制        │
                        │  - 失败重试        │
                        └────────┬─────────┘
                                 │
          ┌──────────────────────┼──────────────────────┐
          │                      │                      │
    ┌─────┴─────┐         ┌──────┴──────┐        ┌──────┴──────┐
    │  WebDriver │         │ HTTP Client │        │  API Client │
    │  (Selenium)│         │ (httpx)     │        │ (aiohttp)   │
    │            │         │             │        │             │
    │ JS 渲染页面│         │ REST/JSON   │        │ 开放 API    │
    └─────┬─────┘         └──────┬──────┘        └──────┬──────┘
          │                      │                      │
    ┌─────┴─────┐         ┌──────┴──────┐        ┌──────┴──────┐
    │ 贴吧       │         │ pvp.qq.com │        │ 百度指数 API│
    │ 动态页面   │         │ 静态 JSON  │        │ Bilibili API│
    └───────────┘         └─────────────┘        └─────────────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                        ┌────────┴─────────┐
                        │   Data Pipeline   │
                        │  - Cleaner       │
                        │  - Deduplicator  │
                        │  - Normalizer    │
                        └────────┬─────────┘
                                 │
                        ┌────────┴─────────┐
                        │  Feature Store    │
                        └──────────────────┘
```

## 核心爬虫接口设计

当前已实现第一阶段官方数据采集脚本：

```bash
python crawlers/wzry_skin_crawler.py --limit 50
```

采集策略：

1. 使用 `https://pvp.qq.com/web201605/js/herolist.json` 作为英雄和皮肤名称全集基准。
2. 使用 `https://pvp.qq.com/zlkdatasys/heroskinlist.json` 补充皮肤 ID、品质、上线时间、获取方式、介绍、图片、详情页和视频。
3. 按 `(hero_name, skin_name)` 合并两个官方源。
4. 没有匹配到详情源的目录皮肤仍然保留，标记为 `has_detail_record = 0`。
5. 图片资产写入 `skin_assets`，默认只下载 `skin_primary`，也可以用 `--skip-images` 只保存 URL。

SQLite 表：

| 表 | 用途 |
|------|------|
| `crawl_runs` | 记录每次采集的来源、数量和失败统计 |
| `heroes` | 英雄基础信息和皮肤数量 |
| `skins` | 皮肤目录与官方增强字段 |
| `skin_assets` | 图片 URL、本地路径、下载状态和 SHA-256 |

查询与过滤统一通过 `data/skin_repository.py`：

- `list_heroes()`：英雄列表，可按定位和关键词过滤。
- `list_skins()`：皮肤列表，可按英雄、品质、详情匹配状态、目录来源和主图状态过滤。
- `search_skins()`：按英雄名、皮肤名或皮肤 ID 搜索。
- `get_skin()` / `list_assets()`：读取单个皮肤详情和图片资产。
- `stats()`：输出英雄数、皮肤数、资产数、缺失详情、缺失主图和下载失败数量。

数据质量检查脚本：

```bash
python scripts/check_wzry_data.py
python scripts/check_wzry_data.py --json
```

第一阶段暂不抓取微博、贴吧、NGA、Bilibili 和二级市场数据。这些来源在官方数据闭环稳定后再接入，用于补充热度、口碑和价格事件。

### 舆论证据采集边界

当前已接入的是“已知证据导入”，不是全网搜索爬虫：

- `scripts/import_market_signals.py`：导入人工整理或半自动整理的维度证据。
- `scripts/fetch_bilibili_evidence.py`：对已知 B 站视频 URL/BVID 拉取视频指标。
- `data/market_signal_repository.py`：存储聚合信号和原始证据条目。

这样可以先保证每条证据可追溯，避免把搜索噪声直接灌进评分系统。后续再做微博、贴吧、评论文本抓取和 NLP 维度归因。

```python
# crawlers/manager.py
class CrawlerManager:
    """爬虫管理器：统一调度、并发控制、结果聚合"""

    def __init__(self):
        self.official = OfficialStoreCrawler()
        self.community = CommunityCrawler()
        self.market = SecondaryMarketCrawler()

    async def collect_all(self, skin_id: str, hero_name: str) -> dict:
        """并行采集所有数据源"""
        official_data, community_data, market_data = await asyncio.gather(
            self.official.crawl_hero_detail(hero_name),
            self.community.crawl_all(hero_name, skin_id),
            self.market.crawl_resale_data(skin_id),
        )
        return {
            "official": official_data,
            "community": community_data,
            "market": market_data,
            "collected_at": datetime.now().isoformat(),
        }


# crawlers/official_store.py
class OfficialStoreCrawler:
    """王者荣耀官方商城爬虫"""

    BASE_URL = "https://pvp.qq.com/web201605/"

    async def crawl_hero_list(self) -> list[dict]:
        """获取全英雄列表 (herolist.json)"""

    async def crawl_hero_detail(self, hero_name: str) -> dict:
        """获取英雄详情页：皮肤名、价格、发布时间"""

    async def crawl_skin_pricing(self) -> list[dict]:
        """获取所有皮肤价格（点券/人民币）及限定类型"""


# crawlers/community.py
class CommunityCrawler:
    """社区数据爬虫（贴吧 + NGA + Bilibili）"""

    async def crawl_tieba_posts(self, hero_name: str, limit: int = 50):
        """贴吧帖子列表 + 情感分析"""

    async def crawl_nga_reviews(self, skin_name: str) -> list[dict]:
        """NGA 皮肤评测帖评分"""

    async def crawl_bilibili_videos(self, skin_name: str) -> dict:
        """B 站视频播放量、弹幕数"""


# crawlers/market.py
class SecondaryMarketCrawler:
    """二级市场数据爬虫"""

    async def crawl_xianyu_listings(self, keyword: str) -> list[dict]:
        """闲鱼账号/CDK 交易价格"""

    def compute_premium_index(self, official_price: float, resale_price: float) -> float:
        """溢价率 = (二手价 - 原价) / 原价"""
```

## 反爬策略矩阵

| 目标站点 | 难度 | 策略 | 频率限制 |
|----------|------|------|----------|
| pvp.qq.com | 低 | 直接请求 JSON API，无 JS 渲染 | 1 req/s |
| 贴吧 | 中 | Selenium + 随机延迟 (2-5s) + Cookie 池 | 10 req/min |
| NGA | 中 | httpx + 登录态 Cookie | 3 req/min |
| Bilibili | 中 | 开放 API (api.bilibili.com) + UA 轮换 | 5 req/s |
| 闲鱼 | 高 | APP 抓包 + Token 刷新 + 代理 IP | 5 req/min |
| 百度指数 | 高 | 官方付费 API 或替代（微信指数） | — |

### 通用反反爬措施
- **User-Agent 池**：20+ 真实浏览器 UA 轮换
- **请求间隔**：正态分布随机延迟 `N(μ=3s, σ=1s)`
- **失败重试**：指数退避 `delay = min(60, 2^retry + random(0,1))`
- **数据完整性校验**：每次爬取后校验 JSON schema，异常则告警

## 增量更新策略

```python
# 仅采集变更数据
async def incremental_update(self, last_crawl_time: datetime):
    # 1. 官网：对比 hero-skin-image 的 git diff
    # 2. 社区：按时间倒序拉取新帖
    # 3. 二级市场：全量刷新（频率低，数据量小）
    ...
```

## 下一步

- → [04 — 特征工程系统](04-feature-engineering.md)

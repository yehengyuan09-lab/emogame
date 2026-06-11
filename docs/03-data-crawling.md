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

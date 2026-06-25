"""
Hero Skin Weibo Comment Crawler

Fetches Weibo comments on 王者荣耀 hero skin posts for emotional premium analysis.

Data sources:
  1. 王者荣耀 official account (UID 5698023579) — skin announcement / teaser posts
  2. Keyword search — hero name + "皮肤" queries on Weibo search

API endpoints (m.weibo.cn, no auth beyond cookie):
  - /comments/hotflow      — hot comments (paginated by max_id)
  - /comments/show         — all comments (paginated by page)
  - /api/container/getIndex — user timeline / keyword search

Output: Structured JSON in data/weibo_comments/ compatible with the
existing wzry_skin_comments_YYYY-MM-DD.json format.

Usage:
  # Crawl comments for specific post MIDs
  python crawlers/weibo_skin_comment_crawler.py --mids 5312215779382153

  # Crawl latest skin posts from 王者荣耀 official account
  python crawlers/weibo_skin_comment_crawler.py --official --days 7

  # Search by hero name
  python crawlers/weibo_skin_comment_crawler.py --search "赵云 皮肤" --limit 5

  # Full scan: official posts + keyword search
  python crawlers/weibo_skin_comment_crawler.py --official --search-heroes --days 3
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

WZRY_UID = 5698023579  # 王者荣耀 official Weibo account
WZRY_CONTAINERID = "1076035698023579"  # containerid for timeline API

HOTFLOW_URL = "https://m.weibo.cn/comments/hotflow"
COMMENTS_SHOW_URL = "https://m.weibo.cn/comments/show"
CONTAINER_URL = "https://m.weibo.cn/api/container/getIndex"
SEARCH_URL = "https://m.weibo.cn/api/container/getIndex"

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (iPhone; CPU iPhone OS 15_0 like Mac OS X) "
    "AppleWebKit/605.1.15 (KHTML, like Gecko) "
    "Version/15.0 Mobile/15E148 Safari/604.1"
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SECRET_PATH = PROJECT_ROOT / "weiboSpider" / ".secret"
OUTPUT_DIR = PROJECT_ROOT / "data" / "weibo_comments"
HERO_SKIN_DB = PROJECT_ROOT / "data" / "wzry_skins" / "skins.sqlite3"

# Hero names to search for when --search-heroes is used.
# Populated at runtime from the skins database if available,
# otherwise falls back to this curated list of popular heroes.
FALLBACK_HERO_NAMES = [
    "赵云", "李白", "孙悟空", "貂蝉", "诸葛亮",
    "韩信", "鲁班七号", "妲己", "安琪拉", "虞姬",
    "孙尚香", "铠", "花木兰", "兰陵王", "露娜",
    "公孙离", "曜", "镜", "上官婉儿", "百里守约",
    "瑶", "大乔", "小乔", "王昭君", "西施",
    "吕布", "关羽", "张飞", "马超", "澜",
]

logger = logging.getLogger("weibo_skin_crawler")


# ---------------------------------------------------------------------------
# data models
# ---------------------------------------------------------------------------

@dataclass
class WeiboComment:
    """A single Weibo comment."""
    user: str
    text: str
    like_count: int
    total_number: int  # reply count
    created_at: str
    source: str = ""  # e.g. "来自安徽"


@dataclass
class WeiboPost:
    """A Weibo post with its metadata."""
    mid: str
    user_id: int
    user_name: str
    text: str
    created_at: str
    reposts_count: int
    comments_count: int
    attitudes_count: int  # likes
    source: str = ""
    pics: list[str] = field(default_factory=list)


@dataclass
class CrawlEntry:
    """Aggregated result for one post — matches existing JSON output format."""
    mid: str
    title: str  # truncated post text used as label
    total_comments: int
    meaningful: int  # count of comments that passed skin-relevance filter
    low_quality: int  # count of comments filtered out
    post: WeiboPost | None = None  # full post metadata (new field)
    comments: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# cookie loading
# ---------------------------------------------------------------------------

def load_cookie() -> str:
    """Read the Weibo cookie from weiboSpider/.secret."""
    if not SECRET_PATH.exists():
        raise FileNotFoundError(
            f"Cookie file not found at {SECRET_PATH}. "
            "Place a valid Weibo cookie in weiboSpider/.secret"
        )
    cookie = SECRET_PATH.read_text().strip()
    if not cookie:
        raise ValueError(f"{SECRET_PATH} is empty — add a valid Weibo cookie")
    return cookie


# ---------------------------------------------------------------------------
# hero name index
# ---------------------------------------------------------------------------

def load_hero_names() -> list[str]:
    """Load hero names from the skins database, fall back to curated list."""
    import sqlite3
    if not HERO_SKIN_DB.exists():
        logger.warning("Skins database not found, using fallback hero list")
        return FALLBACK_HERO_NAMES
    try:
        conn = sqlite3.connect(str(HERO_SKIN_DB))
        rows = conn.execute("SELECT DISTINCT hero_name FROM skins ORDER BY hero_name").fetchall()
        conn.close()
        names = [r[0] for r in rows if r[0]]
        if names:
            logger.info("Loaded %d hero names from skins database", len(names))
            return names
    except Exception:
        logger.exception("Failed to load hero names from database")
    return FALLBACK_HERO_NAMES


# ---------------------------------------------------------------------------
# HTTP client & rate limiting
# ---------------------------------------------------------------------------

class RateLimiter:
    """Token-bucket–inspired rate limiter for polite crawling."""

    def __init__(self, min_delay: float = 0.8, max_delay: float = 2.5):
        self.min_delay = min_delay
        self.max_delay = max_delay

    async def wait(self) -> None:
        delay = random.uniform(self.min_delay, self.max_delay)
        await asyncio.sleep(delay)


class WeiboClient:
    """Async HTTP client for m.weibo.cn with cookie auth and retry logic."""

    def __init__(self, cookie: str, timeout: float = 30.0, max_retries: int = 4):
        self.cookie = cookie
        self.timeout = timeout
        self.max_retries = max_retries
        self.rate_limiter = RateLimiter()
        self._client: httpx.AsyncClient | None = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("Use WeiboClient as async context manager")
        return self._client

    async def __aenter__(self) -> "WeiboClient":
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout),
            headers={
                "User-Agent": DEFAULT_USER_AGENT,
                "Cookie": self.cookie,
                "Accept": "application/json, text/plain, */*",
                "X-Requested-With": "XMLHttpRequest",
                "Referer": "https://m.weibo.cn/",
            },
            follow_redirects=True,
        )
        return self

    async def __aexit__(self, *args: Any) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def _request(
        self, url: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """GET a JSON endpoint with retry + rate-limiting."""
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                await self.rate_limiter.wait()
                resp = await self.client.get(url, params=params)
                resp.raise_for_status()
                data = resp.json()
                # m.weibo.cn may return ok=0 for rate-limit / auth errors
                if isinstance(data, dict) and data.get("ok") == 0:
                    msg = data.get("msg", "unknown error")
                    if "频率" in str(msg) or "太快" in str(msg):
                        wait = min(60, 2 ** (attempt + 1))
                        logger.warning("Rate-limited, waiting %ds ...", wait)
                        await asyncio.sleep(wait)
                        continue
                    raise httpx.HTTPStatusError(
                        f"Weibo API returned ok=0: {msg}",
                        request=resp.request,
                        response=resp,
                    )
                return data
            except (httpx.TimeoutException, httpx.HTTPStatusError) as e:
                last_exc = e
                if attempt < self.max_retries - 1:
                    wait = 2 ** attempt + random.uniform(0, 1)
                    logger.debug("Request failed (attempt %d/%d), retrying in %.1fs: %s",
                                 attempt + 1, self.max_retries, wait, e)
                    await asyncio.sleep(wait)
                else:
                    raise
        raise last_exc  # type: ignore[misc]


# ---------------------------------------------------------------------------
# post discovery
# ---------------------------------------------------------------------------

async def fetch_official_posts(
    client: WeiboClient,
    since_date: str | None = None,
    max_posts: int = 20,
) -> list[WeiboPost]:
    """Fetch recent posts from 王者荣耀 official account timeline."""
    posts: list[WeiboPost] = []
    page = 1

    while len(posts) < max_posts:
        data = await client._request(CONTAINER_URL, params={
            "type": "uid",
            "value": WZRY_UID,
            "containerid": WZRY_CONTAINERID,
            "page": page,
        })
        cards = data.get("data", {}).get("cards", [])
        if not cards:
            break

        for card in cards:
            if card.get("card_type") != 9:  # type 9 = text/image post
                continue
            mblog = card.get("mblog", {})
            if not mblog:
                continue
            mid = mblog.get("id", "")
            created_at = mblog.get("created_at", "")
            # Check date filter — skip (don't stop) for pinned posts
            if since_date and created_at:
                try:
                    post_date = _parse_weibo_date(created_at)
                    cutoff = datetime.strptime(since_date, "%Y-%m-%d").date()
                    if post_date.date() < cutoff:
                        continue  # skip pinned/old posts, keep scanning
                except ValueError:
                    pass

            posts.append(WeiboPost(
                mid=mid,
                user_id=mblog.get("user", {}).get("id", WZRY_UID),
                user_name=mblog.get("user", {}).get("screen_name", ""),
                text=_clean_html(mblog.get("text", "")),
                created_at=created_at,
                reposts_count=mblog.get("reposts_count", 0),
                comments_count=mblog.get("comments_count", 0),
                attitudes_count=mblog.get("attitudes_count", 0),
                source=mblog.get("source", ""),
                pics=[p.get("url", "") for p in mblog.get("pics", [])],
            ))

            if len(posts) >= max_posts:
                break

        page += 1
        if page > 50:  # safety cap
            break

    return posts


async def search_skin_posts(
    client: WeiboClient,
    keyword: str,
    since_date: str | None = None,
    max_posts: int = 5,
) -> list[WeiboPost]:
    """Search Weibo for posts matching a keyword (hero name + 皮肤)."""
    posts: list[WeiboPost] = []
    page = 1

    while len(posts) < max_posts:
        data = await client._request(SEARCH_URL, params={
            "type": "all",
            "queryVal": keyword,
            "containerid": f"100103type=1&q={keyword}",
            "page": page,
        })
        cards = data.get("data", {}).get("cards", [])
        if not cards:
            break

        for card in cards:
            # Search results may have a wrapper card_group
            card_group = card.get("card_group", [card])
            for inner in card_group:
                mblog = inner.get("mblog", {})
                if not mblog:
                    continue
                mid = mblog.get("id", "")
                created_at = mblog.get("created_at", "")
                if since_date and created_at:
                    try:
                        post_date = _parse_weibo_date(created_at)
                        cutoff = datetime.strptime(since_date, "%Y-%m-%d").date()
                        if post_date.date() < cutoff:
                            continue
                    except ValueError:
                        pass

                posts.append(WeiboPost(
                    mid=mid,
                    user_id=mblog.get("user", {}).get("id", 0),
                    user_name=mblog.get("user", {}).get("screen_name", ""),
                    text=_clean_html(mblog.get("text", "")),
                    created_at=created_at,
                    reposts_count=mblog.get("reposts_count", 0),
                    comments_count=mblog.get("comments_count", 0),
                    attitudes_count=mblog.get("attitudes_count", 0),
                    source=mblog.get("source", ""),
                    pics=[p.get("url", "") for p in mblog.get("pics", [])],
                ))

                if len(posts) >= max_posts:
                    break
            if len(posts) >= max_posts:
                break

        page += 1
        if page > 20:
            break

    return posts


# ---------------------------------------------------------------------------
# comment fetching
# ---------------------------------------------------------------------------

async def fetch_hot_comments(
    client: WeiboClient,
    mid: str,
    max_comments: int = 200,
) -> list[dict]:
    """Fetch hot comments for a post using the hotflow API."""
    comments: list[dict] = []
    max_id = 0

    while len(comments) < max_comments:
        data = await client._request(HOTFLOW_URL, params={
            "id": mid,
            "mid": mid,
            "max_id_type": 0,
            "max_id": max_id,
        })
        if data.get("ok") != 1:
            break

        items = data.get("data", {}).get("data", [])
        if not items:
            break

        for item in items:
            if len(comments) >= max_comments:
                break
            comments.append({
                "user": item.get("user", {}).get("screen_name", ""),
                "text": _clean_html(item.get("text", "")),
                "like_count": item.get("like_count", 0),
                "total_number": item.get("total_number", 0),
                "created_at": item.get("created_at", ""),
                "source": item.get("source", ""),
            })

        max_id = data.get("data", {}).get("max_id", 0)
        if max_id == 0:
            break

    return comments[:max_comments]


async def fetch_all_comments(
    client: WeiboClient,
    mid: str,
    max_comments: int = 500,
) -> list[dict]:
    """Fetch all comments (hot + recent) using the comments/show API."""
    comments: list[dict] = []
    page = 1

    while len(comments) < max_comments:
        data = await client._request(COMMENTS_SHOW_URL, params={
            "id": mid,
            "mid": mid,
            "page": page,
        })
        if data.get("ok") != 1:
            break

        items = data.get("data", {}).get("data", [])
        if not items:
            break

        for item in items:
            if len(comments) >= max_comments:
                break
            comments.append({
                "user": item.get("user", {}).get("screen_name", ""),
                "text": _clean_html(item.get("text", "")),
                "like_count": item.get("like_count", 0),
                "total_number": item.get("total_number", 0),
                "created_at": item.get("created_at", ""),
                "source": item.get("source", ""),
            })

        page += 1
        if page > data.get("data", {}).get("max", 10):
            break

    return comments[:max_comments]


# ---------------------------------------------------------------------------
# comment filtering / classification
# ---------------------------------------------------------------------------

# Patterns that indicate a low-quality or bot comment
LOW_QUALITY_PATTERNS = [
    re.compile(p) for p in [
        r"^转发微博$",
        r"^转发$",
        r"^[👍🔥😍❤️💪🌹👏🙏]+$",  # emoji-only
        r"^[.。…]+$",
        r"^打卡$",
        r"^沙发$",
        r"^来了$",
        r"^第一$",
    ]
]

# Patterns that indicate skin-relevant discussion
SKIN_SIGNAL_PATTERNS = [
    re.compile(p) for p in [
        r"皮肤",
        r"特效",
        r"建模",
        r"原画",
        r"海报",
        r"手感",
        r"局内",
        r"语音",
        r"音效",
        r"回城",
        r"出场",
        r"动画",
        r"价格",
        r"点券",
        r"传说",
        r"史诗",
        r"勇者",
        r"限定",
        r"无双",
        r"荣耀典藏",
        r"返场",
        r"入手",
        r"买",
        r"值不值",
        r"好看",
        r"丑",
        r"帅",
        r"美",
        r"炫",
        r"期待",
        r"爆料",
        r"品质",
    ]
]


def is_low_quality(comment: dict) -> bool:
    """Check if a comment is low-quality / spam / bot text."""
    text = comment.get("text", "")
    if len(text) <= 2:
        return True
    for pattern in LOW_QUALITY_PATTERNS:
        if pattern.match(text):
            return True
    return False


def count_skin_signals(comment: dict) -> int:
    """Count how many skin-relevant signal words appear in the comment."""
    text = comment.get("text", "")
    count = 0
    for pattern in SKIN_SIGNAL_PATTERNS:
        if pattern.search(text):
            count += 1
    return count


def filter_comments(comments: list[dict], min_signals: int = 0) -> tuple[list[dict], int]:
    """Split comments into meaningful and low-quality groups.

    A comment is meaningful if it is NOT low-quality AND
    (min_signals == 0 OR has at least min_signals skin signals).
    """
    meaningful: list[dict] = []
    low_quality: int = 0

    for c in comments:
        if is_low_quality(c):
            low_quality += 1
        elif min_signals > 0 and count_skin_signals(c) < min_signals:
            low_quality += 1
        else:
            meaningful.append(c)

    return meaningful, low_quality


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

_HTML_RE = re.compile(r"<[^>]+>")


def _clean_html(text: str) -> str:
    """Strip HTML tags and decode entities."""
    text = _HTML_RE.sub(" ", text)
    text = text.replace("&nbsp;", " ").replace("&amp;", "&")
    text = text.replace("&lt;", "<").replace("&gt;", ">")
    text = text.replace("&quot;", '"').replace("&#39;", "'")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _parse_weibo_date(date_str: str) -> datetime:
    """Parse Weibo's 'Sun Jun 21 12:00:32 +0800 2026' format."""
    # Remove timezone name offset since strptime has limited tz support
    # Format: "Sun Jun 21 12:00:32 +0800 2026"
    return datetime.strptime(date_str, "%a %b %d %H:%M:%S %z %Y")


def _make_title(text: str, max_len: int = 40) -> str:
    """Create a short title from post text."""
    text = _clean_html(text)
    if len(text) <= max_len:
        return text
    return text[:max_len - 1] + "…"


def _output_filename(prefix: str = "wzry_skin_comments") -> str:
    """Generate dated output filename."""
    today = datetime.now().strftime("%Y-%m-%d")
    return f"{prefix}_{today}.json"


# ---------------------------------------------------------------------------
# main crawler orchestrator
# ---------------------------------------------------------------------------

@dataclass
class CrawlConfig:
    """Configuration for a crawl run."""
    mids: list[str] = field(default_factory=list)
    official: bool = False
    search_keywords: list[str] = field(default_factory=list)
    search_heroes: bool = False
    days: int = 7
    limit_per_keyword: int = 5
    hot_only: bool = False
    max_comments_per_post: int = 300
    min_skin_signals: int = 0  # 0 = keep all non-low-quality
    output: str = ""
    quiet: bool = False


async def crawl(config: CrawlConfig) -> list[CrawlEntry]:
    """Run the full crawl pipeline and return results."""
    cookie = load_cookie()
    results: list[CrawlEntry] = []
    all_posts: list[WeiboPost] = []

    since_date = (
        (datetime.now() - timedelta(days=config.days)).strftime("%Y-%m-%d")
        if config.days > 0
        else None
    )

    async with WeiboClient(cookie) as client:
        # ── 1. Discover posts ──

        # Direct MIDs
        for mid in config.mids:
            all_posts.append(WeiboPost(
                mid=mid, user_id=WZRY_UID, user_name="", text="",
                created_at="", reposts_count=0, comments_count=0, attitudes_count=0,
            ))

        # Official account timeline
        if config.official:
            logger.info("Fetching posts from @王者荣耀 official account ...")
            posts = await fetch_official_posts(client, since_date, max_posts=config.limit_per_keyword)
            logger.info("  found %d official posts", len(posts))
            all_posts.extend(posts)

        # Keyword search
        for keyword in config.search_keywords:
            logger.info("Searching: %s", keyword)
            posts = await search_skin_posts(client, keyword, since_date, max_posts=config.limit_per_keyword)
            logger.info("  found %d posts for '%s'", len(posts), keyword)
            all_posts.extend(posts)

        # Hero-name–based search
        if config.search_heroes:
            heroes = load_hero_names()
            logger.info("Searching %d hero names ...", len(heroes))
            for hero in heroes:
                keyword = f"{hero} 皮肤"
                posts = await search_skin_posts(client, keyword, since_date, max_posts=1)
                all_posts.extend(posts)
                if not config.quiet:
                    sys.stderr.write(f"\r  searched {hero} ({len(all_posts)} posts found)")
                    sys.stderr.flush()
            if not config.quiet:
                sys.stderr.write("\n")

        # Deduplicate by mid
        seen: set[str] = set()
        unique_posts: list[WeiboPost] = []
        for p in all_posts:
            if p.mid and p.mid not in seen:
                seen.add(p.mid)
                unique_posts.append(p)
        all_posts = unique_posts

        logger.info("Total unique posts to crawl: %d", len(all_posts))

        # ── 2. Fetch comments for each post ──

        for i, post in enumerate(all_posts):
            if not config.quiet:
                sys.stderr.write(f"\r  [{i + 1}/{len(all_posts)}] fetching comments for {post.mid} ...")
                sys.stderr.flush()

            try:
                if config.hot_only:
                    comments = await fetch_hot_comments(client, post.mid, config.max_comments_per_post)
                else:
                    # Try hot first, fall back to all-comments
                    comments = await fetch_hot_comments(client, post.mid, config.max_comments_per_post)
                    if len(comments) < 20:
                        all_comments = await fetch_all_comments(client, post.mid, config.max_comments_per_post)
                        # Merge, preferring hot order
                        hot_users = {c["user"] for c in comments}
                        for c in all_comments:
                            if c["user"] not in hot_users:
                                comments.append(c)

                meaningful, low_q = filter_comments(comments, config.min_skin_signals)

                entry = CrawlEntry(
                    mid=post.mid,
                    title=_make_title(post.text) if post.text else f"post:{post.mid[:12]}",
                    total_comments=post.comments_count or len(comments),
                    meaningful=len(meaningful),
                    low_quality=low_q,
                    post=post,
                    comments=meaningful,
                )
                results.append(entry)

            except Exception:
                logger.exception("Failed to fetch comments for mid=%s", post.mid)
                # Still record the post with empty comments
                results.append(CrawlEntry(
                    mid=post.mid,
                    title=_make_title(post.text) if post.text else f"post:{post.mid[:12]}",
                    total_comments=post.comments_count or 0,
                    meaningful=0,
                    low_quality=0,
                    post=post,
                    comments=[],
                ))

        if not config.quiet:
            sys.stderr.write("\n")

    return results


def format_output(results: list[CrawlEntry]) -> list[dict]:
    """Convert results to the JSON-compatible list format matching existing data."""
    output: list[dict] = []
    for entry in results:
        output.append({
            "mid": entry.mid,
            "title": entry.title,
            "total_comments": entry.total_comments,
            "meaningful": entry.meaningful,
            "low_quality": entry.low_quality,
            "comments": entry.comments,
        })
    return output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Hero Skin Weibo Comment Crawler — fetch WZRY skin comments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # Post discovery
    g = p.add_argument_group("Post discovery")
    g.add_argument("--mids", nargs="*", default=[],
                   help="Specific Weibo post MIDs to fetch comments for")
    g.add_argument("--official", action="store_true",
                   help="Fetch skin posts from @王者荣耀 official account")
    g.add_argument("--search", nargs="*", default=[],
                   help="Keyword search queries (e.g. '赵云 皮肤')")
    g.add_argument("--search-heroes", action="store_true",
                   help="Search for every hero name + 皮肤 from the skins database")

    # Filters
    f = p.add_argument_group("Filters")
    f.add_argument("--days", type=int, default=7,
                   help="Only fetch posts from the last N days (default: 7, 0=all)")
    f.add_argument("--limit", type=int, default=5, dest="limit_per_keyword",
                   help="Max posts per keyword/official source (default: 5)")
    f.add_argument("--hot-only", action="store_true",
                   help="Only fetch hot comments, skip all-comment pagination")
    f.add_argument("--max-comments", type=int, default=300,
                   help="Max comments to fetch per post (default: 300)")
    f.add_argument("--min-signals", type=int, default=0,
                   help="Min skin signal words per comment to keep (0=keep all non-spam)")

    # Output
    o = p.add_argument_group("Output")
    o.add_argument("--output", "-o", default="",
                   help="Output JSON file path (default: auto-dated in data/weibo_comments/)")
    o.add_argument("--quiet", "-q", action="store_true",
                   help="Suppress progress output")

    return p.parse_args()


async def main() -> None:
    args = parse_args()

    if not args.quiet:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    else:
        logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    # Validate: at least one discovery method
    if not any([args.mids, args.official, args.search, args.search_heroes]):
        print("Error: specify at least one discovery method: --mids, --official, --search, --search-heroes",
              file=sys.stderr)
        sys.exit(1)

    config = CrawlConfig(
        mids=args.mids,
        official=args.official,
        search_keywords=args.search,
        search_heroes=args.search_heroes,
        days=args.days,
        limit_per_keyword=args.limit_per_keyword,
        hot_only=args.hot_only,
        max_comments_per_post=args.max_comments,
        min_skin_signals=args.min_signals,
        output=args.output,
        quiet=args.quiet,
    )

    start = time.monotonic()
    try:
        results = await crawl(config)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    elapsed = time.monotonic() - start

    # ── Output ──
    output_path = Path(args.output) if args.output else OUTPUT_DIR / _output_filename()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    formatted = format_output(results)
    output_path.write_text(
        json.dumps(formatted, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # ── Summary ──
    total_comments = sum(r.meaningful for r in results)
    total_low_q = sum(r.low_quality for r in results)
    print(f"\n{'='*50}")
    print(f"Crawl complete  —  {len(results)} posts, {total_comments} comments ({total_low_q} filtered)")
    print(f"Elapsed: {elapsed:.1f}s")
    print(f"Output:  {output_path}")
    print(f"{'='*50}")


if __name__ == "__main__":
    asyncio.run(main())

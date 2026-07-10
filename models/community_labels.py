"""Community Engagement Labels — turn social/community chatter into a 0-100
emotional-premium signal for skins.

This is the *Community Engagement* evidence source from the ground-truth
strategy (``progress/2026-07-09-ground-truth-data-collection.md``, §3).  It
complements the LLM-expert gold standard with a **data-derived** signal: how
excited / positive / engaged the player community is about a skin, as measured
from Weibo comment dumps (and, by extension, Bilibili / forums / Douyin later).

Design notes
------------
* **Two scoring paths**, like the expert labeler but text-only:

  * ``rule_score`` — fully reproducible, no API.  Filters low-quality comments
    (reusing the crawler's ``filter_comments``), then combines a
    sentiment sub-score (positive/negative keyword ratio) with an engagement
    sub-score (like-weighted, log-scaled).  This is the **default** and the
    canonical community signal.
  * ``--use-llm`` — an *optional* LLM enrichment layer.  The same AutoDL
    endpoint used by :class:`~models.llm_expert_labeler.LLMExpertLabeler` is
    fed the raw comments + the rule sub-scores and asked to refine the
    ``community_premium`` with a ``rationale`` + ``confidence``.  Kept off by
    default so community labels stay cheap and reproducible.

* **Best-effort skin linkage**: Weibo dumps are *post-level* (``mid``), not
  skin-keyed.  We build a keyword index from ``CrawlerManager.list_skins()`` and
  match each post to a ``skin_key`` where a hero/skin name appears in the post
  title or comments.  Unmatched posts keep ``skin_key=None`` (corpus-level
  signal).  This matcher lights up automatically as new crawls target specific
  skins — see the report's ``n_skin_matched`` / ``n_corpus_level`` split.

* **No FeatureStore writes** (mirrors the expert labeler — export only).  Future
  ``models/ground_truth.py`` fuses the JSONL outputs.

Caveat — leakage
----------------
Community labels are text-derived and do **not** leak VLM L3 features.  However,
when ``--use-llm`` is on, the same model family that produces the expert label
may also produce these; the two then correlate by construction.  Keep
``--use-llm`` OFF (the default) for an independent community evidence source.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
import time
from pathlib import Path
from typing import Any

import httpx
from loguru import logger
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from crawlers.manager import CrawlerManager, SkinSummary
from crawlers.weibo_skin_comment_crawler import filter_comments
from vlm.cache import CacheManager
from vlm.config import get_settings

# ── Sentiment keywords (Chinese community slang) ──────────────────────────────
# Mirrors the intent of the strategy doc §3 "Weibo Excitement Score".

POSITIVE_KEYWORDS = [
    "值了", "必买", "绝美", "炫酷", "绝绝子", "yyds", "好看", "帅",
    "美", "炫", "期待", "入手", "买", "爱了", "喜欢", "冲", "氪",
    "良心", "封神", "神作", "顶", "可以",
]

NEGATIVE_KEYWORDS = [
    "不值", "丑", "后悔", "智商税", "坑", "割韭菜", "劝退", "烂",
    "敷衍", "差", "退钱", "糊", "无聊", "无聊", "雷",
]

_POS_RE = [re.compile(re.escape(k)) for k in POSITIVE_KEYWORDS]
_NEG_RE = [re.compile(re.escape(k)) for k in NEGATIVE_KEYWORDS]


# ── Output schema ─────────────────────────────────────────────────────────────

class CommunityLabel(BaseModel):
    """Community engagement signal for one Weibo post (optionally skin-linked)."""

    skin_key: str | None = None       # None = corpus-level (no skin match)
    mid: str = ""                      # Weibo post id
    title: str = ""
    community_premium: float = Field(default=0.0, ge=0.0, le=100.0)
    engagement: float = Field(default=0.0, ge=0.0, le=100.0)   # like-weighted
    sentiment: float = Field(default=0.0, ge=0.0, le=100.0)   # pos-neg ratio
    confidence: int = Field(default=3, ge=1, le=5)
    rationale: str = ""
    n_comments: int = 0
    n_meaningful: int = 0

    # provenance
    model: str = "qwen3-vl-plus"
    source: str = "community_engagement"
    elapsed_s: float = 0.0
    cached: bool = False
    error: str | None = None

    def to_record(self) -> dict[str, Any]:
        """Flatten to a row suitable for CSV/JSONL export."""
        return self.model_dump()


# ── Skin keyword index (best-effort post → skin_key linkage) ──────────────────

def build_skin_index(skins: list[SkinSummary]) -> list[dict[str, Any]]:
    """Build a matchable index from official skins.

    Each entry carries the ``skin_key`` plus the substrings we test against
    post text.  We only match on **meaningful** tokens — full hero name,
    full skin name, and skin-name 2-3 char windows — and never on single
    chars (too noisy).  2-char windows are the *fallback* tier; a full
    hero/skin name always outranks them in :func:`match_skin`.
    """
    index: list[dict[str, Any]] = []
    for s in skins:
        tokens: list[str] = []
        if s.hero_name and len(s.hero_name) >= 2:
            tokens.append(s.hero_name)          # strong token (full hero name)
        if s.skin_name and len(s.skin_name) >= 2:
            tokens.append(s.skin_name)          # strong token (full skin name)
            # overlapping 2-3 char windows catch skin names embedded in comments
            for w in (2, 3):
                if len(s.skin_name) >= w:
                    tokens.extend(
                        s.skin_name[i : i + w]
                        for i in range(len(s.skin_name) - w + 1)
                    )
        # de-dup, drop single chars, keep hero/skin names first (longest first)
        tokens = sorted({t for t in tokens if len(t) >= 2}, key=len, reverse=True)
        index.append(
            {"skin_key": s.source_key, "tokens": tokens,
             "display": f"{s.hero_name}-{s.skin_name}"}
        )
    return index


def match_skin(text: str, index: list[dict[str, Any]]) -> str | None:
    """Return the best-matching ``skin_key`` for ``text``, or ``None``.

    Greedy: the longest matched token wins (a full skin name beats a hero
    name beats a 2-char window).  Deterministic — no randomness.
    """
    if not text or not index:
        return None
    best_key: str | None = None
    best_len = 0
    for entry in index:
        for tok in entry["tokens"]:
            if tok in text and len(tok) > best_len:
                best_len = len(tok)
                best_key = entry["skin_key"]
    return best_key


# ── Labeler ───────────────────────────────────────────────────────────────────

class CommunitySignalLabeler:
    """Score community excitement for Weibo posts, optionally skin-linked.

    Example::

        labeler = CommunitySignalLabeler()
        labels = await labeler.label_batch(posts, skin_index=idx)
        # labels[0].community_premium, labels[0].skin_key, ...
    """

    def __init__(
        self,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 800,
        timeout: float | None = None,
        cache: CacheManager | None = None,
        max_retries: int = 3,
    ) -> None:
        self.settings = get_settings()
        self.model = model or self.settings.expert_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout or self.settings.l3_timeout
        self.cache = cache or CacheManager()
        self.max_retries = max_retries
        self._client: AsyncOpenAI | None = None

    # ── client ──

    def _get_client(self) -> AsyncOpenAI | None:
        if self._client is not None:
            return self._client
        if not self.settings.autodl_token:
            logger.warning("No AUTODL_TOKEN — LLM enrichment disabled")
            return None
        self._client = AsyncOpenAI(
            api_key=self.settings.autodl_token,
            base_url=self.settings.autodl_base_url,
            http_client=httpx.AsyncClient(trust_env=False),
        )
        return self._client

    # ── text hashing (comments are text, not images) ──

    @staticmethod
    def _content_hash(mid: str, title: str, comments: list[dict]) -> str:
        blob = mid + "|" + title + "|" + json.dumps(
            comments, ensure_ascii=False, sort_keys=True
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    # ── rule-based scoring (reproducible, no API) ──

    def rule_score(self, comments: list[dict]) -> dict[str, Any]:
        """Compute sentiment + engagement sub-scores from raw comments.

        Returns a dict with ``sentiment``, ``engagement``, ``community_premium``,
        ``n_comments``, ``n_meaningful``.
        """
        meaningful, low = filter_comments(comments, min_signals=0)
        n = len(comments)
        n_mean = len(meaningful)

        # sentiment: positive vs negative keyword ratio over meaningful comments
        pos = neg = 0
        for c in meaningful:
            text = c.get("text", "") or ""
            pos += sum(1 for r in _POS_RE if r.search(text))
            neg += sum(1 for r in _NEG_RE if r.search(text))
        if pos + neg == 0:
            sentiment = 50.0          # neutral when no signal
        else:
            sentiment = round(100.0 * pos / (pos + neg), 1)

        # engagement: like-weighted, log-scaled, normalized to 0-100
        if n_mean > 0:
            total_likes = sum(
                int(c.get("like_count", 0) or 0) for c in meaningful
            )
            mean_likes = total_likes / n_mean
            # log scale compresses viral outliers; log1p keeps 0→0
            engagement = round(100.0 * math.log1p(mean_likes) / math.log1p(5000), 1)
            engagement = max(0.0, min(100.0, engagement))
        else:
            engagement = 0.0

        # initial fusion weights (tunable — documented in strategy doc §3)
        community_premium = round(0.6 * sentiment + 0.4 * engagement, 1)

        return {
            "sentiment": sentiment,
            "engagement": engagement,
            "community_premium": community_premium,
            "n_comments": n,
            "n_meaningful": n_mean,
        }

    # ── single-post labeling ──

    async def label_post(
        self,
        mid: str,
        title: str,
        comments: list[dict],
        skin_index: list[dict[str, Any]] | None = None,
        force: bool = False,
        use_llm: bool = False,
    ) -> CommunityLabel:
        """Label one post.  Returns a :class:`CommunityLabel` (``error`` set on failure)."""
        content_hash = self._content_hash(mid, title, comments)
        # CacheManager.get/set hash a *Path*; wrap our logical key as a Path
        # under a dedicated namespace so it never collides with image keys.
        cache_path = Path(f"community/{mid}")

        if not force:
            cached = self.cache.get(cache_path, "community", content_hash)
            if cached is not None:
                try:
                    return CommunityLabel(**{**cached, "cached": True})
                except Exception:
                    pass  # fall through to re-label

        rule = self.rule_score(comments)
        skin_key = match_skin(title or "", skin_index)
        # also try matching against the most-liked comment text
        if skin_key is None and skin_index:
            top = max(comments, key=lambda c: int(c.get("like_count", 0) or 0),
                      default=None)
            if top:
                skin_key = match_skin(top.get("text", ""), skin_index)

        label = CommunityLabel(
            skin_key=skin_key,
            mid=mid,
            title=title,
            community_premium=rule["community_premium"],
            engagement=rule["engagement"],
            sentiment=rule["sentiment"],
            confidence=3,
            n_comments=rule["n_comments"],
            n_meaningful=rule["n_meaningful"],
            rationale="rule-based (no LLM)",
        )

        if use_llm:
            enriched = await self._llm_enrich(mid, title, comments, rule, skin_key)
            if enriched is not None:
                label = enriched

        self.cache.set(cache_path, "community", label.to_record(), content_hash)
        return label

    # ── optional LLM enrichment ──

    async def _llm_enrich(
        self,
        mid: str,
        title: str,
        comments: list[dict],
        rule: dict[str, Any],
        skin_key: str | None,
    ) -> CommunityLabel | None:
        client = self._get_client()
        if client is None:
            return None

        # sample up to 40 most-liked comments to stay within token budget
        sampled = sorted(
            comments, key=lambda c: int(c.get("like_count", 0) or 0), reverse=True
        )[:40]
        comment_block = "\n".join(
            f"- (赞{int(c.get('like_count', 0) or 0)}) {c.get('text', '')}"
            for c in sampled
        )

        system_prompt = (
            "你是王者荣耀皮肤社区情绪分析师。给定一条官方微博帖子的玩家评论，"
            "请评估该帖子反映的「社区情绪溢价」——玩家对相关内容（皮肤/活动）的"
            "兴奋度、正面情绪与互动热度，范围 0-100。仅依据评论理性判断，输出严格 JSON。"
        )
        user_prompt = (
            f"# 帖子标题\n{title}\n\n"
            f"# 规则基线分（仅供参考，你可推翻）\n"
            f"- sentiment(正负面情绪比): {rule['sentiment']}\n"
            f"- engagement(点赞加权互动): {rule['engagement']}\n"
            f"- 规则综合 community_premium: {rule['community_premium']}\n"
            f"- 有效评论数: {rule['n_meaningful']}/{rule['n_comments']}\n\n"
            f"# 高赞评论样本\n{comment_block}\n\n"
            "请评估社区情绪溢价，输出 JSON：\n"
            '{\n'
            '  "community_premium": 0-100,\n'
            '  "confidence": 1-5,\n'
            '  "rationale": "一句话判断理由"\n'
            "}\n"
        )

        last_err: str | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                t0 = time.perf_counter()
                resp = await client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    timeout=self.timeout,
                )
                elapsed = round(time.perf_counter() - t0, 2)
                content = resp.choices[0].message.content or ""
                parsed = _parse_jsonish(content) or {}
                premium = float(parsed.get("community_premium",
                                           rule["community_premium"]))
                premium = max(0.0, min(100.0, premium))
                return CommunityLabel(
                    skin_key=skin_key,
                    mid=mid,
                    title=title,
                    community_premium=round(premium, 1),
                    engagement=rule["engagement"],
                    sentiment=rule["sentiment"],
                    confidence=int(parsed.get("confidence", 3)),
                    rationale=str(parsed.get("rationale", "")),
                    n_comments=rule["n_comments"],
                    n_meaningful=rule["n_meaningful"],
                    elapsed_s=elapsed,
                )
            except Exception as exc:  # noqa: BLE001 — retry on any API error
                last_err = str(exc)
                logger.warning(
                    f"Community LLM attempt {attempt}/{self.max_retries} "
                    f"failed for {mid}: {exc}"
                )
                await asyncio.sleep(min(2 ** attempt, 16))

        logger.error(f"Community LLM enrichment failed for {mid}: {last_err}")
        return None

    # ── mini-batch labeling ──

    async def label_batch(
        self,
        posts: list[dict[str, Any]],
        skin_index: list[dict[str, Any]] | None = None,
        batch_size: int = 10,
        concurrency: int = 3,
        force: bool = False,
        use_llm: bool = False,
    ) -> list[CommunityLabel]:
        """Label a list of posts in mini-batches.

        Args:
            posts: each is ``{"mid", "title", "comments": [...]}``.
            skin_index: output of :func:`build_skin_index` (or ``None`` for
                corpus-level only).
            batch_size: posts per chunk.
            concurrency: max in-flight coroutines *within* a batch.

        Returns one :class:`CommunityLabel` per post, in input order.
        """
        if not posts:
            return []
        labels: list[CommunityLabel | None] = [None] * len(posts)
        sem = asyncio.Semaphore(concurrency)

        async def _worker(idx: int, post: dict[str, Any]) -> None:
            async with sem:
                labels[idx] = await self.label_post(
                    mid=post.get("mid", ""),
                    title=post.get("title", ""),
                    comments=post.get("comments", []),
                    skin_index=skin_index,
                    force=force,
                    use_llm=use_llm,
                )

        for start in range(0, len(posts), batch_size):
            chunk = posts[start : start + batch_size]
            idxs = range(start, start + len(chunk))
            logger.info(
                f"Community mini-batch {start // batch_size + 1}: "
                f"posts {start}-{start + len(chunk) - 1}/{len(posts)}"
            )
            await asyncio.gather(
                *(_worker(i, p) for i, p in zip(idxs, chunk))
            )

        return [
            lbl if lbl is not None else CommunityLabel(error="missing")
            for lbl in labels
        ]

    # ── calibration helpers (mirror the expert labeler) ──

    @staticmethod
    def aggregate(labels: list[CommunityLabel]) -> dict[str, float]:
        """Mean of each score across labels."""
        ok = [l for l in labels if l.error is None]
        n = max(len(ok), 1)
        out = {
            "community_premium": round(sum(l.community_premium for l in ok) / n, 2),
            "engagement": round(sum(l.engagement for l in ok) / n, 2),
            "sentiment": round(sum(l.sentiment for l in ok) / n, 2),
        }
        return out


def _parse_jsonish(text: str) -> dict | None:
    """Best-effort JSON extraction (same intent as ``vlm.utils.parse_jsonish``).

    Local copy to avoid pulling VLM utils into a text-only module; handles
    fenced blocks and trailing prose.
    """
    if not text:
        return None
    text = text.strip()
    # strip markdown fences
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    try:
        return json.loads(text)
    except Exception:
        pass
    # find first {...} span
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None

"""LLM-as-Expert labeler — use AutoDL ``qwen3-vl-plus`` as a *standalone domain
expert* to produce the **Expert-Labeled Gold Standard** for emotional premium.

This is intentionally independent of the VLM feature pipeline.  It talks to
the same OpenAI-compatible AutoDL endpoint that ``vlm.l3_semantic`` uses, but
frames the model as a **rating expert** (not a semantic analyzer) and returns
the structured 5-dimension premium label that the ground-truth strategy
document defines for human experts:

    overall_premium (0-100)
    aesthetic / showing_off / belonging / collection / surprise (0-100)
    confidence (1-5), rationale

Design notes
------------
* **Deterministic**: ``temperature=0`` so the expert is reproducible.  Re-running
  the same image returns the same label (validated by the Layer-1 cache-stability
  check in the eval strategy).
* **Cached**: results are stored in the shared ``CacheManager`` under level
  ``"expert"``, keyed by image content-hash, so a re-label is free.
* **Self-contained**: only depends on ``vlm`` config/util/cache helpers — it does
  NOT import the feature pipeline or run L1/L2/L3.
* **Mini-batch friendly**: ``label_batch`` slices work into ``batch_size`` chunks
  and applies per-request concurrency + retry-with-backoff so a large corpus can
  be labeled incrementally without blowing rate limits or losing progress.

Caveat — leakage
----------------
The VLM ``L3`` tier is *also* ``gpt-5.4-mini``.  If you feed this expert's labels
as ground truth to train an XGBoost model whose features include ``L3`` outputs,
you get label/feature leakage.  Use LLM-expert labels for: (1) bulk corpus labeling
where scale beats purity, (2) a *target* to measure human-expert agreement against,
(3) cross-checking the rule engine.  For the human-anchored gold standard, keep a
small human-labeled anchor set and calibrate the LLM-expert against it.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

import httpx
from loguru import logger
from openai import AsyncOpenAI
from pydantic import BaseModel, Field, field_validator

from vlm.cache import CacheManager
from vlm.config import get_settings
from vlm.utils import encode_image, image_content_hash, parse_jsonish

# ── Canonical tier order for monotonicity checks ────────────────────────────
# The crawler ``quality`` field holds variants (传说 / 传说限定 / 珍品传说 …).
# Map them onto the 6 canonical tiers used by the eval strategy.
CANONICAL_TIERS = ["伴生", "勇者", "史诗", "传说", "无双", "荣耀典藏"]

_TIER_ALIASES: dict[str, str] = {
    "伴生": "伴生",
    "勇者": "勇者",
    "史诗": "史诗",
    "传说": "传说",
    "传说限定": "传说",
    "珍品传说": "传说",
    "无双": "无双",
    "无双限定": "无双",
    "荣耀典藏": "荣耀典藏",
}


def canonical_tier(quality: str) -> str:
    """Normalize a raw crawler ``quality`` string to a canonical tier."""
    return _TIER_ALIASES.get((quality or "").strip(), "史诗")


# ── Output schema ───────────────────────────────────────────────────────────

class ExpertLabel(BaseModel):
    """One expert's emotional-premium judgment for a single skin."""

    skin_key: str = ""
    overall_premium: float = Field(default=0.0, ge=0.0, le=100.0)
    aesthetic: float = Field(default=0.0, ge=0.0, le=100.0)
    showing_off: float = Field(default=0.0, ge=0.0, le=100.0)
    belonging: float = Field(default=0.0, ge=0.0, le=100.0)
    collection: float = Field(default=0.0, ge=0.0, le=100.0)
    surprise: float = Field(default=0.0, ge=0.0, le=100.0)
    confidence: int = Field(default=3, ge=1, le=5)
    rationale: str = ""
    dimension_rationale: dict[str, str] = Field(default_factory=dict)

    # provenance
    model: str = "qwen3-vl-plus"
    source: str = "llm_expert"
    elapsed_s: float = 0.0
    cached: bool = False
    error: str | None = None

    @field_validator("overall_premium", "aesthetic", "showing_off",
                     "belonging", "collection", "surprise")
    @classmethod
    def _round(cls, v: float) -> float:
        return round(float(v), 1)

    def to_record(self) -> dict[str, Any]:
        """Flatten to a row suitable for CSV/JSONL export."""
        return self.model_dump()


# ── Prompts ─────────────────────────────────────────────────────────────────

EXPERT_SYSTEM_PROMPT = """\
你是王者荣耀皮肤「情绪溢价」评估专家，拥有十年游戏皮肤策划与社区观察经验。

# 任务
给定一张皮肤壁纸图片与官方属性，请评估该皮肤的「情绪溢价」——玩家愿意为其\
视觉/情感价值（而非功能价值）额外支付的意愿强度，范围 0-100。

# 五个评估维度（均 0-100）
- aesthetic 审美：模型精细度、配色和谐、构图、特效与材质质感带来的纯粹视觉享受
- showing_off 炫耀：皮肤作为社交货币/身份象征的展示价值（稀有度、辨识度、排面）
- belonging 归属：是否呼应玩家对英雄/阵营/文化圈层的身份认同与情感羁绊
- collection 收藏：在系列/图鉴中的收藏完整度与绝版/限定稀缺性
- surprise 惊喜：新颖度、Unexpected delight、打破套路的创意冲击力

# 要求
1. 仅依据图片与给出的官方属性理性判断，不要被品牌光环带偏
2. overall_premium 是对五个维度的综合判断（审美通常权重最高，惊喜通常最弱，\
   但请按你对该皮肤的真实评估给出，而非机械加权）
3. confidence 表示你对该判断的把握（1=很不确定，5=非常确定）
4. 输出严格 JSON，不要任何额外文本
"""

EXPERT_USER_TEMPLATE = """\
# 官方属性
- 英雄: {hero_name}
- 皮肤: {skin_name}
- 稀有度(quality): {quality}
- 获取方式: {acquire_method}
- 定价文本: {price_text}
- 上线日期: {online_date}
- 官方简介: {intro}

{radar_block}

请评估这张皮肤壁纸的情绪溢价，输出 JSON：
{{
  "overall_premium": 0-100,
  "aesthetic": 0-100,
  "showing_off": 0-100,
  "belonging": 0-100,
  "collection": 0-100,
  "surprise": 0-100,
  "confidence": 1-5,
  "rationale": "整体判断的一句话理由",
  "dimension_rationale": {{
    "aesthetic": "...", "showing_off": "...", "belonging": "...",
    "collection": "...", "surprise": "..."
  }}
}}
"""


# ── Labeler ─────────────────────────────────────────────────────────────────

class LLMExpertLabeler:
    """Use ``gpt-5.4-mini`` (AutoDL) as a standalone expert rater.

    Example::

        labeler = LLMExpertLabeler()
        label = await labeler.label_skin(
            skin_key="0001-56304",
            image_path=Path("data/.../wallpaper.jpg"),
            context={"hero_name": "...", "skin_name": "...", "quality": "史诗",
                     "acquire_method": "...", "price_text": "...",
                     "online_date": "...", "intro": "..."},
        )
        # label.overall_premium, label.aesthetic, ...
    """

    def __init__(
        self,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1200,
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
            logger.warning("No AUTODL_TOKEN — LLM expert disabled")
            return None
        self._client = AsyncOpenAI(
            api_key=self.settings.autodl_token,
            base_url=self.settings.autodl_base_url,
            http_client=httpx.AsyncClient(trust_env=False),
        )
        return self._client

    # ── context builder ──

    @staticmethod
    def build_context(
        *,
        hero_name: str,
        skin_name: str,
        quality: str,
        acquire_method: str = "",
        price_text: str | None = None,
        online_date: str = "",
        intro: str = "",
        vlm_radar: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Assemble the ``context`` dict consumed by :meth:`label_skin`.

        ``vlm_radar`` (optional) mirrors what a human expert sees as a radar
        chart — the existing VLM L1/L2/L3 outputs.  Pass it only when you
        *want* the LLM-expert to see the current model's read (useful for
        cross-checking, but introduces coupling — see module docstring).
        """
        ctx: dict[str, Any] = {
            "hero_name": hero_name,
            "skin_name": skin_name,
            "quality": quality,
            "acquire_method": acquire_method or "未知",
            "price_text": price_text or "未知",
            "online_date": online_date or "未知",
            "intro": (intro or "无").strip(),
        }
        if vlm_radar:
            ctx["_radar_block"] = (
                "# 现有视觉分析(供参考，你可推翻)\n"
                + json.dumps(vlm_radar, ensure_ascii=False)
            )
        return ctx

    # ── single-skin labeling ──

    async def label_skin(
        self,
        skin_key: str,
        image_path: str | Path,
        context: dict[str, Any],
        force: bool = False,
    ) -> ExpertLabel:
        """Label one skin.  Returns an :class:`ExpertLabel` (``error`` set on failure)."""
        image_path = Path(image_path)
        content_hash = image_content_hash(image_path)

        # cache hit
        if not force:
            cached = self.cache.get(image_path, "expert", content_hash)
            if cached is not None:
                try:
                    return ExpertLabel(**{**cached, "cached": True})
                except Exception:
                    pass  # fall through to re-label

        client = self._get_client()
        if client is None:
            return ExpertLabel(skin_key=skin_key, error="no_autodl_token")

        image_b64 = encode_image(image_path)
        radar_block = context.get("_radar_block", "（无）")
        user_prompt = EXPERT_USER_TEMPLATE.format(
            hero_name=context.get("hero_name", ""),
            skin_name=context.get("skin_name", ""),
            quality=context.get("quality", ""),
            acquire_method=context.get("acquire_method", "未知"),
            price_text=context.get("price_text", "未知"),
            online_date=context.get("online_date", "未知"),
            intro=context.get("intro", "无"),
            radar_block=radar_block,
        )

        last_err: str | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                t0 = time.perf_counter()
                resp = await client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": EXPERT_SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": user_prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{image_b64}"
                                    },
                                },
                            ],
                        },
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    timeout=self.timeout,
                )
                elapsed = round(time.perf_counter() - t0, 2)
                content = resp.choices[0].message.content or ""
                parsed = parse_jsonish(content) or {}
                label = ExpertLabel(
                    skin_key=skin_key,
                    overall_premium=float(parsed.get("overall_premium", 0)),
                    aesthetic=float(parsed.get("aesthetic", 0)),
                    showing_off=float(parsed.get("showing_off", 0)),
                    belonging=float(parsed.get("belonging", 0)),
                    collection=float(parsed.get("collection", 0)),
                    surprise=float(parsed.get("surprise", 0)),
                    confidence=int(parsed.get("confidence", 3)),
                    rationale=str(parsed.get("rationale", "")),
                    dimension_rationale=parsed.get("dimension_rationale", {}) or {},
                    elapsed_s=elapsed,
                )
                self.cache.set(image_path, "expert", label.to_record(), content_hash)
                return label
            except Exception as exc:  # noqa: BLE001 — retry on any API error
                last_err = str(exc)
                logger.warning(
                    f"Expert label attempt {attempt}/{self.max_retries} failed "
                    f"for {skin_key}: {exc}"
                )
                await asyncio.sleep(min(2 ** attempt, 16))

        return ExpertLabel(skin_key=skin_key, error=last_err or "unknown_error")

    # ── mini-batch labeling ──

    async def label_batch(
        self,
        items: list[dict[str, Any]],
        batch_size: int = 10,
        concurrency: int = 3,
        force: bool = False,
    ) -> list[ExpertLabel]:
        """Label a list of skins in mini-batches.

        Args:
            items: each is ``{"skin_key", "image_path", "context"}``.
            batch_size: number of skins per chunk (controls cost/visibility
                granularity and progress reporting).
            concurrency: max in-flight API calls *within* a batch.

        Returns one :class:`ExpertLabel` per item, in input order.
        """
        if not items:
            return []
        labels: list[ExpertLabel | None] = [None] * len(items)
        sem = asyncio.Semaphore(concurrency)

        async def _worker(idx: int, item: dict[str, Any]) -> None:
            async with sem:
                labels[idx] = await self.label_skin(
                    skin_key=item["skin_key"],
                    image_path=item["image_path"],
                    context=item["context"],
                    force=force,
                )

        for start in range(0, len(items), batch_size):
            chunk = items[start : start + batch_size]
            idxs = range(start, start + len(chunk))
            logger.info(
                f"Expert mini-batch {start // batch_size + 1}: "
                f"skins {start}-{start + len(chunk) - 1}/{len(items)}"
            )
            await asyncio.gather(
                *(_worker(i, it) for i, it in zip(idxs, chunk))
            )

        return [lbl if lbl is not None else ExpertLabel(error="missing") for lbl in labels]

    # ── calibration helpers (mirror the eval-strategy checks) ──

    @staticmethod
    def tier_monotonicity(labels: list[ExpertLabel], qualities: list[str]) -> float:
        """Spearman-style monotonicity: higher canonical tier → higher premium.

        Returns Spearman ρ over (tier_rank, overall_premium).  Pass-through of
        scipy is avoided so this stays dependency-light; uses a rank correlation
        that tolerates ties reasonably for small samples.
        """
        try:
            from scipy.stats import spearmanr  # type: ignore
        except Exception:
            logger.warning("scipy unavailable — monotonicity check skipped")
            return float("nan")
        tier_rank = [CANONICAL_TIERS.index(canonical_tier(q)) + 1 for q in qualities]
        prem = [l.overall_premium for l in labels]
        rho, _ = spearmanr(tier_rank, prem)
        return float(rho)

    @staticmethod
    def aggregate_dimensions(labels: list[ExpertLabel]) -> dict[str, float]:
        """Mean of each dimension across labels (for AHP-alignment review)."""
        dims = ["aesthetic", "showing_off", "belonging", "collection", "surprise"]
        out: dict[str, float] = {}
        n = max(len(labels), 1)
        for d in dims:
            out[d] = round(sum(getattr(l, d) for l in labels) / n, 2)
        out["overall_premium"] = round(
            sum(l.overall_premium for l in labels) / n, 2
        )
        return out

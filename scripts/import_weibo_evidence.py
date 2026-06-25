"""Import Weibo skin comments as market evidence for one known skin."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.market_signal_repository import MarketSignalRepository  # noqa: E402
from data.skin_repository import DEFAULT_DB_PATH, SkinRepository  # noqa: E402
from feature_engineering.features import MarketValidationSignals  # noqa: E402


ASPECT_KEYWORDS = {
    "visual": (
        "皮肤",
        "特效",
        "建模",
        "原画",
        "海报",
        "好看",
        "帅",
        "美",
        "丑",
        "造型",
        "设计",
        "颜色",
    ),
    "feel": (
        "手感",
        "打击感",
        "局内",
        "普攻",
        "技能",
        "音效",
        "回城",
        "出场",
        "动画",
    ),
    "craftsmanship": (
        "品质",
        "传说",
        "史诗",
        "勇者",
        "无双",
        "荣耀典藏",
        "限定",
        "精品",
        "敷衍",
        "缝合",
    ),
    "collection": (
        "限定",
        "返场",
        "绝版",
        "收藏",
        "入手",
        "错过",
        "典藏",
        "联动",
        "ip",
        "IP",
    ),
    "value": (
        "价格",
        "点券",
        "贵",
        "便宜",
        "值不值",
        "值",
        "性价比",
        "折扣",
        "直售",
        "抽",
        "保底",
    ),
    "purchase_intent": (
        "买",
        "入手",
        "必冲",
        "冲",
        "预定",
        "不买",
        "别买",
        "买不起",
        "等返场",
        "氪",
    ),
}

POSITIVE_WORDS = (
    "好看",
    "帅",
    "美",
    "喜欢",
    "期待",
    "必买",
    "必冲",
    "入手",
    "值",
    "香",
    "爱了",
    "高级",
    "顶",
    "绝",
)

NEGATIVE_WORDS = (
    "丑",
    "难看",
    "敷衍",
    "不值",
    "贵",
    "垃圾",
    "拉",
    "失望",
    "别买",
    "不买",
    "买不起",
    "烂",
    "一般",
)

PURCHASE_POSITIVE_WORDS = ("必买", "必冲", "入手", "买", "冲", "预定", "氪")
PURCHASE_NEGATIVE_WORDS = ("不买", "别买", "买不起", "退了", "不冲")


@dataclass(slots=True)
class CommentEvidence:
    external_id: str
    text: str
    title: str
    author: str | None
    published_at: str | None
    url: str | None
    metrics: dict[str, Any]
    aspect_tags: list[str]
    raw_json: dict[str, Any]


@dataclass(slots=True)
class AggregationStats:
    imported: int = 0
    discussion_count: int = 0
    marketing_volume: int = 0
    sentiment_values: list[float] = field(default_factory=list)
    aspect_values: dict[str, list[float]] = field(
        default_factory=lambda: {
            "visual": [],
            "feel": [],
            "craftsmanship": [],
            "collection": [],
            "value": [],
            "purchase_intent": [],
        }
    )


def detect_aspect_tags(text: str) -> list[str]:
    return [
        tag
        for tag, keywords in ASPECT_KEYWORDS.items()
        if any(keyword in text for keyword in keywords)
    ]


def sentiment_value(text: str) -> float | None:
    positive = sum(1 for word in POSITIVE_WORDS if word in text)
    negative = sum(1 for word in NEGATIVE_WORDS if word in text)
    if negative:
        positive = max(0, positive - negative)
    total = positive + negative
    if total == 0:
        return None
    return (positive - negative + total) / (2 * total)


def purchase_intent_value(text: str) -> float | None:
    positive = sum(1 for word in PURCHASE_POSITIVE_WORDS if word in text)
    negative = sum(1 for word in PURCHASE_NEGATIVE_WORDS if word in text)
    if negative:
        positive = max(0, positive - negative)
    total = positive + negative
    if total == 0:
        return None
    return (positive - negative + total) / (2 * total)


def iter_comment_evidence(payload: list[dict[str, Any]]) -> list[CommentEvidence]:
    evidence: list[CommentEvidence] = []
    for post_index, post in enumerate(payload):
        mid = str(post.get("mid") or post_index)
        title = str(post.get("title") or mid)
        post_meta = post.get("post") if isinstance(post.get("post"), dict) else {}
        author = post_meta.get("user_name") or "王者荣耀"
        url = f"https://m.weibo.cn/detail/{mid}" if post.get("mid") else None
        for comment_index, comment in enumerate(post.get("comments") or []):
            text = str(comment.get("text") or "").strip()
            if not text:
                continue
            external_id = f"{mid}:{comment_index}"
            like_count = int(comment.get("like_count") or 0)
            reply_count = int(comment.get("total_number") or 0)
            tags = detect_aspect_tags(text)
            evidence.append(
                CommentEvidence(
                    external_id=external_id,
                    text=text,
                    title=title,
                    author=author,
                    published_at=comment.get("created_at"),
                    url=url,
                    metrics={
                        "comment_count": 1,
                        "like_count": like_count,
                        "reply_count": reply_count,
                    },
                    aspect_tags=tags,
                    raw_json={
                        "post_mid": post.get("mid"),
                        "post_title": title,
                        "comment": comment,
                    },
                )
            )
    return evidence


def aggregate_comment_signals(
    evidence: list[CommentEvidence],
    *,
    min_aspect_comments: int = 3,
) -> MarketValidationSignals:
    stats = AggregationStats()
    for item in evidence:
        stats.imported += 1
        stats.discussion_count += 1 + int(item.metrics.get("reply_count") or 0)
        stats.marketing_volume += int(item.metrics.get("like_count") or 0)

        sentiment = sentiment_value(item.text)
        if sentiment is not None:
            stats.sentiment_values.append(sentiment)
            for tag in item.aspect_tags:
                if tag != "purchase_intent":
                    stats.aspect_values[tag].append(sentiment)

        purchase = purchase_intent_value(item.text)
        if purchase is not None:
            stats.aspect_values["purchase_intent"].append(purchase)

    def average(values: list[float]) -> float | None:
        if len(values) < min_aspect_comments:
            return None
        return round(sum(values) / len(values), 3)

    return MarketValidationSignals(
        visual_score=average(stats.aspect_values["visual"]),
        feel_score=average(stats.aspect_values["feel"]),
        craftsmanship_score=average(stats.aspect_values["craftsmanship"]),
        collection_score=average(stats.aspect_values["collection"]),
        value_score=average(stats.aspect_values["value"]),
        purchase_intent_score=average(stats.aspect_values["purchase_intent"]),
        sentiment_score=average(stats.sentiment_values),
        discussion_count=stats.discussion_count or None,
        marketing_volume=stats.marketing_volume or None,
    )


def import_weibo_payload(
    db_path: Path,
    source_key: str,
    payload: list[dict[str, Any]],
    *,
    strict: bool = True,
    min_aspect_comments: int = 3,
) -> dict[str, Any]:
    if strict and not SkinRepository(db_path).get_skin(source_key):
        raise ValueError(f"skin not found: {source_key}")

    market_repo = MarketSignalRepository(db_path)
    market_repo.ensure_schema()
    evidence = iter_comment_evidence(payload)

    for item in evidence:
        market_repo.add_evidence(
            source_key,
            platform="weibo",
            external_id=item.external_id,
            url=item.url,
            title=item.title,
            author=item.author,
            published_at=item.published_at,
            text=item.text,
            metrics=item.metrics,
            aspect_tags=item.aspect_tags,
            raw_json=item.raw_json,
        )

    signals = aggregate_comment_signals(evidence, min_aspect_comments=min_aspect_comments)
    market_repo.upsert_signals(
        source_key,
        signals,
        signal_source="weibo_comment_aggregate",
        raw_json={
            "source": "weibo_comments",
            "imported_evidence": len(evidence),
            "min_aspect_comments": min_aspect_comments,
        },
        merge=True,
    )
    market_repo.aggregate_evidence_signals(source_key)

    return {
        "source_key": source_key,
        "evidence": len(evidence),
        "signals": {
            name: getattr(signals, name)
            for name in signals.present_fields()
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import Weibo comments as market evidence for one skin.")
    parser.add_argument("json_file", type=Path)
    parser.add_argument("--source-key", required=True, help="Known skin source_key to attach evidence to, e.g. 105-02.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--no-strict", action="store_true", help="Allow source_key values not present in skins table.")
    parser.add_argument(
        "--min-aspect-comments",
        type=int,
        default=3,
        help="Minimum tagged comments needed before emitting an aspect score.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = json.loads(args.json_file.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, list):
        print("Weibo payload must be a list of post entries.", file=sys.stderr)
        return 1
    try:
        result = import_weibo_payload(
            args.db,
            args.source_key,
            payload,
            strict=not args.no_strict,
            min_aspect_comments=args.min_aspect_comments,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

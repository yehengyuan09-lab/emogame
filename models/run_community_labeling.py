#!/usr/bin/env python3
"""Mini-batch runner — produce Community Engagement Labels from Weibo dumps.

Loads the Weibo comment crawls in ``data/weibo_comments/*.json``, flattens them
to posts, builds a best-effort skin keyword index from the crawler, and runs
:class:`CommunitySignalLabeler` over the corpus.  Writes JSONL + CSV + a QA
report (skin-matched vs corpus-level split, score means) to ``--output-dir``.

Usage::

    python -m models.run_community_labeling --dry-run
    python -m models.run_community_labeling --limit 3
    python -m models.run_community_labeling --use-llm --limit 2   # needs AUTODL_TOKEN

Notes
-----
* Scoring is **rule-based by default** (reproducible, no API).  Pass
  ``--use-llm`` to add an LLM refinement layer (same AutoDL endpoint as the
  expert labeler).  See the module docstring for the leakage caveat.
* Posts are matched to a ``skin_key`` via :func:`build_skin_index` /
  :func:`match_skin`; unmatched posts keep ``skin_key=null``.  Current 2026-07
  Weibo events predate most DB skins, so expect mostly corpus-level signals
  until crawls target specific skins — the report surfaces ``n_skin_matched``.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
from pathlib import Path
from typing import Any

from loguru import logger

from crawlers.manager import CrawlerManager
from models.community_labels import (
    CommunitySignalLabeler,
    build_skin_index,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ── load weibo posts ──────────────────────────────────────────────────────────

def load_weibo_posts() -> list[dict[str, Any]]:
    """Flatten all saved Weibo comment crawls into a list of posts."""
    crawler = CrawlerManager()
    sets = crawler.load_weibo_comment_sets()
    posts: list[dict[str, Any]] = []
    for s in sets:
        path = Path(s["path"])
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning(f"Failed to read {path.name}: {exc}")
            continue
        for item in data:
            if not isinstance(item, dict):
                continue
            posts.append(
                {
                    "mid": str(item.get("mid", "")),
                    "title": item.get("title", ""),
                    "comments": item.get("comments", []),
                }
            )
    logger.info(f"Loaded {len(posts)} Weibo posts from {len(sets)} crawl files")
    return posts


# ── main ──────────────────────────────────────────────────────────────────────

async def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Community engagement labeling")
    p.add_argument("--limit", type=int, default=0, help="Max posts (0=all)")
    p.add_argument("--batch-size", type=int, default=10,
                   help="Mini-batch size per chunk")
    p.add_argument("--concurrency", type=int, default=3,
                   help="Max concurrent coroutines within a batch")
    p.add_argument("--use-llm", action="store_true",
                   help="Add LLM enrichment layer (needs AUTODL_TOKEN)")
    p.add_argument("--model", type=str, default=None,
                   help="Override model (default: settings.expert_model)")
    p.add_argument("--output-dir", type=str,
                   default="data/community_labels", help="Export directory")
    p.add_argument("--force", action="store_true", help="Re-label cached posts")
    p.add_argument("--dry-run", action="store_true",
                   help="Show the post queue only")
    args = p.parse_args(argv)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    posts = load_weibo_posts()
    if args.limit > 0:
        posts = posts[: args.limit]

    crawler = CrawlerManager()
    skins = crawler.list_skins(limit=9999)
    skin_index = build_skin_index(skins)
    logger.info(f"Built skin index: {len(skin_index)} skins")

    if args.dry_run:
        for post in posts[:30]:
            from models.community_labels import match_skin
            sk = match_skin(post["title"] or "", skin_index)
            top = max(post["comments"],
                      key=lambda c: int(c.get("like_count", 0) or 0),
                      default=None)
            if sk is None and top:
                sk = match_skin(top.get("text", ""), skin_index)
            logger.info(
                f"  {post['mid']} — {post['title'][:40]} | "
                f"comments={len(post['comments'])} | "
                f"skin={sk or 'corpus'}"
            )
        if len(posts) > 30:
            logger.info(f"  ... and {len(posts) - 30} more")
        return 0

    labeler = CommunitySignalLabeler(model=args.model)
    labels = await labeler.label_batch(
        posts, skin_index=skin_index, batch_size=args.batch_size,
        concurrency=args.concurrency, force=args.force, use_llm=args.use_llm,
    )

    # ── exports ──
    jsonl_path = out_dir / "community_labels.jsonl"
    csv_path = out_dir / "community_labels.csv"
    with jsonl_path.open("w", encoding="utf-8") as fj, \
         csv_path.open("w", encoding="utf-8", newline="") as fc:
        cols = ["skin_key", "mid", "title", "community_premium", "engagement",
                "sentiment", "confidence", "n_comments", "n_meaningful",
                "rationale", "model", "source", "cached", "error"]
        writer = csv.DictWriter(fc, fieldnames=cols)
        writer.writeheader()
        for lbl in labels:
            rec = lbl.to_record()
            fj.write(json.dumps(rec, ensure_ascii=False) + "\n")
            writer.writerow({k: rec.get(k, "") for k in cols})

    # ── QA report ──
    ok = [l for l in labels if l.error is None]
    failed = [l for l in labels if l.error is not None]
    n_skin_matched = sum(1 for l in ok if l.skin_key is not None)
    n_corpus = len(ok) - n_skin_matched
    agg = CommunitySignalLabeler.aggregate(ok)

    report = {
        "total": len(labels),
        "labeled_ok": len(ok),
        "failed": len(failed),
        "failures": [{"mid": l.mid, "error": l.error} for l in failed],
        "n_skin_matched": n_skin_matched,
        "n_corpus_level": n_corpus,
        "score_means": agg,
        "use_llm": args.use_llm,
        "config": {
            "model": labeler.model,
            "temperature": labeler.temperature,
        },
    }
    (out_dir / "community_label_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    logger.info(
        f"=== DONE === labeled={len(ok)}/{len(labels)} failed={len(failed)} "
        f"skin_matched={n_skin_matched} corpus={n_corpus} "
        f"mean_premium={agg['community_premium']}"
    )
    logger.info(f"JSONL : {jsonl_path}")
    logger.info(f"CSV   : {csv_path}")
    logger.info(f"Report: {out_dir / 'community_label_report.json'}")
    return 0 if not failed else 1


if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))

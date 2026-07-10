#!/usr/bin/env python3
"""Mini-batch runner — produce an LLM-expert gold-standard label set.

Pulls skins from the crawler, resolves each wallpaper image, attaches the
official attributes as expert context, and runs :class:`LLMExpertLabeler`
in mini-batches.  Writes JSONL + CSV + a QA report (tier monotonicity,
dimension aggregates) to ``--output-dir``.

Usage::

    python -m models.run_expert_labeling --limit 20 --batch-size 10 --concurrency 3
    python -m models.run_expert_labeling --hero 李白 --use-vlm-context
    python -m models.run_expert_labeling --limit 60 --seed 7 --dry-run

Notes
-----
* ``--limit`` triggers **stratified** sampling across canonical tiers so the
  gold-standard set spans 伴生 → 荣耀典藏 (mirrors the eval-strategy's
  "3 per tier" expert protocol at scale).
* ``--use-vlm-context`` feeds the existing VLM radar chart into the expert
  (cross-check mode).  Omit it for a "blind" expert judgment.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import random
from pathlib import Path
from typing import Any

from loguru import logger

from crawlers.manager import CrawlerManager, SkinSummary
from models.llm_expert_labeler import (
    CANONICAL_TIERS,
    LLMExpertLabeler,
    canonical_tier,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ── image resolution (mirror FeaturePipeline._resolve_image_path) ───────────

def resolve_image_path(skin: SkinSummary) -> Path | None:
    if not skin.image_path:
        return None
    path = Path(skin.image_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path if path.exists() else None


# ── official extra fields ───────────────────────────────────────────────────

def fetch_extra_fields(crawler: CrawlerManager, skin_key: str) -> tuple[str, str]:
    acquire_method, intro = "", ""
    try:
        with crawler._connect_skin_db() as conn:
            row = conn.execute(
                "SELECT acquire_method, intro FROM skins WHERE source_key = ?",
                (skin_key,),
            ).fetchone()
            if row:
                acquire_method = str(row["acquire_method"] or "")
                intro = str(row["intro"] or "")
    except Exception:
        pass
    return acquire_method, intro


# ── stratified sampling ─────────────────────────────────────────────────────

def stratified_sample(
    skins: list[SkinSummary], limit: int, seed: int
) -> list[SkinSummary]:
    """Sample ``limit`` skins with even canonical-tier representation.

    Every tier that has data gets at least ``floor(limit/6)`` skins; the
    remaining budget is filled proportionally to each tier's size.
    """
    if limit >= len(skins):
        return skins

    by_tier: dict[str, list[SkinSummary]] = {t: [] for t in CANONICAL_TIERS}
    for s in skins:
        by_tier[canonical_tier(s.quality)].append(s)

    rng = random.Random(seed)
    # shuffle within tier for deterministic-but-mixed selection
    for t in by_tier:
        rng.shuffle(by_tier[t])

    per_tier_min = max(1, limit // len(CANONICAL_TIERS))
    chosen: list[SkinSummary] = []
    remaining = limit
    for t in CANONICAL_TIERS:
        take = min(per_tier_min, len(by_tier[t]), remaining)
        chosen.extend(by_tier[t][:take])
        remaining -= take

    # fill leftover proportional to tier size
    tier_order = sorted(
        CANONICAL_TIERS, key=lambda t: len(by_tier[t]), reverse=True
    )
    idx = {t: per_tier_min for t in CANONICAL_TIERS}
    while remaining > 0:
        progressed = False
        for t in tier_order:
            if idx[t] < len(by_tier[t]) and remaining > 0:
                chosen.append(by_tier[t][idx[t]])
                idx[t] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            break
    return chosen[:limit]


# ── main ────────────────────────────────────────────────────────────────────

async def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="LLM-expert gold-standard labeling")
    p.add_argument("--limit", type=int, default=0, help="Max skins (0=all)")
    p.add_argument("--batch-size", type=int, default=10,
                   help="Mini-batch size per labeling chunk")
    p.add_argument("--concurrency", type=int, default=3,
                   help="Max concurrent API calls within a batch")
    p.add_argument("--hero", type=str, default=None, help="Filter by hero")
    p.add_argument("--seed", type=int, default=42, help="Sampling seed")
    p.add_argument("--use-vlm-context", action="store_true",
                   help="Feed existing VLM radar into the expert (cross-check)")
    p.add_argument("--output-dir", type=str,
                   default="data/expert_labels", help="Export directory")
    p.add_argument("--force", action="store_true", help="Re-label cached skins")
    p.add_argument("--model", type=str, default=None,
                   help="Override expert model (default: settings.expert_model)")
    p.add_argument("--dry-run", action="store_true",
                   help="Show the sampled work queue only")
    args = p.parse_args(argv)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    crawler = CrawlerManager()
    all_skins = crawler.list_skins(hero_name=args.hero, limit=9999)
    if not all_skins:
        logger.error("No skins found in crawler DB")
        return 1

    if args.limit > 0:
        work = stratified_sample(all_skins, args.limit, args.seed)
    else:
        work = all_skins

    logger.info(
        f"Work queue: {len(work)} skins "
        f"(limit={args.limit}, hero={args.hero}, batch={args.batch_size}, "
        f"concurrency={args.concurrency})"
    )

    # Build items
    items: list[dict[str, Any]] = []
    skipped_no_img = 0
    for s in work:
        img = resolve_image_path(s)
        if img is None:
            skipped_no_img += 1
            continue
        acquire_method, intro = fetch_extra_fields(crawler, s.source_key)
        vlm_radar = None
        if args.use_vlm_context:
            from data.feature_store import FeatureStore
            rec = FeatureStore().get(s.source_key)
            if rec and rec.payload.get("vlm_raw"):
                vlm_radar = rec.payload.get("vlm_raw")
        ctx = LLMExpertLabeler.build_context(
            hero_name=s.hero_name, skin_name=s.skin_name, quality=s.quality,
            acquire_method=acquire_method, price_text=s.price_text,
            online_date=s.online_date, intro=intro, vlm_radar=vlm_radar,
        )
        items.append({"skin_key": s.source_key, "image_path": img, "context": ctx})

    logger.info(f"Resolved {len(items)} images "
                f"({skipped_no_img} skipped: no image)")

    if args.dry_run:
        for it in items[:30]:
            c = it["context"]
            logger.info(f"  {it['skin_key']} — {c['hero_name']}-{c['skin_name']} "
                        f"({c['quality']})")
        if len(items) > 30:
            logger.info(f"  ... and {len(items) - 30} more")
        return 0

    # Run mini-batch labeling
    labeler = LLMExpertLabeler(model=args.model)
    labels = await labeler.label_batch(
        items, batch_size=args.batch_size,
        concurrency=args.concurrency, force=args.force,
    )

    # ── exports ──
    jsonl_path = out_dir / "expert_labels.jsonl"
    csv_path = out_dir / "expert_labels.csv"
    with jsonl_path.open("w", encoding="utf-8") as fj, \
         csv_path.open("w", encoding="utf-8", newline="") as fc:
        cols = ["skin_key", "overall_premium", "aesthetic", "showing_off",
                "belonging", "collection", "surprise", "confidence",
                "rationale", "cached", "error"]
        writer = csv.DictWriter(fc, fieldnames=cols)
        writer.writeheader()
        for lbl in labels:
            rec = lbl.to_record()
            rec.pop("dimension_rationale", None)
            fj.write(json.dumps(rec, ensure_ascii=False) + "\n")
            writer.writerow({k: rec.get(k, "") for k in cols})

    # ── QA report ──
    ok = [l for l in labels if l.error is None]
    failed = [l for l in labels if l.error is not None]
    dims = LLMExpertLabeler.aggregate_dimensions(ok)
    rho = LLMExpertLabeler.tier_monotonicity(
        ok, [it["context"]["quality"] for it in items if it["skin_key"] in
             {l.skin_key for l in ok}]
    ) if ok else float("nan")

    report = {
        "total": len(labels),
        "labeled_ok": len(ok),
        "failed": len(failed),
        "failures": [{"skin_key": l.skin_key, "error": l.error} for l in failed],
        "dimension_means": dims,
        "tier_monotonicity_spearman_rho": rho,
        "batch_size": args.batch_size,
        "use_vlm_context": args.use_vlm_context,
        "config": {
            "model": labeler.model,
            "temperature": labeler.temperature,
        },
    }
    (out_dir / "expert_label_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    logger.info(
        f"=== DONE === labeled={len(ok)}/{len(labels)} failed={len(failed)} "
        f"rho={rho:.3f}" if not (isinstance(rho, float) and rho != rho)
        else f"=== DONE === labeled={len(ok)}/{len(labels)} failed={len(failed)}"
    )
    logger.info(f"JSONL : {jsonl_path}")
    logger.info(f"CSV   : {csv_path}")
    logger.info(f"Report: {out_dir / 'expert_label_report.json'}")
    return 0 if not failed else 1


if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))

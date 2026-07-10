#!/usr/bin/env python3
"""Phase 2 batch CLI — process all skins through the feature pipeline.

Usage::

    python cli.py --limit 10 --l3-policy all
    python cli.py --skin-key "0001-56304" --force
    python cli.py --retry-failures
    python cli.py --l3-policy selective --concurrency 2
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

from loguru import logger

from crawlers.manager import CrawlerManager
from data.feature_store import FeatureStore
from feature_engineering.export import export_from_store
from feature_engineering.features import SkinFeatureVector
from feature_engineering.pipeline import FeaturePipeline
from feature_engineering.report import generate_report

# ═══════════════════════════════════════════════════════════════════════════
# L3 selection policy
# ═══════════════════════════════════════════════════════════════════════════

HIGH_TIERS: set[str] = {
    "传说", "传说限定",
    "珍品传说",
    "无双", "无双限定",
    "荣耀典藏",
}


def should_run_l3(
    skin_quality: str,
    skin_key: str,
    l1_confidence: float | None,
    seed: int = 42,
) -> tuple[bool, str]:
    """Determine whether L3 analysis should run for a given skin.

    Returns ``(should_run, reason)``.
    """
    # High-tier skins always get L3
    if skin_quality in HIGH_TIERS:
        return True, "high_tier"

    # Low-confidence L1 results
    if l1_confidence is not None and l1_confidence < 0.75:
        return True, "low_confidence"

    # Deterministic 5% audit sample
    h = hashlib.md5(f"{seed}:{skin_key}".encode()).digest()
    if int.from_bytes(h[:2], "big") % 100 < 5:
        return True, "audit_5pct"

    return False, ""


# ═══════════════════════════════════════════════════════════════════════════
# CLI argument parsing
# ═══════════════════════════════════════════════════════════════════════════


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 2 feature pipeline — batch VLM + official extraction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--limit", type=int, default=0,
                   help="Max skins to process (0 = all)")
    p.add_argument("--skin-key", type=str, default=None,
                   help="Process a single skin by source_key")
    p.add_argument("--hero", type=str, default=None,
                   help="Filter by hero name")
    p.add_argument("--force", action="store_true",
                   help="Reprocess even if Phase 2 record exists")
    p.add_argument("--retry-failures", action="store_true",
                   help="Only process skins in the retry manifest")
    p.add_argument("--concurrency", type=int, default=1,
                   help="Max concurrent VLM calls (default: 1)")
    p.add_argument("--l3-policy", type=str, default="selective",
                   choices=["selective", "all", "none"],
                   help="L3 trigger policy (default: selective)")
    p.add_argument("--l3-audit-seed", type=int, default=42,
                   help="Seed for deterministic 5%% audit sample (default: 42)")
    p.add_argument("--output-dir", type=str, default="data/phase2_output",
                   help="Directory for exports and report (default: data/phase2_output)")
    p.add_argument("--dry-run", action="store_true",
                   help="List what would be processed without running VLM")
    return p.parse_args(argv)


# ═══════════════════════════════════════════════════════════════════════════
# Main orchestrator
# ═══════════════════════════════════════════════════════════════════════════


async def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    retry_manifest_path = output_dir / "retry_manifest.json"

    # ── Build infrastructure ─────────────────────────────────────────
    crawler = CrawlerManager()
    store = FeatureStore()
    pipeline = FeaturePipeline(
        feature_store=store,
        crawler_manager=crawler,
    )

    # ── Build work queue ─────────────────────────────────────────────
    if args.skin_key:
        skin_keys = [args.skin_key]
        total_in_db = 1
    elif args.retry_failures:
        skin_keys = _load_retry_manifest(retry_manifest_path)
        total_in_db = len(skin_keys)
        if not skin_keys:
            logger.warning("Retry manifest is empty — nothing to do")
            return 0
    else:
        all_skins = crawler.list_skins(hero_name=args.hero, limit=9999)
        skin_keys = [s.source_key for s in all_skins]
        total_in_db = len(skin_keys)
        if args.limit > 0:
            skin_keys = skin_keys[: args.limit]

    logger.info(
        f"Work queue: {len(skin_keys)} skins "
        f"(limit={args.limit}, hero={args.hero}, "
        f"l3_policy={args.l3_policy}, concurrency={args.concurrency})"
    )

    # ── Filter: skip cached records unless --force ───────────────────
    if not args.force and not args.retry_failures:
        filtered: list[str] = []
        skipped = 0
        for sk in skin_keys:
            cached = store.get(sk)
            if cached is not None and cached.is_phase2() and cached.validation_status == "valid":
                skipped += 1
            else:
                filtered.append(sk)
        logger.info(f"Skipping {skipped} cached Phase 2 records")
        skin_keys = filtered
    elif args.retry_failures:
        # For retry mode, we still skip valid cached records unless forced
        if not args.force:
            filtered = []
            for sk in skin_keys:
                cached = store.get(sk)
                if cached is not None and cached.is_phase2() and cached.validation_status == "valid":
                    continue
                filtered.append(sk)
            skin_keys = filtered

    if not skin_keys:
        logger.info("No skins to process — all cached or empty queue")
        return 0

    if args.dry_run:
        logger.info(f"DRY RUN — would process {len(skin_keys)} skins:")
        for sk in skin_keys[:20]:
            s = crawler.get_skin(sk)
            label = s.display_name if s else sk
            logger.info(f"  {sk} — {label}")
        if len(skin_keys) > 20:
            logger.info(f"  ... and {len(skin_keys) - 20} more")
        return 0

    # ── Concurrency control ──────────────────────────────────────────
    sem = asyncio.Semaphore(args.concurrency)
    latency_records: list[dict[str, Any]] = []
    l3_selection_log: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    l1_confidence_map: dict[str, float | None] = {}

    # ═════════════════════════════════════════════════════════════════
    # Pass 1: L1+L2 for all skins
    # ═════════════════════════════════════════════════════════════════
    logger.info(f"=== Pass 1: L1+L2 for {len(skin_keys)} skins ===")

    async def process_l1_l2(skin_key: str) -> SkinFeatureVector:
        async with sem:
            skin = crawler.get_skin(skin_key)
            label = skin.display_name if skin else skin_key
            logger.info(f"[L1+L2] Processing {label}")

            # Fetch extra fields from raw SQLite
            acquire_method, intro = _fetch_extra_fields(crawler, skin_key)

            t0 = time.perf_counter()
            try:
                vector = await pipeline.extract_features(
                    skin_key=skin_key,
                    run_l3=False,
                    force=args.force,
                    acquire_method=acquire_method,
                    intro=intro,
                )
            except Exception as exc:
                logger.error(f"[L1+L2] FAILED {label}: {exc}")
                failures.append({"skin_key": skin_key, "error": str(exc),
                                 "timestamp": time.time()})
                return SkinFeatureVector(skin_key=skin_key,
                                         pipeline_status="error",
                                         validation_errors=[str(exc)])

            elapsed = round(time.perf_counter() - t0, 2)
            latency_records.append(
                {"skin_key": skin_key, "elapsed": elapsed, "mode": "l1_l2"}
            )

            # Extract L1 confidence from VLM raw output
            l1_conf = vector.vlm_raw.get("confidence") if vector.vlm_raw else None
            l1_confidence_map[skin_key] = l1_conf

            avail = sum(vector.availability_array())
            logger.info(
                f"[L1+L2] OK {label}: available={avail}/33 "
                f"elapsed={elapsed}s status={vector.pipeline_status}"
            )
            return vector

    # Run L1+L2 concurrently
    l1_l2_tasks = [process_l1_l2(sk) for sk in skin_keys]
    l1_l2_results = await asyncio.gather(*l1_l2_tasks, return_exceptions=True)

    # ── Determine L3 queue ───────────────────────────────────────────
    l3_queue: list[str] = []
    if args.l3_policy == "all":
        l3_queue = list(skin_keys)
        for sk in l3_queue:
            l3_selection_log.append({"skin_key": sk, "reason": "all"})
    elif args.l3_policy == "selective":
        for sk in skin_keys:
            skin = crawler.get_skin(sk)
            l1_conf = l1_confidence_map.get(sk)
            quality = skin.quality if skin else ""
            run, reason = should_run_l3(quality, sk, l1_conf, args.l3_audit_seed)
            if run:
                l3_queue.append(sk)
                l3_selection_log.append({"skin_key": sk, "reason": reason})
    else:  # "none"
        l3_queue = []

    logger.info(
        f"L3 queue: {len(l3_queue)} skins "
        f"(policy={args.l3_policy}, "
        f"high_tier={sum(1 for e in l3_selection_log if e['reason'] == 'high_tier')}, "
        f"low_confidence={sum(1 for e in l3_selection_log if e['reason'] == 'low_confidence')}, "
        f"audit={sum(1 for e in l3_selection_log if e['reason'] == 'audit_5pct')})"
    )

    # ═════════════════════════════════════════════════════════════════
    # Pass 2: L3 for selected skins
    # ═════════════════════════════════════════════════════════════════
    if l3_queue:
        logger.info(f"=== Pass 2: L3 for {len(l3_queue)} skins ===")

        async def process_l3(skin_key: str) -> SkinFeatureVector:
            async with sem:
                skin = crawler.get_skin(skin_key)
                label = skin.display_name if skin else skin_key
                reason = next(
                    (e["reason"] for e in l3_selection_log
                     if e["skin_key"] == skin_key), "unknown"
                )
                logger.info(f"[L3] Processing {label} (reason={reason})")

                acquire_method, intro = _fetch_extra_fields(crawler, skin_key)

                t0 = time.perf_counter()
                try:
                    vector = await pipeline.extract_features(
                        skin_key=skin_key,
                        run_l3=True,
                        force=True,  # Reprocess to include L3
                        acquire_method=acquire_method,
                        intro=intro,
                    )
                except Exception as exc:
                    logger.error(f"[L3] FAILED {label}: {exc}")
                    failures.append({"skin_key": skin_key, "error": str(exc),
                                     "timestamp": time.time()})
                    return SkinFeatureVector(skin_key=skin_key,
                                             pipeline_status="error",
                                             validation_errors=[str(exc)])

                elapsed = round(time.perf_counter() - t0, 2)
                latency_records.append(
                    {"skin_key": skin_key, "elapsed": elapsed, "mode": "full"}
                )
                avail = sum(vector.availability_array())
                logger.info(
                    f"[L3] OK {label}: available={avail}/33 "
                    f"elapsed={elapsed}s status={vector.pipeline_status}"
                )
                return vector

        l3_tasks = [process_l3(sk) for sk in l3_queue]
        await asyncio.gather(*l3_tasks, return_exceptions=True)

    # ═════════════════════════════════════════════════════════════════
    # Post-processing: exports & report
    # ═════════════════════════════════════════════════════════════════
    logger.info("=== Export & Report ===")

    # Write retry manifest
    if failures:
        retry_manifest_path.write_text(
            json.dumps(failures, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.warning(f"Retry manifest: {len(failures)} failures → {retry_manifest_path}")

    # Export CSV + JSONL
    csv_path, jsonl_path = export_from_store(store, output_dir)
    logger.info(f"CSV export: {csv_path}")
    logger.info(f"JSONL export: {jsonl_path}")

    # Generate Markdown report
    report_path = generate_report(
        store=store,
        output_path=output_dir / "phase2_report.md",
        l3_selection_log=l3_selection_log,
        total_skins=total_in_db,
        latency_records=latency_records,
    )
    logger.info(f"QA Report: {report_path}")

    # Final summary
    record_count = store.count()
    logger.info(
        f"=== DONE === "
        f"records={record_count}/{total_in_db} "
        f"failures={len(failures)} "
        f"l3_processed={len(l3_queue)}"
    )

    return 0 if not failures else 1


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════


def _fetch_extra_fields(crawler: CrawlerManager, skin_key: str) -> tuple[str, str]:
    """Fetch ``acquire_method`` and ``intro`` from the raw SQLite skins table."""
    acquire_method = ""
    intro = ""
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


def _load_retry_manifest(path: Path) -> list[str]:
    """Load skin keys from a retry manifest JSON file."""
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [entry["skin_key"] for entry in data if "skin_key" in entry]
    except (json.JSONDecodeError, TypeError, OSError):
        pass
    return []


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

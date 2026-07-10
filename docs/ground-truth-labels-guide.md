# Ground-Truth Labels — Operation Guide

> Reference for the two implemented evidence sources of the emotional-premium
> ground truth (see `progress/2026-07-09-ground-truth-data-collection.md` for
> the full strategy).  This document is **operation-focused**: what each label
> source is, how to produce it, and the exact output structure.
>
> Both labelers are implemented and verified against the live AutoDL endpoint.
> Neither writes to the FeatureStore — they export JSONL/CSV/report files that a
> future `models/ground_truth.py` fusion module will consume.

---

## 1. Expert-Labeled Gold Standard

**Module:** `models/llm_expert_labeler.py` + `models/run_expert_labeling.py`
**What it is:** An LLM (`qwen3-vl-plus` via AutoDL) acting as a *standalone
domain expert*. Given a skin wallpaper + official attributes, it rates the
emotional premium on a 5-dimension scale. This is the **primary gold-standard
anchor** for training/benchmarking.

**Key properties**
- `temperature=0` → fully reproducible; re-running the same image returns the
  same label.
- Content-hash cached (`CacheManager`, level `"expert"`) → re-label is free.
- **Blind by default** (`--use-vlm-context` off) so the expert does not see the
  VLM L3 radar — keeps it independent of VLM-derived features.
- Mini-batch labeling (`--batch-size` / `--concurrency`) for incremental,
  rate-limit-safe corpus coverage.
- **Leakage caveat:** uses `qwen3-vl-plus`; VLM L3 uses `gpt-5.4-mini`, a
  *different* model family, so label/feature leakage is reduced. Do **not** feed
  these labels as targets to an XGBoost model whose features include L3 outputs.

### Entry commands

```bash
# 1. Dry-run: show the stratified work queue (no API calls)
python -m models.run_expert_labeling --limit 12 --batch-size 4 --dry-run

# 2. Production run: blind expert labeling across all tiers
python -m models.run_expert_labeling --limit 20 --batch-size 10 --concurrency 3

# 3. Focus on one hero
python -m models.run_expert_labeling --hero 李白

# 4. Cross-check mode: feed the existing VLM radar into the expert
python -m models.run_expert_labeling --limit 60 --use-vlm-context

# 5. Override model / re-label cached skins
python -m models.run_expert_labeling --limit 30 --model qwen3-vl-plus --force
```

**Arguments**

| Flag | Default | Purpose |
|------|---------|---------|
| `--limit N` | `0` (=all) | Max skins to label |
| `--batch-size` | `10` | Skins per mini-batch chunk |
| `--concurrency` | `3` | Max in-flight API calls within a batch |
| `--hero NAME` | — | Filter to one hero |
| `--seed` | `42` | Sampling seed for `--limit` |
| `--use-vlm-context` | off | Feed VLM radar into expert (cross-check) |
| `--model` | `expert_model` | Override the expert model |
| `--output-dir` | `data/expert_labels` | Export directory |
| `--force` | off | Re-label cached skins |
| `--dry-run` | off | Print the queue only |

Sampling is **stratified** across the 6 canonical tiers
(伴生/勇者/史诗/传说/无双/荣耀典藏) so the gold set spans the full quality range.

### Output structure

Written to `--output-dir` (default `data/expert_labels/`):

**`expert_labels.jsonl`** — one record per skin:
```json
{
  "skin_key": "0054-17201",
  "overall_premium": 52.0,
  "aesthetic": 78.0,
  "showing_off": 22.0,
  "belonging": 45.0,
  "collection": 28.0,
  "surprise": 58.0,
  "confidence": 4,
  "rationale": "画面完成度和配色有明显加分，但稀缺性与排面有限…",
  "model": "qwen3-vl-plus",
  "source": "llm_expert",
  "elapsed_s": 6.04,
  "cached": true,
  "error": null
}
```

**`expert_labels.csv`** — flat table, columns:
`skin_key, overall_premium, aesthetic, showing_off, belonging, collection,
surprise, confidence, rationale, cached, error`

**`expert_label_report.json`** — QA summary:
```json
{
  "total": 20,
  "labeled_ok": 20,
  "failed": 0,
  "failures": [],
  "dimension_means": {
    "aesthetic": 85.6, "showing_off": 67.65, "belonging": 69.9,
    "collection": 69.15, "surprise": 63.6, "overall_premium": 75.5
  },
  "tier_monotonicity_spearman_rho": 0.868,
  "batch_size": 10,
  "use_vlm_context": false,
  "config": { "model": "qwen3-vl-plus", "temperature": 0.0 }
}
```
The `tier_monotonicity_spearman_rho` (target > 0.6) validates that higher tiers
receive higher premiums — a sanity check on label quality.

---

## 2. Market-Based Proxy Labels

**Status:** *Planned — not yet implemented.*  The strategy defines it as the
second evidence source, derived from **actual player spending behavior** rather
than model judgment. Recorded here so the operations matrix is complete.

**What it would be:** A `MarketPremiumEstimator` (`models/market_labels.py`)
that turns transactional signals into a 0–100 premium proxy:

| Signal | Proxy formula | Meaning |
|--------|--------------|---------|
| Direct spend | `avg_spend_to_obtain / base_price` | Willingness to over-pay |
| Gacha | pull-count distribution | Effort premium |
| Secondary market | resale price | Residual value |
| Time-unlock | opportunity cost | Alt-cost premium |

**Why it matters:** market labels are *behavioral* (not perceptual), so they
anchor the gold standard in real money rather than model opinion. They are the
natural target for `market_weighted` fusion (see strategy §6).

**Blocker:** requires a **market/spend data source** (in-game transaction logs,
gacha pull DB, or secondary-market scrape) that is not yet wired into the
project. Until that exists, this labeler cannot be run.

---

## 3. Community Engagement Labels (implemented, for context)

**Module:** `models/community_labels.py` + `models/run_community_labeling.py`
Listed here because it is the third evidence source and ships alongside the
expert labeler.

**What it is:** Scores community excitement from Weibo comment dumps.
Rule-based by default (reproducible, no API); optional `--use-llm` enrichment.

### Entry commands

```bash
# Dry-run: post queue + skin linkage
python -m models.run_community_labeling --dry-run

# Rule-based full corpus (no API)
python -m models.run_community_labeling

# LLM-enriched (needs AUTODL_TOKEN in .env)
python -m models.run_community_labeling --use-llm --limit 2 --force
```

**Arguments**

| Flag | Default | Purpose |
|------|---------|---------|
| `--limit N` | `0` (=all) | Max posts |
| `--batch-size` | `10` | Posts per chunk |
| `--concurrency` | `3` | Max in-flight coroutines |
| `--use-llm` | off | Add LLM refinement layer |
| `--model` | `expert_model` | Override model |
| `--output-dir` | `data/community_labels` | Export directory |
| `--force` | off | Re-label cached posts |
| `--dry-run` | off | Print queue only |

### Output structure

**`community_labels.jsonl`** — one record per post:
```json
{
  "skin_key": "0097-56406",
  "mid": "5312547970617832",
  "title": "赵云/孙权/姬小满 西行封妖记皮肤CG",
  "community_premium": 94.0,
  "engagement": 53.5,
  "sentiment": 100.0,
  "confidence": 5,
  "rationale": "高赞评论高度集中于角色颜值、主题曲积极反馈…",
  "n_comments": 40,
  "n_meaningful": 39,
  "model": "qwen3-vl-plus",
  "source": "community_engagement",
  "elapsed_s": 3.36,
  "cached": false,
  "error": null
}
```
`skin_key: null` = corpus-level signal (post not matched to a specific skin).

**`community_labels.csv`** — columns:
`skin_key, mid, title, community_premium, engagement, sentiment, confidence,
n_comments, n_meaningful, rationale, model, source, cached, error`

**`community_label_report.json`**:
```json
{
  "total": 15, "labeled_ok": 15, "failed": 0,
  "n_skin_matched": 13, "n_corpus_level": 2,
  "score_means": { "community_premium": 58.21, "engagement": 21.17, "sentiment": 82.91 },
  "use_llm": false,
  "config": { "model": "qwen3-vl-plus", "temperature": 0.0 }
}
```

---

## Operations Matrix

| Source | Module | Status | Entry command | Output dir |
|--------|--------|--------|--------------|------------|
| Expert gold standard | `models/run_expert_labeling.py` | ✅ implemented | `python -m models.run_expert_labeling --limit 20` | `data/expert_labels/` |
| Market proxy | `models/market_labels.py` | ⬜ planned (needs spend data) | — | `data/market_labels/` |
| Community engagement | `models/run_community_labeling.py` | ✅ implemented | `python -m models.run_community_labeling` | `data/community_labels/` |

**Verified runs:** expert labeler — 20/20 skins, tier ρ=0.868; community
labeler — 15/15 posts (rule) + 2/2 LLM-enriched, skin linkage working. Both
cached for free re-runs. `AUTODL_TOKEN` is read from `.env` via pydantic-settings
(never hardcoded).

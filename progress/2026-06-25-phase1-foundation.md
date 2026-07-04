# EmoGame Progress — 2026-06-25

## Phase 1 Foundation Gate Status

Phase 1 is now closed for the local foundation scope in `docs/09-deployment.md`.

## Decisions

- Keep `qwen2.5vl:3b` as the Phase 1 default for both L1 and L2 Ollama tiers.
- Defer `llama3.2-vision:11b` as a candidate L2 model until it has an English prompt and a larger token budget that can satisfy the strict JSON contract.
- Keep AutoDL `gpt-5.4-mini` as L3 semantic analysis and combined L2/L3 fallback when local L2 is unavailable.

## Implemented

- VLM prompt/config cleanup:
  - L1 prompt now requires single enum choices and real visible hex colors.
  - L2 docs now match the working `qwen2.5vl:3b` default.
  - AutoDL client disables environment proxies with `trust_env=False`.
- Crawler coordination:
  - `crawlers/manager.py` exposes WZRY official skin rows and saved Weibo crawl summaries.
- Feature storage:
  - `data/feature_store.py` provides SQLite CRUD for feature payloads in `data/emogame.db`.
  - `feature_engineering/pipeline.py` can seed baseline feature records from official skin metadata.
- Basic UI:
  - `app.py` provides hero selection, skin browsing, local image preview, Weibo crawl summary, and feature-store seeding.

## Remaining Phase 2 Work

- Implement the complete 33-dimensional `SkinFeatureVector`.
- Batch-run VLM outputs over the selected skin image set and write results into the feature store.
- Normalize community/social features from Weibo comments and future Tieba/NGA/Bilibili sources.
- Build rule-engine scoring and later XGBoost training once enough labeled examples exist.

## Validation Target

Run:

```bash
python3 -m py_compile app.py crawlers/manager.py data/feature_store.py feature_engineering/pipeline.py vlm/*.py
```

Then optionally launch:

```bash
streamlit run app.py
```

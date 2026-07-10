# Phase 2: 33-Dimensional Feature and VLM Batch Pipeline

## Summary

Implement a reproducible pipeline for all 800 collected skins. The canonical output will contain 33 nullable model features, 33 availability flags, provenance, raw VLM results, and validation status. Missing data will remain null, never
silently become zero.

Run local L1/L2 for every valid image. Run L3 for:

- Limited or high-tier skins: 传说, 珍品传说, 无双, 荣耀典藏
- L1 confidence below 0.75
- A deterministic 5% audit sample using a fixed seed

## Implementation Changes

- Define a validated SkinFeatureVector with the documented five groups: aesthetic 8, belonging 8, showing-off 6, collection 7, surprise 4.
- Add availability, per-field provenance, schema version, image hash, pipeline status, and validation errors outside the 33 model dimensions.
- Populate official features through deterministic mappings of quality, release date, acquisition text, price, and intro. Preserve source evidence and leave ambiguous values null.
- Map L2 outputs to the four VLM dimensions. Retain L1/L2/L3 raw outputs as auxiliary metadata rather than additional model dimensions.
- Change VlmPipeline to support explicit execution modes: l1_l2 and full. Preserve cache and degradation behavior.
- Extend FeaturePipeline to merge official metadata and VLM results, validate ranges, and write one atomic record per skin.
- Add a resumable batch CLI with filters for limit, skin key, force, retry failures, concurrency, L3 policy, and output directory. Process L1/L2 first, then the selected L3 queue.
- Store canonical records in SQLite and export model-ready CSV plus lossless JSONL. CSV uses empty cells for nulls and separate availability columns.
- Generate a Markdown QA report containing counts, coverage by feature/group, VLM status and source distributions, L3 selection reasons, invalid records, failures, latency, and cache-hit rates.
- Correct the feature documentation’s “31 dimensions” typo and align VLM source descriptions with the implemented L2 mappings.

## Public Interfaces

- SkinFeatureVector.to_array() returns 33 values in a fixed documented order and requires an explicit missing-value policy.
- SkinFeatureVector.availability_array() returns the corresponding 33 binary flags.
- FeaturePipeline.extract_features(skin_key, run_l3=False, force=False) produces and persists one complete record.
- Batch CLI outputs features.jsonl, features.csv, phase2_report.md, and a retry manifest.

## Test Plan

- Unit-test all 33-field ordering, ranges, null handling, availability masks, and serialization.
- Test official quality/date/acquisition mappings, including unknown and ambiguous text.
- Mock L1/L2/L3 responses and verify tier selection, fallback, cache reuse, and partial failures.
- Test resumability, forced refresh, retry manifests, deterministic 5% sampling, and atomic writes.
- Run a 10-skin smoke batch spanning empty quality, limited, high-tier, low-confidence, and cached cases.
- Acceptance requires all 800 skins to receive records, every valid image to attempt L1/L2, no invalid value to enter the canonical vector, and all failures to appear in the report/retry manifest.

## Assumptions

- SQLite is canonical; CSV and JSONL are generated exports.
- Existing records remain readable, but Phase 2 records use a new schema version.
- Missing inputs are not imputed during extraction; model-stage imputation will be designed separately.
- L3 semantics are retained for analysis and future feature revisions but do not alter the fixed 33 dimensions.
"""CSV and JSONL export for SkinFeatureVector records.

Produces model-ready CSV (with availability columns) and lossless JSONL
from FeatureStore records.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from data.feature_store import FeatureRecord, FeatureStore
from feature_engineering.features import FEATURE_FIELD_NAMES, FEATURE_GROUPS


def export_csv(
    records: list[dict[str, Any]],
    output_path: Path,
) -> Path:
    """Write a model-ready CSV.

    Columns:
        * ``skin_key``, ``skin_id``, ``hero_name``, ``skin_name``
        * 33 feature value columns (empty for ``None``)
        * 33 availability columns prefixed ``avail_`` (0/1)

    Returns the output path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build fieldnames
    identity_cols = ["skin_key", "skin_id", "hero_name", "skin_name"]
    feature_cols = list(FEATURE_FIELD_NAMES)
    avail_cols = [f"avail_{name}" for name in FEATURE_FIELD_NAMES]
    extra_cols = ["schema_version", "pipeline_status", "image_hash"]
    fieldnames = identity_cols + feature_cols + avail_cols + extra_cols

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()

        for rec in records:
            row: dict[str, Any] = {}

            # Identity
            for col in identity_cols:
                row[col] = rec.get(col, "")

            # Feature values (empty string for None)
            for name in FEATURE_FIELD_NAMES:
                val = rec.get(name)
                row[name] = "" if val is None else val

            # Availability columns
            for name in FEATURE_FIELD_NAMES:
                row[f"avail_{name}"] = 1 if rec.get(name) is not None else 0

            # Extra metadata
            for col in extra_cols:
                row[col] = rec.get(col, "")

            writer.writerow(row)

    return output_path


def export_jsonl(
    records: list[dict[str, Any]],
    output_path: Path,
) -> Path:
    """Write lossless JSONL — one JSON object per line.

    Each line is a self-contained record including all 33 features,
    metadata, provenance, and raw VLM results.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for rec in records:
            # Preserve None values — use a custom encoder that keeps nulls
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")

    return output_path


def export_from_store(
    store: FeatureStore,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Read all Phase 2 records from the store and export both formats.

    Returns ``(csv_path, jsonl_path)``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    records_raw = store.get_phase2_records()
    records = [r.payload for r in records_raw]

    csv_path = export_csv(records, output_dir / "features.csv")
    jsonl_path = export_jsonl(records, output_dir / "features.jsonl")

    return csv_path, jsonl_path

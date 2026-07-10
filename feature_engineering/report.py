"""Markdown QA report generator for the Phase 2 feature pipeline.

Produces a ``phase2_report.md`` with coverage statistics, VLM status
distributions, L3 selection reasons, failure manifests, and latency metrics.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from data.feature_store import FeatureRecord, FeatureStore
from feature_engineering.features import FEATURE_FIELD_NAMES, FEATURE_GROUPS


def generate_report(
    store: FeatureStore,
    output_path: Path,
    vlm_stats: dict[str, Any] | None = None,
    l3_selection_log: list[dict[str, Any]] | None = None,
    total_skins: int = 0,
    latency_records: list[dict[str, Any]] | None = None,
) -> Path:
    """Generate ``phase2_report.md`` from the feature store.

    Args:
        store: The FeatureStore containing processed records.
        output_path: Where to write the Markdown report.
        vlm_stats: Optional per-tier cache-hit and source counts.
        l3_selection_log: List of ``{skin_key, reason}`` entries.
        total_skins: Total number of skins in the crawler DB (for coverage %).
        latency_records: List of ``{skin_key, elapsed, mode}`` records.

    Returns the output path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = store.get_phase2_records()
    if not records:
        _write_empty_report(output_path, total_skins)
        return output_path

    payloads = [r.payload for r in records]
    vlm_stats = vlm_stats or {}
    l3_log = l3_selection_log or []
    latency = latency_records or []

    lines: list[str] = []
    _h = lines.append

    _h("# Phase 2 — Feature Pipeline QA Report")
    _h("")
    _h(f"Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}Z")
    _h("")

    # ── 1. Summary ─────────────────────────────────────────────────
    _h("## 1. Summary")
    _h("")
    valid = sum(1 for r in records if r.validation_status == "valid")
    invalid = sum(1 for r in records if r.validation_status == "invalid")
    errors = sum(1 for p in payloads if p.get("pipeline_status") == "error")
    partial = sum(1 for p in payloads if p.get("pipeline_status") == "partial")
    complete = sum(1 for p in payloads if p.get("pipeline_status") == "complete")

    _h(f"| Metric | Count |")
    _h(f"|--------|-------|")
    _h(f"| Total skins in crawler DB | {total_skins} |")
    _h(f"| Phase 2 records written | {len(records)} |")
    _h(f"| Complete (all tiers ok) | {complete} |")
    _h(f"| Partial (some degradation) | {partial} |")
    _h(f"| Errors | {errors} |")
    _h(f"| Valid records | {valid} |")
    _h(f"| Invalid records (range violations) | {invalid} |")
    _h(f"| Schema version | 2 |")
    _h("")

    # ── 2. Coverage by feature group ────────────────────────────────
    _h("## 2. Coverage by Feature Group")
    _h("")
    _h("| Group | Weight | Fields | Avg % Populated |")
    _h("|-------|--------|--------|-----------------|")
    weights = {"aesthetic": 0.30, "belonging": 0.20, "showing_off": 0.25,
               "collection": 0.15, "surprise": 0.10}
    for group, names in FEATURE_GROUPS.items():
        populate_rates = []
        for name in names:
            populated = sum(
                1 for p in payloads if p.get(name) is not None
            )
            populate_rates.append(populated / len(payloads) * 100 if payloads else 0)
        avg = sum(populate_rates) / len(populate_rates) if populate_rates else 0
        w = weights.get(group, 0)
        _h(f"| {group} | {w:.2f} | {len(names)} | {avg:.1f}% |")
    _h("")

    # ── 3. Coverage by individual feature ───────────────────────────
    _h("## 3. Coverage by Feature")
    _h("")
    _h("| # | Feature | Group | % Populated |")
    _h("|---|---------|-------|-------------|")
    for i, name in enumerate(FEATURE_FIELD_NAMES, 1):
        group = _group_for(name)
        populated = sum(1 for p in payloads if p.get(name) is not None)
        pct = populated / len(payloads) * 100 if payloads else 0
        _h(f"| {i} | `{name}` | {group} | {pct:.1f}% |")
    _h("")

    # ── 4. VLM status distribution ──────────────────────────────────
    _h("## 4. VLM Pipeline Status Distribution")
    _h("")
    status_counts: dict[str, int] = {}
    for p in payloads:
        s = p.get("status", "unknown")
        status_counts[s] = status_counts.get(s, 0) + 1
    _h("| Status | Count | % |")
    _h("|--------|-------|---|")
    for s, c in sorted(status_counts.items()):
        _h(f"| {s} | {c} | {c/len(payloads)*100:.1f}% |")
    _h("")

    # ── 5. VLM source distribution ──────────────────────────────────
    _h("## 5. Cache-Hit Rates")
    _h("")
    l1_hits = sum(1 for p in payloads if p.get("cache_hit_l1"))
    l2_hits = sum(1 for p in payloads if p.get("cache_hit_l2"))
    l3_hits = sum(1 for p in payloads if p.get("cache_hit_l3"))
    _h("| Tier | Cache Hits | Total | Hit Rate |")
    _h("|------|-----------|-------|----------|")
    _h(f"| L1 | {l1_hits} | {len(payloads)} | {l1_hits/len(payloads)*100:.1f}% |")
    _h(f"| L2 | {l2_hits} | {len(payloads)} | {l2_hits/len(payloads)*100:.1f}% |")
    _h(f"| L3 | {l3_hits} | {len(payloads)} | {l3_hits/len(payloads)*100:.1f}% |")
    _h("")

    # ── 6. L3 selection reasons ─────────────────────────────────────
    if l3_log:
        _h("## 6. L3 Selection Reasons")
        _h("")
        reason_counts: dict[str, int] = {}
        for entry in l3_log:
            r = entry.get("reason", "unknown")
            reason_counts[r] = reason_counts.get(r, 0) + 1
        _h("| Reason | Count |")
        _h("|--------|-------|")
        for r, c in sorted(reason_counts.items(), key=lambda x: -x[1]):
            _h(f"| {r} | {c} |")
        _h("")

    # ── 7. Invalid records ──────────────────────────────────────────
    _h("## 7. Invalid Records")
    _h("")
    invalid_records = [r for r in records if r.validation_status == "invalid"]
    if invalid_records:
        _h("| Skin Key | Validation Errors |")
        _h("|----------|-------------------|")
        for r in invalid_records:
            errs = ", ".join(r.validation_errors)
            _h(f"| {r.skin_key} | {errs} |")
    else:
        _h("No invalid records. ✓")
    _h("")

    # ── 8. Failure manifest ─────────────────────────────────────────
    _h("## 8. Failure Manifest")
    _h("")
    failures = [r for r in records if r.validation_status == "invalid"
                or r.payload.get("pipeline_status") == "error"]
    if failures:
        _h("| Skin Key | Status | Errors |")
        _h("|----------|--------|--------|")
        for r in failures:
            errs = ", ".join(r.validation_errors)
            _h(f"| {r.skin_key} | {r.validation_status} | {errs} |")
    else:
        _h("No failures. ✓")
    _h("")

    # ── 9. Latency statistics ───────────────────────────────────────
    if latency:
        _h("## 9. Latency Statistics")
        _h("")
        times = [r["elapsed"] for r in latency]
        times.sort()
        _h(f"| Stat | Value (seconds) |")
        _h(f"|------|-----------------|")
        _h(f"| Min | {min(times):.1f} |")
        _h(f"| Median | {_percentile(times, 50):.1f} |")
        _h(f"| P95 | {_percentile(times, 95):.1f} |")
        _h(f"| Max | {max(times):.1f} |")
        _h("")

    # ── Write ──────────────────────────────────────────────────────
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════


def _group_for(field_name: str) -> str:
    for group, names in FEATURE_GROUPS.items():
        if field_name in names:
            return group
    return "unknown"


def _percentile(sorted_values: list[float], pct: float) -> float:
    """Return the pct-th percentile of sorted values (linear interpolation)."""
    if not sorted_values:
        return 0.0
    k = (len(sorted_values) - 1) * pct / 100.0
    f = int(k)
    c = k - f
    if f + 1 < len(sorted_values):
        return sorted_values[f] + c * (sorted_values[f + 1] - sorted_values[f])
    return sorted_values[f]


def _write_empty_report(output_path: Path, total_skins: int) -> None:
    lines = [
        "# Phase 2 — Feature Pipeline QA Report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}Z",
        "",
        "## 1. Summary",
        "",
        f"| Metric | Count |",
        f"|--------|-------|",
        f"| Total skins in crawler DB | {total_skins} |",
        f"| Phase 2 records written | 0 |",
        "",
        "⚠️ No Phase 2 records found in the feature store.",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")

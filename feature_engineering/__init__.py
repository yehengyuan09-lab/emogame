"""Feature-engineering entry points.

Exports the full Phase 2 public API: canonical feature vector, official
metadata extractor, FeaturePipeline orchestrator, and export/report utilities.
"""

from feature_engineering.features import (
    FEATURE_FIELD_NAMES,
    FEATURE_GROUPS,
    FIELD_RANGES,
    AcquisitionMethod,
    IPSourceType,
    LimitedType,
    Provenance,
    SkinFeatureVector,
    VisibilityLevel,
)
from feature_engineering.export import export_csv, export_jsonl, export_from_store
from feature_engineering.official_extractor import OfficialFeatureMapper
from feature_engineering.pipeline import FeaturePipeline
from feature_engineering.report import generate_report

__all__ = [
    "AcquisitionMethod",
    "FEATURE_FIELD_NAMES",
    "FEATURE_GROUPS",
    "FIELD_RANGES",
    "FeaturePipeline",
    "IPSourceType",
    "LimitedType",
    "OfficialFeatureMapper",
    "Provenance",
    "SkinFeatureVector",
    "VisibilityLevel",
    "export_csv",
    "export_jsonl",
    "export_from_store",
    "generate_report",
]

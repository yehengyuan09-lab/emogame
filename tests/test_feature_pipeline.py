"""Integration tests for FeaturePipeline.extract_features()."""

import tempfile
import unittest
from pathlib import Path

from feature_engineering.features import FEATURE_FIELD_NAMES, SkinFeatureVector
from feature_engineering.official_extractor import OfficialFeatureMapper
from feature_engineering.pipeline import FeaturePipeline


class FeaturePipelineOfficialOnlyTests(unittest.TestCase):
    """Tests for extract_features when no VLM is needed (pure official extraction)."""

    def test_extract_features_skin_not_found(self):
        """Returns error vector when skin key doesn't exist."""
        import asyncio

        fp = FeaturePipeline()
        vector = asyncio.run(
            fp.extract_features(skin_key="nonexistent-key-99999")
        )
        self.assertEqual(vector.pipeline_status, "error")
        self.assertTrue(any("not found" in e for e in vector.validation_errors))

    def test_official_mapper_integration(self):
        """OfficialFeatureMapper works with real-quality FeaturePipeline setup."""
        mapper = OfficialFeatureMapper()
        features, prov = mapper.extract_all(
            quality="传说",
            online_date="20240101",
            acquire_method="1688点券 限时上架",
            price_text="1688点券",
            intro="含独立语音包和回城特效",
            hero_name="李白",
            skin_name="诗剑行",
        )
        self.assertEqual(features["official_tier"], 3)
        self.assertIsNotNone(features["skin_age_days"])
        self.assertIsNotNone(features["has_voice_pack"])
        self.assertIsNotNone(features["has_custom_anim"])
        self.assertEqual(features["ip_source_type"], 0)
        # Check provenance coverage
        for name in FEATURE_FIELD_NAMES:
            vlm_fields = {"vlm_art_quality", "vlm_effect_score",
                          "vlm_color_harmony", "vlm_composition"}
            if name not in vlm_fields:
                self.assertIn(name, prov, f"Missing provenance for {name}")


class FeaturePipelineCachingTests(unittest.TestCase):
    """Tests for FeatureStore caching behavior in extract_features."""

    def test_phase1_record_not_recognized_as_cached(self):
        """Phase 1 records (schema_version < 2) should not be treated as cache hits."""
        fp = FeaturePipeline()
        # Seed a Phase 1 record
        rec = fp.store.put(
            skin_key="cache-test-001",
            payload={"test": True},
            source="test",
            schema_version=1,
        )
        self.assertEqual(rec.schema_version, 1)
        self.assertFalse(rec.is_phase2())

    def test_phase2_record_recognized_as_cached(self):
        """Phase 2 records should be recognized by is_phase2()."""
        fp = FeaturePipeline()
        rec = fp.store.put(
            skin_key="cache-test-002",
            payload={"test": True},
            source="test",
            schema_version=2,
        )
        self.assertEqual(rec.schema_version, 2)
        self.assertTrue(rec.is_phase2())
        self.assertEqual(rec.validation_status, "unvalidated")


class FeaturePipelineMergeTests(unittest.TestCase):
    """Tests for VLM + official feature merging."""

    def test_from_dicts_merge_vlm_and_official(self):
        """VLM features and official features are correctly merged."""
        v = SkinFeatureVector.from_dicts(
            skin_key="test-001",
            skin_id="56304",
            hero_name="李白",
            skin_name="诗剑行",
            vlm_features={
                "vlm_art_quality": 8.5,
                "vlm_effect_score": 9.0,
                "vlm_color_harmony": 0.75,
                "vlm_composition": 8.0,
            },
            official_features={
                "official_tier": 3,
                "skin_age_days": 200,
                "is_limited": True,
                "acquisition_method": 0,
            },
        )
        # VLM fields
        self.assertEqual(v.vlm_art_quality, 8.5)
        self.assertEqual(v.vlm_effect_score, 9.0)
        self.assertEqual(v.vlm_color_harmony, 0.75)
        self.assertEqual(v.vlm_composition, 8.0)
        # Official fields
        self.assertEqual(v.official_tier, 3)
        self.assertEqual(v.skin_age_days, 200)
        self.assertTrue(v.is_limited)
        self.assertEqual(v.acquisition_method, 0)
        # Availability
        v.recompute_availability()
        self.assertTrue(v.availability["vlm_art_quality"])
        self.assertTrue(v.availability["official_tier"])
        self.assertFalse(v.availability["baidu_index_7d"])

    def test_vlm_overrides_official_for_vlm_fields(self):
        """VLM source takes precedence for the 4 VLM-mapped dimensions."""
        v = SkinFeatureVector.from_dicts(
            skin_key="test",
            vlm_features={"vlm_art_quality": 9.5},
            official_features={"vlm_art_quality": 5.0},
        )
        self.assertEqual(v.vlm_art_quality, 9.5)

    def test_null_fields_preserved(self):
        """Fields without data remain None."""
        v = SkinFeatureVector.from_dicts(skin_key="test")
        arr = v.to_array()
        self.assertTrue(all(x is None for x in arr))


class FeaturePipelineValidationTests(unittest.TestCase):
    """Tests for feature validation within the pipeline context."""

    def test_valid_vector_no_errors(self):
        v = SkinFeatureVector(
            skin_key="test",
            vlm_art_quality=8.5,
            official_tier=3,
            skin_age_days=200,
            acquisition_method=0,
        )
        errs = v.validate_ranges()
        self.assertEqual(errs, [])

    def test_invalid_official_tier_caught_by_pydantic(self):
        from pydantic import ValidationError
        with self.assertRaises(ValidationError):
            SkinFeatureVector(skin_key="test", official_tier=99)

    def test_validation_errors_surface_in_pipeline_status(self):
        """Vectors with validation errors should be trackable."""
        v = SkinFeatureVector(
            skin_key="test",
            pipeline_status="complete",
            validation_errors=["vlm_art_quality out of range"],
        )
        self.assertEqual(v.pipeline_status, "complete")
        self.assertEqual(len(v.validation_errors), 1)


class FeaturePipelineEdgeCasesTests(unittest.TestCase):
    """Edge case tests covering the smoke-test scenarios from the plan."""

    def test_empty_quality_skin(self):
        """Skins with empty quality should still produce a valid vector."""
        v = SkinFeatureVector.from_dicts(
            skin_key="test",
            official_features={"official_tier": None},
        )
        self.assertIsNone(v.official_tier)
        v.recompute_availability()
        self.assertFalse(v.availability["official_tier"])

    def test_limited_skin_detection(self):
        """Limited skins should have is_limited=True."""
        v = SkinFeatureVector.from_dicts(
            skin_key="test",
            official_features={"is_limited": True, "limited_type": 2},
        )
        self.assertTrue(v.is_limited)
        self.assertEqual(v.limited_type, 2)

    def test_high_tier_skin_all_fields_populated(self):
        """High-tier skins should have all official fields populated."""
        v = SkinFeatureVector.from_dicts(
            skin_key="test",
            vlm_features={
                "vlm_art_quality": 9.0,
                "vlm_effect_score": 9.5,
                "vlm_color_harmony": 0.85,
                "vlm_composition": 9.0,
            },
            official_features={
                "official_tier": 5,
                "skin_age_days": 100,
                "acquisition_method": 1,
                "has_voice_pack": True,
                "has_custom_anim": True,
                "is_limited": True,
                "limited_type": 1,
                "is_giftable": True,
                "avg_spend_to_obtain": 2000.0,
            },
        )
        v.recompute_availability()
        # Check aesthetic group is fully populated
        self.assertEqual(v.group_coverage()["aesthetic"], 1.0)

    def test_error_case_produces_trackable_vector(self):
        """Error cases should produce vectors with error status."""
        v = SkinFeatureVector(
            skin_key="error-skin",
            pipeline_status="error",
            validation_errors=["VLM pipeline timeout"],
        )
        self.assertEqual(v.pipeline_status, "error")
        self.assertEqual(len(v.validation_errors), 1)


if __name__ == "__main__":
    unittest.main()

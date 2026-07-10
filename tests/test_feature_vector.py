"""Unit tests for SkinFeatureVector — the 33-dimensional canonical feature vector."""

import unittest

from pydantic import ValidationError

from feature_engineering.features import (
    FEATURE_FIELD_NAMES,
    FEATURE_GROUPS,
    FIELD_RANGES,
    AcquisitionMethod,
    Provenance,
    SkinFeatureVector,
)


class SkinFeatureVectorFieldOrderTests(unittest.TestCase):
    """Verify the 33 fields are in the documented order."""

    def test_field_count_is_33(self):
        self.assertEqual(SkinFeatureVector.field_count(), 33)

    def test_field_names_length_is_33(self):
        self.assertEqual(len(FEATURE_FIELD_NAMES), 33)

    def test_groups_sum_to_33(self):
        total = sum(len(names) for names in FEATURE_GROUPS.values())
        self.assertEqual(total, 33)

    def test_group_sizes_match_documentation(self):
        expected = {
            "aesthetic": 8,
            "belonging": 8,
            "showing_off": 6,
            "collection": 7,
            "surprise": 4,
        }
        for group, size in expected.items():
            self.assertEqual(
                len(FEATURE_GROUPS[group]), size,
                f"Group {group} should have {size} fields",
            )

    def test_all_field_names_have_range_definitions(self):
        for name in FEATURE_FIELD_NAMES:
            self.assertIn(name, FIELD_RANGES,
                          f"Field {name} missing from FIELD_RANGES")


class SkinFeatureVectorToArrayTests(unittest.TestCase):
    """Tests for ``to_array()`` and ``availability_array()``."""

    def test_to_array_returns_33_values(self):
        v = SkinFeatureVector(skin_key="test")
        arr = v.to_array()
        self.assertEqual(len(arr), 33)

    def test_to_array_all_null_by_default(self):
        v = SkinFeatureVector(skin_key="test")
        arr = v.to_array()
        self.assertTrue(all(x is None for x in arr))

    def test_to_array_zero_policy_converts_null_to_zero(self):
        v = SkinFeatureVector(skin_key="test")
        arr = v.to_array(missing_policy="zero")
        self.assertTrue(all(x == 0 for x in arr))

    def test_to_array_rejects_invalid_policy(self):
        v = SkinFeatureVector(skin_key="test")
        with self.assertRaises(ValueError):
            v.to_array(missing_policy="drop")

    def test_to_array_preserves_non_null_values(self):
        v = SkinFeatureVector(
            skin_key="test",
            vlm_art_quality=8.5,
            official_tier=3,
            skin_age_days=200,
        )
        arr = v.to_array()
        # vlm_art_quality is index 0
        self.assertEqual(arr[0], 8.5)
        # official_tier is index 4
        self.assertEqual(arr[4], 3)
        # skin_age_days is index 7 (last aesthetic)
        self.assertEqual(arr[7], 200)

    def test_to_array_zero_policy_keeps_non_null_values(self):
        v = SkinFeatureVector(skin_key="test", vlm_art_quality=8.5)
        arr = v.to_array(missing_policy="zero")
        self.assertEqual(arr[0], 8.5)
        self.assertEqual(arr[1], 0)  # vlm_effect_score is null → 0


class SkinFeatureVectorAvailabilityTests(unittest.TestCase):
    """Tests for ``availability_array()``."""

    def test_availability_array_returns_33_values(self):
        v = SkinFeatureVector(skin_key="test")
        self.assertEqual(len(v.availability_array()), 33)

    def test_availability_all_zero_by_default(self):
        v = SkinFeatureVector(skin_key="test")
        self.assertTrue(all(x == 0 for x in v.availability_array()))

    def test_availability_reflects_populated_fields(self):
        v = SkinFeatureVector(
            skin_key="test",
            vlm_art_quality=8.5,
            official_tier=3,
            has_voice_pack=True,
        )
        self.assertEqual(sum(v.availability_array()), 3)

    def test_recompute_availability_updates_dict(self):
        v = SkinFeatureVector(skin_key="test")
        self.assertEqual(v.availability, {})
        v.recompute_availability()
        self.assertEqual(len(v.availability), 33)
        self.assertFalse(v.availability["vlm_art_quality"])

    def test_recompute_availability_detects_present(self):
        v = SkinFeatureVector(skin_key="test", vlm_art_quality=7.0,
                              renewal_count=3)
        v.recompute_availability()
        self.assertTrue(v.availability["vlm_art_quality"])


class SkinFeatureVectorValidationTests(unittest.TestCase):
    """Tests for range validation."""

    def test_default_vector_has_no_errors(self):
        v = SkinFeatureVector(skin_key="test")
        errors = v.validate_ranges()
        self.assertEqual(errors, [])

    def test_valid_vector_has_no_errors(self):
        v = SkinFeatureVector(
            skin_key="test",
            vlm_art_quality=8.5,
            vlm_effect_score=9.0,
            vlm_color_harmony=0.75,
            vlm_composition=8.0,
            official_tier=3,
            skin_age_days=200,
            acquisition_method=0,
        )
        errors = v.validate_ranges()
        self.assertEqual(errors, [])

    def test_pydantic_rejects_out_of_range_at_construction(self):
        with self.assertRaises(ValidationError):
            SkinFeatureVector(skin_key="test", vlm_art_quality=15.0)

    def test_pydantic_rejects_negative_values(self):
        with self.assertRaises(ValidationError):
            SkinFeatureVector(skin_key="test", vlm_effect_score=-1.0)

    def test_pydantic_rejects_invalid_official_tier(self):
        with self.assertRaises(ValidationError):
            SkinFeatureVector(skin_key="test", official_tier=10)

    def test_validate_ranges_catches_post_hoc_modification(self):
        v = SkinFeatureVector(skin_key="test")
        object.__setattr__(v, "vlm_art_quality", 15.0)
        errors = v.validate_ranges()
        self.assertGreater(len(errors), 0)
        self.assertTrue(any("15.0" in e for e in errors))

    def test_bool_fields_not_range_checked(self):
        v = SkinFeatureVector(skin_key="test", has_voice_pack=True,
                              has_custom_anim=False)
        errors = v.validate_ranges()
        self.assertEqual(errors, [])


class SkinFeatureVectorNullHandlingTests(unittest.TestCase):
    """Tests for null/None handling throughout the model."""

    def test_all_fields_nullable_by_default(self):
        v = SkinFeatureVector(skin_key="test")
        for name in FEATURE_FIELD_NAMES:
            self.assertIsNone(getattr(v, name),
                              f"Field {name} should be None by default")

    def test_null_fields_stay_null_in_array(self):
        v = SkinFeatureVector(skin_key="test")
        for val in v.to_array():
            self.assertIsNone(val)

    def test_null_fields_are_zero_in_availability(self):
        v = SkinFeatureVector(skin_key="test")
        for flag in v.availability_array():
            self.assertEqual(flag, 0)

    def test_boolean_false_is_still_present(self):
        v = SkinFeatureVector(skin_key="test", has_voice_pack=False)
        v.recompute_availability()
        self.assertTrue(v.availability["has_voice_pack"])
        self.assertEqual(v.availability_array()[5], 1)


class SkinFeatureVectorFromDictsTests(unittest.TestCase):
    """Tests for the ``from_dicts()`` factory method."""

    def test_from_dicts_basic(self):
        v = SkinFeatureVector.from_dicts(
            skin_key="test-001",
            skin_id="56304",
            hero_name="李白",
            skin_name="诗剑行",
        )
        self.assertEqual(v.skin_key, "test-001")
        self.assertEqual(v.hero_name, "李白")
        # All 33 fields should be None
        self.assertTrue(all(x is None for x in v.to_array()))

    def test_from_dicts_vlm_features_map_correctly(self):
        v = SkinFeatureVector.from_dicts(
            skin_key="test",
            vlm_features={
                "vlm_art_quality": 8.5,
                "vlm_effect_score": 9.0,
                "vlm_color_harmony": 0.75,
                "vlm_composition": 8.0,
            },
        )
        self.assertEqual(v.vlm_art_quality, 8.5)
        self.assertEqual(v.vlm_effect_score, 9.0)
        self.assertEqual(v.vlm_color_harmony, 0.75)
        self.assertEqual(v.vlm_composition, 8.0)

    def test_from_dicts_official_features(self):
        v = SkinFeatureVector.from_dicts(
            skin_key="test",
            official_features={
                "official_tier": 3,
                "skin_age_days": 200,
                "is_limited": True,
                "acquisition_method": 0,
            },
        )
        self.assertEqual(v.official_tier, 3)
        self.assertEqual(v.skin_age_days, 200)
        self.assertTrue(v.is_limited)
        self.assertEqual(v.acquisition_method, 0)

    def test_from_dicts_vlm_overrides_official_for_vlm_fields(self):
        v = SkinFeatureVector.from_dicts(
            skin_key="test",
            vlm_features={"vlm_art_quality": 9.0},
            official_features={"vlm_art_quality": 5.0},
        )
        self.assertEqual(v.vlm_art_quality, 9.0)

    def test_from_dicts_provenance_tracking(self):
        v = SkinFeatureVector.from_dicts(
            skin_key="test",
            vlm_features={"vlm_art_quality": 8.5},
            official_features={"official_tier": 3, "skin_age_days": 200},
        )
        self.assertEqual(v.provenance.get("vlm_art_quality"), "vlm_l2")
        self.assertEqual(v.provenance.get("official_tier"), "official")
        self.assertEqual(v.provenance.get("baidu_index_7d"), "unavailable")

    def test_from_dicts_availability(self):
        v = SkinFeatureVector.from_dicts(
            skin_key="test",
            vlm_features={"vlm_art_quality": 8.5},
        )
        v.recompute_availability()
        self.assertTrue(v.availability.get("vlm_art_quality"))
        self.assertFalse(v.availability.get("vlm_effect_score"))


class SkinFeatureVectorGroupCoverageTests(unittest.TestCase):
    """Tests for ``group_coverage()``."""

    def test_all_zero_coverage_by_default(self):
        v = SkinFeatureVector(skin_key="test")
        cov = v.group_coverage()
        self.assertEqual(cov["aesthetic"], 0.0)
        self.assertEqual(cov["belonging"], 0.0)

    def test_full_aesthetic_coverage(self):
        v = SkinFeatureVector(
            skin_key="test",
            vlm_art_quality=8.5,
            vlm_effect_score=9.0,
            vlm_color_harmony=0.75,
            vlm_composition=8.0,
            official_tier=3,
            has_voice_pack=True,
            has_custom_anim=False,
            skin_age_days=365,
        )
        cov = v.group_coverage()
        self.assertEqual(cov["aesthetic"], 1.0)

    def test_partial_coverage(self):
        v = SkinFeatureVector(
            skin_key="test",
            vlm_art_quality=8.5,
            official_tier=3,
    )
        cov = v.group_coverage()
        self.assertEqual(cov["aesthetic"], 2.0 / 8.0)


class SkinFeatureVectorEnumsTests(unittest.TestCase):
    """Verify enum definitions match documentation."""

    def test_acquisition_method_values(self):
        self.assertEqual(AcquisitionMethod.DIRECT, 0)
        self.assertEqual(AcquisitionMethod.GACHA, 1)
        self.assertEqual(AcquisitionMethod.BATTLE_PASS, 2)
        self.assertEqual(AcquisitionMethod.SHARD, 3)
        self.assertEqual(AcquisitionMethod.EVENT, 4)

    def test_acquisition_method_valid_in_feature_vector(self):
        v = SkinFeatureVector(skin_key="test", acquisition_method=3)
        self.assertEqual(v.acquisition_method, 3)


class SkinFeatureVectorMetadataTests(unittest.TestCase):
    """Tests for metadata fields outside the 33 dimensions."""

    def test_default_schema_version_is_2(self):
        v = SkinFeatureVector(skin_key="test")
        self.assertEqual(v.schema_version, 2)

    def test_default_pipeline_status_is_pending(self):
        v = SkinFeatureVector(skin_key="test")
        self.assertEqual(v.pipeline_status, "pending")

    def test_validation_errors_default_empty(self):
        v = SkinFeatureVector(skin_key="test")
        self.assertEqual(v.validation_errors, [])

    def test_vlm_raw_storage(self):
        v = SkinFeatureVector(
            skin_key="test",
            vlm_raw={"l1_confidence": 0.92, "l2_model_detail": 8.5},
        )
        self.assertEqual(v.vlm_raw["l1_confidence"], 0.92)

    def test_to_array_excludes_metadata(self):
        """Metadata fields should NOT appear in the 33-dimension array."""
        v = SkinFeatureVector(
            skin_key="meta-test",
            skin_id="extra",
            hero_name="hero",
            skin_name="skin",
            image_hash="abc123",
            pipeline_status="complete",
            vlm_raw={"extra": "data"},
        )
        arr = v.to_array()
        self.assertEqual(len(arr), 33)
        # None of the metadata values should leak into the array
        for val in arr:
            if val is not None:
                self.assertIsInstance(val, (int, float, bool))


class SkinFeatureVectorSerializationTests(unittest.TestCase):
    """Tests for model_dump / dict round-trip."""

    def test_model_dump_includes_identity_and_metadata(self):
        v = SkinFeatureVector(
            skin_key="test",
            skin_id="56304",
            hero_name="李白",
            vlm_art_quality=8.5,
            image_hash="abc123",
            pipeline_status="complete",
        )
        d = v.model_dump()
        self.assertEqual(d["skin_key"], "test")
        self.assertEqual(d["vlm_art_quality"], 8.5)
        self.assertEqual(d["image_hash"], "abc123")
        self.assertEqual(d["pipeline_status"], "complete")

    def test_model_dump_includes_none_by_default(self):
        v = SkinFeatureVector(skin_key="test")
        d = v.model_dump()
        self.assertIn("skin_key", d)
        # Pydantic includes default None values by default
        self.assertIn("vlm_art_quality", d)
        self.assertIsNone(d["vlm_art_quality"])

    def test_model_dump_exclude_none(self):
        v = SkinFeatureVector(skin_key="test")
        d = v.model_dump(exclude_none=True)
        self.assertNotIn("vlm_art_quality", d)


if __name__ == "__main__":
    unittest.main()

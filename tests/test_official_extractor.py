"""Unit tests for OfficialFeatureMapper — deterministic crawler→feature mappings."""

import unittest

from feature_engineering.official_extractor import (
    OfficialFeatureMapper,
    extract_acquisition,
    extract_from_intro,
    extract_official_tier,
    extract_skin_age,
)


class OfficialTierMappingTests(unittest.TestCase):
    """Tests for quality string → official_tier mapping."""

    def test_all_known_qualities(self):
        expected = {
            "伴生": 0,
            "勇者": 1,
            "史诗": 2,
            "传说": 3,
            "无双": 4,
            "荣耀典藏": 5,
        }
        for quality, tier in expected.items():
            value, prov = extract_official_tier(quality)
            self.assertEqual(value, tier,
                             f"{quality!r} should map to {tier}, got {value}")
            self.assertEqual(prov, "official:quality_map")

    def test_limited_variants(self):
        cases = {
            "勇者限定": 1,
            "史诗限定": 2,
            "传说限定": 3,
            "无双限定": 4,
        }
        for quality, tier in cases.items():
            value, _ = extract_official_tier(quality)
            self.assertEqual(value, tier)

    def test_unknown_quality_returns_none(self):
        value, prov = extract_official_tier("未知品质")
        self.assertIsNone(value)
        self.assertEqual(prov, "source:ambiguous")

    def test_empty_quality_returns_none(self):
        value, prov = extract_official_tier("")
        self.assertIsNone(value)
        self.assertEqual(prov, "source:ambiguous")

    def test_none_quality_returns_none(self):
        value, prov = extract_official_tier("")
        self.assertIsNone(value)

    def test_substring_match(self):
        """Quality strings containing known keywords should match."""
        value, _ = extract_official_tier("传说品质皮肤")
        self.assertEqual(value, 3)

    def test_blank_quality(self):
        value, prov = extract_official_tier("   ")
        self.assertIsNone(value)
        self.assertEqual(prov, "source:ambiguous")


class SkinAgeTests(unittest.TestCase):
    """Tests for online_date → skin_age_days."""

    def test_valid_date_yyyymmdd(self):
        age, prov = extract_skin_age("20240101")
        self.assertIsInstance(age, int)
        self.assertGreater(age, 0)
        self.assertEqual(prov, "official:date")

    def test_valid_date_with_dashes(self):
        age, _ = extract_skin_age("2024-01-01")
        self.assertIsInstance(age, int)
        self.assertGreater(age, 0)

    def test_valid_date_with_slashes(self):
        age, _ = extract_skin_age("2024/01/01")
        self.assertIsInstance(age, int)

    def test_valid_date_with_dots(self):
        age, _ = extract_skin_age("2024.01.01")
        self.assertIsInstance(age, int)

    def test_empty_date_returns_none(self):
        age, prov = extract_skin_age("")
        self.assertIsNone(age)
        self.assertEqual(prov, "source:ambiguous")

    def test_invalid_date_returns_none(self):
        age, prov = extract_skin_age("not-a-date")
        self.assertIsNone(age)
        self.assertEqual(prov, "source:ambiguous")

    def test_blank_date_returns_none(self):
        age, prov = extract_skin_age("   ")
        self.assertIsNone(age)


class AcquisitionTests(unittest.TestCase):
    """Tests for acquisition-method parsing."""

    def test_direct_purchase(self):
        result = extract_acquisition("60点券")
        self.assertEqual(result["acquisition_method"], 0)

    def test_direct_purchase_limited(self):
        result = extract_acquisition("1688点券 限时上架")
        self.assertEqual(result["acquisition_method"], 0)

    def test_gacha(self):
        result = extract_acquisition("荣耀水晶兑换")
        self.assertEqual(result["acquisition_method"], 1)

    def test_gacha_lottery(self):
        result = extract_acquisition("抽奖获得")
        self.assertEqual(result["acquisition_method"], 1)

    def test_battle_pass(self):
        result = extract_acquisition("战令进阶获取")
        self.assertEqual(result["acquisition_method"], 2)

    def test_shard_exchange(self):
        result = extract_acquisition("皮肤碎片兑换")
        self.assertEqual(result["acquisition_method"], 3)

    def test_event_free(self):
        result = extract_acquisition("活动赠送")
        self.assertEqual(result["acquisition_method"], 4)

    def test_empty_acquire_method(self):
        result = extract_acquisition("")
        self.assertIsNone(result["acquisition_method"])

    def test_unknown_acquire_method(self):
        result = extract_acquisition("某种未知方式")
        self.assertIsNone(result["acquisition_method"])

    def test_price_extraction(self):
        result = extract_acquisition("888点券", price_text="888点券")
        self.assertEqual(result["avg_spend_to_obtain"], 888.0)

    def test_price_with_decimal(self):
        result = extract_acquisition("", price_text="168.88")
        self.assertEqual(result["avg_spend_to_obtain"], 168.88)

    def test_price_empty(self):
        result = extract_acquisition("直购", price_text="")
        self.assertIsNone(result["avg_spend_to_obtain"])

    def test_is_limited_from_quality(self):
        result = extract_acquisition("888点券", quality="传说限定")
        self.assertTrue(result["is_limited"])

    def test_is_limited_from_acquire_text(self):
        result = extract_acquisition("限时抢购", quality="史诗")
        self.assertTrue(result["is_limited"])

    def test_not_limited(self):
        result = extract_acquisition("888点券", quality="史诗")
        self.assertFalse(result["is_limited"])

    def test_limited_type_anniversary(self):
        result = extract_acquisition("", quality="周年庆限定")
        self.assertEqual(result["limited_type"], 0)

    def test_limited_type_season(self):
        result = extract_acquisition("", quality="赛季限定")
        self.assertEqual(result["limited_type"], 4)

    def test_is_giftable(self):
        result = extract_acquisition("可赠送好友")
        self.assertTrue(result["is_giftable"])

    def test_not_giftable(self):
        result = extract_acquisition("888点券")
        self.assertIsNone(result["is_giftable"])


class IntroExtractionTests(unittest.TestCase):
    """Tests for intro-text feature extraction."""

    def test_has_voice_pack(self):
        result = extract_from_intro("含独立语音包，知名声优配音")
        self.assertTrue(result["has_voice_pack"])

    def test_no_voice_pack(self):
        result = extract_from_intro("精美皮肤设计")
        self.assertIsNone(result["has_voice_pack"])

    def test_has_custom_anim(self):
        result = extract_from_intro("拥有专属回城特效和待机动作")
        self.assertTrue(result["has_custom_anim"])

    def test_no_custom_anim(self):
        result = extract_from_intro("普通皮肤")
        self.assertIsNone(result["has_custom_anim"])

    def test_empty_intro(self):
        result = extract_from_intro("")
        self.assertIsNone(result["has_voice_pack"])
        self.assertIsNone(result["has_custom_anim"])

    def test_series_membership_detected(self):
        result = extract_from_intro("属于山海经系列皮肤")
        self.assertIsNotNone(result["series_membership"])


class FullMapperTests(unittest.TestCase):
    """Integration tests for OfficialFeatureMapper.extract_all()."""

    def setUp(self):
        self.mapper = OfficialFeatureMapper()

    def test_extract_all_legendary_skin(self):
        features, prov = self.mapper.extract_all(
            quality="传说",
            online_date="20240101",
            acquire_method="1688点券 限时上架",
            price_text="1688点券",
            intro="全新传说皮肤，含专属语音和回城特效",
            hero_name="李白",
            skin_name="诗剑行",
        )
        self.assertEqual(features["official_tier"], 3)
        self.assertIsInstance(features["skin_age_days"], int)
        self.assertEqual(features["acquisition_method"], 0)
        self.assertIsNotNone(features["has_voice_pack"])
        self.assertEqual(prov["official_tier"], "official:quality_map")
        self.assertEqual(prov["skin_age_days"], "official:date")

    def test_extract_all_missing_fields(self):
        features, prov = self.mapper.extract_all(
            quality="",
            online_date="",
            acquire_method="",
            intro="",
        )
        self.assertIsNone(features["official_tier"])
        self.assertIsNone(features["skin_age_days"])
        self.assertIsNone(features["acquisition_method"])
        self.assertEqual(prov["baidu_index_7d"], "unavailable")
        self.assertEqual(prov["character_usage_rate"], "unavailable")

    def test_extract_all_provenance_coverage(self):
        """Every non-VLM feature should have a provenance entry."""
        from feature_engineering.features import FEATURE_FIELD_NAMES

        vlm_fields = {"vlm_art_quality", "vlm_effect_score",
                      "vlm_color_harmony", "vlm_composition"}
        _, prov = self.mapper.extract_all()
        for name in FEATURE_FIELD_NAMES:
            if name in vlm_fields:
                continue  # VLM fields come from VlmPipeline, not official extractor
            self.assertIn(name, prov,
                          f"Provenance missing for field {name}")

    def test_extract_all_feature_coverage(self):
        """Every non-VLM feature should have a key in the output."""
        from feature_engineering.features import FEATURE_FIELD_NAMES

        vlm_fields = {"vlm_art_quality", "vlm_effect_score",
                      "vlm_color_harmony", "vlm_composition"}
        features, _ = self.mapper.extract_all()
        for name in FEATURE_FIELD_NAMES:
            if name in vlm_fields:
                continue
            self.assertIn(name, features,
                          f"Feature missing for field {name}")


if __name__ == "__main__":
    unittest.main()

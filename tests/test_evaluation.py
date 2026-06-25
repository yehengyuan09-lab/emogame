import sqlite3
import tempfile
import unittest
from datetime import date
from pathlib import Path

from crawlers.wzry_skin_crawler import AssetRecord, HeroRecord, SkinRecord, ensure_schema
from crawlers.wzry_skin_crawler import save_asset, save_hero, save_skin
from data.skin_repository import SkinRepository
from feature_engineering.features import MarketValidationSignals
from feature_engineering.pipeline import FeatureBuilder
from models.rule_engine import RuleEngine


class EvaluationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp.name) / "skins.sqlite3"
        conn = sqlite3.connect(self.db_path)
        try:
            ensure_schema(conn)
            save_hero(
                conn,
                HeroRecord(
                    hero_id="105",
                    hero_name="廉颇",
                    id_name="lianpo",
                    title="正义爆轰",
                    hero_type_code=3,
                    hero_type="坦克",
                    skin_names=["地狱岩魂"],
                    raw_json={"ename": 105, "cname": "廉颇"},
                ),
            )
            save_skin(
                conn,
                SkinRecord(
                    source_key="105-01",
                    source_index=1,
                    hero_id="105",
                    hero_name="廉颇",
                    skin_index=1,
                    skin_id="10501",
                    skin_name="地狱岩魂",
                    quality="史诗限定",
                    online_date="2024-06-01",
                    intro="sample",
                    acquire_method="商城限时直售获取",
                    price_text=None,
                    image_url="https://example.com/skin.jpg",
                    detail_url="https://example.com/detail.html",
                    mobile_url="",
                    video_id="",
                    catalog_source="herolist",
                    detail_source="heroskinlist",
                    has_detail_record=True,
                    raw_catalog_json={},
                    raw_detail_json={},
                ),
                Path("data/wzry_skins/images/105-01.jpg"),
            )
            save_asset(
                conn,
                AssetRecord(
                    source_key="105-01",
                    asset_type="skin_primary",
                    remote_url="https://example.com/skin.jpg",
                    local_path="data/wzry_skins/images/105-01.jpg",
                    content_hash="abc123",
                    download_status="downloaded",
                ),
            )
            conn.commit()
        finally:
            conn.close()
        self.repo = SkinRepository(self.db_path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_feature_builder_maps_official_fields(self):
        features = FeatureBuilder(self.repo, reference_date=date(2024, 7, 1)).build("105-01")

        self.assertEqual(features.official_tier, 3)
        self.assertTrue(features.is_limited)
        self.assertTrue(features.is_direct_sale)
        self.assertTrue(features.has_primary_asset)
        self.assertEqual(features.skin_age_days, 30)

    def test_rule_engine_marks_missing_market_validation(self):
        features = FeatureBuilder(self.repo, reference_date=date(2024, 7, 1)).build("105-01")
        result = RuleEngine().evaluate(features)

        self.assertEqual(result.validation_status, "insufficient_market_evidence")
        self.assertIn("insufficient_public_opinion_evidence", result.warnings)
        self.assertIsNone(result.evaluation_score)
        self.assertGreater(result.official_prior_score, 0)

    def test_market_signals_raise_confidence_and_validate(self):
        builder = FeatureBuilder(self.repo, reference_date=date(2024, 7, 1))
        baseline = RuleEngine().evaluate(builder.build("105-01"))
        features = builder.build(
            "105-01",
            MarketValidationSignals(
                visual_score=0.75,
                feel_score=0.82,
                craftsmanship_score=0.78,
                collection_score=0.66,
                value_score=0.72,
                purchase_intent_score=0.80,
                sentiment_score=0.85,
                discussion_count=8000,
                video_views=2_000_000,
                marketing_volume=3000,
                sales_volume=120_000,
                avg_spend_to_obtain=180,
                ownership_rate=0.22,
            ),
        )
        result = RuleEngine().evaluate(features)

        self.assertEqual(result.validation_status, "evidence_validated")
        self.assertGreater(result.confidence, baseline.confidence)
        self.assertIsNotNone(result.evaluation_score)
        self.assertGreater(result.aspect_scores["in_game_feel"], 0)


if __name__ == "__main__":
    unittest.main()

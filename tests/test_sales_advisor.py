import sqlite3
import tempfile
import unittest
from pathlib import Path

from business.sales_advisor import SalesAdvisor
from crawlers.wzry_skin_crawler import AssetRecord, HeroRecord, SkinRecord, ensure_schema
from crawlers.wzry_skin_crawler import save_asset, save_hero, save_skin
from data.skin_repository import SkinRepository
from feature_engineering.features import MarketValidationSignals
from feature_engineering.pipeline import FeatureBuilder
from models.rule_engine import RuleEngine


class SalesAdvisorTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp.name) / "skins.sqlite3"
        conn = sqlite3.connect(self.db_path)
        try:
            ensure_schema(conn)
            save_hero(
                conn,
                HeroRecord(
                    hero_id="107",
                    hero_name="赵云",
                    id_name="zhaoyun",
                    title="苍天翔龙",
                    hero_type_code=4,
                    hero_type="刺客",
                    skin_names=["龙胆"],
                    raw_json={},
                ),
            )
            save_skin(
                conn,
                SkinRecord(
                    source_key="107-08",
                    source_index=8,
                    hero_id="107",
                    hero_name="赵云",
                    skin_index=8,
                    skin_id="10708",
                    skin_name="龙胆",
                    quality="史诗限定",
                    online_date="2020-05-05",
                    intro="sample",
                    acquire_method="商城限时直售获取",
                    price_text="888点券",
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
                Path("data/wzry_skins/images/107-08.jpg"),
            )
            save_asset(
                conn,
                AssetRecord(
                    source_key="107-08",
                    asset_type="skin_primary",
                    remote_url="https://example.com/skin.jpg",
                    local_path="data/wzry_skins/images/107-08.jpg",
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

    def _advise(self, signals: MarketValidationSignals):
        features = FeatureBuilder(self.repo).build("107-08", signals)
        evaluation = RuleEngine().evaluate(features)
        return SalesAdvisor().advise(features, evaluation)

    def test_collect_more_evidence_when_validation_missing(self):
        report = self._advise(MarketValidationSignals())

        self.assertEqual(report.decision, "collect_more_evidence")
        self.assertIn("missing_sales_volume", report.evidence_gaps)

    def test_scale_marketing_for_validated_purchase_intent(self):
        report = self._advise(
            MarketValidationSignals(
                visual_score=0.85,
                feel_score=0.8,
                craftsmanship_score=0.82,
                collection_score=0.8,
                value_score=0.72,
                purchase_intent_score=0.83,
                sentiment_score=0.88,
                discussion_count=12000,
                video_views=2_000_000,
                marketing_volume=50000,
                sales_volume=150000,
                ownership_rate=0.15,
            )
        )

        self.assertEqual(report.decision, "scale_marketing")
        self.assertGreaterEqual(report.sales_readiness, 80)
        self.assertEqual(report.pricing_guidance["posture"], "protect_premium")

    def test_price_blocker_changes_decision(self):
        report = self._advise(
            MarketValidationSignals(
                visual_score=0.8,
                feel_score=0.74,
                craftsmanship_score=0.78,
                collection_score=0.72,
                value_score=0.3,
                purchase_intent_score=0.76,
                discussion_count=9000,
                video_views=1_500_000,
                marketing_volume=30000,
                sales_volume=90000,
            )
        )

        self.assertEqual(report.decision, "fix_price_or_bundle")
        self.assertEqual(report.pricing_guidance["posture"], "discount_or_bundle")


if __name__ == "__main__":
    unittest.main()

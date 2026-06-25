import sqlite3
import tempfile
import unittest
from pathlib import Path

from app import build_payload
from crawlers.wzry_skin_crawler import AssetRecord, HeroRecord, SkinRecord, ensure_schema
from crawlers.wzry_skin_crawler import save_asset, save_hero, save_skin
from feature_engineering.features import MarketValidationSignals


class StreamlitAppHelperTest(unittest.TestCase):
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

    def tearDown(self):
        self.tmp.cleanup()

    def test_build_payload_contains_sales_report(self):
        payload = build_payload(
            self.db_path,
            "107-08",
            MarketValidationSignals(
                visual_score=0.8,
                feel_score=0.76,
                craftsmanship_score=0.78,
                collection_score=0.74,
                value_score=0.72,
                purchase_intent_score=0.77,
                discussion_count=8000,
            ),
        )

        self.assertEqual(payload["skin"]["source_key"], "107-08")
        self.assertIn("sales_report", payload)
        self.assertIn("evaluation", payload)
        self.assertIn("sales_gap", payload)


if __name__ == "__main__":
    unittest.main()

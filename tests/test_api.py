import sqlite3
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from api.main import app
from crawlers.wzry_skin_crawler import AssetRecord, HeroRecord, SkinRecord, ensure_schema
from crawlers.wzry_skin_crawler import save_asset, save_hero, save_skin


class ApiTest(unittest.TestCase):
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
        self.client = TestClient(app)

    def tearDown(self):
        self.tmp.cleanup()

    def test_health(self):
        response = self.client.get("/api/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_list_skins(self):
        response = self.client.get("/api/skins", params={"db": str(self.db_path), "search": "龙胆"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["skins"][0]["source_key"], "107-08")

    def test_evaluate_with_inline_signals(self):
        response = self.client.post(
            "/api/evaluate",
            params={"db": str(self.db_path)},
            json={
                "source_key": "107-08",
                "signals": {
                    "visual_score": 0.8,
                    "feel_score": 0.75,
                    "craftsmanship_score": 0.78,
                    "collection_score": 0.72,
                    "value_score": 0.7,
                    "purchase_intent_score": 0.76,
                    "discussion_count": 8000,
                },
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["evaluation"]["validation_status"], "evidence_validated")

    def test_sales_report(self):
        response = self.client.post(
            "/api/sales-report",
            params={"db": str(self.db_path)},
            json={"source_key": "107-08", "ignore_db_signals": True},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["sales_report"]["decision"], "collect_more_evidence")


if __name__ == "__main__":
    unittest.main()

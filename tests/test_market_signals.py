import sqlite3
import tempfile
import unittest
from pathlib import Path

from crawlers.bilibili_evidence import extract_bvid, parse_bilibili_search_results
from crawlers.wzry_skin_crawler import HeroRecord, SkinRecord, ensure_schema
from crawlers.wzry_skin_crawler import save_hero, save_skin
from data.market_signal_repository import MarketSignalRepository
from data.skin_repository import SkinRepository
from feature_engineering.pipeline import FeatureBuilder
from models.rule_engine import RuleEngine
from scripts.import_market_signals import import_payload


class MarketSignalRepositoryTest(unittest.TestCase):
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
                    raw_json={},
                ),
            )
            save_skin(
                conn,
                SkinRecord(
                    source_key="105-02",
                    source_index=2,
                    hero_id="105",
                    hero_name="廉颇",
                    skin_index=2,
                    skin_id="10501",
                    skin_name="地狱岩魂",
                    quality="史诗",
                    online_date="2024-06-01",
                    intro="sample",
                    acquire_method="商城直售获取",
                    price_text=None,
                    image_url="",
                    detail_url="",
                    mobile_url="",
                    video_id="",
                    catalog_source="herolist",
                    detail_source="heroskinlist",
                    has_detail_record=True,
                    raw_catalog_json={},
                    raw_detail_json={},
                ),
                None,
            )
            conn.commit()
        finally:
            conn.close()
        self.market_repo = MarketSignalRepository(self.db_path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_extract_bvid(self):
        self.assertEqual(extract_bvid("https://www.bilibili.com/video/BV1xx411c7mD"), "BV1xx411c7mD")
        self.assertEqual(extract_bvid("BV1xx411c7mD"), "BV1xx411c7mD")

    def test_parse_bilibili_search_results(self):
        payload = {
            "code": 0,
            "data": {
                "result": [
                    {"result_type": "tips", "data": []},
                    {
                        "result_type": "video",
                        "data": [
                            {
                                "bvid": "BV1xx411c7mD",
                                "title": "<em class=\"keyword\">王者荣耀</em> 皮肤测评",
                                "author": "UP主",
                                "play": "1,234",
                                "danmaku": 12,
                                "description": "赵云 新皮肤 手感",
                            }
                        ],
                    },
                ]
            },
        }

        results = parse_bilibili_search_results(payload)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].bvid, "BV1xx411c7mD")
        self.assertEqual(results[0].title, "王者荣耀 皮肤测评")
        self.assertEqual(results[0].play, 1234)

    def test_manual_import_and_evaluation(self):
        import_payload(
            self.db_path,
            {
                "105-02": {
                    "signals": {
                        "visual_score": 0.8,
                        "feel_score": 0.7,
                        "craftsmanship_score": 0.75,
                        "collection_score": 0.6,
                        "value_score": 0.65,
                        "purchase_intent_score": 0.7,
                    }
                }
            },
        )

        signals = self.market_repo.get_signals("105-02")
        features = FeatureBuilder(SkinRepository(self.db_path)).build("105-02", signals)
        result = RuleEngine().evaluate(features)

        self.assertEqual(result.validation_status, "evidence_validated")
        self.assertIsNotNone(result.evaluation_score)

    def test_evidence_aggregation(self):
        self.market_repo.add_evidence(
            "105-02",
            platform="bilibili",
            external_id="BV1xx411c7mD",
            url="https://www.bilibili.com/video/BV1xx411c7mD",
            title="皮肤评测",
            metrics={
                "view": 1000,
                "danmaku": 10,
                "reply": 20,
                "favorite": 30,
                "coin": 40,
                "share": 50,
                "like": 60,
            },
            aspect_tags=["visual", "feel"],
        )
        signals = self.market_repo.aggregate_evidence_signals("105-02")

        self.assertEqual(signals.video_views, 1000)
        self.assertEqual(signals.discussion_count, 30)
        self.assertEqual(signals.marketing_volume, 180)
        self.assertEqual(len(self.market_repo.list_evidence("105-02")), 1)


if __name__ == "__main__":
    unittest.main()

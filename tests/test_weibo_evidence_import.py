import sqlite3
import tempfile
import unittest
from pathlib import Path

from crawlers.wzry_skin_crawler import HeroRecord, SkinRecord, ensure_schema
from crawlers.wzry_skin_crawler import save_hero, save_skin
from data.market_signal_repository import MarketSignalRepository
from scripts.import_weibo_evidence import import_weibo_payload


class WeiboEvidenceImportTest(unittest.TestCase):
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

    def tearDown(self):
        self.tmp.cleanup()

    def test_import_weibo_comments_as_market_evidence(self):
        payload = [
            {
                "mid": "123",
                "title": "新皮肤预告",
                "comments": [
                    {
                        "user": "a",
                        "text": "这个皮肤特效好看，必买",
                        "like_count": 10,
                        "total_number": 2,
                        "created_at": "2026-06-23",
                    },
                    {
                        "user": "b",
                        "text": "局内手感和音效都不错，入手",
                        "like_count": 5,
                        "total_number": 1,
                        "created_at": "2026-06-23",
                    },
                    {
                        "user": "c",
                        "text": "价格有点贵但限定收藏可以冲",
                        "like_count": 7,
                        "total_number": 0,
                        "created_at": "2026-06-23",
                    },
                ],
            }
        ]

        result = import_weibo_payload(self.db_path, "105-02", payload, min_aspect_comments=1)

        self.assertEqual(result["evidence"], 3)
        repo = MarketSignalRepository(self.db_path)
        evidence = repo.list_evidence("105-02")
        signals = repo.get_signals("105-02")

        self.assertEqual(len(evidence), 3)
        self.assertEqual(signals.discussion_count, 6)
        self.assertEqual(signals.marketing_volume, 22)
        self.assertIsNotNone(signals.visual_score)
        self.assertIsNotNone(signals.feel_score)
        self.assertIsNotNone(signals.purchase_intent_score)


if __name__ == "__main__":
    unittest.main()

import sqlite3
import tempfile
import unittest
from pathlib import Path

from crawlers.wzry_skin_crawler import AssetRecord, HeroRecord, SkinRecord, ensure_schema
from crawlers.wzry_skin_crawler import save_asset, save_hero, save_skin
from data.skin_repository import SkinRepository


class SkinRepositoryTest(unittest.TestCase):
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
                    skin_names=["正义爆轰", "地狱岩魂"],
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
                    skin_id="",
                    skin_name="正义爆轰",
                    quality="",
                    online_date="",
                    intro="",
                    acquire_method="",
                    price_text=None,
                    image_url="",
                    detail_url="",
                    mobile_url="",
                    video_id="",
                    catalog_source="herolist",
                    detail_source=None,
                    has_detail_record=False,
                    raw_catalog_json={},
                    raw_detail_json={},
                ),
                None,
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
                Path("data/wzry_skins/images/105-02.jpg"),
            )
            save_asset(
                conn,
                AssetRecord(
                    source_key="105-02",
                    asset_type="skin_primary",
                    remote_url="https://example.com/skin.jpg",
                    local_path="data/wzry_skins/images/105-02.jpg",
                    content_hash="abc123",
                    download_status="downloaded",
                ),
            )
            save_asset(
                conn,
                AssetRecord(
                    source_key="105-02",
                    asset_type="hero_icon",
                    remote_url="https://example.com/icon.jpg",
                    download_status="failed",
                    error="timeout",
                ),
            )
            conn.commit()
        finally:
            conn.close()
        self.repo = SkinRepository(self.db_path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_stats(self):
        stats = self.repo.stats()

        self.assertEqual(stats["heroes"], 1)
        self.assertEqual(stats["skins"], 2)
        self.assertEqual(stats["assets"], 2)
        self.assertEqual(stats["with_detail"], 1)
        self.assertEqual(stats["missing_detail"], 1)
        self.assertEqual(stats["missing_primary_asset"], 1)
        self.assertEqual(stats["failed_assets"], 1)

    def test_list_heroes_filter(self):
        heroes = self.repo.list_heroes(hero_type="坦克")

        self.assertEqual(len(heroes), 1)
        self.assertEqual(heroes[0]["hero_name"], "廉颇")

    def test_list_skins_filters(self):
        detail_skins = self.repo.list_skins(has_detail_record=True)
        missing_asset = self.repo.missing_primary_asset_skins()

        self.assertEqual([row["source_key"] for row in detail_skins], ["105-02"])
        self.assertEqual([row["source_key"] for row in missing_asset], ["105-01"])

    def test_search_get_and_assets(self):
        search_rows = self.repo.search_skins("地狱")
        skin = self.repo.get_skin("105-02")
        assets = self.repo.list_assets("105-02")
        failed_assets = self.repo.failed_assets()

        self.assertEqual(search_rows[0]["source_key"], "105-02")
        self.assertIsNotNone(skin)
        self.assertEqual(skin["primary_asset_status"], "downloaded")
        self.assertEqual(len(assets), 2)
        self.assertEqual(failed_assets[0]["error"], "timeout")


if __name__ == "__main__":
    unittest.main()

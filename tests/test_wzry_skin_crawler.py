import unittest

from crawlers.wzry_skin_crawler import asset_candidates, merge_catalog, parse_price_text


class WzrySkinCrawlerTest(unittest.TestCase):
    def test_parse_price_text(self):
        self.assertEqual(parse_price_text("60点券限时秒杀"), "60点券")
        self.assertEqual(parse_price_text("荣耀水晶兑换"), None)
        self.assertEqual(parse_price_text("288皮肤碎片兑换 / 60点券限时秒杀"), "288皮肤碎片 / 60点券")

    def test_merge_catalog_uses_hero_list_as_baseline(self):
        raw_heroes = [
            {
                "ename": 105,
                "cname": "廉颇",
                "id_name": "lianpo",
                "title": "正义爆轰",
                "hero_type": 3,
                "skin_name": "正义爆轰|地狱岩魂|无尽征程",
            }
        ]
        raw_details = [
            {
                "pfidlb_3934": "1001",
                "pfmclb_7523": "地狱岩魂",
                "yxmclb_9965": "廉颇",
                "sxsjlb_1516": "20240601",
                "pfpzlb_3289": "史诗",
                "hqfs_8609": "商城直售获取",
                "fmlb_4536": "https://example.com/skin.jpg",
            }
        ]

        heroes, skins = merge_catalog(raw_heroes, raw_details)

        self.assertEqual(len(heroes), 1)
        self.assertEqual(len(skins), 3)
        self.assertEqual([skin.skin_name for skin in skins], ["正义爆轰", "地狱岩魂", "无尽征程"])
        self.assertFalse(skins[0].has_detail_record)
        self.assertTrue(skins[1].has_detail_record)
        self.assertEqual(skins[1].skin_id, "1001")
        self.assertEqual(skins[1].online_date, "2024-06-01")
        self.assertEqual(skins[1].quality, "史诗")

    def test_merge_catalog_keeps_detail_only_records(self):
        raw_heroes = []
        raw_details = [
            {
                "pfidlb_3934": "2001",
                "pfmclb_7523": "新增皮肤",
                "yxmclb_9965": "新英雄",
                "fmlb_4536": "https://example.com/new.jpg",
            }
        ]

        _, skins = merge_catalog(raw_heroes, raw_details)

        self.assertEqual(len(skins), 1)
        self.assertEqual(skins[0].catalog_source, "detail_only")
        self.assertTrue(skins[0].has_detail_record)

    def test_asset_candidates_deduplicate_urls(self):
        _, skins = merge_catalog(
            [
                {
                    "ename": 105,
                    "cname": "廉颇",
                    "hero_type": 3,
                    "skin_name": "地狱岩魂",
                }
            ],
            [
                {
                    "pfidlb_3934": "1001",
                    "pfmclb_7523": "地狱岩魂",
                    "yxmclb_9965": "廉颇",
                    "fmlb_4536": "https://example.com/skin.jpg",
                    "fmb1lb_5300": "https://example.com/skin.jpg",
                    "yxtxlb_8443": "https://example.com/icon.jpg",
                }
            ],
        )

        assets = asset_candidates(skins[0])

        self.assertEqual([asset.asset_type for asset in assets], ["skin_primary", "hero_icon"])


if __name__ == "__main__":
    unittest.main()

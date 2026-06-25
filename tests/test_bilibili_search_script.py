import unittest

from scripts.search_bilibili_evidence import (
    build_skin_query,
    is_relevant,
    is_relevant_to_skin,
    skin_name_terms,
)


class BilibiliSearchScriptTest(unittest.TestCase):
    def test_build_skin_query(self):
        query = build_skin_query({"hero_name": "赵云", "skin_name": "龙胆"})

        self.assertEqual(query, "王者荣耀 赵云 龙胆 皮肤")

    def test_relevance_filter_matches_any_skin_term(self):
        self.assertTrue(is_relevant("赵云新皮肤手感测评", ["赵云", "龙胆"]))
        self.assertFalse(is_relevant("鲁班七号电玩小子测评", ["赵云", "龙胆"]))

    def test_skin_relevance_requires_skin_term_when_available(self):
        self.assertTrue(
            is_relevant_to_skin("赵云·龙胆 五虎上将限定皮肤爆料", hero_name="赵云", skin_name="龙胆")
        )
        self.assertFalse(
            is_relevant_to_skin("赵云-飞龙神将 传说皮肤爆料", hero_name="赵云", skin_name="龙胆")
        )

    def test_skin_name_terms_split_alias_punctuation(self):
        self.assertEqual(skin_name_terms("颠倒童话·魔镜"), ["颠倒童话", "魔镜"])


if __name__ == "__main__":
    unittest.main()

import unittest

from app import assets_with_character_names
from server.support import sync_generated_multiview_names


class AssetCharacterNameTests(unittest.TestCase):
    def test_current_bindings_and_old_multiview_results_are_annotated(self):
        state = {
            "projects": {"p": {"characters": [
                {"id": "c1", "name": "小红帽", "portrait_asset": "portrait",
                 "multiview_asset": "new-sheet", "voice_asset": "voice"},
            ]}},
            "assets": {key: {"id": key, "project_id": "p", "name": key}
                       for key in ("portrait", "new-sheet", "old-sheet", "voice", "place")},
            "runs": {"r": {"kind": "character_multiview", "project_id": "p",
                              "character_id": "c1", "asset_id": "old-sheet"}},
        }
        result = {item["id"]: item for item in assets_with_character_names(state, "p")}
        for asset_id in ("portrait", "new-sheet", "old-sheet", "voice"):
            self.assertEqual(result[asset_id]["character_names"], ["小红帽"])
        self.assertEqual(result["place"]["character_names"], [])

    def test_shared_asset_lists_each_character_once(self):
        state = {
            "projects": {"p": {"characters": [
                {"id": "a", "name": "甲", "portrait_asset": "shared"},
                {"id": "b", "name": "乙", "portrait_asset": "shared",
                 "multiview_asset": "shared"},
            ]}},
            "assets": {"shared": {"id": "shared", "project_id": "p"}},
            "runs": {},
        }
        result = assets_with_character_names(state, "p")
        self.assertEqual(result[0]["character_names"], ["甲", "乙"])

    def test_generated_sheet_follows_character_rename_but_custom_asset_name_stays(self):
        state = {
            "projects": {"p": {"characters": [{"id": "c", "name": "阎海文",
                                           "multiview_asset": "new"}]}},
            "assets": {
                "new": {"id": "new", "project_id": "p", "name": "角色 7 · 四视图",
                        "generated_by": "xb_qwen69_multiview"},
                "old": {"id": "old", "project_id": "p", "name": "旧名 · 四视图",
                        "generated_by": "xb_qwen69_multiview"},
                "custom": {"id": "custom", "project_id": "p", "name": "自定义素材名",
                           "generated_by": "xb_qwen69_multiview"},
            },
            "runs": {
                "r1": {"kind": "character_multiview", "project_id": "p",
                       "character_id": "c", "asset_id": "old"},
                "r2": {"kind": "character_multiview", "project_id": "p",
                       "character_id": "c", "asset_id": "custom"},
            },
        }
        shown = {item["id"]: item["name"] for item in assets_with_character_names(state, "p")}
        self.assertEqual(shown["new"], "阎海文 · 四视图")
        self.assertEqual(shown["old"], "阎海文 · 四视图")
        self.assertEqual(shown["custom"], "自定义素材名")
        self.assertEqual(sync_generated_multiview_names(state, "p"), 2)
        self.assertEqual(state["assets"]["new"]["name"], "阎海文 · 四视图")


if __name__ == "__main__":
    unittest.main()

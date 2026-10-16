import unittest

from app import assets_with_character_names


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


if __name__ == "__main__":
    unittest.main()

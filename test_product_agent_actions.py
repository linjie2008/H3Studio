import copy
import unittest
from unittest.mock import patch

import product_agent_actions as actions
from store import new_project


class ProductAgentActionTests(unittest.TestCase):
    def setUp(self):
        self.project = new_project("测试项目")
        self.project["id"] = "project-test"
        self.project["_data_version"] = 7
        self.project["shots"][0]["id"] = "shot-one"
        self.data = {"projects": {self.project["id"]: self.project}, "assets": {}, "runs": {}}
        actions.PENDING.clear()

    def _change(self, callback):
        return callback(self.data)

    def test_proposal_does_not_write_and_confirmation_is_one_time(self):
        before = copy.deepcopy(self.data)
        with patch.object(actions, "read", return_value=self.data), patch.object(actions, "change", side_effect=self._change):
            proposal = actions.propose("project-test", {"name": "set_shot_field", "args": {
                "shot_id": "shot-one", "field": "duration", "value": 6}})
            self.assertEqual(self.data, before)
            applied = actions.confirm(proposal["token"], "project-test")
            self.assertEqual(applied["project"]["shots"][0]["duration"], 6)
            self.assertEqual(applied["project"]["_data_version"], 8)
            with self.assertRaises(ValueError):
                actions.confirm(proposal["token"], "project-test")

    def test_stale_proposal_rejected(self):
        with patch.object(actions, "read", return_value=self.data), patch.object(actions, "change", side_effect=self._change):
            proposal = actions.propose("project-test", {"name": "rename_project", "args": {"name": "新名称"}})
            self.project["_data_version"] = 8
            with self.assertRaisesRegex(ValueError, "预览后已改变"):
                actions.confirm(proposal["token"], "project-test")
            self.assertEqual(self.project["name"], "测试项目")

    def test_foreign_asset_and_wrong_mode_rejected(self):
        asset = {"id": "image-1", "project_id": "other-project", "kind": "image", "name": "别人的图"}
        self.data["assets"][asset["id"]] = asset
        with patch.object(actions, "read", return_value=self.data):
            with self.assertRaisesRegex(ValueError, "不属于当前项目"):
                actions.propose("project-test", {"name": "bind_asset", "args": {
                    "shot_id": "shot-one", "asset_id": "image-1", "slot": "first_image"}})
            asset["project_id"] = "project-test"
            with self.assertRaisesRegex(ValueError, "只能绑定多参"):
                actions.propose("project-test", {"name": "bind_asset", "args": {
                    "shot_id": "shot-one", "asset_id": "image-1", "slot": "ref_image"}})

    def test_last_shot_cannot_be_deleted(self):
        with patch.object(actions, "read", return_value=self.data):
            with self.assertRaisesRegex(ValueError, "至少保留"):
                actions.propose("project-test", {"name": "delete_shot", "args": {"shot_id": "shot-one"}})


if __name__ == "__main__":
    unittest.main()

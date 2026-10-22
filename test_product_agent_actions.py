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

    def test_create_character_and_assign(self):
        with patch.object(actions, "read", return_value=self.data), patch.object(actions, "change", side_effect=self._change):
            prop = actions.propose("project-test", {"name": "create_character", "args": {
                "name": "李逍遥", "description": "蜀山少侠"}})
            self.assertIn("李逍遥", prop["summary"])
            applied = actions.confirm(prop["token"], "project-test")
            chars = applied["project"]["characters"]
            self.assertEqual(len(chars), 1)
            self.assertEqual(chars[0]["name"], "李逍遥")

    def test_reroll_shot(self):
        with patch.object(actions, "read", return_value=self.data), \
             patch.object(actions, "change", side_effect=self._change), \
             patch.object(actions.threading, "Thread") as thread:
            prop = actions.propose("project-test", {"name": "reroll_shot", "args": {
                "shot_id": "shot-one", "new_prompt": "雨夜特写"}})
            applied = actions.confirm(prop["token"], "project-test")
            shot = applied["project"]["shots"][0]
            self.assertEqual(shot["prompt"], "雨夜特写")
            self.assertNotEqual(shot["seed"], 42)
            self.assertEqual(self.data["runs"][applied["run_id"]]["shot_ids"], ["shot-one"])
            thread.return_value.start.assert_called_once()

    def test_create_storyboard_precomputed(self):
        with patch.object(actions, "read", return_value=self.data), patch.object(actions, "change", side_effect=self._change):
            prop = actions.propose("project-test", {
                "name": "create_storyboard",
                "args": {"brief": "李逍遥雨夜激战", "count": 2, "mode": "append"},
                "generated_shots": [
                    {"name": "镜头A", "duration": 4, "prompt": "rainy night temple", "speaker_id": "S1"},
                    {"name": "镜头B", "duration": 5, "prompt": "sword fight", "speaker_id": "S2"}
                ]
            })
            self.assertIn("生成了 2 个分镜", prop["summary"])
            applied = actions.confirm(prop["token"], "project-test")
            shots = applied["project"]["shots"]
            self.assertEqual(len(shots), 3)  # 1 original + 2 new
            self.assertEqual(shots[1]["name"], "镜头A")
            self.assertEqual(shots[2]["name"], "镜头B")

    def test_update_shot_changes_camera_and_duration_together(self):
        with patch.object(actions, "read", return_value=self.data), patch.object(actions, "change", side_effect=self._change):
            prop = actions.propose("project-test", {"name": "update_shot", "args": {
                "shot_id": "shot-one", "fields": {
                    "camera_move": "推近", "camera_speed": "缓慢", "duration": 4}}})
            self.assertIn("camera_move", prop["summary"])
            applied = actions.confirm(prop["token"], "project-test")
        shot = applied["project"]["shots"][0]
        self.assertEqual((shot["camera_move"], shot["camera_speed"], shot["duration"]),
                         ("推近", "缓慢", 4))

    def test_start_render_can_queue_already_selected_shot_after_save(self):
        self.assertTrue(self.project["shots"][0]["selected"])
        started = []
        class DeferredThread:
            def __init__(self, *, target, args, daemon):
                self.args = args
            def start(self):
                run_id = self.args[0]
                self.assert_saved(run_id)
                started.append(run_id)
            def assert_saved(self, run_id):
                assert run_id in self_data["runs"]
        self_data = self.data
        with patch.object(actions, "read", return_value=self.data), \
             patch.object(actions, "change", side_effect=self._change), \
             patch.object(actions.threading, "Thread", DeferredThread):
            prop = actions.propose("project-test", {"name": "start_render", "args": {}})
            applied = actions.confirm(prop["token"], "project-test")
        self.assertEqual(started, [applied["run_id"]])
        self.assertEqual(self.data["runs"][applied["run_id"]]["status"], "queued")


if __name__ == "__main__":
    unittest.main()

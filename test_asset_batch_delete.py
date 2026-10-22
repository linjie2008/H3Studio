"""素材批量删除的回归测试。

1. 一次删多个素材，并清掉镜头、角色、共享参考里的绑定。
2. 删除后 <Picture N> 编号必须按剩余图片重新计算一次。
3. 不允许删别的项目的素材。
4. 素材正被生成任务使用时拒绝删除。
"""
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from server.routes import assets as assets_route


class _FakeRequest:
    def __init__(self, path, body=None):
        self.path = path
        raw = json.dumps(body or {}).encode("utf-8")
        self.headers = {"Content-Length": str(len(raw))}
        self.rfile = io.BytesIO(raw)
        self.status = None
        self.payload = None

    def body_json(self):
        return json.loads(self.rfile.read(int(self.headers["Content-Length"])))

    def json(self, status, payload):
        self.status = status
        self.payload = payload

    def send_error(self, status):
        self.status = status


def _state():
    return {
        "projects": {
            "p1": {
                "id": "p1",
                "reference_image_numbers": {"a": 1, "b": 2, "c": 3},
                "image_descriptions": {"b": "服装图"},
                "shared_refs": {"images": ["a"], "videos": [], "audios": []},
                "characters": [{"id": "ch1", "portrait_asset": "c"}],
                "entities": [],
                "shots": [{
                    "id": "s1", "ref_images": ["a", "b"], "manual_ref_images": ["a"],
                    "ref_videos": [], "ref_audios": [], "first_image": None,
                    "last_image": None, "source_video": None, "h3_guides": [],
                    "prompt": "the face matches <Picture 1> and the outfit in <Picture 2>",
                }],
            },
            "p2": {"id": "p2", "shots": []},
        },
        "assets": {
            "a": {"id": "a", "name": "a.png", "file": "a.png", "kind": "image", "project_id": "p1"},
            "b": {"id": "b", "name": "b.png", "file": "b.png", "kind": "image", "project_id": "p1"},
            "c": {"id": "c", "name": "c.png", "file": "c.png", "kind": "image", "project_id": "p1"},
            "x": {"id": "x", "name": "x.png", "file": "x.png", "kind": "image", "project_id": "p2"},
        },
        "runs": {},
    }


class BatchAssetDeleteTests(unittest.TestCase):
    def _run(self, state, project_id, asset_ids, status=200):
        def fake_read():
            return state

        def fake_change(fn):
            return fn(state)

        request = _FakeRequest("/api/assets/batch-delete", {"project_id": project_id, "asset_ids": asset_ids})
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(assets_route, "ASSETS", Path(directory)), \
                    patch.object(assets_route, "read", fake_read), \
                    patch.object(assets_route, "change", fake_change):
                assets_route.post_api_assets_batch_delete(request, "/api/assets/batch-delete", [], None)
        self.assertEqual(request.status, status)
        return request

    def test_deletes_several_assets_and_clears_references(self):
        state = _state()
        request = self._run(state, "p1", ["a", "b"])
        self.assertEqual(request.payload["deleted"], 2)
        self.assertNotIn("a", state["assets"])
        self.assertNotIn("b", state["assets"])
        shot = state["projects"]["p1"]["shots"][0]
        self.assertEqual(shot["ref_images"], [])
        self.assertEqual(shot["manual_ref_images"], [])
        self.assertEqual(state["projects"]["p1"]["shared_refs"]["images"], [])
        self.assertNotIn("b", state["projects"]["p1"]["image_descriptions"])

    def test_remaining_pictures_are_renumbered_once(self):
        state = _state()
        self._run(state, "p1", ["a", "b"])
        numbers = state["projects"]["p1"]["reference_image_numbers"]
        self.assertNotIn("a", numbers)
        self.assertNotIn("b", numbers)
        self.assertEqual(numbers["c"], 1, "剩下的图片必须重新从 <Picture 1> 开始编号")

    def test_character_binding_is_cleared(self):
        state = _state()
        self._run(state, "p1", ["c"])
        self.assertIsNone(state["projects"]["p1"]["characters"][0]["portrait_asset"])

    def test_cannot_delete_another_projects_asset(self):
        state = _state()
        self._run(state, "p1", ["x"], status=403)
        self.assertIn("x", state["assets"])

    def test_empty_selection_is_rejected(self):
        state = _state()
        with self.assertRaises(ValueError):
            self._run(state, "p1", [])

    def test_asset_in_use_by_a_running_job_is_protected(self):
        state = _state()
        state["runs"] = {"r1": {"status": "running", "project_id": "p1", "shot_ids": ["s1"]}}
        self._run(state, "p1", ["a"], status=409)
        self.assertIn("a", state["assets"])


if __name__ == "__main__":
    unittest.main()

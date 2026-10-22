"""三处修复的回归测试。

1. 上传图片后必须立刻进入项目的 <Picture N> 编号，否则提示词无法引用它。
2. 用户显式指定 reference 角色时，关键词启发式不得把它提升成身份参考。
3. 越界的 <Video N> 必须指回源视频，避免模型收到悬空引用。
"""
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from engine import _is_identity_reference
from runner import prepare_shot


class _FakeRequest:
    def __init__(self, path, headers, body):
        self.path = path
        self.headers = headers
        self.rfile = io.BytesIO(body)
        self.status = None
        self.payload = None

    def json(self, status, payload):
        self.status = status
        self.payload = payload

    def send_error(self, status):
        self.status = status


class AssetUploadNumberingTests(unittest.TestCase):
    def test_uploaded_image_enters_the_project_numbering_right_away(self):
        from server.routes import assets as assets_route
        state = {
            "projects": {"p1": {"id": "p1", "shots": [], "shared_prompt": ""}},
            "assets": {"old": {"id": "old", "kind": "image", "project_id": "p1", "file": "old.png"}},
        }
        state["projects"]["p1"]["reference_image_numbers"] = {"old": 1}

        def fake_read():
            return state

        def fake_change(fn):
            fn(state)

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(assets_route, "ASSETS", Path(directory)), \
                    patch.object(assets_route, "read", fake_read), \
                    patch.object(assets_route, "change", fake_change):
                request = _FakeRequest(
                    "/api/assets?project_id=p1&name=new.png",
                    {"Content-Length": "5"}, b"12345")
                assets_route.post_api_assets_upload(request, "/api/assets", [], None)

        self.assertEqual(request.status, 201)
        new_id = request.payload["id"]
        numbers = state["projects"]["p1"]["reference_image_numbers"]
        self.assertEqual(numbers["old"], 1)
        self.assertEqual(numbers[new_id], 2, "新上传的图片必须立刻拿到 <Picture 2> 编号")


class ExplicitReferenceRoleTests(unittest.TestCase):
    def test_attribute_transfer_role_is_not_promoted_to_identity(self):
        shot = {"replace_person": True, "ref_images": ["sheet", "outfit"],
                "ref_image_roles": {"outfit": "attribute_transfer"},
                "prompt": "The performer is <Picture 1> and wears the outfit shown in <Picture 2>."}
        self.assertTrue(_is_identity_reference(shot, "sheet", "四视图", ""))
        self.assertFalse(_is_identity_reference(shot, "outfit", "服装图", "中式对襟褂正面"))

    def test_weak_reference_role_is_not_promoted_to_identity(self):
        shot = {"replace_person": True, "ref_images": ["sheet", "mood"],
                "ref_image_roles": {"mood": "weak_reference"},
                "prompt": "The performer is <Picture 1>; mood from <Picture 2>."}
        self.assertFalse(_is_identity_reference(shot, "mood", "色调图", ""))

    def test_auto_role_still_uses_the_keyword_heuristic(self):
        shot = {"replace_person": True, "ref_images": ["sheet", "outfit"],
                "ref_image_roles": {"outfit": "auto"},
                "prompt": "The performer's outfit follows <Picture 2>."}
        self.assertTrue(_is_identity_reference(shot, "outfit", "服装图", ""))


class VideoTagNormalizationTests(unittest.TestCase):
    def _project(self):
        return {"id": "p1", "characters": [], "entities": [], "shots": [],
                "shared_prompt": "", "image_descriptions": {}}

    def test_out_of_range_video_tag_points_back_to_the_source(self):
        project = self._project()
        shot = {"id": "s1", "mode": "rv2v", "source_video": "v1", "ref_videos": [],
                "ref_images": [], "ref_audios": [], "prompt": "opens on <Video 2>'s framing"}
        prepared = prepare_shot(dict(shot), project)
        self.assertIn("<Video 1>", prepared["prompt"])
        self.assertNotIn("<Video 2>", prepared["prompt"])

    def test_in_range_video_tags_are_kept(self):
        project = self._project()
        shot = {"id": "s1", "mode": "rv2v", "source_video": "v1", "ref_videos": ["v2"],
                "ref_images": [], "ref_audios": [], "prompt": "<Video 1> and <Video 2>"}
        prepared = prepare_shot(dict(shot), project)
        self.assertIn("<Video 1>", prepared["prompt"])
        self.assertIn("<Video 2>", prepared["prompt"])

    def test_video_tag_without_a_source_video_is_left_alone(self):
        project = self._project()
        shot = {"id": "s1", "mode": "r2v", "ref_videos": ["v2"],
                "ref_images": [], "ref_audios": [], "prompt": "from <Video 3>"}
        prepared = prepare_shot(dict(shot), project)
        self.assertIn("<Video 3>", prepared["prompt"])


if __name__ == "__main__":
    unittest.main()

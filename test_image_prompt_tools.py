import copy
import unittest
from unittest.mock import patch

from image_prompt_tools import optimize_image_prompt


class ImagePromptSkillTests(unittest.TestCase):
    def setUp(self):
        self.project = {
            "name": "水墨短剧", "prompt_skill": "h3-prompt-writing",
            "shared_prompt": "水墨卡通，青蓝雨夜与暖金纸灯。",
            "characters": [
                {"id": "a", "name": "书生", "appearance": "青衫，黑发束冠", "multiview_prompt": ""},
                {"id": "b", "name": "女鬼", "appearance": "白衣红带", "multiview_prompt": ""},
            ],
            "entities": [{"id": "c", "name": "旧书院", "kind": "place",
                          "description": "旧书桌、壁画、雕花木窗", "generation_prompt": ""}],
        }

    def test_character_prompt_uses_project_skill_and_preserves_project(self):
        before = copy.deepcopy(self.project)
        with patch("image_prompt_tools._deepseek_complete", return_value=(
            "One young scholar in a pale teal robe with a dark cap and wooden bookcase. "
            "Chinese ink-wash cartoon, stable face and outfit in every view.")) as model:
            result = optimize_image_prompt(self.project, "character", "a", "all")
        self.assertEqual(result["skill_id"], "h3-prompt-writing")
        self.assertIn("pale teal robe", result["prompt"])
        self.assertEqual(self.project, before)
        request = model.call_args.args[0][1]["content"]
        self.assertIn("青衫，黑发束冠", request)
        self.assertIn("水墨卡通", request)
        self.assertIn("do not blend", request)

    def test_rejects_video_syntax_and_invalid_scope(self):
        with self.assertRaisesRegex(ValueError, "优化范围"):
            optimize_image_prompt(self.project, "entity", "c", "")
        with patch("image_prompt_tools._deepseek_complete", return_value="[Shot 1] <Picture 1> old academy"):
            with self.assertRaisesRegex(RuntimeError, "可用于生图"):
                optimize_image_prompt(self.project, "entity", "c", "all")


if __name__ == "__main__":
    unittest.main()

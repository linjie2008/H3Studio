import json
import unittest
from unittest.mock import patch

import prompt_tools


class SkillAgentTests(unittest.TestCase):
    def test_skill_recommendations_are_limited_to_installed_catalog(self):
        response = {"summary": "追逐短片。", "recommendations": [
            {"id": "chase", "reason": "适合高速动作。"},
            {"id": "invented", "reason": "不存在。"},
            {"id": "chase", "reason": "重复。"},
            {"id": "dialogue", "reason": "适合对白。"},
        ]}
        catalog = [{"id": "h3-prompt-writing", "name": "H3", "description": "通用"},
                   {"id": "chase", "name": "追逐", "description": "动作"},
                   {"id": "dialogue", "name": "对白", "description": "对话"}]
        with patch.object(prompt_tools, "_deepseek_complete", return_value=json.dumps(response)):
            result = prompt_tools.recommend_project_skills(
                {"name": "公路追逐", "shots": [{"prompt": "车辆高速追逐"}]}, catalog)
        self.assertEqual([item["id"] for item in result["recommendations"]],
                         ["chase", "dialogue"])

    def test_proposal_is_allowlisted_and_maps_canvas_dimensions(self):
        response = {
            "summary": "竖屏短剧适合竖版画幅。",
            "settings": {
                "aspect": "9:16", "resolution": 720,
                "generation_speed": "quality", "sage_attention": True,
                "h3_video_vae": "not-installed.safetensors", "memory_policy": "auto_vram",
                "upscale": False, "upscale_resolution": 720,
                "upscale_max_resolution": 1280, "confirm_upscale": True,
                "continuity_frames": 22, "refine_mode": "off", "refine_steps": 4,
            },
            "reasons": {"aspect": "竖屏叙事。", "generation_speed": "强调细节。"},
        }
        with patch.object(prompt_tools, "preview_skill", return_value={"files": []}), \
             patch.object(prompt_tools, "_deepseek_complete", return_value=json.dumps(response)):
            result = prompt_tools.suggest_project_parameters(
                "test", {"width": 832, "height": 480, "generation_speed": "turbo"},
                {"sage_attention": False, "h3_video_vaes": ["installed.safetensors"]}, {})
        self.assertEqual(result["settings"]["width"], 720)
        self.assertEqual(result["settings"]["height"], 1280)
        self.assertEqual(result["settings"]["generation_speed"], "quality")
        self.assertNotIn("sage_attention", result["settings"])
        self.assertNotIn("h3_video_vae", result["settings"])
        self.assertEqual({item["key"] for item in result["changes"]},
                         {"width", "height", "generation_speed", "memory_policy"})

    def test_missing_fields_preserve_project_settings(self):
        response = {"summary": "保持现有设置。", "settings": {}, "reasons": {}}
        current = {"width": 1280, "height": 720, "generation_speed": "quality",
                   "sage_attention": False, "h3_video_vae": "vae.safetensors",
                   "memory_policy": "manual", "upscale": True, "upscale_resolution": 1080,
                   "upscale_max_resolution": 1920, "confirm_upscale": False,
                   "continuity_frames": 39, "refine_mode": "same", "refine_steps": 6}
        with patch.object(prompt_tools, "preview_skill", return_value={"files": []}), \
             patch.object(prompt_tools, "_deepseek_complete", return_value=json.dumps(response)):
            result = prompt_tools.suggest_project_parameters(
                "test", current, {"sage_attention": False, "h3_video_vaes": ["vae.safetensors"]})
        self.assertEqual(result["changes"], [])
        self.assertEqual(result["settings"], {})

    def test_truncated_parameter_json_is_automatically_rebuilt(self):
        truncated = '{"settings":{"aspect":"16:9","resolution":720},"summary":"未结束'
        repaired = {"settings": {"aspect": "16:9", "resolution": 720},
                    "reasons": {}, "summary": "已恢复完整建议。"}
        current = {"width": 1280, "height": 720, "generation_speed": "turbo"}
        with patch.object(prompt_tools, "preview_skill", return_value={"files": []}), \
             patch.object(prompt_tools, "_deepseek_complete",
                          side_effect=[truncated, json.dumps(repaired)]) as complete:
            result = prompt_tools.suggest_project_parameters(
                "test", current, {"sage_attention": False,
                                    "h3_video_vaes": ["vae.safetensors"]})
        self.assertEqual(result["summary"], "已恢复完整建议。")
        self.assertEqual(complete.call_count, 2)
        self.assertEqual(complete.call_args_list[0].kwargs["max_tokens"], 2800)
        self.assertIn("重建", complete.call_args_list[1].args[0][1]["content"])

    def test_twice_truncated_parameter_json_has_readable_error(self):
        with patch.object(prompt_tools, "preview_skill", return_value={"files": []}), \
             patch.object(prompt_tools, "_deepseek_complete",
                          side_effect=['{"settings":"', '{"settings":"']):
            with self.assertRaisesRegex(RuntimeError, "连续两次返回了不完整数据"):
                prompt_tools.suggest_project_parameters(
                    "test", {"width": 1280, "height": 720},
                    {"sage_attention": False, "h3_video_vaes": ["vae.safetensors"]})


if __name__ == "__main__":
    unittest.main()

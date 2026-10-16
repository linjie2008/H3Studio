import unittest
from unittest.mock import patch

import multiview
import profile_images
import store


class ProfileImageGraphTests(unittest.TestCase):
    def test_missing_character_settings_match_local_workflow_defaults(self):
        project = {"id": "p", "name": "test", "characters": [{"id": "c"}]}
        data = store._normalize({"projects": {"p": project}, "assets": {}, "runs": {}})
        character = data["projects"]["p"]["characters"][0]
        self.assertEqual(character["xb_sampling_mode"], "bypass")
        self.assertEqual(character["xb_aspect_ratio"], "16:9")
        self.assertEqual(character["xb_width"], 1920)
        self.assertEqual(character["xb_height"], 1088)
        self.assertFalse(character["xb_canvas_override"])
        self.assertEqual((character["krea_width"], character["krea_height"]), (1536, 1024))

    def test_existing_custom_canvas_is_preserved(self):
        project = {"id": "p", "name": "test", "characters": [
            {"id": "c", "xb_aspect_ratio": "Free", "xb_width": 1280, "xb_height": 768}]}
        data = store._normalize({"projects": {"p": project}, "assets": {}, "runs": {}})
        self.assertTrue(data["projects"]["p"]["characters"][0]["xb_canvas_override"])

    def test_store_preserves_krea2_engine_selection(self):
        project = {"id": "p", "name": "test", "characters": [{"id": "c", "multiview_engine": "krea2"}]}
        data = store._normalize({"projects": {"p": project}, "assets": {}, "runs": {}})
        self.assertEqual(data["projects"]["p"]["characters"][0]["multiview_engine"], "krea2")

    def test_store_preserves_six_view_sheet_selection(self):
        project = {"id": "p", "name": "test", "characters": [
            {"id": "c", "multiview_engine": "qwen_sheet", "multiview_layout": "six"}]}
        data = store._normalize({"projects": {"p": project}, "assets": {}, "runs": {}})
        character = data["projects"]["p"]["characters"][0]
        self.assertEqual((character["multiview_engine"], character["multiview_layout"]),
                         ("qwen_sheet", "six"))

    def test_six_view_graph_uses_saved_character_sheet_workflow(self):
        with patch.object(multiview, "api", return_value={
                "LayerUtility: ImageScaleByAspectRatio V2": {}, "QwenImage21Cache": {},
                "TextEncodeQwenImage21": {}}):
            graph = multiview.build_graph("portrait.png", "fast", "six", "灰色军装",
                                          "six", "qwen_sheet")
        self.assertEqual(graph["8"]["inputs"], {"width": 1536, "height": 2048,
                                                "batch_size": 1})
        self.assertEqual(graph["9"]["inputs"]["steps"], 40)
        self.assertEqual(graph["7"]["inputs"]["images.image_1"], ["2", 0])
        self.assertIn("2行×3列", graph["7"]["inputs"]["prompt"])
        self.assertIn("灰色军装", graph["7"]["inputs"]["prompt"])
        with self.assertRaisesRegex(ValueError, "原图"):
            multiview.build_graph(None, "fast", "six", "角色", "six", "qwen_sheet")

    def test_old_mv_adapter_selection_uses_xb_qwen(self):
        project = {"id": "p", "name": "test", "characters": [{"id": "c", "multiview_engine": "mvadapter"}]}
        data = store._normalize({"projects": {"p": project}, "assets": {}, "runs": {}})
        self.assertEqual(data["projects"]["p"]["characters"][0]["multiview_engine"], "xb_qwen")

    def test_character_four_view_supports_text_and_image_sources(self):
        with patch.object(multiview, "_model_config", return_value=("model", "clip", "vae")):
            text_graph = multiview.build_graph(None, "fast", "job1", "短发飞行员，棕色制服")
            image_graph = multiview.build_graph("portrait.png", "stable", "job2", "短发飞行员")
        self.assertNotIn("5", text_graph)
        self.assertNotIn("images.image_1", text_graph["6"]["inputs"])
        self.assertNotIn("images.image_2", text_graph["6"]["inputs"])
        self.assertEqual(text_graph["7"]["inputs"]["latent_image"], ["10", 0])
        self.assertIn("短发飞行员", text_graph["6"]["inputs"]["prompt"])
        self.assertEqual(image_graph["5"]["inputs"]["image"], "portrait.png")
        self.assertEqual(image_graph["6"]["inputs"]["images.image_2"], ["5", 0])
        self.assertEqual(text_graph["7"]["inputs"]["steps"], 16)
        self.assertEqual(image_graph["7"]["inputs"]["steps"], 25)

    def test_character_five_view_uses_wider_canvas_and_extra_angle(self):
        with patch.object(multiview, "_model_config", return_value=("model", "clip", "vae")):
            graph = multiview.build_graph(None, "fast", "job5", "短发飞行员", "five")
        self.assertEqual(graph["4"]["inputs"]["width"], 2048)
        self.assertEqual(graph["4"]["inputs"]["height"], 1024)
        self.assertIn("exactly five equal-width vertical panels", graph["6"]["inputs"]["prompt"])
        self.assertIn("three-quarter full-body view", graph["6"]["inputs"]["prompt"])
        self.assertIn("rear three-quarter full-body view", graph["6"]["inputs"]["prompt"])
        self.assertIn("no face close-up", graph["6"]["inputs"]["prompt"])

    def test_character_requirements_override_template_defaults(self):
        with patch.object(multiview, "_model_config", return_value=("model", "clip", "vae")):
            graph = multiview.build_graph("portrait.png", "fast", "job", "绿幕背景；换成飞行员制服")
        prompt = graph["6"]["inputs"]["prompt"]
        self.assertTrue(prompt.startswith("MANDATORY USER CHARACTER REQUIREMENTS"))
        self.assertIn("绿幕背景；换成飞行员制服", prompt)
        self.assertNotIn("plain light-grey studio background", prompt)
        self.assertIn("If the mandatory user requirements change clothing", prompt)
        self.assertNotIn("pure chroma-key green screen (#00FF00)", prompt)
        self.assertEqual(graph["4"]["inputs"]["color"], 65280)
        self.assertEqual(graph["200"]["class_type"], "easy imageRemBg")
        self.assertEqual(graph["201"]["inputs"]["color"], 65280)
        self.assertEqual(graph["9"]["inputs"]["images"], ["202", 0])

    def test_mv_adapter_is_no_longer_a_generation_engine(self):
        with self.assertRaisesRegex(ValueError, "生成引擎无效"):
            multiview.build_graph("portrait.png", "fast", "mv", "角色", "five", "mvadapter")

    def test_xb_qwen_reproduces_local_advanced_workflow(self):
        settings = {"output_lang": "zh", "sampling_mode": "on", "skill_mode": "自动",
                    "aspect_ratio": "1:1", "width": 1024, "height": 1024,
                    "seed": 42, "thinking": False, "extra_settings": "纯绿色背景"}
        with patch.object(multiview, "_require_xb_qwen"), patch.object(
                multiview, "_model_config", return_value=("model", "clip", "vae")):
            graph = multiview.build_graph(None, "stable", "xb", "红色飞行员制服", "five", "xb_qwen", settings)
        self.assertEqual(graph["100"]["class_type"], "XB_QwenPromptPreset")
        self.assertEqual(graph["100"]["inputs"]["preset_mode"], "人物五视图")
        preset_text = graph["100"]["inputs"]["three_view_text"]
        self.assertEqual(preset_text, "")
        self.assertEqual(graph["100"]["inputs"]["clip_name"], multiview.XB_TEXT_ENCODER)
        self.assertIn("红色飞行员制服", graph["100"]["inputs"]["internal_prompt"])
        self.assertNotIn("pure chroma-key green screen (#00FF00)", graph["100"]["inputs"]["internal_prompt"])
        self.assertIn("纯绿色背景", graph["100"]["inputs"]["manager_settings"])
        self.assertEqual(graph["4"]["inputs"]["prompt"], ["100", 0])
        self.assertEqual(graph["5"]["inputs"]["latent_image"], ["100", 1])
        self.assertEqual(graph["5"]["inputs"]["steps"], 25)
        self.assertNotIn("green screen", graph["100"]["inputs"]["internal_prompt"])
        self.assertEqual(graph["7"]["inputs"]["images"], ["202", 0])
        self.assertEqual(graph["200"]["inputs"]["images"], ["6", 0])
        self.assertNotIn("101", graph)

    def test_krea2_workflow_produces_four_views_without_prompt_preview(self):
        with patch.object(multiview, "_require_krea2"):
            graph = multiview.build_graph(
                "portrait.png", "fast", "krea", "人物穿蓝色外套", "four", "krea2")
        self.assertEqual(graph["2"]["inputs"]["lora_name"], "Krea2-功能/Krea2-四视图QuadView_krea2_v1.safetensors")
        self.assertEqual(graph["3"]["inputs"]["type"], "krea2")
        self.assertEqual(graph["9"]["class_type"], "Krea2EditGroundedEncode")
        self.assertIn("Character Sheet", graph["9"]["inputs"]["prompt"])
        self.assertIn("人物穿蓝色外套", graph["9"]["inputs"]["prompt"])
        self.assertEqual(graph["12"]["inputs"]["steps"], 10)
        self.assertEqual(graph["8"]["inputs"], {"width": 1536, "height": 1024, "batch_size": 1})
        self.assertEqual(graph["14"]["inputs"]["images"], ["13", 0])
        self.assertFalse(any(node["class_type"] == "PreviewAny" for node in graph.values()))
        with self.assertRaisesRegex(ValueError, "只支持四视图"):
            multiview.build_graph("portrait.png", "fast", "krea5", "角色", "five", "krea2")

    def test_krea2_uses_character_canvas_dimensions(self):
        with patch.object(multiview, "_require_krea2"):
            graph = multiview.build_graph("portrait.png", "fast", "job", "角色", "four",
                                          "krea2", {"krea_width": 1280, "krea_height": 768})
            self.assertEqual(graph["8"]["inputs"]["width"], 1280)
            self.assertEqual(graph["8"]["inputs"]["height"], 768)
            self.assertEqual(graph["6"]["inputs"]["width"], 1024)
            with self.assertRaisesRegex(ValueError, "32 的倍数"):
                multiview.build_graph("portrait.png", "fast", "bad", "角色", "four",
                                      "krea2", {"krea_width": 1279, "krea_height": 768})

    def test_xb_qwen_defaults_to_landscape_aspect_ratio(self):
        with patch.object(multiview, "_require_xb_qwen"), patch.object(
                multiview, "_model_config", return_value=("model", "clip", "vae")):
            graph = multiview.build_graph(None, "fast", "xbfree", "角色", "five", "xb_qwen", {})
        self.assertEqual(graph["100"]["inputs"]["aspect_ratio"], "16:9")
        self.assertEqual(graph["100"]["inputs"]["width"], 1920)
        self.assertEqual(graph["100"]["inputs"]["height"], 1088)
        self.assertEqual(graph["100"]["inputs"]["sampling_mode"], "bypass")

    def test_xb_qwen_passes_saved_user_preset_text_unchanged(self):
        settings = {"preset_text": "用户保存的 XB 预设设定词"}
        with patch.object(multiview, "_require_xb_qwen"), patch.object(
                multiview, "_model_config", return_value=("model", "clip", "vae")):
            graph = multiview.build_graph(None, "fast", "xbpreset", "角色", "five", "xb_qwen", settings)
        self.assertEqual(graph["100"]["inputs"]["preset_mode"], "人物五视图")
        self.assertEqual(graph["100"]["inputs"]["three_view_text"], "用户保存的 XB 预设设定词")

    def test_xb_qwen_image_mode_matches_saved_workflow_69(self):
        settings = {"aspect_ratio": "16:9", "width": 1024, "height": 1024,
                    "sampling_mode": "on"}
        workflow = multiview._xb_i2i_workflow_defaults()
        with patch.object(multiview, "_require_xb_qwen"), patch.object(
                multiview, "_model_config", return_value=("model", "clip", "vae")):
            graph = multiview.build_graph(
                "portrait.png", "fast", "xb", "角色", "four", "xb_qwen", settings)
        self.assertEqual(graph["100"]["inputs"]["io_mode"], "图生图")
        self.assertEqual(graph["100"]["inputs"]["images"], ["9", 0])
        self.assertEqual(graph["9"]["inputs"]["image"], "portrait.png")
        self.assertEqual(graph["100"]["inputs"]["aspect_ratio"], workflow["aspect_ratio"])
        self.assertEqual(graph["100"]["inputs"]["width"], workflow["width"])
        self.assertEqual(graph["100"]["inputs"]["height"], workflow["height"])
        self.assertEqual(graph["100"]["inputs"]["sampling_mode"], workflow["sampling_mode"])
        self.assertEqual(graph["100"]["inputs"]["preset_mode"], "人物四视图")
        self.assertEqual(graph["100"]["inputs"]["three_view_text"], "")
        self.assertIn("角色", graph["100"]["inputs"]["internal_prompt"])
        self.assertEqual(graph["4"]["inputs"]["images.image_1"], ["9", 0])
        self.assertNotIn("images.image_2", graph["4"]["inputs"])
        self.assertEqual(graph["199"]["inputs"]["image"], ["6", 0])
        self.assertEqual(graph["201"]["inputs"]["width"], ["199", 1])
        self.assertEqual(graph["201"]["inputs"]["height"], ["199", 2])
        self.assertEqual(graph["201"]["inputs"]["color"], 65280)
        self.assertEqual(graph["7"]["inputs"]["images"], ["202", 0])
        self.assertEqual(graph["5"]["inputs"]["latent_image"], ["4", 2])

    def test_xb_qwen_can_keep_generated_background(self):
        with patch.object(multiview, "_require_xb_qwen"), patch.object(
                multiview, "_model_config", return_value=("model", "clip", "vae")):
            graph = multiview.build_graph(
                "portrait.png", "fast", "xb-bg", "角色", "four", "xb_qwen",
                {"green_background": "false"})
        self.assertNotIn("200", graph)
        self.assertEqual(graph["7"]["inputs"]["images"], ["6", 0])

    def test_xb_qwen_i2i_defaults_to_green_background(self):
        with patch.object(multiview, "_require_xb_qwen"), patch.object(
                multiview, "_model_config", return_value=("model", "clip", "vae")):
            graph = multiview.build_graph(
                "portrait.png", "fast", "xb-green", "角色", "four", "xb_qwen", {})
        self.assertEqual(graph["199"]["inputs"]["image"], ["6", 0])
        self.assertEqual(graph["201"]["inputs"], {
            "width": ["199", 1], "height": ["199", 2],
            "batch_size": ["199", 3], "color": 65280})
        self.assertEqual(graph["200"]["inputs"]["images"], ["6", 0])
        self.assertEqual(graph["7"]["inputs"]["images"], ["202", 0])

    def test_xb_qwen_five_view_i2i_keeps_workflow_69_preset_text(self):
        with patch.object(multiview, "_require_xb_qwen"), patch.object(
                multiview, "_model_config", return_value=("model", "clip", "vae")):
            graph = multiview.build_graph(
                "portrait.png", "fast", "xb5", "角色", "five", "xb_qwen", {})
        self.assertEqual(graph["100"]["inputs"]["preset_mode"], "人物五视图")
        self.assertIn("生成五宫格排列", graph["100"]["inputs"]["three_view_text"])
        self.assertEqual(graph["4"]["inputs"]["images.image_1"], ["9", 0])

    def test_xb_qwen_i2i_uses_explicit_character_canvas(self):
        settings = {"canvas_override": True, "aspect_ratio": "Free",
                    "width": 1280, "height": 768}
        with patch.object(multiview, "_require_xb_qwen"), patch.object(
                multiview, "_model_config", return_value=("model", "clip", "vae")):
            graph = multiview.build_graph("portrait.png", "fast", "custom", "角色",
                                          "four", "xb_qwen", settings)
        self.assertEqual(graph["100"]["inputs"]["aspect_ratio"], "Free")
        self.assertEqual(graph["100"]["inputs"]["width"], 1280)
        self.assertEqual(graph["100"]["inputs"]["height"], 768)
        self.assertEqual(graph["10"]["inputs"]["width"], 1280)
        self.assertEqual(graph["10"]["inputs"]["height"], 768)
        self.assertEqual(graph["10"]["inputs"]["keep_proportion"], "pad")
        self.assertEqual(graph["4"]["inputs"]["resolution"], 0)
        self.assertEqual(graph["4"]["inputs"]["images.image_1"], ["10", 0])
        self.assertEqual(graph["100"]["inputs"]["images"], ["10", 0])
        self.assertEqual(graph["5"]["inputs"]["latent_image"], ["4", 2])

    def test_each_entity_type_supports_both_generation_modes(self):
        with patch.object(profile_images, "_model_config", return_value=("model", "clip", "vae")):
            for kind in ("place", "prop", "style"):
                with self.subTest(kind=kind):
                    text_graph = profile_images.build_graph(kind, "青灰色、有年代感", "t2i", None, "fast", "job1")
                    image_graph = profile_images.build_graph(kind, "保留轮廓并增强细节", "i2i", "source.png", "stable", "job2")
                    self.assertIn("青灰色、有年代感", text_graph["6"]["inputs"]["prompt"])
                    self.assertNotIn("5", text_graph)
                    self.assertNotIn("images.image_1", text_graph["6"]["inputs"])
                    self.assertEqual(text_graph["7"]["inputs"]["latent_image"], ["10", 0])
                    self.assertEqual(image_graph["5"]["inputs"]["image"], "source.png")
                    self.assertEqual(image_graph["6"]["inputs"]["images.image_2"], ["5", 0])
                    self.assertEqual(text_graph["7"]["inputs"]["steps"], 16)
                    self.assertEqual(image_graph["7"]["inputs"]["steps"], 25)

    def test_image_mode_requires_a_source(self):
        with self.assertRaisesRegex(ValueError, "参考图片"):
            profile_images.build_graph("place", "机场跑道", "i2i", None, "fast", "job")


if __name__ == "__main__":
    unittest.main()

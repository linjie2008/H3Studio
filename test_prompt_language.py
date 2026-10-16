import unittest
from unittest.mock import patch

import prompt_tools
from engine import compile_shot_prompt
from store import new_shot


class PromptLanguageTests(unittest.TestCase):
    def test_new_shot_defaults_to_english(self):
        self.assertEqual(new_shot()["prompt_language"], "en")

    def test_optimization_requests_selected_language_and_preserves_references(self):
        for language, result, expected in (
            ("en", "The person in <Picture 2> walks toward the lens.", "English"),
            ("zh", "<Picture 2> 中的人物正面向镜头走来。", "Simplified Chinese"),
        ):
            with self.subTest(language=language), \
                 patch.object(prompt_tools, "load_skill_context", return_value="basic skill"), \
                 patch.object(prompt_tools, "_deepseek_complete", return_value=result) as complete:
                actual = prompt_tools.enhance("<Picture 2> 中的人物走来。", "r2v",
                                             output_language=language)
                self.assertEqual(actual, result)
                self.assertIn(expected, complete.call_args.args[0][0]["content"])
                self.assertIn(expected, complete.call_args.args[0][1]["content"])

    def test_chinese_long_take_overlay_and_model_prompt_stay_chinese(self):
        skill = "遮挡转场 Prompt 编写公式 continuityOverlapFrames"
        context = {"segment": {"position": 1, "total": 2},
                   "references": [{"tag": "<Picture 1>", "type": "image"}]}
        with patch.object(prompt_tools, "load_skill_context", return_value=skill), \
             patch.object(prompt_tools, "_deepseek_complete", return_value="<Picture 1> 中的人物向镜头走来。"):
            optimized = prompt_tools.enhance("<Picture 1> 中的人物走来。", "r2v",
                                            duration=5, reference_context=context,
                                            output_language="zh")
        self.assertEqual(optimized, "<Picture 1> 中的人物向镜头走来。")
        self.assertNotIn("MANDATORY MOVEMENT", optimized)
        compiled = compile_shot_prompt({"mode": "t2v", "prompt": optimized, "duration": 5,
                                        "prompt_language": "zh"})
        self.assertIn("<Picture 1> 中的人物向镜头走来。", compiled)
        self.assertNotIn("逐秒动作", compiled)


if __name__ == "__main__":
    unittest.main()

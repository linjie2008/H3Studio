import unittest

from prompt_skills import load_skill_context, preview_skill


class PromptSkillModeTests(unittest.TestCase):
    def test_base_modes_receive_only_their_own_keyframe_rule_and_example(self):
        expectations = {
            "t2v": ("### Case 1: T2VA", "### Case 2: I2VA"),
            "i2v": ("### Case 2: I2VA", "### Case 3: FL2VA"),
            "fl2v": ("### Case 3: FL2VA", "### Case 4: L2VA"),
        }
        for mode, (expected, excluded) in expectations.items():
            with self.subTest(mode=mode):
                content = load_skill_context(None, mode)
                self.assertIn(expected, content)
                self.assertNotIn(excluded, content)
                self.assertIn("## Active mode rules", content)
                self.assertIn("integrated_multimodal_description", content)

    def test_reference_modes_receive_distinct_guides(self):
        for mode, heading in (("r2v", "多参考生成"), ("v2v", "视频编辑"),
                              ("rv2v", "视频＋参考")):
            with self.subTest(mode=mode):
                content = load_skill_context(None, mode)
                self.assertIn(heading, content)
                self.assertIn("retention_analysis", content)
                self.assertNotIn("### Case 1: T2VA", content)

    def test_single_shot_reference_context_keeps_labels_without_full_schema(self):
        full = load_skill_context(None, "r2v")
        compact = load_skill_context(None, "r2v", single_shot=True)
        self.assertIn("<Picture N>", compact)
        self.assertIn("### 5.3 Using Reference Labels in Shots", compact)
        self.assertNotIn("## 7. Complete Example", compact)
        self.assertLess(len(compact), len(full) // 2)

    def test_preview_exposes_each_mode_guide(self):
        filenames = {entry["name"] for entry in preview_skill("h3-prompt-writing")["files"]}
        self.assertTrue({f"mode-guides/{mode}.md" for mode in
                         ("t2v", "i2v", "fl2v", "r2v", "v2v", "rv2v")}.issubset(filenames))


if __name__ == "__main__":
    unittest.main()

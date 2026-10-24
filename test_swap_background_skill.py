"""换背景相关的回归测试。

1. 源视频角色降为「弱参考（动作/节奏）」时，说明里必须明确"原背景不保留"，
   否则提示词会同时说"只借动作"和"保留构图"，模型会保住原背景。
2. 默认的「编辑后部分保留」仍应说明保留运镜与构图。
3. 换人 Skill 必须随项目 Skill 一起注入模型上下文。
"""
import unittest

from engine import _compile_full_reference_prompt
from prompt_skills import load_skill_context


class SourceVideoBackgroundTests(unittest.TestCase):
    def _prompt(self, marker):
        return _compile_full_reference_prompt({
            "mode": "rv2v",
            "source_video": "source",
            "replace_person": True,
            "ref_images": ["person"],
            "ref_video_roles": {"source": marker},
            "_h3_reference_names": {"source": "source.mp4", "person": "person.png"},
        }, "The performer matches the identity in <Picture 1> exactly.", "N/A", "N/A")

    def test_weak_source_video_releases_the_background(self):
        prompt = self._prompt("weak_reference")
        self.assertIn("<Video 1> (source edit): weak_reference", prompt)
        self.assertIn("only the source's motion rhythm and body path are borrowed", prompt)
        self.assertIn("background are not kept", prompt)

    def test_default_source_video_keeps_framing(self):
        prompt = self._prompt("partially_preserved")
        self.assertIn("<Video 1> (source edit): partially_preserved", prompt)
        self.assertIn("camera movement, framing and timing are retained", prompt)

    def test_cat_identity_does_not_inherit_stale_human_multiview_description(self):
        shot = {
            "mode": "rv2v", "source_video": "source", "replace_person": True,
            "ref_images": ["sheet"],
            "prompt": "画面主体是 <Picture 1> 里那只猫，保持真实四足体态。",
            "_h3_reference_names": {"sheet": "角色 2 · 四视图", "source": "human.mp4"},
            "_h3_reference_descriptions": {
                "sheet": "同一人物的脸部特写、正面、侧面和背面；用于稳定脸、身形与服装。"},
        }
        prompt = _compile_full_reference_prompt(shot, shot["prompt"], "N/A", "N/A")
        definitions = prompt.split("summary:")[0]
        self.assertIn("for the cat in this shot", definitions)
        self.assertIn("quadruped anatomy", definitions)
        self.assertNotIn("for the person in this shot", definitions)
        self.assertNotIn("Verified user description: 同一人物", definitions)
        self.assertIn("Adapt any human movement to natural quadruped motion", prompt)


class SwapSkillTests(unittest.TestCase):
    def test_swap_skill_is_loaded_with_its_reference(self):
        context = load_skill_context("custom-h3-character-swap", "rv2v", single_shot=True)
        self.assertIn("换人镜头四要素", context)
        self.assertIn("标签纪律", context)
        self.assertIn("换背景（替换源视频的场景）", context)
        self.assertIn("例 A · 四视图单参考", context)


if __name__ == "__main__":
    unittest.main()


class MultiViewRecognitionTests(unittest.TestCase):
    """工作台支持四/五/六视图多视图，引擎必须都认得出来。"""

    def _identity(self, name, description=""):
        from engine import _is_identity_reference
        shot = {"ref_images": ["sheet"], "prompt": "a person stands in frame"}
        return _is_identity_reference(shot, "sheet", name, description)

    def test_six_view_sheet_is_an_identity_reference(self):
        self.assertTrue(self._identity("施亚军身份参考.png", "人物六视图身份参考"))

    def test_six_view_sheet_gets_the_grid_instruction(self):
        from engine import _compile_full_reference_prompt
        prompt = _compile_full_reference_prompt({
            "mode": "r2v", "ref_images": ["sheet"],
            "_h3_reference_names": {"sheet": "施亚军身份参考.png"},
            "_h3_reference_descriptions": {"sheet": "人物六视图身份参考"},
        }, "The person walks forward.", "N/A", "N/A")
        self.assertIn("is a multi-view identity sheet", prompt)
        self.assertIn("never render the sheet", prompt)

    def test_plain_name_is_not_treated_as_a_sheet(self):
        self.assertFalse(self._identity("施亚军身份参考.png", ""))


class ReferenceBackgroundModeTests(unittest.TestCase):
    """镜头级「参考素材底色」控制。"""

    def _shot(self, **kwargs):
        base = {"mode": "rv2v", "source_video": "src", "ref_images": ["person", "scene"],
                "_h3_reference_names": {"person": "六宫格.png", "scene": "草原.png"},
                "_h3_reference_descriptions": {"person": "人物多视图身份参考", "scene": "草原绿茵"}}
        base.update(kwargs)
        return base

    def _description(self, shot):
        from engine import _compile_full_reference_prompt
        out = _compile_full_reference_prompt(shot, "A man in frame.", "N/A", "N/A")
        return out.split("detailed_description:")[1].split("overall_soundscape:")[0].strip()

    def test_auto_inherits_from_the_scene_reference_only(self):
        text = self._description(self._shot())
        self.assertIn("<Picture 2> carry over into this shot", text)
        self.assertNotIn("<Picture 1>", text)

    def test_inherit_uses_every_picture(self):
        text = self._description(self._shot(ref_background="inherit"))
        self.assertIn("<Picture 1> and <Picture 2>", text)

    def test_authored_never_inherits(self):
        text = self._description(self._shot(ref_background="authored"))
        self.assertNotIn("carry over", text)

    def test_identity_reference_may_carry_its_environment(self):
        from engine import _compile_full_reference_prompt
        out = _compile_full_reference_prompt(
            self._shot(ref_images=["person"], ref_background="inherit"),
            "A man in frame.", "N/A", "N/A")
        self.assertIn("Its environment, palette and lighting may also carry into the shot", out)

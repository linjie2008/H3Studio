"""换人镜头最容易踩的两个坑：参考图被压扁、采样步数太低。

两者都会让模型「只换脸、不换衣服」：源视频里原有的服装在每一帧都很清晰，
而参考图里的服装只有很小一块像素，4 步采样又没有足够去噪余量去重绘它。
"""

import unittest

from engine import lint_shot_prompt


def _codes(shot, project=None):
    return [item["code"] for item in lint_shot_prompt(shot, project)]


def _swap_shot(**overrides):
    shot = {
        "mode": "rv2v",
        "replace_person": True,
        "ref_images": ["person"],
        "prompt": "The performer in the frame is <Picture 1>, dressed exactly like the reference.",
        "duration": 5,
        "ref_image_size": "match",
    }
    shot.update(overrides)
    return shot


class SwapQualityLintTests(unittest.TestCase):
    def test_matched_canvas_reference_warns_for_replacement_shots(self):
        self.assertIn("swap_ref_image_size", _codes(_swap_shot()))

    def test_max_reference_size_is_accepted(self):
        self.assertNotIn("swap_ref_image_size", _codes(_swap_shot(ref_image_size="max")))

    def test_four_step_sampling_warns_for_replacement_shots(self):
        codes = _codes(_swap_shot(), {"generation_speed": "turbo4"})
        self.assertIn("swap_fast_steps", codes)

    def test_eight_step_sampling_is_accepted(self):
        codes = _codes(_swap_shot(), {"generation_speed": "turbo"})
        self.assertNotIn("swap_fast_steps", codes)

    def test_plain_shots_are_not_flagged(self):
        plain = {"mode": "rv2v", "prompt": "The camera pushes in on the runner.",
                 "duration": 5, "ref_image_size": "match"}
        codes = _codes(plain, {"generation_speed": "turbo4"})
        self.assertNotIn("swap_ref_image_size", codes)
        self.assertNotIn("swap_fast_steps", codes)


if __name__ == "__main__":
    unittest.main()

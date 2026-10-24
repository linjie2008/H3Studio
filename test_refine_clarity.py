"""二采（selflift / 传统二采）、清晰度护栏与成片尺寸的回归测试。"""
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from engine import (  # noqa: E402
    build_h3_graph,
    lint_shot_prompt,
    refine_sigma_start,
    _guard_refine_settings,
)

SHOT = {"mode": "t2v", "prompt": "A swordsman draws his blade in a bamboo forest at dawn.",
        "seed": 42, "duration": 5}


def _classes(settings):
    graph = build_h3_graph(SHOT, settings, {}, "h3_studio_test_000")
    return {node.get("class_type") for node in graph.values() if isinstance(node, dict)}


class RefineSigmaTests(unittest.TestCase):
    def test_default_denoise_starts_near_zero_noise(self):
        # steps=4 / denoise=0.3 在 H3 的 (i/1000)^2 sigma 网格上从 0.094 起步。
        self.assertAlmostEqual(refine_sigma_start(4, 0.3), 0.0942, places=3)

    def test_higher_denoise_raises_start_noise(self):
        self.assertGreater(refine_sigma_start(4, 0.6), refine_sigma_start(4, 0.3))


class RefineGuardrailTests(unittest.TestCase):
    def test_dangerous_denoise_is_rejected(self):
        with self.assertRaises(ValueError) as caught:
            _guard_refine_settings(4, 0.6)
        self.assertIn("起点噪声", str(caught.exception))

    def test_too_few_steps_for_that_noise_is_rejected(self):
        with self.assertRaises(ValueError) as caught:
            _guard_refine_settings(4, 0.4)
        self.assertIn("步数不足", str(caught.exception))

    def test_safe_combination_passes(self):
        _guard_refine_settings(12, 0.2)
        _guard_refine_settings(16, 0.25)


class RefineGraphTests(unittest.TestCase):
    def test_selflift_uses_the_community_two_stage_sampler(self):
        classes = _classes({"generation_speed": "quality", "width": 1344, "height": 768,
                            "refine_mode": "selflift"})
        self.assertIn("SelfLiftH3Sampler", classes)
        self.assertNotIn("SamplerCustomAdvanced", classes)

    def test_selflift_drops_the_distilled_lora(self):
        classes = _classes({"generation_speed": "smart", "width": 1344, "height": 768,
                            "refine_mode": "selflift"})
        self.assertNotIn("LoraLoaderModelOnly", classes)

    def test_legacy_refine_keeps_the_first_pass_audio(self):
        classes = _classes({"generation_speed": "quality", "width": 1344, "height": 768,
                            "refine_mode": "same", "refine_steps": 12, "refine_denoise": 0.2})
        self.assertIn("SamplerCustomAdvanced", classes)
        self.assertIn("H3StudioKeepFirstAudio", classes)

    def test_invalid_refine_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            _classes({"generation_speed": "quality", "width": 1344, "height": 768,
                      "refine_mode": "chaos"})


class LintRenderSettingsTests(unittest.TestCase):
    def test_canvas_above_native_is_warned(self):
        findings = lint_shot_prompt(SHOT, {"width": 1920, "height": 1088,
                                           "generation_speed": "quality"})
        self.assertIn("canvas_above_native", {item["code"] for item in findings})

    def test_native_canvas_is_not_warned(self):
        findings = lint_shot_prompt(SHOT, {"width": 1344, "height": 768,
                                           "generation_speed": "quality"})
        self.assertNotIn("canvas_above_native", {item["code"] for item in findings})

    def test_unsafe_refine_settings_are_reported(self):
        findings = lint_shot_prompt(SHOT, {"width": 1344, "height": 768,
                                           "refine_mode": "same",
                                           "refine_steps": 4, "refine_denoise": 0.6})
        self.assertIn("refine_sigma_too_high", {item["code"] for item in findings})


class MergeCanvasTests(unittest.TestCase):
    """合并必须按素材实际尺寸，不能把超分后的镜头压回项目画布。"""

    def _video(self, path, width, height):
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
                        "-i", f"color=c=gray:s={width}x{height}:d=1",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)],
                       check=True, timeout=120)

    def test_merge_keeps_the_upscaled_canvas(self):
        from runner import _merge, _video_dimensions
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            shots = [base / "a.mp4", base / "b.mp4"]
            for shot in shots:
                self._video(shot, 1890, 1080)
            target = base / "merged.mp4"
            _merge(shots, target, 1344, 768)  # 传入项目画布也不该被采用
            self.assertEqual(_video_dimensions(target), (1890, 1080))

    def test_merge_handles_a_single_shot(self):
        from runner import _merge, _video_dimensions
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            shot = base / "a.mp4"
            self._video(shot, 1344, 768)
            target = base / "merged.mp4"
            _merge([shot], target, 1344, 768)
            self.assertEqual(_video_dimensions(target), (1344, 768))


class OutputSizeTests(unittest.TestCase):
    def _video(self, path, width, height):
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
                        "-i", f"color=c=gray:s={width}x{height}:d=1",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)],
                       check=True, timeout=120)

    def test_video_is_fitted_to_the_exact_delivery_size(self):
        from runner import _fit_output_size, _video_dimensions
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            source = base / "shot.mp4"
            self._video(source, 1890, 1080)
            fitted = _fit_output_size(source, base, 0, {"output_width": 1920, "output_height": 1080})
            self.assertEqual(_video_dimensions(fitted), (1920, 1080))
            # 原片不能被覆盖：checkpoint 里的「H3 原片」必须还是原始采样结果。
            self.assertEqual(_video_dimensions(source), (1890, 1080))

    def test_zero_target_keeps_the_video(self):
        from runner import _fit_output_size, _video_dimensions
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            source = base / "shot.mp4"
            self._video(source, 1890, 1080)
            result = _fit_output_size(source, base, 0, {"output_width": 0, "output_height": 0})
            self.assertEqual(_video_dimensions(result), (1890, 1080))


if __name__ == "__main__":
    unittest.main()

"""Reference videos must not cost more attention than the frame being generated.

H3 carries every reference token through every sampling step and attention
cost grows with the square of the token count, so a 720x1280 source video
feeding a 480x864 canvas used to cost more attention than the output itself
(37.8 s/step instead of 18.5 s/step).  The runner therefore downscales
reference videos to the generation canvas - the same "match" semantics
ComfyUI applies to reference images.
"""

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import runner


def _probe(path: Path) -> tuple[int, int]:
    result = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                             "-show_entries", "stream=width,height", "-of", "json", str(path)],
                            capture_output=True, text=True, check=True, timeout=30)
    stream = (json.loads(result.stdout).get("streams") or [{}])[0]
    return int(stream.get("width") or 0), int(stream.get("height") or 0)


class ReferenceVideoFitTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.assets = Path(self.temp.name) / "assets"
        self.assets.mkdir(parents=True)
        self.input_dir = Path(self.temp.name) / "input"
        self.input_dir.mkdir(parents=True)
        self._assets_backup = runner.ASSETS
        self._input_backup = runner.COMFY_INPUT
        runner.ASSETS = self.assets
        runner.COMFY_INPUT = self.input_dir
        self.source = self.assets / "source.mp4"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
                        "-i", "testsrc=size=720x1280:rate=24:duration=1",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(self.source)],
                       check=True, timeout=120)
        self.catalog = {"source": {"kind": "video", "file": "source.mp4"}}

    def tearDown(self):
        runner.ASSETS = self._assets_backup
        runner.COMFY_INPUT = self._input_backup
        self.temp.cleanup()

    def test_large_reference_video_is_fitted_to_the_canvas(self):
        name = runner._copy_asset("source", self.catalog, fit_width=480, fit_height=864)
        target = runner.COMFY_INPUT / name
        self.assertTrue(target.is_file(), name)
        self.assertIn("fit480x", name)
        width, height = _probe(target)
        self.assertLessEqual(width, 480)
        self.assertLessEqual(height, 864)
        # H3 works on a 16px canvas grid, so the fitted size is rounded to it.
        self.assertEqual(width % 16, 0)
        self.assertEqual(height % 16, 0)

    def test_small_reference_video_is_copied_unchanged(self):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
                        "-i", "testsrc=size=240x432:rate=24:duration=1",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(self.source)],
                       check=True, timeout=120)
        name = runner._copy_asset("source", self.catalog, fit_width=480, fit_height=864)
        self.assertNotIn("fit", name)
        self.assertEqual(_probe(runner.COMFY_INPUT / name), (240, 432))

    def test_trim_and_fit_combine(self):
        name = runner._copy_asset("source", self.catalog, trim_start=0.0,
                                  trim_duration=0.5, fit_width=480, fit_height=864)
        target = runner.COMFY_INPUT / name
        self.assertTrue(target.is_file(), name)
        self.assertIn("fit480x", name)
        width, height = _probe(target)
        self.assertLessEqual(width, 480)
        self.assertLessEqual(height, 864)

    def test_media_map_uses_the_project_canvas(self):
        media = runner._make_media_map(
            {"source_video": "source", "ref_videos": []},
            {"width": 480, "height": 864, "shared_refs": {}}, self.catalog)
        self.assertIn("fit480x", media["source"])


if __name__ == "__main__":
    unittest.main()

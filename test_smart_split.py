"""Regression test for Director-compatible adaptive video splitting."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from video_tools import boundaries


class SmartSplitTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg unavailable")
    def test_hard_scene_changes_become_three_source_clips(self):
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / "scenes.mp4"
            command = ["ffmpeg", "-y", "-loglevel", "error"]
            for color in ("red", "blue", "green"):
                command += ["-f", "lavfi", "-i", f"color=c={color}:s=160x90:r=24:d=3"]
            command += ["-filter_complex", "[0:v][1:v][2:v]concat=n=3:v=1:a=0",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(video)]
            subprocess.run(command, check=True, timeout=30)
            spans = boundaries(video, "scene")
            self.assertEqual(len(spans), 3)
            self.assertEqual([round(start) for start, _ in spans], [0, 3, 6])
            self.assertTrue(all(2 <= length <= 15 for _, length in spans))


if __name__ == "__main__":
    unittest.main()

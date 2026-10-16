import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import runner


class ComfyVideoDownloadTests(unittest.TestCase):
    def test_first_shot_starts_chain_and_later_shots_use_strong_handoff_by_default(self):
        self.assertEqual(runner._continuity_policy({}, {}, 0),
                         {"enabled": False, "frames": 0, "strength": "none"})
        self.assertEqual(runner._continuity_policy({}, {}, 1),
                         {"enabled": True, "frames": 22, "strength": "strong", "seam_redraw": 0.0})

    def test_fast_motion_adapts_overlap_and_unchecked_option_softens_but_keeps_guide(self):
        fast = runner._continuity_policy({}, {"camera_speed": "快速"}, 1)
        self.assertEqual(fast["frames"], 39)
        self.assertEqual(fast["strength"], "strong")
        soft = runner._continuity_policy({"continuous_take": False}, {}, 1)
        self.assertEqual(soft, {"enabled": True, "frames": 5, "strength": "soft"})

    def test_reads_mp4_returned_in_comfy_images_list(self):
        history = {"outputs": {"17": {"images": [
            {"filename": "preview.png", "type": "output"},
            {"filename": "h3_studio_run_000_00001_.mp4", "subfolder": "video", "type": "output"},
        ]}}}
        response = io.BytesIO(b"video bytes")
        with tempfile.TemporaryDirectory() as temp, patch.object(runner, "COMFY_URL", "http://comfy"), \
             patch.object(runner.urllib.request, "urlopen", return_value=response) as open_url:
            target = Path(temp) / "shot.mp4"
            runner._download_video(history, target, "h3_studio_run_000")
            self.assertEqual(target.read_bytes(), b"video bytes")
            self.assertIn("filename=h3_studio_run_000_00001_.mp4", open_url.call_args.args[0])

    def test_continuity_prefix_trim_uses_exact_24fps_frames(self):
        with tempfile.TemporaryDirectory() as temp:
            def fake_ffmpeg(command, **kwargs):
                Path(command[-1]).write_bytes(b"trimmed")
            with patch.object(runner.subprocess, "run", side_effect=fake_ffmpeg) as run:
                video = Path(temp) / "shot.mp4"
                video.write_bytes(b"source")
                runner._trim_prefix(video, 22 / 24, 5)
                command = run.call_args.args[0]
                self.assertIn("fps=24,trim=start_frame=22:end_frame=142,setpts=PTS-STARTPTS", command)
                self.assertIn("atrim=start=0.916667:duration=5.000000,asetpts=PTS-STARTPTS", command)
                self.assertEqual(command[command.index("-frames:v") + 1], "120")
                self.assertEqual(video.read_bytes(), b"trimmed")


if __name__ == "__main__":
    unittest.main()

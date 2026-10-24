"""Existing-video SeedVR2 jobs must avoid H3 and preserve the source."""

import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import runner
from server.routes import runs as run_routes


class Response:
    def __init__(self):
        self.code = None
        self.body = None

    def json(self, code, body):
        self.code, self.body = code, body

    def send_error(self, code):
        self.code = code


class PostUpscaleTests(unittest.TestCase):
    def test_route_queues_existing_video_without_h3_job(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "original.mp4").write_bytes(b"test")
            data = {"projects": {"p": {"id": "p", "shots": [],
                                      "upscale_resolution": 1080,
                                      "upscale_max_resolution": 1920}},
                    "runs": {"old": {"id": "old", "project_id": "p", "status": "completed",
                                     "merged": "original.mp4", "results": []}}}
            request = Response()
            with patch.object(run_routes, "read", return_value=data), \
                    patch.object(run_routes, "OUTPUTS", root), \
                    patch.object(run_routes, "_video_dimensions", return_value=(640, 360)), \
                    patch.object(run_routes, "reserve_runs",
                                 side_effect=lambda runs: data["runs"].update({run["id"]: run for run in runs}) or None), \
                    patch.object(run_routes, "threading", types.SimpleNamespace(Thread=lambda **kwargs: _FakeThread(kwargs))):
                run_routes.post_api_run_post_upscale(request, "", ["api", "runs", "old", "post-upscale"], None)
            self.assertEqual(request.code, 202)
            self.assertEqual(request.body["kind"], "post_upscale")
            self.assertEqual(request.body["source_file"], "original.mp4")
            self.assertEqual(request.body["upscale_resolution"], 1080)

    def test_worker_uses_seedvr2_and_keeps_source_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "original.mp4"
            source.write_bytes(b"original")
            data = {"runs": {"job": {"id": "job", "source_file": "original.mp4",
                                    "source_run_id": "old", "source_shot_id": "shot",
                                    "upscale_resolution": 1080,
                                    "upscale_max_resolution": 1920, "seed": 42}}}
            def update(_run_id, **fields):
                data["runs"]["job"].update(fields)
            def download(_history, destination, expected_prefix):
                self.assertEqual(expected_prefix, "h3_studio_job_seedvr2")
                destination.write_bytes(b"upscaled")
            with patch.object(runner, "read", return_value=data), \
                    patch.object(runner, "OUTPUTS", root), \
                    patch.object(runner, "COMFY_INPUT", root), \
                    patch.object(runner, "update_run", side_effect=update), \
                    patch.object(runner, "_video_dimensions", side_effect=[(640, 360), (1280, 720)]), \
                    patch.object(runner, "build_seedvr2_graph", return_value={"seedvr2": {}}) as graph, \
                    patch.object(runner, "_submit", return_value="prompt"), \
                    patch.object(runner, "_wait", return_value={}), \
                    patch.object(runner, "_download_video", side_effect=download), \
                    patch.object(runner, "subprocess") as process:
                process.run.side_effect = lambda command, **kwargs: (Path(command[-1]).write_bytes(b"with audio"))
                runner.run_post_upscale("job")
            self.assertEqual(data["runs"]["job"]["status"], "completed")
            self.assertEqual(source.read_bytes(), b"original")
            self.assertEqual((root / "job" / "post_upscaled.mp4").read_bytes(), b"with audio")
            self.assertEqual(graph.call_count, 1)

    def test_route_rejects_while_another_director_task_is_active(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "original.mp4").write_bytes(b"test")
            data = {"projects": {"p": {"id": "p", "shots": [],
                                      "upscale_resolution": 1080,
                                      "upscale_max_resolution": 1920}},
                    "runs": {"old": {"id": "old", "project_id": "p", "status": "completed",
                                     "merged": "original.mp4", "results": []}}}
            request = Response()
            with patch.object(run_routes, "read", return_value=data), \
                    patch.object(run_routes, "OUTPUTS", root), \
                    patch.object(run_routes, "_video_dimensions", return_value=(640, 360)), \
                    patch.object(run_routes, "reserve_runs", return_value={"id": "active"}), \
                    patch.object(run_routes.threading, "Thread") as thread:
                run_routes.post_api_run_post_upscale(request, "", ["api", "runs", "old", "post-upscale"], None)
            self.assertEqual(request.code, 409)
            self.assertEqual(request.body["active_run_id"], "active")
            thread.assert_not_called()

    def test_route_rejects_target_that_cannot_enlarge_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "original.mp4").write_bytes(b"test")
            data = {"projects": {"p": {"id": "p", "shots": [],
                                      "upscale_resolution": 720,
                                      "upscale_max_resolution": 1280}},
                    "runs": {"old": {"id": "old", "project_id": "p", "status": "completed",
                                     "merged": "original.mp4", "results": []}}}
            with patch.object(run_routes, "read", return_value=data), \
                    patch.object(run_routes, "OUTPUTS", root), \
                    patch.object(run_routes, "_video_dimensions", return_value=(1280, 704)):
                with self.assertRaisesRegex(ValueError, "请先在高级设置提高超分尺寸"):
                    run_routes.post_api_run_post_upscale(Response(), "",
                                                         ["api", "runs", "old", "post-upscale"], None)


class _FakeThread:
    def __init__(self, kwargs):
        self.kwargs = kwargs

    def start(self):
        pass

import unittest
import io
from unittest.mock import patch

import runner


class ModelSwitchTests(unittest.TestCase):
    def setUp(self):
        runner._ACTIVE_MODEL_FAMILY = None
        runner._ACTIVE_MODEL_RUN_ID = None

    def test_new_batch_reuses_same_family_model_cache(self):
        graph = {"1": {"class_type": "MiniMaxH3ReferenceToVideo"}}
        def fake_api(path, body=None):
            if path == "/queue":
                return {"queue_running": [], "queue_pending": []}
            if path == "/prompt":
                return {"prompt_id": "prompt"}
            raise AssertionError(path)
        with patch.object(runner, "api", side_effect=fake_api), \
             patch.object(runner, "read", return_value={"runs": {}}), \
             patch.object(runner, "update_run") as update_run, \
             patch.object(runner, "_release_idle_local_models", return_value={}) as helpers, \
             patch.object(runner, "_release_comfy_memory") as release:
            runner._submit(graph, "batch1")
            runner._submit(graph, "batch1")
            runner._submit(graph, "batch2")
        self.assertEqual(release.call_count, 0)
        self.assertEqual(helpers.call_count, 2)
        self.assertTrue(any(call.args == ("batch1",)
                            and call.kwargs.get("memory_switch", {}).get("status") == "preserved"
                            for call in update_run.call_args_list))
        self.assertTrue(any(call.args == ("batch2",)
                            and call.kwargs.get("memory_switch", {}).get("status") == "reused"
                            for call in update_run.call_args_list))

    def test_sleeping_prompt_model_is_not_woken_by_preparation(self):
        with patch.object(runner.urllib.request, "build_opener") as build:
            build.return_value.open.return_value = io.StringIO('{"is_sleeping":true}')
            self.assertEqual(runner._release_idle_local_models()["prompt_model"], "sleeping")
        build.return_value.open.assert_called_once_with("http://127.0.0.1:11434/props", timeout=2)

    def test_preparation_waits_until_idle_prompt_model_has_unloaded(self):
        with patch.object(runner.urllib.request, "build_opener") as build, \
             patch.object(runner.time, "sleep"):
            build.return_value.open.side_effect = [
                io.StringIO('{"is_sleeping":false}'),
                io.StringIO('[{"is_processing":false}]'),
                io.StringIO('{"is_sleeping":false}'),
                io.StringIO('{"is_sleeping":true}'),
            ]
            self.assertEqual(runner._release_idle_local_models()["prompt_model"], "sleeping")

    def test_preparation_preserves_a_prompt_request_already_in_progress(self):
        with patch.object(runner.urllib.request, "build_opener") as build, \
             patch.object(runner.time, "sleep") as sleep:
            build.return_value.open.side_effect = [
                io.StringIO('{"is_sleeping":false}'),
                io.StringIO('[{"is_processing":true}]'),
            ]
            self.assertEqual(runner._release_idle_local_models()["prompt_model"], "busy")
        sleep.assert_not_called()

    def test_detects_supported_model_families(self):
        self.assertEqual(runner._graph_model_family({"1": {"class_type": "MiniMaxH3ModelLoader"}}), "h3")
        self.assertEqual(runner._graph_model_family({"1": {"class_type": "SeedVR2LoadDiTModel"}}), "seedvr2")
        self.assertEqual(runner._graph_model_family({"1": {"class_type": "XB_QwenPromptPreset"}}), "qwen_image")

    def test_switch_releases_once_but_same_family_reuses_models(self):
        qwen = {"1": {"class_type": "XB_QwenPromptPreset"}}
        h3 = {"1": {"class_type": "MiniMaxH3ReferenceToVideo"}}
        prompt_responses = iter(({"prompt_id": "q1"}, {"prompt_id": "q2"}, {"prompt_id": "h1"}))

        def fake_api(path, body=None):
            if path == "/queue":
                return {"queue_running": [], "queue_pending": []}
            if path == "/prompt":
                return next(prompt_responses)
            raise AssertionError(path)

        with patch.object(runner, "api", side_effect=fake_api), \
             patch.object(runner, "_release_comfy_memory") as release:
            self.assertEqual(runner._submit(qwen), "q1")
            self.assertEqual(runner._submit(qwen), "q2")
            self.assertEqual(runner._submit(h3), "h1")

        self.assertEqual(release.call_count, 1)
        self.assertEqual(runner._ACTIVE_MODEL_FAMILY, "h3")

    def test_switch_waits_for_idle_queue_before_release(self):
        runner._ACTIVE_MODEL_FAMILY = "qwen_image"
        h3 = {"1": {"class_type": "MiniMaxH3ReferenceToVideo"}}
        queues = iter((
            {"queue_running": [[1, "old"]], "queue_pending": []},
            {"queue_running": [], "queue_pending": []},
        ))

        def fake_api(path, body=None):
            if path == "/queue":
                return next(queues)
            if path == "/prompt":
                return {"prompt_id": "h1"}
            raise AssertionError(path)

        with patch.object(runner, "api", side_effect=fake_api), \
             patch.object(runner, "_release_comfy_memory") as release, \
             patch.object(runner.time, "sleep") as sleep:
            self.assertEqual(runner._submit(h3), "h1")

        release.assert_called_once_with()
        sleep.assert_called_once_with(1)


if __name__ == "__main__":
    unittest.main()

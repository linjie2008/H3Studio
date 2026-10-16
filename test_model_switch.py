import unittest
from unittest.mock import patch

import runner


class ModelSwitchTests(unittest.TestCase):
    def setUp(self):
        runner._ACTIVE_MODEL_FAMILY = None

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

        self.assertEqual(release.call_count, 2)
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

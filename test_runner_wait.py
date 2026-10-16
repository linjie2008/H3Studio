import unittest
import urllib.error
from unittest.mock import patch

import runner


class RunnerWaitTests(unittest.TestCase):
    def test_prompt_state_finds_history_before_queue(self):
        with patch.object(runner, "api", return_value={"prompt": {"outputs": {}}}) as api:
            self.assertEqual(runner._prompt_state("prompt"), "completed")
        api.assert_called_once_with("/history/prompt")

    def test_prompt_state_finds_running_prompt(self):
        with patch.object(runner, "api", side_effect=[{}, {"queue_running": [[1, "prompt"]]}]):
            self.assertEqual(runner._prompt_state("prompt"), "active")

    def test_resume_prompt_reuses_only_matching_known_stage(self):
        state = {"runs": {"run": {"prompt_id": "prompt", "active_stage": "h3",
                                    "active_shot_id": "shot"}}}
        with patch.object(runner, "read", return_value=state), \
             patch.object(runner, "_prompt_state", return_value="completed"):
            self.assertEqual(runner._resume_prompt("run", "h3", "shot"), "prompt")
            self.assertIsNone(runner._resume_prompt("run", "seedvr2", "shot"))

    def test_transient_comfy_timeout_does_not_fail_submitted_video(self):
        completed = {"status": {"status_str": "success"}, "outputs": {}}
        with patch.object(runner, "read", return_value={"runs": {"run": {}}}), \
             patch.object(runner, "api", side_effect=[urllib.error.URLError("timed out"),
                                                        {"prompt": completed}]), \
             patch.object(runner, "update_run") as update, \
             patch.object(runner.time, "sleep"):
            self.assertEqual(runner._wait("prompt", "run"), completed)
        update.assert_called_once()
        self.assertIn("正在重试", update.call_args.kwargs["current"])


if __name__ == "__main__":
    unittest.main()

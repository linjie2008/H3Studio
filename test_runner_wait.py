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

    def test_completed_job_releases_models_when_director_and_comfy_are_idle(self):
        state = {"runs": {"done": {"id": "done", "status": "completed"}}}
        with patch.object(runner, "read", return_value=state), \
             patch.object(runner, "_comfy_queue_is_idle", return_value=(True, "空闲")), \
             patch.object(runner, "_release_comfy_memory") as release, \
             patch.object(runner, "update_run") as update, \
             patch.object(runner, "change", side_effect=lambda fn: fn(state)), \
             patch.object(runner, "_ACTIVE_MODEL_FAMILY", "h3"), \
             patch.object(runner, "_ACTIVE_MODEL_RUN_ID", "done"):
            runner.release_idle_comfy_after_job("done")
            self.assertIsNone(runner._ACTIVE_MODEL_FAMILY)
            self.assertIsNone(runner._ACTIVE_MODEL_RUN_ID)
        release.assert_called_once()
        self.assertIn("已请求卸载", update.call_args.kwargs["memory_cleanup"])
        self.assertEqual(state["runs"]["done"]["memory_events"][-1]["status"], "released")

    def test_completed_job_keeps_models_for_next_queued_job(self):
        state = {"runs": {"done": {"id": "done", "status": "completed"},
                          "next": {"id": "next", "status": "queued"}}}
        with patch.object(runner, "read", return_value=state), \
             patch.object(runner, "_release_comfy_memory") as release, \
             patch.object(runner, "update_run"), \
             patch.object(runner, "change", side_effect=lambda fn: fn(state)):
            runner.release_idle_comfy_after_job("done")
        release.assert_not_called()
        self.assertEqual(state["runs"]["done"]["memory_events"][-1]["status"], "skipped_queue_busy")


if __name__ == "__main__":
    unittest.main()

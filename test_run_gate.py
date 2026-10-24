"""Admission checks for director jobs that share one GPU."""

import threading
import unittest
from unittest.mock import patch

import store


class RunGateTests(unittest.TestCase):
    def setUp(self):
        self.state = {"runs": {}}
        self.lock = threading.Lock()

        def change(fn):
            with self.lock:
                return fn(self.state)

        self.patcher = patch.object(store, "change", change)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def test_other_project_is_blocked_until_finishing_pass_completes(self):
        first = {"id": "first", "project_id": "a", "status": "running"}
        second = {"id": "second", "project_id": "b", "status": "queued"}
        self.assertIsNone(store.reserve_runs([first]))
        self.assertEqual(store.reserve_runs([second])["id"], "first")
        self.assertNotIn("second", self.state["runs"])
        first["status"] = "completed"
        self.assertIsNone(store.reserve_runs([second]))

    def test_waiting_for_upscale_confirmation_still_blocks(self):
        self.state["runs"]["first"] = {"id": "first", "status": "awaiting_upscale"}
        self.assertEqual(store.reserve_runs([{"id": "second", "status": "queued"}])["id"], "first")

    def test_concurrent_clicks_admit_only_one_job(self):
        barrier = threading.Barrier(3)
        results = []

        def submit(index):
            barrier.wait()
            results.append(store.reserve_runs([{"id": str(index), "status": "queued"}]))

        threads = [threading.Thread(target=submit, args=(index,)) for index in (1, 2)]
        for thread in threads:
            thread.start()
        barrier.wait()
        for thread in threads:
            thread.join()
        self.assertEqual(len(self.state["runs"]), 1)
        self.assertEqual(sum(result is None for result in results), 1)


if __name__ == "__main__":
    unittest.main()

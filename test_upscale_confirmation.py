"""Turning SeedVR2 off must also release a run that waits for upscale confirmation.

The confirmation panel has two ways out: press a button, or turn the upscale
off.  The waiting loop used to only watch for the buttons, so a run stayed
parked in "awaiting_upscale" even after the switch had been turned off.
"""

import unittest
from unittest import mock

import runner


class UpscaleConfirmationTests(unittest.TestCase):
    def _state(self, project):
        return {"projects": {"p1": project}, "runs": {}, "assets": {}}

    def test_switch_off_during_the_wait_releases_the_run(self):
        project = {"id": "p1", "upscale": True, "confirm_upscale": True,
                   "shots": [{"id": "s1", "name": "shot", "upscale": True}]}
        shot = project["shots"][0]
        with mock.patch.object(runner, "read", return_value=self._state(project)):
            self.assertTrue(runner._upscale_still_enabled(shot, project, "p1"))

        # The user turns the shot-level switch off while the run waits.
        project["shots"][0]["upscale"] = False
        with mock.patch.object(runner, "read", return_value=self._state(project)):
            self.assertFalse(runner._upscale_still_enabled(shot, project, "p1"))

    def test_project_switch_off_releases_the_run(self):
        project = {"id": "p1", "upscale": True, "confirm_upscale": True,
                   "shots": [{"id": "s1", "name": "shot"}]}
        shot = project["shots"][0]
        with mock.patch.object(runner, "read", return_value=self._state(project)):
            self.assertTrue(runner._upscale_still_enabled(shot, project, "p1"))

        project["upscale"] = False
        with mock.patch.object(runner, "read", return_value=self._state(project)):
            self.assertFalse(runner._upscale_still_enabled(shot, project, "p1"))

    def test_shot_override_wins_over_a_disabled_project(self):
        """A shot that explicitly asks for the upscale keeps it - that is the
        per-shot switch the user can see in the panel."""
        project = {"id": "p1", "upscale": False, "confirm_upscale": True,
                   "shots": [{"id": "s1", "name": "shot", "upscale": True}]}
        shot = project["shots"][0]
        with mock.patch.object(runner, "read", return_value=self._state(project)):
            self.assertTrue(runner._upscale_still_enabled(shot, project, "p1"))

    def test_missing_live_project_falls_back_to_the_snapshot(self):
        project = {"id": "p1", "upscale": True, "shots": [{"id": "s1"}]}
        shot = project["shots"][0]
        with mock.patch.object(runner, "read", return_value={"projects": {}}):
            self.assertTrue(runner._upscale_still_enabled(shot, project, "p1"))


if __name__ == "__main__":
    unittest.main()

import unittest
from unittest.mock import patch

from runner import _native_take_shot, shot_refine_mode, shot_upscale
from store import new_project, new_shot


class NativeTakeTests(unittest.TestCase):
    def test_short_sequence_uses_one_reference_list_and_timecoded_beats(self):
        project = new_project("native test")
        first, second = new_shot(), new_shot()
        first.update(mode="r2v", duration=5, prompt="Person from <Picture 1> faces camera.",
                     ref_images=["person_a"], audio_mode="mute")
        second.update(mode="r2v", duration=5, prompt="Person from <Picture 2> enters camera.",
                      ref_images=["person_b"], audio_mode="mute")
        project.update(shots=[first, second], reference_image_numbers={"person_a": 1, "person_b": 2},
                       continuous_take=True)
        assets = {"person_a": {"kind": "image", "name": "A", "project_id": project["id"]},
                  "person_b": {"kind": "image", "name": "B", "project_id": project["id"]}}
        with patch("runner.read", return_value={"assets": assets}):
            take = _native_take_shot(project["shots"], project)
        self.assertEqual(take["duration"], 10)
        self.assertEqual(take["ref_images"], ["person_a", "person_b"])
        self.assertIn("[0s-5s]", take["prompt"])
        self.assertIn("[5s-10s]", take["prompt"])
        self.assertIn("Person from <Picture 2> enters", take["prompt"])

    def test_each_native_take_beat_keeps_its_camera_and_action_controls(self):
        project = new_project("directing controls")
        first, second = new_shot(), new_shot()
        first.update(mode="r2v", prompt="A person waits.", camera_move="推近", camera_speed="缓慢",
                     action_intensity="轻微")
        second.update(mode="r2v", prompt="The person runs.", camera_move="跟随主体", camera_speed="快速",
                      action_intensity="明显")
        project["shots"] = [first, second]
        with patch("runner.read", return_value={"assets": {}}):
            take = _native_take_shot(project["shots"], project)
        self.assertIn("continuous dolly-in", take["prompt"])
        self.assertIn("continuously track and follow", take["prompt"])
        self.assertIn("fast and decisive", take["prompt"])
        self.assertTrue(take["suppress_director_controls"])

    def test_long_sequence_keeps_segmented_path(self):
        project = new_project("long test")
        project["shots"] = [new_shot() for _ in range(4)]
        self.assertIsNone(_native_take_shot(project["shots"], project))

    def test_explicit_segmented_mode_keeps_two_five_second_shots(self):
        project = new_project("guided handoff")
        project["shots"] = [new_shot(), new_shot()]
        project["continuous_render_mode"] = "segmented"
        self.assertIsNone(_native_take_shot(project["shots"], project))

    def test_shot_postprocessing_overrides_project_defaults(self):
        project = new_project("shot postprocessing")
        project.update(refine_mode="latent", upscale=True)
        first, second = new_shot(), new_shot()
        first.update(mode="r2v", refine_mode="off", upscale=False)
        second.update(mode="r2v", refine_mode="same", upscale=True)
        project["shots"] = [first, second]
        self.assertEqual(shot_refine_mode(first, project), "off")
        self.assertFalse(shot_upscale(first, project))
        self.assertEqual(shot_refine_mode(second, project), "same")
        self.assertTrue(shot_upscale(second, project))
        self.assertIsNone(_native_take_shot(project["shots"], project))
        project["shots"] = [first, new_shot()]
        self.assertEqual(shot_refine_mode(project["shots"][1], project), "latent")
        self.assertTrue(shot_upscale(project["shots"][1], project))
        project["generation_speed"] = "goldfish"
        self.assertEqual(shot_refine_mode(second, project), "off")


if __name__ == "__main__":
    unittest.main()

import unittest
from unittest.mock import patch

from runner import prepare_shot


class ProjectReferenceNumberingTests(unittest.TestCase):
    def test_global_picture_number_is_localized_for_each_shots_h3_slots(self):
        project = {
            "id": "project",
            "shared_prompt": "The scene follows <Picture 12>.",
            "shared_refs": {},
            "reference_image_numbers": {"first": 5, "third": 12},
            "shots": [],
            "characters": [],
            "entities": [],
        }
        shot = {
            "id": "shot-2",
            "mode": "r2v",
            "ref_images": ["third"],
            "prompt": "Continue using <Picture 12> as the same character.",
        }
        with patch("runner.read", return_value={"assets": {
                "third": {"id": "third", "kind": "image", "name": "third.png"},
                "first": {"id": "first", "kind": "image", "name": "first.png"}}}):
            prepared = prepare_shot(shot, project)
        self.assertIn("The scene follows <Picture 1>.", prepared["prompt"])
        self.assertIn("Continue using <Picture 1> as the same character.", prepared["prompt"])
        self.assertEqual(shot["prompt"], "Continue using <Picture 12> as the same character.")

    def test_missing_global_picture_alias_fails_before_generation(self):
        with patch("runner.read", return_value={"assets": {
                "first": {"id": "first", "kind": "image", "name": "first.png"}}}):
            with self.assertRaisesRegex(ValueError, "不存在的图片编号：99"):
                prepare_shot(
                    {"id": "shot", "mode": "r2v", "ref_images": ["first"], "prompt": "<Picture 99>"},
                    {"id": "project", "shared_refs": {}, "reference_image_numbers": {"first": 5},
                     "shots": [], "characters": [], "entities": []},
                )

    def test_gallery_selected_image_is_not_sent_without_prompt_tag(self):
        project = {
            "id": "project", "shared_refs": {},
            "reference_image_numbers": {"tagged": 1, "clicked": 2},
            "shots": [], "characters": [], "entities": [],
        }
        shot = {
            "id": "shot", "mode": "r2v", "prompt": "Use <Picture 1>.",
            "ref_images": ["tagged", "clicked"],
            "manual_ref_images": ["clicked"],
        }
        with patch("runner.read", return_value={"assets": {
                "tagged": {"id": "tagged", "kind": "image", "name": "tagged.png"},
                "clicked": {"id": "clicked", "kind": "image", "name": "clicked.png"}}}):
            prepared = prepare_shot(shot, project)
        self.assertEqual(prepared["ref_images"], ["tagged"])
        self.assertNotIn("<Picture 2>", shot["prompt"])


if __name__ == "__main__":
    unittest.main()

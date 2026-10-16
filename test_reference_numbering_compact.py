import unittest

from reference_numbering import (assign_single_image_per_shot,
                                 normalized_project_image_references,
                                 renumber_project_image_references)


class CompactImageReferenceNumberingTests(unittest.TestCase):
    def test_delete_compacts_remaining_ids_and_rewrites_prompts(self):
        project = {
            "reference_image_numbers": {"first": 1, "deleted": 2, "third": 3},
            "shared_prompt": "Scene from <Picture 3>; portrait <Picture 2>.",
            "shots": [{"prompt": "Use <Picture 1> then <Picture 3>."}],
        }

        renumber_project_image_references(project, ["first", "third"])

        self.assertEqual(project["reference_image_numbers"], {"first": 1, "third": 2})
        self.assertEqual(project["shared_prompt"], "Scene from <Picture 2>; portrait .")
        self.assertEqual(project["shots"][0]["prompt"], "Use <Picture 1> then <Picture 2>.")

    def test_new_images_follow_survivors_and_unknown_tags_remain(self):
        project = {
            "reference_image_numbers": {"old": 4},
            "shots": [{"prompt": "Use <Picture 4> and <Picture 99>."}],
        }

        renumber_project_image_references(project, ["old", "new"])

        self.assertEqual(project["reference_image_numbers"], {"old": 1, "new": 2})
        self.assertEqual(project["shots"][0]["prompt"], "Use <Picture 1> and <Picture 99>.")

    def test_material_sidebar_order_controls_numbers_without_changing_image_identity(self):
        project = {
            "reference_image_numbers": {"first": 1, "second": 2, "third": 3},
            "asset_order": ["third", "first", "second"],
            "shared_prompt": "Backdrop @图片3",
            "shots": [{"prompt": "Use <Picture 1> with @图片2 and <Picture 3>."}],
        }
        renumber_project_image_references(project, ["first", "second", "third"])
        self.assertEqual(project["reference_image_numbers"], {"third": 1, "first": 2, "second": 3})
        self.assertEqual(project["shared_prompt"], "Backdrop @图片1")
        self.assertEqual(project["shots"][0]["prompt"],
                         "Use <Picture 2> with @图片3 and <Picture 1>.")

    def test_moving_third_character_to_fourth_rewrites_all_bound_prompts(self):
        project = {
            "reference_image_numbers": {"one": 1, "two": 2, "third": 3,
                                        "fourth": 4, "scene": 5},
            "asset_order": ["one", "two", "fourth", "third", "scene"],
            "shared_refs": {"images": ["scene"]},
            "shared_prompt": "Background <Picture 5>",
            "shots": [
                {"mode": "r2v", "prompt": "Person <Picture 3> at <Picture 5>",
                 "ref_images": ["third"]},
                {"mode": "r2v", "prompt": "Person @图片4 at <Picture 5>",
                 "ref_images": ["fourth"]},
            ],
        }

        renumber_project_image_references(
            project, ["one", "two", "third", "fourth", "scene"])

        self.assertEqual(project["shots"][0]["prompt"],
                         "Person <Picture 4> at <Picture 5>")
        self.assertEqual(project["shots"][1]["prompt"],
                         "Person @图片3 at <Picture 5>")
        self.assertEqual([shot["ref_images"] for shot in project["shots"]],
                         [["third"], ["fourth"]])

    def test_all_shots_rebind_from_reordered_prompt_tags_in_one_pass(self):
        project = {
            "reference_image_numbers": {"scene": 1, "one": 2, "two": 3},
            "asset_order": ["two", "scene", "one"],
            "shared_refs": {"images": ["scene"]},
            "shots": [
                {"mode": "r2v", "prompt": "<Picture 2> in <Picture 1>",
                 "ref_images": ["two"]},
                {"mode": "r2v", "prompt": "<Picture 3> in <Picture 1>",
                 "ref_images": ["one"]},
            ],
        }

        renumber_project_image_references(project, ["scene", "one", "two"])

        self.assertEqual(project["reference_image_numbers"],
                         {"two": 1, "scene": 2, "one": 3})
        self.assertEqual([shot["prompt"] for shot in project["shots"]],
                         ["<Picture 3> in <Picture 2>", "<Picture 1> in <Picture 2>"])
        self.assertEqual([shot["ref_images"] for shot in project["shots"]],
                         [["one"], ["two"]])

    def test_explicit_tags_remove_stale_bindings_and_untagged_legacy_refs(self):
        project = {
            "reference_image_numbers": {"scene": 1, "old": 2, "new": 3},
            "shared_refs": {"images": ["scene"]},
            "shots": [
                {"mode": "r2v", "prompt": "<Picture 3> in <Picture 1>",
                 "ref_images": ["old", "new"]},
                {"mode": "r2v", "prompt": "A figure walks forward",
                 "ref_images": ["old"]},
            ],
        }

        renumber_project_image_references(project, ["scene", "old", "new"])

        self.assertEqual(project["shots"][0]["ref_images"], ["new"])
        self.assertEqual(project["shots"][1]["ref_images"], [])

    def test_click_selected_image_is_ui_only_without_prompt_tag(self):
        project = {
            "reference_image_numbers": {"tagged": 1, "clicked": 2},
            "shots": [{"mode": "r2v", "prompt": "Use <Picture 1>",
                       "manual_ref_images": ["clicked"],
                       "ref_images": ["tagged", "clicked"]}],
        }
        renumber_project_image_references(project, ["tagged", "clicked"])
        shot = project["shots"][0]
        self.assertEqual(shot["prompt"], "Use <Picture 1>")
        self.assertEqual(shot["ref_images"], ["tagged"])
        self.assertEqual(shot["manual_ref_images"], ["clicked"])

        shot["prompt"] = "No explicit picture"
        renumber_project_image_references(project, ["tagged", "clicked"])
        self.assertEqual(shot["ref_images"], [])
        self.assertEqual(shot["manual_ref_images"], ["clicked"])

    def test_bulk_sequence_assignment_keeps_shared_scene_and_fixes_all_shots(self):
        project = {
            "reference_image_numbers": {"first": 1, "second": 2, "third": 3,
                                        "wrong": 4, "scene": 5},
            "shared_refs": {"images": ["scene"]},
            "shots": [
                {"id": "a", "mode": "r2v", "prompt": "<Picture 4> in <Picture 5>",
                 "ref_images": ["wrong"]},
                {"id": "b", "mode": "r2v", "prompt": "@图片4 in <Picture 5>",
                 "ref_images": ["wrong"]},
                {"id": "c", "mode": "r2v", "prompt": "A person in <Picture 5>",
                 "ref_images": []},
            ],
        }

        assign_single_image_per_shot(project, {"a": "first", "b": "second", "c": "third"})

        self.assertEqual([shot["prompt"] for shot in project["shots"]],
                         ["<Picture 1> in <Picture 5>",
                          "@图片2 in <Picture 5>",
                          "A person in <Picture 5> <Picture 3>"])
        self.assertEqual([shot["ref_images"] for shot in project["shots"]],
                         [["first"], ["second"], ["third"]])

    def test_render_snapshot_normalizes_stale_order_without_mutating_storage(self):
        project = {
            "id": "project", "asset_order": ["second", "first"],
            "reference_image_numbers": {"first": 1, "second": 2},
            "shared_refs": {"images": []},
            "shots": [{"id": "shot", "mode": "r2v", "prompt": "Use <Picture 1>",
                       "ref_images": ["second"]}],
        }
        assets = {asset_id: {"kind": "image", "project_id": "project"}
                  for asset_id in ("first", "second")}

        snapshot = normalized_project_image_references(project, assets)

        self.assertEqual(snapshot["reference_image_numbers"], {"second": 1, "first": 2})
        self.assertEqual(snapshot["shots"][0]["prompt"], "Use <Picture 2>")
        self.assertEqual(snapshot["shots"][0]["ref_images"], ["first"])
        self.assertEqual(project["shots"][0]["prompt"], "Use <Picture 1>")


if __name__ == "__main__":
    unittest.main()

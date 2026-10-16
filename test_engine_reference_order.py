import unittest
from unittest.mock import patch

from engine import _compile_full_reference_prompt, compile_shot_prompt
from runner import prepare_shot


class EngineReferenceOrderTests(unittest.TestCase):
    def test_creative_keyframes_drop_dormant_gallery_refs_without_changing_r2v(self):
        assets = {key: {"id": key, "name": key + ".png", "kind": "image"}
                  for key in ("first", "last", "stale", "shared")}
        project = {"reference_image_numbers": {"first": 1, "last": 2,
                  "stale": 3, "shared": 4},
                   "shared_refs": {"images": ["shared"]}, "shots": []}
        base = {"id": "s1", "first_image": "first", "last_image": "last",
                "prompt": "人物向前走。", "ref_images": ["stale"],
                "ref_videos": [], "ref_audios": []}
        with patch("runner.read", return_value={"assets": assets}):
            creative = prepare_shot({**base, "mode": "fl2v"}, project)
            multi_reference = prepare_shot({**base, "mode": "r2v",
                "prompt": "人物参考 <Picture 3>，在 <Picture 4> 场景里向前走。"}, project)
        self.assertEqual(creative["ref_images"], [])
        self.assertEqual((creative["first_image"], creative["last_image"]), ("first", "last"))
        self.assertEqual(multi_reference["ref_images"], ["shared", "stale"])
        self.assertIn("<Picture 2>", multi_reference["prompt"])

    def test_image_reference_preserves_original_scene_by_default(self):
        prompt = _compile_full_reference_prompt({
            "mode": "r2v", "ref_images": ["scene"],
            "_h3_reference_names": {"scene": "uploaded-scene.png"},
        }, "The person walks forward.", "N/A", "N/A")
        self.assertIn("<Picture 1> ([Shot 1]): attribute_transfer", prompt)
        self.assertIn("role is exactly the one assigned to this tag in the detailed shot description", prompt)
        self.assertIn("follow any explicit request to reproduce its content or layout", prompt)

    def test_explicit_weak_image_reference_remains_weak(self):
        prompt = _compile_full_reference_prompt({
            "mode": "r2v", "ref_images": ["style"],
            "ref_image_roles": {"style": "weak_reference"},
        }, "The person walks forward.", "N/A", "N/A")
        self.assertIn("<Picture 1> ([Shot 1]): weak_reference", prompt)

    def test_verified_image_description_overrides_conflicting_prompt(self):
        prompt = _compile_full_reference_prompt({
            "mode": "r2v", "ref_images": ["person", "scene"],
            "_h3_reference_descriptions": {
                "person": "Portrait sheet of a uniformed man on a white background; identity only",
                "scene": "A burned battlefield with bare trees and smoke; no paved road",
            },
        }, "The man walks down a road with lane markings.", "N/A", "N/A")
        self.assertIn("A burned battlefield with bare trees and smoke; no paved road", prompt)
        self.assertLess(prompt.index("Verified user description"), prompt.index("lane markings"))
        self.assertIn("Do not assume that every reference must appear as a separate in-frame element", prompt)

    def test_source_video_is_first_and_duplicate_reference_is_not_numbered_twice(self):
        prompt = _compile_full_reference_prompt({
            "mode": "rv2v",
            "source_video": "source",
            "ref_videos": ["source", "motion"],
            "_h3_reference_names": {"source": "source.mp4", "motion": "motion.mp4"},
        }, "", "", "")
        self.assertIn("<Video 1> is the source video being edited (source.mp4)", prompt)
        self.assertIn("<Video 2> is a video reference for motion and temporal structure (motion.mp4)", prompt)
        self.assertNotIn("<Video 3>", prompt)

    def test_continuity_shot_prompt_explicitly_preserves_handoff(self):
        prompt = compile_shot_prompt({"prompt": "A woman walks through a corridor.",
                                      "continuity_video": "previous_tail.mp4", "continuity_strength": "strong",
                                      "continuous_take": True, "mode": "t2v"})
        self.assertIn("One-take continuity", prompt)
        self.assertIn("Continue along one physically possible camera path", prompt)

    def test_explicit_global_picture_tag_binds_asset_to_shot_and_localizes_alias(self):
        assets = {
            "image1": {"id": "image1", "name": "first.png", "kind": "image", "file": "first.png"},
            "image4": {"id": "image4", "name": "fourth.png", "kind": "image", "file": "fourth.png"},
        }
        project = {"reference_image_numbers": {"image1": 1, "image4": 4},
                   "shared_refs": {}, "shots": []}
        shot = {"id": "s1", "mode": "r2v", "prompt": "Use @图片4 as the character reference.",
                "ref_images": ["image1"], "ref_videos": [], "ref_audios": []}
        with patch("runner.read", return_value={"assets": assets}):
            prepared = prepare_shot(shot, project)
        self.assertEqual(prepared["ref_images"], ["image4"])
        self.assertEqual(prepared["ref_image_roles"]["image4"], "auto")
        self.assertIn("<Picture 1>", prepared["prompt"])
        self.assertNotIn("@图片4", prepared["prompt"])
        self.assertEqual(prepared["_h3_image_reference_map"][0]["global_number"], 4)
        self.assertEqual(prepared["_h3_image_reference_map"][0]["local_number"], 1)

    def test_global_picture_alias_carries_project_image_description(self):
        assets = {"scene": {"id": "scene", "name": "scene.png", "kind": "image", "file": "scene.png"}}
        project = {"reference_image_numbers": {"scene": 4}, "shared_refs": {}, "shots": [],
                   "image_descriptions": {"scene": "Burned battlefield with smoke, no paved road"}}
        shot = {"id": "s1", "mode": "r2v", "prompt": "Walk across @图片4.",
                "ref_images": [], "ref_videos": [], "ref_audios": []}
        with patch("runner.read", return_value={"assets": assets}):
            prepared = prepare_shot(shot, project)
        self.assertEqual(prepared["ref_images"], ["scene"])
        self.assertEqual(prepared["_h3_reference_descriptions"]["scene"],
                         "Burned battlefield with smoke, no paved road")
        self.assertIn("<Picture 1>", prepared["prompt"])

    def test_explicit_alias_cannot_push_final_images_past_h3_limit(self):
        assets = {f"image{i}": {"id": f"image{i}", "name": f"{i}.png", "kind": "image"}
                  for i in range(1, 11)}
        project = {"reference_image_numbers": {f"image{i}": i for i in range(1, 11)},
                   "shared_refs": {}, "shots": []}
        shot = {"id": "s1", "name": "过量引用", "mode": "r2v",
                "prompt": " ".join(f"@图片{i}" for i in range(1, 11)),
                "ref_images": [f"image{i}" for i in range(1, 10)],
                "ref_videos": [], "ref_audios": []}
        with patch("runner.read", return_value={"assets": assets}):
            with self.assertRaisesRegex(ValueError, "最终参考素材超过 H3 上限"):
                prepare_shot(shot, project)

    def test_unknown_picture_number_fails_instead_of_pointing_to_wrong_slot(self):
        assets = {"image1": {"id": "image1", "name": "first.png", "kind": "image"}}
        project = {"reference_image_numbers": {"image1": 1}, "shared_refs": {}, "shots": []}
        shot = {"id": "s1", "name": "错误编号", "mode": "r2v", "prompt": "Use <Picture 9>.",
                "ref_images": ["image1"], "ref_videos": [], "ref_audios": []}
        with patch("runner.read", return_value={"assets": assets}):
            with self.assertRaisesRegex(ValueError, "不存在的图片编号：9"):
                prepare_shot(shot, project)

    def test_legacy_untagged_manual_reference_is_not_sent(self):
        assets = {"image1": {"id": "image1", "name": "first.png", "kind": "image"}}
        project = {"reference_image_numbers": {"image1": 1}, "shared_refs": {}, "shots": []}
        shot = {"id": "s1", "mode": "r2v", "prompt": "A person walks forward.",
                "ref_images": ["image1"], "ref_videos": [], "ref_audios": []}
        with patch("runner.read", return_value={"assets": assets}):
            prepared = prepare_shot(shot, project)
        self.assertEqual(prepared["ref_images"], [])

    def test_shared_image_still_sent_but_stale_manual_person_is_filtered(self):
        assets = {key: {"id": key, "name": key + ".png", "kind": "image"}
                  for key in ("scene", "old_person", "new_person")}
        project = {"reference_image_numbers": {"scene": 4, "old_person": 7, "new_person": 13},
                   "shared_refs": {"images": ["scene"]}, "shots": []}
        shot = {"id": "s1", "mode": "r2v", "prompt": "<Picture 13> in <Picture 4>.",
                "ref_images": ["old_person", "new_person"], "ref_videos": [], "ref_audios": []}
        with patch("runner.read", return_value={"assets": assets}):
            prepared = prepare_shot(shot, project)
        self.assertEqual(prepared["ref_images"], ["scene", "new_person"])
        self.assertEqual([(item["global_number"], item["local_number"])
                          for item in prepared["_h3_image_reference_map"]], [(4, 1), (13, 2)])
        self.assertIn("<Picture 2> in <Picture 1>", prepared["prompt"])

    def test_duplicate_global_number_is_rejected(self):
        assets = {key: {"id": key, "name": key + ".png", "kind": "image"}
                  for key in ("first", "second")}
        project = {"reference_image_numbers": {"first": 7, "second": 7},
                   "shared_refs": {}, "shots": []}
        shot = {"id": "s1", "mode": "r2v", "prompt": "Use <Picture 7>.",
                "ref_images": [], "ref_videos": [], "ref_audios": []}
        with patch("runner.read", return_value={"assets": assets}):
            with self.assertRaisesRegex(ValueError, "图片编号 7 对应多张素材"):
                prepare_shot(shot, project)

    def test_three_tagged_references_follow_user_assigned_roles_without_scene_guessing(self):
        prompt = _compile_full_reference_prompt({
            "mode": "r2v", "ref_images": ["person", "plane", "runway"],
            "prompt": "<Picture 1>人物作为飞行员，使用<Picture 2>飞机，在<Picture 3>跑道上起飞。",
            "_h3_reference_names": {"person": "1高志航.png", "plane": "plane.png", "runway": "runway.png"},
        }, "飞行员在跑道上", "N/A", "N/A")
        self.assertIn("<Picture 1> is a visual reference", prompt)
        self.assertIn("<Picture 2> is a visual reference", prompt)
        self.assertIn("<Picture 3> is a visual reference", prompt)
        self.assertIn("role is exactly the one assigned to this tag", prompt)
        self.assertIn("Do not assume that every reference must appear as a separate in-frame element", prompt)
        self.assertEqual(prompt.count("attribute_transfer - transfer the visual attributes requested for this reference"), 3)


if __name__ == "__main__":
    unittest.main()

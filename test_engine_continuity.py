import unittest
from pathlib import Path

from engine import build_h3_graph, compile_shot_prompt, lint_shot_prompt
from runner import (prepare_shot, _continuity_policy, _continuity_subject_relation, _handoff_window,
                    _continuity_anchor_frames,
                    _strengthen_incoming_identity_reference, _shot_identity_refs,
                    _select_handoff_tail_trim, _select_handoff_alignment,
                    _smooth_handoff_motion, _match_handoff_exposure,
                    _stable_matching_view, _append_merge_item)
from unittest.mock import patch


class EngineContinuityTests(unittest.TestCase):
    def test_t2v_ignores_stale_frame_selections(self):
        graph = build_h3_graph(
            {"mode": "t2v", "prompt": "A traveler crosses a market.", "duration": 5,
             "first_image": "old_first", "last_image": "old_last"},
            {"width": 480, "height": 864},
            {}, "plain_t2v",
        )
        nodes = list(graph.values())
        self.assertFalse(any(node["class_type"] == "LoadImage" for node in nodes))
        conditioning = next(node for node in nodes if node["class_type"] == "MiniMaxH3ImageToVideo")
        self.assertNotIn("first_frame", conditioning["inputs"])
        self.assertNotIn("last_frame", conditioning["inputs"])

    def test_i2v_ignores_stale_last_frame(self):
        graph = build_h3_graph(
            {"mode": "i2v", "prompt": "A traveler crosses a market.", "duration": 5,
             "first_image": "first", "last_image": "old_last"},
            {"width": 480, "height": 864},
            {"first": "first.png"}, "plain_i2v",
        )
        conditioning = next(node for node in graph.values()
                            if node["class_type"] == "MiniMaxH3ImageToVideo")
        self.assertIn("first_frame", conditioning["inputs"])
        self.assertNotIn("last_frame", conditioning["inputs"])

    def test_continued_fl2v_places_first_frame_after_handoff_clip(self):
        graph = build_h3_graph(
            {"mode": "fl2v", "prompt": "The traveler crosses the market.", "duration": 5,
             "first_image": "opening", "last_image": "ending",
             "continuity_video": "tail", "continuity_frames": 22},
            {"width": 480, "height": 864},
            {"opening": "opening.png", "ending": "ending.png", "tail": "tail.mp4"},
            "continued_fl2v",
        )
        nodes = list(graph.values())
        conditioning = next(node for node in nodes if node["class_type"] == "MiniMaxH3ImageToVideo")
        self.assertNotIn("first_frame", conditioning["inputs"])
        self.assertIn("last_frame", conditioning["inputs"])
        guides = [node for node in nodes if node["class_type"] == "MiniMaxH3AddGuide"]
        self.assertEqual([guide["inputs"]["frame_idx"] for guide in guides], [0, 46])
        opening_load = next(key for key, node in graph.items()
                            if node["class_type"] == "LoadImage"
                            and node["inputs"]["image"] == "opening.png")
        self.assertEqual(guides[1]["inputs"]["image"], [opening_load, 0])
        prompt = next(node["inputs"]["prompt"] for node in nodes
                      if node["class_type"] == "MiniMaxH3ImageToVideo")
        self.assertIn("after a short bridge", prompt)

    def test_standalone_fl2v_keeps_first_and_last_frames(self):
        graph = build_h3_graph(
            {"mode": "fl2v", "prompt": "The traveler crosses the market.", "duration": 5,
             "first_image": "opening", "last_image": "ending"},
            {"width": 480, "height": 864},
            {"opening": "opening.png", "ending": "ending.png"}, "standalone_fl2v",
        )
        conditioning = next(node for node in graph.values()
                            if node["class_type"] == "MiniMaxH3ImageToVideo")
        self.assertIn("first_frame", conditioning["inputs"])
        self.assertIn("last_frame", conditioning["inputs"])
        self.assertFalse(any(node["class_type"] == "MiniMaxH3AddGuide"
                             for node in graph.values()))

    def test_reference_handoff_still_has_only_its_existing_guide(self):
        graph = build_h3_graph(
            {"mode": "r2v", "prompt": "The traveler crosses the market.", "duration": 5,
             "ref_images": ["person"], "continuity_video": "tail", "continuity_frames": 22},
            {"width": 480, "height": 864},
            {"person": "person.png", "tail": "tail.mp4"}, "unchanged_r2v",
        )
        nodes = list(graph.values())
        guides = [node for node in nodes if node["class_type"] == "MiniMaxH3AddGuide"]
        self.assertEqual([guide["inputs"]["frame_idx"] for guide in guides], [0])
        self.assertTrue(any(node["class_type"] == "MiniMaxH3ReferenceToVideo" for node in nodes))

    def test_shared_scene_does_not_hide_a_change_of_person(self):
        project = {"image_descriptions": {"pilot_a": "陈怀民四视图，稳定脸部和服装",
                                          "pilot_b": "柳哲生四视图，稳定脸部和服装"},
                   "reference_image_numbers": {"runway": 1, "pilot_a": 2, "pilot_b": 3},
                   "characters": []}
        assets = {"runway": {"name": "runway.png"},
                  "pilot_a": {"name": "陈怀民 · 四视图"},
                  "pilot_b": {"name": "柳哲生 · 四视图"}}
        first = {"ref_images": ["runway", "pilot_a"], "prompt": "人物来自 <Picture 2>"}
        second = {"ref_images": ["runway", "pilot_b"], "prompt": "人物来自 <Picture 3>"}
        previous_refs = _shot_identity_refs(first, project, assets)
        current_refs = _shot_identity_refs(second, project, assets)
        self.assertEqual(previous_refs, ["pilot_a"])
        self.assertEqual(current_refs, ["pilot_b"])
        self.assertEqual(_continuity_subject_relation(previous_refs, current_refs), "new")
        _strengthen_incoming_identity_reference(second, current_refs)
        self.assertEqual(second["continuity_identity_picture_tag"], "<Picture 2>")
        self.assertEqual(second["ref_image_roles"], {"pilot_b": "fully_preserved"})

    def test_gallery_only_image_does_not_become_continuity_identity(self):
        project = {"reference_image_numbers": {"pilot": 1}, "characters": [], "shots": []}
        assets = {"pilot": {"name": "人物四视图"}}
        shot = {"mode": "r2v", "prompt": "A person walks forward.",
                "ref_images": ["pilot"], "manual_ref_images": ["pilot"]}
        self.assertEqual(_shot_identity_refs(shot, project, assets), [])

    def test_explicit_person_tag_overrides_extra_selected_characters(self):
        project = {"id": "project", "shared_refs": {"images": []},
                   "reference_image_numbers": {"pilot_a": 1, "pilot_b": 2},
                   "characters": [{"id": "a", "name": "A", "multiview_asset": "pilot_a"},
                                  {"id": "b", "name": "B", "multiview_asset": "pilot_b"}],
                   "shots": []}
        assets = {"pilot_a": {"name": "A 四视图", "kind": "image"},
                  "pilot_b": {"name": "B 四视图", "kind": "image"}}
        first = {"id": "shot_a", "mode": "r2v", "prompt": "Pilot from <Picture 1> in cockpit",
                 "ref_images": ["pilot_a"], "character_ids": ["a", "b"]}
        second = {"id": "shot_b", "mode": "r2v", "prompt": "Pilot from <Picture 2> in next cockpit",
                  "ref_images": ["pilot_b"], "character_ids": []}
        project["shots"] = [first, second]
        with patch("runner.read", return_value={"assets": assets}):
            prepared = prepare_shot(first, project)
        self.assertEqual(prepared["character_ids"], ["a"])
        self.assertEqual(prepared["ref_images"], ["pilot_a"])
        self.assertEqual(_shot_identity_refs(first, project, assets), ["pilot_a"])
        self.assertEqual(_continuity_subject_relation(
            _shot_identity_refs(first, project, assets),
            _shot_identity_refs(second, project, assets)), "new")

    def test_identity_sheet_background_is_not_a_scene_reference(self):
        prompt = compile_shot_prompt({
            "mode": "r2v", "prompt": "Pilot inside the cockpit over <Picture 1>; use <Picture 2> for face.",
            "ref_images": ["runway", "pilot"],
            "_h3_reference_names": {"runway": "runway", "pilot": "Pilot four-view character"},
            "_h3_reference_descriptions": {"pilot": "人物四视图，正面和侧面"},
        })
        self.assertIn("<Picture 1> is a visual reference", prompt)
        self.assertIn("<Picture 2> is the identity reference", prompt)
        self.assertIn("Use it for the identity role requested in the detailed shot description", prompt)

    def test_short_one_take_warns_when_closeup_is_delayed_until_after_wide_action(self):
        delayed = lint_shot_prompt({"prompt": "The airplane crosses the runway, then the camera resolves into a tight close-up of the pilot.",
                                    "duration": 5, "continuous_take": True})
        immediate = lint_shot_prompt({"prompt": "A tight frontal close-up of the pilot fills the frame from the first instant.",
                                      "duration": 5, "continuous_take": True})
        self.assertIn("late_closeup_risk", [item["code"] for item in delayed])
        self.assertNotIn("late_closeup_risk", [item["code"] for item in immediate])

    def test_short_four_phase_motion_warns_about_duration(self):
        prompt = "A cyclist approaches, slows for a face close-up, then accelerates and leaves."
        short = lint_shot_prompt({"prompt": prompt, "duration": 5})
        longer = lint_shot_prompt({"prompt": prompt, "duration": 8})
        self.assertIn("dense_motion_beats", [item["code"] for item in short])
        self.assertNotIn("dense_motion_beats", [item["code"] for item in longer])

    def test_handoff_trim_changes_only_for_clear_visual_gain(self):
        previous = [bytes([100])] * 5 + [bytes([90]), bytes([70]), bytes([50]), bytes([20]), bytes([0])]
        current = [bytes([0])]
        with patch("runner._handoff_gray_frames", side_effect=[previous, current]):
            selected, before, after = _select_handoff_tail_trim(Path("before.mp4"), Path("after.mp4"), 4)
        self.assertEqual(selected, 0)
        self.assertGreater(before, after)
        nearly_equal = [bytes([100])] * 5 + [bytes([100]), bytes([101]), bytes([102]),
                                             bytes([103]), bytes([95])]
        with patch("runner._handoff_gray_frames", side_effect=[nearly_equal, current]):
            selected, _, _ = _select_handoff_tail_trim(Path("before.mp4"), Path("after.mp4"), 4)
        self.assertEqual(selected, 4)

    def test_h3_guide_uses_final_valid_tail_length(self):
        self.assertEqual([_continuity_anchor_frames(frames) for frames in (5, 8, 22, 39, 56)],
                         [5, 5, 22, 39, 56])

    def test_handoff_can_skip_an_unstable_generated_prefix(self):
        previous = [bytes([100])] * 6 + [bytes([40]), bytes([30]), bytes([20]), bytes([10]), bytes([0])]
        current = [bytes([80]), bytes([50]), bytes([2]), bytes([5]), bytes([6])]
        with patch("runner._handoff_gray_frames", side_effect=[previous, current]):
            tail, prefix, before, after = _select_handoff_alignment(
                Path("before.mp4"), Path("after.mp4"), 4)
        self.assertEqual((tail, prefix), (0, 2))
        self.assertLess(after, before)

    def test_motion_bridge_skips_a_different_scene(self):
        previous = [bytes([0]) * 4096] * 6
        current = [bytes([100]) * 4096] * 6
        with patch("runner._handoff_gray_frames", side_effect=[previous, current]), \
                patch("runner.subprocess.run") as ffmpeg:
            result = _smooth_handoff_motion(Path("before.mp4"), Path("after.mp4"), Path("bridge.mp4"))
        self.assertEqual(result, (None, None, 0))
        ffmpeg.assert_not_called()

    def test_exposure_match_skips_a_different_scene(self):
        previous = [bytes([0]) * 4096] * 6
        current = [bytes([100]) * 4096] * 6
        with patch("runner._handoff_gray_frames", side_effect=[previous, current]), \
                patch("runner.subprocess.run") as ffmpeg:
            result = _match_handoff_exposure(Path("before.mp4"), Path("after.mp4"), Path("matched.mp4"))
        self.assertEqual(result, (None, None, None))
        ffmpeg.assert_not_called()

    def test_structure_gate_accepts_same_view_and_rejects_scene_change(self):
        image = bytes((index * 37) % 256 for index in range(4096))
        same_view = bytes(min(255, value + 3) for value in image)
        different_view = image[::-1]
        self.assertTrue(_stable_matching_view([image] * 4, [same_view] * 2))
        self.assertFalse(_stable_matching_view([image] * 4, [different_view] * 2))

    def test_conservative_join_does_not_adjust_exposure_or_interpolate(self):
        paths = [Path("first.mp4")]
        report = []
        with patch("runner._select_handoff_alignment", return_value=(0, 0, 5.0, 4.0)) as align, \
                patch("runner._match_handoff_exposure") as exposure, \
                patch("runner._smooth_handoff_motion") as motion:
            _append_merge_item(paths, report, Path("second.mp4"),
                               {"shot_id": "second", "handoff_visible_frames": 4},
                               1, Path("/tmp"), {"seam_mode": "conservative"})
        align.assert_called_once()
        self.assertEqual(align.call_args.args[-1], 0)
        exposure.assert_not_called()
        motion.assert_not_called()

    def test_continuity_uses_native_h3_guides_for_reference_and_creative_modes(self):
        for mode in ("t2v", "r2v"):
            graph = build_h3_graph(
                {"mode": mode, "prompt": "A traveler crosses a market.", "duration": 5,
                 "ref_images": ["ref"] if mode == "r2v" else [],
                 "continuity_video": "tail", "continuity_frames": 22},
                {"width": 480, "height": 864, "generation_speed": "turbo"},
                {"tail": "tail.mp4", "ref": "ref.png"}, "continuity_test",
            )
            nodes = list(graph.values())
            guide = next(node for node in nodes if node["class_type"] == "MiniMaxH3AddGuide")
            self.assertEqual(guide["inputs"]["frame_idx"], 0)
            frame_node = next(n for n in nodes if n["class_type"] == "GetVideoComponents")
            self.assertEqual(guide["inputs"]["image"], [str(nodes.index(frame_node) + 1), 0])
            conditioning_node = next((i, n) for i, n in enumerate(nodes)
                                     if n["class_type"] == ("MiniMaxH3ReferenceToVideo" if mode == "r2v"
                                                             else "MiniMaxH3ImageToVideo"))
            self.assertEqual(guide["inputs"]["latent"], [str(conditioning_node[0] + 1), 1])
            self.assertFalse(any(node["class_type"] in ("H3StudioLatentContinue", "H3StudioContinueRemask")
                                 for node in nodes))
            video_vae = next((i, n) for i, n in enumerate(nodes) if n["class_type"] == "VAELoader"
                             and "video" in n["inputs"].get("vae_name", ""))
            self.assertEqual(guide["inputs"]["vae"], [str(video_vae[0] + 1), 0])

    def test_reference_shot_can_anchor_its_handoff_frame(self):
        graph = build_h3_graph(
            {"mode": "r2v", "prompt": "A plane passes the camera.", "duration": 5,
             "ref_images": ["pilot"], "last_image": "empty_runway"},
            {"width": 480, "height": 832},
            {"pilot": "pilot.png", "empty_runway": "runway.png"}, "handoff_test",
        )
        guides = [node for node in graph.values() if node["class_type"] == "MiniMaxH3AddGuide"]
        self.assertEqual(len(guides), 1)
        self.assertEqual(guides[0]["inputs"]["frame_idx"], -1)

    def test_continuity_prompt_is_subject_and_scene_agnostic(self):
        prompt = compile_shot_prompt({
            "prompt": "A red paper boat drifts under a bridge.",
            "continuity_video": "tail", "continuity_strength": "strong", "continuous_take": True,
        })
        self.assertIn("one physically possible camera path", prompt)
        self.assertIn("preserve its identity, position, movement", prompt)
        self.assertIn("do not add one as a default transition", prompt)

    def test_new_character_handoff_follows_requested_action_without_forcing_a_wipe(self):
        prompt = compile_shot_prompt({
            "prompt": "Reveal the officer in the new reference image.",
            "continuity_video": "tail", "continuity_strength": "strong",
            "continuity_subject_relation": "new", "continuity_frames": 39,
            "continuity_visible_frames": 15, "duration": 5,
        })
        self.assertIn("incoming person MUST match <Picture 1>", prompt)
        self.assertIn("current shot's explicitly tagged person reference is the identity authority", prompt)
        self.assertIn("use an occlusion only if the described scene supports it", prompt.lower())
        self.assertNotIn("lens wipe", prompt)

    def test_identity_handoff_makes_current_reference_override_previous_identity(self):
        prompt = compile_shot_prompt({
            "prompt": "An officer in a dark coat walks forward.",
            "continuity_video": "tail", "continuity_strength": "strong",
            "continuity_subject_relation": "new", "continuity_frames": 39,
            "continuity_visible_frames": 15, "duration": 5,
            "continuity_identity_picture_tag": "<Picture 2>",
        })
        self.assertLess(prompt.index("NEW PERSON ENTERS:"),
                        prompt.index("An officer in a dark coat walks forward."))
        self.assertIn("explicitly tagged person reference is the identity authority", prompt)
        self.assertIn("Keep the outgoing person unchanged wherever visible", prompt)

    def test_fast_action_and_occlusion_get_longer_exact_overlap(self):
        policy = _continuity_policy({"continuous_take": True, "continuity_frames": 22},
                                    {"camera_speed": "快速"}, 1)
        self.assertEqual(policy["frames"], 39)
        self.assertEqual(policy["seam_redraw"], 0.0)
        wipe = _continuity_policy({"continuous_take": True, "continuity_frames": 22},
                                  {"prompt": "The coat passes the lens in an occlusion wipe."}, 1)
        self.assertEqual(wipe["frames"], 39)

    def test_new_identity_gets_short_context_and_visible_transition(self):
        self.assertEqual(_continuity_subject_relation(["role_a"], ["role_b"]), "new")
        self.assertEqual(_continuity_subject_relation(["role_a"], ["role_a"]), "same")
        self.assertEqual(_continuity_subject_relation(["a", "b"], ["c", "d"]), "new")
        self.assertEqual(_handoff_window("new", 73), (8, 4))
        self.assertEqual(_handoff_window("same", 39), (39, 15))

    def test_incoming_character_reference_is_promoted_even_after_shared_refs(self):
        shot = {"ref_images": ["shared_style", "new_character"],
                "ref_image_roles": {"shared_style": "weak_reference"}}
        _strengthen_incoming_identity_reference(shot, ["new_character"])
        self.assertEqual(shot["ref_image_roles"]["new_character"], "fully_preserved")
        self.assertEqual(shot["ref_image_roles"]["shared_style"], "weak_reference")
        self.assertEqual(shot["continuity_identity_picture_tag"], "<Picture 2>")

    def test_new_identity_keeps_short_segment_guide_and_visible_overlap(self):
        context, visible = _handoff_window("new", 56)
        self.assertEqual((context, visible), (8, 4))
        graph = build_h3_graph(
            {"mode": "r2v", "prompt": "The incoming pilot enters.", "duration": 5,
             "ref_images": ["new_person"], "continuity_video": "tail.mp4",
             "continuity_frames": context, "continuity_visible_frames": visible,
             "continuity_subject_relation": "new", "continuity_strength": "strong",
             "continuity_identity_picture_tag": "<Picture 1>"},
            {"width": 480, "height": 864, "generation_speed": "turbo"},
            {"tail.mp4": "tail.mp4", "new_person": "new_person.png"}, "new_identity_guide_test")
        self.assertTrue(any(node["class_type"] == "MiniMaxH3AddGuide" for node in graph.values()))
        prompt_node = next(node for node in graph.values()
                           if node["class_type"] == "MiniMaxH3ReferenceToVideo")
        self.assertIn("explicitly tagged person reference is the identity authority", prompt_node["inputs"]["prompt"])


if __name__ == "__main__":
    unittest.main()

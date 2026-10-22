import tempfile
import unittest
import json
import subprocess
from pathlib import Path
from unittest.mock import patch

from engine import compile_shot_prompt, lint_shot_prompt, h3_frames, build_h3_graph
from runner import (_append_merge_item, _tail_clip, _grid_duration, _handoff_window,
                    _continuity_anchor_frames, _trim_plan, _trim_downloaded_shot, _narrated_head_skip,
                    _views_are_stable,
                    _has_spoken_audio, _inline_voiceover, continuity_preview,
                    _preserve_spoken_seam, _requests_fresh_composition,
                    _select_voiced_overlap_skip, prepare_shot)


class VoiceoverHandoffTests(unittest.TestCase):
    def test_quoted_narration_moves_to_native_spoken_field(self):
        shot = {"id": "one", "name": "镜头", "mode": "t2v", "duration": 5,
                "prompt": "湖面缓缓推进。旁白“城市与自然在这里相遇。”",
                "dialogue_text": "", "dialogue_voiceover": False,
                "speaker_id": "S1", "character_ids": []}
        project = {"id": "project", "shots": [shot], "characters": [], "entities": [], "shared_refs": {}}
        with patch("runner.read", return_value={"assets": {}}):
            prepared = prepare_shot(shot, project)
        compiled = compile_shot_prompt(prepared)
        self.assertEqual(shot["dialogue_text"], "")
        self.assertEqual(prepared["dialogue_text"], "城市与自然在这里相遇。")
        self.assertTrue(prepared["dialogue_voiceover"])
        self.assertIn("<d>[Chinese] 城市与自然在这里相遇。</d>", compiled)
        self.assertNotIn("旁白“", compiled)

    def test_unquoted_narration_is_extracted_too(self):
        visual, narration = _inline_voiceover("湖面缓缓推进。旁白：城市与自然在这里相遇。")
        self.assertEqual(visual, "湖面缓缓推进。")
        self.assertEqual(narration, "城市与自然在这里相遇。")
        self.assertTrue(_has_spoken_audio({"prompt": "旁白：下一章开始。"}))

    def test_negated_narration_is_not_treated_as_a_line(self):
        prompt = "镜头推进。没有旁白：只有环境声。"
        visual, narration = _inline_voiceover(prompt)
        self.assertEqual(narration, "")
        self.assertEqual(visual, prompt)
        self.assertFalse(_has_spoken_audio({"prompt": prompt}))
        self.assertEqual(_inline_voiceover("无旁白：仅环境声。")[1], "")

    def test_empty_narration_placeholder_is_dropped(self):
        visual, narration = _inline_voiceover("镜头推进。旁白：无")
        self.assertEqual(narration, "")
        self.assertEqual(visual, "镜头推进。")

    def test_explicit_dialogue_is_not_overwritten(self):
        visual, narration = _inline_voiceover('画面转场。旁白"下一章开始。"')
        self.assertEqual(visual, "画面转场。")
        self.assertEqual(narration, "下一章开始。")
        self.assertTrue(_has_spoken_audio({"prompt": '旁白“下一章开始。”'}))

    def test_repeated_narration_is_extracted_once(self):
        prompt = '航拍园区。Narration: "开原，面向未来。"\n旁白“开原，面向未来。”'
        visual, narration = _inline_voiceover(prompt)
        self.assertEqual(narration, '开原，面向未来。')
        self.assertNotIn('开原，面向未来。', visual)

    def test_explicit_dialogue_wins_over_inline_narration(self):
        shot = {"id": "one", "name": "镜头", "mode": "t2v", "duration": 5,
                "prompt": '画面推进过程中旁白响起："旧稿。"',
                "dialogue_text": "正式旁白。", "dialogue_voiceover": True,
                "speaker_id": "S1", "character_ids": []}
        project = {"id": "project", "shots": [shot], "characters": [], "entities": [], "shared_refs": {}}
        with patch("runner.read", return_value={"assets": {}}):
            prepared = prepare_shot(shot, project)
        compiled = compile_shot_prompt(prepared)
        self.assertIn("<d>[Chinese] 正式旁白。</d>", compiled)
        self.assertNotIn("旧稿。", compiled)

    def test_narrated_seam_keeps_both_audio_tracks_intact(self):
        with tempfile.TemporaryDirectory() as directory:
            previous = Path(directory) / "a.mp4"
            current = Path(directory) / "b.mp4"
            paths = [previous]
            seams = []
            with patch("runner._select_handoff_alignment", side_effect=AssertionError("must not trim")):
                _append_merge_item(paths, seams, current,
                                   {"shot_id": "second", "handoff_visible_frames": 9},
                                   1, Path(directory), {}, preserve_spoken_audio=True)
            self.assertEqual(paths, [previous, current])
            self.assertEqual(seams[0]["selected_trim_frames"], 0)
            self.assertTrue(seams[0]["audio_preserved"])

    def test_pasted_note_inside_the_visual_prompt_is_reported(self):
        shot = {"prompt": "镜头以低角度匀速我发现现在的旁白会因为分镜长短自己加速是因为提示词不合理跟拍。",
                "duration": 8, "dialogue_text": "开原，一座充满活力与希望的城市。",
                "dialogue_voiceover": True}
        codes = [finding["code"] for finding in lint_shot_prompt(shot)]
        self.assertIn("prompt_note_leak", codes)
        clean = dict(shot, prompt="镜头以低角度、近距离匀速跟拍，带轻微手持质感。")
        self.assertNotIn("prompt_note_leak", [f["code"] for f in lint_shot_prompt(clean)])

    def test_narrated_shots_warn_about_four_step_quality(self):
        shot = {"mode": "t2v", "prompt": "湖面推进。", "duration": 5,
                "dialogue_text": "开原，一座充满活力与希望的城市。", "dialogue_voiceover": True}
        codes = [f["code"] for f in lint_shot_prompt(shot, {"generation_speed": "turbo4"})]
        self.assertIn("narration_fast_steps", codes)
        reference = dict(shot, mode="r2v")
        self.assertIn("narration_fast_steps",
                      [f["code"] for f in lint_shot_prompt(reference, {"generation_speed": "turbo4"})])

    def test_overlong_narration_is_reported(self):
        findings = lint_shot_prompt({"prompt": "湖面缓缓推进，园区全景。", "duration": 5,
                                      "dialogue_text": "这是一个非常长的旁白句子，需要在五秒钟的画面中完整读出，显然来不及。"})
        self.assertIn("voiceover_too_long", [finding["code"] for finding in findings])

    def test_continuity_clip_carries_picture_without_previous_speech(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "spoken.mp4"
            guide = Path(directory) / "guide.mp4"
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                            "color=c=black:s=64x64:r=24", "-f", "lavfi", "-i",
                            "sine=frequency=440:sample_rate=48000", "-t", "1",
                            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
                            str(source)], check=True, timeout=30)
            _tail_clip(source, guide, 12)
            probe = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-of", "json",
                                    str(guide)], check=True, capture_output=True, text=True, timeout=30)
            kinds = {stream["codec_type"] for stream in json.loads(probe.stdout)["streams"]}
            self.assertEqual(kinds, {"video"})

    def test_timed_narration_leaves_overlap_free_for_visual_alignment(self):
        previous = {"audio_mode": "generate", "dialogue_text": "上一段", "duration": 5,
                    "dialogue_time": "0.8-4.0s"}
        current = {"audio_mode": "generate", "dialogue_text": "下一段", "duration": 5,
                   "dialogue_time": "0.8-4.0s"}
        self.assertFalse(_preserve_spoken_seam(previous, current, 9))
        current["dialogue_time"] = "0-4.0s"
        self.assertTrue(_preserve_spoken_seam(previous, current, 9))
        current["dialogue_time"] = ""
        self.assertTrue(_preserve_spoken_seam(previous, current, 9))

    def test_narrated_handoff_finds_replayed_guide_frame(self):
        previous = [bytes([20]), bytes([100])]
        current = [bytes([value]) for value in (10, 20, 30, 40, 50, 60, 70, 85, 100, 115, 125, 135)]
        with patch("runner._handoff_gray_frames", side_effect=[previous, current]):
            skip, before, after = _select_voiced_overlap_skip(Path("a"), Path("b"), 9)
        self.assertEqual(skip, 8)
        self.assertGreater(before, after)

    def test_new_aerial_setup_uses_transition_instead_of_forced_continuation(self):
        previous = {"prompt": "人物继续前行，最后定格于他的近景侧脸特写。"}
        aerial = {"prompt": "高空无人机广角航拍园区，从正面展示湖泊。"}
        continued = {"prompt": "接续上一镜末帧，无人机保持高度缓慢前行。"}
        self.assertTrue(_requests_fresh_composition(previous, aerial))
        self.assertFalse(_requests_fresh_composition(previous, continued))

    def test_continuation_reserves_the_guide_window_for_the_previous_take(self):
        shot = {"mode": "t2v", "duration": 5, "prompt": "人物继续前行。",
                "continuity_video": "guide", "continuity_frames": 22,
                "dialogue_text": "下一段旁白。", "dialogue_voiceover": True,
                "dialogue_time": "1-3s", "speaker_id": "S1"}
        compiled = compile_shot_prompt(shot)
        self.assertIn("[0s-0.916667s] The supplied opening frames continue", compiled)
        self.assertIn("[0.916667s-5.91667s] 人物继续前行。", compiled)
        # Authored times are relative to the visible shot, so they shift past the
        # replayed guide head instead of being cut by the prefix trim.
        self.assertIn("在 1.91667-3.91667s，", compiled)
        self.assertIn("<d>[Chinese] 下一段旁白。</d>", compiled)

    def test_continuation_without_authored_time_delays_speech_past_the_guide(self):
        shot = {"mode": "t2v", "duration": 5, "prompt": "人物继续前行。",
                "continuity_video": "guide", "continuity_frames": 22,
                "dialogue_text": "下一段旁白。", "dialogue_voiceover": True,
                "dialogue_time": "", "speaker_id": "S1"}
        self.assertIn("在约 0.916667s 之后，", compile_shot_prompt(shot))

    def test_standalone_shot_keeps_the_plain_shot_window(self):
        shot = {"mode": "t2v", "duration": 5, "prompt": "湖面缓缓推进。"}
        compiled = compile_shot_prompt(shot)
        self.assertIn("[Shot 1] [0s-5s] 湖面缓缓推进。", compiled)
        self.assertNotIn("opening frames continue", compiled)

    def test_continued_shot_keeps_every_generated_frame(self):
        # duration + guide must land on H3's 17k+5 grid, otherwise the padding is
        # generated and then trimmed, taking the tail of long narration with it.
        for duration in (2, 5, 6.5, 10, 15):
            for guide in (5, 22, 39, 56):
                effective = _grid_duration(duration, guide)
                generated = h3_frames(effective + guide / 24)
                _, visible = _handoff_window("same", guide)
                head_trim = round(guide / 24 * 24) - visible
                output_frames = round(effective * 24) + visible
                self.assertEqual(generated % 17, 5, (duration, guide))
                self.assertEqual(head_trim + output_frames, generated, (duration, guide))
                self.assertLessEqual(abs(effective - duration) * 24, 8.5, (duration, guide))

    def test_engine_requests_the_grid_aligned_length(self):
        shot = {"mode": "t2v", "duration": 5, "prompt": "人物继续前行。",
                "continuity_video": "guide", "continuity_frames": 22,
                "_h3_effective_duration": _grid_duration(5, 22)}
        graph = build_h3_graph(shot, {"width": 480, "height": 864, "generation_speed": "turbo"},
                               {"guide": "guide.mp4"}, "grid_length")
        length = next(node for node in graph.values()
                      if node["class_type"] == "MiniMaxH3ImageToVideo")["inputs"]["length"]
        self.assertEqual(length, 141)
        self.assertEqual(length % 17, 5)

    def test_quiet_guide_head_is_dropped_on_a_spoken_seam(self):
        with tempfile.TemporaryDirectory() as directory:
            previous = Path(directory) / "a.mp4"
            current = Path(directory) / "b.mp4"
            paths, seams = [previous], []
            with patch("runner._overlap_audio_peak", return_value=(True, -55.0)), \
                    patch("runner._narrated_head_skip", return_value=7), \
                    patch("runner._trim_prefix_frames") as trim, \
                    patch("runner._match_handoff_exposure", return_value=(None, None, None)):
                _append_merge_item(paths, seams, current,
                                   {"shot_id": "second", "handoff_visible_frames": 9},
                                   1, Path(directory), {}, preserve_spoken_audio=True)
            trim.assert_called_once_with(current, 7, trim.call_args.args[2])
            self.assertEqual(seams[0]["skipped_prefix_frames"], 7)
            self.assertEqual(seams[0]["selected_trim_frames"], 0)
            self.assertEqual(seams[0]["spoken_overlap"], "silent")
            self.assertTrue(seams[0]["audio_preserved"])

    def test_speech_in_the_guide_head_keeps_the_whole_clip(self):
        with tempfile.TemporaryDirectory() as directory:
            previous = Path(directory) / "a.mp4"
            current = Path(directory) / "b.mp4"
            paths, seams = [previous], []
            with patch("runner._overlap_audio_peak", return_value=(True, -12.0)), \
                    patch("runner._views_are_stable", return_value=False), \
                    patch("runner._dissolve_entry_keep_audio") as blend, \
                    patch("runner._trim_prefix_frames") as trim, \
                    patch("runner._match_handoff_exposure", return_value=(None, None, None)):
                _append_merge_item(paths, seams, current,
                                   {"shot_id": "second", "handoff_visible_frames": 9},
                                   1, Path(directory), {}, preserve_spoken_audio=True)
            trim.assert_not_called()
            blend.assert_not_called()
            self.assertEqual(paths, [previous, current])
            self.assertEqual(seams[0]["skipped_prefix_frames"], 0)
            self.assertEqual(seams[0]["spoken_overlap"], "speech")
            self.assertEqual(seams[0]["overlap_peak_dbfs"], -12.0)

    def test_speech_in_the_head_is_dissolved_without_touching_audio(self):
        with tempfile.TemporaryDirectory() as directory:
            previous = Path(directory) / "a.mp4"
            current = Path(directory) / "b.mp4"
            paths, seams = [previous], []

            def fake_dissolve(prev, cur, target, frames=12):
                Path(target).write_bytes(b"")
                fake_dissolve.frames = frames

            with patch("runner._overlap_audio_peak", return_value=(True, -5.0)), \
                    patch("runner._views_are_stable", return_value=True), \
                    patch("runner._dissolve_entry_keep_audio", side_effect=fake_dissolve) as blend, \
                    patch("runner._trim_prefix_frames") as trim, \
                    patch("runner._match_handoff_exposure", return_value=(None, None, None)):
                _append_merge_item(paths, seams, current,
                                   {"shot_id": "second", "handoff_visible_frames": 9},
                                   1, Path(directory), {}, preserve_spoken_audio=True)
            blend.assert_called_once()
            self.assertEqual(fake_dissolve.frames, 6)   # capped so the fade stays short
            trim.assert_not_called()
            self.assertEqual(seams[0]["transition_mode"], "dissolve")
            self.assertEqual(seams[0]["transition_frames"], 6)
            self.assertTrue(seams[0]["audio_preserved"])

    def test_view_gate_never_raises_on_unreadable_clips(self):
        with patch("runner._handoff_gray_frames", side_effect=ValueError("cannot read")):
            self.assertFalse(_views_are_stable(Path("a.mp4"), Path("b.mp4")))

    def test_unmeasurable_guide_head_is_left_untouched(self):
        with tempfile.TemporaryDirectory() as directory:
            previous = Path(directory) / "a.mp4"
            current = Path(directory) / "b.mp4"
            paths, seams = [previous], []
            with patch("runner._overlap_audio_peak", return_value=(True, None)), \
                    patch("runner._narrated_head_skip") as skip, \
                    patch("runner._trim_prefix_frames") as trim, \
                    patch("runner._match_handoff_exposure", return_value=(None, None, None)):
                _append_merge_item(paths, seams, current,
                                   {"shot_id": "second", "handoff_visible_frames": 9},
                                   1, Path(directory), {}, preserve_spoken_audio=True)
            trim.assert_not_called()
            skip.assert_not_called()
            self.assertEqual(paths, [previous, current])
            self.assertEqual(seams[0]["spoken_overlap"], "unknown")

    def test_h3_frames_does_not_pad_an_exact_grid_length(self):
        # 51/24 + 56/24 is 107 frames; float noise used to round it up to 124,
        # which silently added a 17-frame pad that the trim then discarded.
        self.assertEqual(h3_frames(107 / 24), 107)
        self.assertEqual(h3_frames(51 / 24 + 56 / 24), 107)
        self.assertEqual(h3_frames(120 / 24), 124)

    def test_prompt_preview_compiles_the_same_continuation_text(self):
        shots = [
            {"id": "a", "name": "镜1", "mode": "t2v", "prompt": "飞机沿跑道加速。", "duration": 5},
            {"id": "b", "name": "镜2", "mode": "t2v", "prompt": "飞机离地爬升。", "duration": 5,
             "dialogue_text": "它终于飞起来了。", "dialogue_voiceover": True},
        ]
        project = {"id": "p", "shots": shots, "characters": [], "entities": [], "shared_refs": {},
                   "continuous_take": True, "continuity_frames": 22}
        prepared = continuity_preview(shots[1], project, {})
        self.assertTrue(prepared["continuity"])
        # Narrated handoffs keep the shortest valid guide so no words can hide in
        # a long trimmed head.
        self.assertEqual(prepared["continuity_frames"], 5)
        self.assertEqual(prepared["continuity_visible_frames"], 4)
        compiled = compile_shot_prompt(prepared)
        self.assertIn("[0s-0.208333s] The supplied opening frames continue", compiled)
        self.assertIn("One-take continuity", compiled)
        self.assertIn("在约 0.208333s 之后，", compiled)
        silent_shot = dict(shots[1], dialogue_text="", dialogue_voiceover=False,
                           prompt="飞机离地爬升。")
        silent = continuity_preview(silent_shot, project, {})
        self.assertEqual((silent["continuity_frames"], silent["continuity_visible_frames"]), (22, 9))
        self.assertNotIn("在约", compile_shot_prompt(silent))

    def test_trim_plan_drops_a_silent_guide_head(self):
        shot = {"duration": 5, "continuity_frames": 8, "continuity_visible_frames": 4,
                "_h3_effective_duration": _grid_duration(5, 8),
                "dialogue_text": "旁白。", "dialogue_voiceover": True}
        with patch("runner.video_duration", return_value=124 / 24), \
                patch("runner._overlap_audio_peak", return_value=(True, -60.0)):
            plan = _trim_plan(Path("clip.mp4"), shot)
        self.assertFalse(plan["head_speech_preserved"])
        self.assertEqual(plan["start_frame"], 4)
        self.assertEqual(plan["output_frames"], 120)
        self.assertEqual(plan["total_frames"], 124)
        self.assertEqual(plan["start_frame"] + plan["output_frames"], plan["total_frames"])

    def test_trim_plan_keeps_a_guide_head_that_already_has_speech(self):
        shot = {"duration": 5, "continuity_frames": 8, "continuity_visible_frames": 4,
                "_h3_effective_duration": _grid_duration(5, 8),
                "dialogue_text": "旁白。", "dialogue_voiceover": True}
        with patch("runner.video_duration", return_value=124 / 24), \
                patch("runner._overlap_audio_peak", return_value=(True, -8.0)):
            plan = _trim_plan(Path("clip.mp4"), shot)
        self.assertTrue(plan["head_speech_preserved"])
        self.assertEqual(plan["start_frame"], 0)
        self.assertEqual(plan["output_frames"], 124)

    def test_trim_plan_ignores_levels_for_unnarrated_shots(self):
        shot = {"duration": 5, "continuity_frames": 8, "continuity_visible_frames": 4,
                "_h3_effective_duration": _grid_duration(5, 8), "prompt": "湖面推进。"}
        with patch("runner.video_duration", return_value=124 / 24), \
                patch("runner._overlap_audio_peak") as peak:
            plan = _trim_plan(Path("clip.mp4"), shot)
        peak.assert_not_called()
        self.assertEqual(plan["start_frame"], 4)
        self.assertFalse(plan["head_speech_preserved"])

    def test_narrated_shot_drops_only_the_silent_aligned_head(self):
        shot = {"duration": 5, "continuity_frames": 5, "continuity_visible_frames": 4,
                "dialogue_text": "旁白。", "dialogue_voiceover": True}

        def fake_trim(video, frames, target):
            Path(target).write_bytes(b"")

        with tempfile.TemporaryDirectory() as directory, \
                patch("runner.video_duration", return_value=124 / 24), \
                patch("runner._narrated_head_skip", return_value=3) as skip, \
                patch("runner._trim_prefix_frames", side_effect=fake_trim) as trim:
            plan = _trim_downloaded_shot(Path(directory) / "clip.mp4", shot,
                                         Path(directory) / "prev.mp4")
        skip.assert_called_once()
        self.assertEqual(trim.call_args.args[1], 3)
        self.assertEqual(plan["mode"], "silent_head")
        self.assertEqual(plan["start_frame"], 3)
        self.assertEqual(plan["output_frames"], 121)

    def test_narrated_head_skip_stops_at_the_first_speech_frame(self):
        with patch("runner._handoff_gray_frames", side_effect=[[b"\x10"], [b"\x00", b"\x05", b"\x12", b"\x30"]]), \
                patch("runner._overlap_audio_peak", side_effect=[(True, -90.0), (True, -70.0), (True, -12.0)]):
            self.assertEqual(_narrated_head_skip(Path("clip.mp4"), Path("prev.mp4"), 5), 2)

    def test_silent_shot_still_drops_the_guide_head(self):
        shot = {"duration": 5, "continuity_frames": 5, "continuity_visible_frames": 4,
                "prompt": "湖面推进。"}
        with patch("runner.video_duration", return_value=124 / 24), \
                patch("runner._retime_video_prefix_keep_audio") as retime, \
                patch("runner._trim_prefix") as trim:
            plan = _trim_downloaded_shot(Path("clip.mp4"), shot)
        retime.assert_not_called()
        trim.assert_called_once()
        self.assertEqual(plan["mode"], "trimmed")
        self.assertEqual(plan["start_frame"], 1)

    def test_grid_alignment_uses_the_handoff_window_not_the_guide_clip(self):
        # The real 彪投 run mixed these up: context 8 with the 5-frame guide clip
        # produced a 17-frame pad, and the trim then cut 0.58 s of narration.
        context, guide_clip = 8, _continuity_anchor_frames(8)
        self.assertEqual(guide_clip, 5)
        effective = _grid_duration(5, context)
        total = h3_frames(effective + context / 24)
        self.assertEqual(total, round(effective * 24) + context)
        self.assertEqual(total % 17, 5)

    def test_transition_preserves_audio_without_handoff_trim(self):
        with tempfile.TemporaryDirectory() as directory:
            previous = Path(directory) / "a.mp4"
            current = Path(directory) / "b.mp4"
            paths = [previous]
            seams = []
            with patch("runner._dissolve_entry_keep_audio") as dissolve:
                _append_merge_item(paths, seams, current,
                                   {"shot_id": "next", "transition_mode": "dissolve",
                                    "handoff_visible_frames": 9}, 1,
                                   Path(directory), {}, preserve_spoken_audio=True)
            dissolve.assert_called_once()
            self.assertEqual(seams[0]["transition_mode"], "dissolve")
            self.assertTrue(seams[0]["audio_preserved"])


if __name__ == "__main__":
    unittest.main()

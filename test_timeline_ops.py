import unittest

from timeline_ops import edit_timeline


def shot(shot_id, prompt, duration=5, **extra):
    return {"id": shot_id, "name": shot_id, "prompt": prompt, "duration": duration,
            "mode": "t2v", "seed": 42, **extra}


class TimelineEditTests(unittest.TestCase):
    def test_same_duration_changes_only_selected_shots(self):
        original = [shot("a", "one"), shot("b", "two"), shot("c", "three")]
        edited = edit_timeline(original, ["a", "c"], "same_duration", 7)
        self.assertEqual([item["duration"] for item in edited], [7, 5, 7])
        self.assertEqual([item["duration"] for item in original], [5, 5, 5])

    def test_merge_preserves_sequence_with_relative_times(self):
        original = [shot("a", "走进房间", 4), shot("b", "坐到窗边", 5), shot("c", "关上窗", 3)]
        edited = edit_timeline(original, ["a", "b"], "merge")
        self.assertEqual([item["id"] for item in edited], ["a", "c"])
        self.assertEqual(edited[0]["duration"], 9)
        self.assertIn("At 0-4s: 走进房间", edited[0]["prompt"])
        self.assertIn("At 4-9s: 坐到窗边", edited[0]["prompt"])

    def test_merge_rejects_non_adjacent_or_different_references(self):
        original = [shot("a", "one"), shot("b", "two"), shot("c", "three")]
        with self.assertRaisesRegex(ValueError, "连续"):
            edit_timeline(original, ["a", "c"], "merge")
        with self.assertRaisesRegex(ValueError, "素材或声音信息"):
            edit_timeline([shot("a", "one", first_image="img1"),
                           shot("b", "two", first_image="img2")], ["a", "b"], "merge")

    def test_merge_rejects_durations_over_h3_limit(self):
        with self.assertRaisesRegex(ValueError, "2 到 15"):
            edit_timeline([shot("a", "one", 8), shot("b", "two", 8)], ["a", "b"], "merge")

    def test_merge_rejects_run_selection_or_dialogue_conflicts(self):
        with self.assertRaisesRegex(ValueError, "selected"):
            edit_timeline([shot("a", "one", selected=True), shot("b", "two", selected=False)],
                          ["a", "b"], "merge")
        with self.assertRaisesRegex(ValueError, "dialogue_text"):
            edit_timeline([shot("a", "one", dialogue_text="你好"), shot("b", "two", dialogue_text="再见")],
                          ["a", "b"], "merge")


if __name__ == "__main__":
    unittest.main()

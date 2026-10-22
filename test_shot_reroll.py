import unittest

from runner import _reroll_result_list


class ShotRerollTests(unittest.TestCase):
    def test_only_target_segment_changes_and_source_stays_untouched(self):
        source = [
            {"shot_id": "first", "file": "old/first.mp4"},
            {"shot_id": "middle", "file": "old/middle.mp4"},
            {"shot_id": "last", "file": "old/last.mp4"},
        ]
        combined = _reroll_result_list(source, "middle",
                                       {"shot_id": "middle", "file": "new/middle.mp4"})
        self.assertEqual([item["file"] for item in combined],
                         ["old/first.mp4", "new/middle.mp4", "old/last.mp4"])
        self.assertEqual(source[1]["file"], "old/middle.mp4")

    def test_ambiguous_or_wrong_replacement_is_rejected(self):
        with self.assertRaises(ValueError):
            _reroll_result_list([{"shot_id": "one"}, {"shot_id": "one"}], "one",
                                {"shot_id": "one", "file": "new.mp4"})
        with self.assertRaises(ValueError):
            _reroll_result_list([{"shot_id": "one"}], "one",
                                {"shot_id": "other", "file": "new.mp4"})


if __name__ == "__main__":
    unittest.main()

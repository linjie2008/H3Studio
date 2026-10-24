"""Missing run-media references are dropped or cleared, not left dangling."""

import tempfile
import unittest
from pathlib import Path

import repair_missing_media_refs as repair


class RepairMissingMediaRefsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.outputs = Path(self.tmp.name)
        (self.outputs / "run-a").mkdir()
        (self.outputs / "run-a" / "shot_000_fit.mp4").write_bytes(b"x")

    def tearDown(self):
        self.tmp.cleanup()

    def test_existing_media_is_left_alone(self):
        run = {"results": [{"file": "run-a/shot_000_fit.mp4",
                            "h3_file": "run-a/shot_000_fit.mp4"}],
               "merged": "run-a/shot_000_fit.mp4"}
        self.assertEqual(repair.repair_run(run, self.outputs), [])
        self.assertNotIn("media_repairs", run)
        self.assertEqual(run["merged"], "run-a/shot_000_fit.mp4")

    def test_vanished_deliverable_drops_the_result(self):
        run = {"results": [
            {"shot_id": "s1", "file": "run-a/shot_000_fit.mp4"},
            {"shot_id": "s2", "file": "gone/shot_001_h3.mp4",
             "h3_file": "gone/shot_001_h3.mp4"},
        ]}
        removed = repair.repair_run(run, self.outputs)
        self.assertEqual(removed, ["gone/shot_001_h3.mp4"])
        self.assertEqual([result["shot_id"] for result in run["results"]], ["s1"])
        self.assertEqual(run["media_repairs"][0]["removed"], ["gone/shot_001_h3.mp4"])

    def test_dangling_secondary_reference_is_cleared_but_result_kept(self):
        run = {"results": [{"shot_id": "s1", "file": "run-a/shot_000_fit.mp4",
                            "h3_file": "gone/shot_000_h3.mp4",
                            "pre_refine_file": "gone/shot_000_pre.mp4"}]}
        removed = repair.repair_run(run, self.outputs)
        self.assertEqual(sorted(removed),
                         ["gone/shot_000_h3.mp4", "gone/shot_000_pre.mp4"])
        result = run["results"][0]
        self.assertIsNone(result["h3_file"])
        self.assertIsNone(result["pre_refine_file"])
        self.assertEqual(result["file"], "run-a/shot_000_fit.mp4")

    def test_missing_merged_video_is_cleared(self):
        run = {"results": [], "merged": "gone/merged.mp4"}
        self.assertEqual(repair.repair_run(run, self.outputs), ["gone/merged.mp4"])
        self.assertIsNone(run["merged"])

    def test_pending_repairs_and_apply_round_trip(self):
        import json
        import sqlite3

        database = self.outputs.parent / "h3studio.test.sqlite3"
        con = sqlite3.connect(database)
        con.execute("CREATE TABLE runs (id TEXT PRIMARY KEY, position INTEGER, data_json TEXT)")
        con.execute("INSERT INTO runs VALUES (?,?,?)", (
            "r1", 0, json.dumps({"id": "r1", "results": [
                {"shot_id": "s1", "file": "gone/shot_000_h3.mp4"}], "merged": None})))
        con.commit()
        con.close()

        repairs = repair.pending_repairs(database, self.outputs)
        self.assertIn("r1", repairs)
        repair.apply_repairs(repairs, database)

        with sqlite3.connect(database) as con:
            run = json.loads(con.execute(
                "SELECT data_json FROM runs WHERE id='r1'").fetchone()[0])
        self.assertEqual(run["results"], [])
        self.assertTrue(run["media_repairs"])
        database.unlink()


if __name__ == "__main__":
    unittest.main()

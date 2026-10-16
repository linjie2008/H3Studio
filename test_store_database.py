import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import store


class SQLiteStoreTests(unittest.TestCase):
    def test_normalizes_legacy_canvas_to_nearest_multiple_of_32(self):
        project = store.new_project("旧项目")
        project.update(width=1280, height=720)
        normalized = store._normalize({"projects": {project["id"]: project}, "assets": {}, "runs": {}})
        actual = normalized["projects"][project["id"]]
        self.assertEqual((actual["width"], actual["height"]), (1280, 704))
        self.assertEqual(store.align_dimension(1080, 576), 1088)

    def test_migrates_existing_json_and_persists_project_media_refs_and_runs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            legacy = {
                "projects": {"p1": {"id": "p1", "name": "测试剧本", "script_text": "保留的台词",
                                    "characters": [{"id": "c1", "name": "主角", "image": "a1"}]}},
                "assets": {"a1": {"id": "a1", "name": "主角参考图", "kind": "image", "file": "hero.png"}},
                "runs": {"r1": {"id": "r1", "project_id": "p1", "status": "completed", "prompt": "雾中列车"}},
            }
            state = root / "state.json"
            state.write_text(json.dumps(legacy, ensure_ascii=False), encoding="utf-8")
            with patch.object(store, "DATABASE", root / "h3studio.sqlite3"), patch.object(store, "STATE", state):
                loaded = store.read()
                self.assertEqual(loaded["projects"]["p1"]["script_text"], "保留的台词")
                self.assertEqual(loaded["projects"]["p1"]["characters"][0]["image"], "a1")
                self.assertEqual(loaded["assets"]["a1"]["file"], "hero.png")
                self.assertEqual(loaded["runs"]["r1"]["prompt"], "雾中列车")
                self.assertEqual(json.loads(state.read_text(encoding="utf-8")), legacy)

                store.change(lambda data: data["runs"].__setitem__("r2", {"id": "r2", "status": "failed"}))
                self.assertEqual(store.read()["runs"]["r2"]["status"], "failed")
                status = store.storage_status()
                self.assertEqual(status["backend"], "SQLite")
                self.assertEqual(status["counts"], {"projects": 1, "assets": 1, "runs": 2})

                # Updating one run must not delete and recreate unrelated
                # project rows.  This trigger would make the old full-table
                # rewrite fail immediately.
                with store._db() as con:
                    con.execute("CREATE TRIGGER reject_project_delete BEFORE DELETE ON projects "
                                "BEGIN SELECT RAISE(FAIL, 'unrelated project deleted'); END")
                store.change(lambda data: data["runs"]["r2"].update(status="completed"))
                self.assertEqual(store.read()["runs"]["r2"]["status"], "completed")


if __name__ == "__main__":
    unittest.main()

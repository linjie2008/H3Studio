import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import store
import app


class ProjectIsolationAndDeletionTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.db_path = self.root / "h3studio.sqlite3"
        self.state_path = self.root / "state.json"
        self.assets_dir = self.root / "assets"
        self.outputs_dir = self.root / "outputs"
        self.assets_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)

        self.patchers = [
            patch.object(store, "ROOT", self.root),
            patch.object(store, "DATABASE", self.db_path),
            patch.object(store, "STATE", self.state_path),
            patch.object(store, "ASSETS", self.assets_dir),
            patch.object(store, "OUTPUTS", self.outputs_dir),
            patch.object(app, "ASSETS", self.assets_dir),
            patch.object(app, "OUTPUTS", self.outputs_dir),
        ]
        for p in self.patchers:
            p.start()

    def tearDown(self):
        for p in reversed(self.patchers):
            p.stop()
        self.temp_dir.cleanup()

    def test_multi_project_asset_isolation(self):
        # Create Project A and Project B
        pa = store.new_project("项目甲")
        pa["id"] = "proj_a"
        pb = store.new_project("项目乙")
        pb["id"] = "proj_b"

        # Create assets
        asset_a = {"id": "asset_a_1", "name": "图A.png", "kind": "image", "file": "asset_a_1.png", "project_id": "proj_a"}
        asset_b = {"id": "asset_b_1", "name": "图B.png", "kind": "image", "file": "asset_b_1.png", "project_id": "proj_b"}

        store.change(lambda data: (
            data["projects"].__setitem__("proj_a", pa),
            data["projects"].__setitem__("proj_b", pb),
            data["assets"].__setitem__("asset_a_1", asset_a),
            data["assets"].__setitem__("asset_b_1", asset_b),
        ))

        state = store.read()
        self.assertIn("proj_a", state["projects"])
        self.assertIn("proj_b", state["projects"])

        # Filter by project_id for proj_a
        assets_a = [a for a in state["assets"].values() if a.get("project_id") == "proj_a"]
        assets_b = [a for a in state["assets"].values() if a.get("project_id") == "proj_b"]

        self.assertEqual(len(assets_a), 1)
        self.assertEqual(assets_a[0]["name"], "图A.png")
        self.assertEqual(len(assets_b), 1)
        self.assertEqual(assets_b[0]["name"], "图B.png")

    def test_project_video_isolation(self):
        pa = store.new_project("项目A")
        pa["id"] = "proj_a"
        pb = store.new_project("项目B")
        pb["id"] = "proj_b"

        run_a = {"id": "run_a_1", "project_id": "proj_a", "status": "completed", "results": [{"shot_id": "s1", "file": "run_a_1/clip1.mp4"}]}
        run_b = {"id": "run_b_1", "project_id": "proj_b", "status": "completed", "results": [{"shot_id": "s2", "file": "run_b_1/clip2.mp4"}]}

        store.change(lambda data: (
            data["projects"].__setitem__("proj_a", pa),
            data["projects"].__setitem__("proj_b", pb),
            data["runs"].__setitem__("run_a_1", run_a),
            data["runs"].__setitem__("run_b_1", run_b),
        ))

        state = store.read()
        runs_a = [r for r in state["runs"].values() if r.get("project_id") == "proj_a"]
        runs_b = [r for r in state["runs"].values() if r.get("project_id") == "proj_b"]
        self.assertEqual([r["id"] for r in runs_a], ["run_a_1"])
        self.assertEqual([r["id"] for r in runs_b], ["run_b_1"])

    def test_project_deletion_guards_against_active_runs(self):
        pa = store.new_project("活跃项目")
        pa["id"] = "proj_active"
        active_run = {"id": "run_active", "project_id": "proj_active", "status": "running"}

        store.change(lambda data: (
            data["projects"].__setitem__("proj_active", pa),
            data["runs"].__setitem__("run_active", active_run),
        ))

        state = store.read()
        has_active_runs = any(
            run.get("project_id") == "proj_active" and run.get("status") in ("queued", "running", "awaiting_upscale")
            for run in state["runs"].values()
        )
        self.assertTrue(has_active_runs)

    def test_project_deletion_removes_project_without_affecting_other_projects(self):
        pa = store.new_project("将被删除的项目")
        pa["id"] = "proj_to_delete"
        pb = store.new_project("保留的项目")
        pb["id"] = "proj_to_keep"

        store.change(lambda data: (
            data["projects"].__setitem__("proj_to_delete", pa),
            data["projects"].__setitem__("proj_to_keep", pb),
        ))

        # Perform deletion
        def remove(data):
            return data["projects"].pop("proj_to_delete", None)
        removed = store.change(remove)

        self.assertIsNotNone(removed)
        self.assertEqual(removed["id"], "proj_to_delete")

        state = store.read()
        self.assertNotIn("proj_to_delete", state["projects"])
        self.assertIn("proj_to_keep", state["projects"])
        self.assertEqual(state["projects"]["proj_to_keep"]["name"], "保留的项目")

    def test_batch_project_deletion_skips_projects_with_active_runs(self):
        idle = store.new_project("可删除")
        idle["id"] = "idle"
        active = store.new_project("生成中")
        active["id"] = "active"
        store.change(lambda data: (
            data["projects"].__setitem__("idle", idle),
            data["projects"].__setitem__("active", active),
            data["runs"].__setitem__("run", {"id": "run", "project_id": "active", "status": "running"}),
        ))

        result = app.delete_projects(["idle", "active", "missing", "idle"])

        self.assertEqual(result, {"deleted": ["idle"], "blocked": ["active"], "missing": ["missing"]})
        self.assertNotIn("idle", store.read()["projects"])
        self.assertIn("active", store.read()["projects"])


if __name__ == "__main__":
    unittest.main()

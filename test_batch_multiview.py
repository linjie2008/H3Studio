import unittest
from unittest.mock import Mock, patch

from app import generate_multiview_batch, make_multiview_job
from server import support
from server.routes import jobs as job_routes


class BatchMultiviewTests(unittest.TestCase):
    def setUp(self):
        self.state = {
            "projects": {"project": {"characters": [
                {"id": "alice", "name": "甲", "appearance": "灰色军装"},
                {"id": "bob", "name": "乙", "portrait_asset": "source"},
            ]}},
            "assets": {"source": {"project_id": "project", "kind": "image"}},
            "runs": {},
        }

    def test_snapshots_each_characters_mode_and_settings(self):
        first = make_multiview_job(self.state, "project", "alice", {
            "mode": "t2i", "engine": "xb_qwen", "view_layout": "five"})
        second = make_multiview_job(self.state, "project", "bob", {
            "mode": "i2i", "engine": "krea2", "view_layout": "four"})
        self.assertEqual((first["source_asset"], first["view_layout"]), (None, "five"))
        self.assertEqual((second["source_asset"], second["engine"]), ("source", "krea2"))
        self.assertEqual((second["xb_settings"]["krea_width"], second["xb_settings"]["krea_height"]),
                         (1536, 1024))
        self.assertNotEqual(first["id"], second["id"])

    def test_six_view_requires_its_workflow_and_source_image(self):
        job = make_multiview_job(self.state, "project", "bob", {
            "mode": "i2i", "engine": "qwen_sheet", "view_layout": "six"})
        self.assertEqual((job["engine"], job["view_layout"], job["source_asset"]),
                         ("qwen_sheet", "six", "source"))
        with self.assertRaisesRegex(ValueError, "六视图请选择"):
            make_multiview_job(self.state, "project", "bob", {
                "mode": "i2i", "engine": "xb_qwen", "view_layout": "six"})

    def test_rejects_invalid_character_before_queuing(self):
        with self.assertRaisesRegex(ValueError, "原图"):
            make_multiview_job(self.state, "project", "alice", {
                "mode": "i2i", "engine": "xb_qwen", "view_layout": "four"})
        with self.assertRaisesRegex(ValueError, "正在生成"):
            self.state["runs"]["old"] = {"kind": "character_multiview", "project_id": "project",
                                           "character_id": "bob", "status": "queued"}
            make_multiview_job(self.state, "project", "bob", {
                "mode": "i2i", "engine": "xb_qwen", "view_layout": "four"})

    def test_batch_runs_sequentially_and_skips_cancelled_job(self):
        state = {"runs": {"one": {}, "two": {"cancel_requested": True}, "three": {}}}
        # generate_multiview_batch 实现在 server.support，patch 目标随之搬家。
        with patch.object(support, "read", return_value=state), \
                patch.object(support, "change") as change, \
                patch.object(support, "generate_multiview") as generate:
            generate_multiview_batch(["one", "two", "three"])
        self.assertEqual([call.args[0] for call in generate.call_args_list], ["one", "three"])
        self.assertEqual(change.call_count, 1)

    def test_single_route_uses_shared_job_builder(self):
        request = Mock()
        request.body_json.return_value = {"mode": "i2i", "engine": "krea2",
                                          "view_layout": "four"}
        parts = ["api", "projects", "project", "characters", "bob", "multiview"]
        with patch.object(job_routes, "read", return_value=self.state), \
                patch.object(job_routes, "reserve_runs",
                             side_effect=lambda runs: self.state["runs"].update(
                                 {job["id"]: job for job in runs}) or None) as reserve, \
                patch.object(job_routes.threading, "Thread") as thread:
            job_routes.post_api_projects_character_multiview(request, "", parts, None)
        job = request.json.call_args.args[1]
        self.assertEqual(request.json.call_args.args[0], 202)
        self.assertEqual((job["character_id"], job["source_asset"], job["engine"]),
                         ("bob", "source", "krea2"))
        reserve.assert_called_once_with([job])
        self.assertIs(self.state["runs"][job["id"]], job)
        thread.assert_called_once_with(target=job_routes.generate_multiview,
                                       args=(job["id"],), daemon=True)
        thread.return_value.start.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()

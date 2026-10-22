import unittest
from unittest.mock import patch

import runner
from server.routes import projects


class SplitExportTests(unittest.TestCase):
    def test_merge_uses_only_selected_segments_in_timeline_order(self):
        shots = [{"id": "first", "source_video": "source"},
                 {"id": "middle", "source_video": "source"},
                 {"id": "last", "source_video": "source"}]
        results = [{"shot_id": "last", "file": "last.mp4"},
                   {"shot_id": "first", "file": "first.mp4"},
                   {"shot_id": "middle", "file": "old-middle.mp4"}]
        selected = runner._selected_results_for_merge(shots, {"first", "last"}, results)
        self.assertEqual([(shot["id"], item["file"]) for shot, item in selected],
                         [("first", "first.mp4"), ("last", "last.mp4")])

    def test_recut_replaces_old_segments_even_without_client_flag(self):
        asset_id = "source"
        state = {"assets": {asset_id: {"kind": "video", "project_id": "project",
                                        "file": "source.mp4", "name": "原片"}},
                 "projects": {"project": {"shots": [
                     {"id": "intro", "mode": "t2v"},
                     {"id": "old-1", "mode": "v2v", "source_video": asset_id},
                     {"id": "old-2", "mode": "v2v", "source_video": asset_id},
                     {"id": "outro", "mode": "t2v"}]}}}

        class Request:
            def __init__(self):
                self.response = None

            def body_json(self):
                return {"asset_id": asset_id, "method": "equal", "count": 2}

            def json(self, status, body):
                self.response = body

        counter = iter(range(10))
        with patch.object(projects, "read", return_value=state), \
             patch.object(projects, "change", side_effect=lambda fn: fn(state)), \
             patch.object(projects, "boundaries", return_value=[(0, 3), (3, 3)]), \
             patch.object(projects, "new_shot", side_effect=lambda: {"id": f"new-{next(counter)}"}):
            for _ in range(2):
                request = Request()
                projects.post_api_projects_split(request, "", ["api", "projects", "project", "split"], None)
                self.assertEqual(len(request.response), 2)
                self.assertEqual([shot["id"] for shot in state["projects"]["project"]["shots"]],
                                 ["intro", *[shot["id"] for shot in request.response], "outro"])


if __name__ == "__main__":
    unittest.main()

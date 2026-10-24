"""Workflow lookup survives ComfyUI sidebar folder renames."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import multiview


SIX_VIEW_NAME = "qwen-image2.1_人物定装照四视图_六视图.json"


class WorkflowPathResolutionTests(unittest.TestCase):
    def _write(self, root: Path, relative: str, payload: dict | None = None) -> Path:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload or {}, ensure_ascii=False), encoding="utf-8")
        return path

    def test_exact_candidate_wins_over_other_copies(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            exact = self._write(root, "Qwen-Image/2.1/" + SIX_VIEW_NAME)
            self._write(root, "elsewhere/" + SIX_VIEW_NAME)
            self.assertEqual(
                multiview._resolve_workflow(["Qwen-Image/2.1/" + SIX_VIEW_NAME], root), exact)

    def test_renamed_category_folder_is_still_found(self):
        # The real machine renamed "Qwen-Image" to "2 Qwen-Image"; exact
        # candidates miss, so the lookup has to fall back to a name search.
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            renamed = self._write(root, "2 Qwen-Image/2.1/" + SIX_VIEW_NAME)
            found = multiview._resolve_workflow(multiview._SIX_VIEW_RELATIVE, root)
            self.assertEqual(found, renamed)

    def test_search_prefers_the_shallowest_copy(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            shallow = self._write(root, "Qwen-Image/" + SIX_VIEW_NAME)
            self._write(root, "archive/2025/old/" + SIX_VIEW_NAME)
            self.assertEqual(
                multiview._resolve_workflow([SIX_VIEW_NAME], root), shallow)

    def test_missing_workflow_and_missing_tree_return_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self.assertIsNone(multiview._resolve_workflow([SIX_VIEW_NAME], root))
            self.assertIsNone(
                multiview._resolve_workflow([SIX_VIEW_NAME], root / "does-not-exist"))

    def test_xb_defaults_give_up_quietly_without_the_workflow(self):
        with patch.object(multiview, "_resolve_workflow", return_value=None):
            self.assertEqual(multiview._xb_i2i_workflow_defaults(), {})

    def test_six_view_builder_reports_the_path_it_could_not_read(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            missing = Path(tmpdir) / "missing-six-view.json"
            with patch.object(multiview, "SIX_VIEW_WORKFLOW", missing), patch.object(
                    multiview, "_resolve_workflow", return_value=None), patch.object(
                    multiview, "api", return_value={}):
                with self.assertRaisesRegex(RuntimeError, "无法读取本机六视图定装照工作流"):
                    multiview.build_six_view_sheet_graph("portrait.png", "job-1")


if __name__ == "__main__":
    unittest.main()

"""ComfyUI prompt submissions must request sampling frames for the workbench."""

import unittest
from unittest.mock import patch

import runner


class LivePreviewSubmitTests(unittest.TestCase):
    def test_submit_enables_per_prompt_preview_without_server_restart(self):
        graph = {"1": {"class_type": "SamplerCustomAdvanced", "inputs": {}}}
        family = runner._graph_model_family(graph)
        with patch.object(runner, "_ACTIVE_MODEL_FAMILY", family), patch.object(
            runner, "api", return_value={"prompt_id": "preview-test"}
        ) as api, patch.object(runner, "update_run") as update_run:
            self.assertEqual(runner._submit(graph, "run123"), "preview-test")
        payload = api.call_args.args[1]
        self.assertEqual(payload["extra_data"]["preview_method"], "auto")
        self.assertEqual(payload["client_id"], "h3_studio_run123")
        update_run.assert_called_with("run123", preview_client_id="h3_studio_run123")


if __name__ == "__main__":
    unittest.main()

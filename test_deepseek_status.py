import io
import json
import unittest
import urllib.error
from unittest.mock import patch

import app
from server import support


class DeepseekStatusTests(unittest.TestCase):
    """函数实现在 server.support；app 只做再导出，因此 patch 目标是 support。"""

    def setUp(self):
        # 云端状态徽章读的是模块级的“最近一次真实推理”记录；别的用例写过
        # 它，这里必须复位，否则本模块的结果取决于用例执行顺序。
        from prompt_tools import _completion_status
        from prompt_tools import _completion_status_lock
        with _completion_status_lock:
            _completion_status.clear()
            _completion_status.update({"ok": None, "checked_at": 0.0, "message": ""})

    def test_unconfigured_key_does_not_contact_provider(self):
        with patch.object(support, "deepseek_key", return_value=""), \
             patch.object(support.urllib.request, "urlopen") as urlopen:
            self.assertEqual(app.deepseek_connection_status(force=True)["state"], "unconfigured")
        urlopen.assert_not_called()

    def test_model_and_balance_are_checked_without_completion(self):
        responses = [io.BytesIO(json.dumps({"data": [{"id": app.DEEPSEEK_MODEL}]}).encode()),
                     io.BytesIO(b'{"is_available": true}')]
        with patch.object(support, "deepseek_key", return_value="test-secret"), \
             patch.object(support, "deepseek_key_source", return_value="saved"), \
             patch.object(support.urllib.request, "urlopen", side_effect=responses) as urlopen:
            status = app.deepseek_connection_status(force=True)
        self.assertEqual(status["state"], "unverified")
        self.assertEqual(urlopen.call_count, 2)
        self.assertEqual([call.args[0].full_url for call in urlopen.call_args_list],
                         ["https://api.deepseek.com/models", "https://api.deepseek.com/user/balance"])

    def test_insufficient_balance_is_not_shown_as_connected(self):
        responses = [io.BytesIO(json.dumps({"data": [{"id": app.DEEPSEEK_MODEL}]}).encode()),
                     io.BytesIO(b'{"is_available": false}')]
        with patch.object(support, "deepseek_key", return_value="test-secret"), \
             patch.object(support.urllib.request, "urlopen", side_effect=responses):
            self.assertEqual(app.deepseek_connection_status(force=True)["state"],
                             "insufficient_balance")

    def test_network_error_is_reported_without_leaking_key(self):
        error = urllib.error.URLError("boom")
        with patch.object(support, "deepseek_key", return_value="test-secret"), \
             patch.object(support.urllib.request, "urlopen", side_effect=error):
            status = app.deepseek_connection_status(force=True)
        self.assertEqual(status["state"], "error")
        self.assertNotIn("test-secret", status["message"])

    def test_recent_completion_failure_overrides_reachable_models(self):
        reachable = {"state": "connected", "message": "接口可达", "checked_at": 1.0}
        with patch.object(support, "deepseek_last_completion", return_value={
                "ok": False, "checked_at": 100.0, "message": "连接 DeepSeek 超时"}), \
             patch.object(support.time, "time", return_value=101.0):
            status = app._with_completion_status(reachable)
        self.assertEqual(status["state"], "completion_error")
        self.assertIn("超时", status["message"])

    def test_old_failure_remains_last_known_result(self):
        reachable = {"state": "connected", "message": "接口可达", "checked_at": 1000.0}
        with patch.object(support, "deepseek_last_completion", return_value={
                "ok": False, "checked_at": 100.0, "message": "连接 DeepSeek 超时"}), \
             patch.object(support.time, "time", return_value=10000.0):
            status = app._with_completion_status(reachable)
        self.assertEqual(status["state"], "completion_error")
        self.assertEqual(status["checked_at"], 100.0)


if __name__ == "__main__":
    unittest.main()

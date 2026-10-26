"""MiniMax 作为提示词模型的回归测试。

覆盖：密钥存储、请求格式、备用顺序、JSON 模式的思考标签清理，
以及 /api/ai/minimax/* 的四个接口。
"""
import json
import os
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

import ai_config
import prompt_tools
from server.routes import ai as ai_routes


class Response:
    def __init__(self, payload=None):
        self.code = None
        self.body = None
        self.payload = payload or {}
        self.same_origin_request = lambda: True

    def json(self, code, body):
        self.code, self.body = code, body

    def send_error(self, code):
        self.code = code

    def body_json(self):
        return self.payload


class FakeHTTPResponse:
    """Minimal stand-in for the object urlopen() returns."""

    def __init__(self, payload):
        self.payload = payload

    def read(self):
        return json.dumps(self.payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


class MinimaxKeyStorageTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.patchers = [
            patch.object(ai_config, "CONFIG_DIR", Path(self.temp_dir.name)),
            patch.object(ai_config, "MINIMAX_KEY_FILE", Path(self.temp_dir.name) / "minimax_api_key"),
        ]
        for patcher in self.patchers:
            patcher.start()
        os.environ.pop("MINIMAX_API_KEY", None)

    def tearDown(self):
        for patcher in reversed(self.patchers):
            patcher.stop()
        os.environ.pop("MINIMAX_API_KEY", None)
        self.temp_dir.cleanup()

    def test_key_round_trip(self):
        ai_config.save_minimax_key("sk-minimax-0123456789abcdef")
        self.assertEqual(ai_config.minimax_key(), "sk-minimax-0123456789abcdef")
        self.assertEqual(ai_config.minimax_key_source(), "saved")

    def test_short_key_is_rejected(self):
        with self.assertRaises(ValueError):
            ai_config.save_minimax_key("too-short")

    def test_key_with_whitespace_is_rejected(self):
        with self.assertRaises(ValueError):
            ai_config.save_minimax_key("sk-minimax 0123456789abcdef")

    def test_environment_key_wins(self):
        os.environ["MINIMAX_API_KEY"] = "env-key-0123456789abcdef"
        self.assertEqual(ai_config.minimax_key(), "env-key-0123456789abcdef")
        self.assertEqual(ai_config.minimax_key_source(), "environment")

    def test_remove_clears_the_key(self):
        ai_config.save_minimax_key("sk-minimax-0123456789abcdef")
        ai_config.remove_saved_minimax_key()
        self.assertEqual(ai_config.minimax_key(), "")
        self.assertEqual(ai_config.minimax_key_source(), "none")

    def test_minimax_is_a_valid_provider_choice(self):
        self.assertIn("minimax", ai_config.PROMPT_PROVIDERS)


class MinimaxRequestTests(unittest.TestCase):
    def _capture(self, payload=None):
        captured = {}

        def fake_urlopen(request, timeout=None):
            captured["url"] = request.full_url
            captured["headers"] = dict(request.header_items())
            captured["body"] = json.loads(request.data.decode("utf-8"))
            return FakeHTTPResponse(payload or {"choices": [{"message": {"content": "OK"}}]})

        return captured, fake_urlopen

    def test_request_uses_bearer_auth_and_openai_shape(self):
        captured, fake = self._capture()
        with patch.object(prompt_tools, "minimax_key", return_value="sk-minimax-0123456789"), \
                patch.object(prompt_tools.urllib.request, "urlopen", side_effect=fake):
            content = prompt_tools._minimax_complete(
                [{"role": "user", "content": "hi"}], max_tokens=64, timeout=5)
        self.assertEqual(content, "OK")
        self.assertIn("chat/completions", captured["url"])
        self.assertEqual(captured["headers"].get("Authorization"), "Bearer sk-minimax-0123456789")
        self.assertEqual(captured["body"]["model"], prompt_tools.MINIMAX_MODEL)
        self.assertEqual(captured["body"]["max_tokens"], 64)

    def test_missing_key_raises_a_readable_error(self):
        with patch.object(prompt_tools, "minimax_key", return_value=""):
            with self.assertRaises(RuntimeError) as caught:
                prompt_tools._minimax_complete([{"role": "user", "content": "hi"}], None, 5)
        self.assertIn("MiniMax API Key", str(caught.exception))

    def test_thinking_block_is_removed(self):
        captured, fake = self._capture(
            {"choices": [{"message": {"content": "<think>reasoning</think>final answer"}}]})
        with patch.object(prompt_tools, "minimax_key", return_value="sk-minimax-0123456789"), \
                patch.object(prompt_tools.urllib.request, "urlopen", side_effect=fake):
            content = prompt_tools._minimax_complete(
                [{"role": "user", "content": "hi"}], None, 5)
        self.assertEqual(content, "final answer")

    def test_non_json_answer_in_json_mode_is_retried(self):
        answers = [{"choices": [{"message": {"content": "抱歉，这是说明文字。"}}]},
                   {"choices": [{"message": {"content": '{"prompts":[{"id":"a","prompt":"b"}]}'}}]}]
        calls = []

        def fake_urlopen(request, timeout=None):
            calls.append(json.loads(request.data.decode("utf-8")))
            return FakeHTTPResponse(answers[min(len(calls) - 1, len(answers) - 1)])

        with patch.object(prompt_tools, "minimax_key", return_value="sk-minimax-0123456789"), \
                patch.object(prompt_tools.urllib.request, "urlopen", side_effect=fake_urlopen):
            content = prompt_tools._minimax_complete(
                [{"role": "user", "content": "hi"}], None, 5, json_mode=True)
        self.assertEqual(json.loads(content)["prompts"][0]["id"], "a")
        self.assertEqual(len(calls), 2)
        # 第二次必须带上「只要 JSON」的明确指令。
        self.assertIn("JSON", str(calls[1]["messages"][-1]["content"]))

    def test_http_error_is_reported_with_the_status(self):
        def fake_urlopen(request, timeout=None):
            raise urllib.error.HTTPError(request.full_url, 401, "Unauthorized", {}, None)

        with patch.object(prompt_tools, "minimax_key", return_value="sk-minimax-0123456789"), \
                patch.object(prompt_tools.urllib.request, "urlopen", side_effect=fake_urlopen):
            with self.assertRaises(RuntimeError) as caught:
                prompt_tools._minimax_complete([{"role": "user", "content": "hi"}], None, 5)
        self.assertIn("无效", str(caught.exception))


class MinimaxProviderRoutingTests(unittest.TestCase):
    """The provider chain records completion status in a module-level dict.

    Restore it after every test so these cases cannot flip the cloud-status
    badge another test module reads.
    """

    def setUp(self):
        self._status_snapshot = dict(prompt_tools._completion_status)

    def tearDown(self):
        with prompt_tools._completion_status_lock:
            prompt_tools._completion_status.clear()
            prompt_tools._completion_status.update(self._status_snapshot)

    def test_explicit_minimax_selection_uses_minimax(self):
        with patch.object(prompt_tools, "prompt_provider", return_value="minimax"), \
                patch.object(prompt_tools, "minimax_key", return_value="sk-minimax-0123456789"), \
                patch.object(prompt_tools, "_minimax_complete", return_value="from-minimax") as minimax, \
                patch.object(prompt_tools, "_deepseek_complete_request",
                             side_effect=AssertionError("DeepSeek 不该被调用")):
            result = prompt_tools._deepseek_complete([{"role": "user", "content": "hi"}], None, 5)
        self.assertEqual(result, "from-minimax")
        self.assertEqual(minimax.call_count, 1)

    def test_auto_falls_back_to_minimax_after_the_others(self):
        with patch.object(prompt_tools, "prompt_provider", return_value="auto"), \
                patch.object(prompt_tools, "deepseek_key", return_value=""), \
                patch.object(prompt_tools, "mimo_key", return_value=""), \
                patch.object(prompt_tools, "minimax_key", return_value="sk-minimax-0123456789"), \
                patch.object(prompt_tools, "_deepseek_complete_request",
                             side_effect=RuntimeError("DeepSeek 挂了")), \
                patch.object(prompt_tools, "_minimax_complete", return_value="from-minimax") as minimax:
            result = prompt_tools._deepseek_complete([{"role": "user", "content": "hi"}], None, 5)
        self.assertEqual(result, "from-minimax")
        self.assertEqual(minimax.call_count, 1)

    def test_auto_keeps_mimo_ahead_of_minimax(self):
        with patch.object(prompt_tools, "prompt_provider", return_value="auto"), \
                patch.object(prompt_tools, "deepseek_key", return_value=""), \
                patch.object(prompt_tools, "mimo_key", return_value="sk-mimo-0123456789"), \
                patch.object(prompt_tools, "minimax_key", return_value="sk-minimax-0123456789"), \
                patch.object(prompt_tools, "_deepseek_complete_request",
                             side_effect=RuntimeError("DeepSeek 挂了")), \
                patch.object(prompt_tools, "_mimo_complete", return_value="from-mimo"), \
                patch.object(prompt_tools, "_minimax_complete",
                             side_effect=AssertionError("不该走到 MiniMax")) as minimax:
            result = prompt_tools._deepseek_complete([{"role": "user", "content": "hi"}], None, 5)
        self.assertEqual(result, "from-mimo")
        self.assertEqual(minimax.call_count, 0)

    def test_all_providers_failing_reports_every_reason(self):
        with patch.object(prompt_tools, "prompt_provider", return_value="auto"), \
                patch.object(prompt_tools, "deepseek_key", return_value=""), \
                patch.object(prompt_tools, "mimo_key", return_value="sk-mimo-0123456789"), \
                patch.object(prompt_tools, "minimax_key", return_value="sk-minimax-0123456789"), \
                patch.object(prompt_tools, "_deepseek_complete_request",
                             side_effect=RuntimeError("DeepSeek 挂了")), \
                patch.object(prompt_tools, "_mimo_complete", side_effect=RuntimeError("MiMo 挂了")), \
                patch.object(prompt_tools, "_minimax_complete", side_effect=RuntimeError("MiniMax 挂了")):
            with self.assertRaises(RuntimeError) as caught:
                prompt_tools._deepseek_complete([{"role": "user", "content": "hi"}], None, 5)
        message = str(caught.exception)
        for fragment in ("DeepSeek", "小米 MiMo", "MiniMax", "挂了"):
            self.assertIn(fragment, message)


class MinimaxRouteTests(unittest.TestCase):
    def setUp(self):
        self.response = Response()

    def _post(self, path, body):
        self.response = Response()
        ai_routes.post_api_ai_minimax_settings(self.response, path, ["api", "ai", "minimax", "settings"], None)
        return self.response

    def test_settings_route_reports_the_model(self):
        with patch.object(ai_routes, "minimax_key_source", return_value="saved"):
            ai_routes.get_api_ai_minimax_settings(
                self.response, "/api/ai/minimax/settings", ["api", "ai", "minimax", "settings"], None)
        self.assertEqual(self.response.code, 200)
        self.assertEqual(self.response.body["provider"], "MiniMax")
        self.assertEqual(self.response.body["model"], prompt_tools.MINIMAX_MODEL)
        self.assertTrue(self.response.body["configured"])

    def test_test_route_requires_same_origin(self):
        request = Response()
        request.same_origin_request = lambda: False
        with patch.object(ai_routes, "_minimax_complete",
                          side_effect=AssertionError("不该发起请求")):
            ai_routes.post_api_ai_minimax_test(
                request, "/api/ai/minimax/test", ["api", "ai", "minimax", "test"], None)
        self.assertEqual(request.code, 403)

    def test_test_route_calls_minimax_when_same_origin(self):
        request = Response()
        request.same_origin_request = lambda: True
        with patch.object(ai_routes, "_minimax_complete", return_value="OK") as complete:
            ai_routes.post_api_ai_minimax_test(
                request, "/api/ai/minimax/test", ["api", "ai", "minimax", "test"], None)
        self.assertEqual(request.code, 200)
        self.assertEqual(request.body["provider"], "MiniMax")
        self.assertEqual(complete.call_count, 1)

    def test_all_minimax_routes_are_registered(self):
        names = {(route.method, route.name) for route in ai_routes.ROUTES}
        self.assertIn(("GET", "get_api_ai_minimax_settings"), names)
        self.assertIn(("POST", "post_api_ai_minimax_test"), names)
        self.assertIn(("POST", "post_api_ai_minimax_settings"), names)
        self.assertIn(("DELETE", "delete_api_ai_minimax_settings"), names)

    def test_provider_options_include_minimax(self):
        ai_routes.get_api_ai_provider(self.response, "/api/ai/provider", ["api", "ai", "provider"], None)
        self.assertIn("minimax", self.response.body["options"])

    def test_settings_route_lists_minimax_as_extra_fallback(self):
        with patch.object(ai_routes, "deepseek_key_source", return_value="saved"), \
                patch.object(ai_routes, "mimo_key_source", return_value="saved"), \
                patch.object(ai_routes, "minimax_key_source", return_value="saved"):
            ai_routes.get_api_ai_settings(self.response, "/api/ai/settings", ["api", "ai", "settings"], None)
        self.assertEqual(self.response.body["extra_fallback"]["provider"], "MiniMax")


if __name__ == "__main__":
    unittest.main()


class ProviderModelChoiceTests(unittest.TestCase):
    """用户自己选择每个提供商使用的模型。"""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.patchers = [
            patch.object(ai_config, "CONFIG_DIR", Path(self.temp_dir.name)),
            patch.object(ai_config, "PROMPT_MODELS_FILE", Path(self.temp_dir.name) / "prompt_models"),
        ]
        for patcher in self.patchers:
            patcher.start()
        for name in ("H3_STUDIO_DEEPSEEK_MODEL", "H3_STUDIO_MIMO_MODEL", "H3_STUDIO_MINIMAX_MODEL"):
            os.environ.pop(name, None)

    def tearDown(self):
        for patcher in reversed(self.patchers):
            patcher.stop()
        for name in ("H3_STUDIO_DEEPSEEK_MODEL", "H3_STUDIO_MIMO_MODEL", "H3_STUDIO_MINIMAX_MODEL"):
            os.environ.pop(name, None)
        self.temp_dir.cleanup()

    def test_default_model_is_the_builtin_one(self):
        self.assertEqual(ai_config.provider_model("minimax"), "MiniMax-M3")
        self.assertEqual(ai_config.provider_model_source("minimax"), "default")

    def test_saved_choice_wins(self):
        ai_config.save_provider_model("minimax", "MiniMax-M2.7")
        self.assertEqual(ai_config.provider_model("minimax"), "MiniMax-M2.7")
        self.assertEqual(ai_config.provider_model_source("minimax"), "saved")

    def test_environment_variable_is_used_when_nothing_is_saved(self):
        os.environ["H3_STUDIO_MINIMAX_MODEL"] = "MiniMax-M2.5"
        self.assertEqual(ai_config.provider_model("minimax"), "MiniMax-M2.5")
        self.assertEqual(ai_config.provider_model_source("minimax"), "environment")

    def test_saved_choice_overrides_environment_variable(self):
        os.environ["H3_STUDIO_MINIMAX_MODEL"] = "MiniMax-M2.5"
        ai_config.save_provider_model("minimax", "MiniMax-M2.7")
        self.assertEqual(ai_config.provider_model("minimax"), "MiniMax-M2.7")

    def test_each_provider_has_its_own_choice(self):
        ai_config.save_provider_model("deepseek", "deepseek-v4-pro")
        ai_config.save_provider_model("minimax", "MiniMax-M2.7")
        self.assertEqual(ai_config.provider_model("deepseek"), "deepseek-v4-pro")
        self.assertEqual(ai_config.provider_model("minimax"), "MiniMax-M2.7")
        self.assertEqual(ai_config.provider_model("mimo"), "mimo-v2.6-pro")

    def test_arbitrary_model_name_is_allowed(self):
        ai_config.save_provider_model("minimax", "MiniMax-M2.7-highspeed")
        self.assertEqual(ai_config.provider_model("minimax"), "MiniMax-M2.7-highspeed")

    def test_dangerous_model_name_is_rejected(self):
        for bad in ("has space", "rm;rf", "a" * 100, ""):
            with self.assertRaises(ValueError):
                ai_config.save_provider_model("minimax", bad)

    def test_unknown_provider_is_rejected(self):
        with self.assertRaises(ValueError):
            ai_config.save_provider_model("nope", "whatever")

    def test_request_uses_the_saved_model(self):
        ai_config.save_provider_model("minimax", "MiniMax-M2.7")
        captured = {}

        def fake_urlopen(request, timeout=None):
            captured["body"] = json.loads(request.data.decode("utf-8"))
            return FakeHTTPResponse({"choices": [{"message": {"content": "OK"}}]})

        with patch.object(prompt_tools, "minimax_key", return_value="sk-minimax-0123456789"), \
                patch.object(prompt_tools.urllib.request, "urlopen", side_effect=fake_urlopen):
            prompt_tools._minimax_complete([{"role": "user", "content": "hi"}], None, 5)
        self.assertEqual(captured["body"]["model"], "MiniMax-M2.7")

    def test_models_route_lists_every_provider(self):
        response = Response()
        ai_routes.get_api_ai_models(response, "/api/ai/models", ["api", "ai", "models"], None)
        self.assertEqual(response.code, 200)
        self.assertEqual(set(response.body["providers"]), {"deepseek", "mimo", "minimax"})
        self.assertIn("MiniMax-M3", response.body["providers"]["minimax"]["catalog"])

    def test_models_route_rejects_cross_site_requests(self):
        response = Response({"provider": "minimax", "model": "MiniMax-M2.7"})
        response.same_origin_request = lambda: False
        ai_routes.post_api_ai_models(response, "/api/ai/models", ["api", "ai", "models"], None)
        self.assertEqual(response.code, 403)

    def test_models_route_saves_the_choice(self):
        response = Response({"provider": "minimax", "model": "MiniMax-M2.7"})
        ai_routes.post_api_ai_models(response, "/api/ai/models", ["api", "ai", "models"], None)
        self.assertEqual(response.code, 200)
        self.assertEqual(response.body["model"], "MiniMax-M2.7")
        self.assertEqual(ai_config.provider_model("minimax"), "MiniMax-M2.7")

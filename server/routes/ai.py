"""DeepSeek / 小米 MiMo 配置、连接测试与提示词翻译。"""
from __future__ import annotations

from server.router import Route

from ai_config import deepseek_key_source
from ai_config import mimo_key_source
from ai_config import prompt_provider
from ai_config import remove_saved_deepseek_key
from ai_config import remove_saved_mimo_key
from ai_config import save_deepseek_key
from ai_config import save_mimo_key
from ai_config import save_prompt_provider
from prompt_tools import DEEPSEEK_MODEL
from prompt_tools import MIMO_MODEL
from prompt_tools import _deepseek_complete_request
from prompt_tools import _mimo_complete
from prompt_tools import stream_translate_prompt
from prompt_tools import translate_prompt_with_backend
from server.support import deepseek_connection_status
import json
import urllib.parse
import urllib.request
import urllib.error

def get_api_ai_settings(req, path, parts, url):
    """原 app.py 分支：`path == "/api/ai/settings"`。"""
    source = deepseek_key_source()
    req.json(200, {"provider": "DeepSeek", "model": DEEPSEEK_MODEL,
                    "configured": source != "none", "key_source": source,
                    "selected_provider": prompt_provider(),
                    "fallback": {"provider": "小米 MiMo", "model": MIMO_MODEL,
                                 "configured": mimo_key_source() != "none",
                                 "key_source": mimo_key_source()}})
    return


def get_api_ai_provider(req, path, parts, url):
    """原 app.py 分支：`path == "/api/ai/provider"`。"""
    selected = prompt_provider()
    req.json(200, {"selected_provider": selected,
                    "options": ["auto", "deepseek", "mimo"]})
    return


def get_api_ai_mimo_settings(req, path, parts, url):
    """原 app.py 分支：`path == "/api/ai/mimo/settings"`。"""
    source = mimo_key_source()
    req.json(200, {"provider": "小米 MiMo", "model": MIMO_MODEL,
                    "configured": source != "none", "key_source": source})
    return


def get_api_ai_status(req, path, parts, url):
    """原 app.py 分支：`path == "/api/ai/status"`。"""
    query = urllib.parse.parse_qs(url.query)
    req.json(200, deepseek_connection_status(force=query.get("refresh") == ["1"]))
    return


def post_api_translate_stream(req, path, parts, url):
    """原 app.py 分支：`path == "/api/translate-prompt-stream"`。"""
    if not req.same_origin_request():
        req.json(403, {"error": "请从工作台调用提示词翻译"})
        return
    body = req.body_json()
    req.send_response(200)
    req.send_header("Content-Type", "text/event-stream; charset=utf-8")
    req.send_header("Cache-Control", "no-cache, no-transform")
    req.send_header("X-Accel-Buffering", "no")
    req.send_header("Connection", "keep-alive")
    req.end_headers()
    try:
        for chunk in stream_translate_prompt(str(body.get("prompt") or "")):
            req.wfile.write(("data: " + json.dumps({"text": chunk}, ensure_ascii=False) + "\n\n").encode("utf-8"))
            req.wfile.flush()
        req.wfile.write(b"data: {\"done\":true}\n\n")
        req.wfile.flush()
    except Exception as error:
        req.wfile.write(("data: " + json.dumps({"error": str(error)}, ensure_ascii=False) + "\n\n").encode("utf-8"))
        req.wfile.flush()
    return


def post_api_translate(req, path, parts, url):
    """原 app.py 分支：`path == "/api/translate-prompt"`。"""
    if not req.same_origin_request():
        req.json(403, {"error": "请从工作台调用提示词翻译"})
        return
    body = req.body_json()
    translation, backend = translate_prompt_with_backend(
        str(body.get("prompt") or ""), str(body.get("quality") or "auto"),
        str(body.get("target_language") or "zh"))
    req.json(200, {"translation": translation, "backend": backend})
    return


def post_api_ai_test(req, path, parts, url):
    """原 app.py 分支：`path == "/api/ai/test"`。"""
    if not req.same_origin_request():
        req.json(403, {"error": "请从工作台页面测试云端 AI"})
        return
    # This button verifies DeepSeek itself. Normal optimization may
    # still transparently use the configured MiMo fallback.
    _deepseek_complete_request(
        [{"role": "system", "content": "Reply with exactly OK."},
         {"role": "user", "content": "OK"}],
        max_tokens=128, timeout=35)
    req.json(200, {"ok": True, "provider": "DeepSeek", "model": DEEPSEEK_MODEL,
                    "message": "连接成功，API Key 与模型均可用。"})
    return


def post_api_ai_mimo_test(req, path, parts, url):
    """原 app.py 分支：`path == "/api/ai/mimo/test"`。"""
    if not req.same_origin_request():
        req.json(403, {"error": "请从工作台页面测试云端 AI"})
        return
    _mimo_complete(
        [{"role": "system", "content": "Reply with exactly OK."},
         {"role": "user", "content": "OK"}],
        max_tokens=128, timeout=35)
    req.json(200, {"ok": True, "provider": "小米 MiMo", "model": MIMO_MODEL,
                    "message": "连接成功，API Key 与模型均可用。"})
    return


def post_api_ai_settings(req, path, parts, url):
    """原 app.py 分支：`path == "/api/ai/settings"`。"""
    if not req.same_origin_request():
        req.json(403, {"error": "请从工作台页面配置云端 AI"})
        return
    body = req.body_json()
    save_deepseek_key(body.get("api_key", ""))
    source = deepseek_key_source()
    req.json(200, {"provider": "DeepSeek", "model": DEEPSEEK_MODEL,
                    "configured": source != "none", "key_source": source})
    return


def post_api_ai_provider(req, path, parts, url):
    """原 app.py 分支：`path == "/api/ai/provider"`。"""
    if not req.same_origin_request():
        req.json(403, {"error": "请从工作台页面配置云端 AI"})
        return
    selected = save_prompt_provider(req.body_json().get("provider"))
    req.json(200, {"selected_provider": selected})
    return


def post_api_ai_mimo_settings(req, path, parts, url):
    """原 app.py 分支：`path == "/api/ai/mimo/settings"`。"""
    if not req.same_origin_request():
        req.json(403, {"error": "请从工作台页面配置云端 AI"})
        return
    body = req.body_json()
    save_mimo_key(body.get("api_key", ""))
    source = mimo_key_source()
    req.json(200, {"provider": "小米 MiMo", "model": MIMO_MODEL,
                    "configured": source != "none", "key_source": source})
    return


def delete_api_ai_settings(req, path, parts, url):
    """原 app.py 分支：`parts == ["api", "ai", "settings"]`。"""
    if not req.same_origin_request():
        req.json(403, {"error": "请从工作台页面管理云端 AI 配置"})
        return
    remove_saved_deepseek_key()
    source = deepseek_key_source()
    req.json(200, {"provider": "DeepSeek", "model": DEEPSEEK_MODEL,
                    "configured": source != "none", "key_source": source})
    return


def delete_api_ai_mimo_settings(req, path, parts, url):
    """原 app.py 分支：`parts == ["api", "ai", "mimo", "settings"]`。"""
    remove_saved_mimo_key()
    source = mimo_key_source()
    req.json(200, {"provider": "小米 MiMo", "model": MIMO_MODEL,
                    "configured": source != "none", "key_source": source})
    return


ROUTES = [
    Route('GET', 'get_api_ai_settings',
         lambda path, parts, url: (path == "/api/ai/settings"),
         get_api_ai_settings),
    Route('GET', 'get_api_ai_provider',
         lambda path, parts, url: (path == "/api/ai/provider"),
         get_api_ai_provider),
    Route('GET', 'get_api_ai_mimo_settings',
         lambda path, parts, url: (path == "/api/ai/mimo/settings"),
         get_api_ai_mimo_settings),
    Route('GET', 'get_api_ai_status',
         lambda path, parts, url: (path == "/api/ai/status"),
         get_api_ai_status),
    Route('POST', 'post_api_translate_stream',
         lambda path, parts, url: (path == "/api/translate-prompt-stream"),
         post_api_translate_stream),
    Route('POST', 'post_api_translate',
         lambda path, parts, url: (path == "/api/translate-prompt"),
         post_api_translate),
    Route('POST', 'post_api_ai_test',
         lambda path, parts, url: (path == "/api/ai/test"),
         post_api_ai_test),
    Route('POST', 'post_api_ai_mimo_test',
         lambda path, parts, url: (path == "/api/ai/mimo/test"),
         post_api_ai_mimo_test),
    Route('POST', 'post_api_ai_settings',
         lambda path, parts, url: (path == "/api/ai/settings"),
         post_api_ai_settings),
    Route('POST', 'post_api_ai_provider',
         lambda path, parts, url: (path == "/api/ai/provider"),
         post_api_ai_provider),
    Route('POST', 'post_api_ai_mimo_settings',
         lambda path, parts, url: (path == "/api/ai/mimo/settings"),
         post_api_ai_mimo_settings),
    Route('DELETE', 'delete_api_ai_settings',
         lambda path, parts, url: (parts == ["api", "ai", "settings"]),
         delete_api_ai_settings),
    Route('DELETE', 'delete_api_ai_mimo_settings',
         lambda path, parts, url: (parts == ["api", "ai", "mimo", "settings"]),
         delete_api_ai_mimo_settings),
]

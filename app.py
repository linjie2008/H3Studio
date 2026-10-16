"""Standalone local director application; ComfyUI supplies inference only.

HTTP 路由已拆分到 server/routes/（顺序见 server/routes/__init__.py 的 ORDER）；
本文件只保留 Handler 基础方法、请求分发与启动逻辑。
"""

from __future__ import annotations

import base64
import hashlib
import json
import mimetypes
import os
import re
import select
import socket
import ssl
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from engine import ALL_MODES, build_h3_graph, compile_shot_prompt, lint_shot_prompt
from packs import export_director_pack, export_pack, import_pack
from prompt_tools import (enhance, enhance_sequence, storyboard as generate_storyboard,
                          suggest_project_parameters, recommend_project_skills,
                          _deepseek_complete, _deepseek_complete_request)
from prompt_skills import DEFAULT_SKILL, delete_skill, import_skill, list_skills, preview_skill
from runner import (COMFY_URL, api, prepare_shot, run_project, remerge_run,
                    _prompt_state, _shot_identity_refs)
from store import (ASSETS, OUTPUTS, align_dimension, change, clone_project, new_project,
                   new_shot, project_asset_ids, read, scope_legacy_assets, storage_status, uid)
from video_tools import boundaries, extract_audio_track
from timeline_ops import edit_timeline
from ai_config import (deepseek_key, deepseek_key_source, remove_saved_deepseek_key,
                       save_deepseek_key, mimo_key, mimo_key_source, remove_saved_mimo_key,
                       save_mimo_key, prompt_provider, save_prompt_provider)
from prompt_tools import (DEEPSEEK_MODEL, stream_translate_prompt, translate_prompt,
                          translate_prompt_with_backend, deepseek_last_completion,
                          MIMO_MODEL, _mimo_complete)
from reference_numbering import normalized_project_image_references, renumber_project_image_references
from multiview import generate as generate_multiview
from profile_images import generate as generate_profile_image
from image_prompt_tools import optimize_image_prompt
from product_agent_actions import ACTION_NAMES as PRODUCT_AGENT_ACTIONS, confirm as confirm_product_agent_action

# 兼容再导出：测试与既有脚本从 app 命名空间引用这些助手（实现在 server/support.py）。
from server.config import COMFY_DIR, EXTENSIONS, HOST, PORT, ROOT
from server.routes import ROUTES
from server.router import resolve
from server.support import (assets_with_character_names, deepseek_connection_status,
                            delete_projects, enhancement_references,
                            generate_multiview_batch, make_multiview_job,
                            parse_byte_range, _with_completion_status)


class Handler(BaseHTTPRequestHandler):
    def proxy_comfy_websocket(self, query: str) -> None:
        """Relay the browser's same-origin WebSocket to ComfyUI."""
        if not self.same_origin_request():
            self.send_error(403, "WebSocket must come from this workbench")
            return
        client_key = self.headers.get("Sec-WebSocket-Key", "")
        if not client_key or self.headers.get("Upgrade", "").lower() != "websocket":
            self.send_error(400, "Invalid WebSocket handshake")
            return
        parsed = urllib.parse.urlsplit(COMFY_URL)
        secure = parsed.scheme == "https"
        port = parsed.port or (443 if secure else 80)
        client_id = urllib.parse.parse_qs(query).get("clientId", ["h3_studio"])[0]
        client_id = re.sub(r"[^a-zA-Z0-9_.-]", "", client_id)[:100] or "h3_studio"
        upstream = None
        upgraded = False
        try:
            upstream = socket.create_connection((parsed.hostname, port), timeout=8)
            if secure:
                upstream = ssl.create_default_context().wrap_socket(upstream, server_hostname=parsed.hostname)
            upstream_key = base64.b64encode(os.urandom(16)).decode("ascii")
            target = f"/ws?clientId={urllib.parse.quote(client_id)}"
            upstream.sendall((
                f"GET {target} HTTP/1.1\r\nHost: {parsed.netloc}\r\n"
                "Upgrade: websocket\r\nConnection: Upgrade\r\n"
                f"Sec-WebSocket-Key: {upstream_key}\r\nSec-WebSocket-Version: 13\r\n\r\n"
            ).encode("ascii"))
            response = bytearray()
            while b"\r\n\r\n" not in response and len(response) < 65536:
                chunk = upstream.recv(4096)
                if not chunk:
                    raise ConnectionError("ComfyUI closed the WebSocket handshake")
                response.extend(chunk)
            header_end = response.find(b"\r\n\r\n")
            if header_end < 0 or not response.startswith(b"HTTP/1.1 101"):
                raise ConnectionError("ComfyUI rejected the WebSocket connection")
            accept = base64.b64encode(hashlib.sha1(
                (client_key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode("ascii")
            ).digest()).decode("ascii")
            self.connection.sendall((
                "HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\n"
                f"Connection: Upgrade\r\nSec-WebSocket-Accept: {accept}\r\n\r\n"
            ).encode("ascii"))
            upgraded = True
            upstream_extra = response[header_end + 4:]
            if upstream_extra:
                self.connection.sendall(upstream_extra)
            self.connection.setblocking(True)
            upstream.setblocking(True)
            while True:
                readable, _, _ = select.select([self.connection, upstream], [], [], 60)
                if not readable:
                    continue
                for source in readable:
                    data = source.recv(65536)
                    if not data:
                        return
                    (upstream if source is self.connection else self.connection).sendall(data)
        except (OSError, ValueError, ConnectionError) as error:
            if not upgraded:
                try:
                    self.send_error(502, f"ComfyUI WebSocket unavailable: {error}")
                except OSError:
                    pass
        finally:
            if upstream:
                upstream.close()

    def json(self, status: int, data) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def body_json(self) -> dict:
        size = int(self.headers.get("Content-Length", "0"))
        if size < 1 or size > 2_000_000:
            raise ValueError("项目数据大小无效")
        data = json.loads(self.rfile.read(size))
        if not isinstance(data, dict):
            raise ValueError("请求必须为对象")
        return data

    def same_origin_request(self) -> bool:
        host = self.headers.get("Host", "")
        return bool(host and self.headers.get("Origin") in (f"http://{host}", f"https://{host}"))

    def file(self, path: Path) -> None:
        if not path.is_file():
            self.send_error(404)
            return
        size = path.stat().st_size
        try:
            requested_range = parse_byte_range(self.headers.get("Range", ""), size)
        except ValueError:
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        partial = requested_range is not None
        start, end = requested_range if partial else (0, size - 1)
        self.send_response(206 if partial else 200)
        self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(end - start + 1))
        if path.name == "index.html":
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
            self.send_header("Pragma", "no-cache")
        self.send_header("Accept-Ranges", "bytes")
        if partial:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with path.open("rb") as source:
            source.seek(start)
            remaining = end - start + 1
            try:
                while remaining and (block := source.read(min(1024 * 1024, remaining))):
                    self.wfile.write(block)
                    remaining -= len(block)
            except (BrokenPipeError, ConnectionResetError):
                # Browsers routinely abandon range requests while scrubbing or
                # switching videos.  That is not a server error.
                return

    def dispatch(self, method: str) -> None:
        """按注册表顺序查找路由并执行；未命中返回 404。"""
        route, path, parts, url = resolve(ROUTES, method, self.path)
        if route is None:
            self.send_error(404)
            return
        route.handler(self, path, parts, url)

    def do_GET(self) -> None:
        self.dispatch("GET")

    def do_POST(self) -> None:
        if not self.same_origin_request():
            self.json(403, {"error": "请从当前 H3 Studio 工作台执行修改操作"})
            return
        try:
            self.dispatch("POST")
        except (ValueError, KeyError, TypeError, zipfile.BadZipFile) as error:
            self.json(400, {"error": str(error)})
        except (OSError, urllib.error.URLError, RuntimeError, subprocess.SubprocessError) as error:
            path = urllib.parse.urlsplit(self.path).path
            if path in ("/api/enhance", "/api/enhance-sequence"):
                self.log_error("Prompt optimization failed (%s): %s", path, error)
            self.json(502, {"error": str(error)})

    def do_PUT(self) -> None:
        if not self.same_origin_request():
            self.json(403, {"error": "请从当前 H3 Studio 工作台执行修改操作"})
            return
        try:
            self.dispatch("PUT")
        except (ValueError, KeyError, TypeError) as error:
            self.json(400, {"error": str(error)})

    def do_DELETE(self) -> None:
        if not self.same_origin_request():
            self.json(403, {"error": "请从当前 H3 Studio 工作台执行修改操作"})
            return
        self.dispatch("DELETE")


if __name__ == "__main__":
    scope_legacy_assets()
    active_runs = [run for run in read()["runs"].values()
                   if run.get("status") in ("queued", "running", "awaiting_upscale")]
    recoverable = []
    for run in active_runs:
        prompt_state = _prompt_state(str(run.get("prompt_id") or ""))
        if (run.get("status") == "awaiting_upscale" or
                (run.get("active_stage") and prompt_state in ("active", "completed"))):
            recoverable.append(run["id"])
            continue
        def interrupt(data, run_id=run["id"], state=prompt_state):
            data["runs"][run_id].update(
                status="interrupted",
                error=f"软件已重启；ComfyUI 任务状态为 {state}，请点击续跑",
            )
        change(interrupt)
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"H3 Studio: http://127.0.0.1:{PORT} → {COMFY_URL}", flush=True)
    for run_id in recoverable:
        threading.Thread(target=run_project, args=(run_id,), daemon=True).start()
    server.serve_forever()

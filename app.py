"""Standalone local director application; ComfyUI supplies inference only."""

from __future__ import annotations

import json
import mimetypes
import os
import re
import base64
import hashlib
import select
import socket
import ssl
import subprocess
import threading
import time
import tempfile
import urllib.parse
import urllib.error
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
from store import (ASSETS, OUTPUTS, align_dimension, change, clone_project, new_project, new_shot, project_asset_ids,
                   read, scope_legacy_assets, storage_status, uid)
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
from product_agent import chat as product_agent_chat
from product_agent_actions import ACTION_NAMES as PRODUCT_AGENT_ACTIONS, confirm as confirm_product_agent_action


ROOT = Path(__file__).resolve().parent
COMFY_DIR = Path(os.environ.get("COMFYUI_DIR", "/home/administrator/ComfyUI"))
HOST = os.environ.get("H3_STUDIO_HOST", "127.0.0.1")
PORT = int(os.environ.get("H3_STUDIO_PORT", "8765"))
_deepseek_status_lock = threading.Lock()
_deepseek_status_cache = {"checked_at": 0.0, "key": "", "result": None}


def _with_completion_status(provider_status: dict) -> dict:
    result = dict(provider_status)
    last_completion = deepseek_last_completion()
    if result["state"] == "connected" and last_completion["ok"] is False:
        result["state"] = "completion_error"
        result["message"] = "最近一次真实推理失败：" + last_completion["message"]
        result["checked_at"] = last_completion["checked_at"]
    elif result["state"] == "connected" and last_completion["ok"] is None:
        result["state"] = "unverified"
        result["message"] += "；真实推理尚未验证"
    elif result["state"] == "connected":
        result["message"] = "最近一次真实推理成功；接口、模型和余额当前可达"
        result["checked_at"] = last_completion["checked_at"]
    return result


def deepseek_connection_status(force: bool = False) -> dict:
    """Check authentication and model availability without a paid completion."""
    source = deepseek_key_source()
    key = deepseek_key()
    if not key:
        return {"state": "unconfigured", "message": "未配置 API Key", "checked_at": time.time()}
    key_id = hashlib.sha256(key.encode("utf-8")).hexdigest()
    with _deepseek_status_lock:
        cached = _deepseek_status_cache
        if (not force and cached["key"] == key_id and cached["result"] is not None
                and time.time() - cached["checked_at"] < 45):
            return _with_completion_status(cached["result"])
    request = urllib.request.Request("https://api.deepseek.com/models",
                                     headers={"Authorization": "Bearer " + key})
    try:
        with urllib.request.urlopen(request, timeout=8) as response:
            models = json.load(response)
        available = any(item.get("id") == DEEPSEEK_MODEL for item in models.get("data", [])
                        if isinstance(item, dict))
        if available:
            balance_request = urllib.request.Request("https://api.deepseek.com/user/balance",
                                                     headers={"Authorization": "Bearer " + key})
            with urllib.request.urlopen(balance_request, timeout=8) as response:
                balance = json.load(response)
            available = balance.get("is_available") is True
            state = "connected" if available else "insufficient_balance"
            message = "接口和模型可达，账户有可用余额" if available else "账户余额不足，无法调用模型"
        else:
            state = "model_unavailable"
            message = "接口可达，但当前模型未列出"
    except urllib.error.HTTPError as error:
        state = "error"
        message = {401: "API Key 无效", 402: "账户余额不足", 403: "API Key 无权访问", 429: "请求过于频繁"}.get(
            error.code, f"DeepSeek 接口返回 HTTP {error.code}")
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as error:
        state = "error"
        message = "连接超时或网络不可达" if isinstance(error, (TimeoutError, urllib.error.URLError)) else "状态响应无效"
    result = {"state": state, "message": message, "model": DEEPSEEK_MODEL,
              "key_source": source, "checked_at": time.time()}
    with _deepseek_status_lock:
        _deepseek_status_cache.update(checked_at=result["checked_at"], key=key_id, result=result)
    return _with_completion_status(result)
EXTENSIONS = {
    ".png": "image", ".jpg": "image", ".jpeg": "image", ".webp": "image", ".bmp": "image", ".gif": "image", ".tif": "image", ".tiff": "image",
    ".mp4": "video", ".mov": "video", ".webm": "video", ".mkv": "video", ".avi": "video", ".m4v": "video",
    ".wav": "audio", ".mp3": "audio", ".flac": "audio", ".m4a": "audio", ".ogg": "audio", ".aac": "audio", ".wma": "audio",
}


def parse_byte_range(header: str, size: int) -> tuple[int, int] | None:
    """Return one inclusive HTTP byte range, including suffix ranges used by video players."""
    if not header.startswith("bytes="):
        return None
    spec = header[6:].strip()
    if size <= 0 or "," in spec or spec.count("-") != 1:
        raise ValueError("invalid byte range")
    raw_start, raw_end = spec.split("-", 1)
    if not raw_start:
        suffix = int(raw_end)
        if suffix <= 0:
            raise ValueError("invalid suffix range")
        return max(0, size - suffix), size - 1
    start = int(raw_start)
    end = min(int(raw_end), size - 1) if raw_end else size - 1
    if start < 0 or start >= size or end < start:
        raise ValueError("invalid byte range")
    return start, end


def assets_with_character_names(state: dict, project_id: str = "") -> list[dict]:
    """Return project assets annotated with current and historical character owners."""
    projects = state.get("projects", {})
    relevant_ids = [project_id] if project_id else list(projects)
    owners: dict[str, list[str]] = {}
    character_names: dict[tuple[str, str], str] = {}
    for current_id in relevant_ids:
        project = projects.get(current_id, {})
        for index, character in enumerate(project.get("characters", []), 1):
            character_id = str(character.get("id") or "")
            name = str(character.get("name") or f"角色 {index}").strip()
            if character_id:
                character_names[(current_id, character_id)] = name
            for key in ("portrait_asset", "multiview_asset", "voice_asset"):
                asset_id = str(character.get(key) or "")
                if asset_id and name not in owners.setdefault(asset_id, []):
                    owners[asset_id].append(name)
    # Keep names on older generated character sheets even after the character
    # switches to a newer bound result.
    for run in state.get("runs", {}).values():
        current_id = str(run.get("project_id") or "")
        if current_id not in relevant_ids or run.get("kind") != "character_multiview":
            continue
        asset_id = str(run.get("asset_id") or "")
        name = character_names.get((current_id, str(run.get("character_id") or "")), "")
        if asset_id and name and name not in owners.setdefault(asset_id, []):
            owners[asset_id].append(name)
    result = []
    for asset in state.get("assets", {}).values():
        asset_project = str(asset.get("project_id") or "")
        if (project_id and asset_project != project_id) or (not project_id and asset_project not in projects):
            continue
        item = dict(asset)
        item["character_names"] = owners.get(str(asset.get("id") or ""), [])
        result.append(item)
    return result


def enhancement_references(prepared: dict, project: dict, catalog: dict,
                           identity_assets: set[str]) -> list[dict]:
    """Describe only media inputs wired into the selected H3 generation mode."""
    image_numbers = project.get("reference_image_numbers") or {
        asset_id: index for index, (asset_id, asset) in enumerate(
            ((key, value) for key, value in catalog.items()
             if value.get("kind") == "image" and value.get("project_id") == project.get("id")), 1)}
    mode = str(prepared.get("mode") or "t2v")
    if mode in {"t2v", "i2v", "fl2v"}:
        frames = [("first_image", "first_frame")]
        if mode == "fl2v":
            frames.append(("last_image", "last_frame"))
        if mode == "t2v":
            frames = []
        result = []
        for key, role in frames:
            asset_id = prepared.get(key)
            if not asset_id:
                continue
            asset = catalog.get(asset_id, {})
            number = image_numbers.get(asset_id)
            result.append({"tag": f"<Picture {number}>" if number else "", "type": "image",
                           "asset_id": asset_id, "role": role,
                           "is_person_identity": asset_id in identity_assets,
                           "name": str(asset.get("name") or asset_id)[:160],
                           "description": str((project.get("image_descriptions") or {}).get(asset_id) or "")[:600]})
        return result
    references = []
    for kind, key, tag_name in (("image", "ref_images", "Picture"),
                                ("video", "ref_videos", "Video"),
                                ("audio", "ref_audios", "Audio")):
        ids = list(prepared.get(key) or [])
        if kind == "video" and prepared.get("source_video"):
            ids = [prepared["source_video"]] + [item for item in ids if item != prepared["source_video"]]
        role_key = {"image": "ref_image_roles", "video": "ref_video_roles", "audio": "ref_audio_roles"}[kind]
        for index, asset_id in enumerate(ids, 1):
            asset = catalog.get(asset_id, {})
            default_role = ("fully_preserved" if kind == "image" else
                            "源视频" if kind == "video" and asset_id == prepared.get("source_video") else "默认参考")
            role = (prepared.get(role_key) or {}).get(asset_id, default_role)
            tag_index = int(image_numbers.get(asset_id, index)) if kind == "image" else index
            references.append({"tag": f"<{tag_name} {tag_index}>", "type": kind,
                "asset_id": asset_id, "is_person_identity": asset_id in identity_assets,
                "name": str(asset.get("name") or asset_id)[:160], "role": str(role)[:80],
                "description": str((project.get("image_descriptions") or {}).get(asset_id) or "")[:600]
                    if kind == "image" else ""})
    return references


def make_multiview_job(state: dict, project_id: str, character_id: str, body: dict) -> dict:
    """Validate one character and snapshot its settings for single or batch generation."""
    preset = str(body.get("preset") or "fast")
    if preset not in ("fast", "stable"):
        raise ValueError("多视图生成模式无效")
    view_layout = str(body.get("view_layout") or "four")
    if view_layout not in ("four", "five", "six"):
        raise ValueError("角色多视图布局无效")
    multiview_engine = str(body.get("engine") or "qwen")
    if multiview_engine not in ("xb_qwen", "qwen", "krea2", "qwen_sheet"):
        raise ValueError("多视图生成引擎无效")
    if multiview_engine == "krea2" and view_layout != "four":
        raise ValueError("Krea2 辅助工作流只支持四视图")
    if multiview_engine == "qwen_sheet" and view_layout != "six":
        raise ValueError("定装照工作流只支持六视图")
    if multiview_engine not in ("qwen_sheet",) and view_layout == "six":
        raise ValueError("六视图请选择 Qwen 2.1 定装照工作流")
    project = state["projects"].get(project_id)
    character = next((item for item in project.get("characters", [])
                      if item.get("id") == character_id), None) if project else None
    if not character:
        raise ValueError("当前项目中找不到这个角色")
    source_id = character.get("portrait_asset")
    mode = str(body.get("mode") or "i2i")
    if mode not in ("t2i", "i2i"):
        raise ValueError("多视图生图模式无效")
    if multiview_engine in ("krea2", "qwen_sheet") and mode != "i2i":
        raise ValueError("当前定装照工作流只支持参考原图生成")
    if mode == "i2i":
        source = state["assets"].get(source_id)
        if not source or source.get("project_id") != project_id or source.get("kind") != "image":
            raise ValueError(f"{character.get('name') or '角色'}：图生图请先绑定当前项目的人物原图")
    else:
        source_id = None
    appearance = str(body.get("appearance") or character.get("appearance") or "").strip()
    if not appearance and mode == "t2i":
        raise ValueError(f"{character.get('name') or '角色'}：请先填写人物外观描述")
    if not appearance:
        appearance = "保持人物原图身份，生成同一人物的一致多视图"
    if any(item.get("kind") == "character_multiview" and item.get("project_id") == project_id
           and item.get("character_id") == character_id
           and item.get("status") in ("queued", "running") for item in state["runs"].values()):
        raise ValueError(f"{character.get('name') or '角色'}的多视图正在生成")
    return {"id": uid(), "kind": "character_multiview", "project_id": project_id,
            "character_id": character_id, "source_asset": source_id, "preset": preset,
            "mode": mode, "view_layout": view_layout, "engine": multiview_engine,
            "appearance": appearance[:2200],
            "xb_settings": {
                "output_lang": character.get("xb_output_lang", "zh"),
                "sampling_mode": character.get("xb_sampling_mode", "bypass"),
                "skill_mode": character.get("xb_skill_mode", "自动"),
                "canvas_override": character.get("xb_canvas_override", False),
                "aspect_ratio": character.get("xb_aspect_ratio", "16:9"),
                "width": character.get("xb_width", 1920),
                "height": character.get("xb_height", 1088),
                "krea_width": character.get("krea_width", 1536),
                "krea_height": character.get("krea_height", 1024),
                "seed": character.get("xb_seed", 0),
                "thinking": character.get("xb_thinking", False),
                "preset_text": character.get("xb_preset_text", ""),
                "extra_settings": character.get("xb_extra_settings", ""),
                "green_background": character.get("xb_green_background", True),
            },
            "status": "queued", "current": "等待 ComfyUI", "created_at": time.time()}


def generate_multiview_batch(job_ids: list[str]) -> None:
    """Run selected characters one at a time to avoid competing for GPU memory."""
    for job_id in job_ids:
        if read()["runs"].get(job_id, {}).get("cancel_requested"):
            change(lambda data: data["runs"][job_id].update(
                status="cancelled", current="已取消", finished_at=time.time()))
            continue
        generate_multiview(job_id)


def delete_projects(project_ids: list[str]) -> dict:
    """Delete existing idle projects in one transaction; retain media/history."""
    requested = list(dict.fromkeys(str(value) for value in project_ids if str(value)))
    if not requested or len(requested) > 500:
        raise ValueError("请选择 1–500 个项目")
    state = read()
    active = {run.get("project_id") for run in state["runs"].values()
              if run.get("status") in ("queued", "running", "awaiting_upscale")}
    blocked = [project_id for project_id in requested
               if project_id in state["projects"] and project_id in active]
    missing = [project_id for project_id in requested if project_id not in state["projects"]]
    deletable = [project_id for project_id in requested
                 if project_id in state["projects"] and project_id not in active]

    def remove(data):
        deleted = []
        for project_id in deletable:
            if data["projects"].pop(project_id, None) is not None:
                deleted.append(project_id)
        return deleted

    return {"deleted": change(remove), "blocked": blocked, "missing": missing}


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

    def do_GET(self) -> None:
        url = urllib.parse.urlsplit(self.path)
        path = url.path
        if path == "/comfy-ws":
            self.proxy_comfy_websocket(url.query)
            return
        if path in ("/", "/index.html"):
            self.file(ROOT / "static" / "index.html")
            return
        if path == "/api/version":
            self.json(200, {"version": "independent-66"})
            return
        if path == "/api/ai/settings":
            source = deepseek_key_source()
            self.json(200, {"provider": "DeepSeek", "model": DEEPSEEK_MODEL,
                            "configured": source != "none", "key_source": source,
                            "selected_provider": prompt_provider(),
                            "fallback": {"provider": "小米 MiMo", "model": MIMO_MODEL,
                                         "configured": mimo_key_source() != "none",
                                         "key_source": mimo_key_source()}})
            return
        if path == "/api/ai/provider":
            selected = prompt_provider()
            self.json(200, {"selected_provider": selected,
                            "options": ["auto", "deepseek", "mimo"]})
            return
        if path == "/api/product-agent/status":
            selected = prompt_provider()
            self.json(200, {"framework": "LangGraph", "read_only": False,
                            "provider": selected,
                            "configured": (bool(deepseek_key()) if selected == "deepseek" else
                                           bool(mimo_key()) if selected == "mimo" else
                                           bool(deepseek_key() or mimo_key())),
                            "knowledge": ["产品功能地图", "项目说明", "开源功能对照", "当前项目摘要"],
                            "actions": list(PRODUCT_AGENT_ACTIONS), "confirmation_required": True})
            return
        if path == "/api/ai/mimo/settings":
            source = mimo_key_source()
            self.json(200, {"provider": "小米 MiMo", "model": MIMO_MODEL,
                            "configured": source != "none", "key_source": source})
            return
        if path == "/api/ai/status":
            query = urllib.parse.parse_qs(url.query)
            self.json(200, deepseek_connection_status(force=query.get("refresh") == ["1"]))
            return
        if path == "/api/storage/status":
            self.json(200, storage_status())
            return
        if path == "/api/skills":
            self.json(200, [{key: value for key, value in skill.items() if key != "path"}
                            for skill in list_skills()])
            return
        if path.startswith("/api/skills/") and path.endswith("/preview"):
            skill_id = urllib.parse.unquote(path.split("/")[3])
            self.json(200, preview_skill(skill_id))
            return
        if path == "/api/videos":
            state = read()
            requested_project = urllib.parse.parse_qs(url.query).get("project_id", [""])[0]
            if requested_project and requested_project not in state["projects"]:
                self.send_error(404)
                return
            project_names = {project_id: project.get("name", "未命名项目")
                             for project_id, project in state["projects"].items()}
            entries, seen = [], set()
            for run in state["runs"].values():
                if requested_project and run.get("project_id") != requested_project:
                    continue
                if not requested_project and run.get("project_id") not in state["projects"]:
                    continue
                if run.get("status") in ("queued", "running", "awaiting_upscale"):
                    continue
                project_id = run.get("project_id", "")
                project = state["projects"].get(project_id, {})
                shot_names = {shot.get("id"): shot.get("name", "镜头")
                              for shot in project.get("shots", [])}
                created = run.get("finished_at") or run.get("created_at", 0)
                def add(relative, kind, title, shot_id=""):
                    if not relative:
                        return
                    rel = Path(str(relative))
                    if rel.is_absolute() or ".." in rel.parts or len(rel.parts) != 2:
                        return
                    key = (run.get("id", ""), rel.as_posix())
                    target = OUTPUTS / rel
                    if key in seen or not target.is_file() or not target.resolve().is_relative_to(OUTPUTS.resolve()):
                        return
                    seen.add(key)
                    entries.append({"id": f"{key[0]}:{key[1]}", "run_id": key[0],
                        "project_id": project_id, "project_name": project_names.get(project_id, "项目已删除"),
                        "shot_id": shot_id, "shot_name": shot_names.get(shot_id, title),
                        "title": title, "kind": kind, "file": rel.as_posix(),
                        "status": run.get("status", ""), "created_at": created,
                        "elapsed_seconds": max(0, int(run["finished_at"] - run["started_at"]))
                            if run.get("finished_at") and run.get("started_at") else None,
                        "size": target.stat().st_size})
                for result in run.get("results", []):
                    shot_id = result.get("shot_id", "")
                    shot_name = result.get("title") or shot_names.get(shot_id, run.get("imported_title") or "镜头成片")
                    add(result.get("file"), "imported" if run.get("kind") == "imported_video" else "clip", shot_name, shot_id)
                    add(result.get("h3_file"), "h3", shot_name + " · H3 原片", shot_id)
                    add(result.get("pre_refine_file"), "pre_refine", shot_name + " · 精修前", shot_id)
                add(run.get("merged"), "merged", run.get("merged_title") or "一镜到底全片")
            entries.sort(key=lambda item: (item.get("kind") != "merged",
                                           -item.get("created_at", 0)))
            self.json(200, entries)
            return
        if path == "/api/health":
            try:
                stats = api("/system_stats")
                try:
                    ops_loaded = ("H3StudioSelfLift" in api("/object_info/H3StudioSelfLift")
                                  and "H3StudioLatentContinue" in api("/object_info/H3StudioLatentContinue")
                                  and "H3StudioContinueRemask" in api("/object_info/H3StudioContinueRemask"))
                except Exception:
                    ops_loaded = False
                self.json(200, {"connected": True, "comfy_url": COMFY_URL,
                                "device": (stats.get("devices") or [{}])[0].get("name", ""),
                                "ops_loaded": ops_loaded})
            except Exception as error:
                self.json(200, {"connected": False, "comfy_url": COMFY_URL, "error": str(error)})
            return
        if path == "/api/models":
            def names(folder):
                root = COMFY_DIR / "models" / folder
                return sorted(p.name for p in root.iterdir() if p.is_file() and p.suffix.lower() in (".safetensors", ".pt", ".pth")) if root.is_dir() else []
            def relative_names(folder, suffixes=(".safetensors", ".pt", ".pth")):
                root = COMFY_DIR / "models" / folder
                return sorted(p.relative_to(root).as_posix() for p in root.rglob("*")
                              if p.is_file() and p.suffix.lower() in suffixes) if root.is_dir() else []
            try:
                sage_available = "MiniMaxH3MemoryEfficientSageAttentionPatch" in api("/object_info/MiniMaxH3MemoryEfficientSageAttentionPatch")
            except Exception:
                sage_available = False
            try:
                veda_node_available = "VedaSparseAttention" in api("/object_info/VedaSparseAttention")
            except Exception:
                veda_node_available = False
            h3_clips = sorted(set(
                [name for name in names("text_encoders") if "minimax_h3" in name] +
                [name for name in names("clip") if "minimax_h3" in name]
            ))
            self.json(200, {"semantic_adapters": names("semantic_bridge"),
                            "latent_models": names("latent_upscale_models"),
                            "h3_unet_models": [name for name in relative_names("diffusion_models") if "minimax_h3" in name.lower()],
                            "h3_lora_models": [name for name in relative_names("loras") if "minimax_h3" in name.lower()],
                            "h3_video_vaes": [name for name in names("vae") if name.startswith("minimax_h3_video_vae")],
                            "h3_audio_vaes": [name for name in names("vae") if name.startswith("minimax_h3_audio_vae")],
                            "h3_clip_models": h3_clips,
                            "seedvr2_models": relative_names("SEEDVR2", (".safetensors", ".pt", ".pth", ".gguf")),
                            "sage_attention": sage_available,
                            "veda_attention": veda_node_available and "minimax_h3_t2va_veda_8nfe_600step_preview_fp8.safetensors" in names("veda")})
            return
        if path == "/api/projects":
            state = read()
            self.json(200, [normalized_project_image_references(project, state["assets"])
                            for project in state["projects"].values()])
            return
        if path.startswith("/api/projects/") and path.endswith("/director-pack"):
            project_id = path.split("/")[3]
            if project_id not in read()["projects"]:
                self.send_error(404)
                return
            self.file(export_director_pack(project_id))
            return
        if path.startswith("/api/projects/") and path.endswith("/pack"):
            project_id = path.split("/")[3]
            if project_id not in read()["projects"]:
                self.send_error(404)
                return
            self.file(export_pack(project_id))
            return
        parts = path.strip("/").split("/")
        if len(parts) == 6 and parts[:2] == ["api", "projects"] and parts[3] == "shots" and parts[5] == "versions":
            state = read()
            project = state["projects"].get(parts[2])
            if not project or not any(shot.get("id") == parts[4] for shot in project.get("shots", [])):
                self.send_error(404)
                return
            versions = []
            for run in state["runs"].values():
                if run.get("project_id") != parts[2]:
                    continue
                for result in run.get("results", []):
                    if result.get("shot_id") == parts[4] and (OUTPUTS / result.get("file", "")).is_file():
                        versions.append({"run_id": run["id"], "created_at": run.get("created_at", 0),
                                         "status": run.get("status", ""),
                                         "elapsed_seconds": max(0, int(run["finished_at"] - run["started_at"]))
                                             if run.get("finished_at") and run.get("started_at") else None,
                                         **result})
            versions.sort(key=lambda item: item.get("created_at", 0), reverse=True)
            self.json(200, versions)
            return
        if path.startswith("/api/projects/"):
            project_id = path.split("/")[3]
            state = read()
            project = state["projects"].get(project_id)
            if project:
                project = normalized_project_image_references(project, state["assets"])
            self.json(200, project) if project else self.send_error(404)
            return
        if path == "/api/assets":
            state = read()
            requested_project = urllib.parse.parse_qs(url.query).get("project_id", [""])[0]
            if requested_project and requested_project not in state["projects"]:
                self.send_error(404)
                return
            self.json(200, assets_with_character_names(state, requested_project))
            return
        if path.startswith("/api/runs/") and path.endswith("/report"):
            run_id = path.split("/")[3]
            run = read()["runs"].get(run_id)
            self.json(200, run) if run else self.send_error(404)
            return
        if path.startswith("/api/runs/"):
            run_id = path.split("/")[3]
            run = read()["runs"].get(run_id)
            self.json(200, run) if run else self.send_error(404)
            return
        if path == "/api/runs":
            self.json(200, [run for run in read()["runs"].values()
                            if run.get("kind") not in ("character_multiview", "entity_image")])
            return
        if path == "/api/profile-images":
            state = read()
            project_id = urllib.parse.parse_qs(url.query).get("project_id", [""])[0]
            if project_id not in state["projects"]:
                self.send_error(404)
                return
            self.json(200, [run for run in state["runs"].values()
                            if run.get("kind") == "entity_image" and run.get("project_id") == project_id])
            return
        if path.startswith("/api/profile-images/"):
            job_id = path.split("/")[3]
            job = read()["runs"].get(job_id)
            requested_project = urllib.parse.parse_qs(url.query).get("project_id", [""])[0]
            self.json(200, job) if (job and job.get("kind") == "entity_image"
                                  and job.get("project_id") == requested_project) else self.send_error(404)
            return
        if path == "/api/multiview":
            state = read()
            project_id = urllib.parse.parse_qs(url.query).get("project_id", [""])[0]
            if project_id not in state["projects"]:
                self.send_error(404)
                return
            self.json(200, [run for run in state["runs"].values()
                            if run.get("kind") == "character_multiview"
                            and run.get("project_id") == project_id])
            return
        if path.startswith("/api/multiview/"):
            job_id = path.split("/")[3]
            job = read()["runs"].get(job_id)
            requested_project = urllib.parse.parse_qs(url.query).get("project_id", [""])[0]
            self.json(200, job) if (job and job.get("kind") == "character_multiview"
                                  and job.get("project_id") == requested_project) else self.send_error(404)
            return
        if path.startswith("/media/assets/"):
            asset_id = path.split("/")[3]
            asset = read()["assets"].get(asset_id)
            self.file(ASSETS / asset["file"]) if asset else self.send_error(404)
            return
        if path.startswith("/media/outputs/"):
            parts = [urllib.parse.unquote(value) for value in path.split("/")[3:]]
            if len(parts) != 2 or any(not value or value in (".", "..") or "/" in value for value in parts):
                self.send_error(400)
                return
            self.file(OUTPUTS / parts[0] / parts[1])
            return
        self.send_error(404)

    def do_POST(self) -> None:
        path = urllib.parse.urlsplit(self.path).path
        if not self.same_origin_request():
            self.json(403, {"error": "请从当前 H3 Studio 工作台执行修改操作"})
            return
        try:
            if path == "/api/product-agent/chat":
                self.json(200, product_agent_chat(self.body_json()))
                return
            if path == "/api/product-agent/confirm":
                body = self.body_json()
                self.json(200, confirm_product_agent_action(
                    str(body.get("token") or ""), str(body.get("project_id") or "")))
                return
            if path == "/api/videos/import":
                if not self.same_origin_request():
                    self.json(403, {"error": "请从工作台导入视频"})
                    return
                params = urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query)
                project_id = (params.get("project_id") or [""])[0]
                if project_id not in read()["projects"]:
                    raise ValueError("请先在视频库选择导入目标项目")
                name = Path((params.get("name") or [""])[0]).name
                suffix = Path(name).suffix.lower()
                if EXTENSIONS.get(suffix) != "video":
                    raise ValueError("请选择 MP4、MOV、WebM、MKV、AVI 或 M4V 视频")
                size = int(self.headers.get("Content-Length", "0"))
                if size < 1 or size > 4_000_000_000:
                    raise ValueError("视频大小无效或超过 4GB")
                run_id = uid()
                folder = OUTPUTS / run_id
                folder.mkdir()
                target = folder / ("imported" + suffix)
                try:
                    remaining = size
                    with target.open("wb") as output:
                        while remaining:
                            block = self.rfile.read(min(1024 * 1024, remaining))
                            if not block:
                                raise ValueError("视频上传中断")
                            output.write(block)
                            remaining -= len(block)
                    relative = f"{run_id}/{target.name}"
                    now = time.time()
                    run = {"id": run_id, "project_id": project_id, "kind": "imported_video",
                           "imported_title": name, "status": "completed", "created_at": now,
                           "finished_at": now, "shot_ids": [],
                           "results": [{"shot_id": "", "file": relative}]}
                    change(lambda data: data["runs"].__setitem__(run_id, run))
                except Exception:
                    target.unlink(missing_ok=True)
                    folder.rmdir()
                    raise
                self.json(201, {"run_id": run_id, "project_id": project_id, "file": relative})
                return
            if path == "/api/projects":
                body = self.body_json()
                project = new_project(str(body.get("name") or ""))
                change(lambda data: data["projects"].__setitem__(project["id"], project))
                self.json(201, project)
                return
            if path.startswith("/api/projects/") and path.endswith("/clone"):
                if not self.same_origin_request():
                    self.json(403, {"error": "请从工作台页面复制项目"})
                    return
                source_id = path.split("/")[3]
                project = clone_project(source_id, str(self.body_json().get("name") or ""))
                self.json(201, project)
                return
            if path == "/api/projects/batch-delete":
                body = self.body_json()
                project_ids = body.get("project_ids")
                if not isinstance(project_ids, list):
                    raise ValueError("项目 ID 列表无效")
                self.json(200, delete_projects(project_ids))
                return
            if path == "/api/skill-agent/parameters":
                if not self.same_origin_request():
                    self.json(403, {"error": "请从工作台调用 Skill 参数智能体"})
                    return
                body = self.body_json()
                project = read()["projects"].get(str(body.get("project_id") or ""))
                if not project:
                    raise ValueError("项目不存在")
                selected_skill = project.get("prompt_skill") or DEFAULT_SKILL
                if selected_skill not in {item["id"] for item in list_skills()}:
                    selected_skill = DEFAULT_SKILL

                def model_names(folder):
                    root = COMFY_DIR / "models" / folder
                    return sorted(path.name for path in root.iterdir() if path.is_file()
                                  and path.suffix.lower() in (".safetensors", ".pt", ".pth")) if root.is_dir() else []
                try:
                    sage_available = "MiniMaxH3MemoryEfficientSageAttentionPatch" in api(
                        "/object_info/MiniMaxH3MemoryEfficientSageAttentionPatch")
                except Exception:
                    sage_available = False
                try:
                    stats = api("/system_stats")
                    device = (stats.get("devices") or [{}])[0]
                    hardware = {key: device.get(key) for key in ("name", "vram_total", "vram_free") if key in device}
                except Exception:
                    hardware = {}
                turbo4_installed = (COMFY_DIR / "models" / "loras" / "minimax_h3"
                                    / "minimax_h3_fl2v_lightx2v_turbo_4step_v0.1_comfy.safetensors").is_file()
                available = {"sage_attention": sage_available, "turbo4": turbo4_installed,
                             "h3_video_vaes": [name for name in model_names("vae")
                                                if name.startswith("minimax_h3_video_vae")],
                             "latent_models": model_names("latent_upscale_models")}
                result = suggest_project_parameters(selected_skill, project, available, hardware)
                result["skill_id"] = selected_skill
                self.json(200, result)
                return
            if path == "/api/skill-agent/recommend":
                if not self.same_origin_request():
                    self.json(403, {"error": "请从工作台调用 Skill 推荐助手"})
                    return
                body = self.body_json()
                project = read()["projects"].get(str(body.get("project_id") or ""))
                if not project:
                    raise ValueError("项目不存在")
                self.json(200, recommend_project_skills(project, list_skills()))
                return
            if path == "/api/translate-prompt-stream":
                if not self.same_origin_request():
                    self.json(403, {"error": "请从工作台调用提示词翻译"})
                    return
                body = self.body_json()
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-cache, no-transform")
                self.send_header("X-Accel-Buffering", "no")
                self.send_header("Connection", "keep-alive")
                self.end_headers()
                try:
                    for chunk in stream_translate_prompt(str(body.get("prompt") or "")):
                        self.wfile.write(("data: " + json.dumps({"text": chunk}, ensure_ascii=False) + "\n\n").encode("utf-8"))
                        self.wfile.flush()
                    self.wfile.write(b"data: {\"done\":true}\n\n")
                    self.wfile.flush()
                except Exception as error:
                    self.wfile.write(("data: " + json.dumps({"error": str(error)}, ensure_ascii=False) + "\n\n").encode("utf-8"))
                    self.wfile.flush()
                return
            if path == "/api/translate-prompt":
                if not self.same_origin_request():
                    self.json(403, {"error": "请从工作台调用提示词翻译"})
                    return
                body = self.body_json()
                translation, backend = translate_prompt_with_backend(
                    str(body.get("prompt") or ""), str(body.get("quality") or "auto"),
                    str(body.get("target_language") or "zh"))
                self.json(200, {"translation": translation, "backend": backend})
                return
            if path == "/api/enhance-sequence":
                body = self.body_json()
                state = read()
                project_id = str(body.get("project_id") or "")
                project_data = state["projects"].get(project_id)
                if not project_data:
                    raise ValueError("项目不存在，请刷新页面后重试")
                if project_data.get("continuous_take") is False:
                    raise ValueError("请先开启“强一镜到底”再统筹优化")
                saved_shots = project_data.get("shots", [])
                if len(saved_shots) < 2:
                    raise ValueError("至少需要两个镜头才能统筹优化")
                skill_id = str(project_data.get("prompt_skill") or DEFAULT_SKILL)
                skill_catalog = {item["id"]: item for item in list_skills()}
                if skill_id not in skill_catalog:
                    skill_id = DEFAULT_SKILL
                catalog = state["assets"]
                sequence = []
                for index, saved_shot in enumerate(saved_shots):
                    prepared = prepare_shot(saved_shot, project_data)
                    identity_assets = set(_shot_identity_refs(saved_shot, project_data, catalog))
                    if prepared.get("mode") in {"t2v", "i2v", "fl2v"}:
                        identity_assets.intersection_update(filter(None, (
                            prepared.get("first_image"), prepared.get("last_image"))))
                    context = {"segment": {"position": index + 1, "total": len(saved_shots),
                        "is_first": index == 0, "is_last": index == len(saved_shots) - 1},
                        "references": [], "characters": prepared.get("_h3_characters", []),
                        "entities": prepared.get("_h3_entities", []),
                        "person_identity_asset_ids": list(identity_assets)}
                    context["references"] = enhancement_references(
                        prepared, project_data, catalog, identity_assets)
                    sequence.append({"id": str(saved_shot.get("id") or index + 1),
                        "name": str(saved_shot.get("name") or f"镜头{index + 1}"),
                        "prompt": str(saved_shot.get("prompt") or ""), "mode": str(saved_shot.get("mode") or "t2v"),
                        "duration": saved_shot.get("duration", 5),
                        "camera_move": saved_shot.get("camera_move", ""),
                        "camera_speed": saved_shot.get("camera_speed", ""),
                        "action_intensity": saved_shot.get("action_intensity", ""),
                        "dialogue_time": saved_shot.get("dialogue_time", ""),
                        "dialogue_text": saved_shot.get("dialogue_text", ""),
                        "output_language": str(saved_shot.get("prompt_language") or "en"), "context": context})
                sequence_brief = (project_data.get("sequence_brief") or project_data.get("script_text")
                                  or project_data.get("shared_prompt") or "")
                optimization_scope = str(body.get("optimization_scope") or "全部提示词").strip()[:500]
                prompts = enhance_sequence(sequence, skill_id, True, str(sequence_brief), optimization_scope)
                self.json(200, {"prompts": [{"id": item["id"], "prompt": prompt}
                    for item, prompt in zip(sequence, prompts)], "skill_id": skill_id,
                    "skill_name": skill_catalog.get(skill_id, {}).get("name", skill_id)})
                return
            if path == "/api/image-prompt":
                if not self.same_origin_request():
                    self.json(403, {"error": "请从工作台调用 AI 生图描述"})
                    return
                body = self.body_json()
                project_id = str(body.get("project_id") or "")
                project_data = read()["projects"].get(project_id)
                if not project_data:
                    raise ValueError("项目不存在，请刷新页面后重试")
                result = optimize_image_prompt(project_data,
                    str(body.get("target_type") or ""), str(body.get("target_id") or ""),
                    str(body.get("scope") or ""))
                self.json(200, result)
                return
            if path == "/api/enhance":
                body = self.body_json()
                reference_context = {}
                state = read()
                project_id, shot_id = str(body.get("project_id") or ""), str(body.get("shot_id") or "")
                project_data = state["projects"].get(project_id) if project_id else None
                skill_id = str(body.get("skill_id") or DEFAULT_SKILL)
                if project_id:
                    if not project_data:
                        raise ValueError("项目不存在，请刷新页面后重试")
                    skill_id = str(project_data.get("prompt_skill") or DEFAULT_SKILL)
                skill_catalog = {item["id"]: item for item in list_skills()}
                if skill_id not in skill_catalog:
                    skill_id = DEFAULT_SKILL
                if project_id and shot_id:
                    if not project_data:
                        raise ValueError("项目不存在，请刷新页面后重试")
                    saved_shot = next((item for item in project_data.get("shots", []) if item.get("id") == shot_id), None)
                    if not saved_shot:
                        raise ValueError("镜头不存在，请刷新页面后重试")
                    catalog = state["assets"]
                    prepared = prepare_shot(saved_shot, project_data)
                    identity_assets = set(_shot_identity_refs(saved_shot, project_data, catalog))
                    if prepared.get("mode") in {"t2v", "i2v", "fl2v"}:
                        identity_assets.intersection_update(filter(None, (
                            prepared.get("first_image"), prepared.get("last_image"))))
                    saved_shots = project_data.get("shots", [])
                    shot_index = next((index for index, item in enumerate(saved_shots)
                                       if item.get("id") == shot_id), 0)
                    def neighbor_context(index):
                        if index < 0 or index >= len(saved_shots):
                            return None
                        item = saved_shots[index]
                        # Never send another shot's full prompt to single-shot
                        # optimization: its subjects and Picture tags can leak.
                        return {"name": str(item.get("name") or "镜头")[:120],
                                "mode": str(item.get("mode") or "t2v")}
                    reference_context = {"segment": {
                        "position": shot_index + 1, "total": len(saved_shots),
                        "is_first": shot_index == 0, "is_last": shot_index == len(saved_shots) - 1,
                        "previous_shot": neighbor_context(shot_index - 1),
                        "next_shot": neighbor_context(shot_index + 1)},
                        "selected_character_ids": list(saved_shot.get("character_ids") or [])}
                    catalog = state["assets"]
                    references = enhancement_references(
                        prepared, project_data, catalog, identity_assets)
                    reference_context.update({"references": references,
                        "characters": prepared.get("_h3_characters", []),
                        "entities": prepared.get("_h3_entities", []),
                        "person_identity_asset_ids": list(identity_assets)})
                output_language = str(body.get("output_language") or "en")
                if output_language not in {"en", "zh"}:
                    raise ValueError("提示词语言只能选择中文或英文")
                optimization_scope = str(body.get("optimization_scope") or "全部提示词").strip()[:500]
                optimized = enhance(str(body.get("prompt") or ""),
                    str((saved_shot.get("mode") or "t2v") if project_id and shot_id else body.get("mode") or "t2v"), skill_id,
                    saved_shot.get("duration", 5) if project_id and shot_id else body.get("duration", 5),
                    reference_context, output_language, optimization_scope)
                self.json(200, {"prompt": optimized, "skill_id": skill_id,
                                "output_language": output_language,
                                "skill_name": skill_catalog.get(skill_id, {}).get("name", skill_id),
                                "base_skill_id": DEFAULT_SKILL})
                return
            if path == "/api/storyboard":
                body = self.body_json()
                shots = generate_storyboard(str(body.get("brief") or ""), body.get("count", 4),
                                            str(body.get("skill_id") or DEFAULT_SKILL))
                self.json(200, {"shots": shots})
                return
            if path == "/api/skills/import":
                if not self.same_origin_request():
                    self.json(403, {"error": "请从工作台导入提示词 Skill"})
                    return
                size = int(self.headers.get("Content-Length", "0"))
                if size < 1 or size > 5_000_000:
                    raise ValueError("Skill 文件为空或超过 5 MB")
                payload = self.rfile.read(size)
                if len(payload) != size:
                    raise ValueError("Skill 文件上传中断")
                name = Path((urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query).get("name") or [""])[0]).name
                skill = import_skill(name, payload)
                self.json(201, {key: value for key, value in skill.items() if key != "path"})
                return
            if path == "/api/videos/delete":
                if not self.same_origin_request():
                    self.json(403, {"error": "请从工作台管理生成视频"})
                    return
                body = self.body_json()
                requested_project = str(body.get("project_id") or "")
                if requested_project not in read()["projects"]:
                    raise ValueError("请选择有效项目后再管理视频")
                requested = body.get("items") or [body]
                if not isinstance(requested, list) or not requested:
                    raise ValueError("请先选择要删除的视频")
                state = read()
                validated = []
                requested_keys = set()
                for entry in requested:
                    run_id = str(entry.get("run_id") or "")
                    relative = str(entry.get("file") or "")
                    request_key = (run_id, relative)
                    if request_key in requested_keys:
                        continue
                    requested_keys.add(request_key)
                    run = state["runs"].get(run_id)
                    if not run:
                        raise ValueError("生成记录不存在，请刷新视频库后重试")
                    if run.get("project_id") != requested_project:
                        raise ValueError("不能删除其他项目的视频")
                    if run.get("status") in ("queued", "running", "awaiting_upscale"):
                        raise ValueError("任务运行中，暂不能删除视频")
                    rel = Path(relative)
                    target = (OUTPUTS / rel).resolve()
                    known = any(relative == result.get(key) for result in run.get("results", [])
                                for key in ("file", "h3_file", "pre_refine_file"))
                    known = known or any(run.get(key) == relative for key in
                                         ("merged", "preview_file", "pre_refine_file"))
                    known = known or any(relative in (checkpoint.get("h3_file"), checkpoint.get("pre_refine_file"))
                                         for checkpoint in run.get("checkpoints", {}).values())
                    if (rel.is_absolute() or ".." in rel.parts or len(rel.parts) != 2
                            or not target.is_relative_to(OUTPUTS.resolve())
                            or not target.is_file() or not known):
                        raise ValueError("视频记录已变化或文件无效，请刷新视频库后重试")
                    validated.append((run_id, rel.as_posix(), target))
                deleted_paths = {(run_id, relative) for run_id, relative, _ in validated}

                def remove_videos(data):
                    for run_id, relative in deleted_paths:
                        item = data["runs"].get(run_id)
                        if not item:
                            continue
                        for result in item.get("results", []):
                            for key in ("file", "h3_file", "pre_refine_file"):
                                if result.get(key) == relative:
                                    result[key] = None
                        item["results"] = [result for result in item.get("results", [])
                                           if any(result.get(key) for key in ("file", "h3_file", "pre_refine_file"))]
                        for key in ("merged", "preview_file", "pre_refine_file"):
                            if item.get(key) == relative:
                                item[key] = None
                        for shot_id, checkpoint in list(item.get("checkpoints", {}).items()):
                            for key in ("h3_file", "pre_refine_file"):
                                if checkpoint.get(key) == relative:
                                    checkpoint[key] = None
                            if not any(checkpoint.get(key) for key in ("h3_file", "pre_refine_file")):
                                item["checkpoints"].pop(shot_id, None)
                change(remove_videos)
                remaining_paths = set()
                for item in read()["runs"].values():
                    for result in item.get("results", []):
                        remaining_paths.update(value for value in
                                                (result.get("file"), result.get("h3_file"), result.get("pre_refine_file"))
                                                if value)
                    remaining_paths.update(value for value in
                                           (item.get("merged"), item.get("preview_file"), item.get("pre_refine_file"))
                                           if value)
                    for checkpoint in item.get("checkpoints", {}).values():
                        remaining_paths.update(value for value in
                                               (checkpoint.get("h3_file"), checkpoint.get("pre_refine_file"))
                                               if value)
                removed_files = set()
                preserved_shared_files = set()
                for _, relative, target in validated:
                    if relative in remaining_paths:
                        preserved_shared_files.add(relative)
                    elif relative not in removed_files:
                        target.unlink(missing_ok=True)
                        removed_files.add(relative)
                self.json(200, {"deleted": [relative for _, relative, _ in validated],
                                "count": len(validated), "files_removed": len(removed_files),
                                "shared_files_preserved": len(preserved_shared_files)})
                return
            if path == "/api/ai/test":
                if not self.same_origin_request():
                    self.json(403, {"error": "请从工作台页面测试云端 AI"})
                    return
                # This button verifies DeepSeek itself. Normal optimization may
                # still transparently use the configured MiMo fallback.
                _deepseek_complete_request(
                    [{"role": "system", "content": "Reply with exactly OK."},
                     {"role": "user", "content": "OK"}],
                    max_tokens=128, timeout=35)
                self.json(200, {"ok": True, "provider": "DeepSeek", "model": DEEPSEEK_MODEL,
                                "message": "连接成功，API Key 与模型均可用。"})
                return
            if path == "/api/ai/mimo/test":
                if not self.same_origin_request():
                    self.json(403, {"error": "请从工作台页面测试云端 AI"})
                    return
                _mimo_complete(
                    [{"role": "system", "content": "Reply with exactly OK."},
                     {"role": "user", "content": "OK"}],
                    max_tokens=128, timeout=35)
                self.json(200, {"ok": True, "provider": "小米 MiMo", "model": MIMO_MODEL,
                                "message": "连接成功，API Key 与模型均可用。"})
                return
            if path == "/api/ai/settings":
                if not self.same_origin_request():
                    self.json(403, {"error": "请从工作台页面配置云端 AI"})
                    return
                body = self.body_json()
                save_deepseek_key(body.get("api_key", ""))
                source = deepseek_key_source()
                self.json(200, {"provider": "DeepSeek", "model": DEEPSEEK_MODEL,
                                "configured": source != "none", "key_source": source})
                return
            if path == "/api/ai/provider":
                if not self.same_origin_request():
                    self.json(403, {"error": "请从工作台页面配置云端 AI"})
                    return
                selected = save_prompt_provider(self.body_json().get("provider"))
                self.json(200, {"selected_provider": selected})
                return
            if path == "/api/ai/mimo/settings":
                if not self.same_origin_request():
                    self.json(403, {"error": "请从工作台页面配置云端 AI"})
                    return
                body = self.body_json()
                save_mimo_key(body.get("api_key", ""))
                source = mimo_key_source()
                self.json(200, {"provider": "小米 MiMo", "model": MIMO_MODEL,
                                "configured": source != "none", "key_source": source})
                return
            if path == "/api/timeline/edit":
                body = self.body_json()
                shots = edit_timeline(body.get("shots"), body.get("shot_ids"),
                                      str(body.get("action") or ""), body.get("value"))
                self.json(200, {"shots": shots})
                return
            if path == "/api/prompt-preview":
                body = self.body_json()
                state = read()
                project = state["projects"].get(str(body.get("project_id") or ""))
                if not project:
                    raise ValueError("项目不存在")
                shot = next((item for item in project.get("shots", []) if item.get("id") == body.get("shot_id")), None)
                if not shot:
                    raise ValueError("镜头不存在")
                prepared = prepare_shot(shot, project)
                prepared["continuous_take"] = bool(project.get("continuous_take", True))
                prompt = compile_shot_prompt(prepared)
                self.json(200, {"prompt": prompt, "findings": lint_shot_prompt(prepared),
                                "image_references": prepared.get("_h3_image_reference_map", [])})
                return
            if path == "/api/assets":
                params = urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query)
                project_id = (params.get("project_id") or [""])[0]
                if project_id not in read()["projects"]:
                    raise ValueError("请先选择有效项目，再上传素材")
                name = Path((params.get("name") or [""])[0]).name
                suffix = Path(name).suffix.lower()
                if suffix not in EXTENSIONS:
                    raise ValueError("只支持图片、视频和音频文件")
                size = int(self.headers.get("Content-Length", "0"))
                if size < 1 or size > 4_000_000_000:
                    raise ValueError("文件大小无效或超过 4GB")
                asset_id = uid()
                filename = asset_id + suffix
                target = ASSETS / filename
                remaining = size
                with target.open("wb") as output:
                    while remaining:
                        block = self.rfile.read(min(1024 * 1024, remaining))
                        if not block:
                            raise ValueError("文件上传中断")
                        output.write(block)
                        remaining -= len(block)
                asset = {"id": asset_id, "name": name, "file": filename,
                         "kind": EXTENSIONS[suffix], "size": size, "created_at": time.time(),
                         "project_id": project_id}
                change(lambda data: data["assets"].__setitem__(asset_id, asset))
                self.json(201, asset)
                return
            if path.startswith("/api/assets/") and path.endswith("/extract-audio"):
                asset_id = path.split("/")[3]
                asset = read()["assets"].get(asset_id)
                if not asset or asset.get("kind") != "video":
                    raise ValueError("请选择视频素材")
                source = ASSETS / asset["file"]
                if not source.is_file():
                    raise ValueError("视频文件不存在")
                new_id = uid()
                filename = new_id + ".flac"
                target = ASSETS / filename
                extract_audio_track(source, target)
                audio_asset = {"id": new_id, "name": Path(asset["name"]).stem + " · 提取音轨.flac",
                               "file": filename, "kind": "audio", "size": target.stat().st_size,
                               "created_at": time.time(), "extracted_from": asset_id,
                               "project_id": asset.get("project_id")}
                change(lambda data: data["assets"].__setitem__(new_id, audio_asset))
                self.json(201, audio_asset)
                return
            if path == "/api/packs/import":
                size = int(self.headers.get("Content-Length", "0"))
                if size < 1 or size > 4_000_000_000:
                    raise ValueError("项目包大小无效")
                with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as temp:
                    remaining = size
                    while remaining:
                        block = self.rfile.read(min(1024 * 1024, remaining))
                        if not block:
                            raise ValueError("项目包上传中断")
                        temp.write(block)
                        remaining -= len(block)
                    temp_path = Path(temp.name)
                try:
                    project = import_pack(temp_path)
                finally:
                    temp_path.unlink(missing_ok=True)
                self.json(201, project)
                return
            parts = path.strip("/").split("/")
            if (len(parts) == 5 and parts[:2] == ["api", "projects"]
                    and parts[3:] == ["characters", "batch-multiview"]):
                body = self.body_json()
                requests = body.get("characters")
                if not isinstance(requests, list) or not 1 <= len(requests) <= 100:
                    raise ValueError("请选择 1–100 个角色批量生成")
                character_ids = [str(item.get("character_id") or "") if isinstance(item, dict) else ""
                                 for item in requests]
                if not all(character_ids) or len(set(character_ids)) != len(character_ids):
                    raise ValueError("批量角色列表无效或存在重复")
                state = read()
                if parts[2] not in state["projects"]:
                    raise ValueError("项目不存在")
                if any(run.get("kind") == "character_multiview" and run.get("project_id") == parts[2]
                       and run.get("status") in ("queued", "running") for run in state["runs"].values()):
                    raise ValueError("当前项目已有角色正在生成，请完成后再启动批量任务")
                jobs = [make_multiview_job(state, parts[2], character_id, item)
                        for character_id, item in zip(character_ids, requests)]
                batch_id = uid()
                for index, job in enumerate(jobs, 1):
                    job["batch_id"] = batch_id
                    job["batch_position"] = index
                    job["batch_total"] = len(jobs)
                    job["current"] = f"批量排队 {index}/{len(jobs)}"
                change(lambda data: data["runs"].update({job["id"]: job for job in jobs}))
                threading.Thread(target=generate_multiview_batch,
                                 args=([job["id"] for job in jobs],), daemon=True).start()
                self.json(202, {"batch_id": batch_id, "jobs": jobs})
                return
            if (len(parts) == 6 and parts[:2] == ["api", "projects"]
                    and parts[3] == "characters" and parts[5] == "multiview"):
                body = self.body_json()
                preset = str(body.get("preset") or "fast")
                if preset not in ("fast", "stable"):
                    raise ValueError("多视图生成模式无效")
                view_layout = str(body.get("view_layout") or "four")
                if view_layout not in ("four", "five", "six"):
                    raise ValueError("角色多视图布局无效")
                multiview_engine = str(body.get("engine") or "qwen")
                if multiview_engine not in ("xb_qwen", "qwen", "krea2", "qwen_sheet"):
                    raise ValueError("多视图生成引擎无效")
                if multiview_engine == "krea2" and view_layout != "four":
                    raise ValueError("Krea2 辅助工作流只支持四视图")
                if multiview_engine == "qwen_sheet" and view_layout != "six":
                    raise ValueError("定装照工作流只支持六视图")
                if multiview_engine != "qwen_sheet" and view_layout == "six":
                    raise ValueError("六视图请选择 Qwen 2.1 定装照工作流")
                state = read()
                project = state["projects"].get(parts[2])
                if any(item.get("kind") == "character_multiview" and item.get("project_id") == parts[2]
                       and item.get("batch_id") and item.get("status") in ("queued", "running")
                       for item in state["runs"].values()):
                    raise ValueError("当前有角色批量任务在生成，请等待批量任务完成")
                character = next((item for item in project.get("characters", [])
                                  if item.get("id") == parts[4]), None) if project else None
                if not character:
                    raise ValueError("当前项目中找不到这个角色")
                source_id = character.get("portrait_asset")
                mode = str(body.get("mode") or "i2i")
                if mode not in ("t2i", "i2i"):
                    raise ValueError("多视图生图模式无效")
                if multiview_engine in ("krea2", "qwen_sheet") and mode != "i2i":
                    raise ValueError("当前定装照工作流只支持参考原图生成")
                if mode == "i2i":
                    source = state["assets"].get(source_id)
                    if not source or source.get("project_id") != parts[2] or source.get("kind") != "image":
                        raise ValueError("图生图请先绑定当前项目的人物原图")
                else:
                    source_id = None
                appearance = str(body.get("appearance") or character.get("appearance") or "").strip()
                if not appearance and mode == "t2i":
                    raise ValueError("请先填写人物外观描述")
                if not appearance:
                    appearance = "保持人物原图身份，生成同一人物的一致多视图"
                if any(item.get("kind") == "character_multiview" and item.get("project_id") == parts[2]
                       and item.get("character_id") == parts[4]
                       and item.get("status") in ("queued", "running") for item in state["runs"].values()):
                    raise ValueError("这个角色的多视图正在生成")
                job_id = uid()
                job = {"id": job_id, "kind": "character_multiview", "project_id": parts[2],
                       "character_id": parts[4], "source_asset": source_id, "preset": preset,
                       "mode": mode, "view_layout": view_layout, "engine": multiview_engine,
                       "appearance": appearance[:2200],
                       "xb_settings": {
                           "output_lang": character.get("xb_output_lang", "zh"),
                           "sampling_mode": character.get("xb_sampling_mode", "bypass"),
                           "skill_mode": character.get("xb_skill_mode", "自动"),
                           "canvas_override": character.get("xb_canvas_override", False),
                           "aspect_ratio": character.get("xb_aspect_ratio", "16:9"),
                           "width": character.get("xb_width", 1920),
                           "height": character.get("xb_height", 1088),
                           "krea_width": character.get("krea_width", 1536),
                           "krea_height": character.get("krea_height", 1024),
                           "seed": character.get("xb_seed", 0),
                           "thinking": character.get("xb_thinking", False),
                           "preset_text": character.get("xb_preset_text", ""),
                           "extra_settings": character.get("xb_extra_settings", ""),
                           "green_background": character.get("xb_green_background", True),
                       },
                       "status": "queued", "current": "等待 ComfyUI", "created_at": time.time()}
                change(lambda data: data["runs"].__setitem__(job_id, job))
                threading.Thread(target=generate_multiview, args=(job_id,), daemon=True).start()
                self.json(202, job)
                return
            if (len(parts) == 6 and parts[:2] == ["api", "projects"]
                    and parts[3] == "entities" and parts[5] == "image"):
                body = self.body_json()
                mode = str(body.get("mode") or "t2i")
                preset = str(body.get("preset") or "fast")
                if mode not in {"t2i", "i2i"} or preset not in {"fast", "stable"}:
                    raise ValueError("档案生图模式无效")
                state = read()
                project = state["projects"].get(parts[2])
                entity = next((item for item in project.get("entities", []) if item.get("id") == parts[4]), None) if project else None
                if not entity:
                    raise ValueError("当前项目中找不到这个档案")
                description = str(body.get("description") or entity.get("description") or "").strip()
                if not description:
                    raise ValueError("请先填写生图描述")
                source_id = str(body.get("source_asset") or "") if mode == "i2i" else ""
                if mode == "i2i":
                    source = state["assets"].get(source_id)
                    if not source or source.get("project_id") != parts[2] or source.get("kind") != "image":
                        raise ValueError("图生图请选择当前项目的参考图片")
                if any(item.get("kind") == "entity_image" and item.get("project_id") == parts[2]
                       and item.get("entity_id") == parts[4] and item.get("status") in ("queued", "running")
                       for item in state["runs"].values()):
                    raise ValueError("这个档案正在生图")
                job_id = uid()
                job = {"id": job_id, "kind": "entity_image", "project_id": parts[2],
                       "entity_id": parts[4], "entity_kind": entity.get("kind", "prop"),
                       "entity_name": str(entity.get("name") or "设定")[:80],
                       "description": description[:1800], "source_asset": source_id or None,
                       "mode": mode, "preset": preset, "status": "queued",
                       "current": "等待 ComfyUI", "created_at": time.time()}
                change(lambda data: data["runs"].__setitem__(job_id, job))
                threading.Thread(target=generate_profile_image, args=(job_id,), daemon=True).start()
                self.json(202, job)
                return
            if len(parts) == 4 and parts[:2] == ["api", "profile-images"] and parts[3] == "cancel":
                job_id = parts[2]
                requested_project = str(self.body_json().get("project_id") or "")
                job = read()["runs"].get(job_id)
                if not job or job.get("kind") != "entity_image" or job.get("project_id") != requested_project:
                    self.send_error(404)
                    return
                if job.get("status") not in ("queued", "running"):
                    raise ValueError("这个生图任务已经结束")
                change(lambda data: data["runs"][job_id].__setitem__("cancel_requested", True))
                self.json(200, {"cancel_requested": True})
                return
            if len(parts) == 4 and parts[:2] == ["api", "multiview"] and parts[3] == "cancel":
                job_id = parts[2]
                requested_project = str(self.body_json().get("project_id") or "")
                job = read()["runs"].get(job_id)
                if not job or job.get("kind") != "character_multiview" or job.get("project_id") != requested_project:
                    self.send_error(404)
                    return
                if job.get("status") not in ("queued", "running"):
                    raise ValueError("这个多视图任务已经结束")
                change(lambda data: data["runs"][job_id].__setitem__("cancel_requested", True))
                self.json(200, {"cancel_requested": True})
                return
            if len(parts) == 4 and parts[:2] == ["api", "projects"] and parts[3] == "shots":
                body = self.body_json()
                shot = new_shot()
                mode = body.get("mode")
                if mode in {"t2v", "i2v", "fl2v", "r2v", "v2v", "rv2v"}:
                    shot["mode"] = mode
                change(lambda data: data["projects"][parts[2]]["shots"].append(shot))
                self.json(201, shot)
                return
            if len(parts) == 4 and parts[:2] == ["api", "projects"] and parts[3] == "split":
                body = self.body_json()
                asset_id = body.get("asset_id")
                asset = read()["assets"].get(asset_id)
                if not asset or asset["kind"] != "video" or asset.get("project_id") != parts[2]:
                    raise ValueError("请选择已上传的视频")
                spans = boundaries(ASSETS / asset["file"], body.get("method", "equal"),
                                   body.get("count", 4), body.get("threshold", 0.35),
                                   body.get("custom_cuts", []))
                shots = []
                for index, (start, length) in enumerate(spans, 1):
                    shot = new_shot()
                    shot.update(name=f"{asset['name']} · {index}", mode="v2v", source_video=asset_id,
                                source_start=start, duration=length, prompt="保留源视频的主体、动作与构图，提升细节和光照。")
                    shots.append(shot)
                # Re-cutting from the director UI is an explicit replacement:
                # otherwise every press kept the old pieces and made a request
                # for four parts look like ten-plus parts in the timeline.
                def save_split(data):
                    project_shots = data["projects"][parts[2]]["shots"]
                    if body.get("replace_source_segments"):
                        project_shots[:] = [item for item in project_shots if not (
                            item.get("source_video") == asset_id
                            and item.get("mode") in {"v2v", "rv2v"}
                        )]
                    project_shots.extend(shots)
                change(save_split)
                self.json(201, shots)
                return
            if len(parts) == 4 and parts[:2] == ["api", "projects"] and parts[3] == "remerge":
                if not self.same_origin_request():
                    self.json(403, {"error": "请从当前工作台重新拼接视频"})
                    return
                state = read()
                project = state["projects"].get(parts[2])
                if not project:
                    self.send_error(404)
                    return
                eligible = [run for run in state["runs"].values()
                            if run.get("project_id") == parts[2] and run.get("status") == "completed"
                            and len(run.get("results", [])) >= 2
                            and all(item.get("file") and (OUTPUTS / item["file"]).is_file()
                                    for item in run["results"])]
                if not eligible:
                    raise ValueError("当前项目没有可重新拼接的完整分段任务")
                latest = max(eligible, key=lambda item: item.get("finished_at") or item.get("created_at") or 0)
                self.json(200, remerge_run(latest["id"], project))
                return
            if len(parts) == 4 and parts[:2] == ["api", "projects"] and parts[3] == "run":
                project = read()["projects"].get(parts[2])
                if not project:
                    self.send_error(404)
                    return
                body = self.body_json()
                selected = body.get("shot_ids") or [shot["id"] for shot in project["shots"] if shot.get("selected", True)]
                valid = {shot["id"] for shot in project["shots"]}
                if not selected or any(item not in valid for item in selected):
                    raise ValueError("请至少选择一个有效镜头")
                assets = read()["assets"]
                foreign_assets = [asset_id for asset_id in project_asset_ids(project)
                                  if asset_id in assets and assets[asset_id].get("project_id") != parts[2]]
                if foreign_assets:
                    raise ValueError("项目引用了其他项目的素材，请刷新项目素材库")
                for shot in project["shots"]:
                    if shot["id"] not in selected:
                        continue
                    preview = prepare_shot(shot, project)
                    ids = [value for key in ("first_image", "last_image", "source_video") if (value := preview.get(key))]
                    for key in ("ref_images", "ref_videos", "ref_audios"):
                        ids.extend(preview.get(key) or [])
                    ids.extend(guide.get("image_id") for guide in preview.get("h3_guides", [])
                               if isinstance(guide, dict) and guide.get("image_id"))
                    missing = [asset_id for asset_id in ids if asset_id not in assets]
                    if missing:
                        raise ValueError(f"镜头 {shot['name']} 引用了不存在的素材：{missing[0]}")
                    media = {asset_id: assets[asset_id]["file"] for asset_id in ids}
                    try:
                        graph = build_h3_graph(preview, project, media, "preflight")
                    except (ValueError, KeyError) as error:
                        raise ValueError(f"镜头 {shot['name']}：{error}") from error
                    for node in graph.values():
                        if node["class_type"] == "LoraLoaderModelOnly":
                            lora_name = node["inputs"]["lora_name"]
                            if not (COMFY_DIR / "models" / "loras" / lora_name).is_file():
                                raise ValueError(f"镜头 {shot['name']}：缺少 H3 LoRA：{lora_name}")
                    if project.get("generation_speed") == "goldfish":
                        from engine import GOLDFISH_MODELS, GOLDFISH_CLIP, GOLDFISH_VAE, GOLDFISH_UPSCALER
                        goldfish_family = "creative" if shot.get("mode", "t2v") in ("t2v", "i2v", "fl2v") else "reference"
                        required = (("H3 主模型", COMFY_DIR / "models" / "diffusion_models" / GOLDFISH_MODELS[goldfish_family]),
                                    ("文本编码器", COMFY_DIR / "models" / "text_encoders" / GOLDFISH_CLIP),
                                    ("视频 VAE", COMFY_DIR / "models" / "vae" / GOLDFISH_VAE),
                                    ("H3 latent 放大器", COMFY_DIR / "models" / "latent_upscale_models" / GOLDFISH_UPSCALER))
                        missing_goldfish = [f"{label}：{path.name}" for label, path in required if not path.is_file()]
                        if missing_goldfish:
                            raise ValueError("金鱼高清流程缺少模型：" + "；".join(missing_goldfish))
                        try:
                            selflift_ready = "SelfLiftH3Sampler" in api("/object_info/SelfLiftH3Sampler")
                        except Exception:
                            selflift_ready = False
                        if not selflift_ready:
                            raise ValueError("金鱼高清流程需要 SelfLift H3 节点；请在安装节点后重启 ComfyUI")
                if project.get("semantic_adapter"):
                    adapter = project["semantic_adapter"]
                    if Path(adapter).name != adapter or not (COMFY_DIR / "models" / "semantic_bridge" / adapter).is_file():
                        raise ValueError("语义桥适配器文件不存在")
                if project.get("generation_speed") != "goldfish" and (project.get("selflift") or project.get("refine_mode") in ("latent", "latent_only")):
                    model_name = project.get("latent_model", "")
                    if Path(model_name).name != model_name or not (COMFY_DIR / "models" / "latent_upscale_models" / model_name).is_file():
                        raise ValueError("H3 3D latent 权重文件不存在")
                vae_name = ("minimax_h3_video_vae_fp16.safetensors" if project.get("generation_speed") == "goldfish"
                            else str(project.get("h3_video_vae") or "minimax_h3_video_vae_fp16.safetensors"))
                if Path(vae_name).name != vae_name or not (COMFY_DIR / "models" / "vae" / vae_name).is_file():
                    raise ValueError("所选 H3 视频 VAE 文件不存在，请在项目设置中重新选择")
                attention_mode = str(project.get("attention_mode") or ("sage" if project.get("sage_attention") else "off"))
                if attention_mode == "sage":
                    try:
                        sage_ready = "MiniMaxH3MemoryEfficientSageAttentionPatch" in api("/object_info/MiniMaxH3MemoryEfficientSageAttentionPatch")
                    except Exception:
                        sage_ready = False
                    if not sage_ready:
                        raise ValueError("当前 ComfyUI 未加载 H3 SageAttention 节点；请安装/启用 KJNodes 后重启 ComfyUI")
                if attention_mode == "veda":
                    try:
                        veda_ready = "VedaSparseAttention" in api("/object_info/VedaSparseAttention")
                    except Exception:
                        veda_ready = False
                    predictor = "minimax_h3_t2va_veda_8nfe_600step_preview_fp8.safetensors"
                    if not veda_ready or not (COMFY_DIR / "models" / "veda" / predictor).is_file():
                        raise ValueError("Veda 节点或 predictor 权重不可用；请安装 Veda 插件并放入对应模型后再选用")
                run_id = uid()
                run = {"id": run_id, "project_id": parts[2], "shot_ids": selected,
                       "status": "queued", "current": "", "results": [], "created_at": time.time()}
                change(lambda data: data["runs"].__setitem__(run_id, run))
                threading.Thread(target=run_project, args=(run_id,), daemon=True).start()
                self.json(202, run)
                return
            if len(parts) == 4 and parts[:2] == ["api", "runs"] and parts[3] == "decision":
                action = self.body_json().get("action")
                def decide(data):
                    run = data["runs"][parts[2]]
                    if action in ("approve", "skip") and run["status"] != "awaiting_upscale":
                        raise ValueError("当前任务不在等待超分确认")
                    if action == "approve":
                        run["upscale_approved"] = True
                    elif action == "skip":
                        run["skip_upscale"] = True
                    elif action == "cancel":
                        run["cancel_requested"] = True
                    else:
                        raise ValueError("未知任务操作")
                    return run
                updated = change(decide)
                if action == "cancel" and updated.get("prompt_id"):
                    prompt_id = str(updated["prompt_id"])
                    try:
                        api("/api/jobs/" + urllib.parse.quote(prompt_id, safe="") + "/cancel", {})
                    except Exception:
                        # Older builds can at least dequeue this exact prompt. Do not
                        # call /interrupt without guaranteed prompt-id targeting.
                        try:
                            api("/queue", {"delete": [prompt_id]})
                        except Exception:
                            pass
                self.json(200, updated)
                return
            if len(parts) == 4 and parts[:2] == ["api", "runs"] and parts[3] == "resume":
                run_id = parts[2]
                state = read()
                run = state["runs"].get(run_id)
                if not run:
                    self.send_error(404)
                    return
                if run.get("status") not in ("error", "cancelled", "interrupted"):
                    raise ValueError("只有失败、取消或中断的任务可以续跑")
                def resume(data):
                    item = data["runs"][run_id]
                    item.update(status="queued", error="", current="", cancel_requested=False,
                                upscale_approved=False, skip_upscale=False, resumed_at=time.time())
                change(resume)
                threading.Thread(target=run_project, args=(run_id,), daemon=True).start()
                self.json(202, read()["runs"][run_id])
                return
            self.send_error(404)
        except (ValueError, KeyError, TypeError, zipfile.BadZipFile) as error:
            self.json(400, {"error": str(error)})
        except (OSError, urllib.error.URLError, RuntimeError, subprocess.SubprocessError) as error:
            if path in ("/api/enhance", "/api/enhance-sequence"):
                self.log_error("Prompt optimization failed (%s): %s", path, error)
            self.json(502, {"error": str(error)})

    def do_PUT(self) -> None:
        path = urllib.parse.urlsplit(self.path).path
        if not self.same_origin_request():
            self.json(403, {"error": "请从当前 H3 Studio 工作台执行修改操作"})
            return
        parts = path.strip("/").split("/")
        if len(parts) != 3 or parts[:2] != ["api", "projects"]:
            self.send_error(404)
            return
        try:
            project = self.body_json()
            if project.get("id") != parts[2]:
                raise ValueError("项目 ID 不匹配")
            project["width"] = align_dimension(project.get("width"), 1344)
            project["height"] = align_dimension(project.get("height"), 768)
            if project.get("memory_policy", "manual") not in ("manual", "auto_vram"):
                raise ValueError("显存画布模式无效")
            if project.get("continuous_render_mode", "auto") not in ("auto", "segmented"):
                raise ValueError("一镜到底生成方式无效")
            if project.get("seam_mode", "auto") not in ("auto", "conservative", "manual"):
                raise ValueError("拼接模式无效")
            try:
                prefix_frames = int(project.get("seam_prefix_frames", 3))
                motion_frames = int(project.get("seam_motion_frames", 3))
            except (TypeError, ValueError):
                raise ValueError("拼接帧数无效")
            if prefix_frames not in (0, 1, 2, 3) or motion_frames not in (0, 2, 3):
                raise ValueError("拼接帧数超出可选范围")
            if not isinstance(project.get("seam_exposure", True), bool):
                raise ValueError("亮度匹配设置无效")
            if not isinstance(project.get("shots"), list):
                raise ValueError("缺少镜头列表")
            for shot in project["shots"]:
                if shot.get("mode") not in ALL_MODES or not shot.get("id"):
                    raise ValueError("镜头模式或 ID 无效")
                if shot.get("prompt_language", "en") not in {"en", "zh"}:
                    raise ValueError("提示词语言只能选择中文或英文")
            characters = project.get("characters", [])
            if not isinstance(characters, list) or any(not isinstance(character, dict) for character in characters):
                raise ValueError("角色档案列表格式无效")
            entities = project.get("entities", [])
            if not isinstance(entities, list) or any(not isinstance(entity, dict) for entity in entities):
                raise ValueError("场景与道具档案列表格式无效")
            entity_ids = [entity.get("id") for entity in entities]
            if any(not item for item in entity_ids) or len(set(entity_ids)) != len(entity_ids):
                raise ValueError("场景与道具档案 ID 无效或重复")
            asset_catalog = read()["assets"]
            image_descriptions = project.get("image_descriptions") or {}
            if not isinstance(image_descriptions, dict) or any(
                not isinstance(description, str) or len(description) > 600
                for description in image_descriptions.values()
            ):
                raise ValueError("图片画面说明无效，请检查素材是否仍存在，且每项不超过 600 字")
            for entity in entities:
                if entity.get("kind", "prop") not in ("prop", "place", "style"):
                    raise ValueError("档案类型只能是道具、地点或画风")
                if entity.get("generation_mode", "t2i") not in ("t2i", "i2i"):
                    raise ValueError("档案生图方式无效")
                if not isinstance(entity.get("image_assets", []), list):
                    raise ValueError("档案参考图片列表格式无效")
                if entity.get("generation_source"):
                    source = asset_catalog.get(entity["generation_source"])
                    if not source or source.get("kind") != "image" or source.get("project_id") != parts[2]:
                        raise ValueError("档案图生图参考图片无效")
                for asset_id in entity.get("image_assets", []):
                    if asset_id not in asset_catalog or asset_catalog[asset_id].get("kind") != "image":
                        raise ValueError(f"档案“{entity.get('name', '')}”引用了不存在的图片素材")
            scenes = project.get("scenes", [])
            if not isinstance(scenes, list) or any(not isinstance(scene, dict) for scene in scenes):
                raise ValueError("场景分组列表格式无效")
            scene_ids = [scene.get("id") for scene in scenes]
            if any(not item for item in scene_ids) or len(set(scene_ids)) != len(scene_ids):
                raise ValueError("场景分组 ID 无效或重复")
            character_ids = [character.get("id") for character in characters]
            if any(not item for item in character_ids) or len(set(character_ids)) != len(character_ids):
                raise ValueError("角色档案 ID 无效或重复")
            profile_ids = set(character_ids)
            state = read()
            previous_project = state["projects"].get(parts[2], {})
            try:
                expected_version = int(project.get("_data_version", 0))
                current_version = int(previous_project.get("_data_version", 0))
            except (ValueError, TypeError):
                raise ValueError("项目版本无效，请刷新工作台后重试")
            if expected_version != current_version:
                raise ValueError("项目已在其他页面更新，请刷新工作台以免覆盖新内容")
            try:
                next_subject_number = max(int(project.get("next_subject_number", 1)),
                                          int(previous_project.get("next_subject_number", 1)))
            except (ValueError, TypeError):
                next_subject_number = int(previous_project.get("next_subject_number", 1) or 1)
            used_subject_numbers = set()
            previous_subject_numbers = {item.get("id"): int(item.get("subject_number", index + 1))
                                         for index, item in enumerate(previous_project.get("characters", []))}
            for character in characters:
                if character.get("id") in previous_subject_numbers:
                    subject_number = previous_subject_numbers[character["id"]]
                else:
                    while next_subject_number in used_subject_numbers:
                        next_subject_number += 1
                    subject_number = next_subject_number
                    next_subject_number += 1
                character["subject_number"] = subject_number
                used_subject_numbers.add(subject_number)
            project["next_subject_number"] = max(next_subject_number, max(used_subject_numbers, default=0) + 1)
            project["_data_version"] = current_version + 1
            asset_catalog = state["assets"]
            foreign_assets = [asset_id for asset_id in project_asset_ids(project)
                              if asset_id in asset_catalog
                              and asset_catalog[asset_id].get("project_id") != parts[2]]
            if foreign_assets:
                raise ValueError("项目不能引用其他项目的素材")
            for character in characters:
                speaker_id = str(character.get("speaker_id") or "S1")
                if not re.fullmatch(r"S\d+", speaker_id):
                    raise ValueError(f"角色“{character.get('name','')}”的说话人编号需为 S1、S2 这样的格式")
                if character.get("portrait_asset") and asset_catalog.get(character["portrait_asset"], {}).get("kind") != "image":
                    raise ValueError(f"角色“{character.get('name','')}”的外观素材必须是图片")
                if character.get("multiview_asset") and asset_catalog.get(character["multiview_asset"], {}).get("kind") != "image":
                    raise ValueError(f"角色“{character.get('name','')}”的多视图素材必须是图片")
                if character.get("voice_asset") and asset_catalog.get(character["voice_asset"], {}).get("kind") != "audio":
                    raise ValueError(f"角色“{character.get('name','')}”的声音素材必须是音频")
                if character.get("multiview_layout", "four") not in ("four", "five", "six"):
                    raise ValueError(f"角色“{character.get('name','')}”的多视图布局无效")
            for shot in project["shots"]:
                assigned_characters = shot.get("character_ids", [])
                if not isinstance(assigned_characters, list):
                    raise ValueError(f"镜头“{shot.get('name','')}”的角色档案格式无效")
                if any(item not in profile_ids for item in assigned_characters):
                    raise ValueError(f"镜头“{shot.get('name','')}”引用了不存在的角色档案")
                assigned_entities = shot.get("entity_ids", [])
                if not isinstance(assigned_entities, list) or any(item not in set(entity_ids) for item in assigned_entities):
                    raise ValueError(f"镜头“{shot.get('name','')}”引用了不存在的地点或道具档案")
                if shot.get("scene_id") and shot["scene_id"] not in set(scene_ids):
                    raise ValueError(f"镜头“{shot.get('name','')}”引用了不存在的场景分组")
            def save(data):
                if parts[2] not in data["projects"]:
                    raise KeyError("项目不存在")
                if int(data["projects"][parts[2]].get("_data_version", 0)) != expected_version:
                    raise ValueError("项目已在其他页面更新，请刷新工作台以免覆盖新内容")
                live_images = {asset_id for asset_id, asset in data["assets"].items()
                               if asset.get("kind") == "image"}
                previous_descriptions = data["projects"][parts[2]].get("image_descriptions") or {}
                project["image_descriptions"] = {
                    **{asset_id: description for asset_id, description in previous_descriptions.items()
                       if asset_id in live_images and isinstance(description, str) and len(description) <= 600},
                    **{asset_id: description for asset_id, description in image_descriptions.items()
                       if asset_id in live_images},
                }
                active_image_ids = [asset_id for asset_id, asset in data["assets"].items()
                                    if asset.get("kind") == "image" and asset.get("project_id") == parts[2]]
                renumber_project_image_references(project, active_image_ids)
                project["_data_version"] = expected_version + 1
                data["projects"][parts[2]] = project
            change(save)
            self.json(200, project)
        except (ValueError, KeyError, TypeError) as error:
            self.json(400, {"error": str(error)})

    def do_DELETE(self) -> None:
        if not self.same_origin_request():
            self.json(403, {"error": "请从当前 H3 Studio 工作台执行修改操作"})
            return
        parts = urllib.parse.urlsplit(self.path).path.strip("/").split("/")
        if parts == ["api", "ai", "settings"]:
            if not self.same_origin_request():
                self.json(403, {"error": "请从工作台页面管理云端 AI 配置"})
                return
            remove_saved_deepseek_key()
            source = deepseek_key_source()
            self.json(200, {"provider": "DeepSeek", "model": DEEPSEEK_MODEL,
                            "configured": source != "none", "key_source": source})
            return
        if parts == ["api", "ai", "mimo", "settings"]:
            remove_saved_mimo_key()
            source = mimo_key_source()
            self.json(200, {"provider": "小米 MiMo", "model": MIMO_MODEL,
                            "configured": source != "none", "key_source": source})
            return
        if len(parts) == 3 and parts[:2] == ["api", "skills"]:
            if not self.same_origin_request():
                self.json(403, {"error": "请从工作台管理提示词 Skill"})
                return
            skill_id = parts[2]
            def reset_skill(data):
                for project in data["projects"].values():
                    if project.get("prompt_skill") == skill_id:
                        project["prompt_skill"] = DEFAULT_SKILL
            change(reset_skill)
            delete_skill(skill_id)
            self.json(200, {"skills": [{key: value for key, value in item.items() if key != "path"}
                                       for item in list_skills()]})
            return
        if len(parts) == 3 and parts[:2] == ["api", "assets"]:
            asset_id = parts[2]
            state = read()
            asset = state["assets"].get(asset_id)
            if not asset:
                self.send_error(404)
                return
            requested_project = urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query).get("project_id", [""])[0]
            if not requested_project or asset.get("project_id") != requested_project:
                self.json(403, {"error": "只能删除当前项目的素材"})
                return
            for run in state["runs"].values():
                if run.get("status") not in ("queued", "running", "awaiting_upscale"):
                    continue
                project = state["projects"].get(run.get("project_id"), {})
                selected = set(run.get("shot_ids") or [])
                selected_characters = {character_id for shot in project.get("shots", [])
                                       if shot.get("id") in selected for character_id in shot.get("character_ids", [])}
                selected_entities = {entity_id for shot in project.get("shots", [])
                                     if shot.get("id") in selected for entity_id in shot.get("entity_ids", [])}
                for character in project.get("characters", []):
                    if character.get("id") in selected_characters and asset_id in (character.get("portrait_asset"), character.get("multiview_asset"), character.get("voice_asset")):
                        self.json(409, {"error": "素材正被生成任务使用，请任务结束后再删除"})
                        return
                if any(entity.get("id") in selected_entities and asset_id in (entity.get("image_assets") or [])
                       for entity in project.get("entities", [])):
                    self.json(409, {"error": "素材正被生成任务使用，请任务结束后再删除"})
                    return
                for shot in project.get("shots", []):
                    if shot.get("id") in selected and asset_id in {
                        shot.get("first_image"), shot.get("last_image"), shot.get("source_video"),
                        *(shot.get("ref_images") or []), *(shot.get("ref_videos") or []), *(shot.get("ref_audios") or []),
                        *(guide.get("image_id") for guide in shot.get("h3_guides", [])
                          if isinstance(guide, dict) and guide.get("image_id")),
                        *(project.get("shared_refs", {}).get("images") or []),
                        *(project.get("shared_refs", {}).get("videos") or []),
                        *(project.get("shared_refs", {}).get("audios") or []),
                    }:
                        self.json(409, {"error": "素材正被生成任务使用，请任务结束后再删除"})
                        return
            def remove_asset(data):
                removed = data["assets"].pop(asset_id, None)
                references = 0
                if removed and removed.get("kind") == "image":
                    active_image_ids = [key for key, value in data["assets"].items()
                                        if value.get("kind") == "image"
                                        and value.get("project_id") == requested_project]
                    renumber_project_image_references(data["projects"][requested_project], active_image_ids)
                for project_id, project in data["projects"].items():
                    if project_id != requested_project:
                        continue
                    (project.get("image_descriptions") or {}).pop(asset_id, None)
                    for shot in project.get("shots", []):
                        for key in ("first_image", "last_image", "source_video"):
                            if shot.get(key) == asset_id:
                                shot[key] = None
                                references += 1
                        for key in ("ref_images", "manual_ref_images", "ref_videos", "ref_audios"):
                            before = shot.get(key) or []
                            after = [value for value in before if value != asset_id]
                            references += len(before) - len(after)
                            shot[key] = after
                        before_guides = shot.get("h3_guides") or []
                        after_guides = [guide for guide in before_guides
                                        if not isinstance(guide, dict) or guide.get("image_id") != asset_id]
                        references += len(before_guides) - len(after_guides)
                        shot["h3_guides"] = after_guides
                    for values in project.get("shared_refs", {}).values():
                        while asset_id in values:
                            values.remove(asset_id)
                            references += 1
                    for character in project.get("characters", []):
                        for key in ("portrait_asset", "multiview_asset", "voice_asset"):
                            if character.get(key) == asset_id:
                                character[key] = None
                                references += 1
                    for entity in project.get("entities", []):
                        before = entity.get("image_assets") or []
                        entity["image_assets"] = [value for value in before if value != asset_id]
                        references += len(before) - len(entity["image_assets"])
                        if entity.get("generation_source") == asset_id:
                            entity["generation_source"] = ""
                            references += 1
                    project["_data_version"] = int(project.get("_data_version", 0)) + 1
                return removed, references
            removed, references = change(remove_asset)
            if removed:
                (ASSETS / Path(removed["file"]).name).unlink(missing_ok=True)
            self.json(200, {"deleted": bool(removed), "references_removed": references})
            return
        if len(parts) != 3 or parts[:2] != ["api", "projects"]:
            self.send_error(404)
            return
        project_id = parts[2]
        state = read()
        if project_id not in state["projects"]:
            self.send_error(404)
            return
        if any(run.get("project_id") == project_id and
               run.get("status") in ("queued", "running", "awaiting_upscale")
               for run in state["runs"].values()):
            self.json(409, {"error": "该项目仍有生成任务，请等待完成或取消后再删除"})
            return
        def remove(data):
            return data["projects"].pop(project_id, None)
        removed = change(remove)
        self.json(200, {"deleted": bool(removed)})


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

"""应用层共享助手：AI 状态、素材标注、参考上下文、多视图任务与项目批量删除。

这些函数被路由模块与测试共同使用；HTTP 层不在这里。
"""
from __future__ import annotations

from prompt_tools import DEEPSEEK_MODEL
from store import change
from ai_config import deepseek_key
from ai_config import deepseek_key_source
from prompt_tools import deepseek_last_completion
from multiview import generate as generate_multiview
import hashlib
import json
import threading
from store import read
import time
from store import uid
import urllib.parse
import urllib.request
import urllib.error

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

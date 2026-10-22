"""素材上传、音轨提取与素材删除。"""
from __future__ import annotations

from server.router import Route

from pathlib import Path
from reference_numbering import renumber_project_image_references
from server.config import EXTENSIONS
from server.support import assets_with_character_names
from store import ASSETS
from store import change
from store import read
from store import uid
from video_tools import extract_audio_track
import time
import urllib.parse
import urllib.request
import urllib.error

def get_api_assets(req, path, parts, url):
    """原 app.py 分支：`path == "/api/assets"`。"""
    state = read()
    requested_project = urllib.parse.parse_qs(url.query).get("project_id", [""])[0]
    if requested_project and requested_project not in state["projects"]:
        req.send_error(404)
        return
    req.json(200, assets_with_character_names(state, requested_project))
    return


def post_api_assets_upload(req, path, parts, url):
    """原 app.py 分支：`path == "/api/assets"`。"""
    params = urllib.parse.parse_qs(urllib.parse.urlsplit(req.path).query)
    project_id = (params.get("project_id") or [""])[0]
    if project_id not in read()["projects"]:
        raise ValueError("请先选择有效项目，再上传素材")
    name = Path((params.get("name") or [""])[0]).name
    suffix = Path(name).suffix.lower()
    if suffix not in EXTENSIONS:
        raise ValueError("只支持图片、视频和音频文件")
    size = int(req.headers.get("Content-Length", "0"))
    if size < 1 or size > 4_000_000_000:
        raise ValueError("文件大小无效或超过 4GB")
    asset_id = uid()
    filename = asset_id + suffix
    target = ASSETS / filename
    remaining = size
    with target.open("wb") as output:
        while remaining:
            block = req.rfile.read(min(1024 * 1024, remaining))
            if not block:
                raise ValueError("文件上传中断")
            output.write(block)
            remaining -= len(block)
    asset = {"id": asset_id, "name": name, "file": filename,
             "kind": EXTENSIONS[suffix], "size": size, "created_at": time.time(),
             "project_id": project_id}

    def add_asset(data):
        data["assets"][asset_id] = asset
        project = data["projects"].get(project_id)
        if project and asset["kind"] == "image":
            # A new image has to enter the project's <Picture N> numbering right
            # away.  Without this it stays unreferenced until the project is
            # saved again, and the prompt cannot name the new material.
            active_image_ids = [key for key, value in data["assets"].items()
                                if value.get("kind") == "image" and value.get("project_id") == project_id]
            renumber_project_image_references(project, active_image_ids)

    change(add_asset)
    req.json(201, asset)
    return


def post_api_assets_extract_audio(req, path, parts, url):
    """原 app.py 分支：`path.startswith("/api/assets/") and path.endswith("/extract-audio")`。"""
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
    req.json(201, audio_asset)
    return


def _asset_in_use(state: dict, asset_id: str) -> bool:
    """True while a queued/running job still needs the asset."""
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
                return True
        if any(entity.get("id") in selected_entities and asset_id in (entity.get("image_assets") or [])
               for entity in project.get("entities", [])):
            return True
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
                return True
    return False


def _remove_asset(data: dict, asset_id: str, requested_project: str) -> tuple:
    """Drop one asset and every reference to it.  Returns (removed, references)."""
    removed = data["assets"].pop(asset_id, None)
    references = 0
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


def _renumber_after_image_removal(data: dict, requested_project: str) -> None:
    """<Picture N> numbers must be rebuilt once, after every removal."""
    active_image_ids = [key for key, value in data["assets"].items()
                        if value.get("kind") == "image"
                        and value.get("project_id") == requested_project]
    renumber_project_image_references(data["projects"][requested_project], active_image_ids)


def delete_api_assets(req, path, parts, url):
    """原 app.py 分支：`len(parts) == 3 and parts[:2] == ["api", "assets"]`。"""
    asset_id = parts[2]
    state = read()
    asset = state["assets"].get(asset_id)
    if not asset:
        req.send_error(404)
        return
    requested_project = urllib.parse.parse_qs(urllib.parse.urlsplit(req.path).query).get("project_id", [""])[0]
    if not requested_project or asset.get("project_id") != requested_project:
        req.json(403, {"error": "只能删除当前项目的素材"})
        return
    if _asset_in_use(state, asset_id):
        req.json(409, {"error": "素材正被生成任务使用，请任务结束后再删除"})
        return
    def remove_asset(data):
        removed, references = _remove_asset(data, asset_id, requested_project)
        if removed and removed.get("kind") == "image":
            _renumber_after_image_removal(data, requested_project)
        return removed, references
    removed, references = change(remove_asset)
    if removed:
        (ASSETS / Path(removed["file"]).name).unlink(missing_ok=True)
    req.json(200, {"deleted": bool(removed), "references_removed": references})
    return


def post_api_assets_batch_delete(req, path, parts, url):
    """批量删除当前项目的多个素材。"""
    body = req.body_json()
    requested_project = str(body.get("project_id") or "")
    asset_ids = body.get("asset_ids")
    if not isinstance(asset_ids, list) or not asset_ids:
        raise ValueError("请选择要删除的素材")
    state = read()
    if requested_project not in state["projects"]:
        raise ValueError("项目不存在")
    wanted = []
    for value in asset_ids:
        asset_id = str(value)
        asset = state["assets"].get(asset_id)
        if not asset or asset.get("project_id") != requested_project:
            req.json(403, {"error": "只能删除当前项目的素材"})
            return
        if asset_id not in wanted:
            wanted.append(asset_id)
    for asset_id in wanted:
        if _asset_in_use(state, asset_id):
            req.json(409, {"error": "素材正被生成任务使用，请任务结束后再删除"})
            return
    def remove_all(data):
        files, references = [], 0
        for asset_id in wanted:
            removed, count = _remove_asset(data, asset_id, requested_project)
            if removed:
                files.append(removed.get("file"))
            references += count
        # One renumber for the whole batch: the <Picture N> numbers have to
        # match the assets that are left, not each intermediate state.
        _renumber_after_image_removal(data, requested_project)
        return files, references
    files, references = change(remove_all)
    for filename in files:
        if filename:
            (ASSETS / Path(filename).name).unlink(missing_ok=True)
    req.json(200, {"deleted": len(files), "references_removed": references})
    return


ROUTES = [
    Route('GET', 'get_api_assets',
         lambda path, parts, url: (path == "/api/assets"),
         get_api_assets),
    Route('POST', 'post_api_assets_upload',
         lambda path, parts, url: (path == "/api/assets"),
         post_api_assets_upload),
    Route('POST', 'post_api_assets_extract_audio',
         lambda path, parts, url: (path.startswith("/api/assets/") and path.endswith("/extract-audio")),
         post_api_assets_extract_audio),
    Route('POST', 'post_api_assets_batch_delete',
         lambda path, parts, url: (path == "/api/assets/batch-delete"),
         post_api_assets_batch_delete),
    Route('DELETE', 'delete_api_assets',
         lambda path, parts, url: (len(parts) == 3 and parts[:2] == ["api", "assets"]),
         delete_api_assets),
]

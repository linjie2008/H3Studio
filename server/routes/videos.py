"""生成视频库的列表、导入与删除。"""
from __future__ import annotations

from server.router import Route

from pathlib import Path
from server.config import EXTENSIONS
from store import OUTPUTS
from store import change
from store import read
from store import uid
import time
import urllib.parse
import urllib.request
import urllib.error

def get_api_videos(req, path, parts, url):
    """原 app.py 分支：`path == "/api/videos"`。"""
    state = read()
    requested_project = urllib.parse.parse_qs(url.query).get("project_id", [""])[0]
    if requested_project and requested_project not in state["projects"]:
        req.send_error(404)
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
    req.json(200, entries)
    return


def post_api_videos_import(req, path, parts, url):
    """原 app.py 分支：`path == "/api/videos/import"`。"""
    if not req.same_origin_request():
        req.json(403, {"error": "请从工作台导入视频"})
        return
    params = urllib.parse.parse_qs(urllib.parse.urlsplit(req.path).query)
    project_id = (params.get("project_id") or [""])[0]
    if project_id not in read()["projects"]:
        raise ValueError("请先在视频库选择导入目标项目")
    name = Path((params.get("name") or [""])[0]).name
    suffix = Path(name).suffix.lower()
    if EXTENSIONS.get(suffix) != "video":
        raise ValueError("请选择 MP4、MOV、WebM、MKV、AVI 或 M4V 视频")
    size = int(req.headers.get("Content-Length", "0"))
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
                block = req.rfile.read(min(1024 * 1024, remaining))
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
    req.json(201, {"run_id": run_id, "project_id": project_id, "file": relative})
    return


def post_api_videos_delete(req, path, parts, url):
    """原 app.py 分支：`path == "/api/videos/delete"`。"""
    if not req.same_origin_request():
        req.json(403, {"error": "请从工作台管理生成视频"})
        return
    body = req.body_json()
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
    req.json(200, {"deleted": [relative for _, relative, _ in validated],
                    "count": len(validated), "files_removed": len(removed_files),
                    "shared_files_preserved": len(preserved_shared_files)})
    return


ROUTES = [
    Route('GET', 'get_api_videos',
         lambda path, parts, url: (path == "/api/videos"),
         get_api_videos),
    Route('POST', 'post_api_videos_import',
         lambda path, parts, url: (path == "/api/videos/import"),
         post_api_videos_import),
    Route('POST', 'post_api_videos_delete',
         lambda path, parts, url: (path == "/api/videos/delete"),
         post_api_videos_delete),
]

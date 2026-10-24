"""角色多视图与档案生图的排队、取消和查询。"""
from __future__ import annotations

from server.router import Route

from multiview import generate as generate_multiview
from profile_images import generate as generate_profile_image
from server.support import generate_multiview_batch
from server.support import make_multiview_job
from store import change
from store import read
from store import reserve_runs
from store import uid
import threading
import time
import urllib.parse
import urllib.request
import urllib.error

def get_api_profile_images(req, path, parts, url):
    """原 app.py 分支：`path == "/api/profile-images"`。"""
    state = read()
    project_id = urllib.parse.parse_qs(url.query).get("project_id", [""])[0]
    if project_id not in state["projects"]:
        req.send_error(404)
        return
    req.json(200, [run for run in state["runs"].values()
                    if run.get("kind") == "entity_image" and run.get("project_id") == project_id])
    return


def get_api_profile_image(req, path, parts, url):
    """原 app.py 分支：`path.startswith("/api/profile-images/")`。"""
    job_id = path.split("/")[3]
    job = read()["runs"].get(job_id)
    requested_project = urllib.parse.parse_qs(url.query).get("project_id", [""])[0]
    req.json(200, job) if (job and job.get("kind") == "entity_image"
                          and job.get("project_id") == requested_project) else req.send_error(404)
    return


def get_api_multiviews(req, path, parts, url):
    """原 app.py 分支：`path == "/api/multiview"`。"""
    state = read()
    project_id = urllib.parse.parse_qs(url.query).get("project_id", [""])[0]
    if project_id not in state["projects"]:
        req.send_error(404)
        return
    req.json(200, [run for run in state["runs"].values()
                    if run.get("kind") == "character_multiview"
                    and run.get("project_id") == project_id])
    return


def get_api_multiview(req, path, parts, url):
    """原 app.py 分支：`path.startswith("/api/multiview/")`。"""
    job_id = path.split("/")[3]
    job = read()["runs"].get(job_id)
    requested_project = urllib.parse.parse_qs(url.query).get("project_id", [""])[0]
    req.json(200, job) if (job and job.get("kind") == "character_multiview"
                          and job.get("project_id") == requested_project) else req.send_error(404)
    return


def post_api_projects_batch_multiview(req, path, parts, url):
    """原 app.py 分支：`len(parts) == 5 and parts[:2] == ["api", "projects"]
                    and parts[3:] == ["characters", "batch-multiview"]`。"""
    body = req.body_json()
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
    busy = reserve_runs(jobs)
    if busy:
        req.json(409, {"error": "请等待当前任务全部完成后再启动批量生图",
                       "active_run_id": busy.get("id")})
        return
    threading.Thread(target=generate_multiview_batch,
                     args=([job["id"] for job in jobs],), daemon=True).start()
    req.json(202, {"batch_id": batch_id, "jobs": jobs})
    return


def post_api_projects_character_multiview(req, path, parts, url):
    """原 app.py 分支：`len(parts) == 6 and parts[:2] == ["api", "projects"]
                    and parts[3] == "characters" and parts[5] == "multiview"`。"""
    body = req.body_json()
    state = read()
    if any(item.get("kind") == "character_multiview" and item.get("project_id") == parts[2]
           and item.get("batch_id") and item.get("status") in ("queued", "running")
           for item in state["runs"].values()):
        raise ValueError("当前有角色批量任务在生成，请等待批量任务完成")
    job = make_multiview_job(state, parts[2], parts[4], body)
    job_id = job["id"]
    busy = reserve_runs([job])
    if busy:
        req.json(409, {"error": "请等待当前任务全部完成后再生图",
                       "active_run_id": busy.get("id")})
        return
    threading.Thread(target=generate_multiview, args=(job_id,), daemon=True).start()
    req.json(202, job)
    return


def post_api_projects_entity_image(req, path, parts, url):
    """原 app.py 分支：`len(parts) == 6 and parts[:2] == ["api", "projects"]
                    and parts[3] == "entities" and parts[5] == "image"`。"""
    body = req.body_json()
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
    busy = reserve_runs([job])
    if busy:
        req.json(409, {"error": "请等待当前任务全部完成后再生图",
                       "active_run_id": busy.get("id")})
        return
    threading.Thread(target=generate_profile_image, args=(job_id,), daemon=True).start()
    req.json(202, job)
    return


def post_api_profile_image_cancel(req, path, parts, url):
    """原 app.py 分支：`len(parts) == 4 and parts[:2] == ["api", "profile-images"] and parts[3] == "cancel"`。"""
    job_id = parts[2]
    requested_project = str(req.body_json().get("project_id") or "")
    job = read()["runs"].get(job_id)
    if not job or job.get("kind") != "entity_image" or job.get("project_id") != requested_project:
        req.send_error(404)
        return
    if job.get("status") not in ("queued", "running"):
        raise ValueError("这个生图任务已经结束")
    change(lambda data: data["runs"][job_id].__setitem__("cancel_requested", True))
    req.json(200, {"cancel_requested": True})
    return


def post_api_multiview_cancel(req, path, parts, url):
    """原 app.py 分支：`len(parts) == 4 and parts[:2] == ["api", "multiview"] and parts[3] == "cancel"`。"""
    job_id = parts[2]
    requested_project = str(req.body_json().get("project_id") or "")
    job = read()["runs"].get(job_id)
    if not job or job.get("kind") != "character_multiview" or job.get("project_id") != requested_project:
        req.send_error(404)
        return
    if job.get("status") not in ("queued", "running"):
        raise ValueError("这个多视图任务已经结束")
    change(lambda data: data["runs"][job_id].__setitem__("cancel_requested", True))
    req.json(200, {"cancel_requested": True})
    return


ROUTES = [
    Route('GET', 'get_api_profile_images',
         lambda path, parts, url: (path == "/api/profile-images"),
         get_api_profile_images),
    Route('GET', 'get_api_profile_image',
         lambda path, parts, url: (path.startswith("/api/profile-images/")),
         get_api_profile_image),
    Route('GET', 'get_api_multiviews',
         lambda path, parts, url: (path == "/api/multiview"),
         get_api_multiviews),
    Route('GET', 'get_api_multiview',
         lambda path, parts, url: (path.startswith("/api/multiview/")),
         get_api_multiview),
    Route('POST', 'post_api_projects_batch_multiview',
         lambda path, parts, url: (len(parts) == 5 and parts[:2] == ["api", "projects"]
                    and parts[3:] == ["characters", "batch-multiview"]),
         post_api_projects_batch_multiview),
    Route('POST', 'post_api_projects_character_multiview',
         lambda path, parts, url: (len(parts) == 6 and parts[:2] == ["api", "projects"]
                    and parts[3] == "characters" and parts[5] == "multiview"),
         post_api_projects_character_multiview),
    Route('POST', 'post_api_projects_entity_image',
         lambda path, parts, url: (len(parts) == 6 and parts[:2] == ["api", "projects"]
                    and parts[3] == "entities" and parts[5] == "image"),
         post_api_projects_entity_image),
    Route('POST', 'post_api_profile_image_cancel',
         lambda path, parts, url: (len(parts) == 4 and parts[:2] == ["api", "profile-images"] and parts[3] == "cancel"),
         post_api_profile_image_cancel),
    Route('POST', 'post_api_multiview_cancel',
         lambda path, parts, url: (len(parts) == 4 and parts[:2] == ["api", "multiview"] and parts[3] == "cancel"),
         post_api_multiview_cancel),
]

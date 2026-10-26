"""生成任务的查询、报告、超分决策与续跑。"""
from __future__ import annotations

from server.router import Route

from runner import api
from runner import run_project
from runner import run_post_upscale
from runner import _seedvr2_enlarges, _video_dimensions
from store import change
from store import OUTPUTS
from store import read
from store import reserve_runs
from store import active_run
from store import uid
import threading
import time
import urllib.parse
import urllib.request
import urllib.error

def get_api_runs_report(req, path, parts, url):
    """原 app.py 分支：`path.startswith("/api/runs/") and path.endswith("/report")`。"""
    run_id = path.split("/")[3]
    run = read()["runs"].get(run_id)
    req.json(200, run) if run else req.send_error(404)
    return


def get_api_run(req, path, parts, url):
    """原 app.py 分支：`path.startswith("/api/runs/")`。"""
    run_id = path.split("/")[3]
    run = read()["runs"].get(run_id)
    req.json(200, run) if run else req.send_error(404)
    return


def get_api_runs(req, path, parts, url):
    """原 app.py 分支：`path == "/api/runs"`。"""
    req.json(200, [run for run in read()["runs"].values()
                    if run.get("kind") not in ("character_multiview", "entity_image")])
    return


def post_api_run_post_upscale(req, path, parts, url):
    """Queue SeedVR2 for a trusted completed output, without another H3 pass."""
    source_id = parts[2]
    state = read()
    source_run = state["runs"].get(source_id)
    if not source_run:
        req.send_error(404)
        return
    project = state["projects"].get(source_run.get("project_id"))
    if not project or source_run.get("status") != "completed":
        raise ValueError("请选择当前项目已完成的成片")
    source_file = source_run.get("merged") or source_run.get("preview_file")
    if not source_file and source_run.get("results"):
        source_file = source_run["results"][-1].get("file")
    if (not source_file or not (OUTPUTS / source_file).is_file()
            or not (OUTPUTS / source_file).resolve().is_relative_to(OUTPUTS.resolve())):
        raise ValueError("原成片文件不存在")
    resolution = int(project.get("upscale_resolution") or 720)
    max_resolution = int(project.get("upscale_max_resolution") or 1280)
    source_size = _video_dimensions(OUTPUTS / source_file)
    if not _seedvr2_enlarges(*source_size, resolution, max_resolution):
        raise ValueError(f"目标 {resolution}p / 最长边 {max_resolution}px 不高于原片 "
                         f"{source_size[0]}×{source_size[1]}；请先在高级设置提高超分尺寸")
    source_shot_id = ""
    if len(source_run.get("results") or []) == 1:
        source_shot_id = str(source_run["results"][0].get("shot_id") or "")
    run_id = uid()
    run = {"id": run_id, "kind": "post_upscale", "project_id": source_run["project_id"],
           "source_run_id": source_id, "source_file": source_file,
           "source_shot_id": source_shot_id, "shot_ids": [source_shot_id] if source_shot_id else [],
           "upscale_resolution": resolution,
           "upscale_max_resolution": max_resolution,
           "seed": next((int(shot.get("seed") or 42) for shot in project.get("shots", [])
                         if shot.get("id") == source_shot_id), 42),
           "status": "queued", "current": "等待处理已有成片", "results": [],
           "created_at": time.time()}
    busy = reserve_runs([run])
    if busy:
        req.json(409, {"error": "请等待当前任务全部完成后再启动下一个任务",
                       "active_run_id": busy.get("id")})
        return
    threading.Thread(target=run_post_upscale, args=(run_id,), daemon=True).start()
    req.json(202, run)
    return


def post_api_runs_decision(req, path, parts, url):
    """原 app.py 分支：`len(parts) == 4 and parts[:2] == ["api", "runs"] and parts[3] == "decision"`。"""
    action = req.body_json().get("action")
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
            if run.get("status") in ("queued", "running", "awaiting_upscale"):
                run["current"] = "正在取消任务并停止后处理"
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
    req.json(200, updated)
    return


def post_api_runs_resume(req, path, parts, url):
    """原 app.py 分支：`len(parts) == 4 and parts[:2] == ["api", "runs"] and parts[3] == "resume"`。"""
    run_id = parts[2]
    state = read()
    run = state["runs"].get(run_id)
    if not run:
        req.send_error(404)
        return
    if run.get("status") not in ("error", "cancelled", "interrupted"):
        raise ValueError("只有失败、取消或中断的任务可以续跑")
    def resume(data):
        busy = active_run(data, exclude_id=run_id)
        if busy:
            return busy
        item = data["runs"][run_id]
        item.update(status="queued", error="", current="", cancel_requested=False,
                    upscale_approved=False, skip_upscale=False, resumed_at=time.time())
        return None
    busy = change(resume)
    if busy:
        req.json(409, {"error": "请等待当前任务全部完成后再续跑",
                       "active_run_id": busy.get("id")})
        return
    worker = run_post_upscale if run.get("kind") == "post_upscale" else run_project
    threading.Thread(target=worker, args=(run_id,), daemon=True).start()
    req.json(202, read()["runs"][run_id])
    return


ROUTES = [
    Route('GET', 'get_api_runs_report',
         lambda path, parts, url: (path.startswith("/api/runs/") and path.endswith("/report")),
         get_api_runs_report),
    Route('GET', 'get_api_run',
         lambda path, parts, url: (path.startswith("/api/runs/")),
         get_api_run),
    Route('GET', 'get_api_runs',
         lambda path, parts, url: (path == "/api/runs"),
         get_api_runs),
    Route('POST', 'post_api_run_post_upscale',
         lambda path, parts, url: (len(parts) == 4 and parts[:2] == ["api", "runs"] and parts[3] == "post-upscale"),
         post_api_run_post_upscale),
    Route('POST', 'post_api_runs_decision',
         lambda path, parts, url: (len(parts) == 4 and parts[:2] == ["api", "runs"] and parts[3] == "decision"),
         post_api_runs_decision),
    Route('POST', 'post_api_runs_resume',
         lambda path, parts, url: (len(parts) == 4 and parts[:2] == ["api", "runs"] and parts[3] == "resume"),
         post_api_runs_resume),
]

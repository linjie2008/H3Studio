"""生成任务的查询、报告、超分决策与续跑。"""
from __future__ import annotations

from server.router import Route

from runner import api
from runner import run_project
from store import change
from store import read
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
        item = data["runs"][run_id]
        item.update(status="queued", error="", current="", cancel_requested=False,
                    upscale_approved=False, skip_upscale=False, resumed_at=time.time())
    change(resume)
    threading.Thread(target=run_project, args=(run_id,), daemon=True).start()
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
    Route('POST', 'post_api_runs_decision',
         lambda path, parts, url: (len(parts) == 4 and parts[:2] == ["api", "runs"] and parts[3] == "decision"),
         post_api_runs_decision),
    Route('POST', 'post_api_runs_resume',
         lambda path, parts, url: (len(parts) == 4 and parts[:2] == ["api", "runs"] and parts[3] == "resume"),
         post_api_runs_resume),
]

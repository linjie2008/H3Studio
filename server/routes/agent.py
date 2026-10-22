"""产品助手（LangGraph 受限操作）的聊天与确认端点。"""
from __future__ import annotations

import json
import urllib.parse

from server.router import Route

from ai_config import deepseek_key
from ai_config import mimo_key
from ai_config import prompt_provider
from product_agent_actions import ACTION_NAMES as PRODUCT_AGENT_ACTIONS
from product_agent_actions import confirm as confirm_product_agent_action

def get_api_product_agent_status(req, path, parts, url):
    """原 app.py 分支：`path == "/api/product-agent/status"`。"""
    selected = prompt_provider()
    req.json(200, {"framework": "LangGraph", "read_only": False,
                    "provider": selected,
                    "configured": (bool(deepseek_key()) if selected == "deepseek" else
                                   bool(mimo_key()) if selected == "mimo" else
                                   bool(deepseek_key() or mimo_key())),
                    "knowledge": ["产品功能地图", "项目说明", "开源功能对照", "当前项目摘要"],
                    "actions": list(PRODUCT_AGENT_ACTIONS), "confirmation_required": True})
    return


def post_api_product_agent_chat(req, path, parts, url):
    """原 app.py 分支：`path == "/api/product-agent/chat"`。"""
    # langgraph 是可选依赖，仅在真实调用时导入。
    from product_agent import chat as product_agent_chat
    req.json(200, product_agent_chat(req.body_json()))
    return


def post_api_product_agent_confirm(req, path, parts, url):
    """原 app.py 分支：`path == "/api/product-agent/confirm"`。"""
    body = req.body_json()
    req.json(200, confirm_product_agent_action(
        str(body.get("token") or ""), str(body.get("project_id") or "")))
    return


def post_api_creative_agent_run(req, path, parts, url):
    """创作智能体：跑一轮「想→做→看」循环。"""
    from agent_loop import run_agent
    if not req.same_origin_request():
        req.json(403, {"error": "请从工作台调用创作智能体"})
        return
    req.json(200, run_agent(req.body_json()))
    return


def _sse_open(req):
    req.send_response(200)
    req.send_header("Content-Type", "text/event-stream; charset=utf-8")
    req.send_header("Cache-Control", "no-cache, no-transform")
    req.send_header("X-Accel-Buffering", "no")
    # 这条响应必须自己结束：之前写的是 keep-alive，浏览器会一直等连接关闭，
    # 前端的读取循环因此永不退出，答案永远渲染不出来（界面看着像卡死）。
    req.send_header("Connection", "close")
    req.close_connection = True
    req.end_headers()


def _sse_write(req, payload: dict) -> None:
    req.wfile.write(("data: " + json.dumps(payload, ensure_ascii=False) + "\n\n").encode("utf-8"))
    req.wfile.flush()


def post_api_creative_agent_run_stream(req, path, parts, url):
    """创作智能体（流式）：边做边把每一步推给前端。"""
    from agent_loop import run_agent
    if not req.same_origin_request():
        req.json(403, {"error": "请从工作台调用创作智能体"})
        return
    body = req.body_json()
    _sse_open(req)
    try:
        result = run_agent(body, emit=lambda event: _sse_write(req, event))
        _sse_write(req, {"type": "result", "result": result})
    except Exception as error:
        _sse_write(req, {"type": "error", "error": str(error)})
    _sse_write(req, {"type": "done"})
    return


def post_api_creative_agent_confirm_stream(req, path, parts, url):
    """创作智能体（流式）：确认后继续执行并继续推送步骤。"""
    from agent_loop import confirm_run
    if not req.same_origin_request():
        req.json(403, {"error": "请从工作台调用创作智能体"})
        return
    body = req.body_json()
    _sse_open(req)
    try:
        result = confirm_run(str(body.get("run_id") or ""), bool(body.get("approved")),
                             emit=lambda event: _sse_write(req, event))
        _sse_write(req, {"type": "result", "result": result})
    except Exception as error:
        _sse_write(req, {"type": "error", "error": str(error)})
    _sse_write(req, {"type": "done"})
    return


def post_api_creative_agent_confirm(req, path, parts, url):
    """创作智能体：用户确认（或拒绝）挂起的删除/渲染操作后继续。"""
    from agent_loop import confirm_run
    if not req.same_origin_request():
        req.json(403, {"error": "请从工作台调用创作智能体"})
        return
    body = req.body_json()
    req.json(200, confirm_run(str(body.get("run_id") or ""), bool(body.get("approved"))))
    return


def post_api_creative_agent_resume(req, path, parts, url):
    """创作智能体：步数用完后继续执行。"""
    from agent_loop import resume_run
    if not req.same_origin_request():
        req.json(403, {"error": "请从工作台调用创作智能体"})
        return
    body = req.body_json()
    req.json(200, resume_run(str(body.get("run_id") or "")))
    return


def post_api_creative_agent_undo(req, path, parts, url):
    """创作智能体：回滚这一轮自动执行的创作类改动。"""
    from agent_audit import undo_run
    if not req.same_origin_request():
        req.json(403, {"error": "请从工作台调用创作智能体"})
        return
    body = req.body_json()
    req.json(200, undo_run(str(body.get("run_id") or ""), str(body.get("project_id") or "")))
    return


def get_api_creative_agent_status(req, path, parts, url):
    """创作智能体能力清单：工具、风险分级、确认策略。"""
    from agent_tools import TOOLS, RISK_LABEL
    req.json(200, {"tools": [{"name": spec["name"], "risk": spec["risk"],
                              "risk_label": RISK_LABEL.get(spec["risk"], spec["risk"]),
                              "summary": spec["summary"]}
                             for spec in TOOLS.values()],
                  "confirmation_required": ["destructive", "gpu"],
                  "framework": "LangGraph"})
    return


def get_api_creative_agent_runs(req, path, parts, url):
    """创作智能体最近几轮的执行记录：做了什么、每步耗时、改了几处。"""
    from agent_audit import list_runs
    query = urllib.parse.parse_qs(url.query or "")
    project_id = (query.get("project_id") or [""])[0]
    try:
        limit = int((query.get("limit") or ["20"])[0])
    except (TypeError, ValueError):
        limit = 20
    req.json(200, {"runs": list_runs(project_id, limit)})
    return


ROUTES = [
    Route('GET', 'get_api_product_agent_status',
         lambda path, parts, url: (path == "/api/product-agent/status"),
         get_api_product_agent_status),
    Route('POST', 'post_api_product_agent_chat',
         lambda path, parts, url: (path == "/api/product-agent/chat"),
         post_api_product_agent_chat),
    Route('POST', 'post_api_product_agent_confirm',
         lambda path, parts, url: (path == "/api/product-agent/confirm"),
         post_api_product_agent_confirm),
    Route('GET', 'get_api_creative_agent_status',
         lambda path, parts, url: (path == "/api/creative-agent/status"),
         get_api_creative_agent_status),
    Route('GET', 'get_api_creative_agent_runs',
         lambda path, parts, url: (path == "/api/creative-agent/runs"),
         get_api_creative_agent_runs),
    Route('POST', 'post_api_creative_agent_run',
         lambda path, parts, url: (path == "/api/creative-agent/run"),
         post_api_creative_agent_run),
    Route('POST', 'post_api_creative_agent_run_stream',
         lambda path, parts, url: (path == "/api/creative-agent/run-stream"),
         post_api_creative_agent_run_stream),
    Route('POST', 'post_api_creative_agent_confirm_stream',
         lambda path, parts, url: (path == "/api/creative-agent/confirm-stream"),
         post_api_creative_agent_confirm_stream),
    Route('POST', 'post_api_creative_agent_confirm',
         lambda path, parts, url: (path == "/api/creative-agent/confirm"),
         post_api_creative_agent_confirm),
    Route('POST', 'post_api_creative_agent_resume',
         lambda path, parts, url: (path == "/api/creative-agent/resume"),
         post_api_creative_agent_resume),
    Route('POST', 'post_api_creative_agent_undo',
         lambda path, parts, url: (path == "/api/creative-agent/undo"),
         post_api_creative_agent_undo),
]

"""产品助手（LangGraph 受限操作）的聊天与确认端点。"""
from __future__ import annotations

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
]

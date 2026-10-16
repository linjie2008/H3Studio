"""新版 Vue 工作台（web/dist）的静态托管。

- `/app` → SPA 入口 index.html（History 路由，未命中文件一律回退到入口）；
- `/app/*` → web/dist 下的构建产物。

旧版界面仍在 `/`，两版并存直到迁移完成。
"""
from __future__ import annotations

import urllib.parse
from pathlib import Path

from server.config import ROOT
from server.router import Route

DIST = ROOT / "web" / "dist"


def _serve_spa(req) -> None:
    index = DIST / "index.html"
    if not index.is_file():
        req.send_error(404, "新版界面尚未构建：cd web && npm run build")
        return
    req.file(index)


def get_app_index(req, path, parts, url):
    """新版工作台入口。"""
    _serve_spa(req)


def get_app_asset(req, path, parts, url):
    """构建产物；未命中的路径回退到 SPA 入口（Vue Router History 模式）。"""
    relative = urllib.parse.unquote(path[len("/app/"):])
    candidate = (DIST / relative).resolve()
    try:
        candidate.relative_to(DIST.resolve())
    except ValueError:
        # 路径穿越尝试：按未命中处理，回退入口而不是泄露目录信息。
        _serve_spa(req)
        return
    if candidate.is_file() and candidate != (DIST / "index.html").resolve():
        req.file(candidate)
        return
    _serve_spa(req)


ROUTES = [
    Route('GET', 'get_app_index',
          lambda path, parts, url: path == "/app" or path == "/app/",
          get_app_index),
    Route('GET', 'get_app_asset',
          lambda path, parts, url: path.startswith("/app/"),
          get_app_asset),
]

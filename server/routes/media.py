"""素材与成片文件的流式响应（含 Range）。"""
from __future__ import annotations

from server.router import Route

from store import ASSETS
from store import OUTPUTS
from store import read
import urllib.parse
import urllib.request
import urllib.error

def get_media_assets(req, path, parts, url):
    """原 app.py 分支：`path.startswith("/media/assets/")`。"""
    asset_id = path.split("/")[3]
    asset = read()["assets"].get(asset_id)
    req.file(ASSETS / asset["file"]) if asset else req.send_error(404)
    return


def get_media_outputs(req, path, parts, url):
    """原 app.py 分支：`path.startswith("/media/outputs/")`。"""
    parts = [urllib.parse.unquote(value) for value in path.split("/")[3:]]
    if len(parts) != 2 or any(not value or value in (".", "..") or "/" in value for value in parts):
        req.send_error(400)
        return
    req.file(OUTPUTS / parts[0] / parts[1])
    return


ROUTES = [
    Route('GET', 'get_media_assets',
         lambda path, parts, url: (path.startswith("/media/assets/")),
         get_media_assets),
    Route('GET', 'get_media_outputs',
         lambda path, parts, url: (path.startswith("/media/outputs/")),
         get_media_outputs),
]

"""H3 Studio 项目包导入。"""
from __future__ import annotations

from server.router import Route

from packs import import_pack
from pathlib import Path
import tempfile

def post_api_packs_import(req, path, parts, url):
    """原 app.py 分支：`path == "/api/packs/import"`。"""
    size = int(req.headers.get("Content-Length", "0"))
    if size < 1 or size > 4_000_000_000:
        raise ValueError("项目包大小无效")
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as temp:
        remaining = size
        while remaining:
            block = req.rfile.read(min(1024 * 1024, remaining))
            if not block:
                raise ValueError("项目包上传中断")
            temp.write(block)
            remaining -= len(block)
        temp_path = Path(temp.name)
    try:
        project = import_pack(temp_path)
    finally:
        temp_path.unlink(missing_ok=True)
    req.json(201, project)
    return


ROUTES = [
    Route('POST', 'post_api_packs_import',
         lambda path, parts, url: (path == "/api/packs/import"),
         post_api_packs_import),
]

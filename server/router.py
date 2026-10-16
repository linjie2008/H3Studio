"""路由注册表与分发。

Route(method, name, match, handler)：
- match(path, parts, url) 决定是否命中（求值顺序 = ORDER，first-match-wins）；
- handler(req, path, parts, url) 执行分支，req 是 HTTP Handler 实例。
"""
from __future__ import annotations

import urllib.parse
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class Route:
    method: str
    name: str
    match: Callable
    handler: Callable


def resolve(routes: list[Route], method: str, path_with_query: str):
    """返回 (route, path, parts, url)；无命中时 route 为 None。"""
    url = urllib.parse.urlsplit(path_with_query)
    path = url.path
    parts = path.strip("/").split("/")
    for route in routes:
        if route.method != method:
            continue
        if route.match(path, parts, url):
            return route, path, parts, url
    return None, path, parts, url

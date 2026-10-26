"""把「土拨鼠」项目换成《马大帅》范德彪 + 开原维多利亚广场背景。

分三步：
  1) 上传范德彪剧照原图 → 新建角色 → 生成四视图身份参考
  2) 生成维多利亚广场环境参考图
  3) 改写镜头提示词并绑定参考图
"""
from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

BASE = "http://127.0.0.1:8765"
PROJECT = "d863280edaab"


OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def call(method: str, path: str, body=None, raw=None, headers=None):
    url = BASE + path
    data = None
    hdrs = dict(headers or {})
    hdrs.setdefault("Origin", BASE)
    if raw is not None:
        data = raw
        hdrs.setdefault("Content-Type", "application/octet-stream")
    elif body is not None:
        data = json.dumps(body).encode()
        hdrs.setdefault("Content-Type", "application/json")
    request = urllib.request.Request(url, data=data, headers=hdrs, method=method)
    try:
        with OPENER.open(request, timeout=120) as response:
            payload = response.read()
            return json.loads(payload) if payload else {}
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", "replace")
        raise SystemExit(f"{method} {path} 失败 {error.code}: {detail[:500]}")


def get_project():
    return call("GET", f"/api/projects/{PROJECT}")


def put_project(project):
    return call("PUT", f"/api/projects/{PROJECT}", project)


def upload(path: str, name: str) -> dict:
    data = Path(path).read_bytes()
    suffix = Path(path).suffix.lower()
    encoded = urllib.parse.quote(name)
    return call("POST", f"/api/assets?project_id={PROJECT}&name={encoded}",
                raw=data, headers={"Content-Type": "application/octet-stream"})


def wait_job(job_id: str, timeout: float = 1800):
    deadline = time.time() + timeout
    while time.time() < deadline:
        run = call("GET", f"/api/runs/{job_id}")
        status = run.get("status")
        current = run.get("current")
        print(f"  [{status}] {current}", flush=True)
        if status in ("completed", "error", "cancelled"):
            return run
        time.sleep(5)
    raise SystemExit("任务超时")


def ensure_character(name: str, appearance: str, portrait_asset: str) -> dict:
    project = get_project()
    characters = project.setdefault("characters", [])
    for character in characters:
        if character.get("name") == name:
            character["appearance"] = appearance
            character["portrait_asset"] = portrait_asset
            character["multiview_mode"] = "i2i"
            character["multiview_engine"] = "krea2"
            character["multiview_layout"] = "four"
            character["character_form"] = "human"
            character["krea_width"] = 1536
            character["krea_height"] = 1024
            put_project(project)
            return character
    character = {
        "id": uuid.uuid4().hex[:12], "subject_number": len(characters) + 2, "name": name,
        "appearance": appearance, "voice_description": "", "speaker_id": "S1",
        "portrait_asset": portrait_asset, "multiview_asset": "", "voice_asset": "",
        "multiview_mode": "i2i", "multiview_prompt": "", "multiview_engine": "krea2",
        "multiview_layout": "four", "xb_output_lang": "zh", "xb_sampling_mode": "bypass",
        "xb_skill_mode": "自动", "xb_canvas_override": False, "xb_aspect_ratio": "16:9",
        "xb_width": 1920, "xb_height": 1088, "krea_width": 1536, "krea_height": 1024,
        "xb_seed": 0, "xb_thinking": False, "xb_preset_text": "", "xb_extra_settings": "",
        "xb_green_background": True, "character_form": "human",
    }
    characters.append(character)
    project["next_subject_number"] = len(characters) + 2
    put_project(project)
    return character


if __name__ == "__main__":
    print("仅作为模块使用")

"""提示词 Skill 的列表、导入、删除与 Skill 参数/推荐建议。"""
from __future__ import annotations

from server.router import Route

from pathlib import Path
from prompt_skills import DEFAULT_SKILL
from prompt_skills import delete_skill
from prompt_skills import import_skill
from prompt_skills import list_skills
from prompt_skills import preview_skill
from prompt_tools import recommend_project_skills
from prompt_tools import suggest_project_parameters
from runner import api
from server.config import COMFY_DIR
from store import change
from store import read
import urllib.parse
import urllib.request
import urllib.error

def get_api_skills(req, path, parts, url):
    """原 app.py 分支：`path == "/api/skills"`。"""
    req.json(200, [{key: value for key, value in skill.items() if key != "path"}
                    for skill in list_skills()])
    return


def get_api_skills_preview(req, path, parts, url):
    """原 app.py 分支：`path.startswith("/api/skills/") and path.endswith("/preview")`。"""
    skill_id = urllib.parse.unquote(path.split("/")[3])
    req.json(200, preview_skill(skill_id))
    return


def post_api_skill_agent_parameters(req, path, parts, url):
    """原 app.py 分支：`path == "/api/skill-agent/parameters"`。"""
    if not req.same_origin_request():
        req.json(403, {"error": "请从工作台调用 Skill 参数智能体"})
        return
    body = req.body_json()
    project = read()["projects"].get(str(body.get("project_id") or ""))
    if not project:
        raise ValueError("项目不存在")
    selected_skill = project.get("prompt_skill") or DEFAULT_SKILL
    if selected_skill not in {item["id"] for item in list_skills()}:
        selected_skill = DEFAULT_SKILL

    def model_names(folder):
        root = COMFY_DIR / "models" / folder
        return sorted(path.name for path in root.iterdir() if path.is_file()
                      and path.suffix.lower() in (".safetensors", ".pt", ".pth")) if root.is_dir() else []
    try:
        sage_available = "MiniMaxH3MemoryEfficientSageAttentionPatch" in api(
            "/object_info/MiniMaxH3MemoryEfficientSageAttentionPatch")
    except Exception:
        sage_available = False
    try:
        stats = api("/system_stats")
        device = (stats.get("devices") or [{}])[0]
        hardware = {key: device.get(key) for key in ("name", "vram_total", "vram_free") if key in device}
    except Exception:
        hardware = {}
    # The 4-step plan always samples with the family's distilled LoRA, so that
    # is what has to be installed.  DMAD is a style LoRA and is optional.
    four_step_names = ("minimax_h3_fl2v_lightx2v_turbo_4step_v0.1_comfy.safetensors",
                       "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors")
    turbo4_installed = all((COMFY_DIR / "models" / "loras" / "minimax_h3" / name).is_file()
                           for name in four_step_names)
    available = {"sage_attention": sage_available, "turbo4": turbo4_installed,
                 "h3_video_vaes": [name for name in model_names("vae")
                                    if name.startswith("minimax_h3_video_vae")],
                 "latent_models": model_names("latent_upscale_models")}
    result = suggest_project_parameters(selected_skill, project, available, hardware)
    result["skill_id"] = selected_skill
    req.json(200, result)
    return


def post_api_skill_agent_recommend(req, path, parts, url):
    """原 app.py 分支：`path == "/api/skill-agent/recommend"`。"""
    if not req.same_origin_request():
        req.json(403, {"error": "请从工作台调用 Skill 推荐助手"})
        return
    body = req.body_json()
    project = read()["projects"].get(str(body.get("project_id") or ""))
    if not project:
        raise ValueError("项目不存在")
    req.json(200, recommend_project_skills(project, list_skills()))
    return


def post_api_skills_import(req, path, parts, url):
    """原 app.py 分支：`path == "/api/skills/import"`。"""
    if not req.same_origin_request():
        req.json(403, {"error": "请从工作台导入提示词 Skill"})
        return
    size = int(req.headers.get("Content-Length", "0"))
    if size < 1 or size > 5_000_000:
        raise ValueError("Skill 文件为空或超过 5 MB")
    payload = req.rfile.read(size)
    if len(payload) != size:
        raise ValueError("Skill 文件上传中断")
    name = Path((urllib.parse.parse_qs(urllib.parse.urlsplit(req.path).query).get("name") or [""])[0]).name
    skill = import_skill(name, payload)
    req.json(201, {key: value for key, value in skill.items() if key != "path"})
    return


def delete_api_skills(req, path, parts, url):
    """原 app.py 分支：`len(parts) == 3 and parts[:2] == ["api", "skills"]`。"""
    if not req.same_origin_request():
        req.json(403, {"error": "请从工作台管理提示词 Skill"})
        return
    skill_id = parts[2]
    def reset_skill(data):
        for project in data["projects"].values():
            if project.get("prompt_skill") == skill_id:
                project["prompt_skill"] = DEFAULT_SKILL
    change(reset_skill)
    delete_skill(skill_id)
    req.json(200, {"skills": [{key: value for key, value in item.items() if key != "path"}
                               for item in list_skills()]})
    return


ROUTES = [
    Route('GET', 'get_api_skills',
         lambda path, parts, url: (path == "/api/skills"),
         get_api_skills),
    Route('GET', 'get_api_skills_preview',
         lambda path, parts, url: (path.startswith("/api/skills/") and path.endswith("/preview")),
         get_api_skills_preview),
    Route('POST', 'post_api_skill_agent_parameters',
         lambda path, parts, url: (path == "/api/skill-agent/parameters"),
         post_api_skill_agent_parameters),
    Route('POST', 'post_api_skill_agent_recommend',
         lambda path, parts, url: (path == "/api/skill-agent/recommend"),
         post_api_skill_agent_recommend),
    Route('POST', 'post_api_skills_import',
         lambda path, parts, url: (path == "/api/skills/import"),
         post_api_skills_import),
    Route('DELETE', 'delete_api_skills',
         lambda path, parts, url: (len(parts) == 3 and parts[:2] == ["api", "skills"]),
         delete_api_skills),
]

"""提示词优化、统筹优化、AI 分镜、生图描述与提示词预览。"""
from __future__ import annotations

from server.router import Route

from engine import compile_shot_prompt
from engine import generation_prompt
from engine import lint_shot_prompt
from image_prompt_tools import optimize_image_prompt
from prompt_skills import DEFAULT_SKILL
from prompt_skills import list_skills
from prompt_tools import enhance
from prompt_tools import enhance_global_prompt
from prompt_tools import enhance_sequence
from prompt_tools import storyboard as generate_storyboard
from runner import _shot_identity_refs
from runner import continuity_preview
from runner import prepare_shot
from server.support import enhancement_references
from store import read

def post_api_enhance_sequence(req, path, parts, url):
    """原 app.py 分支：`path == "/api/enhance-sequence"`。"""
    body = req.body_json()
    state = read()
    project_id = str(body.get("project_id") or "")
    project_data = state["projects"].get(project_id)
    if not project_data:
        raise ValueError("项目不存在，请刷新页面后重试")
    if project_data.get("continuous_take") is False:
        raise ValueError("请先开启“强一镜到底”再统筹优化")
    saved_shots = project_data.get("shots", [])
    if len(saved_shots) < 2:
        raise ValueError("至少需要两个镜头才能统筹优化")
    skill_id = str(project_data.get("prompt_skill") or DEFAULT_SKILL)
    skill_catalog = {item["id"]: item for item in list_skills()}
    if skill_id not in skill_catalog:
        skill_id = DEFAULT_SKILL
    catalog = state["assets"]
    sequence = []
    for index, saved_shot in enumerate(saved_shots):
        prepared = prepare_shot(saved_shot, project_data)
        identity_assets = set(_shot_identity_refs(saved_shot, project_data, catalog))
        if prepared.get("mode") in {"t2v", "i2v", "fl2v"}:
            identity_assets.intersection_update(filter(None, (
                prepared.get("first_image"), prepared.get("last_image"))))
        context = {"segment": {"position": index + 1, "total": len(saved_shots),
            "is_first": index == 0, "is_last": index == len(saved_shots) - 1},
            "references": [], "characters": prepared.get("_h3_characters", []),
            "entities": prepared.get("_h3_entities", []),
            "person_identity_asset_ids": list(identity_assets)}
        context["references"] = enhancement_references(
            prepared, project_data, catalog, identity_assets)
        sequence.append({"id": str(saved_shot.get("id") or index + 1),
            "name": str(saved_shot.get("name") or f"镜头{index + 1}"),
            "prompt": str(saved_shot.get("prompt") or ""), "mode": str(saved_shot.get("mode") or "t2v"),
            "duration": saved_shot.get("duration", 5),
            "camera_move": saved_shot.get("camera_move", ""),
            "camera_speed": saved_shot.get("camera_speed", ""),
            "action_intensity": saved_shot.get("action_intensity", ""),
            "dialogue_time": saved_shot.get("dialogue_time", ""),
            "dialogue_text": saved_shot.get("dialogue_text", ""),
            "output_language": str(saved_shot.get("prompt_language") or "en"), "context": context})
    sequence_brief = (project_data.get("sequence_brief") or project_data.get("script_text")
                      or project_data.get("shared_prompt") or "")
    optimization_scope = str(body.get("optimization_scope") or "全部提示词").strip()[:500]
    prompts = enhance_sequence(sequence, skill_id, True, str(sequence_brief), optimization_scope)
    req.json(200, {"prompts": [{"id": item["id"], "prompt": prompt}
        for item, prompt in zip(sequence, prompts)], "skill_id": skill_id,
        "skill_name": skill_catalog.get(skill_id, {}).get("name", skill_id)})
    return


def post_api_image_prompt(req, path, parts, url):
    """原 app.py 分支：`path == "/api/image-prompt"`。"""
    if not req.same_origin_request():
        req.json(403, {"error": "请从工作台调用 AI 生图描述"})
        return
    body = req.body_json()
    project_id = str(body.get("project_id") or "")
    project_data = read()["projects"].get(project_id)
    if not project_data:
        raise ValueError("项目不存在，请刷新页面后重试")
    result = optimize_image_prompt(project_data,
        str(body.get("target_type") or ""), str(body.get("target_id") or ""),
        str(body.get("scope") or ""))
    req.json(200, result)
    return


def post_api_enhance(req, path, parts, url):
    """原 app.py 分支：`path == "/api/enhance"`。"""
    body = req.body_json()
    reference_context = {}
    state = read()
    project_id, shot_id = str(body.get("project_id") or ""), str(body.get("shot_id") or "")
    project_data = state["projects"].get(project_id) if project_id else None
    requested_skill = str(body.get("skill_id") or "")
    skill_id = requested_skill or DEFAULT_SKILL
    if project_id:
        if not project_data:
            raise ValueError("项目不存在，请刷新页面后重试")
        if not requested_skill:
            skill_id = str(project_data.get("prompt_skill") or DEFAULT_SKILL)
    skill_catalog = {item["id"]: item for item in list_skills()}
    if skill_id not in skill_catalog:
        skill_id = DEFAULT_SKILL
    if project_id and shot_id:
        if not project_data:
            raise ValueError("项目不存在，请刷新页面后重试")
        saved_shot = next((item for item in project_data.get("shots", []) if item.get("id") == shot_id), None)
        if not saved_shot:
            raise ValueError("镜头不存在，请刷新页面后重试")
        catalog = state["assets"]
        prepared = prepare_shot(saved_shot, project_data)
        identity_assets = set(_shot_identity_refs(saved_shot, project_data, catalog))
        if prepared.get("mode") in {"t2v", "i2v", "fl2v"}:
            identity_assets.intersection_update(filter(None, (
                prepared.get("first_image"), prepared.get("last_image"))))
        saved_shots = project_data.get("shots", [])
        shot_index = next((index for index, item in enumerate(saved_shots)
                           if item.get("id") == shot_id), 0)
        def neighbor_context(index):
            if index < 0 or index >= len(saved_shots):
                return None
            item = saved_shots[index]
            # Never send another shot's full prompt to single-shot
            # optimization: its subjects and Picture tags can leak.
            return {"name": str(item.get("name") or "镜头")[:120],
                    "mode": str(item.get("mode") or "t2v")}
        reference_context = {"segment": {
            "position": shot_index + 1, "total": len(saved_shots),
            "is_first": shot_index == 0, "is_last": shot_index == len(saved_shots) - 1,
            "previous_shot": neighbor_context(shot_index - 1),
            "next_shot": neighbor_context(shot_index + 1)},
            "selected_character_ids": list(saved_shot.get("character_ids") or [])}
        catalog = state["assets"]
        references = enhancement_references(
            prepared, project_data, catalog, identity_assets)
        reference_context.update({"references": references,
            "characters": prepared.get("_h3_characters", []),
            "entities": prepared.get("_h3_entities", []),
            "person_identity_asset_ids": list(identity_assets)})
    output_language = str(body.get("output_language") or "en")
    if output_language not in {"en", "zh"}:
        raise ValueError("提示词语言只能选择中文或英文")
    optimization_scope = str(body.get("optimization_scope") or "全部提示词").strip()[:500]
    optimized = enhance(str(body.get("prompt") or ""),
        str((saved_shot.get("mode") or "t2v") if project_id and shot_id else body.get("mode") or "t2v"), skill_id,
        saved_shot.get("duration", 5) if project_id and shot_id else body.get("duration", 5),
        reference_context, output_language, optimization_scope)
    req.json(200, {"prompt": optimized, "skill_id": skill_id,
                    "output_language": output_language,
                    "skill_name": skill_catalog.get(skill_id, {}).get("name", skill_id),
                    "base_skill_id": DEFAULT_SKILL})
    return


def post_api_enhance_global(req, path, parts, url):
    """Preview AI optimization of the current project's shared prompt."""
    body = req.body_json()
    project_id = str(body.get("project_id") or "")
    project_data = read()["projects"].get(project_id)
    if not project_data:
        raise ValueError("项目不存在，请刷新页面后重试")
    prompt = str(body.get("prompt") or "")
    if prompt != str(project_data.get("shared_prompt") or ""):
        raise ValueError("全片提示词已变化，请保存后重试优化")
    skill_catalog = {item["id"]: item for item in list_skills()}
    skill_id = str(project_data.get("prompt_skill") or DEFAULT_SKILL)
    if skill_id not in skill_catalog:
        skill_id = DEFAULT_SKILL
    optimized = enhance_global_prompt(prompt, skill_id)
    req.json(200, {"prompt": optimized, "skill_id": skill_id,
                   "skill_name": skill_catalog[skill_id]["name"]})
    return


def post_api_storyboard(req, path, parts, url):
    """原 app.py 分支：`path == "/api/storyboard"`。"""
    body = req.body_json()
    shots = generate_storyboard(str(body.get("brief") or ""), body.get("count", 4),
                                str(body.get("skill_id") or DEFAULT_SKILL))
    req.json(200, {"shots": shots})
    return


def post_api_prompt_preview(req, path, parts, url):
    """原 app.py 分支：`path == "/api/prompt-preview"`。"""
    body = req.body_json()
    state = read()
    project = state["projects"].get(str(body.get("project_id") or ""))
    if not project:
        raise ValueError("项目不存在")
    shot = next((item for item in project.get("shots", []) if item.get("id") == body.get("shot_id")), None)
    if not shot:
        raise ValueError("镜头不存在")
    prepared = prepare_shot(shot, project)
    # Compile the same continuity fields the run loop would apply, so the preview
    # shows the continuation prelude and handoff wording that actually get sent.
    prepared = continuity_preview(prepared, project, state.get("assets") or {})
    prompt = generation_prompt(prepared)
    req.json(200, {"prompt": prompt, "findings": lint_shot_prompt(prepared, project),
                    "image_references": prepared.get("_h3_image_reference_map", [])})
    return


ROUTES = [
    Route('POST', 'post_api_enhance_sequence',
         lambda path, parts, url: (path == "/api/enhance-sequence"),
         post_api_enhance_sequence),
    Route('POST', 'post_api_image_prompt',
         lambda path, parts, url: (path == "/api/image-prompt"),
         post_api_image_prompt),
    Route('POST', 'post_api_enhance',
         lambda path, parts, url: (path == "/api/enhance"),
         post_api_enhance),
    Route('POST', 'post_api_storyboard',
         lambda path, parts, url: (path == "/api/storyboard"),
         post_api_storyboard),
    Route('POST', 'post_api_prompt_preview',
         lambda path, parts, url: (path == "/api/prompt-preview"),
         post_api_prompt_preview),
    Route('POST', 'post_api_enhance_global',
         lambda path, parts, url: (path == "/api/enhance-global"),
         post_api_enhance_global),
]

"""创作智能体的工具注册表。

每个工具带一个风险等级，决定主脑能不能自己就动手：

    read        只读，随时可执行
    write       创作类写入，自动执行，但留痕并可一键回滚
    destructive 删除类，必须逐次向用户确认
    gpu         会占显卡提交渲染，必须逐次向用户确认

写操作全部走 store.change()，保证原子落库；写之前会把原值记进
agent_audit 的变更流水，用户可以在面板上一键回滚整轮改动。
"""

from __future__ import annotations

import copy
import re
import threading
import time
from pathlib import Path

from agent_audit import MISSING, record_change
from agent_audit import _Missing as _MissingType
from engine import ALL_MODES, lint_shot_prompt
from prompt_skills import list_skills
from runner import _shot_identity_refs, prepare_shot
from server.support import enhancement_references
from store import ASSETS, change, new_shot, read, uid
from store import align_dimension

# 画布档位（MP）与工作台界面保持一致：H3 官方 0.98MP 是最常用的一档。
CANVAS_MP_PRESETS = [("0.2", 608, 352), ("0.3", 736, 416), ("0.4", 864, 480),
                      ("0.5", 960, 544), ("0.6", 1056, 608), ("0.7", 1152, 640),
                      ("0.8", 1216, 672), ("0.9", 1280, 736), ("0.98", 1344, 768),
                      ("1.0", 1376, 768), ("1.2", 1504, 832), ("1.5", 1664, 928),
                      ("1.8", 1824, 1024), ("2.0", 1920, 1088)]

RISK_READ = "read"
RISK_WRITE = "write"
RISK_DESTRUCTIVE = "destructive"
RISK_GPU = "gpu"

RISK_LABEL = {RISK_READ: "只读", RISK_WRITE: "自动执行（可回滚）",
              RISK_DESTRUCTIVE: "需确认（不可撤销）", RISK_GPU: "需确认（占用显卡）"}

TOOLS: dict[str, dict] = {}

PICTURE_TAG = re.compile(r"<Picture\s+(\d+)>", re.IGNORECASE)


def tool(name: str, risk: str, summary: str, args: dict | None = None):
    """Register one agent tool."""
    def wrap(handler):
        TOOLS[name] = {"name": name, "risk": risk, "summary": summary,
                       "args": args or {}, "handler": handler}
        return handler
    return wrap


class ToolContext:
    """Everything a tool needs: the project, the current shot, the run record."""

    def __init__(self, project_id: str, shot_id: str = "", run: dict | None = None):
        self.project_id = project_id
        self.shot_id = shot_id
        self.run = run or {}
        self._state: dict | None = None

    def state(self) -> dict:
        if self._state is None:
            self._state = read()
        return self._state

    def project(self) -> dict:
        return self.state().get("projects", {}).get(self.project_id) or {}

    def invalidate(self) -> None:
        self._state = None


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def project_assets(state: dict, project_id: str) -> list[dict]:
    return [asset for asset in state.get("assets", {}).values()
            if asset.get("project_id") == project_id]


def ordered_assets(state: dict, project: dict, kind: str = "") -> list[dict]:
    """Assets in the user's own gallery order, exactly like the workbench shows them."""
    order = project.get("asset_order") or []
    positions = {asset_id: index for index, asset_id in enumerate(order)}
    items = project_assets(state, project.get("id") or "")
    if kind:
        items = [item for item in items if item.get("kind") == kind]
    return sorted(items, key=lambda item: (positions.get(item["id"], len(positions)),
                                           item.get("created_at") or 0))


def image_numbers(state: dict, project: dict) -> dict[str, int]:
    """asset_id -> <Picture N> number, matching the workbench numbering."""
    return {asset["id"]: index + 1
            for index, asset in enumerate(ordered_assets(state, project, "image"))}


def resolve_shot(ctx: ToolContext, args: dict, required: bool = True):
    project = ctx.project()
    shots = project.get("shots") or []
    shot_id = str(args.get("shot_id") or "").strip()
    if not shot_id and args.get("shot_index") not in (None, ""):
        try:
            index = int(args.get("shot_index")) - 1
        except (TypeError, ValueError):
            index = -1
        if 0 <= index < len(shots):
            return shots[index], index
    if not shot_id:
        shot_id = ctx.shot_id
    for index, shot in enumerate(shots):
        if shot.get("id") == shot_id:
            return shot, index
    if required:
        raise ValueError("找不到目标镜头；先用 list_shots 拿到正确的镜头 id")
    return None, -1


def shot_lint(state: dict, project: dict, shot: dict) -> list[dict]:
    try:
        return lint_shot_prompt(shot, project)
    except Exception:
        return []


def reference_context_for(state: dict, project: dict, shot: dict) -> dict:
    """Same context the workbench's 「AI 优化提示词」 button builds."""
    catalog = state.get("assets") or {}
    prepared = prepare_shot(shot, project)
    identity_assets = set(_shot_identity_refs(shot, project, catalog))
    if prepared.get("mode") in {"t2v", "i2v", "fl2v"}:
        identity_assets.intersection_update(filter(None, (
            prepared.get("first_image"), prepared.get("last_image"))))
    shots = project.get("shots") or []
    index = next((i for i, item in enumerate(shots) if item.get("id") == shot.get("id")), 0)

    def neighbor(position: int):
        if position < 0 or position >= len(shots):
            return None
        item = shots[position]
        return {"name": str(item.get("name") or "镜头")[:120],
                "mode": str(item.get("mode") or "t2v")}

    context = {"segment": {
        "position": index + 1, "total": len(shots),
        "is_first": index == 0, "is_last": index == len(shots) - 1,
        "previous_shot": neighbor(index - 1), "next_shot": neighbor(index + 1)},
        "selected_character_ids": list(shot.get("character_ids") or [])}
    references = enhancement_references(prepared, project, catalog, identity_assets)
    context.update({"references": references,
                    "characters": prepared.get("_h3_characters", []),
                    "entities": prepared.get("_h3_entities", []),
                    "person_identity_asset_ids": list(identity_assets)})
    return context


def mutate_shot(ctx: ToolContext, shot_id: str, mutator, summary: str = ""):
    """Apply one shot-level change and remember the original for undo."""
    outcome: dict = {}

    def apply(data: dict) -> dict:
        project = data["projects"].get(ctx.project_id)
        if not project:
            raise ValueError("项目不存在")
        shots = project.get("shots") or []
        index = next((i for i, shot in enumerate(shots) if shot.get("id") == shot_id), None)
        if index is None:
            raise ValueError("镜头不存在")
        before = copy.deepcopy(shots[index])
        detail = mutator(shots[index], project, data) or {}
        after = copy.deepcopy(shots[index])
        project["_data_version"] = int(project.get("_data_version", 0)) + 1
        outcome["record"] = {"kind": "shot", "shot_id": shot_id, "index": index,
                             "before": before, "after": after}
        outcome["detail"] = detail
        outcome["project"] = copy.deepcopy(project)
        return {"project": outcome["project"]}

    change(apply)
    if outcome.get("record"):
        record_change(ctx.run, outcome["record"])
    ctx.invalidate()
    return outcome.get("detail") or {}


def mutate_project(ctx: ToolContext, mutator):
    """Apply one project-level change and remember the original values for undo."""
    outcome: dict = {}

    def apply(data: dict) -> dict:
        project = data["projects"].get(ctx.project_id)
        if not project:
            raise ValueError("项目不存在")
        before: dict = {}
        detail = mutator(project, data, before) or {}
        project["_data_version"] = int(project.get("_data_version", 0)) + 1
        outcome["record"] = {"kind": "project", "before": before}
        outcome["detail"] = detail
        outcome["project"] = copy.deepcopy(project)
        return {"project": outcome["project"]}

    change(apply)
    if outcome.get("record"):
        record_change(ctx.run, outcome["record"])
    ctx.invalidate()
    return outcome.get("detail") or {}


# --------------------------------------------------------------------------
# 只读工具
# --------------------------------------------------------------------------

@tool("list_shots", RISK_READ,
      "列出本项目所有镜头：序号、id、名称、生成模式、时长、是否选中、参考图尺寸策略、底色策略、提示词里的 <Picture N> 标签。",
      {})
def _list_shots(ctx: ToolContext, args: dict) -> dict:
    project = ctx.project()
    shots = []
    for index, shot in enumerate(project.get("shots") or []):
        shots.append({
            "index": index + 1, "id": shot.get("id"), "name": shot.get("name"),
            "mode": shot.get("mode"), "duration": shot.get("duration"),
            "selected": bool(shot.get("selected", True)),
            "ref_image_size": shot.get("ref_image_size") or "match",
            "ref_background": shot.get("ref_background") or "auto",
            "picture_tags": sorted({int(value) for value in
                                    PICTURE_TAG.findall(str(shot.get("prompt") or ""))}),
            "has_source_video": bool(shot.get("source_video")),
            "prompt_head": str(shot.get("prompt") or "")[:140]})
    return {"summary": f"项目「{project.get('name')}」共 {len(shots)} 个镜头",
            "data": {"shots": shots}}


@tool("get_shot", RISK_READ,
      "读取一个镜头的完整状态：提示词全文、所有参数、绑定的素材、参考素材角色、以及体检结果（lint）。",
      {"shot_id": {"type": "string", "desc": "镜头 id；留空表示当前正在编辑的镜头"}})
def _get_shot(ctx: ToolContext, args: dict) -> dict:
    state = ctx.state()
    project = ctx.project()
    shot, index = resolve_shot(ctx, args)
    numbers = image_numbers(state, project)
    catalog = state.get("assets") or {}
    bound = []
    for asset_id in list(shot.get("ref_images") or []) + list(shot.get("ref_videos") or []) + \
            list(shot.get("ref_audios") or []) + [shot.get("first_image"), shot.get("last_image"),
                                                  shot.get("source_video")]:
        asset = catalog.get(asset_id)
        if asset:
            bound.append({"id": asset_id, "name": asset.get("name"), "kind": asset.get("kind"),
                          "picture": numbers.get(asset_id)})
    roles = {**dict(shot.get("ref_image_roles") or {}), **dict(shot.get("ref_video_roles") or {}),
             **dict(shot.get("ref_audio_roles") or {})}
    return {"summary": f"镜头 {index + 1}「{shot.get('name')}」",
            "data": {"index": index + 1, "shot": copy.deepcopy(shot), "bound_assets": bound,
                     "reference_roles": roles,
                     "lint": shot_lint(state, project, shot)}}


@tool("list_assets", RISK_READ,
      "列出项目素材：id、文件名、类型，以及图片对应的 <Picture N> 编号和它被哪些镜头使用。",
      {"kind": {"type": "string", "desc": "image / video / audio，留空表示全部"}})
def _list_assets(ctx: ToolContext, args: dict) -> dict:
    state = ctx.state()
    project = ctx.project()
    numbers = image_numbers(state, project)
    wanted = str(args.get("kind") or "").strip()
    items = []
    for asset in ordered_assets(state, project, wanted):
        asset_id = asset["id"]
        used_by = []
        for index, shot in enumerate(project.get("shots") or []):
            if asset_id in (shot.get("ref_images") or []) or asset_id in (shot.get("ref_videos") or []) \
                    or asset_id in (shot.get("ref_audios") or []) \
                    or asset_id in (shot.get("first_image"), shot.get("last_image"), shot.get("source_video")):
                used_by.append({"index": index + 1, "id": shot.get("id"), "name": shot.get("name")})
        items.append({"id": asset_id, "name": asset.get("name"), "kind": asset.get("kind"),
                      "picture": numbers.get(asset_id),
                      "description": (project.get("image_descriptions") or {}).get(asset_id, ""),
                      "used_by": used_by})
    return {"summary": f"项目共 {len(items)} 个素材", "data": {"assets": items}}


@tool("list_characters", RISK_READ,
      "列出项目角色：id、姓名、设定描述、肖像图/多视图图/音色素材，以及这些角色出现在哪些镜头。",
      {})
def _list_characters(ctx: ToolContext, args: dict) -> dict:
    project = ctx.project()
    items = []
    for character in project.get("characters") or []:
        shots = [{"index": index + 1, "id": shot.get("id"), "name": shot.get("name")}
                 for index, shot in enumerate(project.get("shots") or [])
                 if character.get("id") in (shot.get("character_ids") or [])]
        items.append({"id": character.get("id"), "name": character.get("name"),
                      "description": character.get("description"),
                      "portrait_asset": character.get("portrait_asset"),
                      "multiview_asset": character.get("multiview_asset"),
                      "voice_asset": character.get("voice_asset"),
                      "shots": shots})
    return {"summary": f"项目共 {len(items)} 个角色", "data": {"characters": items}}


@tool("get_project_settings", RISK_READ,
      "读取项目级生成参数：画布尺寸、帧率、采样档位、换人/风格 LoRA 强度、语义适配、自提升、精修参数、所选 Skill。",
      {})
def _get_project_settings(ctx: ToolContext, args: dict) -> dict:
    project = ctx.project()
    keys = ("width", "height", "fps", "generation_speed", "prompt_skill", "sage_attention",
            "h3_lora_dmad", "h3_lora_dmad_strength", "h3_lora_mystic", "h3_lora_mystic_strength",
            "h3_lora_character_swap_strength", "semantic_adapter", "semantic_alpha",
            "semantic_magnitude_match", "selflift", "selflift_scale", "selflift_highres_steps",
            "refine_mode", "refine_steps", "refine_denoise", "upscale", "upscale_resolution",
            "upscale_max_resolution", "continuous_take", "export_mode", "memory_policy",
            "face_refine", "h3_video_vae", "h3_clip_model", "latent_model")
    settings = {key: project.get(key) for key in keys if key in project}
    settings["shared_prompt"] = str(project.get("shared_prompt") or "")
    settings["script_text_head"] = str(project.get("script_text") or "")[:400]
    settings["sequence_brief_head"] = str(project.get("sequence_brief") or "")[:400]
    return {"summary": "当前项目生成参数", "data": {"settings": settings,
                                                   "all_keys": sorted(project.keys())}}


@tool("list_skills", RISK_READ,
      "列出工作台可用的提示词 Skill（id 与名称），例如专门写换人提示词的 Skill。",
      {})
def _list_skills(ctx: ToolContext, args: dict) -> dict:
    items = [{"id": item.get("id"), "name": item.get("name"),
              "summary": str(item.get("summary") or item.get("description") or "")[:160]}
             for item in list_skills()]
    return {"summary": f"共 {len(items)} 个 Skill", "data": {"skills": items}}


@tool("lint_shot", RISK_READ,
      "体检一个镜头：返回工作台内置的提示词/参数体检结果（空提示词、旁白超时、换人不送参考图、参考图尺寸过小等）。",
      {"shot_id": {"type": "string", "desc": "镜头 id；留空表示当前镜头"}})
def _lint_shot(ctx: ToolContext, args: dict) -> dict:
    state = ctx.state()
    project = ctx.project()
    shot, index = resolve_shot(ctx, args)
    findings = shot_lint(state, project, shot)
    return {"summary": f"镜头 {index + 1} 体检：{len(findings)} 条",
            "data": {"findings": findings}}


@tool("list_runs", RISK_READ,
      "查看最近的生成任务：状态、对应镜头、成片文件名、报错信息。",
      {"limit": {"type": "number", "desc": "返回条数，默认 5"}})
def _list_runs(ctx: ToolContext, args: dict) -> dict:
    state = ctx.state()
    limit = max(1, min(20, int(args.get("limit") or 5)))
    runs = sorted(state.get("runs", {}).values(),
                  key=lambda run: run.get("created_at") or 0, reverse=True)
    items = []
    for run in runs:
        if run.get("project_id") != ctx.project_id:
            continue
        results = run.get("results") or []
        items.append({"id": run.get("id"), "status": run.get("status"),
                      "shot_ids": run.get("shot_ids"),
                      "created_at": run.get("created_at"),
                      "results": [{"shot_id": item.get("shot_id"), "file": item.get("file"),
                                   "error": str(item.get("error") or "")[:200]}
                                  for item in results]})
        if len(items) >= limit:
            break
    return {"summary": f"最近 {len(items)} 个生成任务", "data": {"runs": items}}


@tool("consult_playbook", RISK_READ,
      "查阅工作台知识库：产品说明、可用 Skill 全文、换人/参数排障手册。动手前先查，能少走弯路。",
      {"topic": {"type": "string", "desc": "要查的主题，例如「换人 服装 不生效」「参考图尺寸」", "required": True}})
def _consult_playbook(ctx: ToolContext, args: dict) -> dict:
    from agent_knowledge import search_knowledge
    topic = str(args.get("topic") or "").strip()
    if not topic:
        raise ValueError("请给出要查询的主题")
    hits = search_knowledge(topic, limit=4)
    return {"summary": f"知识库命中 {len(hits)} 条",
            "data": {"hits": [{"title": hit.get("title"), "source": hit.get("source"),
                               "text": hit.get("text")} for hit in hits]}}


@tool("inspect_run_frames", RISK_READ,
      "抽帧并「看图」自评某个生成任务：把成片抽成几帧交给多模态模型看，判断人物/服装/背景是否符合预期。只读，不改项目。",
      {"run_id": {"type": "string", "desc": "生成任务 id", "required": True},
       "question": {"type": "string", "desc": "要问画面的问题，例如「人物的服装是否已换成中式对襟褂」",
                    "required": True},
       "count": {"type": "number", "desc": "抽帧数量，默认 3"}})
def _inspect_run_frames(ctx: ToolContext, args: dict) -> dict:
    from agent_vision import judge_run_frames
    run_id = str(args.get("run_id") or "").strip()
    question = str(args.get("question") or "").strip()
    if not run_id or not question:
        raise ValueError("需要 run_id 和 question")
    verdict = judge_run_frames(ctx.state(), run_id, question, count=int(args.get("count") or 3))
    return {"summary": verdict.get("summary", "已抽帧查看"),
            "data": verdict}


# --------------------------------------------------------------------------
# 创作类写入（自动执行 + 可回滚）
# --------------------------------------------------------------------------

SHOT_PARAM_CHOICES = {
    "camera_move": ["自动/未指定", "固定镜头", "推近", "拉远", "横向移动", "跟随主体", "环绕主体", "手持"],
    "camera_speed": ["自动/未指定", "缓慢", "平稳", "快速"],
    "action_intensity": ["自然", "轻微", "明显", "夸张"],
    "emotion": ["自动/未指定", "轻松", "紧张", "悲伤", "温馨", "热烈", "悬疑", "浪漫"],
}

REFERENCE_ROLES = {
    "image": ["auto", "weak_reference", "fully_preserved", "partially_preserved", "attribute_transfer"],
    "video": ["weak_reference", "fully_preserved", "partially_preserved", "attribute_transfer"],
    "audio": ["reference", "weak_reference", "fully_copy", "partially_copy"],
}


@tool("write_prompt", RISK_WRITE,
      "直接改写某个镜头的提示词全文（原文会记进报告，可回滚）。",
      {"shot_id": {"type": "string", "desc": "镜头 id；留空表示当前镜头"},
       "prompt": {"type": "string", "desc": "新的提示词全文", "required": True},
       "reason": {"type": "string", "desc": "为什么改（一句话，会进报告）"}})
def _write_prompt(ctx: ToolContext, args: dict) -> dict:
    shot, index = resolve_shot(ctx, args)
    shot_id = shot.get("id")
    text = str(args.get("prompt") or "").strip()
    if not text:
        raise ValueError("提示词不能为空")
    old = str(shot.get("prompt") or "")
    if text == old:
        return {"summary": f"镜头 {index + 1} 提示词已经是目标内容，未改动", "data": {}}

    def mutator(target: dict, project: dict, data: dict) -> dict:
        target["prompt"] = text[:6000]
        return {"prompt_before": old, "prompt_after": text[:6000]}

    mutate_shot(ctx, shot_id, mutator)
    return {"summary": f"改写镜头 {index + 1}「{shot.get('name')}」的提示词",
            "data": {"prompt_before": old, "prompt_after": text[:6000],
                     "reason": str(args.get("reason") or "")}}


@tool("rewrite_prompt", RISK_WRITE,
      "按你的要求，用工作台的 Skill 重写某个镜头的提示词（会走云端 AI 优化，可用换人专用 Skill）。",
      {"shot_id": {"type": "string", "desc": "镜头 id；留空表示当前镜头"},
       "instruction": {"type": "string", "desc": "要怎么改，例如「强化服装与人物一致，不要自编场景」", "required": True},
       "language": {"type": "string", "desc": "en 或 zh，默认沿用镜头设置"},
       "skill_id": {"type": "string", "desc": "指定 Skill；留空用项目当前 Skill"}})
def _rewrite_prompt(ctx: ToolContext, args: dict) -> dict:
    from prompt_tools import enhance
    state = ctx.state()
    project = ctx.project()
    shot, index = resolve_shot(ctx, args)
    shot_id = shot.get("id")
    instruction = str(args.get("instruction") or "").strip()
    if not instruction:
        raise ValueError("请说明要怎么改")
    catalog = {item["id"]: item for item in list_skills()}
    skill_id = str(args.get("skill_id") or "").strip()
    swap_skill = "custom-h3-character-swap" if "custom-h3-character-swap" in catalog else ""
    if not skill_id:
        # 换人镜头默认用换人专用 Skill：它管得住身份绑定句式和标签纪律。
        swap_like = bool(shot.get("ref_images")) and str(shot.get("mode") or "") in {"r2v", "rv2v"}
        skill_id = swap_skill if (swap_like and swap_skill) \
            else str(project.get("prompt_skill") or (swap_skill or "h3-prompt-writing"))
    if skill_id not in catalog:
        skill_id = swap_skill or next(iter(catalog), "h3-prompt-writing")
    language = str(args.get("language") or shot.get("prompt_language") or "en")
    if language not in {"en", "zh"}:
        language = "en"
    old = str(shot.get("prompt") or "")
    context = reference_context_for(state, project, shot)
    optimized = enhance(old, str(shot.get("mode") or "t2v"), skill_id,
                         float(shot.get("duration") or 5), context, language, instruction)
    optimized = str(optimized or "").strip()
    if not optimized:
        raise ValueError("AI 没有返回有效提示词")
    if optimized == old:
        return {"summary": f"镜头 {index + 1} 提示词无需改动", "data": {}}

    def mutator(target: dict, project_data: dict, data: dict) -> dict:
        target["prompt"] = optimized[:6000]
        return {"prompt_before": old, "prompt_after": optimized[:6000]}

    mutate_shot(ctx, shot_id, mutator)
    return {"summary": f"用 Skill「{catalog.get(skill_id, {}).get('name', skill_id)}」重写镜头 {index + 1} 提示词",
            "data": {"prompt_before": old, "prompt_after": optimized[:6000],
                     "skill_id": skill_id, "reason": instruction}}


@tool("set_shot_params", RISK_WRITE,
      "修改镜头参数：生成模式、时长、参考图尺寸策略、参考素材底色、种子、音频方式、提示词语言、运镜、情绪等。",
      {"shot_id": {"type": "string", "desc": "镜头 id；留空表示当前镜头"},
       "params": {"type": "object",
                  "desc": "要改的字段，例如 {\"ref_image_size\":\"max\",\"ref_background\":\"inherit\",\"mode\":\"rv2v\",\"duration\":6}",
                  "required": True}})
def _set_shot_params(ctx: ToolContext, args: dict) -> dict:
    shot, index = resolve_shot(ctx, args)
    shot_id = shot.get("id")
    params = args.get("params")
    if not isinstance(params, dict) or not params:
        raise ValueError("请提供要修改的字段")
    allowed = {"mode", "duration", "seed", "ref_image_size", "ref_background", "audio_mode",
               "prompt_language", "camera_move", "camera_speed", "action_intensity", "emotion",
               "continuity", "selected", "replace_person", "source_start", "swap_lora",
               "faceswap_trigger_only", "dialogue_text", "speaker_id", "on_screen_text",
               "overall_soundscape", "non_diegetic_music", "dialogue_voiceover", "identity_first",
               "name", "guide_mode", "dialogue_language", "dialogue_time", "dialogue_cutoff",
               "dialogue_delivery", "dialogue_carries_out", "speaker_description",
               "character_ids", "entity_ids", "scene_id"}
    changed: dict = {}
    before_values: dict = {}

    def mutator(target: dict, project: dict, data: dict) -> dict:
        for key, value in params.items():
            if key not in allowed:
                raise ValueError(f"镜头字段 {key} 不允许修改；可用字段：{'、'.join(sorted(allowed))}")
            before_values[key] = target.get(key, MISSING)
            if key == "mode":
                if value not in ALL_MODES:
                    raise ValueError(f"生成模式无效，只能是 {'/'.join(ALL_MODES)}")
                target["mode"] = value
            elif key == "duration":
                number = float(value)
                if not 2 <= number <= 15:
                    raise ValueError("时长必须在 2–15 秒之间")
                target["duration"] = number
            elif key == "ref_image_size":
                if value not in {"match", "max"}:
                    raise ValueError("参考图尺寸只能是 match 或 max")
                target["ref_image_size"] = value
            elif key == "ref_background":
                if value not in {"auto", "inherit", "authored"}:
                    raise ValueError("参考素材底色只能是 auto / inherit / authored")
                target["ref_background"] = value
            elif key == "audio_mode":
                if value not in {"source", "mute"}:
                    raise ValueError("音频方式只能是 source 或 mute")
                target["audio_mode"] = value
            elif key == "prompt_language":
                if value not in {"en", "zh"}:
                    raise ValueError("提示词语言只能是 en 或 zh")
                target["prompt_language"] = value
            elif key in SHOT_PARAM_CHOICES:
                if str(value) not in SHOT_PARAM_CHOICES[key]:
                    raise ValueError(f"{key} 只能是 {'、'.join(SHOT_PARAM_CHOICES[key])}")
                target[key] = value
            elif key in {"selected", "continuity", "replace_person", "faceswap_trigger_only",
                         "dialogue_voiceover", "identity_first"}:
                if not isinstance(value, bool):
                    raise ValueError(f"{key} 必须是 true 或 false")
                target[key] = value
            elif key == "seed":
                target["seed"] = int(value)
            elif key == "source_start":
                target["source_start"] = float(value)
            elif key == "swap_lora":
                target["swap_lora"] = str(value)
            elif key in {"dialogue_text", "speaker_id", "on_screen_text",
                         "overall_soundscape", "non_diegetic_music"}:
                target[key] = str(value)[:2000]
            changed[key] = target.get(key)
        return {"changed": changed}

    mutate_shot(ctx, shot_id, mutator)
    return {"summary": f"调整镜头 {index + 1} 的 {len(changed)} 项参数",
            "data": {"changed": changed,
                     "before": {key: (None if isinstance(value, _MissingType) else value)
                                for key, value in before_values.items()}}}


@tool("set_project_settings", RISK_WRITE,
      "修改项目级生成参数：画布宽高、帧率、采样档位、换人 LoRA 强度、语义适配、自提升、精修、超分等。",
      {"settings": {"type": "object",
                    "desc": "要改的键值，例如 {\"generation_speed\":\"turbo\",\"h3_lora_character_swap_strength\":1.2}",
                    "required": True}})
def _set_project_settings(ctx: ToolContext, args: dict) -> dict:
    settings = args.get("settings")
    if not isinstance(settings, dict) or not settings:
        raise ValueError("请提供要修改的参数")
    numeric = {"width", "height", "fps", "semantic_alpha", "selflift_scale",
               "selflift_highres_steps", "selflift_rho", "selflift_tile_count",
               "selflift_tile_overlap", "refine_steps", "refine_passes", "refine_denoise",
               "refine_width", "refine_height", "upscale_resolution", "upscale_max_resolution",
               "continuity_frames", "h3_lora_dmad_strength", "h3_lora_mystic_strength",
               "h3_lora_character_swap_strength", "seam_prefix_frames", "seam_motion_frames"}
    boolean = {"upscale", "confirm_upscale", "face_refine", "semantic_magnitude_match",
               "selflift", "selflift_tiling", "clear_vram_between_shots", "sage_attention",
               "continuous_take", "h3_lora_dmad", "h3_lora_mystic", "seam_exposure",
               "identity_first_on_switch"}
    text = {"generation_speed", "prompt_skill", "export_mode", "memory_policy", "h3_video_vae",
            "h3_clip_model", "latent_model", "semantic_adapter", "refine_mode", "seam_mode",
            "attention_mode", "continuous_render_mode", "sequence_brief", "script_text",
            "name"}
    changed: dict = {}

    def mutator(project: dict, data: dict, before: dict) -> dict:
        for key, value in settings.items():
            if key == "shared_prompt":
                raise ValueError("shared_prompt（全片统一提示词）请用 set_shared_prompt 工具修改")
            if key == "aspect":
                ratio = str(value or "")
                if ratio not in ("16:9", "9:16", "1:1"):
                    raise ValueError("画面比例只能是 16:9 / 9:16 / 1:1")
                base = (int(project.get("width") or 1344) * int(project.get("height") or 768)) ** 0.5
                if ratio == "1:1":
                    side = align_dimension(base, 768)
                    before["width"], before["height"] = project.get("width", MISSING), project.get("height", MISSING)
                    project["width"] = project["height"] = side
                else:
                    width = align_dimension(base * (16 / 9 if ratio == "16:9" else 9 / 16), 1344)
                    height = align_dimension(base * (9 / 16 if ratio == "16:9" else 16 / 9), 768)
                    before["width"], before["height"] = project.get("width", MISSING), project.get("height", MISSING)
                    project["width"], project["height"] = width, height
                changed["width"], changed["height"] = project["width"], project["height"]
                continue
            if key == "resolution":
                mp = str(value or "")
                row = next((item for item in CANVAS_MP_PRESETS if item[0] == mp), None)
                if not row:
                    raise ValueError("像素档位无效，可用：" + "、".join(item[0] for item in CANVAS_MP_PRESETS))
                portrait = int(project.get("height") or 768) > int(project.get("width") or 1344)
                before["width"], before["height"] = project.get("width", MISSING), project.get("height", MISSING)
                project["width"] = row[2] if portrait else row[1]
                project["height"] = row[1] if portrait else row[2]
                changed["width"], changed["height"] = project["width"], project["height"]
                continue
            if key not in numeric | boolean | text:
                raise ValueError(f"项目参数 {key} 不可修改")
            before[key] = project.get(key, MISSING)
            if key in numeric:
                number = float(value)
                if key.startswith("h3_lora_"):
                    number = max(0.0, min(1.5, number))
                if key == "seam_prefix_frames":
                    number = int(number)
                    if number not in (0, 1, 2, 3):
                        raise ValueError("拼接前缀帧数只能是 0/1/2/3")
                if key == "seam_motion_frames":
                    number = int(number)
                    if number not in (0, 2, 3):
                        raise ValueError("拼接动作帧数只能是 0/2/3")
                project[key] = number
            elif key in boolean:
                project[key] = bool(value)
            else:
                if key in ("attention_mode", "continuous_render_mode"):
                    allowed = {"attention_mode": {"off", "sage", "veda"},
                               "continuous_render_mode": {"auto", "segmented"}}[key]
                    if str(value) not in allowed:
                        raise ValueError(f"{key} 只能是 {'/'.join(sorted(allowed))}")
                project[key] = str(value)
            changed[key] = project[key]
        return {"changed": changed}

    mutate_project(ctx, mutator)
    return {"summary": f"调整项目参数 {len(changed)} 项", "data": {"changed": changed}}


@tool("set_shared_prompt", RISK_WRITE,
      "改写项目的「全片统一提示词」：对全片所有镜头生效的固定要求"
      "（画面风格、色调、质感、格式约束等），不是单个镜头的提示词。"
      "要改某个镜头的提示词请用 write_prompt。",
      {"prompt": {"type": "string", "desc": "新的全片统一提示词正文", "required": True},
       "reason": {"type": "string", "desc": "为什么这样改，一句话"}})
def _set_shared_prompt(ctx: ToolContext, args: dict) -> dict:
    text = str(args.get("prompt") or "").strip()
    if not text:
        raise ValueError("全片统一提示词不能为空")
    old = str(ctx.project().get("shared_prompt") or "")
    if text == old:
        return {"summary": "全片统一提示词已经是这个内容，未改动", "data": {}}

    def mutator(project: dict, data: dict, before: dict) -> dict:
        before["shared_prompt"] = project.get("shared_prompt", MISSING)
        project["shared_prompt"] = text
        return {"shared_prompt": text}

    mutate_project(ctx, mutator)
    return {"summary": "已改写全片统一提示词",
            "data": {"prompt_before": old, "prompt_after": text,
                     "reason": str(args.get("reason") or "")}}


@tool("list_entities", RISK_READ,
      "列出项目的地点、道具与画风档案（id、名称、类型、描述、参考图）。",
      {})
def _list_entities(ctx: ToolContext, args: dict) -> dict:
    project = ctx.project()
    items = []
    for entity in project.get("entities") or []:
        items.append({"id": entity.get("id"), "name": entity.get("name"),
                      "kind": entity.get("kind") or "prop",
                      "description": str(entity.get("description") or "")[:200],
                      "image_assets": list(entity.get("image_assets") or []),
                      "generation_mode": entity.get("generation_mode") or "t2i"})
    scenes = [{"id": scene.get("id"), "name": scene.get("name")}
              for scene in project.get("scenes") or []]
    return {"summary": f"项目共 {len(items)} 个设定档案、{len(scenes)} 个场景分组",
            "data": {"entities": items, "scenes": scenes}}


@tool("create_entity", RISK_WRITE,
      "新建一个地点、道具或画风档案。",
      {"name": {"type": "string", "desc": "档案名称", "required": True},
       "kind": {"type": "string", "desc": "prop 道具 / place 地点 / style 画风，默认 prop"},
       "description": {"type": "string", "desc": "固定描述，会写进用到它的镜头提示词"},
       "generation_prompt": {"type": "string", "desc": "生图用的提示词"},
       "generation_mode": {"type": "string", "desc": "t2i 文生图 / i2i 图生图，默认 t2i"}})
def _create_entity(ctx: ToolContext, args: dict) -> dict:
    name = str(args.get("name") or "").strip()
    if not name:
        raise ValueError("请给档案起个名字")
    kind = str(args.get("kind") or "prop")
    if kind not in ("prop", "place", "style"):
        raise ValueError("档案类型只能是 prop / place / style")
    generation_mode = str(args.get("generation_mode") or "t2i")
    if generation_mode not in ("t2i", "i2i"):
        raise ValueError("档案生图方式只能是 t2i 或 i2i")
    entity = {"id": uid(), "name": name, "kind": kind,
              "description": str(args.get("description") or "")[:1000],
              "generation_prompt": str(args.get("generation_prompt") or "")[:1000],
              "generation_mode": generation_mode, "image_assets": []}

    def mutator(project: dict, data: dict, before: dict) -> dict:
        before["entities"] = copy.deepcopy(project.get("entities") or [])
        project.setdefault("entities", []).append(entity)
        return {"entity": copy.deepcopy(entity)}

    mutate_project(ctx, mutator)
    return {"summary": f"新建{ {'prop': '道具', 'place': '地点', 'style': '画风'}[kind] }档案「{name}」",
            "data": {"entity": entity}}


@tool("update_entity", RISK_WRITE,
      "修改地点、道具或画风档案的名称、类型、描述、生图设置与参考图。",
      {"entity_id": {"type": "string", "desc": "档案 id（用 list_entities 拿）", "required": True},
       "name": {"type": "string", "desc": "新名称"},
       "kind": {"type": "string", "desc": "prop / place / style"},
       "description": {"type": "string", "desc": "固定描述"},
       "generation_prompt": {"type": "string", "desc": "生图提示词"},
       "generation_mode": {"type": "string", "desc": "t2i / i2i"},
       "image_assets": {"type": "array", "desc": "参考图片素材 id 列表",
                        "items": {"type": "string"}}})
def _update_entity(ctx: ToolContext, args: dict) -> dict:
    entity_id = str(args.get("entity_id") or "")
    catalog = ctx.state().get("assets") or {}
    project = ctx.project()
    entity = next((item for item in project.get("entities") or []
                   if item.get("id") == entity_id), None)
    if not entity:
        raise ValueError("档案不存在；先用 list_entities 查 id")

    def mutator(target: dict, data: dict, before: dict) -> dict:
        item = next((entry for entry in target.get("entities") or []
                     if entry.get("id") == entity_id), None)
        if not item:
            raise ValueError("档案不存在")
        before["entities"] = copy.deepcopy(target.get("entities"))
        if args.get("name") not in (None, ""):
            item["name"] = str(args.get("name")).strip()[:120]
        if args.get("kind") not in (None, ""):
            kind = str(args.get("kind"))
            if kind not in ("prop", "place", "style"):
                raise ValueError("档案类型只能是 prop / place / style")
            item["kind"] = kind
        for key in ("description", "generation_prompt"):
            if args.get(key) not in (None, ""):
                item[key] = str(args.get(key))[:1000]
        if args.get("generation_mode") not in (None, ""):
            mode = str(args.get("generation_mode"))
            if mode not in ("t2i", "i2i"):
                raise ValueError("档案生图方式只能是 t2i 或 i2i")
            item["generation_mode"] = mode
        if isinstance(args.get("image_assets"), list):
            images = [str(value) for value in args["image_assets"]]
            for asset_id in images:
                if asset_id not in catalog or catalog[asset_id].get("kind") != "image":
                    raise ValueError(f"参考图 {asset_id} 不是本项目的图片素材")
            item["image_assets"] = images
        return {"entity": copy.deepcopy(item)}

    mutate_project(ctx, mutator)
    return {"summary": f"更新档案「{entity.get('name')}」", "data": {"entity_id": entity_id}}


@tool("delete_entity", RISK_DESTRUCTIVE,
      "删除一个地点、道具或画风档案（会把它从相关镜头里摘掉）。",
      {"entity_id": {"type": "string", "desc": "档案 id（用 list_entities 拿）", "required": True}})
def _delete_entity(ctx: ToolContext, args: dict) -> dict:
    entity_id = str(args.get("entity_id") or "")
    project = ctx.project()
    entity = next((item for item in project.get("entities") or []
                   if item.get("id") == entity_id), None)
    if not entity:
        raise ValueError("档案不存在")

    def mutator(target: dict, data: dict, before: dict) -> dict:
        before["entities"] = copy.deepcopy(target.get("entities"))
        target["entities"] = [item for item in target.get("entities") or []
                              if item.get("id") != entity_id]
        for shot in target.get("shots") or []:
            if entity_id in (shot.get("entity_ids") or []):
                shot["entity_ids"] = [value for value in shot["entity_ids"]
                                      if value != entity_id]
        return {"removed": entity_id}

    mutate_project(ctx, mutator)
    return {"summary": f"删除档案「{entity.get('name')}」", "data": {"entity_id": entity_id}}


@tool("create_scene", RISK_WRITE,
      "新建一个场景分组（镜头可以归到分组里方便管理）。",
      {"name": {"type": "string", "desc": "分组名称", "required": True}})
def _create_scene(ctx: ToolContext, args: dict) -> dict:
    name = str(args.get("name") or "").strip()
    if not name:
        raise ValueError("请给场景分组起个名字")
    scene = {"id": uid(), "name": name}

    def mutator(project: dict, data: dict, before: dict) -> dict:
        before["scenes"] = copy.deepcopy(project.get("scenes") or [])
        project.setdefault("scenes", []).append(scene)
        return {"scene": copy.deepcopy(scene)}

    mutate_project(ctx, mutator)
    return {"summary": f"新建场景分组「{name}」", "data": {"scene": scene}}


@tool("update_scene", RISK_WRITE,
      "改场景分组的名称。",
      {"scene_id": {"type": "string", "desc": "场景分组 id（用 list_entities 拿）", "required": True},
       "name": {"type": "string", "desc": "新名称", "required": True}})
def _update_scene(ctx: ToolContext, args: dict) -> dict:
    scene_id = str(args.get("scene_id") or "")
    name = str(args.get("name") or "").strip()
    if not name:
        raise ValueError("请给场景分组起个名字")
    project = ctx.project()
    scene = next((item for item in project.get("scenes") or []
                  if item.get("id") == scene_id), None)
    if not scene:
        raise ValueError("场景分组不存在")

    def mutator(target: dict, data: dict, before: dict) -> dict:
        before["scenes"] = copy.deepcopy(target.get("scenes"))
        for item in target.get("scenes") or []:
            if item.get("id") == scene_id:
                item["name"] = name
        return {"scene_id": scene_id}

    mutate_project(ctx, mutator)
    return {"summary": f"场景分组改名为「{name}」", "data": {"scene_id": scene_id}}


@tool("delete_scene", RISK_DESTRUCTIVE,
      "删除场景分组；属于它的镜头会回到「未分组」，镜头本身不删。",
      {"scene_id": {"type": "string", "desc": "场景分组 id（用 list_entities 拿）", "required": True}})
def _delete_scene(ctx: ToolContext, args: dict) -> dict:
    scene_id = str(args.get("scene_id") or "")
    project = ctx.project()
    scene = next((item for item in project.get("scenes") or []
                  if item.get("id") == scene_id), None)
    if not scene:
        raise ValueError("场景分组不存在")

    def mutator(target: dict, data: dict, before: dict) -> dict:
        before["scenes"] = copy.deepcopy(target.get("scenes"))
        target["scenes"] = [item for item in target.get("scenes") or []
                            if item.get("id") != scene_id]
        for shot in target.get("shots") or []:
            if shot.get("scene_id") == scene_id:
                shot["scene_id"] = ""
        return {"removed": scene_id}

    mutate_project(ctx, mutator)
    return {"summary": f"删除场景分组「{scene.get('name')}」", "data": {"scene_id": scene_id}}


@tool("bind_asset", RISK_WRITE,
      "把素材绑定到镜头的某个位置：首帧、尾帧、源视频、参考图、参考视频、参考声音。",
      {"shot_id": {"type": "string", "desc": "镜头 id；留空表示当前镜头"},
       "asset_id": {"type": "string", "desc": "素材 id（用 list_assets 拿）", "required": True},
       "slot": {"type": "string", "desc": "first_image / last_image / source_video / ref_image / ref_video / ref_audio",
                "required": True}})
def _bind_asset(ctx: ToolContext, args: dict) -> dict:
    state = ctx.state()
    project = ctx.project()
    shot, index = resolve_shot(ctx, args)
    shot_id = shot.get("id")
    asset_id = str(args.get("asset_id") or "")
    slot = str(args.get("slot") or "")
    asset = (state.get("assets") or {}).get(asset_id)
    if not asset or asset.get("project_id") != ctx.project_id:
        raise ValueError("素材不存在或不属于当前项目")
    kind = asset.get("kind")
    spec = {"first_image": ("image", "first_image", 1), "last_image": ("image", "last_image", 1),
            "source_video": ("video", "source_video", 1), "ref_image": ("image", "ref_images", 9),
            "ref_video": ("video", "ref_videos", 3), "ref_audio": ("audio", "ref_audios", 3)}
    if slot not in spec:
        raise ValueError("绑定位置无效")
    want_kind, field, maximum = spec[slot]
    if kind != want_kind:
        raise ValueError(f"素材类型不匹配：{slot} 需要 {want_kind}")
    if slot == "first_image" and shot.get("mode") not in {"i2v", "fl2v"}:
        raise ValueError("首帧只在图生/首尾帧模式生效，请先改模式")
    if slot == "last_image" and shot.get("mode") != "fl2v":
        raise ValueError("尾帧只在首尾帧模式生效，请先改模式")
    if slot == "source_video" and shot.get("mode") not in {"v2v", "rv2v"}:
        raise ValueError("源视频只在视频编辑模式生效，请先改模式")
    if slot.startswith("ref_") and shot.get("mode") not in {"r2v", "rv2v"}:
        raise ValueError("参考素材只在多参/视频加参考模式生效，请先改模式")
    numbers = image_numbers(state, project)

    def mutator(target: dict, project_data: dict, data: dict) -> dict:
        if maximum == 1:
            target[field] = asset_id
        else:
            items = list(target.get(field) or [])
            if asset_id not in items:
                if len(items) >= maximum:
                    raise ValueError(f"此类参考素材最多 {maximum} 个")
                items.append(asset_id)
            target[field] = items
            if field == "ref_images":
                manual = list(target.get("manual_ref_images") or [])
                if asset_id not in manual:
                    manual.append(asset_id)
                target["manual_ref_images"] = manual
        return {"slot": slot, "asset": asset.get("name"),
                "picture": numbers.get(asset_id)}

    mutate_shot(ctx, shot_id, mutator)
    return {"summary": f"镜头 {index + 1} 绑定素材「{asset.get('name')}」到 {slot}",
            "data": {"slot": slot, "asset_id": asset_id, "picture": numbers.get(asset_id)}}


@tool("unbind_asset", RISK_WRITE,
      "解除镜头与素材的绑定（首帧、尾帧、源视频、参考图/视频/声音）。",
      {"shot_id": {"type": "string", "desc": "镜头 id；留空表示当前镜头"},
       "slot": {"type": "string", "desc": "要清空的位置"},
       "asset_id": {"type": "string", "desc": "只解绑这个素材；留空则清空该位置全部"}})
def _unbind_asset(ctx: ToolContext, args: dict) -> dict:
    shot, index = resolve_shot(ctx, args)
    shot_id = shot.get("id")
    slot = str(args.get("slot") or "")
    asset_id = str(args.get("asset_id") or "")
    spec = {"first_image": ("first_image",), "last_image": ("last_image",),
            "source_video": ("source_video",), "ref_image": ("ref_images",),
            "ref_video": ("ref_videos",), "ref_audio": ("ref_audios",)}
    if slot not in spec:
        raise ValueError("解绑位置无效")
    field = spec[slot][0]

    def mutator(target: dict, project_data: dict, data: dict) -> dict:
        if field in {"first_image", "last_image", "source_video"}:
            if asset_id and target.get(field) != asset_id:
                return {"removed": []}
            target[field] = None
            return {"removed": [field]}
        items = list(target.get(field) or [])
        keep = [item for item in items if item != asset_id] if asset_id else []
        removed = [item for item in items if item not in keep]
        target[field] = keep
        if field == "ref_images":
            manual = list(target.get("manual_ref_images") or [])
            target["manual_ref_images"] = [item for item in manual if item not in removed]
        return {"removed": removed}

    detail = mutate_shot(ctx, shot_id, mutator)
    return {"summary": f"解除镜头 {index + 1} 的 {slot} 绑定", "data": detail}


@tool("insert_reference_tag", RISK_WRITE,
      "在镜头提示词里插入 <Picture N> 标签（并同步参考图绑定），让参考图真正进入模型输入。",
      {"shot_id": {"type": "string", "desc": "镜头 id；留空表示当前镜头"},
       "asset_id": {"type": "string", "desc": "图片素材 id", "required": True},
       "replace_existing": {"type": "boolean", "desc": "是否先删掉已有的 <Picture N> 标签，默认 false"}})
def _insert_reference_tag(ctx: ToolContext, args: dict) -> dict:
    state = ctx.state()
    project = ctx.project()
    shot, index = resolve_shot(ctx, args)
    shot_id = shot.get("id")
    asset_id = str(args.get("asset_id") or "")
    asset = (state.get("assets") or {}).get(asset_id)
    if not asset or asset.get("project_id") != ctx.project_id:
        raise ValueError("素材不存在或不属于当前项目")
    if asset.get("kind") != "image":
        raise ValueError("只有图片素材有 <Picture N> 编号")
    numbers = image_numbers(state, project)
    number = numbers.get(asset_id)
    if not number:
        raise ValueError("这个素材还没有图片编号")
    old = str(shot.get("prompt") or "")
    tag = f"<Picture {number}>"
    replace = bool(args.get("replace_existing"))

    def mutator(target: dict, project_data: dict, data: dict) -> dict:
        prompt = str(target.get("prompt") or "")
        if replace:
            prompt = PICTURE_TAG.sub("", prompt).strip()
        if tag not in prompt:
            prompt = f"{prompt.rstrip()} {tag}".strip()
        target["prompt"] = prompt
        refs = list(target.get("ref_images") or [])
        if asset_id not in refs:
            refs.append(asset_id)
        target["ref_images"] = refs
        manual = list(target.get("manual_ref_images") or [])
        if asset_id not in manual:
            manual.append(asset_id)
        target["manual_ref_images"] = manual
        return {"prompt_before": old, "prompt_after": prompt, "tag": tag}

    detail = mutate_shot(ctx, shot_id, mutator)
    return {"summary": f"镜头 {index + 1} 插入 {tag}", "data": detail}


@tool("set_reference_role", RISK_WRITE,
      "设定某个参考素材在成片中的保留方式（弱参考/完整保留/部分保留/仅迁移特征）。这决定引擎怎么用它。",
      {"shot_id": {"type": "string", "desc": "镜头 id；留空表示当前镜头"},
       "asset_id": {"type": "string", "desc": "素材 id", "required": True},
       "role": {"type": "string", "desc": "image: auto/weak_reference/fully_preserved/partially_preserved/attribute_transfer；"
                                          "video: weak_reference/fully_preserved/partially_preserved/attribute_transfer；"
                                          "audio: reference/weak_reference/fully_copy/partially_copy", "required": True},
       "kind": {"type": "string", "desc": "image / video / audio，留空自动判断"}})
def _set_reference_role(ctx: ToolContext, args: dict) -> dict:
    state = ctx.state()
    shot, index = resolve_shot(ctx, args)
    shot_id = shot.get("id")
    asset_id = str(args.get("asset_id") or "")
    role = str(args.get("role") or "")
    asset = (state.get("assets") or {}).get(asset_id)
    if not asset:
        raise ValueError("素材不存在")
    kind = str(args.get("kind") or asset.get("kind") or "")
    if kind not in REFERENCE_ROLES:
        raise ValueError("素材类型无效")
    if role not in REFERENCE_ROLES[kind]:
        raise ValueError(f"{kind} 的角色只能是 {'/'.join(REFERENCE_ROLES[kind])}")
    key = {"image": "ref_image_roles", "video": "ref_video_roles",
           "audio": "ref_audio_roles"}[kind]

    def mutator(target: dict, project_data: dict, data: dict) -> dict:
        roles = dict(target.get(key) or {})
        roles[asset_id] = role
        target[key] = roles
        return {"role": role, "asset": asset.get("name")}

    detail = mutate_shot(ctx, shot_id, mutator)
    return {"summary": f"镜头 {index + 1} 的参考素材「{asset.get('name')}」设为 {role}",
            "data": detail}


@tool("set_asset_description", RISK_WRITE,
      "给图片素材写一句说明（这张图里是什么，例如「人物四视图身份参考」「跑道场景」）。引擎靠它判断参考图用途。",
      {"asset_id": {"type": "string", "desc": "素材 id", "required": True},
       "description": {"type": "string", "desc": "图中是什么", "required": True}})
def _set_asset_description(ctx: ToolContext, args: dict) -> dict:
    asset_id = str(args.get("asset_id") or "")
    text = str(args.get("description") or "").strip()[:600]
    state = ctx.state()
    asset = (state.get("assets") or {}).get(asset_id)
    if not asset:
        raise ValueError("素材不存在")

    def mutator(project: dict, data: dict, before: dict) -> dict:
        descriptions = dict(project.get("image_descriptions") or {})
        before["image_descriptions"] = copy.deepcopy(descriptions)
        descriptions[asset_id] = text
        project["image_descriptions"] = descriptions
        return {"description": text}

    detail = mutate_project(ctx, mutator)
    return {"summary": f"素材「{asset.get('name')}」已写说明", "data": detail}


@tool("create_shots", RISK_WRITE,
      "新建镜头（可以一次建多个）。由你自己写好每个镜头的提示词。",
      {"shots": {"type": "array",
                 "desc": "镜头数组，每项 {name, prompt, duration, mode}；mode 用 t2v/i2v/fl2v/r2v/v2v/rv2v",
                 "required": True}})
def _create_shots(ctx: ToolContext, args: dict) -> dict:
    items = args.get("shots")
    if not isinstance(items, list) or not items:
        raise ValueError("请提供镜头数组")
    project = ctx.project()
    if len(project.get("shots") or []) + len(items) > 100:
        raise ValueError("项目镜头数已达上限 100")
    created: list[dict] = []

    def apply(data: dict) -> dict:
        target = data["projects"].get(ctx.project_id)
        shots = target.get("shots") or []
        for item in items:
            if not isinstance(item, dict):
                continue
            shot = new_shot()
            shot["id"] = uid()
            shot["name"] = str(item.get("name") or "新镜头").strip()[:100]
            mode = str(item.get("mode") or "t2v")
            if mode not in ALL_MODES:
                raise ValueError(f"生成模式无效：{mode}")
            shot["mode"] = mode
            shot["prompt"] = str(item.get("prompt") or "")[:6000]
            try:
                duration = float(item.get("duration") or 5)
            except (TypeError, ValueError):
                duration = 5
            shot["duration"] = max(2.0, min(15.0, duration))
            shot["selected"] = True
            shots.append(shot)
            created.append({"id": shot["id"], "name": shot["name"]})
        target["_data_version"] = int(target.get("_data_version", 0)) + 1
        return {"project": copy.deepcopy(target)}

    change(apply)
    for shot in created:
        record_change(ctx.run, {"kind": "shot", "shot_id": shot["id"], "index": None,
                                 "before": None, "after": None})
    ctx.invalidate()
    return {"summary": f"新建 {len(created)} 个镜头", "data": {"created": created}}


@tool("update_character", RISK_WRITE,
      "修改角色设定：姓名、外观描述、声音描述。",
      {"character_id": {"type": "string", "desc": "角色 id", "required": True},
       "name": {"type": "string", "desc": "新姓名"},
       "description": {"type": "string", "desc": "外观描述"},
       "voice_description": {"type": "string", "desc": "声音描述"}})
def _update_character(ctx: ToolContext, args: dict) -> dict:
    character_id = str(args.get("character_id") or "")
    project = ctx.project()
    character = next((item for item in project.get("characters") or []
                      if item.get("id") == character_id), None)
    if not character:
        raise ValueError("角色不存在")

    def mutator(target: dict, data: dict, before: dict) -> dict:
        item = next((entry for entry in target.get("characters") or []
                     if entry.get("id") == character_id), None)
        if not item:
            raise ValueError("角色不存在")
        before["characters"] = copy.deepcopy(target.get("characters"))
        for key in ("name", "description", "voice_description", "appearance",
                    "character_form", "speaker_id", "multiview_layout", "multiview_mode",
                    "multiview_prompt", "multiview_engine", "xb_skill_mode",
                    "xb_aspect_ratio", "xb_sampling_mode", "xb_output_lang",
                    "xb_preset_text", "xb_extra_settings"):
            if args.get(key) not in (None, ""):
                item[key] = str(args.get(key))[:1000]
        for key in ("portrait_asset", "multiview_asset", "voice_asset"):
            if args.get(key) not in (None, ""):
                item[key] = str(args.get(key))
        for key in ("krea_width", "krea_height", "xb_width", "xb_height", "xb_seed"):
            if args.get(key) not in (None, ""):
                try:
                    item[key] = int(args.get(key))
                except (TypeError, ValueError):
                    raise ValueError(f"{key} 必须是数字")
        for key in ("xb_thinking", "xb_green_background"):
            if args.get(key) not in (None, ""):
                item[key] = (str(args.get(key)).lower() in ("true", "1", "yes")
                             or args.get(key) is True)
        return {"character": copy.deepcopy(item)}

    detail = mutate_project(ctx, mutator)
    return {"summary": f"更新角色「{character.get('name')}」", "data": detail}


@tool("assign_character", RISK_WRITE,
      "把角色绑定到某个镜头（或解绑）。多参模式下会带上该角色的形象素材。",
      {"shot_id": {"type": "string", "desc": "镜头 id；留空表示当前镜头"},
       "character_id": {"type": "string", "desc": "角色 id", "required": True},
       "enabled": {"type": "boolean", "desc": "true 绑定 / false 解绑，默认 true"}})
def _assign_character(ctx: ToolContext, args: dict) -> dict:
    shot, index = resolve_shot(ctx, args)
    shot_id = shot.get("id")
    character_id = str(args.get("character_id") or "")
    enabled = args.get("enabled", True)
    project = ctx.project()
    character = next((item for item in project.get("characters") or []
                      if item.get("id") == character_id), None)
    if not character:
        raise ValueError("角色不存在")

    def mutator(target: dict, project_data: dict, data: dict) -> dict:
        assigned = list(target.get("character_ids") or [])
        if enabled and character_id not in assigned:
            assigned.append(character_id)
        if not enabled:
            assigned = [item for item in assigned if item != character_id]
        target["character_ids"] = assigned
        return {"character": character.get("name"), "enabled": bool(enabled)}

    detail = mutate_shot(ctx, shot_id, mutator)
    return {"summary": f"镜头 {index + 1} {'绑定' if enabled else '解绑'}角色「{character.get('name')}」",
            "data": detail}


@tool("set_shot_order", RISK_WRITE,
      "调整镜头在时间轴上的顺序。",
      {"shot_id": {"type": "string", "desc": "镜头 id", "required": True},
       "position": {"type": "number", "desc": "目标位置（从 1 开始）", "required": True}})
def _set_shot_order(ctx: ToolContext, args: dict) -> dict:
    shot, index = resolve_shot(ctx, args)
    shot_id = shot.get("id")
    try:
        position = int(args.get("position"))
    except (TypeError, ValueError):
        raise ValueError("目标位置必须是镜头序号") from None
    shots = ctx.project().get("shots") or []
    if not 1 <= position <= len(shots):
        raise ValueError("目标位置超出镜头数量")

    outcome: dict = {}

    def apply(data: dict) -> dict:
        project = data["projects"].get(ctx.project_id)
        items = project.get("shots") or []
        current = next((i for i, item in enumerate(items) if item.get("id") == shot_id), None)
        if current is None:
            raise ValueError("镜头不存在")
        before_order = [item.get("id") for item in items]
        moving = items.pop(current)
        items.insert(position - 1, moving)
        project["_data_version"] = int(project.get("_data_version", 0)) + 1
        outcome["record"] = {"kind": "order", "before": before_order,
                             "after": [item.get("id") for item in items]}
        outcome["detail"] = {"position": position}
        outcome["project"] = copy.deepcopy(project)
        return {"project": outcome["project"]}

    change(apply)
    if outcome.get("record"):
        record_change(ctx.run, outcome["record"])
    ctx.invalidate()
    return outcome.get("detail") or {}


@tool("set_shot_scene", RISK_WRITE,
      "把镜头归到某个场景（scene）。",
      {"shot_id": {"type": "string", "desc": "镜头 id", "required": True},
       "scene_id": {"type": "string", "desc": "场景 id；留空表示移出场景"}})
def _set_shot_scene(ctx: ToolContext, args: dict) -> dict:
    shot, index = resolve_shot(ctx, args)
    shot_id = shot.get("id")
    scene_id = str(args.get("scene_id") or "")

    def mutator(target: dict, project_data: dict, data: dict) -> dict:
        target["scene_id"] = scene_id
        return {"scene_id": scene_id}

    detail = mutate_shot(ctx, shot_id, mutator)
    return {"summary": f"镜头 {index + 1} 归入场景 {scene_id or '（无）'}", "data": detail}


# --------------------------------------------------------------------------
# 删除类（必须逐次确认）
# --------------------------------------------------------------------------

@tool("delete_shot", RISK_DESTRUCTIVE,
      "删除一个镜头（素材与历史成片不会删除）。不可撤销。",
      {"shot_id": {"type": "string", "desc": "镜头 id", "required": True}})
def _delete_shot(ctx: ToolContext, args: dict) -> dict:
    shot, index = resolve_shot(ctx, args)
    shot_id = shot.get("id")
    name = shot.get("name")
    removed: dict = {}

    def apply(data: dict) -> dict:
        project = data["projects"].get(ctx.project_id)
        shots = project.get("shots") or []
        position = next((i for i, item in enumerate(shots) if item.get("id") == shot_id), None)
        if position is None:
            raise ValueError("镜头不存在")
        if len(shots) <= 1:
            raise ValueError("项目至少保留一个镜头")
        removed["name"] = name
        shots.pop(position)
        project["_data_version"] = int(project.get("_data_version", 0)) + 1
        return {"project": copy.deepcopy(project)}

    change(apply)
    ctx.invalidate()
    return {"summary": f"已删除镜头 {index + 1}「{name}」", "data": {"shot_id": shot_id}}


@tool("delete_asset", RISK_DESTRUCTIVE,
      "删除一个素材，并清除它在镜头、角色与共享参考里的绑定。不可撤销。",
      {"asset_id": {"type": "string", "desc": "素材 id", "required": True}})
def _delete_asset(ctx: ToolContext, args: dict) -> dict:
    from server.routes.assets import _asset_in_use, _remove_asset, _renumber_after_image_removal
    asset_id = str(args.get("asset_id") or "")
    state = ctx.state()
    asset = (state.get("assets") or {}).get(asset_id)
    if not asset or asset.get("project_id") != ctx.project_id:
        raise ValueError("素材不存在或不属于当前项目")
    if _asset_in_use(state, asset_id):
        raise ValueError("素材正被生成任务使用，请任务结束后再删除")

    def apply(data: dict) -> dict:
        removed, references = _remove_asset(data, asset_id, ctx.project_id)
        if removed and removed.get("kind") == "image":
            _renumber_after_image_removal(data, ctx.project_id)
        return {"removed": removed, "references": references}

    result = change(apply)
    removed = result.get("removed") or {}
    if removed.get("file"):
        (ASSETS / Path(removed["file"]).name).unlink(missing_ok=True)
    ctx.invalidate()
    return {"summary": f"已删除素材「{asset.get('name')}」，清理 {result.get('references', 0)} 处引用",
            "data": {"asset_id": asset_id, "references_removed": result.get("references", 0)}}


@tool("batch_delete_assets", RISK_DESTRUCTIVE,
      "批量删除素材，并清除所有引用。不可撤销。",
      {"asset_ids": {"type": "array", "desc": "素材 id 列表", "required": True}})
def _batch_delete_assets(ctx: ToolContext, args: dict) -> dict:
    from server.routes.assets import _asset_in_use, _remove_asset, _renumber_after_image_removal
    wanted = [str(item) for item in (args.get("asset_ids") or [])]
    if not wanted:
        raise ValueError("请提供素材 id 列表")
    state = ctx.state()
    for asset_id in wanted:
        asset = (state.get("assets") or {}).get(asset_id)
        if not asset or asset.get("project_id") != ctx.project_id:
            raise ValueError("素材不存在或不属于当前项目")
        if _asset_in_use(state, asset_id):
            raise ValueError(f"素材「{asset.get('name')}」正被生成任务使用")

    def apply(data: dict) -> dict:
        files, references = [], 0
        for asset_id in wanted:
            removed, count = _remove_asset(data, asset_id, ctx.project_id)
            if removed:
                files.append(removed.get("file"))
            references += count
        _renumber_after_image_removal(data, ctx.project_id)
        return {"files": files, "references": references}

    result = change(apply)
    for filename in result.get("files") or []:
        if filename:
            (ASSETS / Path(filename).name).unlink(missing_ok=True)
    ctx.invalidate()
    return {"summary": f"已删除 {len(result.get('files') or [])} 个素材，清理 {result.get('references', 0)} 处引用",
            "data": {"deleted": len(result.get("files") or []),
                     "references_removed": result.get("references", 0)}}


@tool("delete_character", RISK_DESTRUCTIVE,
      "删除一个角色设定。不可撤销。",
      {"character_id": {"type": "string", "desc": "角色 id", "required": True}})
def _delete_character(ctx: ToolContext, args: dict) -> dict:
    character_id = str(args.get("character_id") or "")
    project = ctx.project()
    character = next((item for item in project.get("characters") or []
                      if item.get("id") == character_id), None)
    if not character:
        raise ValueError("角色不存在")

    def apply(data: dict) -> dict:
        target = data["projects"].get(ctx.project_id)
        target["characters"] = [item for item in target.get("characters") or []
                                if item.get("id") != character_id]
        for shot in target.get("shots") or []:
            shot["character_ids"] = [item for item in (shot.get("character_ids") or [])
                                     if item != character_id]
        target["_data_version"] = int(target.get("_data_version", 0)) + 1
        return {"project": copy.deepcopy(target)}

    change(apply)
    ctx.invalidate()
    return {"summary": f"已删除角色「{character.get('name')}」", "data": {"character_id": character_id}}


# --------------------------------------------------------------------------
# 渲染（必须逐次确认）
# --------------------------------------------------------------------------

@tool("submit_render", RISK_GPU,
      "把选中的镜头提交到 ComfyUI 生成队列。会占用显卡、排队数分钟。",
      {"shot_ids": {"type": "array", "desc": "要渲染的镜头 id；留空表示所有已选中镜头"}})
def _submit_render(ctx: ToolContext, args: dict) -> dict:
    from runner import run_project
    project = ctx.project()
    shots = project.get("shots") or []
    known = {shot.get("id") for shot in shots}
    requested = [str(item) for item in (args.get("shot_ids") or []) if str(item) in known]
    if not requested:
        requested = [shot.get("id") for shot in shots if shot.get("selected", True)]
    if not requested:
        requested = [shot.get("id") for shot in shots]
    if not requested:
        raise ValueError("当前项目没有可渲染的镜头")
    run_id = uid()

    def apply(data: dict) -> dict:
        target = data["projects"].get(ctx.project_id)
        for shot in target.get("shots") or []:
            if shot.get("id") in requested:
                shot["selected"] = True
        runs = data.setdefault("runs", {})
        runs[run_id] = {"id": run_id, "project_id": ctx.project_id, "shot_ids": requested,
                        "status": "queued", "current": "", "results": [], "created_at": time.time()}
        return {"run_id": run_id}

    change(apply)
    ctx.invalidate()
    threading.Thread(target=run_project, args=(run_id,), daemon=True).start()
    names = [shot.get("name") or f"镜头{index + 1}" for index, shot in enumerate(shots)
             if shot.get("id") in requested][:5]
    return {"summary": f"已提交 {len(requested)} 个镜头渲染：{'、'.join(names)}",
            "data": {"run_id": run_id, "shot_ids": requested}}


# --------------------------------------------------------------------------
# 工具目录（给模型看的）
# --------------------------------------------------------------------------

def tool_catalog_prompt() -> str:
    lines = []
    for spec in TOOLS.values():
        params = []
        for key, meta in spec["args"].items():
            detail = meta.get("desc", "")
            if meta.get("enum"):
                detail += f"（只能取 {'/'.join(meta['enum'])}）"
            params.append(f"{key}: {detail}")
        head = f"- {spec['name']}［{RISK_LABEL.get(spec['risk'], spec['risk'])}］{spec['summary']}"
        if params:
            head += "｜参数：" + "；".join(params)
        lines.append(head)
    return "\n".join(lines)


def tool_specs() -> list[dict]:
    return [{"name": spec["name"], "risk": spec["risk"], "summary": spec["summary"],
             "args": spec["args"]} for spec in TOOLS.values()]

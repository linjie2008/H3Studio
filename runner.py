"""Run independent shot plans through ComfyUI's queue API."""

from __future__ import annotations

import copy
import functools
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from engine import build_h3_graph, build_seedvr2_graph, director_control_text
from memory_plan import fit_canvas, plan_canvas
from reference_numbering import normalized_project_image_references
from store import ASSETS, OUTPUTS, change, read
from video_tools import dimensions as video_dimensions
from video_tools import duration as video_duration


COMFY_URL = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188").rstrip("/")
COMFY_INPUT = Path(os.environ.get("COMFYUI_INPUT_DIR", "/home/administrator/ComfyUI/input"))
NEW_IDENTITY_CONTEXT_FRAMES = 8
NEW_IDENTITY_VISIBLE_FRAMES = 4
# A spoken seam keeps every audio sample, so the duplicated guide head may only
# be dropped when that head provably carries no speech. Peak level above this
# threshold is treated as content and the segment is appended untouched.
SPOKEN_OVERLAP_MAX_DBFS = -26.0
# H3 gets the guide as video only, so it starts the voiceover at frame zero even
# when the prompt reserves the opening window. A long guide therefore forces a
# choice between cutting words and replaying a long head; narrated handoffs use
# the shortest valid H3 guide instead (at most one frame is dropped).
NARRATED_HANDOFF_FRAMES = 5
NARRATED_HANDOFF_VISIBLE = 4
INLINE_VOICEOVER_RE = re.compile(
    # Negated forms such as 无旁白 / 没有旁白 / 不需要旁白 are instructions, not lines.
    r'(?<!无)(?<!没有)(?<!不需)(?<!不要)(?<!不需要)'
    r'(?:旁白(?:响起|说道|说)?|画外音|解说(?:词)?|配音|voice[\s-]?over|narration)\s*'
    r'(?:[:：]\s*)?'
    r'(?:“(?P<cn>[^”\n]+)”|「(?P<corner>[^」\n]+)」|『(?P<double_corner>[^』\n]+)』|'
    r'"(?P<ascii>[^"\n]+)"|'
    # Unquoted form: 旁白：城市与自然在这里相遇。 — the front end already
    # rewrites this shape, and projects written that way must still become <d>.
    r'(?<=[:：])\s*(?P<plain>[^。！？!?\n]+[。！？!?]?)|'
    # Unbalanced opening quote (旁白"只有开头没有收尾): take the rest of the line,
    # otherwise the line survives inside the visual description and H3 speaks it.
    r'[“「『"](?P<unclosed>[^”」』"\n]+[。！？!?]?))', re.IGNORECASE)
# Quote marks and spacing that may wrap or trail an authored narration line.
_NARRATION_TRIM = " \t\r\n\"'“”「」『』"
_NARRATION_LEFTOVER_RE = re.compile(r"[ \t]*旁白[ \t]*[\"“「『:]?[ \t]*(?=\n|$)")
# Directives such as 旁白：无 / narration: none mean "no narration", not a line.
NARRATION_PLACEHOLDERS = {"无", "无旁白", "没有", "莫", "留空", "不需要", "暂无",
                         "none", "n/a", "na", "empty", "-"}


def _inline_voiceover(prompt: str) -> tuple[str, str]:
    """Separate explicitly authored narration from the visual direction."""
    lines = []
    seen = set()
    def take(match: re.Match) -> str:
        line = next(value for value in match.groupdict().values() if value is not None)
        line = line.strip(_NARRATION_TRIM)
        if line.lower() in NARRATION_PLACEHOLDERS:
            # Drop the directive from the visual prompt but do not speak it.
            return ""
        if line and line not in seen:
            lines.append(line)
            seen.add(line)
        return ""
    source = str(prompt or "")
    visual = INLINE_VOICEOVER_RE.sub(take, source)
    # A duplicated or partially quoted line can survive the substitution; H3 then
    # reads it as scene content, which is what made narration sound garbled.
    for line in lines:
        if line and line in visual:
            visual = visual.replace(line, "")
    if lines:
        # Drop a marker left dangling at a line end after its text was removed.
        visual = _NARRATION_LEFTOVER_RE.sub("", visual)
    visual = re.sub(r"[ \t]+([。.!?])", r"\1", visual)
    return visual.strip(), "\n".join(line for line in lines if line)


def _has_spoken_audio(shot: dict) -> bool:
    return bool(str(shot.get("dialogue_text") or "").strip()
                or _inline_voiceover(shot.get("prompt") or "")[1]) and shot.get("audio_mode") != "mute"


def _speech_reaches_handoff(shot: dict, *, at_end: bool, overlap_frames: int) -> bool:
    """Protect a seam unless authored speech timing leaves room for its overlap."""
    if not _has_spoken_audio(shot):
        return False
    timing = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*(?:-|~|至)\s*"
                          r"(\d+(?:\.\d+)?)\s*(?:s|秒)?\s*",
                          str(shot.get("dialogue_time") or ""), re.IGNORECASE)
    if not timing:
        return True
    start, end = map(float, timing.groups())
    if start >= end:
        return True
    quiet_margin = max(0, overlap_frames) / 24 + 0.2
    return end > float(shot.get("duration") or 5) - quiet_margin if at_end else start < quiet_margin


def _preserve_spoken_seam(previous: dict, current: dict, overlap_frames: int) -> bool:
    return (_speech_reaches_handoff(previous, at_end=True, overlap_frames=overlap_frames)
            or _speech_reaches_handoff(current, at_end=False, overlap_frames=overlap_frames))


# Serialize submissions across worker threads so a model-family switch cannot
# race another prompt between the idle check, cache release and submission.
_MODEL_SWITCH_LOCK = threading.Lock()
_DIRECTOR_RUN_LOCK = threading.Lock()
_ACTIVE_MODEL_FAMILY: str | None = None
_ACTIVE_MODEL_RUN_ID: str | None = None


def _graph_model_family(graph: dict) -> str:
    """Return the heavyweight model family used by a ComfyUI API graph."""
    classes = {str(node.get("class_type") or "") for node in graph.values()
               if isinstance(node, dict)}
    if any(name.startswith("SeedVR2") for name in classes):
        return "seedvr2"
    if any("MiniMaxH3" in name or name.startswith("H3Studio") for name in classes):
        return "h3"
    if any(name.startswith("Krea2Edit") for name in classes):
        return "krea2"
    if "XB_QwenPromptPreset" in classes or "TextEncodeQwenImage21" in classes:
        return "qwen_image"
    return "other"


def _model_family_label(family: str | None) -> str:
    return {
        None: "已有模型缓存", "h3": "H3 视频模型", "seedvr2": "SeedVR2 超分模型",
        "qwen_image": "Qwen 图像模型",
        "krea2": "Krea2 图像模型",
        "other": "其他模型",
    }.get(family, str(family))


def api(path: str, body: dict | None = None) -> dict:
    payload = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        COMFY_URL + path, data=payload,
        headers={"Content-Type": "application/json"} if payload else {},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def update_run(run_id: str, **fields) -> None:
    change(lambda data: data["runs"][run_id].update(fields))


def _serialized_director_run(worker):
    """Keep director GPU finishing passes separate from later ComfyUI jobs."""
    @functools.wraps(worker)
    def wrapped(run_id: str):
        if _DIRECTOR_RUN_LOCK.locked() and read()["runs"][run_id].get("status") == "queued":
            update_run(run_id, current="等待上一个导演台任务完成")
        with _DIRECTOR_RUN_LOCK:
            if read()["runs"][run_id].get("cancel_requested"):
                update_run(run_id, status="cancelled", current="已取消", finished_at=time.time())
                return
            return worker(run_id)
    return wrapped


def prepare_shot(original: dict, project: dict) -> dict:
    """Apply project refs, common prompt, and selected character and entity profiles."""
    catalog = read()["assets"]
    project = normalized_project_image_references(project, catalog)
    original = next((item for item in project.get("shots", [])
                     if item.get("id") == original.get("id")), original)
    shot = copy.deepcopy(original)
    visual_prompt, inline_narration = _inline_voiceover(shot.get("prompt") or "")
    if inline_narration:
        # Never send a second narration instruction in the visual prompt.
        duplicated = max((str(original.get("prompt") or "").count(line) for line in
                          inline_narration.split("\n")), default=0)
        if duplicated > 1:
            # The same line appears twice (often with mismatched quotes); H3 then
            # repeats or slurs it, so surface it instead of silently deduping.
            shot["_h3_narration_duplicated"] = duplicated
        shot["prompt"] = visual_prompt
        existing_dialogue = str(shot.get("dialogue_text") or "").strip()
        if not existing_dialogue:
            shot["dialogue_text"] = inline_narration
            shot["dialogue_voiceover"] = True
            shot["speaker_description"] = "全片同一位普通话旁白，音色、口音和语速保持一致"
            shot["_h3_inline_voiceover"] = True
    image_numbers = dict(project.get("reference_image_numbers") or {})
    if not image_numbers:
        image_numbers = {asset_id: index for index, (asset_id, asset) in enumerate(
            ((key, value) for key, value in catalog.items()
             if value.get("kind") == "image" and value.get("project_id") == project.get("id")), 1)}
    else:
        next_number = max((int(value) for value in image_numbers.values()
                           if str(value).isdigit()), default=0) + 1
        for asset_id, asset in catalog.items():
            if (asset.get("kind") == "image" and asset.get("project_id") == project.get("id")
                    and asset_id not in image_numbers):
                image_numbers[asset_id] = next_number
                next_number += 1
    authored_text = "\n".join((str(project.get("shared_prompt") or ""),
                               str(original.get("prompt") or "")))
    requested_numbers = {int(match.group(1) or match.group(2)) for match in
                         re.finditer(r"<Picture\s+(\d+)>|@图片\s*(\d+)", authored_text, re.IGNORECASE)}
    number_to_asset = {}
    for asset_id, number in image_numbers.items():
        asset = catalog.get(asset_id, {})
        if (str(number).isdigit() and asset.get("kind") == "image"
                and asset.get("project_id") in (None, project.get("id"))):
            value = int(number)
            if value in requested_numbers and value in number_to_asset and number_to_asset[value] != asset_id:
                raise ValueError(f"镜头 {shot.get('name', '')} 的图片编号 {value} 对应多张素材，请刷新素材编号")
            number_to_asset.setdefault(value, asset_id)
    missing_numbers = sorted(requested_numbers - number_to_asset.keys())
    if missing_numbers and shot.get("mode") in {"r2v", "rv2v"}:
        raise ValueError(f"镜头 {shot.get('name', '')} 引用了不存在的图片编号：{missing_numbers[0]}")
    requested_assets = {number_to_asset[number] for number in requested_numbers if number in number_to_asset}
    # Gallery selection is only a visual planning marker. Only an authored
    # picture tag binds a shot-level image to H3; old ref_images/manual lists
    # must not silently introduce unmentioned people or objects.  Video editing
    # needs the same rule: a leftover gallery image would occupy a slot the
    # prompt then reads as a different reference.
    if shot.get("mode") in {"r2v", "rv2v"}:
        shot["ref_images"] = [asset_id for asset_id in shot.get("ref_images") or []
                              if asset_id in requested_assets]
    for source, destination in (("images", "ref_images"), ("videos", "ref_videos"), ("audios", "ref_audios")):
        shot[destination] = list(dict.fromkeys(
            list(project.get("shared_refs", {}).get(source, [])) + list(shot.get(destination) or [])))
    if shot.get("mode") in {"t2v", "i2v", "fl2v"}:
        # These H3 modes wire only their explicit first/last keyframes. Old
        # gallery selections and project-wide reference lists must not be
        # treated as live media inputs by downstream prompt preparation.
        shot["ref_images"] = []
        shot["ref_videos"] = []
        shot["ref_audios"] = []
    if shot.get("audio_mode") == "mute":
        shot["ref_audios"] = []
    characters = {item.get("id"): item for item in project.get("characters", [])}
    selected_profiles = [characters[item] for item in shot.get("character_ids", []) if item in characters]
    # A single explicit person image in a shot takes precedence over a broad
    # character selection inherited from an earlier group shot. Otherwise all
    # selected faces are sent to H3 and marked as appearing in this segment.
    if len(selected_profiles) > 1:
        authored_numbers = {int(match.group(1) or match.group(2)) for match in
                            re.finditer(r"<Picture\s+(\d+)>|@图片\s*(\d+)",
                                        str(original.get("prompt") or ""), re.IGNORECASE)}
        authored_assets = {number_to_asset[number] for number in authored_numbers
                           if number in number_to_asset}
        matched_profiles = [profile for profile in selected_profiles
                            if profile.get("portrait_asset") in authored_assets
                            or profile.get("multiview_asset") in authored_assets]
        if len(matched_profiles) == 1:
            shot["_h3_ignored_character_ids"] = [profile["id"] for profile in selected_profiles
                                                  if profile["id"] != matched_profiles[0]["id"]]
            selected_profiles = matched_profiles
            shot["character_ids"] = [matched_profiles[0]["id"]]
    if shot.get("dialogue_text") and len(selected_profiles) == 1 and not shot.get("_h3_inline_voiceover"):
        profile_speaker = str(selected_profiles[0].get("speaker_id") or "S1")
        if shot.get("speaker_id", "S1") == "S1" and profile_speaker != "S1":
            shot["speaker_id"] = profile_speaker
        if shot.get("speaker_description", "画面中的人物") == "画面中的人物":
            profile_name = str(selected_profiles[0].get("name") or "画面中的人物")
            profile_appearance = str(selected_profiles[0].get("appearance") or "").strip()
            shot["speaker_description"] = profile_name + ("，" + profile_appearance if profile_appearance else "")
    if shot.get("dialogue_text") and selected_profiles and not shot.get("_h3_inline_voiceover"):
        profile_speakers = {str(item.get("speaker_id") or "S1") for item in selected_profiles}
        if str(shot.get("speaker_id") or "S1") not in profile_speakers:
            shot["dialogue_profile_speaker_mismatch"] = True
    annotations = []
    shot_index = next((i for i, item in enumerate(project.get("shots", [])) if item.get("id") == shot.get("id")), -1)
    previous_shot = project["shots"][shot_index - 1] if shot_index > 0 else None
    next_shot = project["shots"][shot_index + 1] if 0 <= shot_index < len(project.get("shots", [])) - 1 else None
    shot["dialogue_carries_in"] = bool(previous_shot and previous_shot.get("dialogue_carries_out")
                                        and shot.get("dialogue_text"))
    shot["dialogue_carries_out_valid"] = bool(shot.get("dialogue_carries_out") and next_shot
                                              and _has_spoken_audio(next_shot))
    if shot["dialogue_carries_in"] and previous_shot.get("speaker_id", "S1") != shot.get("speaker_id", "S1"):
        shot["dialogue_carry_speaker_mismatch"] = True
    if shot["dialogue_carries_out_valid"] and next_shot.get("speaker_id", "S1") != shot.get("speaker_id", "S1"):
        shot["dialogue_carry_speaker_mismatch"] = True
    reference_modes = {"r2v", "rv2v"}
    for character_id in shot.get("character_ids", []):
        character = characters.get(character_id)
        if not character:
            raise ValueError(f"镜头 {shot.get('name', '')} 选择了不存在的角色档案")
        name = str(character.get("name") or "未命名角色").strip()
        appearance = str(character.get("appearance") or "").strip()
        voice_description = str(character.get("voice_description") or "").strip()
        portrait_id = character.get("portrait_asset")
        multiview_id = character.get("multiview_asset")
        voice_id = character.get("voice_asset")
        visual_tag = ""
        voice_tag = ""
        if shot.get("mode") in reference_modes:
            if portrait_id:
                if portrait_id not in shot["ref_images"]:
                    shot["ref_images"].append(portrait_id)
                visual_tag = f"<Picture {shot['ref_images'].index(portrait_id) + 1}>"
            if multiview_id:
                if multiview_id not in shot["ref_images"]:
                    shot["ref_images"].append(multiview_id)
                multiview_tag = f"<Picture {shot['ref_images'].index(multiview_id) + 1}>"
            if voice_id and shot.get("audio_mode") != "mute":
                if voice_id not in shot["ref_audios"]:
                    shot["ref_audios"].append(voice_id)
                voice_tag = f"<Audio {shot['ref_audios'].index(voice_id) + 1}>"
        parts = [f"角色“{name}”保持身份一致。"]
        if appearance:
            parts.append("外观：" + appearance)
        if visual_tag:
            parts.append("外观参考 " + visual_tag)
        if shot.get("mode") in reference_modes and multiview_id:
            view_label = {"four": "四视图", "five": "五视图", "six": "六视图"}.get(
                character.get("multiview_layout"), "四视图")
            parts.append("同一角色正面、侧面和背面外观参考（" + view_label + "）" + multiview_tag
                         + "；仅用于锁定身份与服装，不要在视频中出现多宫格或分栏，实际朝向遵循镜头提示词")
        if voice_description:
            parts.append("声音：" + voice_description)
        if voice_tag:
            parts.append("声音参考 " + voice_tag)
        annotations.append("；".join(parts))
    shot["_h3_characters"] = [{
        "subject_number": int(characters[item].get("subject_number") or list(characters).index(item) + 1),
        "name": str(characters[item].get("name") or "角色"),
        "appearance": str(characters[item].get("appearance") or "").strip(),
        "portrait_asset": characters[item].get("portrait_asset"),
        "multiview_asset": characters[item].get("multiview_asset"),
        "multiview_layout": characters[item].get("multiview_layout", "four"),
        "voice_description": str(characters[item].get("voice_description") or "").strip(),
        "voice_asset": characters[item].get("voice_asset"),
        "speaker_id": str(characters[item].get("speaker_id") or "S1"),
    } for item in shot.get("character_ids", []) if item in characters]
    entities = {item.get("id"): item for item in project.get("entities", [])}
    entity_annotations = []
    selected_entities = []
    for entity_id in shot.get("entity_ids", []):
        entity = entities.get(entity_id)
        if not entity:
            raise ValueError(f"镜头 {shot.get('name', '')} 选择了不存在的场景或道具档案")
        selected_entities.append(entity)
        kind = entity.get("kind", "prop")
        label = {"prop": "道具", "place": "地点", "style": "画风"}.get(kind, "设定")
        image_tags = []
        for asset_id in entity.get("image_assets", []):
            if shot.get("mode") in reference_modes:
                if asset_id not in shot["ref_images"]:
                    shot["ref_images"].append(asset_id)
                image_tags.append(f"<Picture {shot['ref_images'].index(asset_id) + 1}>")
        description = str(entity.get("description") or "").strip()
        text = f"{label}“{str(entity.get('name') or '未命名')}”在本片保持一致"
        if description:
            text += "：" + description
        if image_tags:
            text += "；视觉参考 " + "、".join(image_tags)
        entity_annotations.append(text + "。")
    shot["_h3_entities"] = [{
        "id": item.get("id"), "kind": item.get("kind", "prop"),
        "name": str(item.get("name") or ""),
        "description": str(item.get("description") or ""),
        "image_assets": list(item.get("image_assets") or []),
    } for item in selected_entities]
    shot["_h3_reference_names"] = {
        asset_id: str(catalog.get(asset_id, {}).get("name") or asset_id)
        for asset_id in set(shot.get("ref_images", []) + shot.get("ref_videos", []) + shot.get("ref_audios", [])
                            + ([shot["source_video"]] if shot.get("source_video") else []))
    }
    if len(shot["ref_images"]) > 9 or len(shot["ref_videos"]) > 3 or len(shot["ref_audios"]) > 3:
        raise ValueError(f"镜头 {shot.get('name', '')} 的共享素材与角色档案引用超过 H3 上限（图片 9、视频 3、声音 3）")
    # Prompts use project-wide image numbers so that the same @ reference keeps
    # its identity across shots. H3 expects slot numbers local to this shot, so
    # translate only user-authored text after assembling this shot's final refs.
    # A typed global alias such as <Picture 4> or @图片4 must bind its asset to
    # this shot. Otherwise the prompt contains a label H3 cannot resolve because
    # no corresponding image was actually sent in ref_images.
    # A typed global alias such as <Picture 4>, @图片4 or a plain 图片4 must bind
    # its asset to this shot. Otherwise the prompt contains a label H3 cannot
    # resolve because no corresponding image was actually sent in ref_images,
    # and the number silently points at whichever image owns that slot.
    if shot.get("mode") in reference_modes:
        for number in sorted(requested_numbers):
            asset_id = number_to_asset.get(number)
            if asset_id and catalog.get(asset_id, {}).get("kind") == "image":
                if asset_id not in shot["ref_images"]:
                    shot["ref_images"].append(asset_id)
                # An @ tag binds an image to the prompt; it does not mean the
                # source image's entire composition should be copied into video.
                # Let the prompt compiler infer identity / scene / object use.
                shot.setdefault("ref_image_roles", {}).setdefault(asset_id, "auto")
    # Explicit <Picture N>/@图片N aliases above can add references after the
    # shared/profile limit check.  Validate the final payload as well; never
    # let engine.py silently truncate a prompt-visible reference.
    if len(shot["ref_images"]) > 9 or len(shot["ref_videos"]) > 3 or len(shot["ref_audios"]) > 3:
        raise ValueError(f"镜头 {shot.get('name', '')} 的最终参考素材超过 H3 上限（图片 9、视频 3、声音 3）")
    shot["_h3_reference_descriptions"] = {
        asset_id: str(description).strip()[:600]
        for asset_id, description in (project.get("image_descriptions") or {}).items()
        if asset_id in shot.get("ref_images", []) and str(description).strip()
    }
    local_image_numbers = {asset_id: index for index, asset_id in enumerate(shot.get("ref_images", []), 1)}
    shot["_h3_image_reference_map"] = [
        {"global_number": int(image_numbers[asset_id]), "local_number": local_number,
         "asset_id": asset_id, "name": str(catalog.get(asset_id, {}).get("name") or asset_id)}
        for asset_id, local_number in local_image_numbers.items()
        if str(image_numbers.get(asset_id, "")).isdigit()
    ]
    global_to_local = {int(number): local_image_numbers[asset_id]
                       for asset_id, number in image_numbers.items()
                       if asset_id in local_image_numbers and str(number).isdigit()}

    def localize_picture_tags(text: str) -> str:
        def localize(match):
            number = int(next(group for group in match.groups() if group))
            local = global_to_local.get(number, number)
            return f"<Picture {local}>"
        # Also normalize the plain Chinese alias 「图片N」, which users type in the
        # shared prompt; H3 only understands <Picture N>, so an un-normalized
        # alias leaves the reference unbound in the description.
        return re.sub(r"<Picture\s+(\d+)>|@图片\s*(\d+)|(?<![<@])图片\s*(\d+)",
                      localize, text, flags=re.IGNORECASE)

    # H3 numbers the edited source video as <Video 1> and any extra reference
    # videos after it.  A hand-edited prompt can drift to a number that has no
    # material behind it; the model then receives a dangling reference.  Point
    # every out-of-range video tag back at the source video, which is what
    # "the original composition / framing / motion" always refers to.
    bound_video_count = len(shot.get("ref_videos") or []) + (1 if shot.get("source_video") else 0)

    def localize_video_tags(text: str) -> str:
        if not shot.get("source_video"):
            return text
        def localize(match):
            number = int(match.group(1))
            if 1 <= number <= bound_video_count:
                return match.group(0)
            return "<Video 1>"
        return re.sub(r"<Video\s+(\d+)>", localize, text, flags=re.IGNORECASE)

    authored_shared_prompt = localize_video_tags(localize_picture_tags(str(project.get("shared_prompt") or "").strip()))
    authored_shot_prompt = localize_video_tags(localize_picture_tags(str(shot.get("prompt") or "").strip()))
    prompt_parts = [authored_shared_prompt, *annotations, *entity_annotations, authored_shot_prompt]
    shot["prompt"] = "\n".join(part for part in prompt_parts if part)
    return shot


# 工作流版本：只要 build_h3_graph 的行为变了（新增/移除节点、改变 LoRA 挂载、改变
# 步数策略等），就把它加一位，让旧成片缓存自动失效——否则同一份镜头设置会命中旧缓存，
# 把“改完引擎却没重新生成”的旧片段拼回成片。
WORKFLOW_REVISION = "h3studio_graph_rev_2026_10_24_depth_identity_roles"


def shot_refine_mode(shot: dict, project: dict) -> str:
    """Use a shot override while preserving settings of existing projects."""
    if project.get("generation_speed") == "goldfish":
        # This preset already owns its latent enlargement chain in engine.py.
        return "off"
    mode = shot.get("refine_mode")
    return str(project.get("refine_mode") or "off") if mode is None else str(mode)


def shot_upscale(shot: dict, project: dict) -> bool:
    enabled = shot.get("upscale")
    return bool(project.get("upscale")) if enabled is None else bool(enabled)


def _shot_signature(shot: dict, project: dict, assets: dict, previous_signature: str | None = None) -> str:
    """Stable cache key for a shot and the settings that affect its output."""
    relevant_project = {key: project.get(key) for key in (
        "width", "height", "fps", "generation_speed", "h3_lora_dmad", "h3_lora_dmad_strength", "h3_lora_mystic", "h3_lora_mystic_strength", "h3_lora_character_swap", "h3_lora_character_swap_strength", "attention_mode", "sage_attention", "h3_video_vae", "upscale", "upscale_resolution", "upscale_max_resolution",
        "face_refine", "semantic_adapter", "semantic_alpha", "selflift", "selflift_scale",
        "selflift_highres_steps", "selflift_rho", "selflift_tiling", "selflift_tile_count",
        "selflift_tile_overlap", "refine_mode", "refine_steps", "refine_denoise", "refine_passes",
        "refine_width", "refine_height", "refine_transition_step", "refine_lowres_scale",
        "refine_rho", "refine_w_min", "refine_w_max", "refine_tiling", "refine_upscaler",
        "output_width", "output_height", "continuous_take", "continuous_render_mode", "continuity_frames", "identity_first_on_switch", "shared_refs", "shared_prompt", "image_descriptions", "clear_vram_between_shots",
    )}
    relevant_project["workflow_revision"] = WORKFLOW_REVISION
    if project.get("face_refine"):
        relevant_project["face_refine_revision"] = "multiface_ultrasharp_v2"
    if shot.get("continuity"):
        # Invalidate cached generations after changing the scene-adaptive
        # handoff rules, without invalidating unrelated projects.
        relevant_project["continuity_algorithm"] = "h3_native_scene_adaptive_continuity_v22_native_canvas"
    elif shot.get("continuous_take"):
        relevant_project["continuity_algorithm"] = "h3studio_single_subject_prompt_v7"
    ids = {value for key in ("first_image", "last_image", "source_video") if (value := shot.get(key))}
    for key in ("ref_images", "ref_videos", "ref_audios"):
        ids.update(shot.get(key) or [])
    ids.update(guide.get("image_id") for guide in shot.get("h3_guides", [])
               if isinstance(guide, dict) and guide.get("image_id"))
    for values in (project.get("shared_refs") or {}).values():
        ids.update(values or [])
    asset_state = {asset_id: {key: assets.get(asset_id, {}).get(key) for key in ("file", "name", "kind")}
                   for asset_id in sorted(ids)}
    continuity_from = previous_signature if shot.get("continuity") else None
    render_shot = {key: value for key, value in shot.items() if key != "scene_id"}
    payload = json.dumps({"shot": render_shot, "project": relevant_project, "assets": asset_state,
                          "continuity_from": continuity_from},
                         ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _continuity_policy(project: dict, shot: dict, index: int) -> dict:
    """Choose strong or softened handoff for this shot in a generated sequence."""
    if index <= 0:
        return {"enabled": False, "frames": 0, "strength": "none"}
    requested = max(5, min(56, int(project.get("continuity_frames", 22))))
    if not project.get("continuous_take", True):
        return {"enabled": True, "frames": min(5, requested), "strength": "soft"}
    prompt = str(shot.get("prompt") or "")
    wipe_motion = bool(re.search(
        r"occlusion|wipe|pass(?:es|ed)?\s+(?:the\s+)?lens|cross(?:es|ed)?\s+(?:the\s+)?lens|"
        r"遮挡|遮满画面|擦镜头|经过镜头|掠过镜头", prompt, re.IGNORECASE))
    fast_motion = (shot.get("camera_speed") == "快速"
                   or shot.get("action_intensity") in ("明显", "夸张")
                   or shot.get("camera_move") in ("跟随主体", "环绕主体", "手持")
                   or wipe_motion)
    frames = max(requested, 39) if fast_motion else requested
    # Fast movement needs room for H3 to redraw the seam; calm shots keep a
    # tighter latent lock so the subject and composition do not drift.
    seam_redraw = 0.0
    return {"enabled": True, "frames": min(56, frames), "strength": "strong",
            "seam_redraw": seam_redraw}


def _requests_fresh_composition(previous: dict | None, current: dict) -> bool:
    """Use an edited transition when a shot explicitly starts at a new camera setup."""
    if not previous:
        return False
    start = str(current.get("prompt") or "")[:320].lower()
    ending = str(previous.get("prompt") or "")[-320:].lower()
    if re.search(r"接续上一|延续上一|上一[镜段]末帧|继续上一|continue from the previous|"
                 r"continue its camera|same unbroken take", start):
        return False
    close = bool(re.search(r"人物.{0,20}(?:近景|特写)|侧脸特写|面部特写|"
                           r"close.?up|portrait", ending))
    aerial = bool(re.search(r"高空.{0,12}(?:无人机|航拍)|广角航拍|"
                            r"wide.{0,20}aerial|bird.?s.?eye", start))
    if close and aerial:
        return True
    return bool(re.search(r"镜头起幅|the shot begins from|the shot opens|"
                          r"camera starts from|从参考图.{0,35}起|"
                          r"以.{0,25}为画面基准与视角来源", start))


def _continuity_subject_relation(previous_refs, current_refs) -> str:
    previous, current = set(previous_refs or []), set(current_refs or [])
    if previous and current and previous.isdisjoint(current):
        return "new"
    return "same"


def _identity_handoff_changed(previous_refs, current_refs, strict=False) -> bool:
    previous, current = set(previous_refs or []), set(current_refs or [])
    return bool(previous and current and
                (previous.isdisjoint(current) or (strict and previous != current)))


def _shot_identity_refs(shot: dict, project: dict, assets: dict) -> list[str]:
    """Separate person references from shared scenery before a continuity handoff."""
    characters = {item.get("id"): item for item in project.get("characters", [])}
    profile_refs = []
    for character_id in shot.get("character_ids") or []:
        profile = characters.get(character_id) or {}
        profile_refs.extend(filter(None, (profile.get("portrait_asset"), profile.get("multiview_asset"))))
    descriptions = project.get("image_descriptions") or {}
    non_person_words = ("飞机", "双翼机", "战机", "机身", "座舱", "机场", "跑道", "场景", "背景", "地点",
                        "aircraft", "airplane", "biplane", "cockpit", "runway", "airfield",
                        "scene", "background", "landscape")
    def is_non_person(asset_id: str) -> bool:
        metadata = (str(descriptions.get(asset_id) or "") + " " +
                    str(assets.get(asset_id, {}).get("name") or "")).lower()
        if any(word in metadata for word in
               ("四视图", "三视图", "五视图", "六视图", "人物设定", "角色设定",
                "portrait", "face sheet", "character sheet")):
            return False
        return any(word in metadata for word in non_person_words)
    image_numbers = project.get("reference_image_numbers") or {
        asset_id: index for index, (asset_id, asset) in enumerate(
            ((key, value) for key, value in assets.items()
             if value.get("kind") == "image" and value.get("project_id") == project.get("id")), 1)}
    number_to_asset = {int(number): asset_id for asset_id, number in image_numbers.items()
                       if str(number).isdigit() and asset_id in assets}
    def tagged_refs(item: dict) -> list[str]:
        text = "\n".join((str(project.get("shared_prompt") or ""), str(item.get("prompt") or "")))
        return list(dict.fromkeys(number_to_asset[number] for match in
                  re.finditer(r"<Picture\s+(\d+)>|@图片\s*(\d+)", text, re.IGNORECASE)
                  if (number := int(match.group(1) or match.group(2))) in number_to_asset))
    authored_refs = tagged_refs(shot)
    refs = list(dict.fromkeys([*(project.get("shared_refs", {}).get("images", []) or []),
                               *authored_refs, *profile_refs]))
    prompt_text = "\n".join((str(project.get("shared_prompt") or ""), str(shot.get("prompt") or "")))
    person_terms = ("人物", "角色", "主角", "人像", "肖像", "男人", "男子", "女人", "女子",
                    "男孩", "女孩", "飞行员", "士兵", "person", "character", "protagonist",
                    "portrait", "man", "woman", "pilot", "soldier", "identity")
    explicit_person_refs = set()
    for match in re.finditer(r"<Picture\s+(\d+)>|@图片\s*(\d+)", prompt_text, re.IGNORECASE):
        nearby = prompt_text[max(0, match.start() - 20):min(len(prompt_text), match.end() + 20)].lower()
        if not any(term in nearby for term in person_terms):
            continue
        asset_id = number_to_asset.get(int(match.group(1) or match.group(2)))
        if asset_id and not is_non_person(asset_id):
            explicit_person_refs.add(asset_id)
            if asset_id not in refs:
                refs.append(asset_id)
    if explicit_person_refs:
        return [asset_id for asset_id in refs if asset_id in explicit_person_refs]
    identity_words = ("六视图", "五视图", "四视图", "三视图", "角色", "人物", "人像", "肖像", "脸部", "面部",
                      "portrait", "character", "identity", "face sheet", "turnaround", "multiview")
    scene_words = ("场景", "背景", "地点", "跑道", "建筑", "风景", "scene", "background",
                   "location", "runway", "landscape", "environment")
    identities = []
    for asset_id in refs:
        if is_non_person(asset_id):
            continue
        if asset_id in explicit_person_refs:
            identities.append(asset_id)
            continue
        if asset_id in profile_refs:
            identities.append(asset_id)
            continue
        metadata = (str(descriptions.get(asset_id) or "") + " " +
                    str(assets.get(asset_id, {}).get("name") or "")).lower()
        if any(word in metadata for word in identity_words) and not any(word in metadata for word in scene_words):
            identities.append(asset_id)
    if identities:
        return identities
    # Simple fallback for the common workflow: each shot has one private
    # character picture plus shared setting references. Exclude shared and
    # entity-profile images, and require exactly one otherwise-unclassified
    # image so scene/prop collections do not become a guessed identity.
    shared = set(project.get("shared_refs", {}).get("images", []) or [])
    entity_assets = {asset_id for entity in project.get("entities", [])
                     for asset_id in (entity.get("image_assets") or [])}
    usage = {}
    for item in project.get("shots", []):
        for asset_id in set(tagged_refs(item)):
            usage[asset_id] = usage.get(asset_id, 0) + 1
    unique_candidates = []
    for asset_id in authored_refs:
        if asset_id in shared or asset_id in entity_assets or usage.get(asset_id, 0) != 1:
            continue
        metadata = (str(descriptions.get(asset_id) or "") + " " +
                    str(assets.get(asset_id, {}).get("name") or "")).lower()
        if any(word in metadata for word in scene_words):
            continue
        unique_candidates.append(asset_id)
    if len(unique_candidates) == 1:
        return unique_candidates
    return []


def _handoff_window(relation: str, context_frames: int) -> tuple[int, int]:
    """Keep a tiny visual bridge at identity changes and a wider bridge otherwise."""
    frames = max(0, int(context_frames))
    if relation == "new":
        frames = min(frames, NEW_IDENTITY_CONTEXT_FRAMES)
        return (frames, min(NEW_IDENTITY_VISIBLE_FRAMES, frames)) if frames >= 5 else (0, 0)
    return frames, min(15, max(0, round(frames * 0.4)))


def _continuity_anchor_frames(context_frames: int) -> int:
    """Match the native H3 guide lengths while anchoring the actual final frames.

    MiniMaxH3AddGuide crops an invalid clip length to its first 17k+5 frames.
    Supplying the final valid-length tail avoids dropping the source's ending.
    """
    frames = max(0, int(context_frames))
    return 0 if frames < 5 else frames - (frames - 5) % 17


def _strengthen_incoming_identity_reference(shot: dict, incoming_refs: list[str]) -> None:
    """Make an incoming identity reference authoritative in the H3 prompt."""
    refs = list(dict.fromkeys(incoming_refs or []))
    if not refs:
        return
    shot["ref_image_roles"] = dict(shot.get("ref_image_roles") or {})
    for asset_id in refs:
        shot["ref_image_roles"][asset_id] = "fully_preserved"
    primary = refs[0]
    if primary in shot.get("ref_images", []):
        shot["continuity_identity_picture_tag"] = f"<Picture {shot['ref_images'].index(primary) + 1}>"


def _copy_asset(asset_id: str, assets: dict, *, trim_start: float | None = None,
                trim_duration: float | None = None,
                fit_width: int | None = None, fit_height: int | None = None) -> str:
    """Copy an asset into ComfyUI's input folder and return its input filename.

    Videos can be trimmed and fitted to the generation canvas.  H3 carries every
    reference token through every sampling step and attention cost grows with
    the square of the token count, so a 720x1280 source video feeding a
    480x864 canvas costs more attention than the frame being generated.  The
    reference is therefore downscaled to the canvas (never upscaled) exactly the
    way ComfyUI's own "match" sizing handles reference images.
    """
    asset = assets.get(asset_id)
    if not asset:
        raise ValueError(f"素材不存在：{asset_id}")
    source = ASSETS / asset["file"]
    if not source.is_file():
        raise FileNotFoundError(source)
    COMFY_INPUT.mkdir(parents=True, exist_ok=True)
    is_video = asset["kind"] == "video"
    wants_trim = is_video and trim_start is not None
    fitted = None
    if is_video and fit_width and fit_height and fit_width > 0 and fit_height > 0:
        source_width, source_height = video_dimensions(source)
        if source_width and source_height and (source_width > fit_width or source_height > fit_height):
            ratio = min(fit_width / source_width, fit_height / source_height)
            # H3 works on a 16px canvas grid, so round the scaled size down to it.
            fitted = (max(16, int(source_width * ratio) // 16 * 16),
                      max(16, int(source_height * ratio) // 16 * 16))
    if wants_trim or fitted:
        parts = ["h3studio_" + asset_id]
        if wants_trim:
            parts += [str(int(trim_start * 1000)), str(int((trim_duration or 0) * 1000))]
        if fitted:
            parts.append(f"fit{fitted[0]}x{fitted[1]}")
        name = "_".join(parts) + ".mp4"
        target = COMFY_INPUT / name
        if not target.exists():
            command = ["ffmpeg", "-y", "-loglevel", "error"]
            if wants_trim:
                command += ["-ss", str(trim_start), "-i", str(source)]
                if trim_duration is not None:
                    command += ["-t", str(trim_duration)]
            else:
                command += ["-i", str(source)]
            filters = []
            if fitted:
                filters.append(f"scale={fitted[0]}:{fitted[1]}")
            filters.append("fps=24")
            command += ["-vf", ",".join(filters), "-c:v", "libx264", "-preset", "fast",
                        "-pix_fmt", "yuv420p", "-c:a", "aac", str(target)]
            subprocess.run(command, check=True, timeout=300)
        return name
    name = "h3studio_" + asset_id + source.suffix.lower()
    target = COMFY_INPUT / name
    if not target.exists() or target.stat().st_size != source.stat().st_size:
        shutil.copy2(source, target)
    return name


def _canvas_of(settings: dict) -> tuple[int, int]:
    """The canvas H3 will actually generate (already adjusted by the memory plan)."""
    try:
        width, height = int(settings.get("width") or 0), int(settings.get("height") or 0)
    except (TypeError, ValueError):
        return 0, 0
    return (width if width > 0 else 0, height if height > 0 else 0)


def _upscale_still_enabled(original: dict, project: dict, project_id: str) -> bool:
    """Re-read the live SeedVR2 switch while a run waits for upscale confirmation.

    The confirmation panel can be answered by turning the upscale off instead of
    pressing a button, so the waiting loop must not block forever once the
    switch is off.  The live project wins over the snapshot taken when the run
    started.
    """
    live_project = read()["projects"].get(project_id) or project
    live_shot = next((item for item in live_project.get("shots", [])
                      if item.get("id") == original.get("id")), original)
    return bool(shot_upscale(live_shot, live_project))


def _make_media_map(shot: dict, project: dict, assets: dict) -> dict[str, str]:
    ids = set()
    for key in ("first_image", "last_image", "source_video"):
        if shot.get(key):
            ids.add(shot[key])
    for key in ("ref_images", "ref_videos", "ref_audios"):
        ids.update(shot.get(key) or [])
    ids.update(guide.get("image_id") for guide in shot.get("h3_guides", [])
               if isinstance(guide, dict) and guide.get("image_id"))
    for values in project.get("shared_refs", {}).values():
        ids.update(values or [])
    canvas_width, canvas_height = _canvas_of(project)
    media = {asset_id: _copy_asset(asset_id, assets, fit_width=canvas_width,
                                   fit_height=canvas_height) for asset_id in ids}
    if shot.get("source_video"):
        media[shot["source_video"]] = _copy_asset(
            shot["source_video"], assets,
            trim_start=float(shot.get("source_start") or 0),
            trim_duration=float(shot.get("duration") or 5),
            fit_width=canvas_width, fit_height=canvas_height,
        )
    return media


def _submit(graph: dict, run_id: str | None = None) -> str:
    """Release cached models only when the next graph changes model family."""
    global _ACTIVE_MODEL_FAMILY, _ACTIVE_MODEL_RUN_ID
    family = _graph_model_family(graph)
    with _MODEL_SWITCH_LOCK:
        previous = _ACTIVE_MODEL_FAMILY
        new_run = run_id is not None and run_id != _ACTIVE_MODEL_RUN_ID
        if new_run:
            update_run(run_id, current="任务准备 · 释放翻译和提示词模型")
            helpers = _release_idle_local_models()
            update_run(run_id, memory_preparation={"time": time.time(), **helpers})
        if previous is not None and previous != family:
            from_label = _model_family_label(previous)
            to_label = _model_family_label(family)
            if run_id:
                update_run(run_id,
                           current=f"正在等待 ComfyUI 空闲并切换：{from_label} → {to_label}",
                           memory_switch={"from": previous or "unknown", "to": family,
                                          "status": "waiting", "time": time.time()})
            failures = 0
            while True:
                if run_id and read()["runs"].get(run_id, {}).get("cancel_requested"):
                    raise RuntimeError("用户取消任务")
                idle, detail = _comfy_queue_is_idle()
                if idle:
                    break
                failures = failures + 1 if detail.startswith("无法确认") else 0
                if failures >= 10:
                    raise RuntimeError("模型切换前无法确认 ComfyUI 队列状态：" + detail)
                time.sleep(1)
            try:
                _release_comfy_memory()
            except Exception as error:
                if run_id:
                    update_run(run_id, memory_switch={"from": previous or "unknown", "to": family,
                               "status": "failed", "time": time.time(), "detail": str(error)})
                raise RuntimeError("模型切换前释放 ComfyUI 内存失败：" + str(error)) from error
            if run_id:
                update_run(run_id, current=f"模型资源已释放，正在加载{to_label}",
                           memory_switch={"from": previous or "unknown", "to": family,
                                          "status": "released", "time": time.time(),
                                          "detail": "已卸载旧模型并释放显存缓存"})
        elif new_run and run_id:
            label = _model_family_label(family)
            known_cache = previous == family
            update_run(run_id, current=f"保留现有模型缓存，正在提交{label}任务",
                       memory_switch={"from": previous or "unknown", "to": family,
                                      "status": "reused" if known_cache else "preserved",
                                      "time": time.time(),
                                      "detail": ("同系列模型已加载，保留缓存" if known_cache else
                                                 "导演台刚启动，保留 ComfyUI 缓存并由其调度模型")})
        # ComfyUI can be slow to accept a prompt while the worker is decoding the
        # previous clip or loading a model. A single 30 s timeout used to abort
        # the whole run; retry instead and surface progress to the workbench.
        response = None
        for attempt in range(1, 7):
            try:
                client_id = f"h3_studio_{run_id}" if run_id else "h3_studio"
                if run_id:
                    update_run(run_id, preview_client_id=client_id)
                response = api("/prompt", {"prompt": graph, "client_id": client_id,
                                           "extra_data": {"preview_method": "auto"}})
                break
            except (TimeoutError, urllib.error.URLError, ConnectionError) as error:
                if read()["runs"].get(run_id or "", {}).get("cancel_requested"):
                    raise RuntimeError("用户取消任务") from error
                if attempt == 6:
                    raise RuntimeError(f"ComfyUI 连续 6 次未能接受任务：{error}") from error
                if run_id:
                    update_run(run_id, current=f"等待 ComfyUI 接受任务（{attempt}/6）：{error}")
                time.sleep(min(5 * attempt, 20))
        if response.get("node_errors"):
            raise RuntimeError(json.dumps(response["node_errors"], ensure_ascii=False))
        _ACTIVE_MODEL_FAMILY = family
        _ACTIVE_MODEL_RUN_ID = run_id
        return response["prompt_id"]


def _prompt_state(prompt_id: str) -> str:
    """Return active/completed/missing without changing the ComfyUI queue."""
    if not prompt_id:
        return "missing"
    try:
        history = api("/history/" + urllib.parse.quote(prompt_id, safe=""))
        if history.get(prompt_id):
            return "completed"
        queue = api("/queue")
        for key in ("queue_running", "queue_pending"):
            for item in queue.get(key, []):
                if isinstance(item, (list, tuple)) and len(item) > 1 and item[1] == prompt_id:
                    return "active"
        return "missing"
    except Exception:
        return "unavailable"


def _resume_prompt(run_id: str, stage: str, shot_id: str) -> str | None:
    run = read()["runs"][run_id]
    if run.get("active_stage") != stage or run.get("active_shot_id") != shot_id:
        return None
    prompt_id = str(run.get("prompt_id") or "")
    return prompt_id if _prompt_state(prompt_id) in ("active", "completed") else None


def _release_idle_local_models() -> dict:
    """Wait for translation to exit and for the local prompt server to sleep."""
    from prompt_tools import _LOCAL_TRANSLATION_LOCK
    with _LOCAL_TRANSLATION_LOCK:
        report = {"translation": "released", "prompt_model": "unavailable"}
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def local_json(path):
        with opener.open("http://127.0.0.1:11434" + path, timeout=2) as response:
            return json.load(response)

    try:
        props = local_json("/props")
        if props.get("is_sleeping") is True:
            report["prompt_model"] = "sleeping"
            return report
        if props.get("is_sleeping") is not False:
            report["prompt_model"] = "unmanaged"
            return report
        if any(slot.get("is_processing") for slot in local_json("/slots")):
            report["prompt_model"] = "busy"
            return report
        # llama-server unloads after one idle second. /props never wakes it.
        for _ in range(12):
            time.sleep(0.25)
            if local_json("/props").get("is_sleeping"):
                report["prompt_model"] = "sleeping"
                return report
        report["prompt_model"] = "pending_sleep"
    except (OSError, ValueError):
        pass
    return report


def _release_comfy_memory() -> None:
    """Ask ComfyUI to unload cached models and release unused accelerator memory."""
    payload = json.dumps({"unload_models": True, "free_memory": True}).encode("utf-8")
    request = urllib.request.Request(COMFY_URL + "/free", data=payload,
                                    headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=30):
        pass


def _comfy_queue_is_idle() -> tuple[bool, str]:
    """Only free model memory when no ComfyUI prompt can be affected."""
    try:
        queue = api("/queue")
        running = len(queue.get("queue_running", []))
        pending = len(queue.get("queue_pending", []))
        if running or pending:
            return False, f"ComfyUI 队列仍有 {running} 个运行中、{pending} 个等待任务"
        return True, "ComfyUI 队列空闲"
    except Exception as error:
        return False, f"无法确认 ComfyUI 队列状态，跳过显存清理：{error}"


def _vram_is_under_pressure() -> tuple[bool, str]:
    """Return whether ComfyUI's first CUDA device has less than 15% or 2 GiB free."""
    try:
        devices = api("/system_stats").get("devices", [])
        device = next((item for item in devices if item.get("type") == "cuda"), None)
        if not device:
            return False, "未检测到 CUDA 设备"
        total, free = int(device.get("vram_total", 0)), int(device.get("vram_free", 0))
        threshold = max(2 * 1024**3, int(total * 0.15))
        return free < threshold, f"显存空闲 {free / 1024**3:.1f}/{total / 1024**3:.1f} GiB"
    except Exception as error:
        return False, f"读取显存状态失败：{error}"


def _auto_memory_plan(width: int, height: int) -> dict:
    """Choose a conservative canvas from ComfyUI device stats; never changes queue/GPU state."""
    try:
        queue = api("/queue")
        busy = bool(queue.get("queue_running") or queue.get("queue_pending"))
    except Exception:
        # Unknown queue state means free-memory readings may be transient; ignore them.
        busy = True
    try:
        devices = api("/system_stats").get("devices", [])
        device = next((item for item in devices if item.get("type") == "cuda"), None)
        if not device:
            return plan_canvas(width, height, queue_busy=busy)
        plan = plan_canvas(width, height, int(device.get("vram_total", 0)),
                           int(device.get("vram_free", 0)), queue_busy=busy)
        return plan
    except Exception as error:
        plan = plan_canvas(width, height, queue_busy=busy)
        plan["reason"] += f"；显存读取失败：{error}"
        return plan


def _wait(prompt_id: str, run_id: str) -> dict:
    connection_failures = 0
    started = time.time()
    while True:
        if read()["runs"][run_id].get("cancel_requested"):
            try:
                api("/api/jobs/" + urllib.parse.quote(prompt_id, safe="") + "/cancel", {})
            except Exception:
                try:
                    api("/queue", {"delete": [prompt_id]})
                except Exception:
                    pass
            raise RuntimeError("用户取消任务")
        try:
            history = api("/history/" + prompt_id).get(prompt_id)
            connection_failures = 0
        except (TimeoutError, urllib.error.URLError, ConnectionError) as error:
            connection_failures += 1
            if connection_failures >= 10:
                raise RuntimeError(f"连续无法读取 ComfyUI 任务状态：{error}") from error
            update_run(run_id, current=(f"ComfyUI 状态暂时不可读，正在重试"
                                        f"（{connection_failures}/10）· 已等待 {int(time.time() - started)}s"))
            time.sleep(min(3 * connection_failures, 15))
            continue
        if history:
            status = history.get("status", {})
            if status.get("status_str") == "error":
                update_run(run_id, active_stage="", active_shot_id="", active_prefix="")
                raise RuntimeError(json.dumps(status.get("messages", []), ensure_ascii=False)[-3000:])
            return history
        time.sleep(3)


def _download_video(history: dict, destination: Path, expected_prefix: str | None = None) -> None:
    for node in history.get("outputs", {}).values():
        for kind in ("videos", "gifs", "images"):
            for item in node.get(kind, []):
                filename = str(item.get("filename", ""))
                # ComfyUI's SaveVideo node exposes MP4s through `images`; only
                # accept video files from that collection to avoid downloading
                # preview PNGs as clips.
                if kind == "images" and Path(filename).suffix.lower() not in (".mp4", ".webm", ".mov", ".mkv"):
                    continue
                if expected_prefix and not re.match(r"^" + re.escape(expected_prefix) + r"_\d{5}_\.", Path(filename).name):
                    continue
                query = urllib.parse.urlencode({
                    "filename": item["filename"], "subfolder": item.get("subfolder", ""),
                    "type": item.get("type", "output"),
                })
                with urllib.request.urlopen(COMFY_URL + "/view?" + query, timeout=120) as source:
                    with destination.open("wb") as output:
                        shutil.copyfileobj(source, output)
                return
    raise RuntimeError("ComfyUI 任务完成，但没有返回视频文件")


def _video_dimensions(path: Path) -> tuple[int, int]:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "json", str(path)],
        capture_output=True, text=True, check=True, timeout=30)
    streams = json.loads(result.stdout).get("streams") or []
    if not streams:
        raise RuntimeError(f"无法读取视频画布：{path.name}")
    return int(streams[0]["width"]), int(streams[0]["height"])


def _seedvr2_enlarges(width: int, height: int, resolution: int, max_resolution: int) -> bool:
    """Both SeedVR2 canvas limits must permit a real enlargement."""
    return min(int(resolution) / min(width, height),
               int(max_resolution) / max(width, height)) > 1.01


@_serialized_director_run
def run_post_upscale(run_id: str) -> None:
    """Apply SeedVR2 to an existing completed video without running H3 again."""
    run = read()["runs"][run_id]
    source = OUTPUTS / run["source_file"]
    output_dir = OUTPUTS / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        update_run(run_id, status="running", started_at=time.time(),
                   current="读取已有成片 · 不重新生成 H3", progress="0/1")
        if not source.is_file():
            raise ValueError("原成片文件已不存在")
        original_size = _video_dimensions(source)
        resolution = int(run["upscale_resolution"])
        max_resolution = int(run["upscale_max_resolution"])
        if not _seedvr2_enlarges(*original_size, resolution, max_resolution):
            raise ValueError("当前超分目标尺寸没有高于原片；请在高级设置中提高短边和最长边")
        input_name = f"h3studio_{run_id}_existing.mp4"
        shutil.copy2(source, COMFY_INPUT / input_name)
        prefix = f"h3_studio_{run_id}_seedvr2"
        graph = build_seedvr2_graph(input_name, prefix, resolution=resolution,
                                     max_resolution=max_resolution, seed=int(run.get("seed", 42)))
        prompt_id = _resume_prompt(run_id, "seedvr2_existing", run["source_run_id"])
        if not prompt_id:
            prompt_id = _submit(graph, run_id)
            update_run(run_id, prompt_id=prompt_id, active_stage="seedvr2_existing",
                       active_shot_id=run["source_run_id"], active_prefix=prefix,
                       current="对已有成片执行 SeedVR2 超分")
        output = output_dir / "post_upscaled.mp4"
        _download_video(_wait(prompt_id, run_id), output, prefix)
        upscaled_size = _video_dimensions(output)
        if min(upscaled_size) <= min(original_size):
            raise RuntimeError("SeedVR2 输出尺寸未超过原片，已保留原成片")
        # SeedVR2 processes pictures. Keep the original soundtrack on the new video.
        with_audio = output_dir / "post_upscaled_audio.mp4"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(output),
                        "-i", str(source), "-map", "0:v:0", "-map", "1:a:0?",
                        "-c:v", "copy", "-c:a", "aac", "-shortest", str(with_audio)],
                       check=True, timeout=300)
        with_audio.replace(output)
        relative = str(output.relative_to(OUTPUTS))
        result = {"file": relative, "source_file": run["source_file"]}
        if run.get("source_shot_id"):
            result["shot_id"] = run["source_shot_id"]
        update_run(run_id, status="completed", finished_at=time.time(),
                   current="已有成片超分完成", progress="1/1", merged=relative,
                   preview_file=relative, results=[result], active_stage="",
                   active_shot_id="", active_prefix="", upscale_result={
                       "source_width": original_size[0], "source_height": original_size[1],
                       "output_width": upscaled_size[0], "output_height": upscaled_size[1],
                       "file": relative})
    except Exception as error:
        status = "cancelled" if "用户取消任务" in str(error) else "error"
        update_run(run_id, status=status, error=str(error), finished_at=time.time(),
                   current="已有成片超分失败", active_stage="", active_shot_id="")


def _continuity_source(item: dict) -> Path:
    """Use the uncropped canvas so delivery fitting cannot zoom the next shot."""
    for key in ("continuity_file", "h3_file", "file"):
        if item.get(key) and (source := OUTPUTS / item[key]).is_file():
            return source
    raise FileNotFoundError("接续镜头的视频文件不存在")


def _tail_clip(video: Path, target: Path, frames: int) -> None:
    seconds = min(frames / 24, video_duration(video))
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-sseof", str(-seconds), "-i", str(video),
                    "-vf", "fps=24,tpad=stop_mode=clone:stop_duration=2", "-frames:v", str(frames),
                    "-c:v", "libx264", "-preset", "fast", "-pix_fmt", "yuv420p", "-an",
                    str(target)], check=True, timeout=120)


def _grid_duration(duration: float, context_frames: int) -> float:
    """Snap a continued shot onto H3's frame grid.

    H3 needs a length of 17k+5 frames for ``duration + context``. Snapping costs
    at most 8 frames (0.33 s) and avoids requesting padding frames that the
    pipeline would then trim off the clip's tail, taking narration with them.
    ``context_frames`` must be the value used for the requested length (the
    handoff window), not the extracted guide-clip length; mixing the two leaves
    a full 17-frame pad behind.
    """
    frames = max(17, round(float(duration) * 24))
    guide = max(0, int(context_frames))
    remainder = (frames + guide - 5) % 17
    if remainder:
        down = frames - remainder
        up = down + 17
        frames = down if (down >= 17 and remainder <= 17 - remainder) else up
    return frames / 24


def _trim_plan(video: Path, shot: dict) -> dict:
    """Decide how much of a generated continued clip may be dropped.

    Dropping the duplicated guide head is only safe when it carries no speech:
    H3 receives the guide as video only, so it can start the narration at frame
    zero even when the prompt reserves the opening window. When that happens the
    head is kept, which costs a few duplicated frames instead of cutting words.
    """
    context = max(0, int(shot.get("continuity_frames") or 0))
    visible = max(0, int(shot.get("continuity_visible_frames") or 0))
    effective = float(shot.get("_h3_effective_duration") or shot.get("duration", 5))
    total = max(1, round(video_duration(video) * 24))
    head = min(max(0, context - visible), max(0, total - 1))
    start = head
    output = min(max(1, round(effective * 24) + visible), total)
    plan = {"total_frames": total, "start_frame": start, "output_frames": output,
            "head_dropped_frames": head, "head_speech_preserved": False,
            "head_peak_dbfs": None}
    if head and _has_spoken_audio(shot):
        has_audio, peak = _overlap_audio_peak(video, head / 24)
        plan["head_peak_dbfs"] = peak
        if has_audio and peak is not None and peak > SPOKEN_OVERLAP_MAX_DBFS:
            plan.update(start_frame=0, output_frames=min(total, head + output),
                        head_speech_preserved=True)
    return plan


def continuity_preview(shot: dict, project: dict, assets: dict) -> dict:
    """Continuity fields the run loop would apply, computed without media.

    Prompt preview and real generation must compile the same text; without this
    the preview silently omits the continuation prelude and the handoff wording
    that usually decide whether a seam reads as one take.
    """
    shots = project.get("shots", [])
    index = next((i for i, item in enumerate(shots) if item.get("id") == shot.get("id")), -1)
    previous = shots[index - 1] if index > 0 else None
    continuity = _continuity_policy(project, shot, max(index, 0))
    if previous and _requests_fresh_composition(previous, shot):
        continuity = {"enabled": False, "frames": 0, "strength": "none"}
    relation = "none"
    if previous:
        relation = _continuity_subject_relation(_shot_identity_refs(previous, project, assets),
                                                _shot_identity_refs(shot, project, assets))
    frames, visible = _handoff_window(relation if continuity.get("enabled") else "none",
                                      continuity.get("frames", 0))
    if frames and _has_spoken_audio(shot):
        frames = min(frames, NARRATED_HANDOFF_FRAMES)
        visible = min(visible, NARRATED_HANDOFF_VISIBLE, frames)
    prepared = dict(shot)
    prepared["continuous_take"] = bool(project.get("continuous_take", True)) and not (
        previous and _requests_fresh_composition(previous, shot))
    prepared["continuity"] = bool(frames)
    prepared["continuity_frames"] = frames
    prepared["continuity_visible_frames"] = visible
    prepared["continuity_strength"] = continuity.get("strength", "none")
    prepared["continuity_subject_relation"] = relation
    if frames:
        prepared["continuity_video"] = "__preview__"
        prepared["continuity_guide_frames"] = _continuity_anchor_frames(frames)
        prepared["_h3_effective_duration"] = _grid_duration(
            float(shot.get("duration", 5)), frames)
    return prepared


def _trim_prefix(video: Path, seconds: float, duration: float, visible_context_frames: int = 0,
                 *, start_frame: int | None = None, output_frames: int | None = None) -> None:
    target = video.with_name(video.stem + "_trim.mp4")
    # Keep a short part of the guided prefix as the visible wipe. The merge
    # path removes the same number of frames from the previous shot's tail.
    # ``start_frame``/``output_frames`` come from _trim_plan, which may keep the
    # head when it already contains speech.
    start = (max(0, round(float(seconds) * 24) - max(0, int(visible_context_frames)))
             if start_frame is None else max(0, int(start_frame)))
    frames = (max(1, round(float(duration) * 24) + max(0, int(visible_context_frames)))
              if output_frames is None else max(1, int(output_frames)))
    audio_start = start / 24
    audio_duration = frames / 24
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(video),
                    "-vf", f"fps=24,trim=start_frame={start}:end_frame={start + frames},setpts=PTS-STARTPTS",
                    "-af", f"atrim=start={audio_start:.6f}:duration={audio_duration:.6f},asetpts=PTS-STARTPTS",
                    "-frames:v", str(frames), "-c:v", "libx264", "-preset", "fast",
                    "-pix_fmt", "yuv420p", "-c:a", "aac", "-ar", "48000", str(target)],
                   check=True, timeout=300)
    target.replace(video)


def _trim_suffix(video: Path, frames: int, target: Path) -> None:
    total_frames = max(1, round(video_duration(video) * 24))
    keep_frames = max(1, total_frames - max(0, int(frames)))
    seconds = keep_frames / 24
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(video),
                    "-vf", f"fps=24,trim=end_frame={keep_frames},setpts=PTS-STARTPTS",
                    "-af", f"atrim=duration={seconds:.6f},asetpts=PTS-STARTPTS",
                    "-frames:v", str(keep_frames), "-c:v", "libx264", "-preset", "fast",
                    "-pix_fmt", "yuv420p", "-c:a", "aac", "-ar", "48000", str(target)],
                   check=True, timeout=300)


def _trim_prefix_frames(video: Path, frames: int, target: Path) -> None:
    """Drop only the redundant guided frames, keeping sound in sync."""
    skip_frames = max(0, int(frames))
    seconds = skip_frames / 24
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(video),
                    "-vf", f"fps=24,trim=start_frame={skip_frames},setpts=PTS-STARTPTS",
                    "-af", f"atrim=start={seconds:.6f},asetpts=PTS-STARTPTS",
                    "-c:v", "libx264", "-preset", "fast", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-ar", "48000", str(target)],
                   check=True, timeout=300)


def _narrated_head_skip(video: Path, previous: Path | None, context_frames: int) -> int:
    """Pick the guide head to drop for a narrated shot: silent and best aligned.

    H3 starts the voiceover at frame zero because the guide carries no audio, so
    the head cannot be dropped on length alone. Only frames whose window stays
    below the speech threshold are candidates; among those, the one whose first
    frame best matches the outgoing shot's last frame gives the calmest cut.
    """
    limit = max(0, int(context_frames))
    if not limit or previous is None:
        return 0
    try:
        before = _handoff_gray_frames(previous)
        after = _handoff_gray_frames(video)
    except (OSError, ValueError, subprocess.SubprocessError):
        return 0
    best_score, best_skip = float("inf"), 0
    for skip in range(0, min(limit, len(after) - 1) + 1):
        if skip:
            has_audio, peak = _overlap_audio_peak(video, skip / 24)
            if has_audio and peak is not None and peak > SPOKEN_OVERLAP_MAX_DBFS:
                break
        score = _gray_distance(before[-1], after[skip])
        if score < best_score:
            best_score, best_skip = score, skip
    return best_skip


ENTRY_FLASH_MIN_LUMA = 8.0
ENTRY_FLASH_DECAY_SECONDS = 0.22


def _frame_luma(video: Path, start: float = 0.0, count: int = 18) -> list[float]:
    """Per-frame average luma (YAVG) for a short window."""
    result = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", f"{max(0.0, start):.4f}", "-i", str(video),
         "-frames:v", str(max(1, int(count))),
         "-vf", "signalstats,metadata=print:key=lavfi.signalstats.YAVG:file=-",
         "-f", "null", "-"], capture_output=True, text=True, timeout=120)
    return [float(line.split("YAVG=")[1]) for line in result.stdout.splitlines()
            if "YAVG=" in line]


def _suppress_entry_flash(video: Path, target: Path | None = None,
                          head_frames: int = 6) -> bool:
    """Damp a bright transient in the opening frames of a generated clip.

    H3 regularly emits a washed-out bright frame a few frames into a new clip.
    At a seam that reads as a white flash. The repair is a short asymmetric luma
    ramp (rise to the measured peak time, linear decay) that leaves audio and
    frame count untouched; it only fires when the peak is clearly above both the
    first frame and the settled level, so a deliberate bright opening is kept.
    """
    try:
        probe = _frame_luma(video, 0.0, head_frames + 12)
    except (OSError, ValueError, subprocess.SubprocessError):
        return False
    if len(probe) < head_frames + 6:
        return False
    settled = probe[head_frames + 2:]
    stable = sorted(settled)[len(settled) // 2]
    head = probe[:head_frames]
    peak = max(head)
    peak_index = head.index(peak)
    if peak_index == 0 or peak < max(head[0], stable) + ENTRY_FLASH_MIN_LUMA:
        return False
    delta = peak - stable
    peak_seconds = peak_index / 24
    rise = max(peak_seconds, 1 / 24)
    decay = max(ENTRY_FLASH_DECAY_SECONDS, 2 / 24)
    expression = (f"clip(lum(X,Y)-{delta:.2f}*if(lt(T,{peak_seconds:.6f}),T/{rise:.6f},"
                  f"max(0,1-(T-{peak_seconds:.6f})/{decay:.6f})),0,255)")
    output = target or video.with_name(video.stem + "_flash.mp4")
    try:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(video),
                        "-vf", f"geq=lum='{expression}':cb='cb(X,Y)':cr='cr(X,Y)'",
                        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                        "-pix_fmt", "yuv420p", "-c:a", "copy", str(output)],
                       check=True, timeout=900)
        fixed = _frame_luma(output, 0.0, head_frames)
    except (OSError, ValueError, subprocess.SubprocessError):
        output.unlink(missing_ok=True)
        return False
    if len(fixed) < head_frames or max(fixed) >= peak - 2:
        output.unlink(missing_ok=True)
        return False
    if target is None:
        output.replace(video)
    return True


def _trim_downloaded_shot(video: Path, shot: dict, previous: Path | None = None) -> dict:
    """Make a downloaded continued clip usable without losing narration.

    A narrated shot may only drop a head that is silent and best aligned with the
    outgoing tail; everything after it keeps its original samples and timing.
    A shot without speech simply drops the guide head.
    """
    if _has_spoken_audio(shot):
        context = max(0, int(shot.get("continuity_frames") or 0))
        total = max(1, round(video_duration(video) * 24))
        skipped = min(_narrated_head_skip(video, previous, context), max(0, total - 1))
        if skipped:
            target = video.with_name(video.stem + "_head.mp4")
            _trim_prefix_frames(video, skipped, target)
            target.replace(video)
        return {"mode": "silent_head", "total_frames": total,
                "start_frame": skipped, "output_frames": max(1, total - skipped),
                "head_dropped_frames": skipped, "head_speech_preserved": skipped < context,
                "head_peak_dbfs": None}
    plan = _trim_plan(video, shot)
    if plan["start_frame"] or plan["output_frames"] < plan["total_frames"]:
        _trim_prefix(video, 0, 0, start_frame=plan["start_frame"],
                     output_frames=plan["output_frames"])
    return {**plan, "mode": "trimmed"}


def _retime_video_prefix_keep_audio(video: Path, skip_frames: int, target: Path) -> None:
    """Remove repeated guide images while retaining every sample of narration.

    The generated clip's first frames replay the preceding shot. Stretch the
    remaining picture by a few percent to its original duration; narration is
    copied without trimming, shifting, or changing its pitch.
    """
    total_frames = max(1, round(video_duration(video) * 24))
    skip = max(0, min(int(skip_frames), total_frames - 2))
    if not skip:
        shutil.copy2(video, target)
        return
    scale = (total_frames - 1) / (total_frames - skip - 1)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(video),
                    "-map", "0:v:0", "-map", "0:a:0?",
                    "-vf", f"fps=24,trim=start_frame={skip},setpts=(PTS-STARTPTS)*{scale:.9f},fps=24",
                    "-frames:v", str(total_frames), "-c:v", "libx264", "-preset", "fast",
                    "-pix_fmt", "yuv420p", "-c:a", "copy", str(target)],
                   check=True, timeout=300)


def _dissolve_entry_keep_audio(previous: Path, current: Path, target: Path,
                               frames: int = 12) -> None:
    """Fade from the outgoing last image into a fresh camera setup.

    The transition occupies the opening frames of the incoming clip, so no
    dialogue samples are cut, overlapped, delayed, or sped up.
    """
    seconds = max(2, int(frames)) / 24
    still = target.with_suffix(".png")
    width, height = _video_dimensions(current)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-sseof", "-0.08",
                    "-i", str(previous), "-frames:v", "1", "-vf", f"scale={width}:{height}",
                    str(still)], check=True, timeout=60)
    try:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-loop", "1",
                        "-framerate", "24", "-t", f"{seconds:.6f}", "-i", str(still),
                        "-i", str(current),
                        "-filter_complex", "[0:v]format=yuv420p,settb=AVTB,setpts=PTS-STARTPTS[a];"
                        "[1:v]fps=24,format=yuv420p,settb=AVTB,setpts=PTS-STARTPTS[b];"
                        f"[a][b]xfade=transition=fade:duration={seconds:.6f}:offset=0[v]",
                        "-map", "[v]", "-map", "1:a:0?", "-c:v", "libx264", "-preset", "fast",
                        "-pix_fmt", "yuv420p", "-c:a", "copy", str(target)],
                       check=True, timeout=300)
    finally:
        still.unlink(missing_ok=True)


_GRAY_FRAME_CACHE: dict[tuple, list[bytes]] = {}
_GRAY_FRAME_CACHE_LOCK = threading.Lock()
_GRAY_FRAME_CACHE_LIMIT = 8


def _decode_handoff_gray_frames(video: Path) -> list[bytes]:
    result = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(video), "-an", "-vf", "fps=24,scale=64:64",
         "-pix_fmt", "gray", "-f", "rawvideo", "pipe:1"],
        capture_output=True, check=True, timeout=120)
    pixels = 64 * 64
    if not result.stdout or len(result.stdout) % pixels:
        raise ValueError("无法读取衔接画面")
    return [result.stdout[i:i + pixels] for i in range(0, len(result.stdout), pixels)]


def _handoff_gray_frames(video: Path) -> list[bytes]:
    """Decode small luma frames for an inexpensive seam alignment check.

    Alignment, exposure matching and motion bridging each decode the same pair
    of clips. Decoding used to run 6–12 times per seam; the cache keeps one copy
    per file revision so the seam cost stays flat.
    """
    try:
        stat = video.stat()
    except OSError:
        return _decode_handoff_gray_frames(video)
    key = (str(video), stat.st_mtime_ns, stat.st_size)
    with _GRAY_FRAME_CACHE_LOCK:
        cached = _GRAY_FRAME_CACHE.get(key)
    if cached is not None:
        return cached
    frames = _decode_handoff_gray_frames(video)
    with _GRAY_FRAME_CACHE_LOCK:
        if len(_GRAY_FRAME_CACHE) >= _GRAY_FRAME_CACHE_LIMIT:
            _GRAY_FRAME_CACHE.pop(next(iter(_GRAY_FRAME_CACHE)))
        _GRAY_FRAME_CACHE[key] = frames
    return frames


def _views_are_stable(previous: Path, current: Path, *, blend: bool = False) -> bool:
    """Same confidence gate used by exposure/motion repair, but never raising.

    ``blend`` relaxes the correlation floor for the short seam dissolve: fading
    two nearly matching frames is far safer than the optical-flow bridge, and a
    slightly looser gate avoids leaving a visible jump on otherwise similar cuts.
    """
    try:
        before = _handoff_gray_frames(previous)
        after = _handoff_gray_frames(current)
    except (OSError, ValueError, subprocess.SubprocessError):
        return False
    if not blend:
        return _stable_matching_view(before, after)
    if len(before) < 2 or len(after) < 2:
        return False
    last, first = before[-1], after[0]
    mean = sum(last) / len(last)
    contrast = (sum((value - mean) ** 2 for value in last) / len(last)) ** 0.5
    return (contrast >= 6
            and _gray_distance(last, first) <= contrast * 0.35
            and _gray_correlation(last, first) >= 0.96)


def _overlap_audio_peak(video: Path, seconds: float) -> tuple[bool, float | None]:
    """(has_audio, peak_dbfs) for the opening window a spoken seam would drop.

    ``peak_dbfs`` is None when the level could not be measured; callers must
    treat that as "unknown" and keep the segment untouched.
    """
    if seconds <= 0:
        return False, None
    if not video.is_file():
        return True, None
    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0",
             "-show_entries", "stream=index", "-of", "csv=p=0", str(video)],
            capture_output=True, text=True, timeout=30)
        if not probe.stdout.strip():
            return False, None
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-nostats", "-t", f"{seconds:.6f}",
             "-i", str(video), "-map", "0:a:0", "-af", "volumedetect",
             "-f", "null", "-"],
            capture_output=True, text=True, timeout=120)
        match = re.search(r"max_volume:\s*(-?inf|-?\d+(?:\.\d+)?)\s*dB", result.stderr)
        if not match:
            return True, None
        return True, -120.0 if "inf" in match.group(1) else float(match.group(1))
    except (OSError, ValueError, subprocess.SubprocessError):
        return True, None


def _select_handoff_tail_trim(previous: Path, current: Path, requested_frames: int) -> tuple[int, float | None, float | None]:
    """Trim less of the preceding clip only when its actual frames match better.

    H3 may regenerate the guided prefix from a state later than the nominal
    overlap. Blindly deleting all visible context frames then skips motion.
    Keep the established trim unless a candidate improves frame similarity
    by at least 5 percent, so calm seams already working stay unchanged.
    """
    requested = max(0, int(requested_frames))
    if requested == 0:
        return 0, None, None
    try:
        before, after = _handoff_gray_frames(previous), _handoff_gray_frames(current)
        limit = min(requested, len(before) - 1)
        if limit <= 0 or not after:
            return requested, None, None
        first = after[0]
        scores = [sum(abs(a - b) for a, b in zip(before[-(trim + 1)], first)) / len(first)
                  for trim in range(limit + 1)]
        baseline = scores[limit]
        selected = min(range(limit + 1), key=lambda trim: scores[trim])
        if scores[selected] >= baseline * 0.95:
            selected = limit
        return selected, round(baseline, 3), round(scores[selected], 3)
    except (OSError, ValueError, subprocess.SubprocessError):
        return requested, None, None


def _select_handoff_alignment(previous: Path, current: Path, requested_frames: int,
                              max_prefix_frames: int = 3) -> tuple[int, int, float | None, float | None]:
    """Align both sides of a guided seam without blending distinct subjects.

    A generated clip can start with one to three duplicate or unstable frames.
    Only discard them when the seam improves by at least 20 percent. The
    incoming action, audio and identity otherwise stay untouched.
    """
    requested = max(0, int(requested_frames))
    if requested == 0:
        return 0, 0, None, None
    try:
        before, after = _handoff_gray_frames(previous), _handoff_gray_frames(current)
        tail_limit = min(requested, len(before) - 1)
        prefix_limit = min(max(0, int(max_prefix_frames)), len(after) - 2)
        if tail_limit <= 0 or prefix_limit <= 0:
            trim, baseline, score = _select_handoff_tail_trim(previous, current, requested)
            return trim, 0, baseline, score

        def distance(tail: int, prefix: int) -> float:
            return sum(abs(a - b) for a, b in zip(before[-(tail + 1)], after[prefix])) / len(after[prefix])

        baseline = distance(tail_limit, 0)
        tail_scores = [(distance(tail, 0), tail) for tail in range(tail_limit + 1)]
        tail_best, tail_trim = min(tail_scores)
        if tail_best >= baseline * 0.95:
            tail_trim, tail_best = tail_limit, baseline

        candidates = [(distance(tail, prefix), tail, prefix)
                      for prefix in range(1, prefix_limit + 1)
                      for tail in range(tail_limit + 1)]
        best = min(score for score, _, _ in candidates)
        if best >= tail_best * 0.80:
            return tail_trim, 0, round(baseline, 3), round(tail_best, 3)
        # Keep the most action possible when nearly equivalent alignments exist.
        prefix, score, tail = min((prefix, score, tail) for score, tail, prefix in candidates
                                  if score <= best * 1.02)
        return tail, prefix, round(baseline, 3), round(score, 3)
    except (OSError, ValueError, subprocess.SubprocessError):
        return requested, 0, None, None


def _select_voiced_overlap_skip(previous: Path, current: Path,
                                requested_frames: int) -> tuple[int, float | None, float | None]:
    """Find the replayed preceding frame at the start of a narrated clip."""
    requested = max(0, int(requested_frames))
    if not requested:
        return 0, None, None
    try:
        before, after = _handoff_gray_frames(previous), _handoff_gray_frames(current)
        if not before or len(after) < 3:
            return 0, None, None
        limit = min(requested + 2, len(after) - 2)
        distances = [_gray_distance(before[-1], frame) for frame in after[:limit + 1]]
        best = min(range(len(distances)), key=distances.__getitem__)
        initial, matched = distances[0], distances[best]
        # Do not advance an unrelated composition. The matched frame must be
        # clearly closer to the outgoing frame and occur near the guide window.
        if best < 2 or matched >= initial * 0.72:
            return 0, round(initial, 3), round(matched, 3)
        return best, round(initial, 3), round(matched, 3)
    except (OSError, ValueError, subprocess.SubprocessError):
        return 0, None, None


def _gray_distance(first: bytes, second: bytes) -> float:
    return sum(abs(a - b) for a, b in zip(first, second)) / len(first)


def _gray_correlation(first: bytes, second: bytes) -> float:
    first_mean = sum(first) / len(first)
    second_mean = sum(second) / len(second)
    first_energy = sum((value - first_mean) ** 2 for value in first)
    second_energy = sum((value - second_mean) ** 2 for value in second)
    if first_energy <= 0 or second_energy <= 0:
        return 0.0
    covariance = sum((a - first_mean) * (b - second_mean) for a, b in zip(first, second))
    return covariance / math.sqrt(first_energy * second_energy)


def _stable_matching_view(before: list[bytes], after: list[bytes]) -> bool:
    """Scene independent confidence gate for optical flow and exposure repair."""
    if len(before) < 4 or len(after) < 2:
        return False
    last, first = before[-1], after[0]
    mean = sum(last) / len(last)
    contrast = math.sqrt(sum((value - mean) ** 2 for value in last) / len(last))
    seam = _gray_distance(last, first)
    return (contrast >= 8 and seam <= contrast * 0.18
            and _gray_correlation(last, first) >= 0.985)


def _match_handoff_exposure(previous: Path, current: Path, target: Path) -> tuple[float | None, float | None, float | None]:
    """Ease a small generated exposure step over the incoming clip's first 0.75 s."""
    try:
        before, after = _handoff_gray_frames(previous), _handoff_gray_frames(current)
        if not _stable_matching_view(before, after):
            return None, None, None
        seam = _gray_distance(before[-1], after[0])
        delta = (sum(before[-1]) - sum(after[0])) / len(after[0])
        previous_drift = abs(sum(before[-1]) - sum(before[-2])) / len(before[-1])
        next_drift = abs(sum(after[1]) - sum(after[0])) / len(after[0])
        mean = sum(before[-1]) / len(before[-1])
        contrast = math.sqrt(sum((value - mean) ** 2 for value in before[-1]) / len(before[-1]))
        if not (max(0.75, previous_drift * 1.5, next_drift * 1.5) <= abs(delta)
                <= contrast * 0.20):
            return None, None, None
        # The geq luma scale measured on generated YUV clips is about 1.15
        # grayscale units per requested unit. Preserve hue and saturation.
        gain = delta / 1.15
        filter_graph = ("format=yuv420p,geq="
                        f"lum='clip(lum(X,Y)+({gain:.4f})*max(0,1-T/0.75),0,255)':"
                        "cb='cb(X,Y)':cr='cr(X,Y)'")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(current),
                        "-vf", filter_graph, "-c:v", "libx264", "-preset", "fast",
                        "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "copy", str(target)],
                       check=True, timeout=300)
        matched = _handoff_gray_frames(target)
        adjusted = _gray_distance(before[-1], matched[0])
        brightness_error = abs(sum(before[-1]) - sum(matched[0])) / len(matched[0])
        # A one-level flash has little effect on structural distance, but is
        # visible across an otherwise stable view. Verify its luma separately.
        improvement = 0.95 if abs(delta) < 1.5 else 0.85
        if (len(matched) != len(after) or adjusted >= seam * improvement
                or brightness_error > abs(delta) * 0.5):
            target.unlink(missing_ok=True)
            return None, None, None
        return round(seam, 3), round(adjusted, 3), round(gain, 3)
    except (OSError, ValueError, subprocess.SubprocessError):
        target.unlink(missing_ok=True)
        return None, None, None


def _smooth_handoff_motion(previous: Path, current: Path, target: Path,
                           max_frames: int = 3) -> tuple[float | None, float | None, int]:
    """Replace up to three tail frames with motion estimates for a calm seam.

    This preserves clip length and audio timing. An unrelated scene, a close
    subject, or a poor optical-flow estimate keeps the original frames.
    """
    try:
        before, after = _handoff_gray_frames(previous), _handoff_gray_frames(current)
        if len(before) < 5 or len(after) < 2 or max_frames < 2:
            return None, None, 0
        seam = _gray_distance(before[-1], after[0])
        prior_motion = max(_gray_distance(before[-3], before[-2]),
                           _gray_distance(before[-2], before[-1]))
        next_motion = _gray_distance(after[0], after[1])
        # Flow across a large scene change or a moving close-up can make a
        # distracting hybrid person. Only bridge stable, visually similar views.
        if not (_stable_matching_view(before, after)
                and seam >= max(prior_motion, next_motion, 0.5) * 1.5):
            return None, None, 0
        with tempfile.TemporaryDirectory(prefix="h3_bridge_", dir=target.parent) as temporary:
            work = Path(temporary)
            incoming = work / "incoming.png"
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(current),
                            "-vf", "select=eq(n\\,0)", "-frames:v", "1", str(incoming)],
                           check=True, timeout=120)
            original_peak = max(prior_motion, seam)
            best: tuple[float, int, Path] | None = None
            for bridge_frames in range(2, min(3, int(max_frames)) + 1):
                stable = work / f"stable_{bridge_frames}.png"
                subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(previous),
                                "-vf", f"reverse,select=eq(n\\,{bridge_frames})",
                                "-frames:v", "1", str(stable)], check=True, timeout=120)
                for index, image in enumerate((stable, stable, incoming, incoming, incoming)):
                    shutil.copyfile(image, work / f"input_{index}.png")
                fps = 12 * (bridge_frames + 1)
                subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "12",
                                "-i", str(work / "input_%d.png"),
                                "-vf", f"minterpolate=fps={fps}:mi_mode=mci:mc_mode=obmc:me_mode=bidir:vsbmc=1",
                                "-frames:v", str(2 * (bridge_frames + 1)),
                                str(work / "estimate_%02d.png")], check=True, timeout=120)
                estimates = [work / f"estimate_{bridge_frames + index + 2:02d}.png"
                             for index in range(1, bridge_frames + 1)]
                if not all(image.is_file() for image in estimates):
                    continue
                inputs = []
                filters = [f"[0:v]fps=24,trim=end_frame={len(before) - bridge_frames},"
                           "setpts=PTS-STARTPTS,format=yuv420p[v0]"]
                for index, image in enumerate(estimates, 1):
                    inputs.extend(["-loop", "1", "-framerate", "24", "-i", str(image)])
                    filters.append(f"[{index}:v]trim=end_frame=1,setpts=PTS-STARTPTS,format=yuv420p[v{index}]")
                filters.append("".join(f"[v{index}]" for index in range(bridge_frames + 1))
                               + f"concat=n={bridge_frames + 1}:v=1:a=0[v]")
                candidate = work / f"candidate_{bridge_frames}.mp4"
                subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(previous),
                                *inputs, "-filter_complex", ";".join(filters), "-map", "[v]",
                                "-map", "0:a:0?", "-frames:v", str(len(before)), "-r", "24",
                                "-fps_mode", "cfr", "-c:v", "libx264", "-preset", "fast",
                                "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "copy", str(candidate)],
                               check=True, timeout=300)
                bridged = _handoff_gray_frames(candidate)
                if len(bridged) != len(before):
                    continue
                peak = max([_gray_distance(bridged[-index - 1], bridged[-index])
                            for index in range(bridge_frames, 0, -1)]
                           + [_gray_distance(bridged[-1], after[0])])
                if best is None or peak < best[0] * 0.95:
                    best = (peak, bridge_frames, candidate)
            if best is None or best[0] > original_peak * 0.85:
                return None, None, 0
            shutil.copyfile(best[2], target)
            return round(original_peak, 3), round(best[0], 3), best[1]
    except (OSError, ValueError, subprocess.SubprocessError):
        target.unlink(missing_ok=True)
        return None, None, 0


def _audio_mode(video: Path, source_video: Path | None, mode: str) -> None:
    if mode not in ("source", "mute"):
        return
    temp = video.with_name(video.stem + "_audio.mp4")
    if mode == "mute":
        command = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(video), "-c:v", "copy", "-an", str(temp)]
    elif source_video:
        command = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(video), "-i", str(source_video),
                   "-map", "0:v:0", "-map", "1:a:0?", "-c:v", "copy", "-c:a", "aac", "-shortest", str(temp)]
    else:
        return
    subprocess.run(command, check=True, timeout=300)
    temp.replace(video)


def _write_merge_audio(videos: list[Path], target: Path) -> None:
    """Join decoded samples at the exact segment boundaries, without overlap.

    A sample clock avoids concat-demuxer timestamp rounding and AAC priming
    at every cut. Only codec padding outside each picture's duration is
    removed; missing audio is filled with silence.
    """
    with target.open("wb") as master:
        for path in videos:
            info = subprocess.run(
                ["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(path)],
                capture_output=True, text=True, check=True, timeout=30)
            streams = json.loads(info.stdout).get("streams", [])
            picture = next(stream for stream in streams if stream.get("codec_type") == "video")
            seconds = float(picture.get("duration") or video_duration(path))
            samples = round(seconds * 48000)
            has_audio = any(stream.get("codec_type") == "audio" for stream in streams)
            command = ["ffmpeg", "-v", "error"]
            if has_audio:
                command += ["-i", str(path), "-map", "0:a:0"]
            else:
                command += ["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo"]
            command += ["-af", f"aresample=48000,apad=whole_len={samples},atrim=end_sample={samples}",
                        "-ac", "2", "-ar", "48000", "-c:a", "pcm_s16le", "-f", "s16le", "pipe:1"]
            subprocess.run(command, stdout=master, check=True, timeout=600)


def _fit_output_size(video: Path, output_dir: Path, index: int, project: dict) -> Path:
    """Rescale the finished shot to the exact delivery size.

    H3's native canvas is 1344x768 (1.75:1), so a 16:9 delivery comes out of
    the upscaler as 1890x1080 rather than 1920x1080.  This stage scales to
    cover the target and center-crops the overflow: no letterbox bars, no
    stretching.  A target of 0 on either side keeps the video as it is.
    """
    target_w = int(project.get("output_width") or 0)
    target_h = int(project.get("output_height") or 0)
    if target_w <= 0 or target_h <= 0:
        return video
    width, height = _video_dimensions(video)
    if (width, height) == (target_w, target_h):
        return video
    fitted = output_dir / f"shot_{index:03d}_fit.mp4"
    vf = (f"scale={target_w}:{target_h}:force_original_aspect_ratio=increase,"
          f"crop={target_w}:{target_h}:(iw-{target_w})/2:(ih-{target_h})/2,"
          "setsar=1")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-threads", "2", "-i", str(video),
                    "-vf", vf, "-filter_threads", "2", "-c:v", "libx264", "-threads", "2",
                    "-preset", "fast", "-crf", "17",
                    "-pix_fmt", "yuv420p", "-c:a", "copy", str(fitted)],
                   check=True, timeout=900)
    # 不覆盖原文件：checkpoint 里的「H3 原片」必须还是原始采样结果。
    return fitted


def _merge(videos: list[Path], target: Path, width: int, height: int) -> None:
    """把分镜拼成一条片。

    画布以素材实际尺寸为准，而不是项目画布：开了超分后每个镜头是
    1890×1080，若这里再按项目画布（1344×768）缩放，超分的收益会在最后
    一步被吃掉。
    """
    normalized = []
    sizes = []
    probed = []
    for path in videos:
        info = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(path)],
                              capture_output=True, text=True, check=True, timeout=30)
        streams = json.loads(info.stdout).get("streams", [])
        picture = next((stream for stream in streams if stream.get("codec_type") == "video"), None)
        if picture:
            sizes.append((int(picture.get("width") or width), int(picture.get("height") or height)))
        probed.append((path, any(stream.get("codec_type") == "audio" for stream in streams)))
    if sizes:
        # 画布以素材实际尺寸为准：开了超分后镜头是 1890×1080，若这里再按项目
        # 画布（1344×768）缩放，超分的收益会在最后一步被吃掉。
        width = max(item[0] for item in sizes)
        height = max(item[1] for item in sizes)
    for index, (path, has_audio) in enumerate(probed):
        output = target.with_name(f"merge_norm_{index:03d}.mkv")
        command = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(path)]
        if not has_audio:
            command += ["-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000"]
        command += ["-map", "0:v:0", "-map", "0:a:0" if has_audio else "1:a:0",
                    "-vf", f"scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,fps=24,setpts=PTS-STARTPTS",
                    "-af", "aresample=48000:async=1:first_pts=0,asetpts=PTS-STARTPTS,apad",
                    "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                    "-pix_fmt", "yuv420p", "-c:a", "pcm_s16le", "-ar", "48000", "-ac", "2",
                    "-shortest", str(output)]
        subprocess.run(command, check=True, timeout=600)
        normalized.append(output)
    manifest = target.with_suffix(".txt")
    manifest.write_text("".join("file '" + str(path).replace("'", "'\\''") + "'\n" for path in normalized))
    audio_master = target.with_suffix(".audio.s16le")
    try:
        _write_merge_audio(videos, audio_master)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
                        "-i", str(manifest), "-f", "s16le", "-ar", "48000", "-ac", "2",
                        "-i", str(audio_master), "-map", "0:v:0", "-map", "1:a:0",
                        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                        "-ar", "48000", "-ac", "2", "-movflags", "+faststart",
                        str(target)], check=True, timeout=600)
    finally:
        audio_master.unlink(missing_ok=True)
        manifest.unlink(missing_ok=True)
        for path in normalized:
            path.unlink(missing_ok=True)


def _append_merge_item(merge_items: list[Path], seam_adjustments: list[dict], path: Path,
                       item: dict, item_index: int, output_dir: Path, project: dict,
                       file_prefix: str = "merge", preserve_spoken_audio: bool = False) -> None:
    if merge_items and item.get("transition_mode") == "dissolve":
        transitioned = output_dir / f"{file_prefix}_dissolve_{item_index:03d}.mp4"
        _dissolve_entry_keep_audio(merge_items[-1], path, transitioned)
        seam_adjustments.append({"into_shot_id": item.get("shot_id"),
                                 "transition_mode": "dissolve", "transition_frames": 12,
                                 "audio_preserved": True})
        merge_items.append(transitioned)
        return
    flash_fixed = output_dir / f"{file_prefix}_flash_{item_index:03d}.mp4"
    if _suppress_entry_flash(path, flash_fixed):
        path = flash_fixed
    visible_frames = max(0, int(item.get("handoff_visible_frames", 0) or 0))
    if visible_frames and merge_items:
        mode = project.get("seam_mode", "auto")
        if preserve_spoken_audio:
            # Spoken seams never trim the outgoing tail. The duplicated guide head
            # is either dropped at the frame that is both silent and best aligned,
            # or hidden behind a short dissolve: both keep every audio sample.
            has_audio, peak = _overlap_audio_peak(path, visible_frames / 24)
            quiet = not has_audio or (peak is not None and peak <= SPOKEN_OVERLAP_MAX_DBFS)
            report = {"into_shot_id": item.get("shot_id"),
                      "requested_trim_frames": visible_frames,
                      "selected_trim_frames": 0,
                      "skipped_prefix_frames": 0,
                      "audio_preserved": True, "audio_video_timeline_preserved": True,
                      "spoken_overlap": "silent" if quiet else ("unknown" if peak is None else "speech"),
                      "overlap_peak_dbfs": peak}
            if quiet:
                # Align the cut instead of blindly dropping the visible window.
                skip = _narrated_head_skip(path, merge_items[-1], visible_frames)
                report["skipped_prefix_frames"] = skip
                if skip:
                    trimmed_head = output_dir / f"{file_prefix}_spoken_head_{item_index:03d}.mp4"
                    _trim_prefix_frames(path, skip, trimmed_head)
                    path = trimmed_head
                elif _views_are_stable(merge_items[-1], path, blend=True):
                    dissolve_frames = max(2, min(visible_frames, 6))
                    blended = output_dir / f"{file_prefix}_spoken_blend_{item_index:03d}.mp4"
                    _dissolve_entry_keep_audio(merge_items[-1], path, blended, dissolve_frames)
                    path = blended
                    report["transition_mode"] = "dissolve"
                    report["transition_frames"] = dissolve_frames
            elif _views_are_stable(merge_items[-1], path, blend=True):
                # Speech occupies the duplicated head, so time cannot be removed.
                # Fade that head out of the outgoing last frame instead; audio and
                # clip length stay untouched.
                dissolve_frames = max(2, min(visible_frames, 6))
                blended = output_dir / f"{file_prefix}_spoken_blend_{item_index:03d}.mp4"
                _dissolve_entry_keep_audio(merge_items[-1], path, blended, dissolve_frames)
                path = blended
                report["transition_mode"] = "dissolve"
                report["transition_frames"] = dissolve_frames
            exposure_enabled = mode == "auto" or (mode == "manual" and project.get("seam_exposure", True))
            if exposure_enabled:
                matched_current = output_dir / f"{file_prefix}_spoken_exposure_{item_index:03d}.mp4"
                exposure_before, exposure_after, gain = _match_handoff_exposure(
                    merge_items[-1], path, matched_current)
                if exposure_after is not None:
                    path = matched_current
                    report.update({"exposure_score_before": exposure_before,
                                   "exposure_score_after": exposure_after, "exposure_gain": gain})
            seam_adjustments.append(report)
            merge_items.append(path)
            return
        max_prefix = (0 if mode == "conservative" else
                      max(0, min(3, int(project.get("seam_prefix_frames", 3)))) if mode == "manual" else 3)
        chosen_frames, skipped_frames, before_score, after_score = _select_handoff_alignment(
            merge_items[-1], path, visible_frames, max_prefix)
        report = {"into_shot_id": item.get("shot_id"), "requested_trim_frames": visible_frames,
                  "selected_trim_frames": chosen_frames, "skipped_prefix_frames": skipped_frames,
                  "score_before": before_score, "score_after": after_score}
        seam_adjustments.append(report)
        if chosen_frames:
            trimmed_previous = output_dir / f"{file_prefix}_handoff_tail_{item_index:03d}.mp4"
            _trim_suffix(merge_items[-1], chosen_frames, trimmed_previous)
            merge_items[-1] = trimmed_previous
        if skipped_frames:
            trimmed_current = output_dir / f"{file_prefix}_handoff_prefix_{item_index:03d}.mp4"
            _trim_prefix_frames(path, skipped_frames, trimmed_current)
            path = trimmed_current
        exposure_enabled = mode == "auto" or (mode == "manual" and project.get("seam_exposure", True))
        if exposure_enabled:
            matched_current = output_dir / f"{file_prefix}_handoff_exposure_{item_index:03d}.mp4"
            exposure_before, exposure_after, gain = _match_handoff_exposure(
                merge_items[-1], path, matched_current)
            if exposure_after is not None:
                path = matched_current
                report.update({"exposure_score_before": exposure_before,
                               "exposure_score_after": exposure_after, "exposure_gain": gain})
        max_motion = (0 if mode == "conservative" else
                      int(project.get("seam_motion_frames", 3)) if mode == "manual" else 3)
        if max_motion >= 2:
            bridged_previous = output_dir / f"{file_prefix}_handoff_motion_{item_index:03d}.mp4"
            motion_before, motion_after, bridge_frames = _smooth_handoff_motion(
                merge_items[-1], path, bridged_previous, max_motion)
            if motion_after is not None:
                merge_items[-1] = bridged_previous
                report.update({"motion_bridge_frames": bridge_frames,
                               "motion_peak_before": motion_before,
                               "motion_peak_after": motion_after})
    merge_items.append(path)


def _reroll_result_list(source_items: list[dict], shot_id: str, replacement: dict) -> list[dict]:
    """Replace exactly one segment while preserving all other saved files and order."""
    if sum(item.get("shot_id") == shot_id for item in source_items) != 1:
        raise ValueError("原成片中无法唯一定位要重抽的镜头")
    if replacement.get("shot_id") != shot_id or not replacement.get("file"):
        raise ValueError("重抽结果与选定镜头不一致")
    return [replacement if item.get("shot_id") == shot_id else item.copy()
            for item in source_items]


def remerge_run(run_id: str, project: dict) -> dict:
    """Apply current project seam settings to saved segment files, without rerendering H3."""
    run = read()["runs"].get(run_id)
    if not run or run.get("project_id") != project.get("id") or run.get("status") != "completed":
        raise ValueError("找不到当前项目已完成的生成任务")
    items = [item for item in run.get("results", []) if item.get("file")]
    if len(items) < 2 or any(not (OUTPUTS / item["file"]).is_file() for item in items):
        raise ValueError("至少需要两个仍存在的分段视频才能重新拼接")
    output_dir = OUTPUTS / run_id
    output_dir.mkdir(exist_ok=True)
    stamp = str(time.time_ns())
    paths: list[Path] = []
    seams: list[dict] = []
    shot_by_id = {shot.get("id"): shot for shot in project.get("shots", [])}
    for index, item in enumerate(items):
        current_shot = shot_by_id.get(item.get("shot_id"), {})
        previous_shot = shot_by_id.get(items[index - 1].get("shot_id"), {}) if index else {}
        voiced = _preserve_spoken_seam(previous_shot, current_shot,
                                       int(item.get("handoff_visible_frames") or 0))
        _append_merge_item(paths, seams, OUTPUTS / item["file"], item, index,
                           output_dir, project, f"remerge_{stamp}", voiced)
    merged = output_dir / f"merged_remerge_{stamp}.mp4"
    _merge(paths, merged, int(project.get("width", 1024)), int(project.get("height", 576)))
    update_run(run_id, merged=str(merged.relative_to(OUTPUTS)),
               merge_seam_algorithm="adaptive_structural_v7_continuous_audio", merge_seam_adjustments=seams,
               remerged_at=time.time(), remerge_settings={key: project.get(key) for key in
                                                          ("seam_mode", "seam_prefix_frames",
                                                           "seam_motion_frames", "seam_exposure")})
    return read()["runs"][run_id]


def _native_take_shot(shots: list[dict], project: dict) -> dict | None:
    """Fit a short sequence in one H3 sample so it has no generated segment seam."""
    if (not project.get("continuous_take", True)
            or project.get("continuous_render_mode", "auto") == "segmented"
            or len(shots) < 2
            or len(shots) != len(project.get("shots", []))
            or any(shot_refine_mode(shot, project) != "off" or shot_upscale(shot, project)
                   for shot in shots)
            or project.get("face_refine") or project.get("selflift")):
        return None
    if any(shot.get("mode") != "r2v" or shot.get("source_video")
           or shot.get("first_image") or shot.get("last_image")
           or _has_spoken_audio(shot) for shot in shots):
        return None
    duration = sum(float(shot.get("duration") or 5) for shot in shots)
    if duration > 15:
        return None
    prepared = [prepare_shot(shot, project) for shot in shots]
    images = list(dict.fromkeys(asset_id for shot in prepared for asset_id in shot.get("ref_images", [])))
    videos = list(dict.fromkeys(asset_id for shot in prepared for asset_id in shot.get("ref_videos", [])))
    audios = list(dict.fromkeys(asset_id for shot in prepared for asset_id in shot.get("ref_audios", [])))
    # Video/audio tags also have local slot numbers; keep those workflows on
    # the existing path until their per-beat bindings can be remapped safely.
    if len(images) > 9 or videos or audios:
        return None
    if (len({shot.get("audio_mode") for shot in prepared}) != 1
            or len({shot.get("prompt_language", "en") for shot in prepared}) != 1):
        return None
    fused = copy.deepcopy(prepared[0])
    fused.update(duration=duration, ref_images=images, ref_videos=videos, ref_audios=audios,
                 continuous_take=True, continuity=False)
    fused["_h3_reference_names"] = {key: value for shot in prepared
                                    for key, value in shot.get("_h3_reference_names", {}).items()}
    fused["_h3_reference_descriptions"] = {key: value for shot in prepared
                                           for key, value in shot.get("_h3_reference_descriptions", {}).items()}
    fused["ref_image_roles"] = {key: value for shot in prepared
                                  for key, value in shot.get("ref_image_roles", {}).items()}
    fused["_h3_characters"] = list({item["subject_number"]: item for shot in prepared
                                    for item in shot.get("_h3_characters", [])}.values())
    fused["_h3_entities"] = list({item["id"]: item for shot in prepared
                                  for item in shot.get("_h3_entities", [])}.values())
    parts = ["One physically continuous camera take from beginning to end. "
             "The following beats are successive moments of the same unbroken scene; camera motion, "
             "screen direction, and subject actions continue without reset or repetition."]
    elapsed = 0.0
    for index, shot in enumerate(prepared):
        # Prepared prompts use image slots local to each shot. Remap them to
        # the one reference list that the fused H3 request actually receives.
        local_images = shot.get("ref_images", [])
        def remap(match):
            local = int(match.group(1)) - 1
            if not 0 <= local < len(local_images):
                return match.group(0)
            return f"<Picture {images.index(local_images[local]) + 1}>"
        prompt = str(shot.get("prompt", ""))
        controls = director_control_text(shot)
        if controls:
            prompt += "\n" + controls
        prompt = re.sub(r"<Picture\s+(\d+)>", remap, prompt, flags=re.IGNORECASE)
        end = elapsed + float(shots[index].get("duration") or 5)
        parts.append(f"[{elapsed:g}s-{end:g}s] Beat {index + 1}: {prompt}")
        elapsed = end
    fused["prompt"] = "\n".join(parts)
    # Every beat already carries its own controls. Do not repeat the first
    # shot's selections over the whole native take during final compilation.
    fused["suppress_director_controls"] = True
    return fused


def _run_native_take(run_id: str, shot: dict, project: dict, assets: dict,
                     output_dir: Path, shot_count: int) -> None:
    """Render a short multi-beat take once; never present stitching as native continuity."""
    update_run(run_id, current="原生一镜到底 · H3 视频生成中", progress=f"0/{shot_count}",
               render_strategy="native_single_generation")
    media = _make_media_map(shot, project, assets)
    graph = build_h3_graph(shot, project, media, f"h3_studio_{run_id}_native")
    prompt_id = _resume_prompt(run_id, "native_h3", shot["id"])
    if not prompt_id:
        prompt_id = _submit(graph, run_id)
        update_run(run_id, prompt_id=prompt_id, active_stage="native_h3",
                   active_shot_id=shot["id"], active_prefix=f"h3_studio_{run_id}_native")
    history = _wait(prompt_id, run_id)
    if read()["runs"][run_id].get("cancel_requested"):
        raise RuntimeError("用户取消任务")
    video = output_dir / "merged.mp4"
    _download_video(history, video, f"h3_studio_{run_id}_native")
    _audio_mode(video, None, shot.get("audio_mode", "generate"))
    update_run(run_id, merged=str(video.relative_to(OUTPUTS)), progress=f"{shot_count}/{shot_count}",
               results=[{"shot_id": shot["id"], "file": str(video.relative_to(OUTPUTS)),
                         "native_take": True}], status="completed", current="", finished_at=time.time(),
               active_stage="", active_shot_id="", active_prefix="")


def _selected_results_for_merge(shots: list[dict], selected_ids: set[str],
                                results: list[dict]) -> list[tuple[dict, dict]]:
    """Only authored shots selected for this run belong in its normal export."""
    by_id = {item["shot_id"]: item for item in results}
    return [(shot, by_id[shot["id"]]) for shot in shots
            if shot["id"] in selected_ids and shot["id"] in by_id]


@_serialized_director_run
def run_project(run_id: str) -> None:
    """Process selected shots sequentially; keep results even if a later shot fails."""
    state = read()
    run = state["runs"][run_id]
    assets = state["assets"]
    project = normalized_project_image_references(state["projects"][run["project_id"]], assets)
    source_run = state["runs"].get(run.get("source_run_id")) if run.get("reroll_shot_id") else None
    output_dir = OUTPUTS / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    saved_results = {}
    prior_runs = sorted((candidate for candidate in state["runs"].values()
                         if candidate.get("project_id") == run["project_id"] and candidate.get("id") != run_id),
                        key=lambda candidate: candidate.get("created_at", 0))
    for prior in prior_runs:
        for item in prior.get("results", []):
            if item.get("signature") and (OUTPUTS / item.get("file", "")).is_file():
                saved_results[item.get("shot_id")] = item
    saved_results.update({item.get("shot_id"): item for item in run.get("results", [])})
    previous = None
    previous_signature = None
    try:
        update_run(run_id, status="running", started_at=time.time())
        shots = [shot for shot in project["shots"] if shot["id"] in run["shot_ids"]]
        native_take = _native_take_shot(shots, project)
        if native_take:
            _run_native_take(run_id, native_take, project, assets, output_dir, len(shots))
            return
        update_run(run_id, render_strategy="segmented_continuation" if len(shots) > 1 else "single_shot")
        reused_shot_ids: list[str] = []
        for index, original in enumerate(shots):
            if read()["runs"][run_id].get("cancel_requested"):
                raise RuntimeError("用户取消任务")
            shot = prepare_shot(original, project)
            source_items = source_run.get("results", []) if source_run else []
            source_index = next((i for i, item in enumerate(source_items)
                                 if item.get("shot_id") == original["id"]), -1)
            preceding_item = source_items[source_index - 1] if source_index > 0 else None
            preceding_original = (next((item for item in project["shots"]
                                        if item["id"] == preceding_item.get("shot_id")), None)
                                  if preceding_item else shots[index - 1] if index > 0 else None)
            if source_run and preceding_item and preceding_original:
                previous = _continuity_source(preceding_item)
                previous_signature = preceding_item.get("signature")
            continuity = _continuity_policy(project, original,
                                            source_index if source_run else index)
            fresh_composition = (bool(project.get("continuous_take", True)) and
                                 _requests_fresh_composition(preceding_original, original))
            if fresh_composition:
                continuity = {"enabled": False, "frames": 0, "strength": "none"}
                shot["transition_mode"] = "dissolve"
            if preceding_original:
                # Selected character references identify who continues across
                # the segment boundary; shared references identify the same cast.
                previous_identities = _shot_identity_refs(preceding_original, project, assets)
                current_identities = _shot_identity_refs(original, project, assets)
                shot["continuity_subject_relation"] = _continuity_subject_relation(
                    previous_identities, current_identities)
                identity_changed = _identity_handoff_changed(
                    previous_identities, current_identities,
                    strict=bool(project.get("identity_first_on_switch")))
                if identity_changed:
                    _strengthen_incoming_identity_reference(shot, current_identities)
                    if project.get("identity_first_on_switch"):
                        continuity = {"enabled": False, "frames": 0, "strength": "none"}
                    elif continuity.get("enabled"):
                        # Keep the final, usually empty scene frames for camera
                        # continuity without carrying the old person's closeup
                        # through the long H3 guide window.
                        continuity["frames"] = min(
                            continuity["frames"], NEW_IDENTITY_CONTEXT_FRAMES)
            shot["continuous_take"] = bool(project.get("continuous_take", True)) and not fresh_composition
            shot["continuity"] = continuity["enabled"]
            shot["continuity_frames"] = continuity["frames"]
            shot["continuity_strength"] = continuity["strength"]
            shot["continuity_seam_redraw"] = continuity.get("seam_redraw", 0.5)
            shot["continuity_frames"], shot["continuity_visible_frames"] = _handoff_window(
                shot.get("continuity_subject_relation", "same") if continuity.get("enabled") else "none",
                shot["continuity_frames"])
            if shot["continuity_frames"] and _has_spoken_audio(original):
                # Keep the drop-able head at one frame for narrated handoffs.
                shot["continuity_frames"] = min(shot["continuity_frames"], NARRATED_HANDOFF_FRAMES)
                shot["continuity_visible_frames"] = min(shot["continuity_visible_frames"],
                                                       NARRATED_HANDOFF_VISIBLE, shot["continuity_frames"])
            if shot["continuity_frames"]:
                shot["continuity_guide_frames"] = _continuity_anchor_frames(shot["continuity_frames"])
                # Align the effective duration with the *length* contribution of the
                # handoff window: build_h3_graph requests duration + continuity_frames.
                shot["_h3_effective_duration"] = _grid_duration(
                    float(shot.get("duration", 5)), shot["continuity_frames"])
            render_settings = copy.deepcopy(project)
            render_settings["refine_mode"] = shot_refine_mode(original, project)
            render_settings["_h3_has_continuity"] = bool(original.get("continuity_video"))
            if project.get("memory_policy") == "auto_vram":
                memory_plan = _auto_memory_plan(int(project.get("width", 1024)), int(project.get("height", 576)))
                render_settings["width"], render_settings["height"] = memory_plan["width"], memory_plan["height"]
                if render_settings.get("refine_mode") in ("latent", "latent_only"):
                    requested_refine = (int(project.get("refine_width") or memory_plan["width"]),
                                        int(project.get("refine_height") or memory_plan["height"]))
                    refined = fit_canvas(*requested_refine, memory_plan["budget_megapixels"])
                    refined = (max(memory_plan["width"], refined[0]), max(memory_plan["height"], refined[1]))
                    render_settings["refine_width"], render_settings["refine_height"] = refined
                    memory_plan["refine_requested_width"], memory_plan["refine_requested_height"] = requested_refine
                    memory_plan["refine_width"], memory_plan["refine_height"] = refined
                shot["_h3_memory_plan"] = memory_plan
                shot["_h3_effective_refine_mode"] = render_settings.get("refine_mode", "off")
                old_plans = read()["runs"][run_id].get("memory_plans", [])
                plans = [item for item in old_plans if item.get("shot_id") != shot["id"]]
                plans.append({"shot_id": shot["id"], "shot_name": shot.get("name", ""), **memory_plan})
                update_run(run_id, memory_plans=plans)
            signature = _shot_signature(shot, project, assets, previous_signature)
            cached = saved_results.get(shot["id"])
            if (not run.get("reroll_shot_id") and cached and cached.get("signature") == signature
                    and (OUTPUTS / cached.get("file", "")).is_file()):
                # Refresh the length from the file so cached shots report what the
                # film actually contains, not the authored duration.
                cached_seconds = round(round(video_duration(OUTPUTS / cached["file"]) * 24) / 24, 3)
                results.append({**cached, "effective_duration": cached_seconds,
                                "planned_duration": cached.get("planned_duration") or cached_seconds})
                previous = _continuity_source(cached)
                previous_signature = signature
                reused_shot_ids.append(shot["id"])
                update_run(run_id, results=results, progress=f"{index + 1}/{len(shots)}",
                           current=shot["name"] + " · 已复用缓存", reused_shot_ids=reused_shot_ids)
                continue
            update_run(run_id, current=shot["name"] + " · 准备生成", progress=f"{index}/{len(shots)}")
            media = _make_media_map(shot, render_settings, assets)
            if previous and shot.get("continuity"):
                context_frames = int(shot.get("continuity_frames", 22))
                guide_frames = int(shot.get("continuity_guide_frames") or _continuity_anchor_frames(context_frames))
                clip = output_dir / f"continuity_{index:03d}.mp4"
                _tail_clip(previous, clip, guide_frames)
                filename = f"h3studio_{run_id}_continuity_{index:03d}.mp4"
                shutil.copy2(clip, COMFY_INPUT / filename)
                continuity_id = "__continuity__"
                media[continuity_id] = filename
                shot["continuity_video"] = continuity_id
                shot["continuity_frames"] = context_frames
            prefix = f"h3_studio_{run_id}_{index:03d}"
            checkpoints = read()["runs"][run_id].get("checkpoints", {})
            checkpoint = checkpoints.get(shot["id"], {})
            first_pass = OUTPUTS / checkpoint.get("h3_file", "")
            pre_refine = OUTPUTS / checkpoint["pre_refine_file"] if checkpoint.get("pre_refine_file") else None
            # 采样计划先算好：界面写着 8 步、实际跑 4 步这种事必须一眼可见。
            try:
                from engine import sampler_summary
                sampler_info = sampler_summary(render_settings, shot.get("mode", "t2v"))
                sampler_label = sampler_info["label"]
            except Exception:
                sampler_info, sampler_label = None, ""
            if checkpoint.get("signature") != signature or not first_pass.is_file() or (pre_refine and not pre_refine.is_file()):
                prompt_id = _resume_prompt(run_id, "h3", shot["id"])
                if not prompt_id:
                    graph = build_h3_graph(shot, render_settings, media, prefix)
                    if sampler_info:
                        try:
                            update_run(run_id, sampler=sampler_info)
                        except Exception:  # 摘要失败不能影响渲染
                            pass
                    if read()["runs"][run_id].get("cancel_requested"):
                        raise RuntimeError("用户取消任务")
                    prompt_id = _submit(graph, run_id)
                    update_run(run_id, prompt_id=prompt_id, active_stage="h3",
                               active_shot_id=shot["id"], active_prefix=prefix,
                               current=shot["name"] + " · H3 视频生成中"
                               + (f"（{sampler_label}）" if sampler_label else ""))
                else:
                    update_run(run_id, current=shot["name"] + " · 恢复 ComfyUI H3 任务")
                first_pass = output_dir / f"shot_{index:03d}_h3.mp4"
                history = _wait(prompt_id, run_id)
                _download_video(history, first_pass, prefix)
                pre_refine = None
                if render_settings.get("refine_mode") in ("same", "latent", "latent_only"):
                    pre_refine = output_dir / f"shot_{index:03d}_pre_refine.mp4"
                    _download_video(history, pre_refine, prefix + "_pre")
                if read()["runs"][run_id].get("cancel_requested"):
                    raise RuntimeError("用户取消任务")
                if shot.get("continuity_video"):
                    plan = _trim_downloaded_shot(first_pass, shot, previous)
                    shot["_h3_trim_plan"] = plan
                    if pre_refine:
                        _trim_downloaded_shot(pre_refine, shot, previous)
                shot["_h3_entry_flash"] = _suppress_entry_flash(first_pass)
                checkpoint = {"signature": signature, "h3_file": str(first_pass.relative_to(OUTPUTS)),
                              "pre_refine_file": str(pre_refine.relative_to(OUTPUTS)) if pre_refine else None}
                def save_checkpoint(data):
                    data["runs"][run_id].setdefault("checkpoints", {})[shot["id"]] = checkpoint
                    data["runs"][run_id].update(active_stage="", active_shot_id="", active_prefix="")
                change(save_checkpoint)
            update_run(run_id, preview_file=str(first_pass.relative_to(OUTPUTS)),
                       pre_refine_file=str(pre_refine.relative_to(OUTPUTS)) if pre_refine else None)
            final = first_pass
            perform_upscale = shot_upscale(original, project)
            if perform_upscale:
                original_size = _video_dimensions(first_pass)
                if not _seedvr2_enlarges(*original_size,
                                         project.get("upscale_resolution", 720),
                                         project.get("upscale_max_resolution", 1280)):
                    perform_upscale = False
                    update_run(run_id, upscale_skipped_reason=(
                        f"SeedVR2 已跳过：目标 {project.get('upscale_resolution', 720)}p / "
                        f"最长边 {project.get('upscale_max_resolution', 1280)}px "
                        f"不高于 H3 原片 {original_size[0]}×{original_size[1]}，"
                        "运行超分不会增加输出像素。"))
            if perform_upscale:
                resumed_upscale_prompt = _resume_prompt(run_id, "seedvr2", shot["id"])
                if project.get("confirm_upscale", True) and not resumed_upscale_prompt:
                    update_run(run_id, status="awaiting_upscale", current=shot["name"] + " · 等待确认超分",
                               upscale_approved=False, skip_upscale=False)
                    while True:
                        decision = read()["runs"][run_id]
                        if decision.get("cancel_requested"):
                            raise RuntimeError("用户取消任务")
                        if decision.get("upscale_approved") or decision.get("skip_upscale"):
                            break
                        # 超分可能在等待确认期间被关掉。这时不该继续拦着用户点确认，
                        # 直接按"跳过超分"处理，继续跑后面的镜头。
                        if not _upscale_still_enabled(original, project, run["project_id"]):
                            decision = {"skip_upscale": True}
                            update_run(run_id, skip_upscale=True, upscale_skipped_reason=(
                                "超分已在等待确认期间关闭，自动跳过超分。"))
                            break
                        time.sleep(1)
                    update_run(run_id, status="running")
                else:
                    decision = {}
                if decision.get("skip_upscale"):
                    final = first_pass
                else:
                    upscale_started_at = time.time()
                    input_name = f"h3studio_{run_id}_{index:03d}_first.mp4"
                    shutil.copy2(first_pass, COMFY_INPUT / input_name)
                    graph = build_seedvr2_graph(input_name, prefix + "_seedvr2",
                        resolution=project.get("upscale_resolution", 720),
                        max_resolution=project.get("upscale_max_resolution", 1280), seed=shot.get("seed", 42))
                    prompt_id = resumed_upscale_prompt
                    if not prompt_id:
                        prompt_id = _submit(graph, run_id)
                        update_run(run_id, prompt_id=prompt_id, active_stage="seedvr2",
                                   active_shot_id=shot["id"], active_prefix=prefix + "_seedvr2",
                                   current=shot["name"] + " · SeedVR2")
                    else:
                        update_run(run_id, current=shot["name"] + " · 恢复 ComfyUI SeedVR2 任务")
                    final = output_dir / f"shot_{index:03d}_seedvr2.mp4"
                    _download_video(_wait(prompt_id, run_id), final, prefix + "_seedvr2")
                    upscaled_size = _video_dimensions(final)
                    if min(upscaled_size) <= min(original_size):
                        raise RuntimeError(
                            f"SeedVR2 输出未放大：H3 原片 {original_size[0]}×{original_size[1]}，"
                            f"超分文件 {upscaled_size[0]}×{upscaled_size[1]}；已保留 H3 原片，"
                            "不把相同尺寸文件标记为超分成片。")
                    update_run(run_id, upscale_result={
                        "shot_id": shot["id"],
                        "source_width": original_size[0], "source_height": original_size[1],
                        "output_width": upscaled_size[0], "output_height": upscaled_size[1],
                        "elapsed_seconds": round(time.time() - upscale_started_at, 1),
                        "file": str(final.relative_to(OUTPUTS)),
                    })
                    update_run(run_id, active_stage="", active_shot_id="", active_prefix="")
                    if read()["runs"][run_id].get("cancel_requested"):
                        raise RuntimeError("用户取消任务")
            # 先裁到交付尺寸，再在人脸恢复：这样脸部检测和细节恢复都作用在
            # 最终分辨率上（1080p 下脸约 194px，4K 下约 388px）。
            continuity_source = final
            final = _fit_output_size(final, output_dir, index, project)
            if project.get("face_refine"):
                refined = output_dir / f"shot_{index:03d}_face.mp4"
                face_python = os.environ.get("H3_STUDIO_FACE_PYTHON", "/home/administrator/ComfyUI/venv/bin/python")
                env = dict(os.environ)
                env["H3_STUDIO_FACE_USE_ESRGAN"] = "1"
                subprocess.run([face_python, str(Path(__file__).with_name("face_refine.py")),
                                str(final), str(refined)], check=True, timeout=3600, env=env)
                refined.replace(final)
            source = COMFY_INPUT / media[shot["source_video"]] if shot.get("source_video") else None
            _audio_mode(final, source, shot.get("audio_mode", "generate"))
            previous = continuity_source
            previous_signature = signature
            plan = shot.get("_h3_trim_plan") or {}
            actual_seconds = round(video_duration(final) * 24) / 24
            results.append({"shot_id": shot["id"], "signature": signature, "file": str(final.relative_to(OUTPUTS)),
                            "h3_file": str(first_pass.relative_to(OUTPUTS)),
                            "continuity_file": str(continuity_source.relative_to(OUTPUTS)),
                            "pre_refine_file": str(pre_refine.relative_to(OUTPUTS)) if pre_refine else None,
                            "transition_mode": shot.get("transition_mode", "continue"),
                            "handoff_visible_frames": shot.get("continuity_visible_frames", 0),
                            "handoff_guide_frames": shot.get("continuity_guide_frames", 0),
                            "effective_duration": round(actual_seconds, 3),
                            "planned_duration": round(float(plan.get("output_frames") or round(
                                float(shot.get("_h3_effective_duration") or shot["duration"]) * 24)) / 24, 3),
                            "dropped_head_frames": plan.get("head_dropped_frames", 0),
                            "head_speech_preserved": bool(plan.get("head_speech_preserved")),
                            "head_peak_dbfs": plan.get("head_peak_dbfs"),
                            "narration_mode": plan.get("mode")})
            update_run(run_id, results=results)
            update_run(run_id, progress=f"{index + 1}/{len(shots)}", current=shot["name"] + " · 已完成")
            under_pressure, vram_detail = _vram_is_under_pressure()
            cleanup_action = "manual" if project.get("clear_vram_between_shots") else "automatic" if under_pressure else "none"
            queue_detail = "ComfyUI 队列状态未知"
            if project.get("clear_vram_between_shots") or under_pressure:
                try:
                    queue_idle, queue_detail = _comfy_queue_is_idle()
                    if not queue_idle:
                        cleanup_action = "skipped_queue_busy"
                        cleanup_message = f"{vram_detail}；{queue_detail}，为避免影响其它任务已跳过显存清理"
                        update_run(run_id, memory_cleanup=cleanup_message)
                    else:
                        _release_comfy_memory()
                        cleanup_action = "released"
                        reason = "手动开启镜头间清理" if project.get("clear_vram_between_shots") else "检测到低可用显存"
                        cleanup_message = f"{reason}：{vram_detail}；{queue_detail}；已请求 ComfyUI 释放模型与显存缓存"
                        update_run(run_id, memory_cleanup=cleanup_message)
                except Exception as error:
                    cleanup_message = f"{vram_detail}；{queue_detail}；显存清理请求未成功：{error}"
                    update_run(run_id, memory_cleanup=cleanup_message)
                    cleanup_action = "failed"
            else:
                cleanup_message = f"自动显存检查：{vram_detail}；无需清理"
                update_run(run_id, memory_cleanup=cleanup_message)
            def record_memory_event(data):
                data["runs"][run_id].setdefault("memory_events", []).append({
                    "shot_id": shot["id"], "shot_name": shot["name"],
                    "time": time.time(), "status": cleanup_action, "detail": vram_detail,
                    "queue": queue_detail, "message": cleanup_message,
                })
            change(record_memory_event)
        # Strong continuous-take mode always needs a single deliverable. Keep
        # per-shot files as source segments, but do not let "segments" export
        # suppress the stitched long take.
        if source_run:
            replacement = next((item for item in results
                                if item.get("shot_id") == run["reroll_shot_id"]), None)
            if not replacement:
                raise RuntimeError("重抽镜头没有生成结果")
            combined = _reroll_result_list(source_run.get("results", []),
                                           run["reroll_shot_id"], replacement)
            merge_items: list[Path] = []
            seams: list[dict] = []
            shot_by_id = {shot["id"]: shot for shot in project["shots"]}
            for item_index, item in enumerate(combined):
                current_shot = shot_by_id.get(item.get("shot_id"), {})
                previous_shot = shot_by_id.get(combined[item_index - 1].get("shot_id"), {}) if item_index else {}
                _append_merge_item(merge_items, seams, OUTPUTS / item["file"], item,
                                   item_index, output_dir, project,
                                   preserve_spoken_audio=_preserve_spoken_seam(
                                       previous_shot, current_shot,
                                       int(item.get("handoff_visible_frames") or 0)))
            if len(merge_items) > 1:
                merged = output_dir / "merged.mp4"
                _merge(merge_items, merged, int(project.get("width", 1024)),
                       int(project.get("height", 576)))
                update_run(run_id, merged=str(merged.relative_to(OUTPUTS)),
                           merge_seam_algorithm="adaptive_structural_v5",
                           merge_seam_adjustments=seams)
            update_run(run_id, results=combined,
                       reused_shot_ids=[item.get("shot_id") for item in combined
                                        if item.get("shot_id") != run["reroll_shot_id"]])
        elif project.get("export_mode") == "all" or project.get("continuous_take", True):
            selected_ids = set(run["shot_ids"])
            merge_items = []
            seam_adjustments = []
            merge_previous_shot = None
            for index, (shot, item) in enumerate(_selected_results_for_merge(
                    project["shots"], selected_ids, results)):
                _append_merge_item(merge_items, seam_adjustments, OUTPUTS / item["file"],
                                   item, index, output_dir, project,
                                   preserve_spoken_audio=_preserve_spoken_seam(
                                       merge_previous_shot or {}, shot,
                                       int(item.get("handoff_visible_frames") or 0)))
                merge_previous_shot = shot
            omitted = [shot["id"] for shot in project["shots"] if shot["id"] not in selected_ids]
            if merge_items:
                merged = output_dir / "merged.mp4"
                _merge(merge_items, merged,
                       int(project.get("width", 1024)), int(project.get("height", 576)))
                update_run(run_id, merged=str(merged.relative_to(OUTPUTS)),
                           merge_seam_algorithm="adaptive_structural_v7_continuous_audio", merge_seam_adjustments=seam_adjustments,
                           omitted_shots=omitted)
        update_run(run_id, status="completed", current="", finished_at=time.time(),
                   active_stage="", active_shot_id="", active_prefix="",
                   reused_shot_ids=reused_shot_ids)
    except Exception as error:
        status = "cancelled" if str(error) == "用户取消任务" else "error"
        update_run(run_id, status=status, error=str(error), finished_at=time.time(), results=results)

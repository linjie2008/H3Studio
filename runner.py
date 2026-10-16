"""Run independent shot plans through ComfyUI's queue API."""

from __future__ import annotations

import copy
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
from video_tools import duration as video_duration


COMFY_URL = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188").rstrip("/")
COMFY_INPUT = Path(os.environ.get("COMFYUI_INPUT_DIR", "/home/administrator/ComfyUI/input"))
NEW_IDENTITY_CONTEXT_FRAMES = 8
NEW_IDENTITY_VISIBLE_FRAMES = 4

# Serialize submissions across worker threads so a model-family switch cannot
# race another prompt between the idle check, cache release and submission.
_MODEL_SWITCH_LOCK = threading.Lock()
_ACTIVE_MODEL_FAMILY: str | None = None


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


def prepare_shot(original: dict, project: dict) -> dict:
    """Apply project refs, common prompt, and selected character and entity profiles."""
    catalog = read()["assets"]
    project = normalized_project_image_references(project, catalog)
    original = next((item for item in project.get("shots", [])
                     if item.get("id") == original.get("id")), original)
    shot = copy.deepcopy(original)
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
    # must not silently introduce unmentioned people or objects.
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
    if shot.get("dialogue_text") and len(selected_profiles) == 1:
        profile_speaker = str(selected_profiles[0].get("speaker_id") or "S1")
        if shot.get("speaker_id", "S1") == "S1" and profile_speaker != "S1":
            shot["speaker_id"] = profile_speaker
        if shot.get("speaker_description", "画面中的人物") == "画面中的人物":
            profile_name = str(selected_profiles[0].get("name") or "画面中的人物")
            profile_appearance = str(selected_profiles[0].get("appearance") or "").strip()
            shot["speaker_description"] = profile_name + ("，" + profile_appearance if profile_appearance else "")
    if shot.get("dialogue_text") and selected_profiles:
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
                                              and next_shot.get("dialogue_text"))
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
            number = int(match.group(1) or match.group(2))
            local = global_to_local.get(number, number)
            return f"<Picture {local}>"
        return re.sub(r"<Picture\s+(\d+)>|@图片\s*(\d+)", localize, text, flags=re.IGNORECASE)

    authored_shared_prompt = localize_picture_tags(str(project.get("shared_prompt") or "").strip())
    authored_shot_prompt = localize_picture_tags(str(shot.get("prompt") or "").strip())
    prompt_parts = [authored_shared_prompt, *annotations, *entity_annotations, authored_shot_prompt]
    shot["prompt"] = "\n".join(part for part in prompt_parts if part)
    return shot


def _shot_signature(shot: dict, project: dict, assets: dict, previous_signature: str | None = None) -> str:
    """Stable cache key for a shot and the settings that affect its output."""
    relevant_project = {key: project.get(key) for key in (
        "width", "height", "fps", "generation_speed", "h3_lora_dmad", "h3_lora_dmad_strength", "h3_lora_mystic", "h3_lora_mystic_strength", "attention_mode", "sage_attention", "h3_video_vae", "upscale", "upscale_resolution", "upscale_max_resolution",
        "face_refine", "semantic_adapter", "semantic_alpha", "selflift", "selflift_scale",
        "selflift_highres_steps", "selflift_rho", "selflift_tiling", "selflift_tile_count",
        "selflift_tile_overlap", "refine_mode", "refine_steps", "refine_denoise", "refine_passes",
        "refine_width", "refine_height", "continuous_take", "continuous_render_mode", "continuity_frames", "identity_first_on_switch", "shared_refs", "shared_prompt", "image_descriptions", "clear_vram_between_shots",
    )}
    if shot.get("continuity"):
        # Invalidate cached generations after changing the scene-adaptive
        # handoff rules, without invalidating unrelated projects.
        relevant_project["continuity_algorithm"] = "h3_native_scene_adaptive_continuity_v18"
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


def _continuity_subject_relation(previous_refs, current_refs) -> str:
    previous, current = set(previous_refs or []), set(current_refs or [])
    if previous and current and previous.isdisjoint(current):
        return "new"
    return "same"


def _shot_identity_refs(shot: dict, project: dict, assets: dict) -> list[str]:
    """Separate person references from shared scenery before a continuity handoff."""
    characters = {item.get("id"): item for item in project.get("characters", [])}
    profile_refs = []
    for character_id in shot.get("character_ids") or []:
        profile = characters.get(character_id) or {}
        profile_refs.extend(filter(None, (profile.get("portrait_asset"), profile.get("multiview_asset"))))
    descriptions = project.get("image_descriptions") or {}
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
        if asset_id:
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


def _copy_asset(asset_id: str, assets: dict, *, trim_start: float | None = None, trim_duration: float | None = None) -> str:
    asset = assets.get(asset_id)
    if not asset:
        raise ValueError(f"素材不存在：{asset_id}")
    source = ASSETS / asset["file"]
    if not source.is_file():
        raise FileNotFoundError(source)
    COMFY_INPUT.mkdir(parents=True, exist_ok=True)
    name = "h3studio_" + asset_id + source.suffix.lower()
    target = COMFY_INPUT / name
    if trim_start is not None and asset["kind"] == "video":
        name = "h3studio_" + asset_id + "_" + str(int(trim_start * 1000)) + "_" + str(int((trim_duration or 0) * 1000)) + ".mp4"
        target = COMFY_INPUT / name
        if not target.exists():
            command = ["ffmpeg", "-y", "-loglevel", "error", "-ss", str(trim_start), "-i", str(source)]
            if trim_duration is not None:
                command += ["-t", str(trim_duration)]
            command += ["-vf", "fps=24", "-c:v", "libx264", "-preset", "fast", "-pix_fmt", "yuv420p", "-c:a", "aac", str(target)]
            subprocess.run(command, check=True, timeout=300)
    elif not target.exists() or target.stat().st_size != source.stat().st_size:
        shutil.copy2(source, target)
    return name


def _source_passthrough(shot: dict, assets: dict, destination: Path) -> None:
    """Prepare an unselected video-edit shot from its trimmed source clip."""
    asset = assets.get(shot.get("source_video"))
    if not asset:
        raise ValueError(f"源视频素材不存在：{shot.get('source_video')}")
    source = ASSETS / asset["file"]
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", str(float(shot.get("source_start") or 0)),
                    "-i", str(source), "-t", str(float(shot.get("duration") or 5)),
                    "-map", "0:v:0", "-map", "0:a:0?", "-vf", "fps=24",
                    "-c:v", "libx264", "-preset", "fast", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-shortest", str(destination)], check=True, timeout=600)


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
    media = {asset_id: _copy_asset(asset_id, assets) for asset_id in ids}
    if shot.get("source_video"):
        media[shot["source_video"]] = _copy_asset(
            shot["source_video"], assets,
            trim_start=float(shot.get("source_start") or 0),
            trim_duration=float(shot.get("duration") or 5),
        )
    return media


def _submit(graph: dict, run_id: str | None = None) -> str:
    """Submit a graph, releasing old model caches at model-family boundaries."""
    global _ACTIVE_MODEL_FAMILY
    family = _graph_model_family(graph)
    with _MODEL_SWITCH_LOCK:
        previous = _ACTIVE_MODEL_FAMILY
        if previous != family:
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
        response = api("/prompt", {"prompt": graph, "client_id": "h3_studio"})
        if response.get("node_errors"):
            raise RuntimeError(json.dumps(response["node_errors"], ensure_ascii=False))
        _ACTIVE_MODEL_FAMILY = family
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
            update_run(run_id, current=f"ComfyUI 状态暂时不可读，正在重试（{connection_failures}/10）")
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


def _tail_clip(video: Path, target: Path, frames: int) -> None:
    seconds = min(frames / 24, video_duration(video))
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-sseof", str(-seconds), "-i", str(video),
                    "-vf", "fps=24,tpad=stop_mode=clone:stop_duration=2", "-frames:v", str(frames),
                    "-c:v", "libx264", "-preset", "fast", "-pix_fmt", "yuv420p", "-c:a", "aac",
                    str(target)], check=True, timeout=120)


def _trim_prefix(video: Path, seconds: float, duration: float, visible_context_frames: int = 0) -> None:
    target = video.with_name(video.stem + "_trim.mp4")
    # Keep a short part of the guided prefix as the visible wipe. The merge
    # path removes the same number of frames from the previous shot's tail.
    start_frame = max(0, round(float(seconds) * 24) - max(0, int(visible_context_frames)))
    output_frames = max(1, round(float(duration) * 24) + max(0, int(visible_context_frames)))
    audio_start = start_frame / 24
    audio_duration = output_frames / 24
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(video),
                    "-vf", f"fps=24,trim=start_frame={start_frame}:end_frame={start_frame + output_frames},setpts=PTS-STARTPTS",
                    "-af", f"atrim=start={audio_start:.6f}:duration={audio_duration:.6f},asetpts=PTS-STARTPTS",
                    "-frames:v", str(output_frames), "-c:v", "libx264", "-preset", "fast",
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


def _handoff_gray_frames(video: Path) -> list[bytes]:
    """Decode small luma frames for an inexpensive seam alignment check."""
    result = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(video), "-an", "-vf", "fps=24,scale=64:64",
         "-pix_fmt", "gray", "-f", "rawvideo", "pipe:1"],
        capture_output=True, check=True, timeout=120)
    pixels = 64 * 64
    if not result.stdout or len(result.stdout) % pixels:
        raise ValueError("无法读取衔接画面")
    return [result.stdout[i:i + pixels] for i in range(0, len(result.stdout), pixels)]


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
        if not (max(1.5, previous_drift * 1.5, next_drift * 1.5) <= abs(delta)
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
        if len(matched) != len(after) or adjusted >= seam * 0.85:
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


def _merge(videos: list[Path], target: Path, width: int, height: int) -> None:
    normalized = []
    for index, path in enumerate(videos):
        info = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(path)],
                              capture_output=True, text=True, check=True, timeout=30)
        has_audio = any(stream.get("codec_type") == "audio" for stream in json.loads(info.stdout).get("streams", []))
        output = target.with_name(f"merge_norm_{index:03d}.mp4")
        command = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(path)]
        if not has_audio:
            command += ["-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000"]
        command += ["-map", "0:v:0", "-map", "0:a:0" if has_audio else "1:a:0",
                    "-vf", f"scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,fps=24",
                    "-af", "apad", "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                    "-pix_fmt", "yuv420p", "-c:a", "aac", "-ar", "48000", "-ac", "2",
                    "-shortest", str(output)]
        subprocess.run(command, check=True, timeout=600)
        normalized.append(output)
    manifest = target.with_suffix(".txt")
    manifest.write_text("".join("file '" + str(path).replace("'", "'\\''") + "'\n" for path in normalized))
    try:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
                        "-i", str(manifest), "-c", "copy", str(target)], check=True, timeout=600)
    finally:
        manifest.unlink(missing_ok=True)
        for path in normalized:
            path.unlink(missing_ok=True)


def _append_merge_item(merge_items: list[Path], seam_adjustments: list[dict], path: Path,
                       item: dict, item_index: int, output_dir: Path, project: dict,
                       file_prefix: str = "merge") -> None:
    visible_frames = max(0, int(item.get("handoff_visible_frames", 0) or 0))
    if visible_frames and merge_items:
        mode = project.get("seam_mode", "auto")
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
    for index, item in enumerate(items):
        _append_merge_item(paths, seams, OUTPUTS / item["file"], item, index,
                           output_dir, project, f"remerge_{stamp}")
    merged = output_dir / f"merged_remerge_{stamp}.mp4"
    _merge(paths, merged, int(project.get("width", 1024)), int(project.get("height", 576)))
    update_run(run_id, merged=str(merged.relative_to(OUTPUTS)),
               merge_seam_algorithm="adaptive_structural_v5", merge_seam_adjustments=seams,
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
            or project.get("refine_mode", "off") != "off" or project.get("upscale")
            or project.get("face_refine") or project.get("selflift")):
        return None
    if any(shot.get("mode") != "r2v" or shot.get("source_video")
           or shot.get("first_image") or shot.get("last_image")
           or shot.get("dialogue_text") for shot in shots):
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


def run_project(run_id: str) -> None:
    """Process selected shots sequentially; keep results even if a later shot fails."""
    state = read()
    run = state["runs"][run_id]
    assets = state["assets"]
    project = normalized_project_image_references(state["projects"][run["project_id"]], assets)
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
        for index, original in enumerate(shots):
            if read()["runs"][run_id].get("cancel_requested"):
                raise RuntimeError("用户取消任务")
            shot = prepare_shot(original, project)
            continuity = _continuity_policy(project, original, index)
            if index > 0:
                # Selected character references identify who continues across
                # the segment boundary; shared references identify the same cast.
                previous_identities = _shot_identity_refs(shots[index - 1], project, assets)
                current_identities = _shot_identity_refs(original, project, assets)
                shot["continuity_subject_relation"] = _continuity_subject_relation(
                    previous_identities, current_identities)
                if shot["continuity_subject_relation"] == "new":
                    _strengthen_incoming_identity_reference(shot, current_identities)
                    if project.get("identity_first_on_switch"):
                        # A hard guide of the outgoing person's final frames can
                        # override the incoming face and uniform. For projects
                        # that prioritize distinct people, keep the shared scene
                        # and prompt but do not feed the old person as a guide.
                        continuity = {"enabled": False, "frames": 0, "strength": "none"}
                    elif continuity.get("enabled") and continuity.get("strength") == "strong":
                        continuity["frames"] = NEW_IDENTITY_CONTEXT_FRAMES
            shot["continuous_take"] = bool(project.get("continuous_take", True))
            shot["continuity"] = continuity["enabled"]
            shot["continuity_frames"] = continuity["frames"]
            shot["continuity_strength"] = continuity["strength"]
            shot["continuity_seam_redraw"] = continuity.get("seam_redraw", 0.5)
            shot["continuity_frames"], shot["continuity_visible_frames"] = _handoff_window(
                shot.get("continuity_subject_relation", "same") if continuity.get("enabled") else "none",
                shot["continuity_frames"])
            if shot["continuity_frames"]:
                shot["continuity_guide_frames"] = _continuity_anchor_frames(shot["continuity_frames"])
            render_settings = project
            if project.get("memory_policy") == "auto_vram":
                memory_plan = _auto_memory_plan(int(project.get("width", 1024)), int(project.get("height", 576)))
                render_settings = copy.deepcopy(project)
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
            if cached and cached.get("signature") == signature and (OUTPUTS / cached.get("file", "")).is_file():
                results.append(cached)
                previous = OUTPUTS / cached["file"]
                previous_signature = signature
                update_run(run_id, results=results, progress=f"{index + 1}/{len(shots)}", current=shot["name"] + " · 已复用缓存")
                continue
            update_run(run_id, current=shot["name"] + " · 准备生成", progress=f"{index}/{len(shots)}")
            media = _make_media_map(shot, project, assets)
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
            if checkpoint.get("signature") != signature or not first_pass.is_file() or (pre_refine and not pre_refine.is_file()):
                prompt_id = _resume_prompt(run_id, "h3", shot["id"])
                if not prompt_id:
                    graph = build_h3_graph(shot, render_settings, media, prefix)
                    if read()["runs"][run_id].get("cancel_requested"):
                        raise RuntimeError("用户取消任务")
                    prompt_id = _submit(graph, run_id)
                    update_run(run_id, prompt_id=prompt_id, active_stage="h3",
                               active_shot_id=shot["id"], active_prefix=prefix,
                               current=shot["name"] + " · H3 视频生成中")
                else:
                    update_run(run_id, current=shot["name"] + " · 恢复 ComfyUI H3 任务")
                first_pass = output_dir / f"shot_{index:03d}_h3.mp4"
                history = _wait(prompt_id, run_id)
                _download_video(history, first_pass, prefix)
                pre_refine = None
                if project.get("refine_mode") in ("same", "latent", "latent_only"):
                    pre_refine = output_dir / f"shot_{index:03d}_pre_refine.mp4"
                    _download_video(history, pre_refine, prefix + "_pre")
                if read()["runs"][run_id].get("cancel_requested"):
                    raise RuntimeError("用户取消任务")
                if shot.get("continuity_video"):
                    _trim_prefix(first_pass, shot["continuity_frames"] / 24, float(shot["duration"]),
                                 shot.get("continuity_visible_frames", 0))
                    if pre_refine:
                        _trim_prefix(pre_refine, shot["continuity_frames"] / 24, float(shot["duration"]),
                                     shot.get("continuity_visible_frames", 0))
                if project.get("face_refine"):
                    refined = output_dir / f"shot_{index:03d}_face.mp4"
                    face_python = os.environ.get("H3_STUDIO_FACE_PYTHON", "/home/administrator/ComfyUI/venv/bin/python")
                    subprocess.run([face_python, str(Path(__file__).with_name("face_refine.py")),
                                    str(first_pass), str(refined)], check=True, timeout=3600)
                    refined.replace(first_pass)
                checkpoint = {"signature": signature, "h3_file": str(first_pass.relative_to(OUTPUTS)),
                              "pre_refine_file": str(pre_refine.relative_to(OUTPUTS)) if pre_refine else None}
                def save_checkpoint(data):
                    data["runs"][run_id].setdefault("checkpoints", {})[shot["id"]] = checkpoint
                    data["runs"][run_id].update(active_stage="", active_shot_id="", active_prefix="")
                change(save_checkpoint)
            update_run(run_id, preview_file=str(first_pass.relative_to(OUTPUTS)),
                       pre_refine_file=str(pre_refine.relative_to(OUTPUTS)) if pre_refine else None)
            final = first_pass
            perform_upscale = bool(project.get("upscale"))
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
                    _download_video(_wait(prompt_id, run_id), final)
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
            source = COMFY_INPUT / media[shot["source_video"]] if shot.get("source_video") else None
            _audio_mode(final, source, shot.get("audio_mode", "generate"))
            previous = final
            previous_signature = signature
            results.append({"shot_id": shot["id"], "signature": signature, "file": str(final.relative_to(OUTPUTS)),
                            "h3_file": str(first_pass.relative_to(OUTPUTS)),
                            "pre_refine_file": str(pre_refine.relative_to(OUTPUTS)) if pre_refine else None,
                            "handoff_visible_frames": shot.get("continuity_visible_frames", 0),
                            "handoff_guide_frames": shot.get("continuity_guide_frames", 0)})
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
        if project.get("export_mode") == "all" or project.get("continuous_take", True):
            selected_ids = set(run["shot_ids"])
            current_by_shot = {item["shot_id"]: item for item in results}
            merge_items = []
            seam_adjustments = []
            passthrough = []
            omitted = []
            merge_previous_signature = None

            for index, shot in enumerate(project["shots"]):
                item = current_by_shot.get(shot["id"])
                if item:
                    _append_merge_item(merge_items, seam_adjustments, OUTPUTS / item["file"],
                                       item, index, output_dir, project)
                    merge_previous_signature = item.get("signature")
                    continue
                signature = _shot_signature(shot, project, assets, merge_previous_signature)
                candidate = saved_results.get(shot["id"])
                if (shot["id"] not in selected_ids and candidate and candidate.get("signature") == signature
                        and (OUTPUTS / candidate.get("file", "")).is_file()):
                    _append_merge_item(merge_items, seam_adjustments, OUTPUTS / candidate["file"],
                                       candidate, index, output_dir, project)
                    merge_previous_signature = signature
                    continue
                if shot["id"] not in selected_ids and shot.get("source_video"):
                    passthrough_file = output_dir / f"passthrough_{index:03d}.mp4"
                    try:
                        _source_passthrough(shot, assets, passthrough_file)
                        merge_items.append(passthrough_file)
                        passthrough.append(shot["id"])
                        merge_previous_signature = signature
                        continue
                    except Exception:
                        passthrough_file.unlink(missing_ok=True)
                omitted.append(shot["id"])
                merge_previous_signature = None
            if merge_items:
                merged = output_dir / "merged.mp4"
                _merge(merge_items, merged,
                       int(project.get("width", 1024)), int(project.get("height", 576)))
                update_run(run_id, merged=str(merged.relative_to(OUTPUTS)),
                           merge_seam_algorithm="adaptive_structural_v5", merge_seam_adjustments=seam_adjustments,
                           source_passthrough_shots=passthrough, omitted_shots=omitted)
            elif passthrough or omitted:
                update_run(run_id, source_passthrough_shots=passthrough, omitted_shots=omitted)
        update_run(run_id, status="completed", current="", finished_at=time.time(),
                   active_stage="", active_shot_id="", active_prefix="")
    except Exception as error:
        status = "cancelled" if str(error) == "用户取消任务" else "error"
        update_run(run_id, status=status, error=str(error), finished_at=time.time(), results=results)

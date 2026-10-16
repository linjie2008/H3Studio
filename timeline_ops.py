"""Validated, pure timeline edit operations used by the browser workbench."""

from __future__ import annotations

import copy
import math


def edit_timeline(shots: list[dict], shot_ids: list[str], action: str, value=None) -> list[dict]:
    if not isinstance(shots, list) or any(not isinstance(shot, dict) for shot in shots):
        raise ValueError("镜头列表格式无效")
    if (not isinstance(shot_ids, list)
            or any(not isinstance(item, str) for item in shot_ids)
            or len(set(shot_ids)) != len(shot_ids)):
        raise ValueError("编辑选择无效或重复")
    selected = set(shot_ids)
    if not selected or any(not item for item in selected):
        raise ValueError("请先勾选要编辑的镜头")
    indexed = [index for index, shot in enumerate(shots) if shot.get("id") in selected]
    if len(indexed) != len(selected):
        raise ValueError("编辑选择中包含不存在的镜头")
    if action == "same_duration":
        try:
            duration = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError("镜头时长必须是数字") from error
        if not math.isfinite(duration) or not 2 <= duration <= 15:
            raise ValueError("统一时长需在 2 到 15 秒之间")
        if len(indexed) < 2:
            raise ValueError("至少勾选两个镜头才能统一时长")
        result = copy.deepcopy(shots)
        for index in indexed:
            result[index]["duration"] = duration
        return result
    if action != "merge":
        raise ValueError("未知时间轴编辑操作")
    if len(indexed) < 2:
        raise ValueError("至少勾选两个镜头才能合并")
    if indexed != list(range(indexed[0], indexed[-1] + 1)):
        raise ValueError("只能合并时间轴上连续的镜头")

    group = [shots[index] for index in indexed]
    total_duration = sum(float(shot.get("duration") or 0) for shot in group)
    if not 2 <= total_duration <= 15:
        raise ValueError("合并后时长必须在 2 到 15 秒之间；请先调整时长")
    same_fields = (
        "mode", "seed", "selected", "first_image", "last_image", "source_video", "source_start",
        "ref_images", "ref_videos", "ref_audios", "character_ids", "entity_ids", "scene_id",
        "audio_mode", "ref_image_size", "emotion", "camera_move", "camera_speed", "action_intensity",
        "speaker_id", "speaker_description", "dialogue_text", "dialogue_language", "dialogue_delivery",
        "dialogue_time", "dialogue_voiceover", "dialogue_cutoff", "dialogue_carries_out",
        "on_screen_text", "overall_soundscape", "non_diegetic_music", "ref_image_roles",
        "ref_video_roles", "ref_audio_roles",
    )
    first = group[0]
    for field in same_fields:
        if any(shot.get(field) != first.get(field) for shot in group[1:]):
            raise ValueError(f"镜头的“{field}”设置不一致；为避免丢失素材或声音信息，不能合并")
    timed_prompts = []
    cursor = 0.0
    for shot in group:
        duration = float(shot.get("duration") or 0)
        prompt = str(shot.get("prompt") or "").strip()
        if not prompt:
            raise ValueError("待合并镜头都需要填写提示词")
        timed_prompts.append(f"At {cursor:g}-{cursor + duration:g}s: {prompt}")
        cursor += duration
    merged = copy.deepcopy(first)
    merged["name"] = "合并镜头"
    merged["duration"] = total_duration
    merged["prompt"] = "\n".join(timed_prompts)
    result = copy.deepcopy(shots)
    result[indexed[0]:indexed[-1] + 1] = [merged]
    return result

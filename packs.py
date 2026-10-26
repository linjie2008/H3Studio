"""Portable H3 Studio project bundles with media assets."""

from __future__ import annotations

import copy
import json
import shutil
import zipfile
from pathlib import Path

from engine import h3_frames
from store import ASSETS, ROOT, change, new_project, new_shot, read, uid
from store import project_asset_ids as store_project_asset_ids

PACKS = ROOT / "packs"
PACKS.mkdir(exist_ok=True)


def project_asset_ids(project: dict) -> set[str]:
    """Every asset the project refers to.

    This must stay the same set the studio collects, otherwise a manual
    reference image or an official time anchor is left out of the pack and
    the imported project points at ids that do not exist.
    """
    return store_project_asset_ids(project)


def export_pack(project_id: str) -> Path:
    data = read()
    project = data["projects"][project_id]
    out = PACKS / (project_id + ".h3pack.zip")
    temp = out.with_name(out.name + ".tmp")
    missing = []
    with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        archive.writestr("project.json", json.dumps({"format": "h3-studio-1", "project": project}, ensure_ascii=False))
        for asset_id in sorted(project_asset_ids(project)):
            asset = data["assets"].get(asset_id)
            # A reference to a deleted asset must not abort the export and
            # leave a half-written pack behind.
            if not asset or not (ASSETS / asset["file"]).is_file():
                missing.append(asset_id)
                continue
            archive.write(ASSETS / asset["file"], "assets/" + asset["file"])
            archive.writestr("asset_meta/" + asset_id + ".json", json.dumps(asset, ensure_ascii=False))
    temp.replace(out)
    if missing:
        raise ValueError("项目引用了 " + str(len(missing)) + " 个已不存在的素材，无法完整导出：" + missing[0])
    return out


def export_director_pack(project_id: str) -> Path:
    """Write a Director-compatible timeline pack for interchange."""
    data = read()
    project = data["projects"][project_id]
    out = PACKS / (project_id + ".mmxpack.zip")
    shots = project.get("shots", [])
    task = shots[0].get("mode", "t2v") if shots else "t2v"
    timeline = {
        "version": 5, "timelineMode": "fl2v" if task == "fl2v" else "video" if task in ("v2v", "rv2v") else "prompt_batch",
        "frameRate": 24, "width": project["width"], "height": project["height"],
        "output": {"width": project["width"], "height": project["height"],
                   "exportMode": project.get("export_mode", "all"),
                   "continuityEnabled": True,
                   "continuityOverlapFrames": project.get("continuity_frames", 22)},
        "global": {"taskType": task, "prompt": project.get("shared_prompt", ""), "refs": [], "refVideos": [], "refAudios": []},
        "segments": [], "shots": [], "video": {},
    }
    meta = {"format": "minimax-h3-director-pack", "formatVersion": 1,
            "taskType": task, "output": timeline["output"]}
    meta["h3StudioMemoryPolicy"] = project.get("memory_policy", "manual")
    shot_data_fields = ("speaker_id", "speaker_description", "dialogue_text", "dialogue_language",
                        "dialogue_delivery", "dialogue_time", "dialogue_voiceover", "dialogue_cutoff",
                        "dialogue_carries_out", "on_screen_text", "overall_soundscape", "non_diegetic_music",
                        "ref_image_roles", "ref_video_roles", "ref_audio_roles", "emotion", "camera_move",
                        "camera_speed", "action_intensity")
    meta["h3StudioShotData"] = [{"id": shot.get("id"), **{key: shot.get(key) for key in shot_data_fields if key in shot}}
                                for shot in shots]
    cursor = 0
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        def put(asset_id, kind, folder, stem):
            if not asset_id:
                return None
            asset = data["assets"][asset_id]
            suffix = Path(asset["file"]).suffix
            rel = f"{folder}/{stem}{suffix}"
            archive.write(ASSETS / asset["file"], rel)
            return {"image": {"imageFile": rel}, "video": {"videoFile": rel},
                    "audio": {"audioFile": rel}}[kind]

        shared = project.get("shared_refs", {})
        for kind, key, stem in (("image", "images", "Picture"), ("video", "videos", "Video"), ("audio", "audios", "Audio")):
            field = {"image": "refs", "video": "refVideos", "audio": "refAudios"}[kind]
            for index, asset_id in enumerate(shared.get(key, []), 1):
                timeline["global"][field].append(put(asset_id, kind, "shared_params", f"{stem}{index}"))
        for index, shot in enumerate(shots, 1):
            folder = f"asset_groups/{index:02d}"
            frames = h3_frames(float(shot.get("duration", 5)))
            card = {"id": shot["id"], "start": cursor, "length": frames, "frameCount": frames,
                    "durationSec": shot.get("duration", 5), "taskType": shot.get("mode", task),
                    "prompt": shot.get("prompt", ""), "continuityFromPrev": shot.get("continuity", False),
                    "h3StudioCharacterIds": list(shot.get("character_ids", [])),
                    "h3StudioEntityIds": list(shot.get("entity_ids", [])),
                    "h3StudioSceneId": shot.get("scene_id", ""),
                    "refs": [], "refVideos": [], "refAudios": []}
            cursor += frames
            card["startImage"] = put(shot.get("first_image"), "image", folder, "start")
            card["endImage"] = put(shot.get("last_image"), "image", folder, "end")
            card["referenceVideo"] = put(shot.get("source_video"), "video", folder, "source")
            for kind, key, stem in (("image", "ref_images", "Picture"), ("video", "ref_videos", "Video"), ("audio", "ref_audios", "Audio")):
                field = {"image": "refs", "video": "refVideos", "audio": "refAudios"}[kind]
                offset = len(shared.get({"image": "images", "video": "videos", "audio": "audios"}[kind], []))
                for slot, asset_id in enumerate(shot.get(key, []), offset + 1):
                    card[field].append(put(asset_id, kind, folder, f"{stem}{slot}"))
            timeline["segments"].append(card)
            timeline["shots"].append(card)
            archive.writestr(folder + "/group.json", json.dumps(card, ensure_ascii=False))
        profiles = []
        for index, character in enumerate(project.get("characters", []), 1):
            folder = f"characters/{index:02d}"
            portrait = put(character.get("portrait_asset"), "image", folder, "portrait")
            multiview = put(character.get("multiview_asset"), "image", folder, "multiview")
            voice = put(character.get("voice_asset"), "audio", folder, "voice")
            profiles.append({"id": character.get("id"), "subject_number": character.get("subject_number", index),
                             "name": character.get("name", ""),
                             "appearance": character.get("appearance", ""),
                             "voice_description": character.get("voice_description", ""),
                             "speaker_id": character.get("speaker_id", "S1"),
                             "portrait": portrait.get("imageFile") if portrait else None,
                             "multiview": multiview.get("imageFile") if multiview else None,
                             "voice": voice.get("audioFile") if voice else None})
        meta["h3StudioCharacters"] = profiles
        entities = []
        for index, entity in enumerate(project.get("entities", []), 1):
            refs = [put(asset_id, "image", f"entities/{index:02d}", f"reference{slot:02d}")
                    for slot, asset_id in enumerate(entity.get("image_assets", []), 1)]
            source = put(entity.get("generation_source"), "image", f"entities/{index:02d}", "generation_source")
            entities.append({"id": entity.get("id"), "kind": entity.get("kind", "prop"),
                             "name": entity.get("name", ""), "description": entity.get("description", ""),
                             "images": [ref["imageFile"] for ref in refs if ref],
                             "generation_mode": entity.get("generation_mode", "t2i"),
                             "generation_prompt": entity.get("generation_prompt", ""),
                             "generation_source": source.get("imageFile") if source else None})
        meta["h3StudioEntities"] = entities
        meta["h3StudioScenes"] = list(project.get("scenes", []))
        timeline["totalFrames"] = cursor
        archive.writestr("pack.json", json.dumps(meta, ensure_ascii=False))
        archive.writestr("timeline.json", json.dumps(timeline, ensure_ascii=False))
        archive.writestr("shared_params/shared_params.json", json.dumps(timeline["global"], ensure_ascii=False))
    return out


def import_pack(source: Path) -> dict:
    with zipfile.ZipFile(source) as archive:
        items = archive.infolist()
        if len(items) > 500 or sum(item.file_size for item in items) > 4_000_000_000:
            raise ValueError("项目包过大")
        if "project.json" not in archive.namelist() and "timeline.json" in archive.namelist():
            return _import_director_pack(archive)
        payload = json.loads(archive.read("project.json"))
        if not isinstance(payload, dict) or payload.get("format") != "h3-studio-1":
            raise ValueError("不是 H3 Studio 项目包")
        if not isinstance(payload.get("project"), dict):
            raise ValueError("项目包内容损坏：缺少项目数据")
        project = copy.deepcopy(payload["project"])
        for key, value in new_project(project.get("name", "导入项目")).items():
            project.setdefault(key, copy.deepcopy(value))
        old_ids = project_asset_ids(project)
        new_assets = {}
        mapping = {}
        written = []
        try:
            for old_id in old_ids:
                meta = json.loads(archive.read("asset_meta/" + old_id + ".json"))
                file_name = meta["file"]
                if "/" in file_name or "\\" in file_name or file_name in (".", ".."):
                    raise ValueError("项目包包含非法素材路径")
                new_id = uid()
                suffix = "." + file_name.rsplit(".", 1)[-1]
                new_name = new_id + suffix
                with archive.open("assets/" + file_name) as src, (ASSETS / new_name).open("wb") as dst:
                    shutil.copyfileobj(src, dst, 1024 * 1024)
                written.append(ASSETS / new_name)
                new_assets[new_id] = {**meta, "id": new_id, "file": new_name}
                mapping[old_id] = new_id
        except Exception:
            # A half-imported pack must not leave orphan files behind.
            for path in written:
                path.unlink(missing_ok=True)
            raise
    project["id"] = uid()
    project["name"] = project.get("name", "导入项目") + "（导入）"
    for asset in new_assets.values():
        asset["project_id"] = project["id"]
    for key, values in project.get("shared_refs", {}).items():
        project["shared_refs"][key] = [mapping[item] for item in values if item in mapping]
    for shot in project.get("shots", []):
        shot["id"] = uid()
        for key in ("first_image", "last_image", "source_video"):
            if shot.get(key):
                shot[key] = mapping.get(shot[key], "")
        for key in ("ref_images", "ref_videos", "ref_audios", "manual_ref_images"):
            shot[key] = [mapping[item] for item in shot.get(key, []) if item in mapping]
        for guide in shot.get("h3_guides") or []:
            if isinstance(guide, dict) and guide.get("image_id"):
                guide["image_id"] = mapping.get(guide["image_id"], "")
        shot.setdefault("character_ids", [])
    for character in project.get("characters", []):
        for key in ("portrait_asset", "multiview_asset", "voice_asset"):
            if character.get(key):
                character[key] = mapping.get(character[key], "")
    for entity in project.get("entities", []):
        entity["image_assets"] = [mapping[item] for item in entity.get("image_assets", []) if item in mapping]
        if entity.get("generation_source"):
            entity["generation_source"] = mapping.get(entity["generation_source"], "")
    descriptions = project.get("image_descriptions") or {}
    project["image_descriptions"] = {
        mapping[asset_id]: text for asset_id, text in descriptions.items() if asset_id in mapping
    }
    def save(data):
        data["projects"][project["id"]] = project
        data["assets"].update(new_assets)
    change(save)
    return project


def _import_director_pack(archive: zipfile.ZipFile) -> dict:
    """Convert the Director plugin's portable timeline and assets into our model."""
    timeline = json.loads(archive.read("timeline.json"))
    meta = json.loads(archive.read("pack.json")) if "pack.json" in archive.namelist() else {}
    global_block = timeline.get("global") or {}
    output = timeline.get("output") or {}
    project = new_project("导入导演包")
    project["width"] = int(output.get("width") or timeline.get("width") or 1344)
    project["height"] = int(output.get("height") or timeline.get("height") or 768)
    project["export_mode"] = "segments" if output.get("exportMode") == "segments" else "all"
    project["continuity_frames"] = int(output.get("continuityOverlapFrames") or 22)
    if meta.get("h3StudioMemoryPolicy") in ("manual", "auto_vram"):
        project["memory_policy"] = meta["h3StudioMemoryPolicy"]
    project["shots"] = []
    new_assets = {}
    path_to_id = {}
    names = set(archive.namelist())

    def media_id(ref, kind: str):
        if isinstance(ref, str):
            old_path = ref
        elif isinstance(ref, dict):
            keys = {"image": ("imageFile", "image_file", "fileName"),
                    "video": ("videoFile", "video_file", "fileName"),
                    "audio": ("audioFile", "audio_file", "fileName")}[kind]
            old_path = next((ref[key] for key in keys if ref.get(key)), "")
        else:
            return None
        old_path = str(old_path).replace("\\", "/").strip("/")
        if not old_path or old_path not in names or ".." in Path(old_path).parts:
            return None
        if old_path in path_to_id:
            return path_to_id[old_path]
        extension = Path(old_path).suffix.lower()
        if extension not in (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff",
                             ".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v", ".wav", ".mp3",
                             ".flac", ".ogg", ".aac", ".wma"):
            return None
        asset_id = uid()
        file_name = asset_id + extension
        with archive.open(old_path) as src, (ASSETS / file_name).open("wb") as dst:
            shutil.copyfileobj(src, dst, 1024 * 1024)
        new_assets[asset_id] = {"id": asset_id, "name": Path(old_path).name, "file": file_name,
                                "kind": kind, "size": (ASSETS / file_name).stat().st_size, "created_at": 0}
        path_to_id[old_path] = asset_id
        return asset_id

    def media_list(values, kind):
        return [item for ref in (values or []) if (item := media_id(ref, kind))]

    project["shared_refs"] = {
        "images": media_list(global_block.get("refs"), "image"),
        "videos": media_list(global_block.get("refVideos"), "video"),
        "audios": media_list(global_block.get("refAudios"), "audio"),
    }
    project["shared_prompt"] = str(global_block.get("prompt") or "")
    project["characters"] = []
    character_mapping = {}
    used_subject_numbers = set()
    next_subject_number = 1
    for index, profile in enumerate(meta.get("h3StudioCharacters", []), 1):
        new_character_id = uid()
        if profile.get("id"):
            character_mapping[profile["id"]] = new_character_id
        try:
            subject_number = int(profile.get("subject_number") or index)
        except (ValueError, TypeError):
            subject_number = index
        if subject_number < 1 or subject_number in used_subject_numbers:
            subject_number = max(next_subject_number, max(used_subject_numbers, default=0) + 1)
        used_subject_numbers.add(subject_number)
        next_subject_number = max(next_subject_number, subject_number + 1)
        project["characters"].append({
            "id": new_character_id, "name": str(profile.get("name") or "未命名角色"),
            "subject_number": subject_number,
            "appearance": str(profile.get("appearance") or ""),
            "voice_description": str(profile.get("voice_description") or ""),
            "speaker_id": str(profile.get("speaker_id") or "S1"),
            "portrait_asset": media_id(profile.get("portrait"), "image"),
            "multiview_asset": media_id(profile.get("multiview"), "image"),
            "voice_asset": media_id(profile.get("voice"), "audio"),
        })
    project["next_subject_number"] = next_subject_number
    entity_mapping = {}
    for entity in meta.get("h3StudioEntities", []):
        entity_id = uid()
        if entity.get("id"):
            entity_mapping[entity["id"]] = entity_id
        project["entities"].append({
            "id": entity_id, "kind": entity.get("kind", "prop"),
            "name": str(entity.get("name") or "未命名设定"),
            "description": str(entity.get("description") or ""),
            "image_assets": media_list(entity.get("images"), "image"),
            "generation_mode": str(entity.get("generation_mode") or "t2i"),
            "generation_prompt": str(entity.get("generation_prompt") or ""),
            "generation_source": media_id(entity.get("generation_source"), "image"),
        })
    scene_mapping = {}
    for scene in meta.get("h3StudioScenes", []):
        scene_id = uid()
        if scene.get("id"):
            scene_mapping[scene["id"]] = scene_id
        project["scenes"].append({"id": scene_id, "name": str(scene.get("name") or "未命名场景")})
    task = str(meta.get("taskType") or global_block.get("taskType") or "t2v").split(" ")[0].split("—")[0]
    cards = timeline.get("shots") if task == "fl2v" and timeline.get("shots") else timeline.get("segments") or []
    clips = timeline.get("videoClips") or []
    global_source = media_id(timeline.get("video") or (clips[0] if clips else None) or global_block.get("referenceVideo"), "video")
    studio_shot_data = {str(item.get("id")): item for item in meta.get("h3StudioShotData", []) if isinstance(item, dict)}
    shot_data_fields = ("speaker_id", "speaker_description", "dialogue_text", "dialogue_language",
                        "dialogue_delivery", "dialogue_time", "dialogue_voiceover", "dialogue_cutoff",
                        "dialogue_carries_out", "on_screen_text", "overall_soundscape", "non_diegetic_music",
                        "ref_image_roles", "ref_video_roles", "ref_audio_roles", "emotion", "camera_move",
                        "camera_speed", "action_intensity")
    for index, card in enumerate(cards):
        mode = str(card.get("taskType") or task).split(" ")[0].split("—")[0]
        if mode not in ("t2v", "i2v", "fl2v", "r2v", "v2v", "rv2v"):
            mode = task if task in ("t2v", "i2v", "fl2v", "r2v", "v2v", "rv2v") else "t2v"
        shot = new_shot()
        shot["name"] = f"镜头 {index + 1}"
        shot["mode"] = mode
        shot["prompt"] = str(card.get("prompt") or "")
        shot["duration"] = float(card.get("durationSec") or (int(card.get("frameCount") or 124) / 24))
        shot["continuity"] = bool(card.get("continuityFromPrev"))
        shot["first_image"] = media_id(card.get("startImage") or card.get("genImage"), "image")
        shot["last_image"] = media_id(card.get("endImage"), "image")
        shot["source_video"] = media_id(card.get("referenceVideo"), "video") or global_source
        shot["ref_images"] = media_list(card.get("refs"), "image")
        shot["ref_videos"] = media_list(card.get("refVideos"), "video")
        shot["ref_audios"] = media_list(card.get("refAudios"), "audio")
        shot["character_ids"] = [character_mapping[item] for item in card.get("h3StudioCharacterIds", [])
                                  if item in character_mapping]
        shot["entity_ids"] = [entity_mapping[item] for item in card.get("h3StudioEntityIds", [])
                               if item in entity_mapping]
        shot["scene_id"] = scene_mapping.get(card.get("h3StudioSceneId"), "")
        extra = studio_shot_data.get(str(card.get("id")), {})
        for key in shot_data_fields:
            if key in extra:
                shot[key] = extra[key]
        project["shots"].append(shot)
    if not project["shots"]:
        project["shots"] = [new_shot()]
    for asset in new_assets.values():
        asset["project_id"] = project["id"]
    def save(data):
        data["projects"][project["id"]] = project
        data["assets"].update(new_assets)
    change(save)
    return project

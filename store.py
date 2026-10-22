"""Small on-disk project store for the standalone director app."""

from __future__ import annotations

import json
import shutil
import sqlite3
import threading
import uuid
from copy import deepcopy
from contextlib import contextmanager
from pathlib import Path


ROOT = Path(__file__).resolve().parent / "data"
ROOT.mkdir(exist_ok=True)
ASSETS = ROOT / "assets"
OUTPUTS = ROOT / "outputs"
ASSETS.mkdir(exist_ok=True)
OUTPUTS.mkdir(exist_ok=True)
STATE = ROOT / "state.json"
DATABASE = ROOT / "h3studio.sqlite3"
LOCK = threading.RLock()


def align_dimension(value, default: int) -> int:
    """Return the nearest positive multiple of 32, preferring lower on ties."""
    try:
        number = max(32, int(value))
    except (TypeError, ValueError):
        number = int(default)
    lower = max(32, number // 32 * 32)
    upper = lower + 32
    return lower if number - lower <= upper - number else upper


def uid() -> str:
    return uuid.uuid4().hex[:12]


def _empty() -> dict:
    return {"projects": {}, "assets": {}, "runs": {}}


def _normalize(data: dict) -> dict:
    data = {key: data.get(key, {}) for key in ("projects", "assets", "runs")}
    defaults = new_project("")
    for project in data.get("projects", {}).values():
        for key, value in defaults.items():
            if key not in project:
                project[key] = value
        project["width"] = align_dimension(project.get("width"), defaults["width"])
        project["height"] = align_dimension(project.get("height"), defaults["height"])
        used = set()
        next_number = 1
        for character in project.get("characters", []):
            if character.get("character_form") not in ("auto", "human", "animal", "anthropomorphic"):
                character["character_form"] = "auto"
            if character.get("multiview_layout") not in ("four", "five", "six"):
                character["multiview_layout"] = "four"
            if character.get("multiview_engine") not in ("xb_qwen", "krea2", "qwen_sheet"):
                character["multiview_engine"] = "xb_qwen"
            character.setdefault("xb_output_lang", "zh")
            character.setdefault("xb_sampling_mode", "bypass")
            character.setdefault("xb_skill_mode", "自动")
            # Older projects had no override flag. Preserve dimensions that a
            # user actually changed, while leaving untouched workflow 69 jobs
            # on that workflow's own 1024x1024 canvas.
            character.setdefault("xb_canvas_override", any((
                character.get("xb_aspect_ratio", "16:9") != "16:9",
                str(character.get("xb_width", 1920)) != "1920",
                str(character.get("xb_height", 1088)) != "1088",
            )))
            character.setdefault("xb_aspect_ratio", "16:9")
            character.setdefault("xb_width", 1920)
            character.setdefault("xb_height", 1088)
            character.setdefault("krea_width", 1536)
            character.setdefault("krea_height", 1024)
            character.setdefault("xb_seed", 0)
            character.setdefault("xb_thinking", False)
            character.setdefault("xb_preset_text", "")
            character.setdefault("xb_extra_settings", "")
            character.setdefault("xb_green_background", True)
            try:
                number = int(character.get("subject_number", 0))
            except (ValueError, TypeError):
                number = 0
            if number < 1 or number in used:
                number = max(next_number, max(used, default=0) + 1)
            character["subject_number"] = number
            used.add(number)
            next_number = max(next_number, number + 1)
        try:
            project["next_subject_number"] = max(int(project.get("next_subject_number", 1)), next_number)
        except (ValueError, TypeError):
            project["next_subject_number"] = next_number
    return data


def _connect() -> sqlite3.Connection:
    con = sqlite3.connect(DATABASE, timeout=30)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys=ON")
    con.execute("PRAGMA synchronous=FULL")
    return con


@contextmanager
def _db():
    con = _connect()
    try:
        with con:
            yield con
    finally:
        con.close()


def _initialize() -> None:
    DATABASE.parent.mkdir(parents=True, exist_ok=True)
    with _db() as con:
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA synchronous=FULL")
        for table in ("projects", "assets", "runs"):
            con.execute(f"CREATE TABLE IF NOT EXISTS {table} (id TEXT PRIMARY KEY, position INTEGER NOT NULL, data_json TEXT NOT NULL)")
        con.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        if con.execute("SELECT 1 FROM metadata WHERE key='initialized' ").fetchone():
            return
        # One-time, non-destructive migration. Keep state.json as a recoverable source copy.
        legacy = json.loads(STATE.read_text(encoding="utf-8")) if STATE.is_file() else _empty()
        data = _normalize(legacy)
        for table in ("projects", "assets", "runs"):
            for pos, (key, value) in enumerate(data[table].items()):
                con.execute(f"INSERT INTO {table}(id,position,data_json) VALUES(?,?,?)",
                            (str(key), pos, json.dumps(value, ensure_ascii=False)))
        con.execute("INSERT INTO metadata(key,value) VALUES('initialized','1')")
        con.execute("INSERT OR REPLACE INTO metadata(key,value) VALUES('migrated_from',?)",
                    (str(STATE) if STATE.is_file() else "new installation",))


def _load() -> dict:
    _initialize()
    data = _empty()
    with _db() as con:
        for table in data:
            data[table] = {row["id"]: json.loads(row["data_json"])
                           for row in con.execute(f"SELECT id,data_json FROM {table} ORDER BY position,id")}
    return _normalize(data)


def _serialized_rows(data: dict) -> dict:
    return {
        table: {str(key): (pos, json.dumps(value, ensure_ascii=False))
                for pos, (key, value) in enumerate(data[table].items())}
        for table in ("projects", "assets", "runs")
    }


def _save(data: dict, previous: dict | None = None) -> None:
    _initialize()
    current = _serialized_rows(data)
    with _db() as con:
        con.execute("BEGIN IMMEDIATE")
        for table in ("projects", "assets", "runs"):
            before = (previous or {}).get(table, {})
            removed = [(key,) for key in before.keys() - current[table].keys()]
            if removed:
                con.executemany(f"DELETE FROM {table} WHERE id=?", removed)
            changed = [(key, pos, raw) for key, (pos, raw) in current[table].items()
                       if before.get(key) != (pos, raw)]
            if changed:
                con.executemany(
                    f"INSERT INTO {table}(id,position,data_json) VALUES(?,?,?) "
                    "ON CONFLICT(id) DO UPDATE SET position=excluded.position,data_json=excluded.data_json",
                    changed,
                )


def storage_status() -> dict:
    _initialize()
    with _db() as con:
        counts = {table: con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                  for table in ("projects", "assets", "runs")}
        migrated = con.execute("SELECT value FROM metadata WHERE key='migrated_from'").fetchone()
    return {"backend": "SQLite", "database": str(DATABASE), "counts": counts,
            "media_directory": str(ASSETS), "migrated_from": migrated[0] if migrated else ""}


def read() -> dict:
    with LOCK:
        return _load()


def change(fn):
    with LOCK:
        data = _load()
        previous = _serialized_rows(data)
        result = fn(data)
        _save(data, previous)
        return result


def project_asset_ids(project: dict) -> set[str]:
    """Return every asset ID referenced by one project."""
    ids = set((project.get("image_descriptions") or {}).keys())
    for values in (project.get("shared_refs") or {}).values():
        ids.update(values or [])
    for shot in project.get("shots", []):
        ids.update(value for key in ("first_image", "last_image", "source_video")
                   if (value := shot.get(key)))
        for key in ("ref_images", "manual_ref_images", "ref_videos", "ref_audios"):
            ids.update(shot.get(key) or [])
        ids.update(guide.get("image_id") for guide in shot.get("h3_guides", [])
                   if isinstance(guide, dict) and guide.get("image_id"))
    for character in project.get("characters", []):
        ids.update(value for key in ("portrait_asset", "multiview_asset", "voice_asset")
                   if (value := character.get(key)))
    for entity in project.get("entities", []):
        ids.update(entity.get("image_assets") or [])
        if entity.get("generation_source"):
            ids.add(entity["generation_source"])
    return ids


def _replace_asset_id(value, old: str, new: str):
    if isinstance(value, dict):
        return {_replace_asset_id(key, old, new): _replace_asset_id(item, old, new)
                for key, item in value.items()}
    if isinstance(value, list):
        return [_replace_asset_id(item, old, new) for item in value]
    return new if value == old else value


def scope_legacy_assets() -> None:
    """Give old global assets project owners, copying assets shared by projects."""
    state = read()
    if not any(not asset.get("project_id") for asset in state["assets"].values()):
        return
    backup = DATABASE.with_name("h3studio.pre_asset_scope.sqlite3")
    if not backup.exists():
        with _connect() as source, sqlite3.connect(backup) as destination:
            source.backup(destination)

    def migrate(data):
        for asset_id, asset in list(data["assets"].items()):
            if asset.get("project_id"):
                continue
            owners = [project_id for project_id, project in data["projects"].items()
                      if asset_id in project_asset_ids(project)]
            if not owners:
                owners = list(data["projects"])[:1]
            if not owners:
                continue
            asset["project_id"] = owners[0]
            source = ASSETS / Path(asset["file"]).name
            for project_id in owners[1:]:
                if not source.is_file():
                    raise FileNotFoundError(f"共享素材文件不存在：{source}")
                clone_id = uid()
                clone_name = clone_id + source.suffix
                shutil.copy2(source, ASSETS / clone_name)
                clone = {**asset, "id": clone_id, "file": clone_name,
                         "project_id": project_id}
                data["assets"][clone_id] = clone
                project = _replace_asset_id(data["projects"][project_id], asset_id, clone_id)
                project["_data_version"] = int(project.get("_data_version", 0)) + 1
                data["projects"][project_id] = project
    change(migrate)


def clone_project(source_id: str, name: str = "") -> dict:
    """Copy a project and its referenced media into an independently editable project."""
    source_id = str(source_id or "")

    def clone(data):
        source = data["projects"].get(source_id)
        if not source:
            raise KeyError("项目不存在")
        project = deepcopy(source)
        project_id = uid()
        project["id"] = project_id
        project["name"] = str(name or "").strip() or f"{source.get('name') or '未命名项目'} 副本"
        project["_data_version"] = 0

        asset_map = {}
        copies = []
        for asset_id in project_asset_ids(source):
            asset = data["assets"].get(asset_id)
            if not asset:
                raise ValueError("项目引用了不存在的素材，无法完整复制")
            source_file = ASSETS / Path(str(asset.get("file") or "")).name
            if not source_file.is_file():
                raise FileNotFoundError(f"素材文件不存在：{source_file.name}")
            cloned_id = uid()
            target_file = cloned_id + source_file.suffix.lower()
            asset_map[asset_id] = cloned_id
            copies.append((source_file, ASSETS / target_file,
                           {**deepcopy(asset), "id": cloned_id, "file": target_file,
                            "project_id": project_id}))
        for old_id, new_id in asset_map.items():
            project = _replace_asset_id(project, old_id, new_id)

        character_map = {item.get("id"): uid() for item in project.get("characters", []) if item.get("id")}
        entity_map = {item.get("id"): uid() for item in project.get("entities", []) if item.get("id")}
        scene_map = {item.get("id"): uid() for item in project.get("scenes", []) if item.get("id")}
        for item in project.get("characters", []):
            if item.get("id") in character_map:
                item["id"] = character_map[item["id"]]
        for item in project.get("entities", []):
            if item.get("id") in entity_map:
                item["id"] = entity_map[item["id"]]
        for item in project.get("scenes", []):
            if item.get("id") in scene_map:
                item["id"] = scene_map[item["id"]]
        shot_map = {}
        for shot in project.get("shots", []):
            old_shot_id = shot.get("id")
            shot["id"] = uid()
            if old_shot_id:
                shot_map[old_shot_id] = shot["id"]
            shot["character_ids"] = [character_map.get(item, item) for item in shot.get("character_ids", [])]
            shot["entity_ids"] = [entity_map.get(item, item) for item in shot.get("entity_ids", [])]
            if shot.get("scene_id"):
                shot["scene_id"] = scene_map.get(shot["scene_id"], shot["scene_id"])

        run_copies = []
        cloned_runs = []
        for run in data.get("runs", {}).values():
            if run.get("project_id") != source_id:
                continue
            cloned_run = deepcopy(run)
            cloned_run_id = uid()
            cloned_run["id"] = cloned_run_id
            cloned_run["project_id"] = project_id
            cloned_run["shot_ids"] = [shot_map.get(item, item) for item in cloned_run.get("shot_ids", [])]
            output_map = {}

            def copy_output_reference(value):
                raw = str(value or "")
                if not raw:
                    return value
                if raw in output_map:
                    return output_map[raw]
                relative = Path(raw)
                source_file = OUTPUTS / relative
                try:
                    source_file.resolve().relative_to(OUTPUTS.resolve())
                except ValueError:
                    return value
                if not source_file.is_file():
                    return value
                target_file = OUTPUTS / cloned_run_id / source_file.name
                output_map[raw] = str(target_file.relative_to(OUTPUTS))
                run_copies.append((source_file, target_file))
                return output_map[raw]

            for result in cloned_run.get("results", []):
                if not isinstance(result, dict):
                    continue
                result["shot_id"] = shot_map.get(result.get("shot_id"), result.get("shot_id"))
                if result.get("file"):
                    result["file"] = copy_output_reference(result["file"])
            if cloned_run.get("preview_file"):
                cloned_run["preview_file"] = copy_output_reference(cloned_run["preview_file"])
            cloned_runs.append(cloned_run)

        copied_paths = []
        try:
            for source_file, target_file, asset in copies:
                shutil.copy2(source_file, target_file)
                copied_paths.append(target_file)
                data["assets"][asset["id"]] = asset
            for source_file, target_file in run_copies:
                target_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_file, target_file)
                copied_paths.append(target_file)
            for run in cloned_runs:
                data["runs"][run["id"]] = run
            data["projects"][project_id] = project
        except Exception:
            for target in copied_paths:
                target.unlink(missing_ok=True)
            raise
        return project

    return change(clone)


def new_shot() -> dict:
    return {
        "id": uid(), "name": "新镜头", "mode": "t2v", "prompt": "", "prompt_language": "en",
        "duration": 5, "seed": 42, "selected": True,
        "first_image": None, "last_image": None, "source_video": None,
        "source_start": 0, "ref_images": [], "manual_ref_images": [],
        "ref_videos": [], "ref_audios": [],
        "character_ids": [], "entity_ids": [], "scene_id": "",
        "continuity": True, "audio_mode": "generate", "ref_image_size": "match",
        "emotion": "自动/未指定", "camera_move": "自动/未指定",
        "camera_speed": "自动/未指定", "action_intensity": "自然",
        "speaker_id": "S1", "speaker_description": "画面中的人物",
        "dialogue_text": "", "dialogue_language": "Chinese",
        "dialogue_delivery": "", "dialogue_time": "", "dialogue_voiceover": False,
        "dialogue_cutoff": False, "on_screen_text": "",
        "dialogue_carries_out": False,
        "overall_soundscape": "", "non_diegetic_music": "",
        "ref_image_roles": {}, "ref_video_roles": {}, "ref_audio_roles": {},
        "replace_person": False, "swap_lora": "character", "faceswap_trigger_only": False,
    }


def new_project(name: str) -> dict:
    return {
        "id": uid(), "name": name.strip() or "未命名项目", "marked": False, "width": 1216, "height": 672,
        "_data_version": 0,
        "memory_policy": "manual",
        "prompt_skill": "h3-prompt-writing",
        "generation_speed": "smart",
        "h3_lora_dmad": False, "h3_lora_dmad_strength": 0.6,
        "h3_lora_mystic": False, "h3_lora_mystic_strength": 0.5,
        "sage_attention": False,
        "attention_mode": "sage",
        "h3_clip_model": "qwen3vl_32b_heretic_minimax_h3_nvfp4.safetensors",
        "h3_video_vae": "minimax_h3_video_vae_fp16.safetensors",
        "fps": 24, "export_mode": "all", "upscale": False, "confirm_upscale": True,
        "face_refine": False,
        "semantic_adapter": "", "semantic_alpha": 0.15, "semantic_magnitude_match": True,
        "selflift": False, "selflift_scale": 0.5, "selflift_highres_steps": 2,
        "selflift_rho": 0, "latent_model": "minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors",
        "selflift_tiling": False, "selflift_tile_count": 2, "selflift_tile_overlap": 128,
        "refine_mode": "off", "refine_steps": 4, "refine_denoise": 0.3,
        "refine_passes": 1,
        "refine_width": 1536, "refine_height": 864,
        "clear_vram_between_shots": False,
        "continuous_take": True,
        "identity_first_on_switch": False,
        "continuous_render_mode": "auto",
        "continuity_frames": 22,
        "seam_mode": "auto",
        "seam_prefix_frames": 3,
        "seam_motion_frames": 3,
        "seam_exposure": True,
        "upscale_resolution": 720, "upscale_max_resolution": 1280,
        "shared_refs": {"images": [], "videos": [], "audios": []},
        "image_descriptions": {},
        "shared_prompt": "",
        "sequence_brief": "",
        "characters": [],
        "entities": [],
        "next_subject_number": 1,
        "script_text": "", "scenes": [], "asset_catalog": [],
        "shots": [new_shot()],
    }

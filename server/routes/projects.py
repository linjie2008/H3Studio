"""项目 CRUD、克隆、分镜操作、切分、重拼、运行与整包保存。"""
from __future__ import annotations

from server.router import Route

from engine import ALL_MODES
from engine import build_h3_graph
from packs import export_director_pack
from packs import export_pack
from pathlib import Path
from reference_numbering import normalized_project_image_references
from reference_numbering import renumber_project_image_references
from runner import api
from runner import prepare_shot
from runner import remerge_run
from runner import run_project
from server.config import COMFY_DIR
from server.support import delete_projects
from store import ASSETS
from store import OUTPUTS
from store import align_dimension
from store import change
from store import clone_project
from store import new_project
from store import new_shot
from store import project_asset_ids
from store import read
from store import uid
from timeline_ops import edit_timeline
from video_tools import boundaries
import re
import threading
import time

def get_api_projects(req, path, parts, url):
    """原 app.py 分支：`path == "/api/projects"`。"""
    state = read()
    req.json(200, [normalized_project_image_references(project, state["assets"])
                    for project in state["projects"].values()])
    return


def get_api_projects_director_pack(req, path, parts, url):
    """原 app.py 分支：`path.startswith("/api/projects/") and path.endswith("/director-pack")`。"""
    project_id = path.split("/")[3]
    if project_id not in read()["projects"]:
        req.send_error(404)
        return
    req.file(export_director_pack(project_id))
    return


def get_api_projects_pack(req, path, parts, url):
    """原 app.py 分支：`path.startswith("/api/projects/") and path.endswith("/pack")`。"""
    project_id = path.split("/")[3]
    if project_id not in read()["projects"]:
        req.send_error(404)
        return
    req.file(export_pack(project_id))
    return


def get_api_projects_shot_versions(req, path, parts, url):
    """原 app.py 分支：`len(parts) == 6 and parts[:2] == ["api", "projects"] and parts[3] == "shots" and parts[5] == "versions"`。"""
    state = read()
    project = state["projects"].get(parts[2])
    if not project or not any(shot.get("id") == parts[4] for shot in project.get("shots", [])):
        req.send_error(404)
        return
    versions = []
    for run in state["runs"].values():
        if run.get("project_id") != parts[2]:
            continue
        for result in run.get("results", []):
            if result.get("shot_id") == parts[4] and (OUTPUTS / result.get("file", "")).is_file():
                versions.append({"run_id": run["id"], "created_at": run.get("created_at", 0),
                                 "status": run.get("status", ""),
                                 "elapsed_seconds": max(0, int(run["finished_at"] - run["started_at"]))
                                     if run.get("finished_at") and run.get("started_at") else None,
                                 **result})
    versions.sort(key=lambda item: item.get("created_at", 0), reverse=True)
    req.json(200, versions)
    return


def get_api_project(req, path, parts, url):
    """原 app.py 分支：`path.startswith("/api/projects/")`。"""
    project_id = path.split("/")[3]
    state = read()
    project = state["projects"].get(project_id)
    if project:
        project = normalized_project_image_references(project, state["assets"])
    req.json(200, project) if project else req.send_error(404)
    return


def post_api_projects_create(req, path, parts, url):
    """原 app.py 分支：`path == "/api/projects"`。"""
    body = req.body_json()
    project = new_project(str(body.get("name") or ""))
    change(lambda data: data["projects"].__setitem__(project["id"], project))
    req.json(201, project)
    return


def post_api_projects_clone(req, path, parts, url):
    """原 app.py 分支：`path.startswith("/api/projects/") and path.endswith("/clone")`。"""
    if not req.same_origin_request():
        req.json(403, {"error": "请从工作台页面复制项目"})
        return
    source_id = path.split("/")[3]
    project = clone_project(source_id, str(req.body_json().get("name") or ""))
    req.json(201, project)
    return


def post_api_projects_batch_delete(req, path, parts, url):
    """原 app.py 分支：`path == "/api/projects/batch-delete"`。"""
    body = req.body_json()
    project_ids = body.get("project_ids")
    if not isinstance(project_ids, list):
        raise ValueError("项目 ID 列表无效")
    req.json(200, delete_projects(project_ids))
    return


def post_api_timeline_edit(req, path, parts, url):
    """原 app.py 分支：`path == "/api/timeline/edit"`。"""
    body = req.body_json()
    shots = edit_timeline(body.get("shots"), body.get("shot_ids"),
                          str(body.get("action") or ""), body.get("value"))
    req.json(200, {"shots": shots})
    return


def post_api_projects_shots(req, path, parts, url):
    """原 app.py 分支：`len(parts) == 4 and parts[:2] == ["api", "projects"] and parts[3] == "shots"`。"""
    body = req.body_json()
    shot = new_shot()
    mode = body.get("mode")
    if mode in {"t2v", "i2v", "fl2v", "r2v", "v2v", "rv2v"}:
        shot["mode"] = mode
    change(lambda data: data["projects"][parts[2]]["shots"].append(shot))
    req.json(201, shot)
    return


def post_api_projects_split(req, path, parts, url):
    """原 app.py 分支：`len(parts) == 4 and parts[:2] == ["api", "projects"] and parts[3] == "split"`。"""
    body = req.body_json()
    asset_id = body.get("asset_id")
    asset = read()["assets"].get(asset_id)
    if not asset or asset["kind"] != "video" or asset.get("project_id") != parts[2]:
        raise ValueError("请选择已上传的视频")
    spans = boundaries(ASSETS / asset["file"], body.get("method", "equal"),
                       body.get("count", 4), body.get("threshold", 0.35),
                       body.get("custom_cuts", []), body.get("sensitivity", "medium"))
    shots = []
    for index, (start, length) in enumerate(spans, 1):
        shot = new_shot()
        shot.update(name=f"{asset['name']} · {index}", mode="v2v", source_video=asset_id,
                    source_start=start, duration=length, prompt="保留源视频的主体、动作与构图，提升细节和光照。")
        shots.append(shot)
    # A source video has one active split. Replace its previous segments at
    # their original timeline position, including requests from older clients.
    def save_split(data):
        project_shots = data["projects"][parts[2]]["shots"]
        old_positions = [index for index, item in enumerate(project_shots)
                         if item.get("source_video") == asset_id
                         and item.get("mode") in {"v2v", "rv2v"}]
        insert_at = old_positions[0] if old_positions else len(project_shots)
        project_shots[:] = [item for item in project_shots if not (
            item.get("source_video") == asset_id
            and item.get("mode") in {"v2v", "rv2v"}
        )]
        project_shots[insert_at:insert_at] = shots
    change(save_split)
    req.json(201, shots)
    return


def post_api_projects_remerge(req, path, parts, url):
    """原 app.py 分支：`len(parts) == 4 and parts[:2] == ["api", "projects"] and parts[3] == "remerge"`。"""
    if not req.same_origin_request():
        req.json(403, {"error": "请从当前工作台重新拼接视频"})
        return
    state = read()
    project = state["projects"].get(parts[2])
    if not project:
        req.send_error(404)
        return
    eligible = [run for run in state["runs"].values()
                if run.get("project_id") == parts[2] and run.get("status") == "completed"
                and len(run.get("results", [])) >= 2
                and all(item.get("file") and (OUTPUTS / item["file"]).is_file()
                        for item in run["results"])]
    if not eligible:
        raise ValueError("当前项目没有可重新拼接的完整分段任务")
    latest = max(eligible, key=lambda item: item.get("finished_at") or item.get("created_at") or 0)
    req.json(200, remerge_run(latest["id"], project))
    return


def post_api_projects_run(req, path, parts, url):
    """原 app.py 分支：`len(parts) == 4 and parts[:2] == ["api", "projects"] and parts[3] == "run"`。"""
    project = read()["projects"].get(parts[2])
    if not project:
        req.send_error(404)
        return
    body = req.body_json()
    reroll_shot_id = body.get("reroll_shot_id")
    source_run_id = body.get("source_run_id")
    selected = ([reroll_shot_id] if reroll_shot_id else
                body.get("shot_ids") or [shot["id"] for shot in project["shots"] if shot.get("selected", True)])
    valid = {shot["id"] for shot in project["shots"]}
    if not selected or any(item not in valid for item in selected):
        raise ValueError("请至少选择一个有效镜头")
    if reroll_shot_id:
        source_run = read()["runs"].get(str(source_run_id or ""))
        if (not source_run or source_run.get("project_id") != parts[2]
                or source_run.get("status") != "completed"
                or not any(item.get("shot_id") == reroll_shot_id for item in source_run.get("results", []))
                or any(item.get("native_take") for item in source_run.get("results", []))
                or any(not item.get("file") or not (OUTPUTS / item["file"]).is_file()
                       for item in source_run.get("results", []))):
            raise ValueError("请先选择包含这个镜头的已完成成片，再重抽")
    assets = read()["assets"]
    foreign_assets = [asset_id for asset_id in project_asset_ids(project)
                      if asset_id in assets and assets[asset_id].get("project_id") != parts[2]]
    if foreign_assets:
        raise ValueError("项目引用了其他项目的素材，请刷新项目素材库")
    for shot in project["shots"]:
        if shot["id"] not in selected:
            continue
        preview = prepare_shot(shot, project)
        ids = [value for key in ("first_image", "last_image", "source_video") if (value := preview.get(key))]
        for key in ("ref_images", "ref_videos", "ref_audios"):
            ids.extend(preview.get(key) or [])
        ids.extend(guide.get("image_id") for guide in preview.get("h3_guides", [])
                   if isinstance(guide, dict) and guide.get("image_id"))
        missing = [asset_id for asset_id in ids if asset_id not in assets]
        if missing:
            raise ValueError(f"镜头 {shot['name']} 引用了不存在的素材：{missing[0]}")
        media = {asset_id: assets[asset_id]["file"] for asset_id in ids}
        try:
            graph = build_h3_graph(preview, project, media, "preflight")
        except (ValueError, KeyError) as error:
            raise ValueError(f"镜头 {shot['name']}：{error}") from error
        for node in graph.values():
            if node["class_type"] == "LoraLoaderModelOnly":
                lora_name = node["inputs"]["lora_name"]
                if not (COMFY_DIR / "models" / "loras" / lora_name).is_file():
                    raise ValueError(f"镜头 {shot['name']}：缺少 H3 LoRA：{lora_name}")
        if project.get("generation_speed") == "goldfish":
            from engine import GOLDFISH_MODELS, GOLDFISH_CLIP, GOLDFISH_VAE, GOLDFISH_UPSCALER
            goldfish_family = "creative" if shot.get("mode", "t2v") in ("t2v", "i2v", "fl2v") else "reference"
            required = (("H3 主模型", COMFY_DIR / "models" / "diffusion_models" / GOLDFISH_MODELS[goldfish_family]),
                        ("文本编码器", COMFY_DIR / "models" / "text_encoders" / GOLDFISH_CLIP),
                        ("视频 VAE", COMFY_DIR / "models" / "vae" / GOLDFISH_VAE),
                        ("H3 latent 放大器", COMFY_DIR / "models" / "latent_upscale_models" / GOLDFISH_UPSCALER))
            missing_goldfish = [f"{label}：{path.name}" for label, path in required if not path.is_file()]
            if missing_goldfish:
                raise ValueError("金鱼高清流程缺少模型：" + "；".join(missing_goldfish))
            try:
                selflift_ready = "SelfLiftH3Sampler" in api("/object_info/SelfLiftH3Sampler")
            except Exception:
                selflift_ready = False
            if not selflift_ready:
                raise ValueError("金鱼高清流程需要 SelfLift H3 节点；请在安装节点后重启 ComfyUI")
    if project.get("semantic_adapter"):
        adapter = project["semantic_adapter"]
        if Path(adapter).name != adapter or not (COMFY_DIR / "models" / "semantic_bridge" / adapter).is_file():
            raise ValueError("语义桥适配器文件不存在")
    if project.get("generation_speed") != "goldfish" and (project.get("selflift") or project.get("refine_mode") in ("latent", "latent_only")):
        model_name = project.get("latent_model", "")
        if Path(model_name).name != model_name or not (COMFY_DIR / "models" / "latent_upscale_models" / model_name).is_file():
            raise ValueError("H3 3D latent 权重文件不存在")
    vae_name = ("minimax_h3_video_vae_fp16.safetensors" if project.get("generation_speed") == "goldfish"
                else str(project.get("h3_video_vae") or "minimax_h3_video_vae_fp16.safetensors"))
    if Path(vae_name).name != vae_name or not (COMFY_DIR / "models" / "vae" / vae_name).is_file():
        raise ValueError("所选 H3 视频 VAE 文件不存在，请在项目设置中重新选择")
    attention_mode = str(project.get("attention_mode") or ("sage" if project.get("sage_attention") else "off"))
    if attention_mode == "sage":
        try:
            sage_ready = "MiniMaxH3MemoryEfficientSageAttentionPatch" in api("/object_info/MiniMaxH3MemoryEfficientSageAttentionPatch")
        except Exception:
            sage_ready = False
        if not sage_ready:
            raise ValueError("当前 ComfyUI 未加载 H3 SageAttention 节点；请安装/启用 KJNodes 后重启 ComfyUI")
    if attention_mode == "veda":
        try:
            veda_ready = "VedaSparseAttention" in api("/object_info/VedaSparseAttention")
        except Exception:
            veda_ready = False
        predictor = "minimax_h3_t2va_veda_8nfe_600step_preview_fp8.safetensors"
        if not veda_ready or not (COMFY_DIR / "models" / "veda" / predictor).is_file():
            raise ValueError("Veda 节点或 predictor 权重不可用；请安装 Veda 插件并放入对应模型后再选用")
    run_id = uid()
    run = {"id": run_id, "project_id": parts[2], "shot_ids": selected,
           "status": "queued", "current": "", "results": [], "created_at": time.time()}
    if reroll_shot_id:
        run.update(reroll_shot_id=reroll_shot_id, source_run_id=source_run_id)
    change(lambda data: data["runs"].__setitem__(run_id, run))
    threading.Thread(target=run_project, args=(run_id,), daemon=True).start()
    req.json(202, run)
    return


def delete_api_projects(req, path, parts, url):
    """原 app.py 分支：`len(parts) == 3 and parts[:2] == ["api", "projects"]`。"""
    project_id = parts[2]
    state = read()
    if project_id not in state["projects"]:
        req.send_error(404)
        return
    if any(run.get("project_id") == project_id and
           run.get("status") in ("queued", "running", "awaiting_upscale")
           for run in state["runs"].values()):
        req.json(409, {"error": "该项目仍有生成任务，请等待完成或取消后再删除"})
        return
    def remove(data):
        return data["projects"].pop(project_id, None)
    removed = change(remove)
    req.json(200, {"deleted": bool(removed)})


def put_api_projects(req, path, parts, url):
    """原 app.py 分支：`len(parts) == 3 and parts[:2] == ["api", "projects"]`。"""
    project = req.body_json()
    if project.get("id") != parts[2]:
        raise ValueError("项目 ID 不匹配")
    project["name"] = str(project.get("name") or "").strip()
    if not project["name"] or len(project["name"]) > 120:
        raise ValueError("项目名称需为 1–120 字")
    project["marked"] = bool(project.get("marked", False))
    project["width"] = align_dimension(project.get("width"), 1344)
    project["height"] = align_dimension(project.get("height"), 768)
    if project.get("memory_policy", "manual") not in ("manual", "auto_vram"):
        raise ValueError("显存画布模式无效")
    if project.get("continuous_render_mode", "auto") not in ("auto", "segmented"):
        raise ValueError("一镜到底生成方式无效")
    if project.get("seam_mode", "auto") not in ("auto", "conservative", "manual"):
        raise ValueError("拼接模式无效")
    try:
        prefix_frames = int(project.get("seam_prefix_frames", 3))
        motion_frames = int(project.get("seam_motion_frames", 3))
    except (TypeError, ValueError):
        raise ValueError("拼接帧数无效")
    if prefix_frames not in (0, 1, 2, 3) or motion_frames not in (0, 2, 3):
        raise ValueError("拼接帧数超出可选范围")
    if not isinstance(project.get("seam_exposure", True), bool):
        raise ValueError("亮度匹配设置无效")
    if not isinstance(project.get("shots"), list):
        raise ValueError("缺少镜头列表")
    for shot in project["shots"]:
        if shot.get("mode") not in ALL_MODES or not shot.get("id"):
            raise ValueError("镜头模式或 ID 无效")
        if shot.get("prompt_language", "en") not in {"en", "zh"}:
            raise ValueError("提示词语言只能选择中文或英文")
    characters = project.get("characters", [])
    if not isinstance(characters, list) or any(not isinstance(character, dict) for character in characters):
        raise ValueError("角色档案列表格式无效")
    entities = project.get("entities", [])
    if not isinstance(entities, list) or any(not isinstance(entity, dict) for entity in entities):
        raise ValueError("场景与道具档案列表格式无效")
    entity_ids = [entity.get("id") for entity in entities]
    if any(not item for item in entity_ids) or len(set(entity_ids)) != len(entity_ids):
        raise ValueError("场景与道具档案 ID 无效或重复")
    asset_catalog = read()["assets"]
    image_descriptions = project.get("image_descriptions") or {}
    if not isinstance(image_descriptions, dict) or any(
        not isinstance(description, str) or len(description) > 600
        for description in image_descriptions.values()
    ):
        raise ValueError("图片画面说明无效，请检查素材是否仍存在，且每项不超过 600 字")
    for entity in entities:
        if entity.get("kind", "prop") not in ("prop", "place", "style"):
            raise ValueError("档案类型只能是道具、地点或画风")
        if entity.get("generation_mode", "t2i") not in ("t2i", "i2i"):
            raise ValueError("档案生图方式无效")
        if not isinstance(entity.get("image_assets", []), list):
            raise ValueError("档案参考图片列表格式无效")
        if entity.get("generation_source"):
            source = asset_catalog.get(entity["generation_source"])
            if not source or source.get("kind") != "image" or source.get("project_id") != parts[2]:
                raise ValueError("档案图生图参考图片无效")
        for asset_id in entity.get("image_assets", []):
            if asset_id not in asset_catalog or asset_catalog[asset_id].get("kind") != "image":
                raise ValueError(f"档案“{entity.get('name', '')}”引用了不存在的图片素材")
    scenes = project.get("scenes", [])
    if not isinstance(scenes, list) or any(not isinstance(scene, dict) for scene in scenes):
        raise ValueError("场景分组列表格式无效")
    scene_ids = [scene.get("id") for scene in scenes]
    if any(not item for item in scene_ids) or len(set(scene_ids)) != len(scene_ids):
        raise ValueError("场景分组 ID 无效或重复")
    character_ids = [character.get("id") for character in characters]
    if any(not item for item in character_ids) or len(set(character_ids)) != len(character_ids):
        raise ValueError("角色档案 ID 无效或重复")
    profile_ids = set(character_ids)
    state = read()
    previous_project = state["projects"].get(parts[2], {})
    try:
        expected_version = int(project.get("_data_version", 0))
        current_version = int(previous_project.get("_data_version", 0))
    except (ValueError, TypeError):
        raise ValueError("项目版本无效，请刷新工作台后重试")
    if expected_version != current_version:
        raise ValueError("项目已在其他页面更新，请刷新工作台以免覆盖新内容")
    try:
        next_subject_number = max(int(project.get("next_subject_number", 1)),
                                  int(previous_project.get("next_subject_number", 1)))
    except (ValueError, TypeError):
        next_subject_number = int(previous_project.get("next_subject_number", 1) or 1)
    used_subject_numbers = set()
    previous_subject_numbers = {item.get("id"): int(item.get("subject_number", index + 1))
                                 for index, item in enumerate(previous_project.get("characters", []))}
    for character in characters:
        if character.get("character_form", "auto") not in ("auto", "human", "animal", "anthropomorphic"):
            raise ValueError(f"角色“{character.get('name','')}”的体态类型无效")
        if character.get("id") in previous_subject_numbers:
            subject_number = previous_subject_numbers[character["id"]]
        else:
            while next_subject_number in used_subject_numbers:
                next_subject_number += 1
            subject_number = next_subject_number
            next_subject_number += 1
        character["subject_number"] = subject_number
        used_subject_numbers.add(subject_number)
    project["next_subject_number"] = max(next_subject_number, max(used_subject_numbers, default=0) + 1)
    project["_data_version"] = current_version + 1
    asset_catalog = state["assets"]
    foreign_assets = [asset_id for asset_id in project_asset_ids(project)
                      if asset_id in asset_catalog
                      and asset_catalog[asset_id].get("project_id") != parts[2]]
    if foreign_assets:
        raise ValueError("项目不能引用其他项目的素材")
    for character in characters:
        speaker_id = str(character.get("speaker_id") or "S1")
        if not re.fullmatch(r"S\d+", speaker_id):
            raise ValueError(f"角色“{character.get('name','')}”的说话人编号需为 S1、S2 这样的格式")
        if character.get("portrait_asset") and asset_catalog.get(character["portrait_asset"], {}).get("kind") != "image":
            raise ValueError(f"角色“{character.get('name','')}”的外观素材必须是图片")
        if character.get("multiview_asset") and asset_catalog.get(character["multiview_asset"], {}).get("kind") != "image":
            raise ValueError(f"角色“{character.get('name','')}”的多视图素材必须是图片")
        if character.get("voice_asset") and asset_catalog.get(character["voice_asset"], {}).get("kind") != "audio":
            raise ValueError(f"角色“{character.get('name','')}”的声音素材必须是音频")
        if character.get("multiview_layout", "four") not in ("four", "five", "six"):
            raise ValueError(f"角色“{character.get('name','')}”的多视图布局无效")
    for shot in project["shots"]:
        assigned_characters = shot.get("character_ids", [])
        if not isinstance(assigned_characters, list):
            raise ValueError(f"镜头“{shot.get('name','')}”的角色档案格式无效")
        if any(item not in profile_ids for item in assigned_characters):
            raise ValueError(f"镜头“{shot.get('name','')}”引用了不存在的角色档案")
        assigned_entities = shot.get("entity_ids", [])
        if not isinstance(assigned_entities, list) or any(item not in set(entity_ids) for item in assigned_entities):
            raise ValueError(f"镜头“{shot.get('name','')}”引用了不存在的地点或道具档案")
        if shot.get("scene_id") and shot["scene_id"] not in set(scene_ids):
            raise ValueError(f"镜头“{shot.get('name','')}”引用了不存在的场景分组")
    def save(data):
        if parts[2] not in data["projects"]:
            raise KeyError("项目不存在")
        if int(data["projects"][parts[2]].get("_data_version", 0)) != expected_version:
            raise ValueError("项目已在其他页面更新，请刷新工作台以免覆盖新内容")
        live_images = {asset_id for asset_id, asset in data["assets"].items()
                       if asset.get("kind") == "image"}
        previous_descriptions = data["projects"][parts[2]].get("image_descriptions") or {}
        project["image_descriptions"] = {
            **{asset_id: description for asset_id, description in previous_descriptions.items()
               if asset_id in live_images and isinstance(description, str) and len(description) <= 600},
            **{asset_id: description for asset_id, description in image_descriptions.items()
               if asset_id in live_images},
        }
        active_image_ids = [asset_id for asset_id, asset in data["assets"].items()
                            if asset.get("kind") == "image" and asset.get("project_id") == parts[2]]
        renumber_project_image_references(project, active_image_ids)
        project["_data_version"] = expected_version + 1
        data["projects"][parts[2]] = project
    change(save)
    req.json(200, project)


ROUTES = [
    Route('GET', 'get_api_projects',
         lambda path, parts, url: (path == "/api/projects"),
         get_api_projects),
    Route('GET', 'get_api_projects_director_pack',
         lambda path, parts, url: (path.startswith("/api/projects/") and path.endswith("/director-pack")),
         get_api_projects_director_pack),
    Route('GET', 'get_api_projects_pack',
         lambda path, parts, url: (path.startswith("/api/projects/") and path.endswith("/pack")),
         get_api_projects_pack),
    Route('GET', 'get_api_projects_shot_versions',
         lambda path, parts, url: (len(parts) == 6 and parts[:2] == ["api", "projects"] and parts[3] == "shots" and parts[5] == "versions"),
         get_api_projects_shot_versions),
    Route('GET', 'get_api_project',
         lambda path, parts, url: (path.startswith("/api/projects/")),
         get_api_project),
    Route('POST', 'post_api_projects_create',
         lambda path, parts, url: (path == "/api/projects"),
         post_api_projects_create),
    Route('POST', 'post_api_projects_clone',
         lambda path, parts, url: (path.startswith("/api/projects/") and path.endswith("/clone")),
         post_api_projects_clone),
    Route('POST', 'post_api_projects_batch_delete',
         lambda path, parts, url: (path == "/api/projects/batch-delete"),
         post_api_projects_batch_delete),
    Route('POST', 'post_api_timeline_edit',
         lambda path, parts, url: (path == "/api/timeline/edit"),
         post_api_timeline_edit),
    Route('POST', 'post_api_projects_shots',
         lambda path, parts, url: (len(parts) == 4 and parts[:2] == ["api", "projects"] and parts[3] == "shots"),
         post_api_projects_shots),
    Route('POST', 'post_api_projects_split',
         lambda path, parts, url: (len(parts) == 4 and parts[:2] == ["api", "projects"] and parts[3] == "split"),
         post_api_projects_split),
    Route('POST', 'post_api_projects_remerge',
         lambda path, parts, url: (len(parts) == 4 and parts[:2] == ["api", "projects"] and parts[3] == "remerge"),
         post_api_projects_remerge),
    Route('POST', 'post_api_projects_run',
         lambda path, parts, url: (len(parts) == 4 and parts[:2] == ["api", "projects"] and parts[3] == "run"),
         post_api_projects_run),
    Route('DELETE', 'delete_api_projects',
         lambda path, parts, url: (len(parts) == 3 and parts[:2] == ["api", "projects"]),
         delete_api_projects),
    Route('PUT', 'put_api_projects',
         lambda path, parts, url: (len(parts) == 3 and parts[:2] == ["api", "projects"]),
         put_api_projects),
]

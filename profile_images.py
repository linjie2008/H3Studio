"""Generate project entity reference images with native Qwen-Image 2.1 nodes."""

from __future__ import annotations

import secrets
import shutil
import time
from pathlib import Path

from multiview import COMFY_OUTPUT, _model_config, wait_for_image
from runner import COMFY_INPUT, _submit, update_run, release_idle_comfy_after_job
from store import ASSETS, change, read, uid


LABELS = {"place": "地点", "prop": "道具", "style": "画风"}
NEGATIVE = "text, letters, labels, watermark, collage, split screen, blurry, deformed, duplicate objects"


def build_graph(kind: str, description: str, mode: str, source_name: str | None,
                preset: str, job_id: str) -> dict:
    if kind not in LABELS or mode not in {"t2i", "i2i"} or preset not in {"fast", "stable"}:
        raise ValueError("档案生图参数无效")
    if mode == "i2i" and not source_name:
        raise ValueError("图生图需要参考图片")
    description = description.strip()
    if not description:
        raise ValueError("请先填写生图描述")
    model, clip, vae = _model_config()
    width, height = (1024, 1024) if kind == "prop" else (1536, 864)
    kind_rule = {
        "place": "Create one coherent environment reference image. Preserve spatial layout, architecture and lighting. No inset panels or people unless explicitly requested.",
        "prop": "Create one clear prop reference image. Show its shape, material, construction and distinctive details, centered and fully visible.",
        "style": "Create one coherent visual-style reference image. Show palette, lighting, texture, rendering technique and mood in a representative scene.",
    }[kind]
    source_rule = ("Image 1 is only a blank output canvas. Image 2 is the source reference; "
                   "preserve its recognizable subject and intended details while applying the requested changes. "
                   if mode == "i2i" else "")
    graph = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": model, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": clip, "type": "qwen_image", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": vae}},
        "4": {"class_type": "EmptyImage", "inputs": {"width": width, "height": height,
                                                     "batch_size": 1, "color": 15132390}},
        "6": {"class_type": "TextEncodeQwenImage21", "inputs": {
            "clip": ["2", 0], "vae": ["3", 0],
            "prompt": source_rule + kind_rule + " User requirements: " + description[:1800] +
                      " No text, labels or watermark.",
            "negative_prompt": NEGATIVE, "resolution": 0}},
        "7": {"class_type": "KSampler", "inputs": {
            "model": ["1", 0], "positive": ["6", 0], "negative": ["6", 1],
            "latent_image": ["6", 2], "seed": secrets.randbits(48),
            "steps": 16 if preset == "fast" else 25, "cfg": 1.0,
            "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["3", 0]}},
        "9": {"class_type": "SaveImage", "inputs": {
            "images": ["8", 0], "filename_prefix": "h3studio_profile_" + job_id}},
    }
    if source_name:
        graph["5"] = {"class_type": "LoadImage", "inputs": {"image": source_name}}
        graph["6"]["inputs"]["images.image_1"] = ["4", 0]
        graph["6"]["inputs"]["images.image_2"] = ["5", 0]
    else:
        graph["10"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["4", 0], "vae": ["3", 0]}}
        graph["7"]["inputs"]["latent_image"] = ["10", 0]
    return graph


def generate(job_id: str) -> None:
    job = read()["runs"][job_id]
    source_input = None
    submitted = False
    completed = False
    try:
        if job.get("source_asset"):
            source = read()["assets"].get(job["source_asset"])
            source_path = ASSETS / source["file"] if source else None
            if not source_path or not source_path.is_file():
                raise FileNotFoundError("参考图片已不存在")
            COMFY_INPUT.mkdir(parents=True, exist_ok=True)
            source_input = COMFY_INPUT / ("h3studio_profile_" + job_id + source_path.suffix.lower())
            shutil.copy2(source_path, source_input)
        graph = build_graph(job["entity_kind"], job["description"], job["mode"],
                            source_input.name if source_input else None, job["preset"], job_id)
        if read()["runs"][job_id].get("cancel_requested"):
            raise RuntimeError("用户取消生图任务")
        prompt_id = _submit(graph, job_id)
        submitted = True
        update_run(job_id, status="running", prompt_id=prompt_id, current="Qwen2.1 档案生图中")
        history = wait_for_image(prompt_id, job_id)
        images = history.get("outputs", {}).get("9", {}).get("images", [])
        if not images:
            raise RuntimeError("Qwen2.1 没有输出图片")
        image = images[0]
        if image.get("type") != "output" or Path(str(image.get("filename", ""))).suffix.lower() not in {".png", ".webp"}:
            raise RuntimeError("ComfyUI 返回了无效图片")
        output = (COMFY_OUTPUT / Path(str(image.get("subfolder") or "")) / str(image["filename"])).resolve()
        if not output.is_relative_to(COMFY_OUTPUT.resolve()) or not output.is_file():
            raise FileNotFoundError("ComfyUI 图片文件不存在")
        asset_id = uid()
        destination = ASSETS / (asset_id + output.suffix.lower())
        shutil.copy2(output, destination)
        asset = {"id": asset_id, "name": job["entity_name"] + " · 生成参考图",
                 "file": destination.name, "kind": "image", "size": destination.stat().st_size,
                 "created_at": time.time(), "project_id": job["project_id"],
                 "generated_by": "qwen_image_2.1_profile", "generation_mode": job["mode"]}

        def finish(data):
            current_job = data["runs"].get(job_id)
            project = data["projects"].get(job["project_id"])
            entity = next((item for item in project.get("entities", []) if item.get("id") == job["entity_id"]), None) if project else None
            if not current_job or current_job.get("cancel_requested") or not entity:
                return False
            data["assets"][asset_id] = asset
            entity.setdefault("image_assets", []).append(asset_id)
            project.setdefault("image_descriptions", {})[asset_id] = (job["entity_name"] + "：" + job["description"])[:600]
            project["_data_version"] = int(project.get("_data_version", 0)) + 1
            current_job.update(status="completed", asset_id=asset_id, bound=True,
                               finished_at=time.time(), current="参考图已生成并绑定")
            return True

        if not change(finish):
            destination.unlink(missing_ok=True)
            raise RuntimeError("项目或档案已变化，生成图片未保存")
        completed = True
    except Exception as error:
        update_run(job_id, status="cancelled" if read()["runs"][job_id].get("cancel_requested") else "error",
                   error=str(error), finished_at=time.time())
    finally:
        if source_input and (completed or not submitted):
            source_input.unlink(missing_ok=True)
        release_idle_comfy_after_job(job_id)

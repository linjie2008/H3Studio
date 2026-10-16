"""Generate character multi-view sheets with Qwen-Image or Krea2."""

from __future__ import annotations

import os
import json
import math
import secrets
import shutil
import time
import urllib.error
import urllib.parse
from pathlib import Path

from runner import COMFY_INPUT, api, _submit, update_run
from store import ASSETS, change, read, uid


COMFY_OUTPUT = Path(os.environ.get("COMFYUI_DIR", "/home/administrator/ComfyUI")) / "output"
NEGATIVE = (
    "different people, identity drift, changed clothes, extra person, duplicate panel, "
    "missing panel, wrong number of panels, front view in back panel, cropped feet, weapon, text, labels, "
    "watermark, deformed hands, extra limbs"
)
KREA2_NODES = ("Krea2EditGroundedEncode", "Krea2EditModelPatch", "ImageResizeKJv2")
XB_NODE = "XB_QwenPromptPreset"
XB_TEXT_ENCODER = "qwen3.5_9b_qwen_image_2.1_pe_t2i.int8_convrot.safetensors"
GREEN_BG_NODE = "easy imageRemBg"
XB_I2I_WORKFLOW = (Path(os.environ.get("COMFYUI_DIR", "/home/administrator/ComfyUI")) /
                   "user/default/workflows/Qwen-Image/2.1/图片编辑/"
                   "Qwen-Image-2.1_69-图片编辑进阶工作流-本机适配.json")
_SIX_VIEW_WORKFLOW_CANDIDATES = tuple(
    Path(os.environ.get("COMFYUI_DIR", "/home/administrator/ComfyUI")) / relative
    for relative in (
        # Current location inside the Qwen-Image category folder.
        "user/default/workflows/Qwen-Image/2.1/"
        "qwen-image2.1_人物定装照四视图_六视图.json",
        # Legacy location at the workflows root.
        "user/default/workflows/qwen-image2.1_人物定装照四视图_六视图.json",
    ))
SIX_VIEW_WORKFLOW = next((path for path in _SIX_VIEW_WORKFLOW_CANDIDATES if path.is_file()),
                         _SIX_VIEW_WORKFLOW_CANDIDATES[0])


def _xb_i2i_workflow_defaults() -> dict:
    """Read the user's saved XB node settings from their workflow 69."""
    try:
        workflow = json.loads(XB_I2I_WORKFLOW.read_text(encoding="utf-8"))
        node = next(item for item in workflow.get("nodes", [])
                    if item.get("type") == XB_NODE)
        values = node.get("widgets_values") or []
        return {
            "output_lang": values[1], "preset_mode": values[3], "preset_text": values[4],
            "skill_mode": values[5], "skill_name": values[6], "aspect_ratio": values[7],
            "width": values[8], "height": values[9], "batch_size": values[10],
            "clip_name": values[11], "clip_type": values[12], "device": values[13],
            "max_length": values[14], "sampling_mode": values[15], "temperature": values[16],
            "top_k": values[17], "top_p": values[18], "min_p": values[19],
            "repetition_penalty": values[20], "presence_penalty": values[21],
            "seed": values[22], "thinking": values[24],
            "use_default_template": values[25], "mtp": values[26],
            "internal_prompt": values[27], "manager_settings": values[28],
        }
    except (OSError, ValueError, KeyError, IndexError, StopIteration, TypeError):
        return {}


def _xb_manager_settings(config: dict, extra: str) -> str:
    try:
        payload = json.loads(config.get("manager_settings") or "{}")
    except (TypeError, ValueError):
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    qwen = payload.setdefault("qwen", {})
    if not isinstance(qwen, dict):
        qwen = payload["qwen"] = {}
    qwen["extra_settings"] = extra
    return json.dumps(payload, ensure_ascii=False)


def _attach_green_background(graph: dict, image_link: list, save_node: str,
                             width: int | list, height: int | list,
                             batch_size: int | list = 1) -> None:
    """Remove generated backgrounds and composite the subject onto exact chroma green."""
    graph["200"] = {"class_type": GREEN_BG_NODE, "inputs": {
        # RMBG-2.0 is gated on Hugging Face and fails on machines without
        # explicit access. RMBG-1.4 is the public EasyUse-supported model.
        "images": image_link, "rem_mode": "RMBG-1.4", "image_output": "Hide",
        "save_prefix": "h3studio_green_matte"}}
    graph["201"] = {"class_type": "EmptyImage", "inputs": {
        "width": width, "height": height, "batch_size": batch_size, "color": 65280}}
    graph["202"] = {"class_type": "ImageCompositeMasked", "inputs": {
        "destination": ["201", 0], "source": ["200", 0], "x": 0, "y": 0,
        "resize_source": False, "mask": ["200", 1]}}
    graph[save_node]["inputs"]["images"] = ["202", 0]


def _sheet_prompt(source_name: str | None, view_layout: str, appearance: str = "") -> tuple[str, int, int]:
    if view_layout == "five":
        layout = (
            "exactly five equal-width vertical panels, all showing the complete body at exactly the same scale: "
            "panel 1 a straight front full-body view at 0 degrees, "
            "panel 2 a front three-quarter full-body view at 45 degrees, "
            "panel 3 a strict profile side full-body view at 90 degrees, "
            "panel 4 a rear three-quarter full-body view at 135 degrees, "
            "panel 5 a straight back full-body view at 180 degrees"
        )
        width, height = 2048, 1024
    elif view_layout == "four":
        layout = (
            "exactly four vertical panels: panel 1 a large face close-up, panel 2 a front full-body view, "
            "panel 3 a strict side full-body view, panel 4 a back full-body view"
        )
        width, height = 1536, 1024
    else:
        raise ValueError("角色多视图布局无效")
    mandatory = appearance.strip()[:2200]
    user_rule = (
        "MANDATORY USER CHARACTER REQUIREMENTS — these override every default below: "
        + mandatory + ". "
    ) if mandatory else ""
    source_rule = (
        "Use image 2 only to preserve the person's identity, facial structure and body proportions. "
        "If the mandatory user requirements change clothing, hairstyle or art style, follow them instead. "
        "Image 1 is only a blank output canvas. "
    ) if source_name else ""
    prompt = (
        user_rule + source_rule + "Create one clean professional character reference sheet divided into " + layout + ". "
        "Keep exactly the same person and the user-requested outfit in every panel. Full body panels show "
        "head through shoes, neutral standing pose and empty hands. " +
        "Use the exact requested clothing, "
        "art style, colors and character traits consistently in all panels. "
        "Keep consistent scale and lighting. Follow the requested visual style. "
        "No text, no labels, no letters, no watermark."
    )
    if view_layout == "five":
        prompt += (" Every panel must be a head-to-toe full-body turnaround view; no face close-up, no portrait crop, "
                   "no repeated angle, and no perspective camera variation. Keep feet aligned on one horizontal baseline.")
    return prompt, width, height


def _model_config() -> tuple[str, str, str]:
    names = {}
    for node, field in (("UNETLoader", "unet_name"), ("CLIPLoader", "clip_name"),
                        ("VAELoader", "vae_name")):
        info = api("/object_info/" + node).get(node, {})
        names[node] = info.get("input", {}).get("required", {}).get(field, [[]])[0]
    if "TextEncodeQwenImage21" not in api("/object_info/TextEncodeQwenImage21"):
        raise RuntimeError("ComfyUI 缺少 Qwen-Image 2.1 编码节点")

    def choose(items, preferred, token):
        return next((name for name in preferred if name in items), None) or next(
            (name for name in items if token in name.lower()), None)

    model = choose(names["UNETLoader"], ("qwen_image_2.1_int8_convrot.safetensors",
                                          "qwen_image_2.1_bf16.safetensors"), "qwen_image_2.1")
    clip = choose(names["CLIPLoader"], ("qwen3vl_8b_w4a8.safetensors",
                                           "qwen3vl_8b_int8_convrot.safetensors"), "qwen3vl_8b")
    vae = choose(names["VAELoader"], ("qwen_image_2.1_vae_bf16.safetensors",), "qwen_image_2.1_vae")
    if not all((model, clip, vae)):
        raise RuntimeError("ComfyUI 缺少 Qwen-Image 2.1 模型、视觉编码器或 VAE")
    return model, clip, vae


def build_qwen_graph(source_name: str | None, preset: str, job_id: str,
                     appearance: str = "", view_layout: str = "four") -> dict:
    if preset not in ("fast", "stable"):
        raise ValueError("多视图生成模式无效")
    model, clip, vae = _model_config()
    prompt, width, height = _sheet_prompt(source_name, view_layout, appearance)
    graph = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": model, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": clip, "type": "qwen_image", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": vae}},
        "4": {"class_type": "EmptyImage", "inputs": {"width": width, "height": height,
                                                     "batch_size": 1, "color": 65280}},
        "6": {"class_type": "TextEncodeQwenImage21", "inputs": {
            "clip": ["2", 0],
            "vae": ["3", 0], "prompt": prompt, "negative_prompt": NEGATIVE,
            "resolution": 0}},
        "7": {"class_type": "KSampler", "inputs": {
            "model": ["1", 0], "positive": ["6", 0], "negative": ["6", 1],
            "latent_image": ["6", 2], "seed": secrets.randbits(48),
            "steps": 16 if preset == "fast" else 25, "cfg": 1.0,
            "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["3", 0]}},
        "9": {"class_type": "SaveImage", "inputs": {
            "images": ["8", 0], "filename_prefix": "h3studio_multiview_" + job_id}},
    }
    _attach_green_background(graph, ["8", 0], "9", width, height)
    if source_name:
        graph["5"] = {"class_type": "LoadImage", "inputs": {"image": source_name}}
        graph["6"]["inputs"]["images.image_1"] = ["4", 0]
        graph["6"]["inputs"]["images.image_2"] = ["5", 0]
    else:
        # Pure text conditioning with a correctly shaped Qwen VAE latent.
        graph["10"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["4", 0], "vae": ["3", 0]}}
        graph["7"]["inputs"]["latent_image"] = ["10", 0]
    return graph


def _require_krea2() -> None:
    missing = [name for name in KREA2_NODES if name not in api("/object_info/" + name)]
    if missing:
        raise RuntimeError("ComfyUI 尚未载入 Krea2 四视图节点：" + ", ".join(missing))
    if not (Path(os.environ.get("COMFYUI_DIR", "/home/administrator/ComfyUI")) /
            "models/diffusion_models/Krea2/Krea2-MuseByStable_v15Turbo_fp8.safetensors").is_file():
        raise RuntimeError("找不到 Krea2 Muse 模型，请检查 ComfyUI 模型文件")
    if not (Path(os.environ.get("COMFYUI_DIR", "/home/administrator/ComfyUI")) /
            "models/loras/Krea2-功能/Krea2-四视图QuadView_krea2_v1.safetensors").is_file():
        raise RuntimeError("找不到 Krea2 四视图 LoRA，请检查 ComfyUI 模型文件")


def build_krea2_graph(source_name: str, job_id: str, appearance: str = "",
                      settings: dict | None = None) -> dict:
    """Port the local Krea2_MiniMaxH3 four-view workflow without XB prompt preview."""
    if not source_name:
        raise ValueError("Krea2 四视图需要绑定人物原图")
    _require_krea2()
    options = settings or {}
    try:
        width = int(options.get("krea_width") or 1536)
        height = int(options.get("krea_height") or 1024)
    except (TypeError, ValueError):
        raise ValueError("Krea2 画布宽高必须是数字") from None
    if not (512 <= width <= 2048 and 512 <= height <= 2048
            and width % 32 == 0 and height % 32 == 0):
        raise ValueError("Krea2 画布宽高需为 512–2048 且是 32 的倍数")
    prompt = "Convert the character in the image to a Character Sheet showing a face close-up, front full body, side full body and back full body views"
    if appearance.strip():
        prompt += ". Character appearance requirements: " + appearance.strip()[:1800]
    return {
        "1": {"class_type": "UNETLoader", "inputs": {
            "unet_name": "Krea2/Krea2-MuseByStable_v15Turbo_fp8.safetensors", "weight_dtype": "default"}},
        "2": {"class_type": "LoraLoaderModelOnly", "inputs": {
            "model": ["1", 0], "lora_name": "Krea2-功能/Krea2-四视图QuadView_krea2_v1.safetensors",
            "strength_model": 1.0}},
        "3": {"class_type": "CLIPLoader", "inputs": {
            "clip_name": "qwen3vl_4b_fp8_scaled.safetensors", "type": "krea2", "device": "default"}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_vae.safetensors"}},
        "5": {"class_type": "LoadImage", "inputs": {"image": source_name}},
        "6": {"class_type": "ImageResizeKJv2", "inputs": {
            "image": ["5", 0], "width": 1024, "height": 1536,
            "upscale_method": "lanczos", "keep_proportion": "resize",
            "pad_color": "0, 0, 0", "crop_position": "center", "divisible_by": 2,
            "device": "cpu"}},
        "7": {"class_type": "VAEEncode", "inputs": {"pixels": ["6", 0], "vae": ["4", 0]}},
        "8": {"class_type": "EmptySD3LatentImage", "inputs": {
            "width": width, "height": height, "batch_size": 1}},
        "9": {"class_type": "Krea2EditGroundedEncode", "inputs": {
            "clip": ["3", 0], "prompt": prompt, "image": ["6", 0],
            "grounding_px": 0, "system_prompt": ""}},
        "10": {"class_type": "Krea2EditGroundedEncode", "inputs": {
            "clip": ["3", 0], "prompt": "", "image": ["6", 0],
            "grounding_px": 768, "system_prompt": ""}},
        "11": {"class_type": "Krea2EditModelPatch", "inputs": {
            "model": ["2", 0], "source_latent": ["7", 0], "vae": ["4", 0],
            "source_image": ["6", 0], "ref_boost": 1.0, "ref_boost_a": 1.0, "fit_mode": "fit"}},
        "12": {"class_type": "KSampler", "inputs": {
            "model": ["11", 0], "positive": ["9", 0], "negative": ["10", 0],
            "latent_image": ["8", 0], "seed": secrets.randbits(48), "steps": 10,
            "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
        "13": {"class_type": "VAEDecode", "inputs": {"samples": ["12", 0], "vae": ["4", 0]}},
        "14": {"class_type": "SaveImage", "inputs": {
            "images": ["13", 0], "filename_prefix": "h3studio_krea2_multiview_" + job_id}},
    }


def _require_xb_qwen() -> None:
    if XB_NODE not in api("/object_info/" + XB_NODE):
        raise RuntimeError("ComfyUI 尚未载入 XB_QwenPromptPreset，请重启 ComfyUI 后重试")
    info = api("/object_info/" + XB_NODE).get(XB_NODE, {})
    encoders = info.get("input", {}).get("required", {}).get("clip_name", [[]])[0]
    if XB_TEXT_ENCODER not in encoders:
        raise RuntimeError("缺少 68 号工作流使用的本机 Qwen3.5 提示词增强模型")


def build_xb_qwen_graph(source_name: str | None, preset: str, job_id: str,
                        appearance: str = "", view_layout: str = "four",
                        settings: dict | None = None) -> dict:
    """Reproduce the local Qwen-Image-2.1_68 text/image-to-image workflow."""
    if preset not in ("fast", "stable"):
        raise ValueError("多视图生成模式无效")
    if view_layout not in ("four", "five"):
        raise ValueError("角色多视图布局无效")
    _require_xb_qwen()
    model, clip, vae = _model_config()
    options = settings or {}
    green_background = options.get("green_background", True) not in (False, "false", 0, "0")
    workflow = _xb_i2i_workflow_defaults() if source_name else {}
    config = dict(workflow) if source_name else dict(options)
    # A non-empty role-level preset/extra setting is an intentional override;
    # otherwise preserve the values saved in the user's workflow 69.
    if source_name:
        for key in ("preset_text", "extra_settings"):
            if options.get(key):
                config[key] = options[key]
        if options.get("canvas_override"):
            for key in ("aspect_ratio", "width", "height"):
                if key in options:
                    config[key] = options[key]
    language = "英文 [EN]" if config.get("output_lang") == "en" else "中文 [ZH]"
    sampling_mode = config.get("sampling_mode") if config.get("sampling_mode") in ("bypass", "on", "off") else "bypass"
    skill_mode = config.get("skill_mode") if config.get("skill_mode") in ("自动", "不用") else "自动"
    aspect = config.get("aspect_ratio") if config.get("aspect_ratio") in ("Free", "1:1", "16:9", "9:16", "4:3", "3:4", "21:9") else "16:9"
    fallback_size = 1024 if source_name else 1920
    width = max(32, min(2048, round(int(config.get("width") or fallback_size) / 32) * 32))
    height = max(32, min(2048, round(int(config.get("height") or (1024 if source_name else 1088)) / 32) * 32))
    ratios = {"1:1": 1.0, "16:9": 16 / 9, "9:16": 9 / 16,
              "4:3": 4 / 3, "3:4": 3 / 4, "21:9": 21 / 9}
    ratio = ratios.get(aspect)
    snap = lambda value: max(32, min(2048, int(math.floor(float(value) / 32 + .5)) * 32))
    if ratio:
        if width >= height:
            height = snap(width / ratio)
        else:
            width = snap(height * ratio)
    seed = int(options.get("seed") or config.get("seed") or secrets.randbits(32)) & 0xFFFFFFFFFFFFFFFF
    preset_mode = "人物五视图" if view_layout == "five" else "人物四视图"
    # Pass the character's saved XB preset text through unchanged. When blank,
    # XB_QwenPromptPreset itself resolves its own built-in text for preset_mode.
    saved_preset = options.get("preset_text") or (workflow.get("preset_text") if view_layout == "five" else "")
    custom_preset = str(saved_preset or "").strip()[:4000]
    extra = str(config.get("extra_settings") or "").strip()[:2000]
    body = appearance.strip()[:4000]
    workflow_text = str(workflow.get("internal_prompt") or "").strip()
    if source_name and workflow_text:
        body = (workflow_text + ("；" + body if body else ""))[:4000]
    body = body[:5000]
    extra = extra[:3000]
    graph = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": model, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": clip, "type": "qwen_image", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": vae}},
        "100": {"class_type": XB_NODE, "inputs": {
            "latent_kind": "Qwen-image", "output_lang": language,
            "io_mode": "图生图" if source_name else "文生图",
            "preset_mode": preset_mode, "three_view_text": custom_preset,
            "skill_mode": skill_mode, "skill_name": config.get("skill_name") or "不使用", "aspect_ratio": aspect,
            "width": width, "height": height, "batch_size": int(config.get("batch_size") or 1),
            "clip_name": config.get("clip_name") or XB_TEXT_ENCODER,
            "clip_type": config.get("clip_type") or "qwen_image", "device": config.get("device") or "default",
            "max_length": int(config.get("max_length") or 512), "sampling_mode": sampling_mode,
            "temperature": float(config.get("temperature") or 0.7),
            "top_k": int(config.get("top_k") or 64), "top_p": float(config.get("top_p") or 0.95),
            "min_p": float(config.get("min_p") or 0.05),
            "repetition_penalty": float(config.get("repetition_penalty") or 1.05),
            "presence_penalty": float(config.get("presence_penalty") or 0.0), "seed": seed,
            "thinking": config.get("thinking") in (True, "true", 1, "1"),
            "use_default_template": config.get("use_default_template", True), "mtp": config.get("mtp") or "auto",
            "internal_prompt": body,
            "manager_settings": _xb_manager_settings(config, extra)},},
        "4": {"class_type": "TextEncodeQwenImage21", "inputs": {
            "clip": ["2", 0], "prompt": ["100", 0], "negative_prompt": NEGATIVE,
            "resolution": max(width, height)}},
        "5": {"class_type": "KSampler", "inputs": {
            "model": ["1", 0], "positive": ["4", 0], "negative": ["4", 1],
            "latent_image": ["100", 1], "seed": seed,
            "steps": 16 if preset == "fast" else 25, "cfg": 1.0,
            "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
        "6": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["3", 0]}},
        "7": {"class_type": "SaveImage", "inputs": {
            "images": ["6", 0], "filename_prefix": "h3studio_xb_multiview_" + job_id}},
    }
    if source_name:
        # Match workflow 69: the source portrait is the encoder's first and
        # only reference image. The optional green matte runs after decode.
        graph["9"] = {"class_type": "LoadImage", "inputs": {"image": source_name}}
        reference_image = ["9", 0]
        if options.get("canvas_override"):
            # TextEncodeQwenImage21 creates the sampler's latent from its first
            # reference image, not from XB node 100's empty latent. Its normal
            # resolution setting retains the source image's aspect ratio, so
            # changing only XB width/height cannot alter an i2i output canvas.
            # Pad without stretching/cropping the character and keep the exact
            # requested dimensions through the encoder (resolution=0).
            graph["10"] = {"class_type": "ImageResizeKJv2", "inputs": {
                "image": ["9", 0], "width": width, "height": height,
                "upscale_method": "lanczos", "keep_proportion": "pad",
                "pad_color": "0, 255, 0", "crop_position": "center",
                "divisible_by": 1, "device": "cpu"}}
            reference_image = ["10", 0]
            graph["4"]["inputs"]["resolution"] = 0
        graph["100"]["inputs"]["images"] = reference_image
        graph["4"]["inputs"].update({
            "vae": ["3", 0], "images.image_1": reference_image})
        graph["5"]["inputs"]["latent_image"] = ["4", 2]
        if green_background:
            # The encoder can retain the source image's dimensions. Size the
            # matte from the decoded result so the subject is never cropped.
            graph["199"] = {"class_type": "GetImageSizeAndCount", "inputs": {
                "image": ["6", 0]}}
            _attach_green_background(graph, ["6", 0], "7",
                                     ["199", 1], ["199", 2], ["199", 3])
    elif green_background:
        _attach_green_background(graph, ["6", 0], "7", width, height)
    return graph


def build_six_view_sheet_graph(source_name: str | None, job_id: str,
                               appearance: str = "") -> dict:
    """Use the user's Qwen-Image 2.1 six-view character-sheet workflow."""
    if not source_name:
        raise ValueError("六视图定装照需要先绑定人物原图")
    try:
        workflow = json.loads(SIX_VIEW_WORKFLOW.read_text(encoding="utf-8"))
        nodes = {node["id"]: node for node in workflow["nodes"]}
        template = str(nodes[21]["widgets_values"][0]).strip()
        model_name = str(nodes[50]["widgets_values"][0])
        clip_name = str(nodes[52]["widgets_values"][0])
        vae_name = str(nodes[54]["widgets_values"][0])
        steps = int(nodes[40]["widgets_values"][2])
        resolution = int(nodes[30]["widgets_values"][2])
        scale = nodes[10]["widgets_values"]
    except (OSError, ValueError, KeyError, IndexError, TypeError) as error:
        raise RuntimeError("无法读取本机六视图定装照工作流") from error
    if not template or not 1 <= steps <= 100:
        raise RuntimeError("本机六视图工作流的模板或采样步数无效")
    required_nodes = ("LayerUtility: ImageScaleByAspectRatio V2", "QwenImage21Cache",
                      "TextEncodeQwenImage21")
    for node_name in required_nodes:
        if node_name not in api("/object_info/" + urllib.parse.quote(node_name)):
            raise RuntimeError("ComfyUI 缺少六视图工作流节点：" + node_name)
    prompt = ("优先遵守此角色的外观设定：" + appearance.strip()[:2200] + "。\n" if appearance.strip() else "") + template
    return {
        "1": {"class_type": "LoadImage", "inputs": {"image": source_name}},
        "2": {"class_type": "LayerUtility: ImageScaleByAspectRatio V2", "inputs": {
            "image": ["1", 0], "aspect_ratio": scale[0], "proportional_width": scale[1],
            "proportional_height": scale[2], "fit": scale[3], "method": scale[4],
            "round_to_multiple": scale[5], "scale_to_side": scale[6],
            "scale_to_length": scale[7], "background_color": scale[8]}},
        "3": {"class_type": "UNETLoader", "inputs": {"unet_name": model_name, "weight_dtype": "default"}},
        "4": {"class_type": "QwenImage21Cache", "inputs": {"model": ["3", 0], "device": "auto", "dtype": "default"}},
        "5": {"class_type": "CLIPLoader", "inputs": {"clip_name": clip_name, "type": "qwen_image", "device": "default"}},
        "6": {"class_type": "VAELoader", "inputs": {"vae_name": vae_name}},
        "7": {"class_type": "TextEncodeQwenImage21", "inputs": {
            "clip": ["5", 0], "images.image_1": ["2", 0], "vae": ["6", 0],
            "prompt": prompt, "negative_prompt": "", "resolution": resolution}},
        "8": {"class_type": "EmptyLatentImage", "inputs": {"width": 1536, "height": 2048, "batch_size": 1}},
        "9": {"class_type": "KSampler", "inputs": {
            "model": ["4", 0], "positive": ["7", 0], "negative": ["7", 1],
            "latent_image": ["8", 0], "seed": secrets.randbits(48), "steps": steps,
            "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
        "10": {"class_type": "VAEDecode", "inputs": {"samples": ["9", 0], "vae": ["6", 0]}},
        "11": {"class_type": "SaveImage", "inputs": {
            "images": ["10", 0], "filename_prefix": "h3studio_six_view_" + job_id}},
    }


def build_graph(source_name: str | None, preset: str, job_id: str,
                appearance: str = "", view_layout: str = "four",
                engine: str = "qwen", settings: dict | None = None) -> dict:
    if engine == "xb_qwen":
        return build_xb_qwen_graph(source_name, preset, job_id, appearance, view_layout, settings)
    if engine == "qwen_sheet":
        if view_layout != "six":
            raise ValueError("定装照工作流只支持六视图")
        return build_six_view_sheet_graph(source_name, job_id, appearance)
    if engine == "krea2":
        if view_layout != "four":
            raise ValueError("Krea2 辅助工作流只支持四视图")
        return build_krea2_graph(source_name or "", job_id, appearance, settings)
    if engine != "qwen":
        raise ValueError("多视图生成引擎无效")
    return build_qwen_graph(source_name, preset, job_id, appearance, view_layout)


def _output_paths(history: dict, node_id: str) -> list[Path]:
    paths = []
    for image in history.get("outputs", {}).get(node_id, {}).get("images", []):
        if image.get("type") != "output" or Path(str(image.get("filename", ""))).suffix.lower() not in (".png", ".webp"):
            continue
        relative = Path(str(image.get("subfolder") or "")) / str(image["filename"])
        output = (COMFY_OUTPUT / relative).resolve()
        if output.is_relative_to(COMFY_OUTPUT.resolve()) and output.is_file():
            paths.append(output)
    return paths


def wait_for_image(prompt_id: str, job_id: str) -> dict:
    failures = 0
    while True:
        if read()["runs"][job_id].get("cancel_requested"):
            try:
                api("/api/jobs/" + urllib.parse.quote(prompt_id, safe="") + "/cancel", {})
            except (OSError, RuntimeError):
                try:
                    api("/queue", {"delete": [prompt_id]})
                except OSError:
                    pass
            raise RuntimeError("用户取消生图任务")
        try:
            history = api("/history/" + prompt_id).get(prompt_id)
            failures = 0
        except (OSError, urllib.error.URLError, TimeoutError) as error:
            failures += 1
            if failures >= 12:
                raise RuntimeError("ComfyUI 连续多次无响应：" + str(error)) from error
            time.sleep(3)
            continue
        if history:
            status = history.get("status", {})
            if status.get("status_str") == "error":
                raise RuntimeError(str(status.get("messages", []))[-2000:])
            return history
        time.sleep(3)


def generate(job_id: str) -> None:
    job = read()["runs"][job_id]
    source_input = None
    submitted = False
    completed = False
    try:
        if job.get("source_asset"):
            source = read()["assets"][job["source_asset"]]
            source_path = ASSETS / source["file"]
            if not source_path.is_file():
                raise FileNotFoundError("角色外观原图已不存在")
            COMFY_INPUT.mkdir(parents=True, exist_ok=True)
            source_input = COMFY_INPUT / ("h3studio_multiview_" + job_id + source_path.suffix.lower())
            shutil.copy2(source_path, source_input)
        view_layout = job.get("view_layout") if job.get("view_layout") in ("four", "five", "six") else "four"
        view_label = {"four": "四视图", "five": "五视图", "six": "六视图"}[view_layout]
        engine = job.get("engine") if job.get("engine") in ("xb_qwen", "krea2", "qwen_sheet") else "qwen"
        engine_label = ({"xb_qwen": "XB Qwen 69 图生图" if source_input else "XB Qwen 68 文生图",
                         "krea2": "Krea2 辅助四视图",
                         "qwen_sheet": "Qwen 2.1 定装照",
                         "qwen": "Qwen2.1"}[engine])
        graph = build_graph(source_input.name if source_input else None, job["preset"],
                            job_id, str(job.get("appearance") or ""), view_layout, engine,
                            job.get("xb_settings") if isinstance(job.get("xb_settings"), dict) else None)
        if read()["runs"][job_id].get("cancel_requested"):
            raise RuntimeError("用户取消多视图任务")
        submitted = True
        prompt_id = _submit(graph, job_id)
        update_run(job_id, status="running", prompt_id=prompt_id,
                   current=f"{engine_label} {view_label}生成中")
        history = wait_for_image(prompt_id, job_id)
        outputs = _output_paths(history, {"xb_qwen": "7", "krea2": "14", "qwen_sheet": "11"}.get(engine, "9"))
        if not outputs:
            raise RuntimeError(f"{engine_label} 工作流没有输出{view_label}图片")
        state = read()
        project = state["projects"].get(job["project_id"])
        character = next((item for item in project.get("characters", []) if item.get("id") == job["character_id"]), None) if project else None
        if not character:
            raise RuntimeError(f"项目或角色已删除，{view_label}没有绑定")
        asset_id = uid()
        destination = ASSETS / (asset_id + ".png")
        shutil.copy2(outputs[0], destination)
        asset = {"id": asset_id, "name": str(character.get("name") or "角色") + " · " + view_label,
                 "file": destination.name, "kind": "image", "size": destination.stat().st_size,
                 "created_at": time.time(), "project_id": job["project_id"],
                 "generated_by": {"xb_qwen": "xb_qwen69_multiview",
                                   "krea2": "krea2_aux_multiview",
                                   "qwen_sheet": "qwen_image_2.1_six_view_sheet"}.get(engine, "qwen_image_2.1_multiview"),
                 "generation_mode": job.get("mode", "i2i"), "multiview_engine": engine,
                 "view_layout": view_layout}

        def finish(data):
            current_job = data["runs"].get(job_id)
            current_project = data["projects"].get(job["project_id"])
            current_character = next((item for item in current_project.get("characters", [])
                                      if item.get("id") == job["character_id"]), None) if current_project else None
            if not current_job or current_job.get("cancel_requested") or not current_character:
                return False
            data["assets"][asset_id] = asset
            current_project.setdefault("image_descriptions", {})[asset_id] = (
                f"{current_character.get('name') or '角色'}的{view_label}：同一人物的"
                + ({"five": "正面、前侧四分之三、侧面、后侧四分之三和背面全身；",
                    "six": "上排正面、45°侧面、背面头部；下排正面、45°侧面、背面全身；"}
                   .get(view_layout, "脸部特写、正面、侧面和背面；"))
                + "用于稳定脸、身形与服装，不作为视频分栏画面。")
            current_project["_data_version"] = int(current_project.get("_data_version", 0)) + 1
            if not job.get("source_asset") or current_character.get("portrait_asset") == job["source_asset"]:
                current_character["multiview_asset"] = asset_id
                current_job["bound"] = True
            else:
                current_job["bound"] = False
            current_job.update(status="completed", asset_id=asset_id, finished_at=time.time(),
                               current=f"{view_label}已生成")
            return True

        if not change(finish):
            destination.unlink(missing_ok=True)
            raise RuntimeError(f"角色已变化，{view_label}生成结果未保存")
        completed = True
    except Exception as error:
        update_run(job_id, status="cancelled" if read()["runs"][job_id].get("cancel_requested") else "error",
                   error=str(error), finished_at=time.time())
    finally:
        if source_input and (completed or not submitted):
            source_input.unlink(missing_ok=True)

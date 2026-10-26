"""用本地 Qwen-Image 2.1 生成一张参考图（t2i / i2i），供导演台绑定身份参考。

用法：
    python3 work_qwen_ref.py --out /tmp/face.png --source /path/to.jpg --prompt "..." [--size 1024] [--steps 28]
"""
from __future__ import annotations

import argparse
import json
import secrets
import shutil
import time
import urllib.request
from pathlib import Path

COMFY = "http://127.0.0.1:8188"
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
NEGATIVE = ("text, letters, labels, watermark, collage, split screen, multi panel, grid, blurry, "
            "deformed, distorted face, extra limbs, extra fingers, cartoon, 3d render, cgi, "
            "plaid jacket, checkered jacket, skull, skull print, graphic print, logo, text on shirt")


def api(path: str, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    headers = {"Content-Type": "application/json"} if data else {}
    request = urllib.request.Request(COMFY + path, data=data, headers=headers)
    with OPENER.open(request, timeout=60) as response:
        return json.loads(response.read())


def model_config():
    names = {}
    for node, field in (("UNETLoader", "unet_name"), ("CLIPLoader", "clip_name"), ("VAELoader", "vae_name")):
        info = api("/object_info/" + node).get(node, {})
        names[node] = info.get("input", {}).get("required", {}).get(field, [[]])[0]

    def choose(items, preferred, token):
        return next((name for name in preferred if name in items), None) or next(
            (name for name in items if token in name.lower()), None)

    return (choose(names["UNETLoader"], ("qwen_image_2.1_int8_convrot.safetensors",
                                         "qwen_image_2.1_bf16.safetensors"), "qwen_image_2.1"),
            choose(names["CLIPLoader"], ("qwen3vl_8b_w4a8.safetensors",
                                        "qwen3vl_8b_int8_convrot.safetensors"), "qwen3vl_8b"),
            choose(names["VAELoader"], ("qwen_image_2.1_vae_bf16.safetensors",), "qwen_image_2.1_vae"))


def build(prompt: str, source_name: str | None, size: int, steps: int, prefix: str):
    unet, clip, vae = model_config()
    graph = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": unet, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": clip, "type": "qwen_image", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": vae}},
        "4": {"class_type": "EmptyImage", "inputs": {"width": size, "height": size,
                                                     "batch_size": 1, "color": 15132390}},
        "6": {"class_type": "TextEncodeQwenImage21", "inputs": {
            "clip": ["2", 0], "vae": ["3", 0], "prompt": prompt,
            "negative_prompt": NEGATIVE, "resolution": 0}},
        "7": {"class_type": "KSampler", "inputs": {
            "model": ["1", 0], "positive": ["6", 0], "negative": ["6", 1],
            "latent_image": ["6", 2], "seed": secrets.randbits(48),
            "steps": steps, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["3", 0]}},
        "9": {"class_type": "SaveImage", "inputs": {
            "images": ["8", 0], "filename_prefix": prefix}},
    }
    if source_name:
        graph["5"] = {"class_type": "LoadImage", "inputs": {"image": source_name}}
        graph["6"]["inputs"]["images.image_1"] = ["4", 0]
        graph["6"]["inputs"]["images.image_2"] = ["5", 0]
    else:
        graph["10"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["4", 0], "vae": ["3", 0]}}
        graph["7"]["inputs"]["latent_image"] = ["10", 0]
    return graph


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--source")
    parser.add_argument("--out", required=True)
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=28)
    args = parser.parse_args()

    input_dir = Path("/home/administrator/ComfyUI/input")
    output_dir = Path("/home/administrator/ComfyUI/output")
    source_name = None
    if args.source:
        input_dir.mkdir(parents=True, exist_ok=True)
        target = input_dir / ("work_ref_" + Path(args.source).name)
        shutil.copy2(args.source, target)
        source_name = target.name

    graph = build(args.prompt, source_name, args.size, args.steps, "work_ref")
    prompt_id = api("/prompt", {"prompt": graph})["prompt_id"]
    print("prompt", prompt_id, flush=True)
    for _ in range(200):
        history = api("/history/" + prompt_id)
        if prompt_id in history:
            images = history[prompt_id].get("outputs", {}).get("9", {}).get("images", [])
            if not images:
                raise SystemExit("没有输出图片")
            source = output_dir / images[0]["filename"]
            shutil.copy2(source, args.out)
            print("saved", args.out)
            return
        time.sleep(3)
    raise SystemExit("超时")


if __name__ == "__main__":
    main()

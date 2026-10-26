"""静态入口、健康检查、模型清单与存储状态。"""
from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.request

from server.router import Route

from runner import COMFY_URL
from runner import api
from server.config import COMFY_DIR
from server.config import ROOT
from store import storage_status


SETTINGS_HUB_ASSETS = {
    "/ui/settings-hub.js": "settings-hub.js",
    "/ui/settings-hub.css": "settings-hub.css",
    "/ui/help.js": "help.js",
}

def get_comfy_ws(req, path, parts, url):
    """原 app.py 分支：`path == "/comfy-ws"`。"""
    req.proxy_comfy_websocket(url.query)
    return


def get_index(req, path, parts, url):
    """原 app.py 分支：`path in ("/", "/index.html")`。"""
    req.file(ROOT / "static" / "index.html")
    return


def get_settings_hub_asset(req, path, parts, url):
    """Serve only the explicit static assets used by the legacy workbench."""
    req.file(ROOT / "static" / "ui" / SETTINGS_HUB_ASSETS[path])


def get_api_version(req, path, parts, url):
    """原 app.py 分支：`path == "/api/version"`。"""
    req.json(200, {"version": "independent-93"})
    return


def get_api_storage_status(req, path, parts, url):
    """原 app.py 分支：`path == "/api/storage/status"`。"""
    req.json(200, storage_status())
    return


def get_api_health(req, path, parts, url):
    """原 app.py 分支：`path == "/api/health"`。"""
    try:
        stats = api("/system_stats")
        try:
            ops_loaded = ("H3StudioSelfLift" in api("/object_info/H3StudioSelfLift")
                          and "H3StudioLatentContinue" in api("/object_info/H3StudioLatentContinue")
                          and "H3StudioContinueRemask" in api("/object_info/H3StudioContinueRemask"))
        except Exception:
            ops_loaded = False
        req.json(200, {"connected": True, "comfy_url": COMFY_URL,
                        "device": (stats.get("devices") or [{}])[0].get("name", ""),
                        "ops_loaded": ops_loaded})
    except Exception as error:
        req.json(200, {"connected": False, "comfy_url": COMFY_URL, "error": str(error)})
    return


def get_api_hardware(req, path, parts, url):
    """Compact ComfyUI and host hardware snapshot for the workbench header."""
    result = {"updated_at": time.time(), "comfy_connected": False, "gpus": []}
    try:
        memory = {}
        with open("/proc/meminfo", encoding="ascii") as source:
            for line in source:
                key, _, value = line.partition(":")
                if key in ("MemTotal", "MemAvailable"):
                    memory[key] = int(value.strip().split()[0]) * 1024
        result["wsl_ram_total"] = memory["MemTotal"]
        result["wsl_ram_available"] = memory["MemAvailable"]
    except (OSError, KeyError, ValueError, IndexError):
        pass
    try:
        with urllib.request.urlopen(COMFY_URL + "/system_stats", timeout=3) as response:
            stats = json.load(response)
        result["comfy_connected"] = True
        result["ram_total"] = stats.get("system", {}).get("ram_total")
        result["ram_free"] = stats.get("system", {}).get("ram_free")
        result["comfy_devices"] = [{key: device.get(key) for key in
                                    ("name", "index", "vram_total", "vram_free")}
                                   for device in stats.get("devices", [])]
    except Exception as error:
        result["comfy_error"] = str(error)
    try:
        with urllib.request.urlopen(COMFY_URL + "/queue", timeout=3) as response:
            queue = json.load(response)
        result["queue_running"] = len(queue.get("queue_running", []))
        result["queue_pending"] = len(queue.get("queue_pending", []))
    except Exception:
        pass
    try:
        query = "index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw"
        probe = subprocess.run(["nvidia-smi", "--query-gpu=" + query,
                                "--format=csv,noheader,nounits"], capture_output=True,
                               text=True, timeout=3, check=True)
        for row in probe.stdout.splitlines():
            fields = [field.strip() for field in row.split(",")]
            if len(fields) != 7:
                continue
            def number(value):
                try:
                    return float(value)
                except ValueError:
                    return None
            result["gpus"].append({"index": int(fields[0]), "name": fields[1],
                                   "utilization": number(fields[2]),
                                   "memory_used_mib": number(fields[3]),
                                   "memory_total_mib": number(fields[4]),
                                   "temperature_c": number(fields[5]),
                                   "power_w": number(fields[6])})
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
    try:
        result["load_1m"] = round(os.getloadavg()[0], 2)
    except OSError:
        pass
    req.json(200, result)


def get_api_models(req, path, parts, url):
    """原 app.py 分支：`path == "/api/models"`。"""
    def names(folder):
        root = COMFY_DIR / "models" / folder
        return sorted(p.name for p in root.iterdir() if p.is_file() and p.suffix.lower() in (".safetensors", ".pt", ".pth")) if root.is_dir() else []
    def relative_names(folder, suffixes=(".safetensors", ".pt", ".pth")):
        root = COMFY_DIR / "models" / folder
        return sorted(p.relative_to(root).as_posix() for p in root.rglob("*")
                      if p.is_file() and p.suffix.lower() in suffixes) if root.is_dir() else []
    try:
        sage_available = "MiniMaxH3MemoryEfficientSageAttentionPatch" in api("/object_info/MiniMaxH3MemoryEfficientSageAttentionPatch")
    except Exception:
        sage_available = False
    try:
        veda_node_available = "VedaSparseAttention" in api("/object_info/VedaSparseAttention")
    except Exception:
        veda_node_available = False
    h3_clips = sorted(set(
        [name for name in names("text_encoders") if "minimax_h3" in name] +
        [name for name in names("clip") if "minimax_h3" in name]
    ))
    req.json(200, {"semantic_adapters": names("semantic_bridge"),
                    "latent_models": names("latent_upscale_models"),
                    "h3_unet_models": [name for name in relative_names("diffusion_models") if "minimax_h3" in name.lower()],
                    "h3_lora_models": [name for name in relative_names("loras") if "minimax_h3" in name.lower()],
                    "h3_video_vaes": [name for name in names("vae") if name.startswith("minimax_h3_video_vae")],
                    "h3_audio_vaes": [name for name in names("vae") if name.startswith("minimax_h3_audio_vae")],
                    "h3_clip_models": h3_clips,
                    "seedvr2_models": relative_names("SEEDVR2", (".safetensors", ".pt", ".pth", ".gguf")),
                    "sage_attention": sage_available,
                    "veda_attention": veda_node_available and "minimax_h3_t2va_veda_8nfe_600step_preview_fp8.safetensors" in names("veda")})
    return


ROUTES = [
    Route('GET', 'get_comfy_ws',
         lambda path, parts, url: (path == "/comfy-ws"),
         get_comfy_ws),
    Route('GET', 'get_index',
         lambda path, parts, url: (path in ("/", "/index.html")),
         get_index),
    Route('GET', 'get_settings_hub_asset',
         lambda path, parts, url: path in SETTINGS_HUB_ASSETS,
         get_settings_hub_asset),
    Route('GET', 'get_api_version',
         lambda path, parts, url: (path == "/api/version"),
         get_api_version),
    Route('GET', 'get_api_storage_status',
         lambda path, parts, url: (path == "/api/storage/status"),
         get_api_storage_status),
    Route('GET', 'get_api_health',
         lambda path, parts, url: (path == "/api/health"),
         get_api_health),
    Route('GET', 'get_api_hardware',
         lambda path, parts, url: (path == "/api/hardware"),
         get_api_hardware),
    Route('GET', 'get_api_models',
         lambda path, parts, url: (path == "/api/models"),
         get_api_models),
]

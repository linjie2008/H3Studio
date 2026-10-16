"""进程配置：路径、监听地址与文件类型表。"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COMFY_DIR = Path(os.environ.get("COMFYUI_DIR", "/home/administrator/ComfyUI"))
HOST = os.environ.get("H3_STUDIO_HOST", "127.0.0.1")
PORT = int(os.environ.get("H3_STUDIO_PORT", "8765"))
EXTENSIONS = {
    ".png": "image", ".jpg": "image", ".jpeg": "image", ".webp": "image",
    ".bmp": "image", ".gif": "image", ".tif": "image", ".tiff": "image",
    ".mp4": "video", ".mov": "video", ".webm": "video", ".mkv": "video",
    ".avi": "video", ".m4v": "video",
    ".wav": "audio", ".mp3": "audio", ".flac": "audio", ".m4a": "audio",
    ".ogg": "audio", ".aac": "audio", ".wma": "audio",
}

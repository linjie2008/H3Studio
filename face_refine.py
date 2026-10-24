"""Robust, flicker-free face-region detail recovery for rendered H3 shots.

Upgrades:
1. Every-frame detection with Exponential Moving Average (EMA) trajectory smoothing.
2. Uses GPU execution for InsightFace to prevent frame drops/delays.
3. Hysteresis detection and bounding box temporal damping to eliminate 8Hz strobe/flicker.
4. Smooth luminance contrast blending and sub-pixel edge feathering.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Tuple

import numpy as np

try:
    import cv2
except ImportError:
    raise SystemExit("face_refine 需要 OpenCV（cv2）")

# This is a background finishing pass.  Keep CPU headroom for the desktop and
# ComfyUI while processing high-resolution video frames.
cv2.setNumThreads(2)


ROI_EXPAND = 0.25
CLAHE_CLIP = 1.3
CLAHE_BLEND = 0.25
UNSHARP_RADIUS = 1.0
UNSHARP_AMOUNT = 0.75
GRAIN_SIGMA = 0.0

# Real ESRGAN
ESRGAN_MODELS = {
    "ultrasharp": "/home/administrator/ComfyUI/models/upscale_models/4x-UltraSharp.pth",
    "realesrgan": "/home/administrator/ComfyUI/models/upscale_models/RealESRGAN_x4plus.pth",
}
ESRGAN_MODEL = os.environ.get("H3_STUDIO_FACE_ESRGAN", "ultrasharp")
ESRGAN_ENABLED = os.environ.get("H3_STUDIO_FACE_USE_ESRGAN", "0") == "1"
_ESRGAN = {"model": None, "device": None}


def _load_esrgan():
    if not ESRGAN_ENABLED:
        return None
    if _ESRGAN["model"] is not None:
        return _ESRGAN["model"]
    path = ESRGAN_MODELS.get(ESRGAN_MODEL, ESRGAN_MODEL)
    if not os.path.isfile(path):
        print(f"[face_refine] 找不到超分模型：{path}", flush=True)
        return None
    try:
        sys.path.insert(0, "/home/administrator/ComfyUI")
        import comfy.utils
        import torch
        torch.set_num_threads(2)
        from spandrel import ModelLoader
        descriptor = ModelLoader().load_from_state_dict(
            comfy.utils.load_torch_file(path, safe_load=True))
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = descriptor.model.to(device).eval()
        _ESRGAN["model"] = (model, torch, device)
        print(f"[face_refine] 超分模型已加载：{Path(path).name} ({device})", flush=True)
        return _ESRGAN["model"]
    except Exception as exc:
        print(f"[face_refine] 超分模型加载失败：{exc}", flush=True)
        return None


def _esrgan_face(face_rgb: np.ndarray) -> np.ndarray:
    loaded = _load_esrgan()
    if loaded is None:
        return face_rgb
    try:
        model, torch, device = loaded
        h, w = face_rgb.shape[:2]
        tensor = torch.from_numpy(face_rgb.copy()).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
        with torch.no_grad():
            upscaled = model(tensor)
        out = (upscaled[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255.0).round().astype(np.uint8)
        if (out.shape[1], out.shape[0]) != (w, h):
            out = cv2.resize(out, (w, h), interpolation=cv2.INTER_AREA)
        return out
    except Exception as exc:
        print(f"[face_refine] 超分推理失败：{exc}", flush=True)
        return face_rgb


class TemporalFaceTracker:
    """Detect every face and smooth each box across adjacent frames."""

    def __init__(self) -> None:
        self.app = None
        self.yolo = None
        self.haar = None
        self.backend = "none"

        cascade_names = [
            Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml",
            Path(__file__).resolve().parent.parent / "cascades" / "haarcascade_frontalface_default.xml",
        ]
        for cascade_path in cascade_names:
            if cascade_path.is_file():
                cascade = cv2.CascadeClassifier(str(cascade_path))
                if not cascade.empty():
                    self.haar = cascade
                    break

        # 1. Try Ultralytics face_yolov8m (same as Director node)
        yolo_path = "/home/administrator/ComfyUI/models/ultralytics/bbox/face_yolov8m.pt"
        if os.path.isfile(yolo_path):
            try:
                from ultralytics import YOLO
                self.yolo = YOLO(yolo_path)
                self.backend = "yolov8_face"
            except Exception as e:
                self.yolo = None

        # 2. Try InsightFace on CUDA
        if self.yolo is None:
            try:
                from insightface.app import FaceAnalysis
                app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
                app.prepare(ctx_id=0, det_size=(640, 640))
                self.app = app
                self.backend = "insightface_cuda"
            except Exception as exc:
                try:
                    from insightface.app import FaceAnalysis
                    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
                    app.prepare(ctx_id=-1, det_size=(640, 640))
                    self.app = app
                    self.backend = "insightface_cpu"
                except Exception:
                    self.app = None

        self.tracks: list[tuple[np.ndarray, int]] = []
        self.alpha = 0.35

    def detect_raw(self, frame_bgr: np.ndarray) -> list[Tuple[int, int, int, int]]:

        # Method 1: YOLO
        if self.yolo is not None:
            try:
                # A larger inference canvas preserves faces occupying only a
                # small part of a wide shot. Keep every detection, not just one.
                results = self.yolo(frame_bgr, verbose=False, conf=0.25, imgsz=1280)
                boxes = []
                for r in results:
                    for b in r.boxes.xyxy.cpu().numpy():
                        x1, y1, x2, y2 = b[:4]
                        boxes.append((int(x1), int(y1), int(x2 - x1), int(y2 - y1)))
                return boxes
            except Exception as exc:
                print(f"[face_refine] YOLO 检测失败：{exc}", flush=True)

        # Method 2: InsightFace
        if self.app is not None:
            try:
                faces = self.app.get(frame_bgr)
                boxes = []
                for face in faces:
                    x1, y1, x2, y2 = [int(v) for v in face.bbox]
                    boxes.append((x1, y1, max(1, x2 - x1), max(1, y2 - y1)))
                return boxes
            except Exception as exc:
                print(f"[face_refine] InsightFace 检测失败：{exc}", flush=True)

        # Method 3: OpenCV Haar fallback
        try:
            if self.haar is None:
                return []
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            boxes = self.haar.detectMultiScale(gray, scaleFactor=1.12, minNeighbors=5, minSize=(24, 24))
            if len(boxes) > 0:
                return [(int(x), int(y), int(bw), int(bh)) for x, y, bw, bh in boxes]
        except Exception:
            pass

        return []

    def update(self, frame_rgb: np.ndarray) -> list[Tuple[int, int, int, int]]:
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        detections = self.detect_raw(frame_bgr)
        remaining = set(range(len(self.tracks)))
        updated: list[tuple[np.ndarray, int]] = []
        for box in detections:
            cur = np.array(box, dtype=np.float32)
            center = cur[:2] + cur[2:] / 2
            candidates = [(float(np.linalg.norm(center - (self.tracks[i][0][:2] + self.tracks[i][0][2:] / 2))), i)
                          for i in remaining]
            nearest = min(candidates, default=(float("inf"), -1))
            if nearest[0] < max(cur[2], cur[3]) * 0.8:
                old = self.tracks[nearest[1]][0]
                cur = self.alpha * cur + (1 - self.alpha) * old
                remaining.remove(nearest[1])
            updated.append((cur, 0))
        # Bridge brief detector misses without painting over an empty shot.
        updated.extend((self.tracks[i][0], self.tracks[i][1] + 1)
                       for i in remaining if self.tracks[i][1] < 2)
        self.tracks = updated
        return [tuple(int(v) for v in box) for box, _ in updated]


def _expand(box: Tuple[int, int, int, int], width: int, height: int) -> Tuple[int, int, int, int]:
    x, y, w, h = box
    pad_x, pad_y = int(w * ROI_EXPAND), int(h * ROI_EXPAND)
    x0 = max(0, x - pad_x)
    y0 = max(0, y - pad_y)
    x1 = min(width, x + w + pad_x)
    y1 = min(height, y + h + pad_y)
    if x1 - x0 < 20 or y1 - y0 < 20:
        return (0, 0, 0, 0)
    return (x0, y0, x1 - x0, y1 - y0)


def _feather_mask(width: int, height: int, ratio: float = 0.35) -> np.ndarray:
    fx = max(2, int(width * ratio))
    fy = max(2, int(height * ratio))
    ramp_x = np.clip(np.minimum(np.arange(width), np.arange(width)[::-1]) / fx, 0.0, 1.0)
    ramp_y = np.clip(np.minimum(np.arange(height), np.arange(height)[::-1]) / fy, 0.0, 1.0)
    mask = ramp_x[None, :] * ramp_y[:, None]
    return mask * mask * (3.0 - 2.0 * mask)


def _enhance_roi(frame: np.ndarray, box: Tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = box
    pad = 16
    px0, py0 = max(0, x - pad), max(0, y - pad)
    px1, py1 = min(frame.shape[1], x + w + pad), min(frame.shape[0], y + h + pad)

    patch = frame[py0:py1, px0:px1]
    if patch.size == 0:
        return frame

    gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=(8, 8))
    lifted = clahe.apply(gray)
    blurred = cv2.GaussianBlur(lifted, (0, 0), UNSHARP_RADIUS)
    detail_full = (lifted.astype(np.float32) - blurred.astype(np.float32)) * UNSHARP_AMOUNT + \
                  (lifted.astype(np.float32) - gray.astype(np.float32)) * CLAHE_BLEND
    detail = detail_full[y - py0:y - py0 + h, x - px0:x - px0 + w]

    roi = frame[y:y + h, x:x + w]
    if roi.size == 0:
        return frame

    restored = _esrgan_face(roi)
    lab = cv2.cvtColor(restored, cv2.COLOR_RGB2LAB)
    luminance, chroma_a, chroma_b = cv2.split(lab)
    boosted = np.clip(luminance.astype(np.float32) + detail, 0, 255).astype(np.uint8)
    out = cv2.cvtColor(cv2.merge((boosted, chroma_a, chroma_b)), cv2.COLOR_LAB2RGB)
    alpha = _feather_mask(w, h)[..., None]
    frame[y:y + h, x:x + w] = np.clip(
        roi.astype(np.float32) * (1 - alpha) + out.astype(np.float32) * alpha, 0, 255).astype(np.uint8)
    return frame


def _video_meta(path: Path) -> Tuple[int, int, float]:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,avg_frame_rate", "-of", "json", str(path)],
        capture_output=True, text=True, check=True, timeout=60)
    stream = (json.loads(result.stdout).get("streams") or [{}])[0]
    width, height = int(stream.get("width", 0)), int(stream.get("height", 0))
    rate = str(stream.get("avg_frame_rate") or "24/1")
    try:
        num, den = rate.split("/")
        fps = float(num) / float(den or 1)
    except ValueError:
        fps = 24.0
    return width, height, fps


def refine(input_path: Path, output_path: Path) -> None:
    width, height, fps = _video_meta(input_path)
    tracker = TemporalFaceTracker()
    print(f"[face_refine] 使用检测追踪器: {tracker.backend} (EMA平滑防闪烁已启用)")

    frame_bytes = width * height * 3
    decode = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-threads", "2", "-i", str(input_path),
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    temp = output_path.with_suffix(".tmp.mp4")
    encode = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-threads", "2", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{width}x{height}", "-r", f"{fps:.6f}", "-i", "-",
         "-c:v", "libx264", "-threads", "2", "-preset", "fast", "-crf", "16",
         "-pix_fmt", "yuv420p", str(temp)],
        stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    touched = 0
    index = 0
    try:
        assert decode.stdout is not None and encode.stdin is not None
        while True:
            raw = decode.stdout.read(frame_bytes)
            if len(raw) < frame_bytes:
                break
            frame = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3).copy()

            # Track each face independently; a group shot must not refine only
            # the largest face.
            for box in tracker.update(frame):
                region = _expand(box, width, height)
                if region[2] > 0 and region[3] > 0:
                    frame = _enhance_roi(frame, region)
                    touched += 1

            if GRAIN_SIGMA > 0:
                noise = np.random.default_rng(index).normal(0.0, GRAIN_SIGMA, frame.shape)
                frame = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)

            encode.stdin.write(frame.tobytes())
            index += 1
    finally:
        if encode.stdin is not None:
            encode.stdin.close()
        decode_error = decode.stderr.read().decode(errors="replace") if decode.stderr else ""
        encode_error = encode.stderr.read().decode(errors="replace") if encode.stderr else ""
        decode.wait()
        encode.wait()

    if decode.returncode != 0 or encode.returncode != 0:
        temp.unlink(missing_ok=True)
        raise RuntimeError(f"人脸处理视频编解码失败：{decode_error or encode_error}")

    if touched == 0:
        print("[face_refine] 未检测到人脸，保留原视频", flush=True)
        shutil.copy2(input_path, output_path)
        temp.unlink(missing_ok=True)
        return

    # Mux original audio
    mux = subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(temp), "-i", str(input_path),
         "-map", "0:v:0", "-map", "1:a:0?", "-c:v", "copy", "-c:a", "copy",
         "-movflags", "+faststart", str(output_path)],
        capture_output=True, text=True, timeout=120)
    if mux.returncode != 0:
        temp.unlink(missing_ok=True)
        raise RuntimeError(f"人脸处理音轨合并失败：{mux.stderr}")
    temp.unlink(missing_ok=True)

    print(f"[face_refine] 成功处理 {touched}/{index} 帧，输出已平滑保存: {output_path.name}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法：face_refine.py input.mp4 output.mp4", file=sys.stderr)
        sys.exit(1)
    if hasattr(os, "nice"):
        try:
            os.nice(10)
        except OSError:
            pass
    refine(Path(sys.argv[1]), Path(sys.argv[2]))

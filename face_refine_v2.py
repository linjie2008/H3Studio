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
from typing import Tuple, List, Optional

import numpy as np

try:
    import cv2
except ImportError:
    raise SystemExit("face_refine 需要 OpenCV（cv2）")


ROI_EXPAND = 0.25
CLAHE_CLIP = 1.3
CLAHE_BLEND = 0.25
UNSHARP_RADIUS = 1.0
UNSHARP_AMOUNT = 0.75
GRAIN_SIGMA = 0.20

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
        return None
    try:
        from comfy_extras.nodes_upscale_model import ImageUpscaleWithModel
        import comfy.model_management as mm
        from comfy_extras.nodes_upscale_model import UpscaleModelLoader
        device = mm.get_torch_device()
        model, = UpscaleModelLoader().load_model(path)
        _ESRGAN["model"] = model
        _ESRGAN["device"] = device
        return model
    except Exception:
        return None


def _esrgan_face(face_bgr: np.ndarray) -> np.ndarray:
    model = _load_esrgan()
    if model is None:
        return face_bgr
    try:
        import torch
        from comfy_extras.nodes_upscale_model import ImageUpscaleWithModel
        h, w = face_bgr.shape[:2]
        tensor = torch.from_numpy(face_bgr).float() / 255.0
        tensor = tensor.unsqueeze(0)
        upscaled, = ImageUpscaleWithModel().upscale(model, tensor)
        out = (upscaled[0].clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)
        if (out.shape[1], out.shape[0]) != (w, h):
            out = cv2.resize(out, (w, h), interpolation=cv2.INTER_AREA)
        return out
    except Exception:
        return face_bgr


class TemporalFaceTracker:
    """Per-frame InsightFace tracker with YOLO fallback and EMA temporal smoothing."""

    def __init__(self) -> None:
        self.app = None
        self.yolo = None
        self.backend = "none"
        
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

        # State for temporal smoothing (x, y, w, h)
        self.smooth_box: Optional[np.ndarray] = None
        self.missed_frames = 0
        self.alpha = 0.35  # EMA smoothing factor (0.0 = frozen, 1.0 = raw)

    def detect_raw(self, frame_bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        h, w = frame_bgr.shape[:2]
        
        # Method 1: YOLO
        if self.yolo is not None:
            try:
                results = self.yolo(frame_bgr, verbose=False, conf=0.35)
                boxes = []
                for r in results:
                    for b in r.boxes.xyxy.cpu().numpy():
                        x1, y1, x2, y2 = b[:4]
                        boxes.append((int(x1), int(y1), int(x2 - x1), int(y2 - y1)))
                if boxes:
                    return max(boxes, key=lambda b: b[2] * b[3])
            except Exception:
                pass

        # Method 2: InsightFace
        if self.app is not None:
            try:
                faces = self.app.get(frame_bgr)
                if faces:
                    boxes = []
                    for face in faces:
                        x1, y1, x2, y2 = [int(v) for v in face.bbox]
                        boxes.append((x1, y1, max(1, x2 - x1), max(1, y2 - y1)))
                    return max(boxes, key=lambda b: b[2] * b[3])
            except Exception:
                pass

        # Method 3: OpenCV Haar fallback
        try:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            boxes = cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5, minSize=(48, 48))
            if len(boxes) > 0:
                b = max(boxes, key=lambda item: item[2] * item[3])
                return (int(b[0]), int(b[1]), int(b[2]), int(b[3]))
        except Exception:
            pass

        return None

    def update(self, frame_rgb: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        raw_box = self.detect_raw(frame_bgr)

        if raw_box is not None:
            cur = np.array(raw_box, dtype=np.float32)
            self.missed_frames = 0
            if self.smooth_box is None:
                self.smooth_box = cur
            else:
                # Check for sudden scene cut or jump
                center_dist = np.linalg.norm(cur[:2] - self.smooth_box[:2])
                if center_dist > max(cur[2], cur[3]) * 2.5:
                    self.smooth_box = cur
                else:
                    self.smooth_box = self.alpha * cur + (1.0 - self.alpha) * self.smooth_box
        else:
            self.missed_frames += 1
            if self.missed_frames > 15:  # Lost face for over 0.5s
                self.smooth_box = None

        if self.smooth_box is not None:
            sb = self.smooth_box.astype(int)
            return (int(sb[0]), int(sb[1]), max(10, int(sb[2])), max(10, int(sb[3])))
        return None


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

    # ESRGAN detail pass if active
    roi_bgr = cv2.cvtColor(roi, cv2.COLOR_RGB2BGR)
    roi_bgr = _esrgan_face(roi_bgr)
    roi = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)

    lab = cv2.cvtColor(roi, cv2.COLOR_RGB2LAB)
    luminance, chroma_a, chroma_b = cv2.split(lab)
    alpha = _feather_mask(w, h)
    boosted = np.clip(luminance.astype(np.float32) + detail * alpha, 0, 255).astype(np.uint8)
    out = cv2.cvtColor(cv2.merge((boosted, chroma_a, chroma_b)), cv2.COLOR_LAB2RGB)
    frame[y:y + h, x:x + w] = out
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
        ["ffmpeg", "-v", "error", "-i", str(input_path), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    temp = output_path.with_suffix(".tmp.mp4")
    encode = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{width}x{height}", "-r", f"{fps:.6f}", "-i", "-",
         "-c:v", "libx264", "-preset", "fast", "-crf", "16", "-pix_fmt", "yuv420p", str(temp)],
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

            # Track face with temporal smoothing
            box = tracker.update(frame)
            if box is not None:
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
        decode.wait()
        encode.wait()

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
    temp.unlink(missing_ok=True)
    if mux.returncode != 0:
        shutil.copy2(temp, output_path)

    print(f"[face_refine] 成功处理 {touched}/{index} 帧，输出已平滑保存: {output_path.name}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法：face_refine.py input.mp4 output.mp4", file=sys.stderr)
        sys.exit(1)
    refine(Path(sys.argv[1]), Path(sys.argv[2]))

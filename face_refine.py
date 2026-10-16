"""Standalone face detail pass for generated video; run with ComfyUI's Python env."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def refine(source: Path, output: Path, model_file: Path, strength: float = 0.55) -> None:
    detector = YOLO(str(model_file))
    capture = cv2.VideoCapture(str(source))
    fps = capture.get(cv2.CAP_PROP_FPS) or 24
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width < 1 or height < 1:
        raise RuntimeError("无法读取视频画面")
    silent = output.with_name(output.stem + "_silent.mp4")
    writer = cv2.VideoWriter(str(silent), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("无法创建人脸精修临时视频")
    last_box = None
    frame_number = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_number % 3 == 0:
                result = detector.predict(frame, conf=0.35, verbose=False, device="cpu")[0]
                boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else []
                if len(boxes):
                    box = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
                    last_box = box if last_box is None else 0.65 * last_box + 0.35 * box
                else:
                    last_box = None
            if last_box is not None:
                x1, y1, x2, y2 = last_box
                pad_x = (x2 - x1) * 0.35
                pad_y = (y2 - y1) * 0.35
                left = max(0, int(x1 - pad_x)); top = max(0, int(y1 - pad_y))
                right = min(width, int(x2 + pad_x)); bottom = min(height, int(y2 + pad_y))
                if right - left >= 24 and bottom - top >= 24:
                    crop = frame[top:bottom, left:right]
                    enhanced = cv2.detailEnhance(crop, sigma_s=8, sigma_r=0.12)
                    smooth = cv2.GaussianBlur(enhanced, (0, 0), 1.1)
                    enhanced = cv2.addWeighted(enhanced, 1.25, smooth, -0.25, 0)
                    mask = np.zeros(crop.shape[:2], np.float32)
                    center = (mask.shape[1] // 2, mask.shape[0] // 2)
                    axes = (int(mask.shape[1] * 0.42), int(mask.shape[0] * 0.42))
                    cv2.ellipse(mask, center, axes, 0, 0, 360, 1, -1)
                    mask = cv2.GaussianBlur(mask, (0, 0), max(2, min(mask.shape) * 0.08))
                    alpha = (mask * strength)[..., None]
                    frame[top:bottom, left:right] = np.clip(crop * (1 - alpha) + enhanced * alpha, 0, 255).astype(np.uint8)
            writer.write(frame)
            frame_number += 1
    finally:
        capture.release()
        writer.release()
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(silent), "-i", str(source),
                    "-map", "0:v:0", "-map", "1:a:0?", "-c:v", "libx264", "-preset", "fast",
                    "-crf", "18", "-c:a", "copy", "-shortest", str(output)], check=True, timeout=600)
    silent.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", type=Path, default=Path("/home/administrator/ComfyUI/models/ultralytics/bbox/face_yolov8m.pt"))
    parser.add_argument("--strength", type=float, default=0.55)
    args = parser.parse_args()
    refine(args.source, args.output, args.model, args.strength)

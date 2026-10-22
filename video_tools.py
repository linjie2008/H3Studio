"""Video probing with FFmpeg and adaptive shot-boundary detection."""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path


def duration(path: Path) -> float:
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                             "-of", "json", str(path)], capture_output=True, text=True, check=True, timeout=30)
    return float(json.loads(result.stdout)["format"]["duration"])


def extract_audio_track(source: Path, target: Path) -> None:
    """Extract the first audio stream to lossless FLAC for use as an H3 reference."""
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0",
                            "-show_entries", "stream=index", "-of", "json", str(source)],
                           capture_output=True, text=True, check=True, timeout=30)
    if not json.loads(probe.stdout).get("streams"):
        raise ValueError("此视频没有可提取的音轨")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(source),
                    "-map", "0:a:0", "-vn", "-c:a", "flac", "-compression_level", "5", str(target)],
                   check=True, timeout=600)


def _adaptive_scene_cuts(path: Path, end: float, sensitivity: str = "medium") -> list[float]:
    """Match ComfyUI Director's PySceneDetect AdaptiveDetector cut detection."""
    try:
        from scenedetect import AdaptiveDetector, detect
    except ImportError as error:
        raise RuntimeError("智能分割需要 PySceneDetect；请重新安装 requirements-agent.txt") from error

    thresholds = {"low": 4.5, "medium": 3.0, "high": 2.0}
    adaptive_threshold = thresholds.get(str(sensitivity).lower(), thresholds["medium"])
    scenes = detect(str(path), AdaptiveDetector(
        adaptive_threshold=adaptive_threshold,
        min_scene_len=12,
    ), show_progress=False, start_in_scene=True)
    cuts = [0.0]
    for start, _ in scenes[1:]:
        point = round(start.seconds, 3)
        if 0 < point < end:
            cuts.append(point)
    cuts.append(end)
    return sorted(set(cuts))


def _h3_spans(cuts: list[float], end: float) -> list[tuple[float, float]]:
    """Keep detected cuts where both sides can form valid 2–15s H3 shots."""
    if end < 2:
        raise ValueError("源视频不足 2 秒，无法生成 H3 镜头")
    ordered = sorted(set([0.0, end, *(float(value) for value in cuts if 0 < value < end)]))
    kept = [0.0]
    for point in ordered[1:-1]:
        if point - kept[-1] >= 2 and end - point >= 2:
            kept.append(point)
    if end - kept[-1] < 2 and len(kept) > 1:
        kept.pop()
    kept.append(end)
    spans: list[tuple[float, float]] = []
    for start, stop in zip(kept, kept[1:]):
        length = stop - start
        pieces = max(1, math.ceil(length / 15))
        for index in range(pieces):
            a = start + length * index / pieces
            b = start + length * (index + 1) / pieces
            spans.append((round(a, 3), round(b - a, 3)))
    return spans


def boundaries(path: Path, method: str, count: int = 4, threshold: float = 0.35,
               custom_cuts: list[float] | None = None, sensitivity: str = "medium") -> list[tuple[float, float]]:
    end = duration(path)
    if method == "equal":
        count = max(1, min(int(count), 100, max(1, int(end // 2))))
        cuts = [end * index / count for index in range(count + 1)]
    elif method == "manual":
        cuts = [0.0]
        try:
            values = [float(value) for value in (custom_cuts or [])]
        except (TypeError, ValueError) as error:
            raise ValueError("手动切分点必须是秒数") from error
        if len(values) > 99:
            raise ValueError("手动切分点最多 99 个")
        if any(not math.isfinite(value) or value <= 0 or value >= end for value in values):
            raise ValueError(f"切分点必须大于 0 且小于视频时长 {end:.2f} 秒")
        cuts.extend(sorted(set(values)))
        cuts.append(end)
    elif method == "scene":
        cuts = _adaptive_scene_cuts(path, end, sensitivity)
    else:
        raise ValueError("未知切分方式")
    return _h3_spans(cuts, end)

"""Video probing and shot boundary detection using local ffmpeg."""

from __future__ import annotations

import json
import math
import re
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


def boundaries(path: Path, method: str, count: int = 4, threshold: float = 0.35,
               custom_cuts: list[float] | None = None) -> list[tuple[float, float]]:
    end = duration(path)
    if method == "equal":
        count = max(1, min(int(count), 100))
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
        threshold = min(0.95, max(0.05, float(threshold)))
        filter_graph = f"fps=2,scale=320:-1,select=gt(scene\\,{threshold}),showinfo"
        result = subprocess.run(["ffmpeg", "-hide_banner", "-i", str(path), "-vf", filter_graph,
                                 "-an", "-f", "null", "-"], capture_output=True, text=True, timeout=600)
        if result.returncode:
            raise RuntimeError(result.stderr[-1000:])
        times = [float(value) for value in re.findall(r"pts_time:([0-9.]+)", result.stderr)]
        cuts = [0.0]
        for value in sorted(set(times)):
            if 1 <= value <= end - 1 and value - cuts[-1] >= 1:
                cuts.append(value)
        cuts.append(end)
    else:
        raise ValueError("未知切分方式")
    spans = []
    for a, b in zip(cuts, cuts[1:]):
        while b - a > 15:
            spans.append((round(a, 3), 15.0))
            a += 15
        if b - a >= 0.25:
            spans.append((round(a, 3), round(b - a, 3)))
    return spans

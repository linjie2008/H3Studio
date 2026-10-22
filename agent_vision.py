"""成片抽帧 + 多模态看图自评。

这是「创作智能体」和「自动化脚本」的分界线：主脑必须能真的看到成片，
才能判断「衣服到底换了没有」，再决定要不要重写提示词、重调参数。

链路：ffmpeg 抽帧 → 临时上传拿 URL → 多模态看图 → 结论回灌给主脑。
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from store import OUTPUTS

FFMPEG = shutil.which("ffmpeg") or "/usr/bin/ffmpeg"
VISION_TOOL = "connector__matrix__describe_images"
FRAME_TIMEOUT = 120
VISION_TIMEOUT = 300


def _mcode_tools() -> str:
    """Locate the CLI even when the server was started without the node PATH."""
    found = shutil.which("mcode-tools")
    if found:
        return found
    candidates = [Path.home() / ".local" / "bin" / "mcode-tools"]
    nvm = Path.home() / ".nvm" / "versions" / "node"
    if nvm.is_dir():
        candidates.extend(sorted(nvm.glob("*/lib/node_modules/@minimax-ai/code/internal-bin/mcode-tools")))
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return "mcode-tools"


def _run(command: list[str], timeout: int) -> str:
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip().splitlines()
        raise RuntimeError(detail[-1] if detail else f"命令失败：{command[0]}")
    return result.stdout


def extract_frames(video_path: Path, count: int = 3, workdir: Path | None = None) -> list[Path]:
    """Pull a few evenly spaced frames out of a clip."""
    if not Path(FFMPEG).exists():
        raise RuntimeError("系统没有 ffmpeg，无法抽帧")
    target_dir = Path(workdir or tempfile.mkdtemp(prefix="agent_frames_"))
    target_dir.mkdir(parents=True, exist_ok=True)
    count = max(1, min(6, int(count)))
    command = [FFMPEG, "-y", "-v", "error", "-i", str(video_path),
               "-vf", f"fps=1/2,scale=640:-2", "-frames:v", str(count),
               str(target_dir / "frame_%02d.jpg")]
    subprocess.run(command, capture_output=True, text=True, timeout=FRAME_TIMEOUT)
    frames = sorted(target_dir.glob("frame_*.jpg"))
    if not frames:
        # Fall back to a single still frame for very short clips.
        fallback = [FFMPEG, "-y", "-v", "error", "-i", str(video_path),
                    "-frames:v", "1", str(target_dir / "frame_01.jpg")]
        subprocess.run(fallback, capture_output=True, text=True, timeout=FRAME_TIMEOUT)
        frames = sorted(target_dir.glob("frame_*.jpg"))
    return frames


def upload_temp_url(path: Path) -> str:
    payload = json.loads(_run([_mcode_tools(), "upload_temp_url", str(path)], timeout=180))
    url = str(payload.get("temp_url") or "")
    if not url:
        raise RuntimeError("临时上传没有返回可用链接")
    return url


def describe_images(items: list[dict], timeout: int = VISION_TIMEOUT) -> list[dict]:
    """Ask the vision model about one or more frames."""
    if not items:
        return []
    args_file = Path(tempfile.mkdtemp(prefix="agent_vision_")) / "args.json"
    args_file.write_text(json.dumps({"image_info": items}, ensure_ascii=False), encoding="utf-8")
    raw = _run([_mcode_tools(), "connector", "call", VISION_TOOL, "--args-file", str(args_file)],
               timeout=timeout)
    try:
        payload = json.loads(raw)
    except (TypeError, ValueError):
        return [{"description": raw.strip()[:2000], "error": ""}]
    results = []
    for item in payload.get("results") or []:
        results.append({"description": str(item.get("description") or ""),
                        "error": str(item.get("error") or "")})
    return results


def run_videos(state: dict, run_id: str) -> list[Path]:
    """Every playable output file of one generation run."""
    run = (state.get("runs") or {}).get(str(run_id or ""))
    if not run:
        return []
    files: list[Path] = []
    for result in run.get("results") or []:
        for key in ("file", "h3_file", "pre_refine_file"):
            name = result.get(key)
            if not name:
                continue
            path = OUTPUTS / name
            if path.is_file() and path not in files:
                files.append(path)
    return files


def judge_run_frames(state: dict, run_id: str, question: str, count: int = 3) -> dict:
    """Extract frames from a run and ask a vision model whether it looks right."""
    run_id = str(run_id or "").strip()
    if not run_id:
        raise ValueError("缺少生成任务 id")
    videos = run_videos(state, run_id)
    if not videos:
        raise ValueError("这个任务还没有可查看的成片（可能仍在排队或已被清理）")
    workdir = Path(tempfile.mkdtemp(prefix="agent_frames_"))
    frames: list[Path] = []
    for video in videos[:2]:
        frames.extend(extract_frames(video, count=count, workdir=workdir))
        if len(frames) >= count:
            break
    frames = frames[:max(1, count)]
    if not frames:
        raise ValueError("无法从成片里抽帧")
    items = []
    for frame in frames:
        try:
            items.append({"url": upload_temp_url(frame), "prompt": question})
        except Exception as error:  # one bad frame should not kill the review
            items.append({"url": "", "prompt": question, "error": str(error)})
    usable = [item for item in items if item.get("url")]
    if not usable:
        raise RuntimeError("帧图上传失败，无法自评")
    answers = describe_images(usable)
    verdict = "\n".join(answer.get("description", "") for answer in answers if answer.get("description"))
    return {"run_id": run_id, "question": question, "frames": [frame.name for frame in frames],
            "verdict": verdict.strip(), "answers": answers,
            "summary": (verdict.strip()[:200] or "已完成看图自评")}

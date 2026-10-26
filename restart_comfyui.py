"""Restart this machine's ComfyUI process and return without waiting for model loading."""

from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path


import os

COMFY_DIR = Path(os.environ.get("COMFYUI_DIR", "/home/administrator/ComfyUI"))
START_SCRIPT = Path(os.environ.get("COMFYUI_START_SCRIPT", "/home/administrator/start_comfyui.sh"))
LOG = COMFY_DIR / "comfyui_boot.log"


def running_comfyui() -> list[int]:
    found = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            pid = int(entry.name)
            args = (entry / "cmdline").read_bytes().split(b"\0")
            args = [arg.decode(errors="replace") for arg in args if arg]
            cwd = os.readlink(entry / "cwd")
        except (OSError, PermissionError):
            continue
        if cwd == str(COMFY_DIR) and "main.py" in args and "8188" in args:
            if "--port" in args and args.index("--port") + 1 < len(args):
                if args[args.index("--port") + 1] == "8188":
                    found.append(pid)
    return found


def stop_process(pid: int) -> None:
    try:
        os.kill(pid, signal.SIGINT)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.5)
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.5)
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def main() -> None:
    pids = running_comfyui()
    if pids:
        print("正在停止 ComfyUI（当前生成任务会中断）…", flush=True)
        for pid in pids:
            stop_process(pid)
    else:
        print("没有检测到运行中的 ComfyUI，将直接启动。", flush=True)

    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("ab", buffering=0) as log:
        subprocess.Popen(
            ["bash", str(START_SCRIPT)], cwd=COMFY_DIR,
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    print("ComfyUI 已启动，模型与自定义节点仍在载入；工作台可以先打开。", flush=True)


if __name__ == "__main__":
    main()

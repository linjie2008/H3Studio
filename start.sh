#!/usr/bin/env bash
set -euo pipefail

studio_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
studio_port="${H3_STUDIO_PORT:-8765}"
studio_host="${H3_STUDIO_HOST:-127.0.0.1}"
studio_url="http://127.0.0.1:$studio_port"
comfy_url="${COMFYUI_URL:-http://127.0.0.1:8188}"
agent_python="$studio_dir/.venv-agent/bin/python"
if [ ! -x "$agent_python" ] || ! "$agent_python" -c 'import langgraph, scenedetect' >/dev/null 2>&1; then
    if ! command -v uv >/dev/null 2>&1; then
        echo "产品助手需要 uv。安装 uv 后重试，或在 .venv-agent 中安装 requirements-agent.txt。" >&2
        exit 1
    fi
    uv venv --clear --python 3.12 "$studio_dir/.venv-agent"
    uv pip install --python "$agent_python" -r "$studio_dir/requirements-agent.txt"
fi
mkdir -p "$studio_dir/logs"

ready() {
    curl --noproxy '*' -fsS --max-time 3 "$1" >/dev/null 2>&1
}

stop_studio_listener() {
    local pids pid cmdline process_cwd matched=0
    pids="$(fuser -n tcp "$studio_port" 2>/dev/null || true)"
    if [ -z "$pids" ]; then
        pids="$(pgrep -f '(^|/)app\.py($| )' || true)"
    fi
    for pid in $pids; do
        [ -r "/proc/$pid/cmdline" ] || continue
        cmdline="$(tr '\0' ' ' <"/proc/$pid/cmdline" 2>/dev/null || true)"
        process_cwd="$(readlink "/proc/$pid/cwd" 2>/dev/null || true)"
        if [[ "$cmdline" == *"$studio_dir/app.py"* || ( "$process_cwd" == "$studio_dir" && "$cmdline" == *app.py* ) ]]; then
            echo "正在关闭工作台进程 $pid…"
            kill "$pid" 2>/dev/null || true
            matched=1
        elif [ -n "$pids" ]; then
            echo "端口 $studio_port 被其他进程占用（PID $pid），为避免误关未执行终止。" >&2
        fi
    done
    return 0
}

studio_version="$(curl --noproxy '*' -fsS --max-time 3 "$studio_url/api/version" 2>/dev/null || true)"
if [[ -n "$studio_version" && "$studio_version" != *"independent-91"* ]]; then
    echo "正在关闭旧版工作台…"
    stop_studio_listener
    for attempt in $(seq 1 20); do
        if ! ready "$studio_url/api/version"; then break; fi
        sleep 1
    done
    if ready "$studio_url/api/version"; then
        echo "旧版工作台未退出，请检查占用 $studio_port 端口的进程" >&2
        exit 1
    fi
fi

if ! ready "$studio_url/api/version"; then
    echo "启动导演工作台…"
    setsid env H3_STUDIO_HOST="$studio_host" "$agent_python" "$studio_dir/app.py" >"$studio_dir/logs/studio.log" 2>&1 </dev/null &
    disown -a 2>/dev/null || true
    for attempt in $(seq 1 20); do
        if ready "$studio_url/api/version"; then break; fi
        sleep 1
    done
    if ! ready "$studio_url/api/version"; then
        echo "导演工作台启动失败。查看 $studio_dir/logs/studio.log" >&2
        exit 1
    fi
fi

echo "导演工作台已就绪：$studio_url"
if ready "$comfy_url/system_stats"; then
    echo "ComfyUI 推理服务已连接：$comfy_url"
    if ! curl -fsS --max-time 3 "$comfy_url/object_info/H3StudioContinueRemask" 2>/dev/null | grep -q 'H3StudioContinueRemask'; then
        echo "H3 Studio 一镜到底接续节点尚未载入。请重启 ComfyUI 后再生成连续镜头。"
    fi
else
    echo "ComfyUI 暂未连接。项目编辑可用；生成任务需等待 ComfyUI 启动。"
fi
wsl_ip="$(hostname -I 2>/dev/null | awk '{print $1}')"
if [ -n "$wsl_ip" ] && [ "$studio_host" = "0.0.0.0" ]; then
    echo "若 Windows 浏览器无法访问 127.0.0.1，请改用：http://$wsl_ip:$studio_port"
fi
if command -v explorer.exe >/dev/null 2>&1; then
    explorer.exe "$studio_url" >/dev/null 2>&1 &
elif command -v xdg-open >/dev/null 2>&1; then
    xdg-open "$studio_url" >/dev/null 2>&1 &
fi

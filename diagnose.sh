#!/usr/bin/env bash
set -u

echo "WSL 地址：$(hostname -I 2>/dev/null | awk '{print $1}')"
for port in 8188 8765; do
    echo
    echo "端口 $port："
    curl --noproxy '*' -sS --max-time 3 -o /dev/null -w 'HTTP %{http_code}\n' "http://127.0.0.1:$port/" 2>&1 || true
done
echo
echo "相关进程："
ps -eo pid,args | rg '[m]ain.py|[a]pp.py|[s]tart.sh' || true
echo
echo "最近启动日志："
for log_file in /home/administrator/minimax-h3-studio/logs/comfyui.log /home/administrator/minimax-h3-studio/logs/studio.log; do
    if [ -f "$log_file" ]; then
        echo "$log_file"
        tail -n 12 "$log_file"
    else
        echo "$log_file：未生成"
    fi
done

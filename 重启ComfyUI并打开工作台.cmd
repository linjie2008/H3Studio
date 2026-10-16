@echo off
chcp 65001 >nul
echo 正在重启 ComfyUI 并打开 H3 漫剧工作台…
echo 如果当前有生成任务，它会被中断。
wsl.exe -d Ubuntu -- bash -lc "python3 /home/administrator/minimax-h3-studio/restart_comfyui.py && bash /home/administrator/minimax-h3-studio/start.sh"
if errorlevel 1 (
    echo.
    echo 启动失败，请将上方错误信息发给我。
    pause
)

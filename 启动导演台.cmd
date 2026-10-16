@echo off
chcp 65001 >nul
echo 正在启动 MiniMax H3 导演工作台，请等待终端显示“已就绪”…
wsl.exe -d Ubuntu -- bash /home/administrator/minimax-h3-studio/start.sh
if errorlevel 1 (
    echo.
    echo 启动失败，请将上方错误信息发给我。
    pause
)

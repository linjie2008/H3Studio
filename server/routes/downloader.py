"""抖音视频下载：解析分享链接，下载最高清无水印视频。

依赖本机 Windows Chrome（无头 + CDP）绕过抖音反爬签名：
1. 解析 v.douyin.com 短链或视频链接，得到 aweme_id；
2. 复用/启动 Windows Chrome 无头实例，用真实浏览器指纹请求
   aweme/v1/web/aweme/detail 拿到播放地址列表；
3. 选最高码率的 play_url，下载到 outputs/<uid>/<标题>-<id>.mp4，
   通过 /media/outputs/... 提供访问与下载。
"""
from __future__ import annotations

import json
import re
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import requests

from server.router import Route
from store import OUTPUTS, uid

CHROME_EXE = "/mnt/c/Program Files/Google/Chrome/Application/chrome.exe"
CHROME_PROFILE_WIN = r"C:\Users\aiuse\AppData\Local\Temp\dy_chrome_profile"
CDP_HOST = "127.0.0.1"
CDP_PORT = 9223

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/155.0.0.0 Safari/537.36")

# 单用户本地工具：串行化下载，避免多线程共用同一个 CDP 浏览器。
_LOCK = threading.Lock()

# 抖音为国内服务，直连即可；不继承环境里的 http_proxy/https_proxy，
# 避免代理不稳定导致解析/下载失败。
_HTTP = requests.Session()
_HTTP.trust_env = False
_HTTP.headers.update({"User-Agent": UA})


def _cdp_available() -> bool:
    try:
        with urllib.request.urlopen(
                f"http://{CDP_HOST}:{CDP_PORT}/json/version", timeout=2) as resp:
            return resp.status == 200
    except Exception:
        return False


def _ensure_chrome() -> None:
    if _cdp_available():
        return
    subprocess.Popen(
        [CHROME_EXE, "--headless=new", "--disable-gpu", "--no-sandbox",
         "--disable-dev-shm-usage",
         f"--remote-debugging-port={CDP_PORT}",
         f"--user-data-dir={CHROME_PROFILE_WIN}",
         "about:blank"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    for _ in range(40):
        if _cdp_available():
            return
        time.sleep(0.5)
    raise RuntimeError("无法启动 Windows Chrome 无头浏览器，请确认已安装 Google Chrome")


def _extract_video_id(url: str) -> str:
    text = (url or "").strip()
    if not text:
        raise ValueError("请粘贴抖音视频链接")
    # 支持直接粘贴整段分享文案：从文本中提取第一个 http(s) 链接。
    link = re.search(r"https?://[^\s]+", text)
    if link:
        text = link.group(0).rstrip("，。；！？、）)】》〉\"'")
    elif not text.lower().startswith(("http://", "https://")):
        raise ValueError("没有找到抖音链接，请粘贴链接或包含链接的分享文案")
    if "v.douyin.com" in text:
        try:
            resp = _HTTP.get(text, allow_redirects=True, timeout=20)
            text = resp.url
        except requests.RequestException as error:
            raise RuntimeError(f"短链解析失败：{error}")
    match = re.search(r"/video/(\d+)", text)
    if match:
        return match.group(1)
    match = re.search(r"[?&]modal_id=(\d+)", text)
    if match:
        return match.group(1)
    raise ValueError("无法从链接中识别视频 ID，请粘贴抖音视频链接或分享短链")


def _best_play_url(aweme: dict):
    video = aweme.get("video") or {}
    bitrates = video.get("bit_rate") or []
    best = None
    for entry in bitrates:
        if best is None or (entry.get("bit_rate") or 0) > (best.get("bit_rate") or 0):
            best = entry
    play_addr = (best or {}).get("play_addr") or video.get("play_addr") or {}
    urls = play_addr.get("url_list") or []
    if not urls:
        raise RuntimeError("未找到视频播放地址（可能触发验证码，请稍后重试）")
    return urls[0], (best or {}).get("bit_rate"), (best or {}).get("gear_name"), play_addr


def _fetch_detail(video_id: str) -> dict:
    _ensure_chrome()
    from playwright.sync_api import sync_playwright

    with sync_playwright() as playwright:
        browser = playwright.chromium.connect_over_cdp(f"http://{CDP_HOST}:{CDP_PORT}")
        try:
            context = browser.contexts[0]
            page = context.pages[0] if context.pages else context.new_page()

            # 首次访问先落到抖音首页，让 JS 生成 s_v_web_id / UIFID 等反爬 cookie。
            cookies = context.cookies()
            if not any(cookie.get("name") == "s_v_web_id" for cookie in cookies):
                try:
                    page.goto("https://www.douyin.com/",
                              wait_until="domcontentloaded", timeout=60000)
                    page.wait_for_timeout(5000)
                except Exception:
                    pass

            bodies = []

            def on_response(resp):
                if "/aweme/v1/web/aweme/detail/" in resp.url and video_id in resp.url:
                    try:
                        bodies.append(resp.body())
                    except Exception:
                        pass

            page.on("response", on_response)
            try:
                page.goto(f"https://www.douyin.com/video/{video_id}",
                          wait_until="domcontentloaded", timeout=60000)
            except Exception:
                pass
            for _ in range(12):
                page.wait_for_timeout(2000)
                if any(len(body) > 10000 for body in bodies):
                    break
                try:
                    page.mouse.wheel(0, 400)
                except Exception:
                    pass

            usable = [body for body in bodies if len(body) > 1000]
            if not usable:
                raise RuntimeError("解析失败：抖音触发了验证码，请稍后重试或换个链接")
            data = json.loads(usable[-1].decode("utf-8", "replace"))
            aweme = data.get("aweme_detail") or {}
            if not aweme.get("aweme_id"):
                raise RuntimeError("解析失败：未返回视频信息")
            return aweme
        finally:
            # 只断开 CDP 连接，保留 Chrome 进程与 cookie 供下次复用。
            try:
                browser.close()
            except Exception:
                pass


def _safe_title(title: str, video_id: str) -> str:
    base = re.sub(r"[\\/:*?\"<>|#%?\s]+", "_", (title or "").strip())[:80].strip("_")
    return base or f"douyin_{video_id}"


def _download_video(play_url: str, title: str, video_id: str) -> tuple[str, int]:
    folder = OUTPUTS / uid()
    folder.mkdir(parents=True, exist_ok=True)
    filename = f"{_safe_title(title, video_id)}-{video_id}.mp4"
    target = folder / filename
    headers = {"Referer": "https://www.douyin.com/"}
    try:
        with _HTTP.get(play_url, headers=headers, stream=True, timeout=120) as resp:
            resp.raise_for_status()
            with target.open("wb") as output:
                for chunk in resp.iter_content(1024 * 1024):
                    if chunk:
                        output.write(chunk)
    except Exception:
        target.unlink(missing_ok=True)
        raise
    relative = f"{folder.name}/{filename}"
    return relative, target.stat().st_size


def post_api_downloader_download(req, path, parts, url):
    """POST /api/downloader/download  {url: "..."}。"""
    body = req.body_json()
    target_url = str(body.get("url") or "").strip()
    if not target_url:
        raise ValueError("请粘贴抖音视频链接")
    with _LOCK:
        video_id = _extract_video_id(target_url)
        aweme = _fetch_detail(video_id)
        play_url, bitrate, gear, play_addr = _best_play_url(aweme)
        video = aweme.get("video") or {}
        title = (aweme.get("desc") or "").strip() or f"douyin_{video_id}"
        author = ((aweme.get("author") or {}).get("nickname") or "").strip()
        relative, size = _download_video(play_url, title, video_id)
        duration_ms = video.get("duration") or 0
        folder, filename = relative.split("/", 1)
        media_url = "/media/outputs/" + urllib.parse.quote(folder) + "/" + urllib.parse.quote(filename)
        req.json(200, {
            "video_id": video_id,
            "title": title,
            "author": author,
            "file": relative,
            "url": media_url,
            "size": size,
            "bitrate": bitrate,
            "gear": gear,
            "width": play_addr.get("width"),
            "height": play_addr.get("height"),
            "duration": round((duration_ms or 0) / 1000, 2),
        })


ROUTES = [
    Route('POST', 'post_api_downloader_download',
          lambda path, parts, url: path == "/api/downloader/download",
          post_api_downloader_download),
]

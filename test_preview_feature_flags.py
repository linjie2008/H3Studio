"""ComfyUI 只有在客户端声明 supports_preview_metadata 时才推送实时预览帧。

ComfyUI 把客户端的第一条 WebSocket 消息当作特性协商消息（server.py 的
websocket_handler）。缺少这条声明时，comfy_execution/progress.py 不会发送任何
预览帧，工作台就只停在第一帧画面。本模块锁住工作台代理的这条握手。
"""

import base64
import json
import os
import struct
import unittest
from unittest.mock import patch

import app as studio_app


def _decode_masked_text_frame(frame: bytes) -> str:
    """解码一个带掩码的客户端→服务端 WebSocket 文本帧。"""
    assert frame[0] & 0x0F == 0x01, "opcode 必须是文本帧"
    assert frame[1] & 0x80, "客户端帧必须带掩码位"
    length = frame[1] & 0x7F
    offset = 2
    if length == 126:
        length = struct.unpack(">H", frame[offset:offset + 2])[0]
        offset += 2
    elif length == 127:
        length = struct.unpack(">Q", frame[offset:offset + 8])[0]
        offset += 8
    mask = frame[offset:offset + 4]
    offset += 4
    payload = frame[offset:offset + length]
    return bytes(byte ^ mask[index % 4] for index, byte in enumerate(payload)).decode("utf-8")


class _FakeUpstream:
    """模拟 ComfyUI 的 WebSocket 端点：回送 101 握手后等待。"""

    def __init__(self, response: bytes):
        self.response = response
        self.sent = bytearray()

    def sendall(self, data):
        self.sent.extend(data)

    def recv(self, size):
        chunk, self.response = self.response[:size], self.response[size:]
        return chunk

    def setblocking(self, flag):
        pass

    def close(self):
        pass


class _FakeBrowser:
    def __init__(self):
        self.sent = bytearray()

    def sendall(self, data):
        self.sent.extend(data)

    def setblocking(self, flag):
        pass


class PreviewFeatureFlagTests(unittest.TestCase):
    def test_text_frame_is_masked_and_round_trips(self):
        payload = studio_app._ws_masked_text_frame("你好 ComfyUI")
        self.assertEqual(_decode_masked_text_frame(payload), "你好 ComfyUI")

    def test_handshake_payload_declares_preview_metadata(self):
        payload = json.loads(_decode_masked_text_frame(
            studio_app._ws_masked_text_frame(json.dumps(studio_app.COMFY_CLIENT_FEATURE_FLAGS))))
        self.assertEqual(payload["type"], "feature_flags")
        self.assertTrue(payload["data"]["supports_preview_metadata"])

    def test_proxy_negotiates_preview_before_forwarding_browser_data(self):
        upstream = _FakeUpstream(
            b"HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\n\r\n")
        browser = _FakeBrowser()
        handler = studio_app.Handler.__new__(studio_app.Handler)
        handler.headers = {
            "Host": "127.0.0.1:8765",
            "Origin": "http://127.0.0.1:8765",
            "Upgrade": "websocket",
            "Sec-WebSocket-Key": base64.b64encode(os.urandom(16)).decode("ascii"),
        }
        handler.connection = browser

        # 转发循环读到上游返回空数据时结束，避免代理一直阻塞。
        with patch.object(studio_app.socket, "create_connection", return_value=upstream), \
                patch.object(studio_app.select, "select", return_value=([upstream], [], [])):
            handler.proxy_comfy_websocket("clientId=h3_studio_run123")

        sent = bytes(upstream.sent)
        request_end = sent.find(b"\r\n\r\n")
        self.assertNotEqual(request_end, -1, "应先发出 WebSocket 升级请求")
        self.assertIn(b"clientId=h3_studio_run123", sent[:request_end])

        handshake = _decode_masked_text_frame(sent[request_end + 4:])
        payload = json.loads(handshake)
        self.assertEqual(payload["type"], "feature_flags")
        self.assertTrue(payload["data"]["supports_preview_metadata"])


if __name__ == "__main__":
    unittest.main()

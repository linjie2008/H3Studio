"""Exercise the browser-facing creative-agent SSE and confirmation endpoints."""

import copy
import http.client
import json
import threading
import unittest
from http.server import ThreadingHTTPServer
from unittest.mock import patch

import agent_audit
import agent_loop
import agent_tools
from app import Handler
from store import new_project


class CreativeAgentHTTPTests(unittest.TestCase):
    def setUp(self):
        project = new_project("HTTP 智能体测试")
        project["id"] = "http-test-project"
        project["shots"][0]["id"] = "http-shot"
        project["shots"][0]["prompt"] = "原文"
        second = copy.deepcopy(project["shots"][0])
        second["id"] = "other-shot"
        project["shots"].append(second)
        self.data = {"projects": {project["id"]: project}, "assets": {}, "runs": {}}
        agent_audit.RUNS.clear()
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=3)

    def post_events(self, route, body):
        port = self.server.server_address[1]
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=15)
        conn.request("POST", route, json.dumps(body).encode(),
                     {"Content-Type": "application/json", "Origin": f"http://127.0.0.1:{port}"})
        response = conn.getresponse()
        payload = response.read().decode()
        conn.close()
        self.assertEqual(response.status, 200, payload)
        events = [json.loads(line[6:]) for line in payload.splitlines() if line.startswith("data: ")]
        self.assertFalse([event for event in events if event["type"] == "error"], events)
        return events

    def test_stream_confirm_and_undo(self):
        decisions = iter([
            {"type": "tool", "name": "delete_shot", "args": {"shot_id": "http-shot"}},
            {"type": "final", "answer": "已删除镜头"},
        ])
        mutate = lambda callback: callback(self.data)
        with patch.object(agent_tools, "read", return_value=self.data), \
                patch.object(agent_tools, "change", side_effect=mutate), \
                patch.object(agent_audit, "change", side_effect=mutate), \
                patch.object(agent_loop, "_decide", side_effect=lambda *args, **kwargs: next(decisions)):
            initial = self.post_events("/api/creative-agent/run-stream", {
                "message": "删除这个镜头", "project_id": "http-test-project", "shot_id": "http-shot"})
            pending = next(event["result"] for event in initial if event["type"] == "result")
            self.assertEqual(pending["status"], "needs_confirmation")
            self.assertEqual(len(self.data["projects"]["http-test-project"]["shots"]), 2)
            resumed = self.post_events("/api/creative-agent/confirm-stream", {
                "run_id": pending["run_id"], "approved": True})
            completed = next(event["result"] for event in resumed if event["type"] == "result")
            self.assertEqual(completed["status"], "done")
            self.assertEqual(len(self.data["projects"]["http-test-project"]["shots"]), 1)
            # Destructive deletion is intentionally irreversible. Verify the
            # separate reversible prompt-write path through HTTP as well.
            decisions = iter([{"type": "tool", "name": "write_prompt",
                               "args": {"shot_id": "other-shot", "prompt": "新文案"}}])
            with patch.object(agent_loop, "_decide", side_effect=lambda *args, **kwargs: next(decisions)):
                edited = self.post_events("/api/creative-agent/run-stream", {
                    "message": "改当前镜头提示词", "project_id": "http-test-project",
                    "shot_id": "other-shot"})
            edit_result = next(event["result"] for event in edited if event["type"] == "result")
            self.assertEqual(edit_result["status"], "done")
            self.assertEqual(self.data["projects"]["http-test-project"]["shots"][0]["prompt"], "新文案")
            port = self.server.server_address[1]
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=15)
            conn.request("POST", "/api/creative-agent/undo",
                         json.dumps({"run_id": edit_result["run_id"], "project_id": "http-test-project"}),
                         {"Content-Type": "application/json", "Origin": f"http://127.0.0.1:{port}"})
            response = conn.getresponse()
            self.assertEqual(response.status, 200, response.read().decode())
            self.assertEqual(self.data["projects"]["http-test-project"]["shots"][0]["prompt"], "原文")
            conn.close()

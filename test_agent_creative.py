"""创作智能体的循环、分级确认与回滚测试。"""

import copy
import json
import types
import unittest
from unittest.mock import MagicMock, patch

import agent_audit
import agent_loop
import agent_tools
from store import new_project


def build_data():
    project = new_project("智能体测试")
    project["id"] = "agent-project"
    project["_data_version"] = 1
    project["shots"][0]["id"] = "shot-one"
    project["shots"][0]["prompt"] = "原始提示词"
    second = copy.deepcopy(project["shots"][0])
    second["id"] = "shot-two"
    second["name"] = "第二个镜头"
    project["shots"].append(second)
    return {"projects": {project["id"]: project}, "assets": {}, "runs": {}}


class CreativeAgentTests(unittest.TestCase):
    def setUp(self):
        self.data = build_data()
        agent_audit.RUNS.clear()

    def _change(self, callback):
        return callback(self.data)

    def _patches(self, decisions):
        return (patch.object(agent_tools, "read", return_value=self.data),
                patch.object(agent_tools, "change", side_effect=self._change),
                patch.object(agent_audit, "change", side_effect=self._change),
                patch.object(agent_loop, "_decide", side_effect=decisions))

    def test_tool_registry_covers_four_risk_levels(self):
        risks = {spec["risk"] for spec in agent_tools.TOOLS.values()}
        self.assertEqual(risks, {"read", "write", "destructive", "gpu"})
        for spec in agent_tools.TOOLS.values():
            self.assertTrue(spec["summary"], spec["name"])
            self.assertIn(spec["name"], agent_tools.tool_catalog_prompt())

    def test_write_tool_runs_automatically_and_is_undoable(self):
        decisions = [
            {"type": "tool", "name": "list_shots", "args": {}},
            {"type": "tool", "name": "write_prompt",
             "args": {"shot_id": "shot-one", "prompt": "新提示词内容"}},
            {"type": "final", "answer": "已改好"},
        ]
        read_patch, change_patch, audit_patch, decide_patch = self._patches(decisions)
        with read_patch, change_patch, audit_patch, decide_patch:
            result = agent_loop.run_agent({"message": "把这个镜头的提示词改一下",
                                          "project_id": "agent-project", "shot_id": "shot-one"})
            self.assertEqual(result["status"], "done")
            self.assertEqual(result["answer"], "已改好")
            shot = self.data["projects"]["agent-project"]["shots"][0]
            self.assertEqual(shot["prompt"], "新提示词内容")
            self.assertEqual(result["report"]["applied"], 2)
            self.assertTrue(result["report"]["prompt_changes"])

            undone = agent_audit.undo_run(result["run_id"], "agent-project")
            self.assertEqual(undone["undone"], 1)
            shot = self.data["projects"]["agent-project"]["shots"][0]
            self.assertEqual(shot["prompt"], "原始提示词")

    def test_destructive_tool_pauses_until_confirmed(self):
        decisions = [
            {"type": "tool", "name": "delete_shot", "args": {"shot_id": "shot-one"}},
            {"type": "final", "answer": "已删除"},
        ]
        read_patch, change_patch, audit_patch, decide_patch = self._patches(decisions)
        with read_patch, change_patch, audit_patch, decide_patch:
            result = agent_loop.run_agent({"message": "删掉这个镜头",
                                          "project_id": "agent-project", "shot_id": "shot-one"})
            self.assertEqual(result["status"], "needs_confirmation")
            self.assertEqual(result["pending"]["name"], "delete_shot")
            self.assertEqual(result["pending"]["risk"], "destructive")
            # 确认之前项目不能被动过
            self.assertEqual(len(self.data["projects"]["agent-project"]["shots"]), 2)

            resumed = agent_loop.confirm_run(result["run_id"], True)
            self.assertEqual(resumed["status"], "done")
            self.assertEqual(len(self.data["projects"]["agent-project"]["shots"]), 1)

    def test_rejected_action_is_reported_back_to_the_agent(self):
        decisions = [
            {"type": "tool", "name": "delete_shot", "args": {"shot_id": "shot-one"}},
            {"type": "final", "answer": "好的，那就不删了"},
        ]
        read_patch, change_patch, audit_patch, decide_patch = self._patches(decisions)
        with read_patch, change_patch, audit_patch, decide_patch:
            result = agent_loop.run_agent({"message": "删掉这个镜头",
                                          "project_id": "agent-project", "shot_id": "shot-one"})
            resumed = agent_loop.confirm_run(result["run_id"], False)
            self.assertEqual(resumed["status"], "done")
            self.assertEqual(len(self.data["projects"]["agent-project"]["shots"]), 2)
            observations = agent_audit.get_run(result["run_id"])["observations"]
            self.assertIn("拒绝", observations[-1]["result"])

    def test_repeated_read_call_reuses_previous_result(self):
        """同样的只读查询不该再跑一遍：每次都要等模型，重复查询很费时间。"""
        decisions = [
            {"type": "tool", "name": "list_shots", "args": {}},
            {"type": "tool", "name": "list_shots", "args": {}},
            {"type": "final", "answer": "看完了"},
        ]
        read_patch, change_patch, audit_patch, decide_patch = self._patches(decisions)
        with read_patch, change_patch, audit_patch, decide_patch:
            result = agent_loop.run_agent({"message": "看一下镜头",
                                          "project_id": "agent-project", "shot_id": "shot-one"})
        self.assertEqual(result["status"], "done")
        steps = result["report"]["steps"]
        self.assertEqual(len(steps), 2)
        self.assertTrue(steps[1].get("cached"))
        self.assertIn("复用", steps[1]["result"])

    def test_write_call_invalidates_the_read_cache(self):
        """写过之后必须重新读，不能拿旧状态当新状态。"""
        decisions = [
            {"type": "tool", "name": "list_shots", "args": {}},
            {"type": "tool", "name": "write_prompt",
             "args": {"shot_id": "shot-one", "prompt": "新的提示词"}},
            {"type": "tool", "name": "list_shots", "args": {}},
            {"type": "final", "answer": "改完了"},
        ]
        read_patch, change_patch, audit_patch, decide_patch = self._patches(decisions)
        with read_patch, change_patch, audit_patch, decide_patch:
            result = agent_loop.run_agent({"message": "改提示词",
                                          "project_id": "agent-project", "shot_id": "shot-one"})
        steps = result["report"]["steps"]
        self.assertFalse(steps[2].get("cached"))
        self.assertIn("新的提示词", str(steps[2]["data"]))

    def test_read_only_batch_runs_in_one_round_trip(self):
        """一批互不依赖的只读查询应该一次问完，而不是一个一个来。"""
        decisions = [
            {"type": "tools", "calls": [
                {"name": "list_shots", "args": {}},
                {"name": "list_assets", "args": {}},
                {"name": "list_characters", "args": {}},
            ]},
            {"type": "final", "answer": "都看过了"},
        ]
        read_patch, change_patch, audit_patch, decide_patch = self._patches(decisions)
        with read_patch, change_patch, audit_patch, decide_patch:
            result = agent_loop.run_agent({"message": "看看项目里有什么",
                                          "project_id": "agent-project", "shot_id": ""})
        self.assertEqual(result["status"], "done")
        tools = [step["tool"] for step in result["report"]["steps"]]
        self.assertEqual(tools, ["list_shots", "list_assets", "list_characters"])
        self.assertTrue(all(step["ok"] for step in result["report"]["steps"]))

    def test_write_tool_is_not_run_in_a_read_batch(self):
        """批量里混进写工具时，写工具必须退回单条执行（确认与留痕都靠它）。"""
        decisions = [
            {"type": "tools", "calls": [
                {"name": "list_shots", "args": {}},
                {"name": "write_prompt", "args": {"shot_id": "shot-one", "prompt": "批量里的写入"}},
            ]},
            {"type": "final", "answer": "好了"},
        ]
        read_patch, change_patch, audit_patch, decide_patch = self._patches(decisions)
        with read_patch, change_patch, audit_patch, decide_patch:
            result = agent_loop.run_agent({"message": "批量里带个写入",
                                          "project_id": "agent-project", "shot_id": "shot-one"})
        tools = [step["tool"] for step in result["report"]["steps"]]
        self.assertNotIn("write_prompt", tools)
        self.assertEqual(self.data["projects"]["agent-project"]["shots"][0]["prompt"], "原始提示词")

    def test_model_failure_is_announced_before_retrying(self):
        """模型没接上话时要立刻出声，不然界面会像卡死。"""
        calls = {"n": 0}

        def flaky(run, history):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("模型超时")
            return {"type": "final", "answer": "接上了"}

        events = []
        read_patch, change_patch, audit_patch, decide_patch = self._patches([])
        with read_patch, change_patch, audit_patch, decide_patch:
            with patch.object(agent_loop, "_decide", side_effect=flaky):
                result = agent_loop.run_agent({"message": "做点事",
                                              "project_id": "agent-project", "shot_id": ""},
                                             emit=events.append)
        self.assertEqual(result["status"], "done")
        retries = [event for event in events if event.get("type") == "retry"]
        self.assertTrue(retries, events)
        self.assertIn("模型没接上话", retries[0]["note"])

    def test_tool_call_cap_stops_the_endless_research_loop(self):
        """同一个工具问太多次要拦下来，不然智能体会一直查下去不给结论。"""
        decisions = [
            {"type": "tool", "name": "consult_playbook", "args": {"query": "换人"}},
            {"type": "tool", "name": "consult_playbook", "args": {"query": "换人参数"}},
            {"type": "tool", "name": "consult_playbook", "args": {"query": "再问一次"}},
            {"type": "final", "answer": "查够了，说结论"},
        ]
        read_patch, change_patch, audit_patch, decide_patch = self._patches(decisions)
        with read_patch, change_patch, audit_patch, decide_patch:
            result = agent_loop.run_agent({"message": "查查换人怎么做",
                                          "project_id": "agent-project", "shot_id": ""})
        steps = result["report"]["steps"]
        self.assertEqual(len(steps), 3)
        self.assertFalse(steps[2]["ok"])
        self.assertIn("不要再查", steps[2]["result"])

    def test_budget_exhaustion_still_gives_a_conclusion(self):
        """步数用完也必须给结论，而不是一句「步数用完」。"""
        calls = {"n": 0}

        def decide(run, history, force_final=False):
            calls["n"] += 1
            if force_final:
                return {"type": "final", "answer": "收尾结论：镜头 3 缺少参考图。"}
            return {"type": "tool", "name": "list_shots", "args": {}}

        read_patch, change_patch, audit_patch, decide_patch = self._patches([])
        with read_patch, change_patch, audit_patch, decide_patch:
            with patch.object(agent_loop, "_decide", side_effect=decide):
                result = agent_loop.run_agent({"message": "一直查",
                                              "project_id": "agent-project", "shot_id": ""})
        self.assertEqual(result["status"], "budget")
        self.assertIn("收尾结论", result["answer"])

    def test_unparsable_model_output_is_delivered_as_the_answer(self):
        """模型写出内容但不是合法 JSON 时，那段文字就是结论，不该报“没接上话”。"""
        def decide(run, history, force_final=False):
            raise agent_loop.DecisionError("模型没有返回可解析的 JSON",
                                          "检查结果：镜头 3 的提示词缺少参考图标签，建议补上。")

        read_patch, change_patch, audit_patch, decide_patch = self._patches([])
        with read_patch, change_patch, audit_patch, decide_patch:
            with patch.object(agent_loop, "_decide", side_effect=decide):
                result = agent_loop.run_agent({"message": "检查一下",
                                              "project_id": "agent-project", "shot_id": ""})
        self.assertEqual(result["status"], "done")
        self.assertIn("缺少参考图标签", result["answer"])

    def test_repeated_prompt_rewrites_show_one_diff(self):
        """同一镜头改写两次，报告里只应该有一条「改前/改后」对比。"""
        decisions = [
            {"type": "tool", "name": "write_prompt",
             "args": {"shot_id": "shot-one", "prompt": "第一版"}},
            {"type": "tool", "name": "write_prompt",
             "args": {"shot_id": "shot-one", "prompt": "第二版"}},
            {"type": "final", "answer": "改好了"},
        ]
        read_patch, change_patch, audit_patch, decide_patch = self._patches(decisions)
        with read_patch, change_patch, audit_patch, decide_patch:
            result = agent_loop.run_agent({"message": "改提示词",
                                          "project_id": "agent-project", "shot_id": "shot-one"})
        diffs = result["report"]["prompt_changes"]
        self.assertEqual(len(diffs), 1)
        self.assertEqual(diffs[0]["before"], "原始提示词")
        self.assertEqual(diffs[0]["after"], "第二版")

    def test_run_history_lists_recent_runs(self):
        decisions = [{"type": "tool", "name": "list_shots", "args": {}},
                     {"type": "final", "answer": "好了"}]
        read_patch, change_patch, audit_patch, decide_patch = self._patches(decisions)
        with read_patch, change_patch, audit_patch, decide_patch:
            result = agent_loop.run_agent({"message": "打个招呼",
                                          "project_id": "agent-project", "shot_id": ""})
        history = agent_audit.list_runs("agent-project")
        self.assertTrue(history)
        self.assertEqual(history[0]["id"], result["run_id"])
        self.assertEqual(history[0]["status"], "done")
        self.assertEqual(history[0]["applied"], 1)
        self.assertEqual(history[0]["steps"][0]["tool"], "list_shots")

    def test_gpu_tool_needs_confirmation_before_queueing(self):
        decisions = [
            {"type": "tool", "name": "submit_render", "args": {}},
            {"type": "final", "answer": "已排队"},
        ]
        # 只替换 agent_tools 里的 Thread 引用：直接 patch 全局 threading.Thread
        # 会把 LangGraph 自己的线程也一起 mock 掉，图会等一个永远不启动的线程。
        thread = MagicMock()
        fake_threading = types.SimpleNamespace(Thread=thread)
        read_patch, change_patch, audit_patch, decide_patch = self._patches(decisions)
        with read_patch, change_patch, audit_patch, decide_patch, \
             patch.object(agent_tools, "threading", fake_threading):
            result = agent_loop.run_agent({"message": "跑一下",
                                          "project_id": "agent-project", "shot_id": "shot-one"})
            self.assertEqual(result["status"], "needs_confirmation")
            self.assertEqual(result["pending"]["risk"], "gpu")
            self.assertEqual(len(self.data["runs"]), 0)
            thread.return_value.start.assert_not_called()

            resumed = agent_loop.confirm_run(result["run_id"], True)
            self.assertEqual(resumed["status"], "done")
            self.assertEqual(len(self.data["runs"]), 1)
            thread.return_value.start.assert_called_once()

    def test_unknown_tool_does_not_break_the_loop(self):
        decisions = [
            {"type": "tool", "name": "make_coffee", "args": {}},
            {"type": "final", "answer": "我换个做法"},
        ]
        read_patch, change_patch, audit_patch, decide_patch = self._patches(decisions)
        with read_patch, change_patch, audit_patch, decide_patch:
            result = agent_loop.run_agent({"message": "来个咖啡",
                                          "project_id": "agent-project"})
            self.assertEqual(result["status"], "done")
            observations = agent_audit.get_run(result["run_id"])["observations"]
            self.assertIn("没有名为", observations[-1]["result"])

    def test_failed_tool_call_is_reported_not_raised(self):
        decisions = [
            {"type": "tool", "name": "set_shot_params",
             "args": {"shot_id": "shot-one", "params": {"nonsense": 1}}},
            {"type": "final", "answer": "这个字段我不能改"},
        ]
        read_patch, change_patch, audit_patch, decide_patch = self._patches(decisions)
        with read_patch, change_patch, audit_patch, decide_patch:
            result = agent_loop.run_agent({"message": "改个参数",
                                          "project_id": "agent-project", "shot_id": "shot-one"})
            self.assertEqual(result["status"], "done")
            observations = agent_audit.get_run(result["run_id"])["observations"]
            self.assertIn("失败", observations[-1]["result"])

    def test_shot_params_accept_only_known_fields(self):
        ctx = agent_tools.ToolContext("agent-project", "shot-one", {})
        with patch.object(agent_tools, "read", return_value=self.data), \
             patch.object(agent_tools, "change", side_effect=self._change):
            detail = agent_tools.TOOLS["set_shot_params"]["handler"](
                ctx, {"shot_id": "shot-one", "params": {"ref_image_size": "max",
                                                        "ref_background": "inherit"}})
            self.assertEqual(detail["data"]["changed"]["ref_image_size"], "max")
            shot = self.data["projects"]["agent-project"]["shots"][0]
            self.assertEqual(shot["ref_image_size"], "max")
            self.assertEqual(shot["ref_background"], "inherit")
            with self.assertRaisesRegex(ValueError, "不允许修改"):
                agent_tools.TOOLS["set_shot_params"]["handler"](
                    ctx, {"shot_id": "shot-one", "params": {"prompt": "x"}})

    def test_insert_reference_tag_syncs_prompt_and_binding(self):
        self.data["assets"]["img-1"] = {"id": "img-1", "project_id": "agent-project",
                                         "kind": "image", "name": "身份图"}
        self.data["projects"]["agent-project"]["asset_order"] = ["img-1"]
        ctx = agent_tools.ToolContext("agent-project", "shot-one", {})
        with patch.object(agent_tools, "read", return_value=self.data), \
             patch.object(agent_tools, "change", side_effect=self._change):
            detail = agent_tools.TOOLS["insert_reference_tag"]["handler"](
                ctx, {"shot_id": "shot-one", "asset_id": "img-1"})
            self.assertEqual(detail["data"]["tag"], "<Picture 1>")
            shot = self.data["projects"]["agent-project"]["shots"][0]
            self.assertIn("<Picture 1>", shot["prompt"])
            self.assertEqual(shot["ref_images"], ["img-1"])

    def test_new_project_key_is_undoable_and_serializable(self):
        decisions = [
            {"type": "tool", "name": "set_project_settings",
             "args": {"settings": {"h3_lora_character_swap_strength": 1.2}}},
            {"type": "final", "answer": "已设置"},
        ]
        read_patch, change_patch, audit_patch, decide_patch = self._patches(decisions)
        with read_patch, change_patch, audit_patch, decide_patch:
            result = agent_loop.run_agent({"message": "把换人 LoRA 强度调到 1.2",
                                          "project_id": "agent-project"})
            json.dumps(result)  # 回滚记录里的哨兵不能破坏响应
            project = self.data["projects"]["agent-project"]
            self.assertEqual(project["h3_lora_character_swap_strength"], 1.2)
            agent_audit.undo_run(result["run_id"], "agent-project")
            self.assertNotIn("h3_lora_character_swap_strength", project)

    def test_knowledge_search_finds_the_swap_playbook(self):
        from agent_knowledge import search_knowledge
        hits = search_knowledge("换人 服装 不生效 参考图尺寸", limit=5)
        self.assertTrue(hits)
        self.assertTrue(any("playbook" in hit["source"] or "swap" in hit["source"]
                            for hit in hits))

    def test_undo_restores_created_shot_creation(self):
        decisions = [
            {"type": "tool", "name": "create_shots",
             "args": {"shots": [{"name": "新镜头", "prompt": "p", "duration": 5, "mode": "t2v"}]}},
            {"type": "final", "answer": "建好了"},
        ]
        read_patch, change_patch, audit_patch, decide_patch = self._patches(decisions)
        with read_patch, change_patch, audit_patch, decide_patch:
            result = agent_loop.run_agent({"message": "加个镜头",
                                          "project_id": "agent-project"})
            self.assertEqual(len(self.data["projects"]["agent-project"]["shots"]), 3)
            undone = agent_audit.undo_run(result["run_id"], "agent-project")
            self.assertEqual(undone["undone"], 1)
            self.assertEqual(len(self.data["projects"]["agent-project"]["shots"]), 2)


if __name__ == "__main__":
    unittest.main()

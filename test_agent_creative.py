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
            self.assertIn("已更新当前镜头的提示词", result["answer"])
            shot = self.data["projects"]["agent-project"]["shots"][0]
            self.assertEqual(shot["prompt"], "新提示词内容")
            self.assertEqual(result["report"]["applied"], 2)
            self.assertTrue(result["report"]["prompt_changes"])

            undone = agent_audit.undo_run(result["run_id"], "agent-project")
            self.assertEqual(undone["undone"], 1)
            shot = self.data["projects"]["agent-project"]["shots"][0]
            self.assertEqual(shot["prompt"], "原始提示词")

    def test_optimization_uses_project_skill_even_if_model_calls_write_prompt(self):
        self.data["projects"]["agent-project"]["prompt_skill"] = "custom-continuous-close-pass"
        decisions = [{"type": "tool", "name": "write_prompt",
                      "args": {"shot_id": "shot-one", "prompt": "模型自己写的文本"}}]
        read_patch, change_patch, audit_patch, decide_patch = self._patches(decisions)
        with read_patch, change_patch, audit_patch, decide_patch, \
                patch.object(agent_tools, "list_skills", return_value=[
                    {"id": "custom-continuous-close-pass", "name": "连续驶近特写运镜"}]), \
                patch("prompt_tools.enhance", return_value="遵循项目 Skill 的优化文本") as enhance:
            result = agent_loop.run_agent({"message": "优化这个镜头的提示词，保持运动连续",
                                           "project_id": "agent-project", "shot_id": "shot-one"})
        self.assertEqual(result["status"], "done")
        self.assertEqual(result["report"]["steps"][0]["tool"], "rewrite_prompt")
        self.assertEqual(result["report"]["steps"][0]["data"]["skill_id"], "custom-continuous-close-pass")
        self.assertEqual(self.data["projects"]["agent-project"]["shots"][0]["prompt"],
                         "遵循项目 Skill 的优化文本")
        self.assertEqual(enhance.call_args.args[2], "custom-continuous-close-pass")

    def test_current_shot_optimization_skips_planner_round_trip(self):
        read_patch, change_patch, audit_patch, decide_patch = self._patches([])
        with read_patch, change_patch, audit_patch, decide_patch as decide, \
                patch("prompt_tools.enhance", return_value="优化后的提示词"):
            result = agent_loop.run_agent({"message": "优化当前镜头提示词，保持动作连续",
                                           "project_id": "agent-project", "shot_id": "shot-one"})
        self.assertEqual(result["status"], "done")
        self.assertEqual([step["tool"] for step in result["report"]["steps"]], ["rewrite_prompt"])
        decide.assert_not_called()
        self.assertEqual(self.data["projects"]["agent-project"]["shots"][0]["prompt"], "优化后的提示词")

    def test_shared_prompt_edit_stops_without_touching_shot_prompts(self):
        decisions = [
            {"type": "tool", "name": "set_shared_prompt",
             "args": {"prompt": "全片保持超现实主义风格", "reason": "统一风格"}},
            {"type": "tool", "name": "write_prompt",
             "args": {"shot_id": "shot-one", "prompt": "不应执行"}},
        ]
        read_patch, change_patch, audit_patch, decide_patch = self._patches(decisions)
        with read_patch, change_patch, audit_patch, decide_patch as decide:
            result = agent_loop.run_agent({"message": "全局统一提示词中的风格一定要是超现实主义",
                                           "project_id": "agent-project", "shot_id": "shot-one"})
        self.assertEqual(result["status"], "done")
        self.assertEqual([step["tool"] for step in result["report"]["steps"]], ["set_shared_prompt"])
        self.assertEqual(self.data["projects"]["agent-project"]["shots"][0]["prompt"], "原始提示词")
        self.assertEqual(self.data["projects"]["agent-project"]["shared_prompt"], "全片保持超现实主义风格")
        decide.assert_called_once()

    def test_rewrite_does_not_override_project_skill_for_reference_shot(self):
        project = self.data["projects"]["agent-project"]
        project["prompt_skill"] = "custom-continuous-close-pass"
        project["shots"][0].update(mode="r2v", ref_images=["image-one"])
        ctx = agent_tools.ToolContext("agent-project", "shot-one")
        with patch.object(agent_tools, "read", return_value=self.data), \
                patch.object(agent_tools, "change", side_effect=self._change), \
                patch.object(agent_tools, "list_skills", return_value=[
                    {"id": "custom-continuous-close-pass", "name": "连续驶近特写运镜"},
                    {"id": "custom-h3-character-swap", "name": "换人"}]), \
                patch.object(agent_tools, "reference_context_for", return_value={}), \
                patch("prompt_tools.enhance", return_value="项目 Skill 的结果") as enhance:
            result = agent_tools._rewrite_prompt(ctx, {
                "shot_id": "shot-one", "instruction": "按当前项目要求优化",
                "skill_id": "custom-h3-character-swap"})
        self.assertEqual(result["data"]["skill_id"], "custom-continuous-close-pass")
        self.assertEqual(enhance.call_args.args[2], "custom-continuous-close-pass")

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
            result = agent_loop.run_agent({"message": "改提示词和参数后再检查",
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

    def test_batch_read_refreshes_after_write_but_stalls_when_unchanged(self):
        read_patch, change_patch, audit_patch, _ = self._patches([])
        run = {"project_id": "agent-project", "shot_id": "shot-one", "question": "检查并修改",
               "steps": [], "signatures": [], "changes": []}
        calls = [{"name": "get_shot", "args": {"shot_id": "shot-one"}}]
        with read_patch, change_patch, audit_patch:
            first = agent_loop._execute_batch(run, calls)
            self.assertEqual(first[0]["data"]["shot"]["prompt"], "原始提示词")
            self.assertEqual(agent_loop._execute_batch(run, calls), [])
            self.assertTrue(run["read_batch_stalled"])
            edited = agent_loop._execute(run, "write_prompt",
                                         {"shot_id": "shot-one", "prompt": "改后文本"})
            self.assertTrue(edited["ok"], edited)
            refreshed = agent_loop._execute_batch(run, calls)
        self.assertEqual(refreshed[0]["data"]["shot"]["prompt"], "改后文本")
        self.assertEqual(len([step for step in run["steps"] if step["tool"] == "get_shot"]), 2)

    def test_batch_respects_read_limits_with_duplicate_calls(self):
        run = {"steps": [], "read_cache": {}, "project_id": "agent-project", "shot_id": "shot-one"}
        calls = [{"name": "get_shot", "args": {"shot_id": f"shot-{index}"}} for index in range(6)]
        with patch.object(agent_loop, "GLOBAL_READ_CAP", 2), \
                patch.object(agent_loop, "_run_tool", return_value={"tool": "get_shot", "ok": True,
                                                                   "result": "ok", "data": {}}) as execute:
            result = agent_loop._execute_batch(run, calls)
        self.assertEqual(len(result), 2)
        self.assertEqual(execute.call_count, 2)

    def test_write_tool_in_batch_uses_audited_single_path(self):
        """批量里混进写工具时，转成单条执行并留下可回滚记录。"""
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
        self.assertEqual(tools, ["write_prompt"])
        self.assertEqual(result["report"]["changes"], 1)
        self.assertEqual(self.data["projects"]["agent-project"]["shots"][0]["prompt"], "批量里的写入")

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
            return {"type": "tool", "name": "list_shots" if calls["n"] == 1 else "list_assets",
                    "args": {}}

        read_patch, change_patch, audit_patch, decide_patch = self._patches([])
        with read_patch, change_patch, audit_patch, decide_patch:
            with patch.object(agent_loop, "_decide", side_effect=decide), \
                 patch.object(agent_loop, "MAX_STEPS", 2):
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

    def test_dsml_tool_array_is_executed_instead_of_shown_to_user(self):
        raw = ('[{"name":"get_shot","args":{"shot_id":"shot-one"}},'
               '{"name":"get_shot","args":{"shot_id":"shot-two"}}]'
               '｜｜DSML｜｜ parameter>\n</｜｜DSML｜｜ invoke>')
        first = agent_loop._parse_json(raw)
        self.assertEqual(first["type"], "tools")
        events = []
        read_patch, change_patch, audit_patch, decide_patch = self._patches(
            [first, {"type": "final", "answer": "已检查两个镜头。"}])
        with read_patch, change_patch, audit_patch, decide_patch:
            result = agent_loop.run_agent({"message": "检查两个镜头",
                                           "project_id": "agent-project"}, emit=events.append)
        self.assertEqual(result["answer"], "已检查两个镜头。")
        self.assertEqual([step["tool"] for step in result["report"]["steps"]],
                         ["get_shot", "get_shot"])
        self.assertEqual(len([event for event in events if event.get("type") == "step"]), 2)

    def test_repeated_read_batch_reports_progress_instead_of_spinning(self):
        first = {"type": "tools", "calls": [{"name": "get_shot",
                                               "args": {"shot_id": "shot-one"}}]}
        read_patch, change_patch, audit_patch, decide_patch = self._patches(
            [first, first, {"type": "final", "answer": "已检查。"}])
        with read_patch, change_patch, audit_patch, decide_patch:
            result = agent_loop.run_agent({"message": "检查镜头",
                                           "project_id": "agent-project"})
        self.assertEqual(result["status"], "done")
        self.assertEqual(len(result["report"]["steps"]), 1)
        self.assertEqual(result["answer"], "已检查。")
        self.assertNotIn("batch", [step["tool"] for step in result["report"]["steps"]])

    def test_repeating_same_single_read_twice_finishes(self):
        repeated = {"type": "tool", "name": "get_shot", "args": {"shot_id": "shot-one"}}
        read_patch, change_patch, audit_patch, decide_patch = self._patches(
            [repeated, repeated, repeated, {"type": "final", "answer": "已经核对。"}])
        with read_patch, change_patch, audit_patch, decide_patch:
            result = agent_loop.run_agent({"message": "核对镜头", "project_id": "agent-project"})
        self.assertEqual(result["status"], "done")
        self.assertEqual(result["answer"], "已经核对。")
        self.assertEqual(len(result["report"]["steps"]), 3)

    def test_provider_function_arguments_are_normalized(self):
        raw = '{"tool_calls":[{"function":{"name":"get_shot","arguments":"{\\"shot_id\\":\\"shot-one\\"}"}}]}'
        decision = agent_loop._parse_json(raw)
        self.assertEqual(decision, {"type": "tools", "calls": [
            {"name": "get_shot", "args": {"shot_id": "shot-one"}}]})

    def test_explicit_read_only_request_blocks_model_write(self):
        decisions = [
            {"type": "tool", "name": "write_prompt",
             "args": {"shot_id": "shot-one", "prompt": "不应写入"}},
            {"type": "final", "answer": "只读核对完成，未修改项目。"},
        ]
        read_patch, change_patch, audit_patch, decide_patch = self._patches(decisions)
        with read_patch, change_patch, audit_patch, decide_patch:
            result = agent_loop.run_agent({"message": "只读核对提示词，不要修改项目",
                                           "project_id": "agent-project", "shot_id": "shot-one"})
        self.assertEqual(result["status"], "done")
        self.assertEqual(result["report"]["changes"], 0)
        self.assertEqual(self.data["projects"]["agent-project"]["shots"][0]["prompt"], "原始提示词")
        self.assertFalse(agent_loop._simple_prompt_edit({"question": "只读核对提示词，不要修改",
                                                         "shot_id": "shot-one"}))

    def test_read_only_request_does_not_interrupt_for_delete(self):
        decisions = [
            {"type": "tool", "name": "delete_shot", "args": {"shot_id": "shot-one"}},
            {"type": "final", "answer": "只读查看完成。"},
        ]
        read_patch, change_patch, audit_patch, decide_patch = self._patches(decisions)
        with read_patch, change_patch, audit_patch, decide_patch:
            result = agent_loop.run_agent({"message": "仅查看镜头，不要执行删除",
                                           "project_id": "agent-project", "shot_id": "shot-one"})
        self.assertEqual(result["status"], "done")
        self.assertFalse(result.get("pending"))
        self.assertEqual(len(self.data["projects"]["agent-project"]["shots"]), 2)

    def test_get_shot_digest_keeps_prompt_and_bound_reference(self):
        item = {"tool": "get_shot", "ok": True, "result": "镜头详情",
                "data": {"shot": {"id": "shot-one", "prompt": "猫使用 <Picture 2>"},
                         "bound_assets": [{"id": "cat-image", "picture": 2}]}}
        digest = agent_loop._compact_observation(item)["data"]
        self.assertIn("猫使用 <Picture 2>", digest)
        self.assertIn("cat-image", digest)

    def test_prompt_edit_cannot_report_success_without_writing(self):
        from agent_graph import _plan_node
        run = {"question": "你来更新提示词", "shot_id": "shot-one", "steps": []}
        with patch.object(agent_loop, "_decide", return_value={"type": "final", "answer": "已处理"}):
            planned = _plan_node({"run": run})
        self.assertEqual(planned["action"]["name"], "rewrite_prompt")
        self.assertEqual(planned["action"]["args"]["shot_id"], "shot-one")
        self.assertTrue(planned["run"]["prompt_fallback_attempted"])

    def test_unrecognized_internal_markup_is_not_delivered_as_answer(self):
        raw = '<｜DSML｜tool_calls>get_shot</｜DSML｜tool_calls>'
        self.assertEqual(agent_loop._plain_answer(raw), "")
        read_patch, change_patch, audit_patch, decide_patch = self._patches([])
        with read_patch, change_patch, audit_patch, decide_patch:
            with patch.object(agent_loop, "_decide", side_effect=agent_loop.DecisionError(
                    "模型没有返回可解析的 JSON", raw)), patch("agent_graph.time.sleep"):
                result = agent_loop.run_agent({"message": "检查镜头",
                                               "project_id": "agent-project"})
        self.assertNotIn("DSML", result["answer"])
        self.assertIn("无法识别", result["answer"])

    def test_repeated_prompt_rewrites_show_one_diff(self):
        """同一镜头改写两次，报告里只应该有一条「改前/改后」对比。"""
        decisions = [
            {"type": "tool", "name": "write_prompt",
             "args": {"shot_id": "shot-one", "prompt": "第一版"}},
            {"type": "tool", "name": "write_prompt",
             "args": {"shot_id": "shot-one", "prompt": "第二版"}},
            {"type": "tool", "name": "write_prompt",
             "args": {"shot_id": "shot-one", "prompt": "第三版"}},
            {"type": "final", "answer": "改好了"},
        ]
        read_patch, change_patch, audit_patch, decide_patch = self._patches(decisions)
        with read_patch, change_patch, audit_patch, decide_patch:
            result = agent_loop.run_agent({"message": "改提示词然后再次修改",
                                          "project_id": "agent-project", "shot_id": "shot-one"})
        diffs = result["report"]["prompt_changes"]
        self.assertEqual(len(diffs), 1)
        self.assertEqual(diffs[0]["before"], "原始提示词")
        self.assertEqual(diffs[0]["after"], "第二版")
        self.assertFalse(result["report"]["steps"][2]["ok"])
        self.assertIn("已改写两次", result["report"]["steps"][2]["result"])

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

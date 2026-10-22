import unittest
from unittest.mock import patch

import product_agent


class ProductAgentTests(unittest.TestCase):
    def test_search_has_mode_guidance_and_allowlisted_sources(self):
        result = product_agent.search_product_docs("首尾帧和多参有什么区别")
        self.assertTrue(result)
        self.assertTrue(all(item["source"] in product_agent.DOCUMENTS for item in result))
        self.assertTrue(any("首尾帧" in item["text"] for item in result))

    def test_chat_uses_project_context_without_writes(self):
        project = {"name": "测试片", "shots": [{"id": "s1", "name": "开场", "mode": "fl2v",
                                             "prompt": "转身", "duration": 5}], "characters": [], "entities": []}
        state = {"projects": {"p1": project}, "assets": {}, "runs": {}}
        with patch.object(product_agent, "read", return_value=state), \
             patch.object(product_agent, "deepseek_key", return_value="test-key"), \
             patch.object(product_agent, "_deepseek_complete", return_value="在创作页选择首尾帧。") as complete:
            result = product_agent.chat({"message": "当前镜头如何生成？", "project_id": "p1", "shot_id": "s1"})
        self.assertEqual(result["answer"], "在创作页选择首尾帧。")
        self.assertTrue(result["read_only"])
        self.assertIn("selected_shot", complete.call_args.args[0][-1]["content"])

    def test_invalid_history_and_question_rejected(self):
        with self.assertRaises(ValueError):
            product_agent.chat({"message": ""})
        with self.assertRaises(ValueError):
            product_agent.chat({"message": "帮助", "history": "bad"})

    def test_selected_provider_without_key_does_not_call_model(self):
        with patch.object(product_agent, "prompt_provider", return_value="mimo"), \
             patch.object(product_agent, "mimo_key", return_value=""), \
             patch.object(product_agent, "deepseek_key", return_value="other-provider-key"), \
             patch.object(product_agent, "_deepseek_complete") as complete:
            result = product_agent.chat({"message": "设置在哪里？"})
        self.assertIn("未配置", result["answer"])
        complete.assert_not_called()

    def test_write_request_only_returns_proposal(self):
        catalog = {"project": {"id": "p1", "name": "片子", "shots": [{"index": 1, "id": "s1",
                    "name": "镜头1", "mode": "t2v", "duration": 5}]}, "assets": []}
        proposal = {"token": "preview-token", "summary": "修改镜头 1 的时长：5 → 6"}
        with patch.object(product_agent, "catalog_for_planner", return_value=catalog), \
             patch.object(product_agent, "_deepseek_complete", return_value='{"type":"action","name":"set_shot_field","args":{"shot_id":"s1","field":"duration","value":6}}'), \
             patch.object(product_agent, "propose", return_value=proposal) as propose:
            result = product_agent.chat({"message": "把镜头1时长改成6秒", "project_id": "p1", "shot_id": "s1"})
        self.assertEqual(result["proposal"], proposal)
        propose.assert_called_once()

    def test_action_intent_does_not_depend_on_keyword_list(self):
        catalog = {"project": {"id": "p1", "name": "片子", "shots": [
            {"index": 1, "id": "s1", "name": "镜头1", "mode": "t2v", "duration": 5}]}, "assets": []}
        with patch.object(product_agent, "catalog_for_planner", return_value=catalog), \
             patch.object(product_agent, "prompt_provider", return_value="mimo"), \
             patch.object(product_agent, "mimo_key", return_value="configured"), \
             patch.object(product_agent, "_deepseek_complete", return_value='{"type":"action","name":"set_shot_field","args":{"shot_id":"s1","field":"duration","value":6}}'), \
             patch.object(product_agent, "propose", return_value={"token": "preview", "summary": "修改时长"}):
            result = product_agent.chat({"message": "把第1个镜头时长改为6秒",
                                         "project_id": "p1", "shot_id": "s1"})
        self.assertEqual(result["proposal"]["token"], "preview")


if __name__ == "__main__":
    unittest.main()

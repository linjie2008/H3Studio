import unittest
import json
import urllib.error
from unittest.mock import patch

import prompt_tools
from app import enhancement_references


class PromptEnhancementTests(unittest.TestCase):
    def test_fl2v_context_ignores_stale_gallery_references(self):
        project = {"id": "p", "reference_image_numbers": {
            "first": 1, "last": 2, "stale3": 3, "stale4": 4}}
        catalog = {key: {"name": key, "kind": "image", "project_id": "p"}
                   for key in ("first", "last", "stale3", "stale4")}
        shot = {"mode": "fl2v", "first_image": "first", "last_image": "last",
                "ref_images": ["stale3", "stale4"]}
        refs = enhancement_references(shot, project, catalog, set())
        self.assertEqual([(item["asset_id"], item["role"]) for item in refs],
                         [("first", "first_frame"), ("last", "last_frame")])

    def test_fl2v_optimization_drops_unbound_picture_tags(self):
        context = {"references": [
            {"tag": "<Picture 1>", "asset_id": "first", "role": "first_frame"},
            {"tag": "<Picture 2>", "asset_id": "last", "role": "last_frame"}],
            "segment": {"position": 1, "next_shot": {"name": "second", "mode": "fl2v"}}}
        with patch.object(prompt_tools, "load_skill_context", return_value="Skill rules"), \
             patch.object(prompt_tools, "_deepseek_complete",
                          return_value="人物 <Picture 3> 走近，随后 <Picture 4> 出现。") as complete:
            result = prompt_tools.enhance(
                "首帧人物 <Picture 3> 向前，尾帧人物 <Picture 4> 出现。",
                "fl2v", duration=5, reference_context=context, output_language="zh")
        self.assertNotIn("<Picture", result)
        self.assertNotIn("<Picture 3>", complete.call_args.args[0][1]["content"])
        self.assertNotIn("<Picture 4>", complete.call_args.args[0][1]["content"])
        self.assertNotIn("<Picture 1>", complete.call_args.args[0][1]["content"])

    def test_mixed_creative_sequence_cannot_copy_other_shots_picture_tags(self):
        shots = [
            {"id": "first", "prompt": "First person <Picture 3> walks.", "mode": "fl2v",
             "duration": 5, "context": {"references": [
                 {"tag": "<Picture 1>", "role": "first_frame"},
                 {"tag": "<Picture 2>", "role": "last_frame"}]}},
            {"id": "second", "prompt": "Second person <Picture 4> walks.", "mode": "fl2v",
             "duration": 5, "context": {"references": [
                 {"tag": "<Picture 2>", "role": "first_frame"},
                 {"tag": "<Picture 1>", "role": "last_frame"}]}},
        ]
        output = json.dumps({"prompts": [
            {"id": "first", "prompt": "First person <Picture 4> walks."},
            {"id": "second", "prompt": "Second person <Picture 3> walks."}]})
        with patch.object(prompt_tools, "load_skill_context", return_value="Skill rules"), \
             patch.object(prompt_tools, "_deepseek_complete", return_value=output) as complete:
            results = prompt_tools.enhance_sequence(shots, sequence_brief="Two people cross paths")
        self.assertTrue(all("<Picture" not in item for item in results))
        request = complete.call_args.args[0][1]["content"]
        self.assertNotIn("<Picture 3>", request)
        self.assertNotIn("<Picture 4>", request)

    def test_multi_reference_context_is_sent_to_deepseek(self):
        captured = {}

        def complete(messages, max_tokens, timeout, json_mode=False):
            captured['messages'] = messages
            captured['max_tokens'] = max_tokens
            return 'A woman in a red coat walks through the station, matching <Picture 1>.'

        context = {
            'references': [{'tag': '<Picture 1>', 'type': 'image',
                            'name': 'hero portrait.png', 'role': 'fully_preserved'}],
            'characters': [{'name': 'Lin', 'appearance': 'short black hair'}],
            'entities': [],
        }
        with patch.object(prompt_tools, 'load_skill_context', return_value='Skill rules'), \
             patch.object(prompt_tools, '_deepseek_complete', side_effect=complete):
            output = prompt_tools.enhance('女孩走进车站', 'r2v', 'custom-skill', 5, context)
        self.assertIn('<Picture 1>', output)
        request = captured['messages'][1]['content']
        self.assertIn('hero portrait.png', request)
        self.assertIn('fully_preserved', request)
        self.assertIn('short black hair', request)
        self.assertIn('metadata only', request)
        system = captured['messages'][0]['content']
        self.assertIn("Apply the selected Skill's timing and directing rules", system)
        self.assertIn("Selected H3 writing Skill", system)
        self.assertIn("Target duration: 5 seconds", system)
        self.assertNotIn("Do not replace or redesign their background", system)
        self.assertNotIn("never force a person", system)
        self.assertNotIn("SUBJECT ISOLATION WITH CONTINUITY", system)

    def test_existing_reference_tags_must_be_preserved(self):
        with patch.object(prompt_tools, 'load_skill_context', return_value='Skill rules'), \
             patch.object(prompt_tools, '_deepseek_complete', return_value='A cinematic shot.'):
            with self.assertRaisesRegex(RuntimeError, '参考素材标记'):
                prompt_tools.enhance('Use <Picture 2> as the character reference.', 'r2v')

    def test_skill_can_add_only_selected_reference_tags(self):
        context = {'references': [
            {'tag': '<Picture 1>', 'type': 'image', 'name': 'one.png'},
            {'tag': '<Picture 2>', 'type': 'image', 'name': 'two.png'},
        ]}
        with patch.object(prompt_tools, 'load_skill_context', return_value='Skill rules'), \
             patch.object(prompt_tools, '_deepseek_complete',
                          return_value='The first subject <Picture 1> hands off to <Picture 2>.'):
            result = prompt_tools.enhance('A continuous shot with <Picture 1>.', 'r2v',
                                          'custom-skill', 4, context)
        self.assertIn('<Picture 2>', result)

    def test_enhancement_rejects_unselected_reference_tags(self):
        context = {'references': [{'tag': '<Picture 1>', 'type': 'image', 'name': 'one.png'}]}
        with patch.object(prompt_tools, 'load_skill_context', return_value='Skill rules'), \
             patch.object(prompt_tools, '_deepseek_complete',
                          return_value='A shot using <Picture 1> and <Picture 9>.'):
            with self.assertRaisesRegex(RuntimeError, '未选择的参考素材标记'):
                prompt_tools.enhance('A shot with <Picture 1>.', 'r2v',
                                     'custom-skill', 4, context)

    def test_translation_returns_chinese_without_mutating_source(self):
        source = 'The woman in <Picture 1> walks forward without stopping.'
        with patch.object(prompt_tools, 'translate_prompt_with_backend',
                          return_value=('图片中的女人不停步地向前走。', 'Qwen3 0.6B（本地）')) as translate:
            translated = prompt_tools.translate_prompt(source)
        self.assertEqual(translated, '图片中的女人不停步地向前走。')
        translate.assert_called_once_with(source)

    def test_accurate_translation_uses_cloud_and_preserves_reference_markers(self):
        source = 'The person in <Picture 1> walks toward <Picture 4> at [0.0s-1.0s].'
        translation = '<Picture 1> 中的人在 [0.0s-1.0s] 朝 <Picture 4> 走去。'
        with patch.object(prompt_tools, 'deepseek_key', return_value='configured'), \
             patch.object(prompt_tools, '_deepseek_complete', return_value=translation), \
             patch.object(prompt_tools, '_local_translation') as local:
            result, backend = prompt_tools.translate_prompt_with_backend(source, 'accurate')
        self.assertEqual(result, translation)
        self.assertIn('DeepSeek', backend)
        local.assert_not_called()

    def test_chinese_prompt_can_be_translated_to_english(self):
        source = '镜头跟随 <Picture 2> 中的人走向门口。'
        translation = 'The camera follows the person in <Picture 2> toward the door.'
        with patch.object(prompt_tools, 'deepseek_key', return_value='configured'), \
             patch.object(prompt_tools, '_deepseek_complete', return_value=translation) as cloud:
            result, backend = prompt_tools.translate_prompt_with_backend(source, 'accurate', 'en')
        self.assertEqual(result, translation)
        self.assertIn('DeepSeek', backend)
        self.assertIn('English', cloud.call_args.args[0][0]['content'])

    def test_changed_cloud_reference_marker_falls_back_to_local(self):
        source = 'The person in <Picture 1> walks.'
        with patch.object(prompt_tools, 'deepseek_key', return_value='configured'), \
             patch.object(prompt_tools, '_deepseek_complete', return_value='图片 2 中的人走路。'), \
             patch.object(prompt_tools, '_local_translation', return_value='<Picture 1> 中的人走路。'):
            result, backend = prompt_tools.translate_prompt_with_backend(source, 'accurate')
        self.assertIn('<Picture 1>', result)
        self.assertIn('本地回退', backend)

    def test_default_translation_prefers_local(self):
        source = 'The person in <Picture 1> walks.'
        with patch.object(prompt_tools, 'deepseek_key', return_value='configured'), \
             patch.object(prompt_tools, '_deepseek_complete') as cloud, \
             patch.object(prompt_tools, '_local_translation', return_value='<Picture 1> 中的人走路。') as local:
            result, backend = prompt_tools.translate_prompt_with_backend(source)
        self.assertEqual(result, '<Picture 1> 中的人走路。')
        self.assertIn('Qwen3 0.6B（本地快译）', backend)
        cloud.assert_not_called()
        local.assert_called_once_with(source)

    def test_fast_translation_prefers_local_cpu_model(self):
        with patch.object(prompt_tools, '_local_translation', return_value='本地译文'), \
             patch.object(prompt_tools, '_deepseek_complete') as cloud:
            result, backend = prompt_tools.translate_prompt_with_backend('Video prompt', 'fast')
        self.assertEqual(result, '本地译文')
        self.assertIn('Qwen3', backend)
        self.assertIn('本地快译', backend)
        cloud.assert_not_called()

    def test_fast_translation_never_sends_text_to_cloud_when_local_unavailable(self):
        with patch.object(prompt_tools, '_local_translation', return_value=None), \
             patch.object(prompt_tools, 'deepseek_key', return_value='configured'), \
             patch.object(prompt_tools, '_deepseek_complete') as cloud:
            with self.assertRaisesRegex(RuntimeError, '本地翻译不可用'):
                prompt_tools.translate_prompt_with_backend('Video prompt', 'fast')
        cloud.assert_not_called()

    def test_selected_skill_is_the_source_of_motion_and_transition_rules(self):
        skill = '遮挡转场 Prompt 编写公式; continuityOverlapFrames'
        context = {'segment': {'position': 1, 'total': 1}, 'references': [
            {'tag': '<Picture 1>', 'type': 'image', 'name': 'historical-person.png'}]}
        with patch.object(prompt_tools, 'load_skill_context', return_value=skill), \
             patch.object(prompt_tools, '_deepseek_complete',
                          return_value='The person in <Picture 1> stays in the shot.') as complete:
            output = prompt_tools.enhance('A person in <Picture 1>.', 'r2v',
                                          'custom-minimax-h3-director-longtake', 4, context)
        self.assertEqual(output, 'The person in <Picture 1> stays in the shot.')
        self.assertIsNone(complete.call_args.kwargs['max_tokens'])
        system = complete.call_args.args[0][0]['content']
        self.assertIn(skill, system)
        self.assertNotIn('fully occludes the lens', system)

    def test_reoptimizing_longtake_replaces_its_prior_timeline(self):
        skill = '遮挡转场 Prompt 编写公式; continuityOverlapFrames'
        context = {'segment': {'position': 1, 'total': 2}, 'references': [
            {'tag': '<Picture 1>', 'type': 'image', 'name': 'person.png'}]}
        captured = []

        def complete(messages, max_tokens, timeout, json_mode=False):
            captured.append((messages[0]['content'], messages[1]['content']))
            return 'The person in <Picture 1> walks forward.'

        with patch.object(prompt_tools, 'load_skill_context', return_value=skill), \
             patch.object(prompt_tools, '_deepseek_complete', side_effect=complete):
            first = prompt_tools.enhance('A person in <Picture 1>.', 'r2v',
                                         'custom-longtake', 5, context)
            second = prompt_tools.enhance(first, 'r2v',
                                          'custom-longtake', 5, context)
        self.assertEqual(second, 'The person in <Picture 1> walks forward.')
        self.assertNotIn('MANDATORY MOVEMENT', captured[1][0])
        self.assertIn(skill, captured[1][0])

    def test_longtake_timeline_uses_selected_duration(self):
        skill = '遮挡转场 Prompt 编写公式; continuityOverlapFrames; 逐秒时间轴'
        context = {'segment': {'position': 1, 'total': 1}}
        with patch.object(prompt_tools, 'load_skill_context', return_value=skill), \
             patch.object(prompt_tools, '_deepseek_complete', return_value='A person walks forward.'):
            output = prompt_tools.enhance('A person walks forward.', 't2v',
                                          'custom-minimax-h3-director-longtake', 6, context)
        self.assertEqual(output, 'A person walks forward.')

    def test_longtake_uses_person_picture_after_scene_and_fractional_duration(self):
        skill = '遮挡转场 Prompt 编写公式; continuityOverlapFrames; 逐秒时间轴'
        context = {'segment': {'position': 1, 'total': 2}, 'references': [
            {'tag': '<Picture 4>', 'type': 'image', 'description': '场景参考：焦土和烟雾'},
            {'tag': '<Picture 2>', 'type': 'image', 'description': '人物身份参考：灰色制服男子'},
        ]}
        with patch.object(prompt_tools, 'load_skill_context', return_value=skill), \
             patch.object(prompt_tools, '_deepseek_complete',
                          return_value='The person in <Picture 2> walks across <Picture 4>.'):
            output = prompt_tools.enhance('The person in <Picture 2> walks across <Picture 4>.',
                                          'r2v', 'custom-minimax-h3-director-longtake', 4.5, context)
        self.assertEqual(output, 'The person in <Picture 2> walks across <Picture 4>.')

    def test_sequence_accepts_complete_ordered_response_when_model_rewrites_ids(self):
        shots = [{"id": "shot-a", "prompt": "A", "duration": 4, "mode": "t2v"},
                 {"id": "shot-b", "prompt": "B", "duration": 4, "mode": "t2v"}]
        response = json.dumps({"prompts": [
            {"id": "1", "prompt": "First improved shot."},
            {"id": "2", "prompt": "Second improved shot."},
        ]})
        with patch.object(prompt_tools, "load_skill_context", return_value="Skill rules"), \
             patch.object(prompt_tools, "_deepseek_complete", return_value=response) as complete:
            result = prompt_tools.enhance_sequence(shots, sequence_brief="A then B")
        self.assertEqual(result, ["First improved shot.", "Second improved shot."])
        complete.assert_called_once()
        self.assertIsNone(complete.call_args.kwargs["max_tokens"])

    def test_sequence_retries_only_missing_prompts(self):
        shots = [{"id": "shot-a", "prompt": "A", "duration": 4, "mode": "t2v"},
                 {"id": "shot-b", "prompt": "B", "duration": 4, "mode": "t2v"}]
        partial = json.dumps({"prompts": [{"id": "shot-a", "prompt": "First improved shot."}]})
        repaired = json.dumps({"prompts": [{"id": "shot-b", "prompt": "Second improved shot."}]})
        with patch.object(prompt_tools, "load_skill_context", return_value="Skill rules"), \
             patch.object(prompt_tools, "_deepseek_complete", side_effect=[partial, repaired]) as complete:
            result = prompt_tools.enhance_sequence(shots, sequence_brief="A then B")
        self.assertEqual(result, ["First improved shot.", "Second improved shot."])
        self.assertEqual(complete.call_count, 2)
        self.assertTrue(all(call.kwargs["max_tokens"] is None for call in complete.call_args_list))
        self.assertIn("shot-b", complete.call_args_list[1].args[0][1]["content"])
        self.assertNotIn('"id": "shot-a"', complete.call_args_list[1].args[0][1]["content"].split('"missing_shots_to_rewrite"', 1)[1].split('"already_completed_neighbor_prompts"', 1)[0])

    def test_transient_deepseek_502_is_retried(self):
        response = unittest.mock.MagicMock()
        response.__enter__.return_value = response
        response.__exit__.return_value = False
        response.read.return_value = json.dumps({"choices": [{"message": {"content": "恢复后的回答"}}]}).encode()
        outage = urllib.error.HTTPError('https://api.deepseek.com/chat/completions', 502,
                                        'Bad Gateway', {}, None)
        with patch.object(prompt_tools, 'deepseek_key', return_value='test-key'), \
             patch.object(prompt_tools.urllib.request, 'urlopen', side_effect=[outage, response]) as call, \
             patch.object(prompt_tools.time, 'sleep'):
            result = prompt_tools._deepseek_complete([{"role": "user", "content": "test"}], 20, 10)
        self.assertEqual(result, '恢复后的回答')
        self.assertEqual(call.call_count, 2)

    def test_sequence_request_omits_application_output_token_limit(self):
        response = unittest.mock.MagicMock()
        response.__enter__.return_value = response
        response.__exit__.return_value = False
        response.read.return_value = json.dumps({"choices": [{"message": {"content": '{"prompts":[]}'}}]}).encode()
        with patch.object(prompt_tools, "deepseek_key", return_value="test-key"), \
             patch.object(prompt_tools.urllib.request, "urlopen", return_value=response) as open_request:
            prompt_tools._deepseek_complete([{"role": "user", "content": "Return JSON"}],
                                            max_tokens=None, timeout=10, json_mode=True)
        payload = json.loads(open_request.call_args.args[0].data)
        self.assertNotIn("max_tokens", payload)
        self.assertEqual(payload["response_format"], {"type": "json_object"})


if __name__ == '__main__':
    unittest.main()

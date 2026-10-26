import unittest
import os
import sys
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import patch

import prompt_tools
from engine import compile_shot_prompt
from store import new_shot


class LocalTranslationLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        interpreter = self.root / ".venv-translation/bin/python"
        interpreter.parent.mkdir(parents=True)
        interpreter.symlink_to(sys.executable)
        model = self.root / "data/translation/qwen3-0.6b-int8/model.bin"
        model.parent.mkdir(parents=True)
        model.touch()
        (self.root / "local_translation.py").write_text(
            "import json, os, sys, time\nfrom pathlib import Path\n"
            "Path(__file__).with_suffix('.pid').write_text(str(os.getpid()))\n"
            "for line in sys.stdin:\n"
            " request = json.loads(line)\n"
            " if request['text'] == 'timeout': time.sleep(60)\n"
            " if request['text'] == 'error': print(json.dumps({'error': 'failed'}), flush=True)\n"
            " elif request['text'] == 'invalid': print('invalid json', flush=True)\n"
            " else: print(json.dumps({'translation': request['text'].replace('walks', '走路')}), flush=True)\n"
        )
        self.module_path = patch.object(prompt_tools, "__file__", str(self.root / "prompt_tools.py"))
        self.module_path.start()

    def tearDown(self):
        self.module_path.stop()
        self.temporary.cleanup()

    def assert_worker_exited(self):
        pid = int((self.root / "local_translation.pid").read_text())
        with self.assertRaises(ProcessLookupError):
            os.kill(pid, 0)

    def test_translation_success_and_errors_leave_no_worker(self):
        for prompt, expected in [("<Picture 1> walks", "<Picture 1> 走路"),
                                 ("error", None), ("invalid", None)]:
            with self.subTest(prompt=prompt):
                self.assertEqual(prompt_tools._local_translation(prompt), expected)
                self.assert_worker_exited()

    def test_translation_timeout_kills_and_reaps_worker(self):
        run = subprocess.run
        def short_timeout(*args, **kwargs):
            return run(*args, **{**kwargs, "timeout": 0.2})
        with patch.object(prompt_tools.subprocess, "run", side_effect=short_timeout):
            self.assertIsNone(prompt_tools._local_translation("timeout"))
        self.assert_worker_exited()


class PromptLanguageTests(unittest.TestCase):
    def test_enhancement_uses_current_video_and_shared_project_constraints(self):
        from types import SimpleNamespace
        from server.routes import prompts
        shot = {"id": "shot", "mode": "rv2v", "duration": 5,
                "source_start": 9.5,
                "source_video": "new-video", "ref_images": ["identity"],
                "replace_person": True, "prompt": "Follow <Picture 3>."}
        project = {"id": "project", "shots": [shot], "prompt_skill": "custom-h3-character-swap",
                   "shared_prompt": "Keep the authored daylight setting in every segment.",
                   "reference_image_numbers": {"identity": 3},
                   "image_descriptions": {"identity": "拟人角色的多视图身份参考"}}
        state = {"projects": {"project": project}, "assets": {
            "identity": {"id": "identity", "kind": "image", "name": "reference.png"},
            "new-video": {"id": "new-video", "kind": "video", "name": "new_depth_transfer.mp4"}}}
        request = SimpleNamespace(
            same_origin_request=lambda: True,
            body_json=lambda: {"project_id": "project", "shot_id": "shot", "prompt": shot["prompt"]},
            json=lambda *_: None)
        with patch.object(prompts, "read", return_value=state), \
             patch.object(prompts, "prepare_shot", return_value=shot), \
             patch.object(prompts, "_shot_identity_refs", return_value=["identity"]), \
             patch.object(prompts, "list_skills", return_value=[{"id": "custom-h3-character-swap"}]), \
             patch.object(prompts, "enhance", return_value="rewritten") as enhance:
            prompts.post_api_enhance(request, "/api/enhance-prompt", [], None)
        context = enhance.call_args.args[4]
        self.assertEqual(context["project_shared_prompt"], project["shared_prompt"])
        self.assertTrue(context["replacement_settings"]["replace_person"])
        source = next(item for item in context["references"] if item["type"] == "video")
        self.assertEqual(source["asset_id"], "new-video")
        self.assertTrue(source["is_source"])
        self.assertEqual(source["source_kind"], "depth")
        self.assertEqual(source["source_start_seconds"], 9.5)
        self.assertEqual(source["source_duration_seconds"], 5)

    def test_new_shot_defaults_to_english(self):
        self.assertEqual(new_shot()["prompt_language"], "en")

    def test_optimization_requests_selected_language_and_preserves_references(self):
        for language, result, expected in (
            ("en", "The person in <Picture 2> walks toward the lens.", "English"),
            ("zh", "<Picture 2> 中的人物正面向镜头走来。", "Simplified Chinese"),
        ):
            with self.subTest(language=language), \
                 patch.object(prompt_tools, "load_skill_context", return_value="basic skill"), \
                 patch.object(prompt_tools, "_deepseek_complete", return_value=result) as complete:
                actual = prompt_tools.enhance("<Picture 2> 中的人物走来。", "r2v",
                                             output_language=language)
                self.assertEqual(actual, result)
                self.assertIn(expected, complete.call_args.args[0][0]["content"])
                self.assertIn(expected, complete.call_args.args[0][1]["content"])

    def test_chinese_long_take_overlay_and_model_prompt_stay_chinese(self):
        skill = "遮挡转场 Prompt 编写公式 continuityOverlapFrames"
        context = {"segment": {"position": 1, "total": 2},
                   "references": [{"tag": "<Picture 1>", "type": "image"}]}
        with patch.object(prompt_tools, "load_skill_context", return_value=skill), \
             patch.object(prompt_tools, "_deepseek_complete", return_value="<Picture 1> 中的人物向镜头走来。"):
            optimized = prompt_tools.enhance("<Picture 1> 中的人物走来。", "r2v",
                                            duration=5, reference_context=context,
                                            output_language="zh")
        self.assertEqual(optimized, "<Picture 1> 中的人物向镜头走来。")
        self.assertNotIn("MANDATORY MOVEMENT", optimized)
        compiled = compile_shot_prompt({"mode": "t2v", "prompt": optimized, "duration": 5,
                                        "prompt_language": "zh"})
        self.assertIn("<Picture 1> 中的人物向镜头走来。", compiled)
        self.assertNotIn("逐秒动作", compiled)


if __name__ == "__main__":
    unittest.main()

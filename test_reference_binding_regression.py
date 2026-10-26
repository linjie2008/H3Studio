"""参考图绑定与缓存键的回归测试。

覆盖这些曾经会让成片用错参考图或反复重渲染的问题：

1. 提示词里的「图片N」必须绑定到用户指的那张素材（改写侧认、绑定侧不认的写法
   会让编号悬空，进而指向另一张图）。
2. 没有素材的编号不能沿用原值，否则它会占用别人的槽位。
3. 视频编辑模式同样只发送提示词点名的参考图。
4. 源视频自带原声占用 <Audio 1>，独立音频的编号要整体后移。
5. 显存策略的瞬时读数不能进缓存键。
6. 合并镜头不能悄悄丢掉第二个镜头的参考素材设置。
7. 项目之间不能共享同一份默认列表。
"""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import store
import runner
from engine import _compile_full_reference_prompt
from reference_numbering import normalize_picture_aliases, picture_numbers
from timeline_ops import edit_timeline


class _TempStore(unittest.TestCase):
    """Point the studio store at a throwaway database."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.assets_dir = root / "assets"
        self.outputs_dir = root / "outputs"
        self.assets_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.patchers = [
            patch.object(store, "ROOT", root),
            patch.object(store, "DATABASE", root / "h3studio.sqlite3"),
            patch.object(store, "STATE", root / "state.json"),
            patch.object(store, "ASSETS", self.assets_dir),
            patch.object(store, "OUTPUTS", self.outputs_dir),
            patch.object(runner, "ASSETS", self.assets_dir),
            patch.object(runner, "OUTPUTS", self.outputs_dir),
        ]
        for patcher in self.patchers:
            patcher.start()

    def tearDown(self):
        for patcher in reversed(self.patchers):
            patcher.stop()
        self.temp_dir.cleanup()

    def seed(self):
        assets = {
            "a1": {"id": "a1", "name": "场景照", "kind": "image", "file": "a1.png", "project_id": "p1"},
            "a2": {"id": "a2", "name": "女孩正面照", "kind": "image", "file": "a2.png", "project_id": "p1"},
            "a3": {"id": "a3", "name": "男孩正面照", "kind": "image", "file": "a3.png", "project_id": "p1"},
            "c1": {"id": "c1", "name": "角色档案肖像", "kind": "image", "file": "c1.png", "project_id": "p1"},
        }
        project = {
            "id": "p1", "name": "测试项目", "shared_prompt": "", "shared_refs": {},
            "reference_image_numbers": {"a1": 1, "a2": 2, "a3": 3},
            "shots": [], "characters": [], "entities": [], "image_descriptions": {},
            "characters": [{"id": "ch1", "name": "男孩", "subject_number": 1,
                            "portrait_asset": "c1", "appearance": "十六岁男孩"}],
        }
        store.change(lambda data: (data["projects"].update({"p1": project}),
                                   data["assets"].update(assets)))
        return project, assets


class PictureNumberParsingTests(unittest.TestCase):
    def test_all_spellings_are_recognised(self):
        for text in ("<Picture 3>", "@图片3", "图片3", "参考图片3"):
            self.assertEqual(picture_numbers(text), [3], text)

    def test_tag_without_a_space_is_recognised(self):
        # 真实项目里出现过 <Picture3>：旧正则要求标签与编号之间必须有空格，
        # 于是这张图根本没被绑定，模型只能拿别的图顶上。
        self.assertEqual(picture_numbers("<Picture3>"), [3])
        self.assertEqual(picture_numbers("<Picture 3>、<Picture 4>"), [3, 4])

    def test_at_alias_is_not_read_twice(self):
        self.assertEqual(picture_numbers("@图片3"), [3])

    def test_plain_alias_is_normalized(self):
        self.assertEqual(normalize_picture_aliases("参考图片3"), "参考@图片3")
        self.assertEqual(normalize_picture_aliases("@图片3"), "@图片3")
        self.assertEqual(normalize_picture_aliases("<Picture 3>"), "<Picture 3>")


class PictureBindingTests(_TempStore):
    def test_plain_chinese_alias_binds_the_named_asset(self):
        project, assets = self.seed()
        shot = {"id": "s1", "name": "镜头1", "mode": "r2v",
                "prompt": "让男孩保持 参考图片3 的外观", "ref_images": ["a3"],
                "duration": 5, "character_ids": [], "entity_ids": []}
        prepared = runner.prepare_shot(shot, project)
        self.assertIn("a3", prepared["ref_images"])
        # 绑定的图必须真的被挂到 H3 的输入里，而不是只在提示词里留个编号。
        self.assertIn("<Picture", prepared["prompt"])

    def test_tag_without_a_space_binds_the_named_asset(self):
        project, assets = self.seed()
        shot = {"id": "s1", "name": "镜头1", "mode": "r2v",
                "prompt": "驾驶位上的飞行员，面容以 <Picture3> 为准",
                "ref_images": [], "duration": 5,
                "character_ids": [], "entity_ids": []}
        prepared = runner.prepare_shot(shot, project)
        # <Picture3> 必须被当成真标签：图3 得进 ref_images，提示词也得落到它的槽位。
        self.assertIn("a3", prepared["ref_images"])
        self.assertNotIn("<Picture3>", prepared["prompt"])

    def test_prompt_slot_points_at_the_named_image(self):
        project, assets = self.seed()
        shot = {"id": "s1", "name": "镜头1", "mode": "r2v",
                "prompt": "让男孩保持 @图片3 的外观", "ref_images": ["a3"],
                "duration": 5, "character_ids": ["ch1"], "entity_ids": []}
        prepared = runner.prepare_shot(shot, project)
        mounted = prepared["ref_images"]
        self.assertEqual(assets[mounted[0]]["name"], "男孩正面照")
        # 用户写的那句必须指向他点名的那张图。
        sentence = [line for line in prepared["prompt"].splitlines() if "让男孩保持" in line][0]
        number = int(sentence.split("<Picture ")[1].split(">")[0])
        self.assertEqual(assets[mounted[number - 1]]["name"], "男孩正面照")

    def test_unknown_number_fails_loudly_in_reference_mode(self):
        project, assets = self.seed()
        shot = {"id": "s1", "name": "镜头1", "mode": "r2v",
                "prompt": "参考图片9 的构图", "ref_images": [],
                "duration": 5, "character_ids": [], "entity_ids": []}
        # 编号不存在时必须报错，而不是悄悄指向别的图。
        with self.assertRaises(ValueError):
            runner.prepare_shot(shot, project)

    def test_creative_mode_sends_no_reference_images(self):
        project, assets = self.seed()
        shot = {"id": "s1", "name": "镜头1", "mode": "t2v",
                "prompt": "参考图片3 的构图", "ref_images": ["a3"],
                "duration": 5, "character_ids": [], "entity_ids": []}
        prepared = runner.prepare_shot(shot, project)
        self.assertEqual(prepared["ref_images"], [])

    def test_video_edit_mode_binds_only_the_tagged_asset(self):
        project, assets = self.seed()
        project["shared_refs"] = {"images": ["a1"]}
        store.change(lambda data: data["projects"].update({"p1": project}))
        shot = {"id": "s1", "name": "镜头1", "mode": "v2v",
                "prompt": "让人物保持 @图片3 的外观", "ref_images": ["a3", "a2"],
                "source_video": "v1", "duration": 5,
                "character_ids": [], "entity_ids": []}
        prepared = runner.prepare_shot(shot, project)
        self.assertIn("a3", prepared["ref_images"])
        # 没被点名的图不能混进来，否则编号会整体错位。
        self.assertNotIn("a2", prepared["ref_images"])


class AudioNumberingTests(unittest.TestCase):
    def _prompt(self, **overrides):
        shot = {"mode": "rv2v", "source_video": "v1", "audio_mode": "source",
                "ref_images": [], "ref_videos": [], "ref_audios": ["aud1"],
                "_h3_characters": [{"id": "ch1", "name": "男孩", "subject_number": 1,
                                    "voice_asset": "aud1"}],
                "prompt": ""}
        shot.update(overrides)
        return _compile_full_reference_prompt(shot, "", "", "")

    def test_source_soundtrack_shifts_standalone_audio(self):
        prompt = self._prompt()
        # 源视频原声占 <Audio 1>，音色样本必须是 <Audio 2>。
        self.assertIn("<Audio 2> is the voice-timbre reference", prompt)
        self.assertNotIn("<Audio 1> is the voice-timbre reference", prompt)

    def test_without_source_audio_the_number_starts_at_one(self):
        prompt = self._prompt(audio_mode="generate")
        self.assertIn("<Audio 1> is the voice-timbre reference", prompt)


class CacheKeyTests(_TempStore):
    def test_memory_reading_does_not_change_the_cache_key(self):
        from memory_plan import GIB, plan_canvas
        shot = {"id": "s1", "prompt": "一只猫", "duration": 5, "ref_images": []}
        busy = dict(shot, _h3_memory_plan=plan_canvas(1920, 1088, 24 * GIB, 12.0 * GIB))
        idle = dict(shot, _h3_memory_plan=plan_canvas(1920, 1088, 24 * GIB, 7.3 * GIB))
        self.assertEqual(runner._shot_signature(busy, {}, {}),
                         runner._shot_signature(idle, {}, {}))


class TimelineMergeTests(unittest.TestCase):
    def _shot(self, shot_id, **overrides):
        shot = {"id": shot_id, "name": shot_id, "mode": "r2v", "duration": 4,
                "prompt": "内容 " + shot_id, "seed": 42, "selected": True,
                "first_image": None, "last_image": None, "source_video": None,
                "ref_images": [], "ref_videos": [], "ref_audios": [],
                "character_ids": [], "entity_ids": [], "scene_id": None,
                "audio_mode": "generate", "ref_image_size": "match",
                "emotion": "自动/未指定", "camera_move": "自动/未指定",
                "camera_speed": "自动/未指定", "action_intensity": "自然",
                "speaker_id": "S1", "speaker_description": "", "dialogue_text": "",
                "dialogue_language": "Chinese", "dialogue_delivery": "",
                "dialogue_time": "", "dialogue_voiceover": False,
                "dialogue_cutoff": False, "dialogue_carries_out": False,
                "on_screen_text": "", "overall_soundscape": "", "non_diegetic_music": "",
                "ref_image_roles": {}, "ref_video_roles": {}, "ref_audio_roles": {}}
        shot.update(overrides)
        return shot

    def test_merge_refuses_shots_with_different_reference_settings(self):
        shots = [self._shot("a", manual_ref_images=["img1"]),
                 self._shot("b", manual_ref_images=["img2"])]
        with self.assertRaises(ValueError):
            edit_timeline(shots, ["a", "b"], "merge")

    def test_merge_refuses_shots_with_different_anchors(self):
        shots = [self._shot("a", h3_guides=[{"image_id": "img1", "time_seconds": 1}]),
                 self._shot("b", h3_guides=[{"image_id": "img2", "time_seconds": 2}])]
        with self.assertRaises(ValueError):
            edit_timeline(shots, ["a", "b"], "merge")

    def test_merge_refuses_a_zero_length_shot(self):
        shots = [self._shot("a", duration=0), self._shot("b", duration=4)]
        with self.assertRaises(ValueError):
            edit_timeline(shots, ["a", "b"], "merge")

    def test_merge_records_the_absorbed_shot_ids(self):
        shots = [self._shot("a"), self._shot("b")]
        merged = edit_timeline(shots, ["a", "b"], "merge")
        self.assertEqual(merged[0]["_merged_shot_ids"], ["b"])


class StoreIsolationTests(_TempStore):
    def test_projects_do_not_share_the_default_collections(self):
        project = store.new_project("项目甲")
        project["id"] = "p1"
        other = store.new_project("项目乙")
        other["id"] = "p2"
        # 模拟一个缺 characters 的旧项目行。
        stripped = {key: value for key, value in project.items() if key != "characters"}
        store.change(lambda data: data["projects"].update({"p1": stripped, "p2": other}))
        data = store.read()
        self.assertIsNot(data["projects"]["p1"]["characters"],
                         data["projects"]["p2"]["characters"])
        data["projects"]["p1"]["characters"].append({"id": "c1"})
        self.assertEqual(len(store.read()["projects"]["p2"]["characters"]), 0)


if __name__ == "__main__":
    unittest.main()

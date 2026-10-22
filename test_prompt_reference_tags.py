"""提示词增强的参考标签校验回归测试。

1. 有素材支撑的标签（图片身份图）必须原样保留。
2. 悬空标签（例如只绑了两段视频却写了 <Video 9>）允许模型改写成有效编号，
   与 runner.localize_video_tags 的口径一致——引擎本来就会把它指回源视频。
3. 无论哪种情况，某一类引用（图片/视频/声音）都不能被整类丢掉。
"""
import unittest

from prompt_tools import _reference_tag_sets, _require_reference_kinds


class ReferenceTagSetTests(unittest.TestCase):
    def setUp(self):
        self.references = [
            {"tag": "<Picture 1>", "type": "image"},
            {"tag": "<Video 1>", "type": "video"},
            {"tag": "<Video 2>", "type": "video"},
        ]
        self.pattern = __import__("re").compile(r"<(?:Picture|Video|Audio)\s+\d+>")

    def test_bound_tags_must_survive(self):
        authored = "the face matches <Picture 1> exactly, the camera holds to <Video 1>"
        required, allowed = _reference_tag_sets(authored, self.references)
        self.assertEqual(required, {"<Picture 1>", "<Video 1>"})
        self.assertIn("<Picture 1>", allowed)
        self.assertIn("<Video 2>", allowed)

    def test_dangling_video_tag_is_not_required(self):
        authored = "the camera holds to <Video 9> and the face matches <Picture 1>"
        required, allowed = _reference_tag_sets(authored, self.references)
        self.assertNotIn("<Video 9>", required)
        self.assertIn("<Picture 1>", required)

    def test_dangling_picture_tag_is_not_required(self):
        references = [{"tag": "<Picture 1>", "type": "image"}]
        required, _ = _reference_tag_sets("the face matches <Picture 7>", references)
        self.assertNotIn("<Picture 7>", required)

    def test_missing_kind_is_rejected(self):
        authored = "the camera holds to <Video 9>"
        with self.assertRaises(RuntimeError):
            _require_reference_kinds(authored, set(), self.pattern)

    def test_renumbered_dangling_tag_is_accepted(self):
        authored = "the camera holds to <Video 9>"
        _require_reference_kinds(authored, {"<Video 1>"}, self.pattern)

    def test_tag_without_material_is_still_required_when_no_context(self):
        # 没有上下文时无法判断编号是否有效，保守起见仍要求原样保留。
        required, _ = _reference_tag_sets("the face matches <Picture 3>", [])
        self.assertIn("<Picture 3>", required)


if __name__ == "__main__":
    unittest.main()

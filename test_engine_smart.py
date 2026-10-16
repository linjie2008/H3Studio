import unittest

from engine import build_h3_graph
from runner import _seedvr2_enlarges
from store import new_project


def graph_for(mode, *, dialogue=""):
    shot = {"mode": mode, "prompt": "A person moves naturally.", "duration": 5,
            "dialogue_text": dialogue, "seed": 7}
    media = {}
    if mode in ("i2v", "fl2v"):
        shot["first_image"] = "first"
        media["first"] = "first.png"
    if mode == "fl2v":
        shot["last_image"] = "last"
        media["last"] = "last.png"
    if mode == "r2v":
        shot["ref_images"] = ["first"]
        media["first"] = "first.png"
    if mode in ("v2v", "rv2v"):
        shot["source_video"] = "source"
        media["source"] = "source.mp4"
    return build_h3_graph(shot, {"width": 1216, "height": 672,
                                  "generation_speed": "smart",
                                  "attention_mode": "off"}, media, "smart_test")


class SmartPresetTests(unittest.TestCase):
    def test_new_projects_start_with_smart_canvas_without_seedvr2(self):
        project = new_project("test")
        self.assertEqual(project["generation_speed"], "smart")
        self.assertEqual((project["width"], project["height"]), (1216, 672))
        self.assertFalse(project["upscale"])

    def test_each_mode_uses_its_recommended_steps_and_lora(self):
        for mode, steps, turbo in (("t2v", 8, True), ("i2v", 8, True),
                                   ("fl2v", 16, False), ("r2v", 20, False),
                                   ("v2v", 20, False), ("rv2v", 20, False)):
            with self.subTest(mode=mode):
                nodes = list(graph_for(mode).values())
                self.assertEqual(next(n["inputs"]["steps"] for n in nodes
                                      if n["class_type"] == "BasicScheduler"), steps)
                self.assertEqual(any(n["class_type"] == "LoraLoaderModelOnly"
                                     for n in nodes), turbo)

    def test_dialogue_uses_native_sixteen_steps(self):
        nodes = list(graph_for("i2v", dialogue="你好").values())
        self.assertEqual(next(n["inputs"]["steps"] for n in nodes
                              if n["class_type"] == "BasicScheduler"), 16)
        self.assertFalse(any(n["class_type"] == "LoraLoaderModelOnly" for n in nodes))

    def test_seedvr2_only_when_target_larger_than_source(self):
        self.assertFalse(_seedvr2_enlarges(1344, 768, 720, 1280))
        self.assertTrue(_seedvr2_enlarges(1344, 768, 1080, 1920))
        self.assertFalse(_seedvr2_enlarges(1344, 768, 1080, 1280))


if __name__ == "__main__":
    unittest.main()

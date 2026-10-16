import unittest

from engine import build_h3_graph, GOLDFISH_MODEL, GOLDFISH_MODELS, GOLDFISH_CLIP, GOLDFISH_VAE


class GoldfishGraphTests(unittest.TestCase):
    def test_reference_uses_goldfish_hybrid_selflift_path(self):
        shot = {"mode": "r2v", "prompt": "A moving portrait.", "duration": 5,
                "ref_images": ["portrait"], "seed": 19}
        settings = {"width": 1664, "height": 928, "generation_speed": "goldfish",
                    "h3_lora_mystic": True, "refine_mode": "same", "selflift": True}
        graph = build_h3_graph(shot, settings, {"portrait": "portrait.png"}, "test")
        nodes = list(graph.values())
        by_type = lambda kind: [node["inputs"] for node in nodes if node["class_type"] == kind]
        self.assertEqual(by_type("UNETLoader")[0]["unet_name"], GOLDFISH_MODEL)
        self.assertEqual(by_type("MiniMaxH3ReferenceToVideo")[0]["ref_image_size"], "max")
        self.assertEqual(by_type("BasicScheduler")[0]["scheduler"], "beta")
        self.assertEqual(by_type("BasicScheduler")[0]["steps"], 10)
        self.assertEqual(by_type("KSamplerSelect")[0]["sampler_name"], "euler")
        self.assertEqual(by_type("CLIPLoader")[0]["clip_name"], GOLDFISH_CLIP)
        self.assertEqual(by_type("VAELoader")[0]["vae_name"], GOLDFISH_VAE)
        self.assertEqual(len(by_type("SamplerCustomAdvanced")), 0)
        self.assertEqual(len(by_type("SelfLiftH3Sampler")), 1)
        self.assertEqual(by_type("SelfLiftH3Sampler")[0]["transition_step"], 6)
        self.assertEqual(by_type("SelfLiftH3Sampler")[0]["lowres_scale"], 0.6)
        self.assertEqual(by_type("SelfLiftH3Sampler")[0]["upscaler_model"], "h3_upscaler_lms_v0.1.safetensors")
        self.assertEqual(len(by_type("BlockSparseAttention")), 1)
        self.assertEqual(len(by_type("MiniMaxChunkFeedForward")), 1)
        loras = {node["lora_name"]: node["strength_model"] for node in by_type("LoraLoaderModelOnly")}
        self.assertEqual(loras["minimax_h3/minimax_h3_dmad_full_dareties_v4_step600_comfy_fro.safetensors"], 0.8)
        self.assertEqual(loras["minimax_h3/MysticXXX_MMH3-V1.safetensors"], 0.45)
        self.assertEqual(len(by_type("SaveVideo")), 1)

    def test_creative_mode_uses_goldfish_hybrid_model(self):
        graph = build_h3_graph({"mode": "t2v", "prompt": "A scene"},
                               {"width": 1664, "height": 928, "generation_speed": "goldfish"},
                               {}, "test")
        nodes = list(graph.values())
        self.assertEqual(next(n["inputs"]["unet_name"] for n in nodes
                              if n["class_type"] == "UNETLoader"), GOLDFISH_MODELS["creative"])
        self.assertTrue(any(n["class_type"] == "MiniMaxH3ImageToVideo" for n in nodes))
        self.assertTrue(any(n["class_type"] == "SelfLiftH3Sampler" for n in nodes))

    def test_action_loras_are_explicitly_opt_in(self):
        shot = {"mode": "r2v", "prompt": "A short scene", "duration": 5,
                "ref_images": ["portrait"]}
        graph = build_h3_graph(shot, {"width": 1664, "height": 928,
                                      "generation_speed": "goldfish",
                                      "goldfish_action_loras": True},
                               {"portrait": "portrait.png"}, "test")
        names = [node["inputs"]["lora_name"] for node in graph.values()
                 if node["class_type"] == "LoraLoaderModelOnly"]
        self.assertIn("minimax_h3/H3_Combat_V2.safetensors", names)
        self.assertIn("minimax_h3/Motion_Repair_V2.safetensors", names)
        self.assertIn("minimax_h3/H3_Fight_Flow_V1.safetensors", names)

    def test_action_loras_are_explicitly_opt_in(self):
        shot = {"mode": "r2v", "prompt": "A short scene", "duration": 5,
                "ref_images": ["portrait"]}
        graph = build_h3_graph(shot, {"width": 1664, "height": 928,
                                      "generation_speed": "goldfish",
                                      "goldfish_action_loras": True},
                               {"portrait": "portrait.png"}, "test")
        names = [node["inputs"]["lora_name"] for node in graph.values()
                 if node["class_type"] == "LoraLoaderModelOnly"]
        self.assertIn("minimax_h3/H3_Combat_V2.safetensors", names)
        self.assertIn("minimax_h3/Motion_Repair_V2.safetensors", names)
        self.assertIn("minimax_h3/H3_Fight_Flow_V1.safetensors", names)


if __name__ == "__main__":
    unittest.main()

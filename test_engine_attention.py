import unittest

from engine import build_h3_graph


class EngineAttentionTests(unittest.TestCase):
    def test_four_step_preset_uses_matching_creative_lora(self):
        graph = build_h3_graph(
            {"mode": "fl2v", "prompt": "A person walks toward the camera.",
             "duration": 4, "first_image": "start", "last_image": "end"},
            {"width": 480, "height": 864, "generation_speed": "turbo4"},
            {"start": "start.png", "end": "end.png"}, "four_step_test",
        )
        lora = next(node for node in graph.values() if node["class_type"] == "LoraLoaderModelOnly")
        scheduler = next(node for node in graph.values() if node["class_type"] == "BasicScheduler")
        self.assertEqual(lora["inputs"]["lora_name"],
                         "minimax_h3/minimax_h3_fl2v_lightx2v_turbo_4step_v0.1_comfy.safetensors")
        self.assertEqual(scheduler["inputs"]["steps"], 4)

    def test_four_step_preset_uses_reference_lora_in_all_reference_modes(self):
        for mode in ("r2v", "v2v", "rv2v"):
            with self.subTest(mode=mode):
                shot = {"mode": mode, "prompt": "A person walks toward the camera.",
                        "duration": 4, "ref_images": ["person"]}
                if mode != "r2v":
                    shot["source_video"] = "source"
                media = {"person": "person.png", "source": "source.mp4"}
                graph = build_h3_graph(
                    shot, {"width": 480, "height": 864, "generation_speed": "turbo4"},
                    media, "reference_speed_test",
                )
                lora = next(node for node in graph.values() if node["class_type"] == "LoraLoaderModelOnly")
                scheduler = next(node for node in graph.values() if node["class_type"] == "BasicScheduler")
                self.assertEqual(lora["inputs"]["lora_name"],
                                 "minimax_h3/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors")
                self.assertEqual(scheduler["inputs"]["steps"], 4)

    def test_reference_four_step_choice_is_not_overridden_for_person_replacement(self):
        for dmad, expected in (
            (False, "minimax_h3/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors"),
            (True, "minimax_h3/minimax_h3_dmad_full_dareties_v4_step600_comfy_fro.safetensors"),
        ):
            with self.subTest(dmad=dmad):
                graph = build_h3_graph(
                    {"mode": "rv2v", "prompt": "Replace the person with the reference character.",
                     "duration": 4, "source_video": "source", "ref_images": ["person"],
                     "replace_person": True},
                    {"width": 480, "height": 864, "generation_speed": "turbo4",
                     "h3_lora_dmad": dmad},
                    {"source": "source.mp4", "person": "person.png"}, "replacement_speed_test",
                )
                loras = [node["inputs"]["lora_name"] for node in graph.values()
                         if node["class_type"] == "LoraLoaderModelOnly"]
                scheduler = next(node for node in graph.values() if node["class_type"] == "BasicScheduler")
                self.assertIn(expected, loras)
                self.assertEqual(scheduler["inputs"]["steps"], 4)

    def test_faceswap_variant_uses_author_lora_and_trigger(self):
        shot = {"mode": "rv2v", "prompt": "Replace the person with the reference character.",
                "duration": 4, "source_video": "source", "ref_images": ["person"],
                "replace_person": True, "swap_lora": "faceswap"}
        graph = build_h3_graph(
            shot, {"width": 480, "height": 864, "generation_speed": "turbo4"},
            {"source": "source.mp4", "person": "person.png"}, "faceswap_test",
        )
        loras = [node["inputs"]["lora_name"] for node in graph.values()
                 if node["class_type"] == "LoraLoaderModelOnly"]
        self.assertIn("minimax_h3/SS_FaceSwap_MiniMax_H3_REF2VA.safetensors", loras)
        self.assertNotIn("minimax_h3/h3_character_swap_pro4500_1000.safetensors", loras)
        self.assertIn("minimax_h3/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors", loras)
        reference = next(node for node in graph.values()
                         if node["class_type"] == "MiniMaxH3ReferenceToVideo")
        self.assertIn("[Shot 1] Faceswap.", reference["inputs"]["prompt"])
        scheduler = next(node for node in graph.values() if node["class_type"] == "BasicScheduler")
        self.assertEqual(scheduler["inputs"]["steps"], 4)

    def test_faceswap_selection_is_loaded_in_reference_and_goldfish_modes(self):
        for mode, speed in (("r2v", "turbo4"), ("rv2v", "goldfish")):
            with self.subTest(mode=mode, speed=speed):
                shot = {"mode": mode, "prompt": "Use the reference person's face.",
                        "duration": 4, "ref_images": ["person"],
                        "replace_person": True, "swap_lora": "faceswap"}
                if mode == "rv2v":
                    shot["source_video"] = "source"
                graph = build_h3_graph(
                    shot, {"width": 480, "height": 864, "generation_speed": speed,
                           "h3_lora_character_swap": False},
                    {"person": "person.png", "source": "source.mp4"}, "faceswap_mode_test",
                )
                loras = [node["inputs"]["lora_name"] for node in graph.values()
                         if node["class_type"] == "LoraLoaderModelOnly"]
                self.assertIn("minimax_h3/SS_FaceSwap_MiniMax_H3_REF2VA.safetensors", loras)

    def test_faceswap_author_prompt_preset_sends_only_trigger(self):
        graph = build_h3_graph(
            {"mode": "rv2v", "prompt": "Replace face and clothes with the reference.",
             "duration": 3, "source_video": "source", "ref_images": ["person"],
             "replace_person": True, "swap_lora": "faceswap",
             "faceswap_trigger_only": True},
            {"width": 480, "height": 864, "generation_speed": "turbo"},
            {"source": "source.mp4", "person": "person.png"}, "faceswap_simple_test",
        )
        reference = next(node for node in graph.values()
                         if node["class_type"] == "MiniMaxH3ReferenceToVideo")
        self.assertEqual(reference["inputs"]["prompt"], "Faceswap")
        scheduler = next(node for node in graph.values() if node["class_type"] == "BasicScheduler")
        self.assertEqual(scheduler["inputs"]["steps"], 8)

    def test_veda_sparse_attention_is_applied_after_turbo_lora(self):
        graph = build_h3_graph(
            {"mode": "t2v", "prompt": "A person walks through a quiet garden.", "duration": 4},
            {"width": 480, "height": 864, "generation_speed": "turbo", "attention_mode": "veda"},
            {}, "attention_test",
        )
        nodes = list(graph.values())
        lora_index = next(i for i, item in enumerate(nodes) if item["class_type"] == "LoraLoaderModelOnly")
        veda_index = next(i for i, item in enumerate(nodes) if item["class_type"] == "VedaSparseAttention")
        sampler_index = next(i for i, item in enumerate(nodes) if item["class_type"] == "SamplerCustomAdvanced")
        self.assertLess(lora_index, veda_index)
        self.assertLess(veda_index, sampler_index)
        veda = nodes[veda_index]["inputs"]
        self.assertEqual(veda["generated_sparsity"], "90%")
        self.assertEqual(veda["reference_sparsity"], "90%")
        self.assertEqual(veda["predictor"], "minimax_h3_t2va_veda_8nfe_600step_preview_fp8.safetensors")

    def test_h3_clip_model_selection(self):
        default_graph = build_h3_graph(
            {"mode": "t2v", "prompt": "test prompt", "duration": 4},
            {"width": 480, "height": 864},
            {}, "clip_test_default",
        )
        clip_loader = next(item for item in default_graph.values() if item["class_type"] == "CLIPLoader")
        self.assertEqual(clip_loader["inputs"]["clip_name"], "qwen3vl_32b_heretic_minimax_h3_nvfp4.safetensors")

        custom_graph = build_h3_graph(
            {"mode": "t2v", "prompt": "test prompt", "duration": 4},
            {"width": 480, "height": 864, "h3_clip_model": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"},
            {}, "clip_test_custom",
        )
        custom_clip = next(item for item in custom_graph.values() if item["class_type"] == "CLIPLoader")
        self.assertEqual(custom_clip["inputs"]["clip_name"], "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors")


if __name__ == "__main__":
    unittest.main()

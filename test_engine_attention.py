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

    def test_four_step_preset_uses_reference_eight_step_lora_for_reference_mode(self):
        graph = build_h3_graph(
            {"mode": "r2v", "prompt": "A person walks toward the camera.",
             "duration": 4, "ref_images": ["person"]},
            {"width": 480, "height": 864, "generation_speed": "turbo4"},
            {"person": "person.png"}, "reference_speed_test",
        )
        lora = next(node for node in graph.values() if node["class_type"] == "LoraLoaderModelOnly")
        scheduler = next(node for node in graph.values() if node["class_type"] == "BasicScheduler")
        self.assertEqual(lora["inputs"]["lora_name"],
                         "minimax_h3/minimax_h3_ref2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors")
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

"""Independent H3 task graphs built from ComfyUI's official nodes."""

from __future__ import annotations

import math
import re


CREATIVE = {"t2v", "i2v", "fl2v"}
REFERENCE = {"r2v", "v2v", "rv2v"}
ALL_MODES = CREATIVE | REFERENCE

MODELS = {
    "creative": {
        "unet": "Minimax_H3/minimax_h3_fl2va_pruned_int8_convrot.safetensors",
        "lora": "minimax_h3/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",
        "lora_4step": "minimax_h3/minimax_h3_fl2v_lightx2v_turbo_4step_v0.1_comfy.safetensors",
    },
    "reference": {
        "unet": "Minimax_H3/minimax_h3_ref2va_pruned_int8_convrot.safetensors",
        "lora": "minimax_h3/minimax_h3_ref2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors",
    },
}

# Optional project-level H3 LoRAs.  These are deliberately separate from the
# sampler-speed LoRA above: users can turn them on for a comparison without
# changing the project's chosen 4/8/20-step sampling policy.
OPTIONAL_H3_LORAS = {
    "dmad": "minimax_h3/minimax_h3_dmad_full_dareties_v4_step600_comfy_fro.safetensors",
    "mystic": "minimax_h3/MysticXXX_MMH3-V1.safetensors",
}

# The YZ Goldfish workflow uses one hybrid FL2VA/Ref2VA checkpoint for both
# creative and reference-conditioned shots, plus the H3 SelfLift sampler.
GOLDFISH_MODEL = "Minimax_H3/minimax_h3_hybrid_fl2va_ref2va_b25-49-int8.safetensors"
GOLDFISH_MODELS = {family: GOLDFISH_MODEL for family in MODELS}
GOLDFISH_CLIP = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
GOLDFISH_VAE = "minimax_h3_video_vae_fp16.safetensors"
GOLDFISH_UPSCALER = "h3_upscaler_lms_v0.1.safetensors"
GOLDFISH_ACTION_LORAS = (
    ("minimax_h3/H3_Combat_V2.safetensors", 0.65),
    ("minimax_h3/Motion_Repair_V2.safetensors", 0.35),
    ("minimax_h3/H3_Fight_Flow_V1.safetensors", 0.35),
)

# Recommended starting points, not universal optima.  Creative one-frame
# tasks benefit from the installed 8-step LoRA; two-frame handoffs and
# reference/edit tasks keep more native denoising steps for conditioning.
SMART_PRESETS = {
    "t2v": (8, True),
    "i2v": (8, True),
    "fl2v": (16, False),
    "r2v": (20, False),
    "v2v": (20, False),
    "rv2v": (20, False),
}


def h3_frames(seconds: float) -> int:
    """Round up to MiniMax H3's 17k+5 grid at 24 fps."""
    requested = max(5, math.ceil(float(seconds) * 24))
    return requested + (5 - requested) % 17


def safe_prefix(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9_-]+", "_", value).strip("_")
    return value[:70] or "h3_studio"


class Graph:
    def __init__(self):
        self.nodes: dict[str, dict] = {}
        self.next_id = 1

    def add(self, kind: str, **inputs):
        node_id = str(self.next_id)
        self.next_id += 1
        self.nodes[node_id] = {"class_type": kind, "inputs": inputs}
        return [node_id, 0]


def build_h3_graph(shot: dict, settings: dict, media: dict[str, str], prefix: str) -> dict:
    """Build one shot from official MiniMax H3 nodes, without Director nodes."""
    mode = shot.get("mode", "t2v")
    if mode not in ALL_MODES:
        raise ValueError(f"未知任务模式：{mode}")
    prompt = compile_shot_prompt(shot)
    if not prompt:
        raise ValueError("镜头提示词不能为空")
    width = int(settings.get("width", 1344))
    height = int(settings.get("height", 768))
    if width < 32 or height < 32 or width % 32 or height % 32:
        raise ValueError("画布宽高需为 32 的倍数")
    context_frames = int(shot.get("continuity_frames") or 0) if shot.get("continuity_video") else 0
    length = h3_frames(float(shot.get("duration", 5)) + context_frames / 24)
    if length > 3600:
        raise ValueError("单镜头时长超过 H3 节点上限")
    seed = int(shot.get("seed", 42))
    family = "creative" if mode in CREATIVE else "reference"
    speed = str(settings.get("generation_speed", "turbo"))
    goldfish = speed == "goldfish"
    model_names = MODELS[family]

    g = Graph()
    model = g.add("UNETLoader", unet_name=GOLDFISH_MODELS[family] if goldfish else model_names["unet"], weight_dtype="default")
    if speed not in ("turbo4", "turbo", "quality", "goldfish", "smart"):
        raise ValueError("H3 生成速度无效，请重新选择")
    smart = speed == "smart"
    turbo = SMART_PRESETS[mode][1] if smart else speed not in ("quality", "goldfish")
    steps = SMART_PRESETS[mode][0] if smart else 4 if speed == "turbo4" and family == "creative" else 8 if turbo else 20
    if smart and mode in ("t2v", "i2v") and (shot.get("dialogue_text") or "<d>" in prompt):
        # Short 8-step visual previews are a poor default for speech.
        turbo, steps = False, 16
    # Reference modes use their own eight-step LoRA, even when the project
    # selects a creative four-step variant.  A dormant choice must not block
    # mixed-mode projects or quietly add another sampler LoRA.
    use_dmad = bool(settings.get("h3_lora_dmad")) and speed == "turbo4" and family == "creative"
    if goldfish:
        # Match the supplied YZ workflow: DMAD is run at 0.8 alongside a
        # ten-step beta/Euler schedule, not used as the four-step preset.
        model = g.add("LoraLoaderModelOnly", model=model,
                      lora_name=OPTIONAL_H3_LORAS["dmad"], strength_model=0.8)
    elif turbo:
        # DMAD is itself a 4-step H3 sampler LoRA.  It replaces LightX2V;
        # loading both would apply two incompatible 4-step deltas to the
        # same model and makes any comparison meaningless.
        lora_name = (OPTIONAL_H3_LORAS["dmad"] if use_dmad
                     else model_names["lora_4step"] if steps == 4 else model_names["lora"])
        model = g.add("LoraLoaderModelOnly", model=model, lora_name=lora_name, strength_model=1)
    # Artistic/action LoRAs are opt-in.  Loading them only when checked keeps
    # the default workflow's VRAM footprint and appearance unchanged.
    for key, lora_name in OPTIONAL_H3_LORAS.items():
        if key == "dmad":
            continue
        if (not goldfish and settings.get(f"h3_lora_{key}")) or (
                goldfish and key == "mystic" and settings.get("h3_lora_mystic", True)):
            strength = float(settings.get(f"h3_lora_{key}_strength", 0.45 if goldfish else 0.6))
            if not 0 <= strength <= 1.5:
                raise ValueError(f"{key} LoRA 强度需在 0 到 1.5 之间")
            model = g.add("LoraLoaderModelOnly", model=model, lora_name=lora_name,
                          strength_model=strength)
    if goldfish and settings.get("goldfish_action_loras"):
        for lora_name, strength in GOLDFISH_ACTION_LORAS:
            model = g.add("LoraLoaderModelOnly", model=model, lora_name=lora_name,
                          strength_model=strength)
    attention_mode = str(settings.get("attention_mode") or ("sage" if settings.get("sage_attention") else "off"))
    if goldfish:
        model = g.add("ModelAttentionBackend", model=model, attention="comfy kitchen attention")
        model = g.add("BlockSparseAttention", model=model,
                      selection={"selection": "sol-attn", "tau": 1.3},
                      start_percent=0.2, end_percent=1.0, dense_blocks="",
                      min_tokens=12288, extra_tokens=256,
                      sink_conditioning="exact_kv_and_rows", verbose=True)
    elif attention_mode == "veda":
        model = g.add("VedaSparseAttention", model=model,
                      predictor=str(settings.get("veda_predictor") or "minimax_h3_t2va_veda_8nfe_600step_preview_fp8.safetensors"),
                      generated_sparsity=str(settings.get("veda_generated_sparsity") or "90%"),
                      reference_sparsity=str(settings.get("veda_reference_sparsity") or "90%"),
                      full_attention_layers="", full_attention_steps="", verbose=False)
    elif attention_mode == "sage":
        model = g.add("MiniMaxH3MemoryEfficientSageAttentionPatch", model=model)
    base_model = model
    model = g.add("MiniMaxH3SigmaShift", model=model, shift_video=12, shift_audio=3)
    if goldfish:
        model = g.add("MiniMaxChunkFeedForward", model=model, chunks=2, seq_threshold=4096)
    clip_name = GOLDFISH_CLIP if goldfish else str(settings.get("h3_clip_model") or "qwen3vl_32b_heretic_minimax_h3_nvfp4.safetensors")
    clip = g.add("CLIPLoader", clip_name=clip_name, type="minimax", device="default")
    vae = g.add("VAELoader", vae_name=GOLDFISH_VAE if goldfish else str(settings.get("h3_video_vae") or "minimax_h3_video_vae_fp16.safetensors"))
    audio_vae = g.add("VAELoader", vae_name="minimax_h3_audio_vae_fp32.safetensors")

    common = {"clip": clip, "prompt": prompt, "width": width, "height": height, "length": length}
    if family == "creative":
        inputs = {**common, "vae": vae}
        # Only wire frame inputs that belong to the selected mode.  The UI
        # intentionally keeps previous selections when users switch modes, so
        # treating those dormant fields as active would silently turn t2v into
        # image-conditioned generation.
        first_id = shot.get("first_image") if mode in ("i2v", "fl2v") else None
        last_id = shot.get("last_image") if mode == "fl2v" else None
        # A continued FL2V shot has a guide clip at frame 0. Feeding its
        # separately selected first image into ImageToVideo would create a
        # second, potentially conflicting keyframe at that same frame. Give
        # the outgoing clip its own time, then leave a short interval for the
        # model to bridge into the selected first image.
        handoff_first_id = (first_id if mode == "fl2v"
                            and shot.get("continuity_video") and context_frames >= 5 else None)
        if mode == "i2v" and not first_id:
            raise ValueError("图生视频需要首帧图片")
        if mode == "fl2v" and not (first_id or last_id):
            raise ValueError("首尾帧模式需要至少一张图片")
        if first_id and not handoff_first_id:
            inputs["first_frame"] = g.add("LoadImage", image=media[first_id])
        if last_id:
            inputs["last_frame"] = g.add("LoadImage", image=media[last_id])
        conditioning = g.add("MiniMaxH3ImageToVideo", **inputs)
    else:
        inputs = {**common, "vae": vae, "audio_vae": audio_vae,
                  "ref_image_size": "max" if goldfish else shot.get("ref_image_size", "match")}
        image_ids = list(shot.get("ref_images") or [])[:9]
        video_ids = list(shot.get("ref_videos") or [])
        audio_ids = list(shot.get("ref_audios") or [])[:3]
        if mode in ("v2v", "rv2v"):
            source = shot.get("source_video")
            if not source:
                raise ValueError("视频编辑需要源视频")
            video_ids = list(dict.fromkeys([source, *video_ids]))[:3]
        else:
            video_ids = video_ids[:3]
        for index, media_id in enumerate(image_ids):
            inputs[f"ref_images.ref_image_{index}"] = g.add("LoadImage", image=media[media_id])
        for index, media_id in enumerate(video_ids):
            video = g.add("LoadVideo", file=media[media_id])
            frames = g.add("GetVideoComponents", video=video)
            inputs[f"ref_videos.ref_video_{index}"] = frames
            if index == 0 and mode in ("v2v", "rv2v") and shot.get("audio_mode") == "source":
                inputs[f"ref_video_audios.ref_video_audio_{index}"] = [frames[0], 1]
        for index, media_id in enumerate(audio_ids):
            inputs[f"ref_audios.ref_audio_{index}"] = g.add("LoadAudio", audio=media[media_id])
        if mode == "r2v" and not (image_ids or video_ids or audio_ids):
            raise ValueError("参考素材模式需要至少一个参考文件")
        conditioning = g.add("MiniMaxH3ReferenceToVideo", **inputs)

    latent = [conditioning[0], 1]
    if shot.get("continuity_video") and context_frames >= 5:
        guide_video = g.add("LoadVideo", file=media[shot["continuity_video"]])
        guide_frames = g.add("GetVideoComponents", video=guide_video)
        # Use ComfyUI's native H3 frame-guide conditioning for the overlap.
        # This keeps the target latent fresh and avoids replacing its prefix
        # with a VAE round-trip plus a sampler noise mask.
        conditioning = g.add("MiniMaxH3AddGuide", positive=conditioning, latent=latent,
                             image=guide_frames, vae=vae, frame_idx=0)
    if family == "creative" and handoff_first_id:
        opening_frame = g.add("LoadImage", image=media[handoff_first_id])
        bridge_frames = min(24, max(5, round(float(shot.get("duration", 5)) * 6)))
        conditioning = g.add("MiniMaxH3AddGuide", positive=conditioning, latent=latent,
                             image=opening_frame, vae=vae, frame_idx=context_frames + bridge_frames)
    # Reference mode can use its selected first frame as an actual starting
    # composition. A single start guide leaves subsequent frames free to move
    # continuously, unlike pinning a second still as a mid-shot close-up.
    if family == "reference" and shot.get("first_image") and not shot.get("continuity_video"):
        start_frame = g.add("LoadImage", image=media[shot["first_image"]])
        conditioning = g.add("MiniMaxH3AddGuide", positive=conditioning, latent=latent,
                             image=start_frame, vae=vae, frame_idx=0)
    # A reference-mode end frame can pin a deliberate handoff composition.
    # This is especially useful when the next segment should begin on the
    # same empty space after a subject leaves the camera view.
    if family == "reference" and shot.get("last_image"):
        end_frame = g.add("LoadImage", image=media[shot["last_image"]])
        conditioning = g.add("MiniMaxH3AddGuide", positive=conditioning, latent=latent,
                             image=end_frame, vae=vae, frame_idx=-1)
    # Optional user-authored keyframes use ComfyUI's official H3 guide node.
    # Times are relative to the visible shot; continuation context, when
    # present, is prepended and therefore shifts each anchor forward.
    if shot.get("guide_mode") == "official":
        guides = shot.get("h3_guides") or []
        if len(guides) > 8:
            raise ValueError("官方节点模式最多支持 8 个时间锚点")
        duration = float(shot.get("duration", 5))
        normalized_guides = []
        for guide in guides:
            if not isinstance(guide, dict):
                raise ValueError("官方节点时间锚点格式无效")
            image_id = guide.get("image_id")
            try:
                time_seconds = float(guide.get("time_seconds", 0))
            except (TypeError, ValueError):
                raise ValueError("官方节点锚点时间必须是秒数") from None
            if not image_id or image_id not in media:
                raise ValueError("官方节点锚点图片不存在，请重新选择图片")
            if not 0 < time_seconds < duration:
                raise ValueError(f"官方节点锚点时间需大于 0 且小于镜头时长 {duration:g} 秒")
            frame_idx = round(time_seconds * 24) + context_frames
            if frame_idx <= context_frames:
                raise ValueError("官方节点锚点至少从第 1 帧开始")
            normalized_guides.append((frame_idx, image_id))
        positions = [frame_idx for frame_idx, _ in normalized_guides]
        if len(positions) != len(set(positions)):
            raise ValueError("官方节点锚点不能设置在相同的视频帧")
        for frame_idx, image_id in sorted(normalized_guides):
            guide_image = g.add("LoadImage", image=media[image_id])
            conditioning = g.add("MiniMaxH3AddGuide", positive=conditioning, latent=latent,
                                 image=guide_image, vae=vae, frame_idx=frame_idx)
    adapter = str(settings.get("semantic_adapter") or "").strip()
    if adapter:
        conditioning = g.add("H3StudioSemanticBridge", positive=conditioning, adapter=adapter,
                             alpha=float(settings.get("semantic_alpha", 0.15)),
                             magnitude_match=bool(settings.get("semantic_magnitude_match", True)))
    # Release the 15~26GB text encoder immediately so all VRAM is reserved for DiT
    conditioning = g.add("DenoTextEncoderUnload", positive_conditioning=conditioning, text_encoder=clip)
    # H3 Studio's current SelfLift sampler performs its own two-stage noise
    # schedule and bypasses native H3 guide conditioning. Use the standard
    # sampler for continuation shots so the guide reaches MiniMaxH3.
    # SelfLift is a speed-oriented two-stage sampler. It is useful for
    # creative H3 modes, but reference/video-edit conditioning is sensitive:
    # on RV2V it can let the source video dominate the Picture identity.
    # Keep reference modes on the native sampler so Picture conditioning stays
    # intact all the way through denoising.
    if goldfish:
        negative = g.add("ConditioningZeroOut", conditioning=conditioning)
        sampler = g.add("KSamplerSelect", sampler_name="euler")
        schedule = g.add("BasicScheduler", model=model, scheduler="beta", steps=10, denoise=1)
        sampled = g.add("SelfLiftH3Sampler", model=model, positive=conditioning,
                        negative=negative, vae=vae, latent_image=latent, sampler=sampler,
                        sigmas=schedule, seed=seed, cfg=1.0, transition_step=6,
                        lowres_scale=0.6, rho=0.0, w_min=0.5, w_max=1.0,
                        upscaler_model=GOLDFISH_UPSCALER,
                        highres_tiling=True, upscaler_unload=True)
    elif settings.get("selflift") and family == "creative" and not shot.get("continuity_video"):
        sampled = g.add("H3StudioSelfLift", model=base_model, positive=conditioning,
                        latent=latent, vae=vae, seed=seed,
                        lowres_scale=float(settings.get("selflift_scale", 0.5)),
                        highres_steps=int(settings.get("selflift_highres_steps", 2)),
                        rho=float(settings.get("selflift_rho", 0)),
                        latent_model=str(settings.get("latent_model") or "minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors"),
                        enable_tiling=bool(settings.get("selflift_tiling", False)),
                        tile_count=int(settings.get("selflift_tile_count", 2)),
                        tile_overlap=int(settings.get("selflift_tile_overlap", 128)))
    else:
        sampler = g.add("KSamplerSelect", sampler_name="res_multistep")
        schedule = g.add("BasicScheduler", model=model, scheduler="simple", steps=steps, denoise=1)
        guider = g.add("BasicGuider", model=model, conditioning=conditioning)
        noise = g.add("RandomNoise", noise_seed=seed)
        sampled = g.add("SamplerCustomAdvanced", noise=noise, guider=guider, sampler=sampler, sigmas=schedule, latent_image=latent)
    refine_mode = "off" if goldfish else str(settings.get("refine_mode") or "off")
    if refine_mode in ("same", "latent", "latent_only"):
        pre_images = g.add("VAEDecode", samples=sampled, vae=vae)
        pre_audio = g.add("VAEDecodeAudio", samples=sampled, vae=audio_vae)
        pre_video = g.add("CreateVideo", images=pre_images, audio=pre_audio, fps=24, bit_depth=8)
        g.add("SaveVideo", video=pre_video, filename_prefix="video/" + safe_prefix(prefix) + "_pre",
              format="auto", codec="auto")
    if refine_mode in ("latent", "latent_only"):
        target_w = int(settings.get("refine_width") or width)
        target_h = int(settings.get("refine_height") or height)
        if target_w % 32 or target_h % 32 or target_w < width or target_h < height:
            raise ValueError("H3 latent 放大画布须为 32 的倍数且不小于原画布")
        lifted = g.add("H3StudioLatentUpscale", positive=conditioning, latent=sampled,
                       target_width=target_w, target_height=target_h,
                       latent_model=str(settings.get("latent_model") or "minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors"))
        conditioning, sampled = lifted, [lifted[0], 1]
    if refine_mode in ("same", "latent"):
        refine_steps = max(1, min(16, int(settings.get("refine_steps", 4))))
        refine_denoise = max(0.01, min(1.0, float(settings.get("refine_denoise", 0.3))))
        passes = max(1, min(8, int(settings.get("refine_passes", 1))))
        for pass_index in range(passes):
            refine_guider = g.add("BasicGuider", model=model, conditioning=conditioning)
            refine_sampler = g.add("KSamplerSelect", sampler_name="euler")
            refine_schedule = g.add("BasicScheduler", model=model, scheduler="simple",
                                    steps=refine_steps, denoise=refine_denoise)
            refine_noise = g.add("RandomNoise", noise_seed=(seed + pass_index + 1) % (1 << 64))
            sampled = g.add("SamplerCustomAdvanced", noise=refine_noise, guider=refine_guider,
                            sampler=refine_sampler, sigmas=refine_schedule, latent_image=sampled)
    images = g.add("VAEDecode", samples=sampled, vae=vae)
    audio = g.add("VAEDecodeAudio", samples=sampled, vae=audio_vae)
    video = g.add("CreateVideo", images=images, audio=audio, fps=24, bit_depth=8)
    g.add("SaveVideo", video=video, filename_prefix="video/" + safe_prefix(prefix), format="auto", codec="auto")
    return g.nodes


def director_control_text(shot: dict) -> str:
    """Translate UI directing choices into explicit, model-facing instructions."""
    if shot.get("suppress_director_controls"):
        return ""
    camera_moves = {
        "固定镜头": "Camera movement: locked-off tripod shot; keep the camera position and framing fixed with no pan, tilt, zoom, dolly, or orbit.",
        "推近": "Camera movement: perform one clear, continuous dolly-in toward the main subject; the subject grows progressively larger in frame.",
        "拉远": "Camera movement: perform one clear, continuous dolly-out away from the main subject; reveal more of the surrounding environment.",
        "横向移动": "Camera movement: perform a clear lateral tracking move parallel to the scene; preserve screen direction and visible parallax.",
        "跟随主体": "Camera movement: continuously track and follow the moving main subject while keeping it consistently framed.",
        "环绕主体": "Camera movement: make a continuous orbit around the main subject with visible background parallax while keeping the subject centered.",
        "手持": "Camera movement: use controlled handheld camera motion with subtle natural sway; do not become a locked-off shot.",
    }
    camera_speeds = {
        "缓慢": "Camera motion pace: slow and deliberate throughout the shot.",
        "平稳": "Camera motion pace: smooth, even, and mechanically stable throughout the shot.",
        "快速": "Camera motion pace: fast and decisive, with clearly visible displacement during the shot.",
    }
    action_levels = {
        "自然": "Subject action intensity: natural, physically realistic movement with readable motion.",
        "轻微": "Subject action intensity: subtle and restrained; use small movements without becoming completely still.",
        "明显": "Subject action intensity: clearly visible and purposeful; complete the described action with readable body motion.",
        "夸张": "Subject action intensity: large, energetic, and strongly exaggerated while remaining physically coherent.",
    }
    emotions = {
        "轻松": "Scene mood: relaxed and easygoing.", "紧张": "Scene mood: tense and urgent.",
        "悲伤": "Scene mood: sorrowful and subdued.", "温馨": "Scene mood: warm and intimate.",
        "热烈": "Scene mood: lively and passionate.", "悬疑": "Scene mood: suspenseful and uncertain.",
        "浪漫": "Scene mood: romantic and tender.",
    }
    instructions = []
    for value, mapping in ((shot.get("emotion"), emotions), (shot.get("camera_move"), camera_moves),
                           (shot.get("camera_speed"), camera_speeds),
                           (shot.get("action_intensity"), action_levels)):
        if value in mapping:
            instructions.append(mapping[value])
    return "DIRECTOR CONTROLS — MUST FOLLOW:\n- " + "\n- ".join(instructions) if instructions else ""


def compile_shot_prompt(shot: dict) -> str:
    """Compile the exact prompt sent to H3 from one timeline segment."""
    description = str(shot.get("prompt") or "").strip()
    identity_picture = str(shot.get("continuity_identity_picture_tag") or "<Picture 1>")
    if shot.get("continuity_subject_relation") == "new":
        description = (
            f"NEW PERSON ENTERS: the incoming person MUST match {identity_picture} for face, age, build, "
            "hair, headwear, and clothing. Keep this person visibly distinct from the outgoing person. "
            "The outgoing person remains the same physical person in the preceding footage; never turn or "
            "morph them into the incoming person. Continue the established camera and scene; let the "
            "incoming person enter along a physically possible path supported by the requested action. "
            "When the shot calls for a separate incoming vehicle or prop, keep it physically distinct from "
            "the outgoing one and preserve each object's own motion and identifying details. "
            "Do not restart the action or force an unsupported entrance.\n" + description
        )
    if shot.get("continuous_take"):
        if shot.get("continuity_subject_relation") == "new":
            description += (
                "\nOne-take continuity: continue the camera path, environment, screen direction, and ongoing action "
                "as one physical take. The incoming person is a separate person whose identity comes from the "
                "current shot's tagged reference; the outgoing person keeps their own identity if still visible. "
                "Do not restart the action, teleport, clone, ghost, or morph either person. Use an occlusion only "
                "when the requested action and visible scene support it."
            )
        else:
            description += (
                "\nOne-take continuity: keep every continuing subject as one physical entity and preserve its identity, "
                "position, movement, and contact with the environment. Continue along one physically possible camera path. "
                "Do not restart the action, teleport, clone, double, or ghost a subject. Use a foreground obstruction "
                "only when the requested action and visible scene support it; otherwise preserve clear spatial continuity "
                "and do not add one as a default transition."
            )
    if shot.get("continuity_video"):
        if shot.get("continuity_strength") == "strong":
            if shot.get("continuity_subject_relation") == "new":
                description += (
                    "\nContinuous-take handoff: the supplied video shows the immediately preceding moments of this "
                    "same take. Continue its camera position and motion, environment, object positions, screen direction, "
                    "action phase, exposure, and ambience. The current shot's explicitly tagged person reference is the "
                    "identity authority for the incoming person only. Keep the outgoing person unchanged wherever visible. "
                    "Move the camera or subjects along the described physical path to reveal the new person; do not "
                    "replace one person's face or body with another's. Do not restart, reset the camera, teleport, morph, "
                    "clone, or freeze. Use an occlusion only if the described scene supports it. Keep the take continuous."
                )
            else:
                description += (
                    "\nContinuous-take handoff: the supplied video shows the immediately preceding moments of this "
                    "same take. Continue from its final visible state. Preserve the established subject identities, "
                    "environment, object positions, screen direction, action phase, camera position and motion, "
                    "exposure, and ambience. Let the requested next action grow from what is already happening; "
                    "carry motion through the overlap without restarting, resetting the camera, jumping to an "
                    "unmotivated composition, or freezing. Treat reference images of an established subject as "
                    "appearance guides for that same subject, not extra copies. If this prompt specifies a real "
                    "foreground obstruction, carry that same element through its motion; otherwise continue the scene "
                    "without introducing an unsupported foreground object. Keep the take physically continuous."
                )
        else:
            description += (
                "\nUse the supplied preceding frames as a light continuity cue. Preserve the main subject and "
                "screen direction broadly while following the requested action into a natural continuation."
            )
        if shot.get("mode") == "fl2v" and shot.get("first_image"):
            description += (
                "\nFor this continued first-and-last-frame shot, begin with the supplied preceding video frames, "
                "move continuously into the selected opening image after a short bridge, then progress toward "
                "the selected ending image. Do not cut or reset the scene at either handoff."
            )
    controls = director_control_text(shot)
    if controls:
        description += "\n" + controls
    dialogue = str(shot.get("dialogue_text") or "").strip()
    if dialogue:
        speaker = str(shot.get("speaker_description") or "画面中的人物").strip()
        speaker_id = re.sub(r"[^A-Za-z0-9]", "", str(shot.get("speaker_id") or "S1")) or "S1"
        delivery = str(shot.get("dialogue_delivery") or "").strip()
        language = str(shot.get("dialogue_language") or "Chinese").strip()
        when = str(shot.get("dialogue_time") or "").strip()
        timing = f"在 {when}，" if when else ""
        if shot.get("dialogue_voiceover"):
            line = f"{timing}{speaker} ({speaker_id}) says in an off-screen voiceover{': ' + delivery if delivery else ''}: <d>[{language}] {dialogue}</d>. The on-screen character's lips remain completely closed."
        else:
            line = f"{timing}{speaker} ({speaker_id}) says{': ' + delivery if delivery else ''}: <d>[{language}] {dialogue}</d>"
        if shot.get("dialogue_carries_in"):
            line = "<scenetrans> The audio carries over from the previous shot. " + line
        if shot.get("dialogue_cutoff"):
            line += " <cutoff>"
        if shot.get("dialogue_carries_out_valid"):
            line += " The audio continues across the cut into the next shot. <scenetrans>"
        description += "\n" + line
    visible_text = str(shot.get("on_screen_text") or "").strip()
    if visible_text:
        quoted = "\n".join('Visible on-screen text reads "' + line.replace('"', '\\"') + '".'
                             for line in visible_text.splitlines() if line.strip())
        description += "\n" + quoted
    if not description:
        return ""
    duration = max(1, float(shot.get("duration", 5)))
    shot_description = (f"[Shot 1] {description}" if shot.get("mode") in ("r2v", "v2v", "rv2v")
                        else f"[Shot 1] [0s-{duration:g}s] {description}")
    soundscape = str(shot.get("overall_soundscape") or "").strip()
    music = str(shot.get("non_diegetic_music") or "").strip()
    if shot.get("audio_mode") == "mute":
        soundscape = music = "N/A"
    else:
        if not soundscape:
            soundscape = "Ambient sound and synchronized physical sounds matching the scene."
        if not music:
            music = "N/A"
    if shot.get("mode") in ("r2v", "v2v", "rv2v"):
        return _compile_full_reference_prompt(shot, description, soundscape, music)
    return ("integrated_multimodal_description: " + shot_description + "\n"
            + "overall_soundscape: " + soundscape + "\n"
            + "non_diegetic_music: " + music)


def lint_shot_prompt(shot: dict) -> list[dict]:
    """Report actionable prompt and duration issues without blocking drafts."""
    prompt = str(shot.get("prompt") or "").strip()
    findings = []
    def add(level, code, message):
        findings.append({"level": level, "code": code, "message": message})
    if not prompt:
        add("error", "empty_prompt", "请填写镜头画面或动作描述。")
    elif len(prompt) < 12:
        add("warning", "short_prompt", "描述较短；可补充主体、动作或环境信息。")
    if len(prompt) > 4000:
        add("warning", "long_prompt", "提示词超过 4000 字符，建议拆分或精简。")
    dialogue = str(shot.get("dialogue_text") or "").strip()
    if dialogue:
        if not re.fullmatch(r"S\d+", str(shot.get("speaker_id") or "S1")):
            add("error", "speaker_id", "说话人编号需使用 S1、S2 这样的格式。")
        time_range = str(shot.get("dialogue_time") or "").strip()
        if time_range:
            match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*(?:-|~|至)\s*(\d+(?:\.\d+)?)\s*(?:s|秒)?\s*", time_range, re.I)
            if not match or float(match.group(1)) >= float(match.group(2)) or float(match.group(2)) > float(shot.get("duration", 5)):
                add("error", "dialogue_time", "对白时间请写成 1-3s，且范围应落在当前镜头时长内。")
        if "<d>" in prompt or "</d>" in prompt:
            add("warning", "duplicate_dialogue", "对白字段会自动生成 <d> 标记；提示词正文里已有对白标记，请检查是否重复。")
    if shot.get("dialogue_voiceover") and not dialogue:
        add("warning", "empty_voiceover", "已选旁白，但还没有填写旁白内容。")
    if shot.get("audio_mode") == "mute" and dialogue:
        add("warning", "muted_dialogue", "当前镜头设置为静音，但仍填写了对白；请切换为生成声音或清空对白。")
    if shot.get("_h3_ignored_character_ids"):
        add("warning", "character_refs_narrowed",
            "此镜头选了多个角色，但只 @ 了一个角色参考图；生成时仅使用该角色档案。若要多人同框，请分别 @ 对应角色图片。")
    if shot.get("dialogue_carries_out") and not shot.get("dialogue_carries_out_valid"):
        add("warning", "dialogue_carry_target", "已启用对白跨镜头，但下一镜头没有对白内容。")
    if shot.get("dialogue_carry_speaker_mismatch"):
        add("error", "dialogue_carry_speaker", "跨镜头对白的说话人编号不一致。")
    if shot.get("dialogue_profile_speaker_mismatch"):
        add("error", "dialogue_profile_speaker", "对白说话人编号与所选角色档案不一致。")
    duration = float(shot.get("duration", 5))
    if not 2 <= duration <= 15:
        add("error", "duration_range", "H3 单镜头时长需在 2–15 秒内。")
    if (duration <= 6
            and re.search(r"approach|toward|驶近|靠近|走近|接近", prompt, re.I)
            and re.search(r"slow|brak|decel|减速|放慢", prompt, re.I)
            and re.search(r"close[ -]?up|特写|近景", prompt, re.I)
            and re.search(r"leav|depart|accelerat|驶离|离开|远去|走开", prompt, re.I)):
        add("warning", "dense_motion_beats",
            "短镜头同时要求驶近、减速特写和驶离，模型可能来不及完成。可延长单段时长，检查完整动作后再裁剪。")
    if (shot.get("continuous_take") and duration <= 6
            and re.search(r"(?:then|finally|resolves? into|随后|然后|最终).{0,120}(?:close[ -]?up|特写|近景)",
                          prompt, re.IGNORECASE | re.DOTALL)):
        add("warning", "late_closeup_risk",
            "短镜头把人物特写放在远景或移动之后，实测易停在远景或突然跳近。请让上一段结束时已靠近人物，并让本段从近景开始。")
    if re.search(r"<(?:Picture|Video|Audio)\s*\d+>", prompt, re.I):
        refs = {"Picture": len(shot.get("ref_images") or []),
                "Video": len(shot.get("ref_videos") or []), "Audio": len(shot.get("ref_audios") or [])}
        for kind, count in refs.items():
            for match in re.finditer(rf"<{kind}\s*(\d+)", prompt, re.I):
                if int(match.group(1)) > count:
                    add("warning", "unbound_reference", f"提示词中的 {match.group(0)} 未绑定到当前镜头素材。")
                    break
    if not findings:
        add("ok", "ready", "未发现明显问题。")
    return findings


def _is_identity_reference(shot: dict, asset_id: str, name: str, visual_description: str) -> bool:
    """Only mark identity when a project character or an explicit portrait sheet binds it."""
    metadata = (name + " " + visual_description).lower()
    characters = shot.get("_h3_characters") or []
    if any(asset_id in (item.get("portrait_asset"), item.get("multiview_asset"))
           for item in characters):
        return True
    return any(word in metadata for word in
               ("五视图", "四视图", "三视图", "多视图", "portrait", "face sheet", "turnaround", "multiview"))
def _compile_full_reference_prompt(shot: dict, description: str, soundscape: str, music: str) -> str:
    """Use MiniMax's six-section rewrite format for video/reference tasks."""
    mode = shot.get("mode")
    images = list(shot.get("ref_images") or [])[:9]
    videos = list(shot.get("ref_videos") or [])[:3]
    source_video = shot.get("source_video") if mode in ("v2v", "rv2v") else None
    ordered_videos = list(dict.fromkeys(([source_video] if source_video else []) + videos))[:3]
    audios = list(shot.get("ref_audios") or [])[:3]
    names = shot.get("_h3_reference_names") or {}
    descriptions = shot.get("_h3_reference_descriptions") or {}
    definitions, retention = [], []
    def asset_name(asset_id):
        return str(names.get(asset_id) or "selected reference asset").replace("\n", " ")
    identity_images = {asset_id for asset_id in images
                       if _is_identity_reference(shot, asset_id, asset_name(asset_id),
                                                 str(descriptions.get(asset_id) or ""))}
    for i, asset_id in enumerate(images, 1):
        name = asset_name(asset_id)
        visual_description = str(descriptions.get(asset_id) or "").strip()
        detail = f' Verified user description: {visual_description}.' if visual_description else ''
        if asset_id in identity_images:
            definitions.append(f'<Picture {i}> is the identity reference ({name}) for the person in this shot.'
                               f'{detail} Preserve this person’s face, hair, build, and clothing. '
                               'Use it for the identity role requested in the detailed shot description. The user’s '
                               'description and loaded Skill determine whether any other visible property of this image '
                               'should be used; do not add extra copies of the person by default.')
        else:
            definitions.append(f'<Picture {i}> is a visual reference ({name}).{detail} Its role is exactly the one '
                               'assigned to this tag in the detailed shot description. The user’s description and '
                               'loaded Skill determine which attributes to use and whether the reference itself should '
                               'be reproduced in the video. Do not make every reference an in-frame element merely '
                               'because it was supplied; follow any explicit request to reproduce its content or layout. '
                               'Preserve distinctive colors, silhouettes, markings and their placement when the assigned '
                               'role or verified description calls for them; do not invent conflicting insignia or details.')
        marker = (shot.get("ref_image_roles") or {}).get(asset_id, "auto")
        if marker == "auto":
            marker = "fully_preserved" if asset_id in identity_images else "attribute_transfer"
        if marker not in ("fully_preserved", "partially_preserved", "attribute_transfer", "weak_reference"):
            marker = "weak_reference" if len(images) > 1 else "fully_preserved"
        explanation = {
            "fully_preserved": ("preserve this character’s requested identity details"
                                if asset_id in identity_images else
                                "preserve the depicted content according to its explicitly assigned role and the detailed shot request"),
            "partially_preserved": "retain only details relevant to the role explicitly assigned to this reference",
            "attribute_transfer": "transfer the visual attributes requested for this reference",
            "weak_reference": "use only broad visual guidance; the reference image itself is not a requested video element",
        }[marker]
        retention.append(f'<Picture {i}> ([Shot 1]): {marker} - {explanation}.')
    if len(images) > 1:
        definitions.append("Each picture is reference input. Use each tag only for the role assigned to it in the "
                           "detailed shot description and loaded Skill. Do not assume that every reference must appear "
                           "as a separate in-frame element; equally, follow an explicit request to reproduce or combine "
                           "reference content. Let the user’s description determine subjects, actions, composition, and "
                           "which visual properties from each reference are used.")
    for i, asset_id in enumerate(ordered_videos, 1):
        name = asset_name(asset_id)
        is_source = bool(source_video and i == 1 and asset_id == source_video)
        role = "the source video being edited" if is_source else "a video reference for motion and temporal structure"
        definitions.append(f'<Video {i}> is {role} ({name}).')
        default = "partially_preserved" if is_source else "weak_reference"
        marker = (shot.get("ref_video_roles") or {}).get(asset_id, default)
        if marker not in ("fully_preserved", "partially_preserved", "attribute_transfer", "weak_reference"):
            marker = default
        retention.append(f'<Video {i}> ({"source edit" if is_source else "motion and temporal structure"}): {marker} - the reference video role is followed in the target shot.')
    characters = shot.get("_h3_characters") or []
    picture_index = {asset_id: i for i, asset_id in enumerate(images, 1)}
    audio_index = {asset_id: i for i, asset_id in enumerate(audios, 1)}
    for fallback_index, character in enumerate(characters, 1):
        i = max(1, int(character.get("subject_number") or fallback_index))
        sources = []
        portrait = character.get("portrait_asset")
        multiview = character.get("multiview_asset")
        voice = character.get("voice_asset")
        if portrait in picture_index:
            sources.append(f"appearance comes from <Picture {picture_index[portrait]}>")
        if multiview in picture_index:
            view_label = {"four": "four-view", "five": "five-view", "six": "six-view"}.get(
                character.get("multiview_layout"), "four-view")
            sources.append(f"face, body proportions and outfit stay consistent with the {view_label} sheet <Picture {picture_index[multiview]}>")
            definitions.append(f"<Picture {picture_index[multiview]}> is a {view_label} identity sheet, not a panel layout for the video frame.")
        if voice in audio_index:
            sources.append(f"voice timbre comes from <Audio {audio_index[voice]}>")
        appearance = str(character.get("appearance") or "").strip()
        source_text = " whose " + " and ".join(sources) if sources else ""
        detail = f", {appearance}" if appearance else ""
        definitions.append(f'<Subject {i}> is {character.get("name") or "the selected character"}{detail}{source_text}.')
        retention.append(f'<Subject {i}> (appears in [Shot 1]): fully_preserved - the selected character identity and described appearance are retained.')
        if voice in audio_index:
            speaker_id = re.sub(r"[^A-Za-z0-9]", "", str(character.get("speaker_id") or shot.get("speaker_id") or "S1")) or "S1"
            definitions.append(f'<Audio {audio_index[voice]}> is the voice-timbre reference for <Subject {i}> ({speaker_id}).')
    audio_markers = {"fully_copy", "partially_copy", "reference", "weak_reference"}
    profile_voice_assets = {character.get("voice_asset") for character in characters if character.get("voice_asset")}
    for i, asset_id in enumerate(audios, 1):
        role = (shot.get("ref_audio_roles") or {}).get(asset_id, "reference")
        if role not in audio_markers:
            role = "reference"
        name = asset_name(asset_id)
        if asset_id not in profile_voice_assets:
            definitions.append(f'<Audio {i}> is an audio reference ({name}) for the target shot.')
        explanations = {
            "fully_copy": "the source audio is reused as the complete final audio track",
            "partially_copy": "selected audio content is copied and may be combined with other sounds",
            "reference": "audio timbre, rhythm, delivery, or sound texture guides newly generated sound without copying the signal",
            "weak_reference": "only broad audio category or atmosphere is retained",
        }
        retention.append(f'<Audio {i}>: {role} - {explanations[role]}.')
    if mode == "v2v":
        task = "video editing"
        summary = "The target video is an edited version of <Video 1>."
    elif mode == "rv2v":
        task = "video editing + reference generation"
        summary = "The target video is an edited version of <Video 1> and follows the selected visual and audio references."
    else:
        task = "reference generation"
        summary = "The target video follows the selected visual and audio references for the described shot."
    if audios:
        relations = {(shot.get("ref_audio_roles") or {}).get(asset_id, "reference") for asset_id in audios}
        task += " + audio reuse" if relations & {"fully_copy", "partially_copy"} else " + audio reference"
    if not definitions:
        definitions.append("No explicit reference assets are assigned.")
    if not retention:
        retention.append("No reference assets require a retention analysis.")
    sections = [
        "subject_definitions:\n" + "\n".join(definitions),
        "summary:\n[" + task + "] " + summary,
        "retention_analysis:\n" + "\n".join(retention),
        "detailed_description:\n[Shot 1] " + description,
        "overall_soundscape:\n" + soundscape,
        "non_diegetic_music:\n" + music,
    ]
    return "\n\n".join(sections)


def build_seedvr2_graph(video_file: str, prefix: str, *, resolution=720, max_resolution=1280, seed=42) -> dict:
    """Run SeedVR2 separately so H3 weights can unload before upscaling."""
    g = Graph()
    video = g.add("LoadVideo", file=video_file)
    frames = g.add("GetVideoComponents", video=video)
    vae = g.add("SeedVR2LoadVAEModel", model="ema_vae_fp16.safetensors", device="cuda:0", encode_tiled=True,
                encode_tile_size=768, encode_tile_overlap=96, decode_tiled=True, decode_tile_size=512,
                decode_tile_overlap=64, tile_debug="false", offload_device="cpu", cache_model=False)
    dit = g.add("SeedVR2LoadDiTModel", model="seedvr2_ema_3b-Q4_K_M.gguf", device="cuda:0",
                blocks_to_swap=0, swap_io_components=False, offload_device="cpu", cache_model=False,
                attention_mode="sageattn_2")
    upscaled = g.add("SeedVR2VideoUpscaler", image=frames, dit=dit, vae=vae, seed=int(seed),
                     resolution=int(resolution), max_resolution=int(max_resolution), batch_size=41,
                     uniform_batch_size=True, color_correction="lab", temporal_overlap=3,
                     prepend_frames=0, input_noise_scale=0, latent_noise_scale=0,
                     offload_device="cpu", enable_debug=False)
    output = g.add("CreateVideo", images=upscaled, audio=[frames[0], 1], fps=[frames[0], 2])
    g.add("SaveVideo", video=output, filename_prefix="video/" + safe_prefix(prefix), format="auto", codec="auto")
    return g.nodes

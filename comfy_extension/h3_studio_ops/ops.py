"""Isolated model operators used by H3 Studio API graphs."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

import folder_paths

from .selflift.sample import sample_selflift_stage
from .selflift.cond import resize_positive_spatial
from .h3_latent_upscale import upscale_h3_video_latent
from .h3_motion_context import _repack_av_streams, _streams_from_latent


class _Student(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(5120, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 5120)

    def forward(self, value):
        return self.fc3(F.silu(self.fc2(F.silu(self.fc1(value)))))


_STUDENTS = {}


def _adapter_path(name: str) -> Path:
    root = Path(folder_paths.models_dir) / "semantic_bridge"
    candidate = (root / name).resolve()
    if not candidate.is_relative_to(root.resolve()) or not candidate.is_file():
        raise ValueError("Semantic Bridge 权重不存在：" + name)
    return candidate


def _student(name: str):
    path = _adapter_path(name)
    stamp = path.stat().st_mtime_ns
    key = (str(path), stamp)
    if key not in _STUDENTS:
        if path.suffix == ".safetensors":
            from safetensors.torch import load_file
            state = load_file(str(path))
        else:
            state = torch.load(path, map_location="cpu", weights_only=True)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
        student = _Student()
        student.load_state_dict(state, strict=True)
        student.eval()
        _STUDENTS.clear()
        _STUDENTS[key] = student
    return _STUDENTS[key]


class H3StudioSemanticBridge:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "positive": ("CONDITIONING",),
            "adapter": ("STRING", {"default": ""}),
            "alpha": ("FLOAT", {"default": 0.15, "min": 0, "max": 1, "step": 0.01}),
            "magnitude_match": ("BOOLEAN", {"default": True}),
        }}

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "apply"
    CATEGORY = "H3 Studio"

    def apply(self, positive, adapter, alpha, magnitude_match):
        if not positive or not isinstance(positive[0][0], torch.Tensor):
            raise ValueError("Semantic Bridge 需要 MiniMax H3 conditioning")
        hidden = positive[0][0]
        if hidden.ndim != 3 or hidden.shape[-1] != 5120:
            raise ValueError(f"Semantic Bridge 需要 [B,T,5120]，收到 {tuple(hidden.shape)}")
        student = _student(adapter).to(device=hidden.device, dtype=torch.float32)
        with torch.no_grad():
            source = hidden.float()
            normalized = source * torch.rsqrt(source.square().mean(dim=-1, keepdim=True) + 1e-6)
            output = student(normalized)
            if magnitude_match:
                output = output * (source.norm(dim=-1, keepdim=True) / (output.norm(dim=-1, keepdim=True) + 1e-8))
            output = (source + float(alpha) * (output - source)).to(dtype=hidden.dtype)
        student.to("cpu")
        result = list(positive)
        first = list(positive[0])
        first[0] = output
        result[0] = first
        return (result,)


class H3StudioSelfLift:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",), "positive": ("CONDITIONING",),
            "latent": ("LATENT",), "vae": ("VAE",),
            "seed": ("INT", {"default": 42, "min": 0, "max": 2**64 - 1}),
            "lowres_scale": ("FLOAT", {"default": 0.5, "min": 0.25, "max": 1, "step": 0.05}),
            "highres_steps": ("INT", {"default": 2, "min": 1, "max": 7}),
            "rho": ("FLOAT", {"default": 0, "min": 0, "max": 1, "step": 0.05}),
            "latent_model": ("STRING", {"default": "minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors"}),
            "enable_tiling": ("BOOLEAN", {"default": False}),
            "tile_count": ("INT", {"default": 2, "min": 1, "max": 8}),
            "tile_overlap": ("INT", {"default": 128, "min": 0, "max": 2048}),
        }}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "H3 Studio"

    def sample(self, model, positive, latent, vae, seed, lowres_scale, highres_steps, rho,
               latent_model, enable_tiling, tile_count, tile_overlap):
        root = Path(folder_paths.models_dir) / "latent_upscale_models"
        path = (root / latent_model).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            raise ValueError("SelfLift 3D latent 权重不存在：" + latent_model)
        pack = {
            "enabled": True, "split_mode": "highres_steps",
            "highres_steps": int(highres_steps), "lowres_scale": float(lowres_scale),
            "h3_latent_model": latent_model, "rho": float(rho),
            "w_min": 0.5, "w_max": 1.0, "latent_upsample": "bilinear",
            "native_low_carry": False, "enable_latent_chunking": False,
            "enable_tiling": bool(enable_tiling), "tile_count": int(tile_count),
            "tile_overlap": int(tile_overlap),
        }
        output, _ = sample_selflift_stage(
            model=model, positive=positive, negative=None, latent=latent,
            seed=int(seed), cfg=1.0, steps=8, sampler_name="euler",
            scheduler="simple", pack=pack, vae=vae,
        )
        return (output,)


class H3StudioLatentUpscale:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "positive": ("CONDITIONING",), "latent": ("LATENT",),
            "target_width": ("INT", {"default": 1536, "min": 32, "step": 32}),
            "target_height": ("INT", {"default": 864, "min": 32, "step": 32}),
            "latent_model": ("STRING", {"default": "minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors"}),
        }}

    RETURN_TYPES = ("CONDITIONING", "LATENT")
    FUNCTION = "upscale"
    CATEGORY = "H3 Studio"

    def upscale(self, positive, latent, target_width, target_height, latent_model):
        streams = _streams_from_latent(latent)
        src_h, src_w = int(streams[0].shape[-2]), int(streams[0].shape[-1])
        dst_h, dst_w = int(target_height) // 16, int(target_width) // 16
        if dst_h < src_h or dst_w < src_w:
            raise ValueError("H3 3D latent 放大不支持缩小画布")
        lifted = upscale_h3_video_latent(
            {"samples": streams[0]}, target_width=int(target_width), target_height=int(target_height),
            source_width=src_w * 16, source_height=src_h * 16, model_name=latent_model,
        )["samples"]
        out = dict(latent)
        out["samples"] = _repack_av_streams([lifted, *streams[1:]], latent)
        out.pop("noise_mask", None)
        return (resize_positive_spatial(positive, src_h, src_w, dst_h, dst_w), out)


class H3StudioLatentContinue:
    """Seed a new H3 AV latent with the previous shot's decoded video tail.

    The copied prefix gets a tapered noise mask: the disposable overlap can
    redraw freely while the last few latent tokens stay close to the prior
    shot, giving the next segment a real temporal starting state.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "latent": ("LATENT",), "frames": ("IMAGE",), "vae": ("VAE",),
            "context_frames": ("INT", {"default": 22, "min": 5, "max": 56}),
            "seam_redraw": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.95, "step": 0.05}),
        }}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "apply"
    CATEGORY = "H3 Studio"

    def apply(self, latent, frames, vae, context_frames, seam_redraw):
        from .h3_latent_continue import apply_pixel_tail
        return (apply_pixel_tail(latent, frames, vae, int(context_frames), float(seam_redraw)),)


class H3StudioContinueRemask:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",), "latent": ("LATENT",), "sigmas": ("SIGMAS",)}}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "H3 Studio"

    def apply(self, model, latent, sigmas):
        from .h3_latent_continue import install_continue_prefix_remask
        return (install_continue_prefix_remask(model, latent, sigmas),)

"""Latent handoff for continuous H3 sequences (Apache-2.0 adapted algorithm)."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from typing import Any

from .h3_motion_context import _streams_from_latent

PREFIX_STEPS_KEY = "_h3studio_continue_prefix_steps"
CONTINUE_SEAM_KEY = "_h3studio_continue_seam"


def _nested(video, audio, template):
    try:
        from comfy.nested_tensor import NestedTensor
        return NestedTensor((video, audio))
    except Exception:
        return type(template)((video, audio))


def _resize_frames(frames, width, height):
    # Comfy IMAGE convention is [B,H,W,C], VAE expects that convention too.
    x = frames[..., :3].movedim(-1, 1).float()
    x = F.interpolate(x, size=(height, width), mode="bilinear", align_corners=False)
    return x.movedim(1, -1).to(dtype=frames.dtype)


def _prefix_weights(count, seam):
    if float(seam) <= 1e-8:
        return [0.0] * int(count)
    # H3 video temporal VAE uses one token per 17 frames after its first token.
    taper = min(4, count)
    head = count - taper
    values = [1.0] * head
    values += [1.0 + (seam - 1.0) * (i + 1) / taper for i in range(taper)]
    return values


def apply_pixel_tail(latent, frames, vae, context_frames, seam_redraw):
    streams = _streams_from_latent(latent)
    if len(streams) < 2:
        raise ValueError("H3 Studio latent continuation requires video and audio streams")
    video, audio = streams[:2]
    if video.ndim != 5:
        raise ValueError(f"Expected H3 video latent [B,C,T,H,W], got {tuple(video.shape)}")
    available = int(frames.shape[0])
    pixel_count = min(int(context_frames), available)
    # Use supported H3 VAE context windows; this avoids an off-grid encode.
    grid = (1, 5, 22, 39, 56, 73, 90, 107, 124)
    pixel_count = max((n for n in grid if n <= pixel_count), default=0)
    if pixel_count < 5:
        raise ValueError("Previous shot does not contain enough frames for H3 continuation")
    source = frames[-pixel_count:]
    pixel_width, pixel_height = int(video.shape[-1]) * 16, int(video.shape[-2]) * 16
    encoded = vae.encode(_resize_frames(source, pixel_width, pixel_height))
    if encoded.ndim != 5:
        raise ValueError(f"H3 VAE returned unexpected continuation shape {tuple(encoded.shape)}")
    prefix = int(encoded.shape[2])
    if prefix >= int(video.shape[2]):
        raise ValueError("Continuation overlap must be shorter than the new segment")

    patched = video.clone()
    patched[:, :, :prefix] = encoded[:, :, :prefix].to(device=video.device, dtype=video.dtype)
    out = dict(latent)
    template = latent["samples"]
    out["samples"] = _nested(patched, audio, template)

    seam = max(0.0, min(0.95, float(seam_redraw)))
    mask = torch.ones((1, 1, video.shape[2], video.shape[3], video.shape[4]),
                      device=video.device, dtype=torch.float32)
    weights = torch.tensor(_prefix_weights(prefix, seam), device=video.device, dtype=mask.dtype)
    mask[:, :, :prefix] = weights.view(1, 1, prefix, 1, 1)
    audio_mask = torch.ones((1, 1, 1, audio.shape[-1]), device=audio.device, dtype=torch.float32)
    out["noise_mask"] = _nested(mask, audio_mask, template)
    out[PREFIX_STEPS_KEY] = prefix
    out[CONTINUE_SEAM_KEY] = seam
    return out


def _schedule_values(sigmas: Any) -> tuple[float, ...]:
    raw = sigmas.detach().float().reshape(-1).cpu().tolist() if torch.is_tensor(sigmas) else list(sigmas or ())
    values = [float(value) for value in raw if math.isfinite(float(value)) and float(value) >= 0]
    return tuple(sorted(set(values), reverse=True))


def _next_sigma_ratio(current: float, sigmas: Any) -> float:
    current = float(current)
    if not math.isfinite(current) or current <= 0:
        return 0.0
    tolerance = max(1e-7, abs(current) * 1e-6)
    for candidate in _schedule_values(sigmas):
        if candidate < current - tolerance:
            return max(0.0, min(1.0, candidate / current))
    return 0.0


def _apply_prefix_weights(mask: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    n = int(weights.numel())
    if not n or not torch.is_tensor(mask):
        return mask
    out = mask.clone()
    values = weights.to(device=mask.device, dtype=mask.dtype)
    if mask.ndim == 5 and n <= int(mask.shape[2]):
        view = [1] * 5
        view[2] = n
        out[:, :, :n] = values.view(*view)
    elif mask.ndim == 3 and n <= int(mask.shape[-1]):
        out[..., :n] = values.view(1, 1, n)
    elif mask.ndim == 4 and n <= int(mask.shape[2]):
        view = [1] * 4
        view[2] = n
        out[:, :, :n] = values.view(*view)
    return out


class _PrefixRemask:
    """Follow H3's current sigma at every denoise step to keep the handoff pinned."""
    def __init__(self, prefix_steps, sigmas, video_shape, seam_min):
        self.prefix_steps = int(prefix_steps)
        self.sigmas = _schedule_values(sigmas)
        self.video_shape = tuple(int(v) for v in video_shape)
        self.seam_min = max(0.0, min(0.95, float(seam_min)))

    def _live_weights(self, sigma, extra_options=None):
        current = float(torch.as_tensor(sigma).detach().float().reshape(-1)[0])
        schedule = self.sigmas or _schedule_values((extra_options or {}).get("sigmas", ()))
        ratio = _next_sigma_ratio(current, schedule)
        values = []
        for base in _prefix_weights(self.prefix_steps, self.seam_min):
            value = base * ratio
            if self.seam_min > 0:
                value = max(self.seam_min, value)
            values.append(max(0.0, min(1.0, value)))
        return torch.tensor(values, dtype=torch.float32)

    def denoise_mask_function(self, sigma, denoise_mask, extra_options=None):
        weights = self._live_weights(sigma, extra_options)
        if torch.is_tensor(denoise_mask) and denoise_mask.ndim == 5:
            return _apply_prefix_weights(denoise_mask, weights)
        if torch.is_tensor(denoise_mask) and denoise_mask.ndim == 3:
            return _apply_prefix_weights(denoise_mask, weights)
        streams = list(denoise_mask.unbind()) if hasattr(denoise_mask, "unbind") else None
        if streams and torch.is_tensor(streams[0]) and streams[0].ndim == 5:
            video = _apply_prefix_weights(streams[0], weights)
            try:
                from comfy.nested_tensor import NestedTensor
                return NestedTensor((video, *streams[1:]))
            except Exception:
                return type(denoise_mask)((video, *streams[1:]))
        return denoise_mask

    def apply_model_wrapper(self, executor, *args, **kwargs):
        sigma = args[1] if len(args) > 1 else kwargs.get("t")
        mask = kwargs.get("denoise_mask")
        if sigma is not None and torch.is_tensor(mask) and mask.ndim == 5 and mask.shape[2] >= self.prefix_steps:
            kwargs["denoise_mask"] = _apply_prefix_weights(mask, self._live_weights(sigma))
        return executor(*args, **kwargs)


def install_continue_prefix_remask(model, latent, sigmas):
    prefix = int(latent.get(PREFIX_STEPS_KEY, 0) or 0)
    streams = _streams_from_latent(latent)
    if prefix < 1 or not streams or prefix >= int(streams[0].shape[2]):
        return model
    if not callable(getattr(model, "clone", None)):
        return model
    patched = model.clone()
    setter = getattr(patched, "set_model_denoise_mask_function", None)
    if not callable(setter):
        return model
    seam = max(0.0, min(0.95, float(latent.get(CONTINUE_SEAM_KEY, 0.1))))
    state = _PrefixRemask(prefix, sigmas, streams[0].shape, seam)
    setter(state.denoise_mask_function)
    try:
        from comfy.patcher_extension import WrappersMP
        patched.add_wrapper_with_key(WrappersMP.APPLY_MODEL, "h3studio_continue_prefix_remask",
                                     state.apply_model_wrapper, state.apply_model_wrapper)
    except Exception:
        pass
    setattr(patched, "_h3studio_continue_remask", state)
    return patched


def uninstall_continue_prefix_remask(model):
    """Compatibility hook retained for SelfLift sampler cleanup."""
    return model

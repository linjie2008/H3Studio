"""AV latent helpers used by the standalone SelfLift operator."""

import torch


def _streams_from_latent(latent):
    samples = latent["samples"]
    if torch.is_tensor(samples):
        raise ValueError("MiniMax H3 AV NestedTensor required")
    return list(samples.unbind()) if hasattr(samples, "unbind") else list(samples)


def _repack_av_streams(streams, template=None):
    import comfy.nested_tensor
    return comfy.nested_tensor.NestedTensor(tuple(streams))


def av_pixel_size(latent):
    video = _streams_from_latent(latent)[0]
    return int(video.shape[-1]) * 16, int(video.shape[-2]) * 16

"""Small execution operators for the independent H3 Studio application.

H3 Studio owns the director UI, timeline and queue. These nodes only transform
conditioning or run model internal sampling. Adapted algorithms are licensed
under Apache-2.0; see LICENSE in this folder.
"""

from .ops import H3StudioContinueRemask, H3StudioLatentContinue, H3StudioLatentUpscale, H3StudioSemanticBridge, H3StudioSelfLift

NODE_CLASS_MAPPINGS = {
    "H3StudioSemanticBridge": H3StudioSemanticBridge,
    "H3StudioSelfLift": H3StudioSelfLift,
    "H3StudioLatentUpscale": H3StudioLatentUpscale,
    "H3StudioLatentContinue": H3StudioLatentContinue,
    "H3StudioContinueRemask": H3StudioContinueRemask,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3StudioSemanticBridge": "H3 Studio Semantic Bridge",
    "H3StudioSelfLift": "H3 Studio SelfLift Sample",
    "H3StudioLatentUpscale": "H3 Studio H3 Latent Upscale",
    "H3StudioLatentContinue": "H3 Studio H3 Latent Continue",
    "H3StudioContinueRemask": "H3 Studio Continuity Remask",
}

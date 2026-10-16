# H3 Studio execution nodes

This small ComfyUI extension exposes H3 Studio operators for semantic bridging,
SelfLift sampling, 3D latent upscaling, and temporal latent continuation.
Project management, timeline planning, shot sequencing, preview decisions, and
exports remain in the standalone application.

`core_sampling.py`, `h3_latent_upscale.py`, `spatial_tiled_sampling.py`, the
`selflift/` sampling modules, and the latent continuation algorithm are adapted
from AIMixer's ComfyUI MiniMax H3 Director under Apache-2.0. The included
`LICENSE` applies. `ops.py` and the
integration API are H3 Studio code. This extension has no runtime import from
the original Director plugin.

After adding or updating these files, restart ComfyUI. The application health
panel checks whether the nodes were loaded. A semantic bridge adapter is
optional and belongs in `ComfyUI/models/semantic_bridge/`. The SelfLift and
3D latent upscale modes need a H3 3D latent weight under
`ComfyUI/models/latent_upscale_models/`.

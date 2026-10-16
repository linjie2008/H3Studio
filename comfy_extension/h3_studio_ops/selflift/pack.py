"""SelfLift transition controls for an eight step H3 graph."""


def require_euler(sampler_name, pack):
    del pack
    return "euler" if sampler_name != "euler" else sampler_name


def resolve_transition_k(pack, n_steps):
    if pack.get("split_mode") == "transition_step":
        return max(1, min(n_steps - 1, int(pack.get("transition_step", 6))))
    return max(1, min(n_steps - 1, n_steps - int(pack.get("highres_steps", 2))))

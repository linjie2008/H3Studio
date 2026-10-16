"""No previous latent carry is needed: H3 Studio uses official AddGuide clips."""


def prefix_steps_from_pin_frames(n_frames):
    if n_frames:
        raise ValueError("SelfLift latent pin is unavailable in H3 Studio; use AddGuide continuity")
    return 0


def stamp_continue_keys(target, source):
    return dict(target)


def copy_low_tail(*args, **kwargs):
    raise ValueError("Native low latent carry is unavailable; use AddGuide continuity")


def lock_prefix_keep_mask(*args, **kwargs):
    raise ValueError("SelfLift latent pin is unavailable; use AddGuide continuity")


def match_lift_prefix_dc(*args, **kwargs):
    raise ValueError("SelfLift latent pin is unavailable; use AddGuide continuity")

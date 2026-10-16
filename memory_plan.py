"""Conservative H3 canvas planning from ComfyUI CUDA memory statistics."""

from __future__ import annotations

import math


GIB = 1024**3
H3_NATIVE_MEGAPIXELS = 1.04


def fit_canvas(width: int, height: int, budget_megapixels: float) -> tuple[int, int]:
    """Scale down to a 32-pixel grid without exceeding an MP budget."""
    width, height = max(32, int(width)), max(32, int(height))
    budget_pixels = max(32 * 32, float(budget_megapixels) * 1_000_000)
    pixels = width * height
    if pixels <= budget_pixels:
        return width, height
    scale = math.sqrt(budget_pixels / pixels)
    out_w = max(32, int(width * scale // 32) * 32)
    out_h = max(32, int(height * scale // 32) * 32)
    while out_w * out_h > budget_pixels:
        if out_w >= out_h and out_w > 32:
            out_w -= 32
        elif out_h > 32:
            out_h -= 32
        else:
            break
    return out_w, out_h


def plan_canvas(width: int, height: int, total_bytes: int = 0, free_bytes: int = 0,
                queue_busy: bool = False) -> dict:
    """Plan an H3 canvas; VRAM bands are conservative heuristics, not guarantees."""
    total_gib = max(0, int(total_bytes)) / GIB
    free_gib = max(0, int(free_bytes)) / GIB
    if total_gib <= 0:
        budget = 0.40
        reason = "无法读取显存容量，使用保守预览画布预算"
    elif total_gib < 16:
        budget = 0.40
        reason = f"总显存 {total_gib:.1f} GiB，使用预览画布预算"
    elif total_gib < 23:
        budget = 0.60
        reason = f"总显存 {total_gib:.1f} GiB，使用标准画布预算"
    elif total_gib < 32:
        budget = 0.78
        reason = f"总显存 {total_gib:.1f} GiB，使用较高画布预算"
    else:
        budget = H3_NATIVE_MEGAPIXELS
        reason = f"总显存 {total_gib:.1f} GiB，可使用 H3 原生画布预算"

    if not queue_busy and free_bytes > 0:
        low_free_limit = max(4 * GIB, int(total_bytes * 0.15))
        if free_bytes < low_free_limit and budget > 0.40:
            budget = 0.40
            reason += f"；当前空闲显存仅 {free_gib:.1f} GiB，降至预览预算"
    elif queue_busy:
        reason += "；ComfyUI 队列繁忙，未使用瞬时空闲显存判断"

    budget = min(budget, H3_NATIVE_MEGAPIXELS)
    out_width, out_height = fit_canvas(width, height, budget)
    return {
        "requested_width": int(width), "requested_height": int(height),
        "width": out_width, "height": out_height,
        "requested_megapixels": round(int(width) * int(height) / 1_000_000, 3),
        "actual_megapixels": round(out_width * out_height / 1_000_000, 3),
        "budget_megapixels": budget,
        "total_vram_gib": round(total_gib, 2),
        "free_vram_gib": round(free_gib, 2) if free_bytes else None,
        "adjusted": (out_width, out_height) != (int(width), int(height)),
        "reason": reason,
        "queue_busy": bool(queue_busy),
    }

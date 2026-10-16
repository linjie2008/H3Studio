"""Keep project-wide image aliases compact after an image is deleted."""

from __future__ import annotations

import copy
import re


_PICTURE_TAG = re.compile(r"<Picture\s+(\d+)>|@图片\s*(\d+)", re.IGNORECASE)


def normalized_project_image_references(project: dict, assets: dict) -> dict:
    """Return a render-safe project snapshot without changing stored user data."""
    normalized = copy.deepcopy(project)
    project_id = normalized.get("id")
    if not project_id or not normalized.get("asset_order"):
        return normalized
    active_image_ids = [asset_id for asset_id, asset in assets.items()
                        if asset.get("kind") == "image" and asset.get("project_id") == project_id]
    renumber_project_image_references(normalized, active_image_ids)
    return normalized


def renumber_project_image_references(project: dict, active_image_ids: list[str]) -> None:
    """Renumber surviving images from 1 and preserve prompt references by asset ID.

    Tags that pointed to a deleted image are removed so they cannot accidentally
    bind to a different image after its number is reused.
    """
    old_numbers = project.get("reference_image_numbers") or {}
    active = set(active_image_ids)
    original_positions = {asset_id: index for index, asset_id in enumerate(active_image_ids)}
    positions = {asset_id: index for index, asset_id in enumerate(project.get("asset_order") or [])}
    ordered = sorted(active_image_ids, key=lambda asset_id: (
        positions.get(asset_id, len(positions)), original_positions[asset_id]))
    new_numbers = {asset_id: index for index, asset_id in enumerate(ordered, 1)}
    old_to_new: dict[int, int] = {}
    deleted_numbers: set[int] = set()
    for asset_id, value in old_numbers.items():
        try:
            number = int(value)
        except (TypeError, ValueError):
            continue
        if number > 0:
            if asset_id in active:
                old_to_new.setdefault(number, new_numbers[asset_id])
            else:
                deleted_numbers.add(number)

    project["reference_image_numbers"] = new_numbers

    def rewrite(text: str) -> str:
        def replace(match: re.Match) -> str:
            number = int(match.group(1) or match.group(2))
            is_at_alias = match.group(2) is not None
            if number in old_to_new:
                return f"@图片{old_to_new[number]}" if is_at_alias else f"<Picture {old_to_new[number]}>"
            if number in deleted_numbers:
                return ""
            return match.group(0)
        return _PICTURE_TAG.sub(replace, text)

    if "shared_prompt" in project:
        project["shared_prompt"] = rewrite(str(project.get("shared_prompt") or ""))
    for shot in project.get("shots", []):
        if "prompt" in shot:
            shot["prompt"] = rewrite(str(shot.get("prompt") or ""))
        if isinstance(shot.get("manual_ref_images"), list):
            shot["manual_ref_images"] = [asset_id for asset_id in shot["manual_ref_images"]
                                         if asset_id in active]
    synchronize_project_image_bindings(project)


def synchronize_project_image_bindings(project: dict) -> None:
    """Only authored picture tags become shot-level image inputs.

    ``manual_ref_images`` stores gallery selection for the UI; it is not a
    generation binding, including for legacy shots without any picture tags.
    """
    by_number = {int(number): asset_id for asset_id, number in
                 (project.get("reference_image_numbers") or {}).items()
                 if str(number).isdigit() and int(number) > 0}
    shared = set((project.get("shared_refs") or {}).get("images") or [])
    for shot in project.get("shots", []):
        if shot.get("mode") not in ("r2v", "rv2v"):
            continue
        numbers = [int(match.group(1) or match.group(2)) for match in
                   _PICTURE_TAG.finditer(str(shot.get("prompt") or ""))]
        if not numbers:
            shot["ref_images"] = []
            continue
        missing = [number for number in numbers if number not in by_number]
        if missing:
            # The runner reports the invalid alias instead of silently sending
            # a different image occupying the same local H3 slot.
            continue
        tagged = [by_number[number] for number in numbers if by_number[number] not in shared]
        shot["ref_images"] = list(dict.fromkeys(tagged))


def assign_single_image_per_shot(project: dict, assignments: dict[str, str]) -> None:
    """Bulk-assign one non-shared image to each chosen shot by immutable asset ID.

    This is an explicit sequence operation, not a guess based on shot position.
    Shared scene references and their prompt tags are left untouched.
    """
    numbers = project.get("reference_image_numbers") or {}
    by_number = {int(number): asset_id for asset_id, number in numbers.items()
                 if str(number).isdigit() and int(number) > 0}
    shared = set((project.get("shared_refs") or {}).get("images") or [])
    if len(by_number) != len(numbers):
        raise ValueError("图片编号存在重复或无效值，请先同步素材顺序")
    for shot in project.get("shots", []):
        target_id = assignments.get(shot.get("id"))
        if not target_id:
            continue
        if target_id not in numbers or target_id in shared:
            raise ValueError("顺序绑定的目标图片无效或属于共享场景")
        target_number = int(numbers[target_id])
        replaced = False

        def replace(match: re.Match) -> str:
            nonlocal replaced
            source_number = int(match.group(1) or match.group(2))
            if by_number.get(source_number) in shared:
                return match.group(0)
            replaced = True
            return (f"@图片{target_number}" if match.group(2) is not None
                    else f"<Picture {target_number}>")

        prompt = _PICTURE_TAG.sub(replace, str(shot.get("prompt") or ""))
        if not replaced:
            prompt = prompt.rstrip() + (" " if prompt.strip() else "") + f"<Picture {target_number}>"
        shot["prompt"] = prompt
        shot["ref_images"] = [target_id]



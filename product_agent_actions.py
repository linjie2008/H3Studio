"""Bounded project tools with preview, explicit confirmation and version checks."""

from __future__ import annotations

import copy
import secrets
import threading
import time

from engine import ALL_MODES
from store import OUTPUTS, change, new_shot, read, uid

PENDING: dict[str, dict] = {}
PENDING_LOCK = threading.Lock()
TTL_SECONDS = 600
ACTION_NAMES = (
    "rename_project", "add_shot", "duplicate_shot", "delete_shot", "move_shot",
    "set_shot_field", "update_shot", "set_all_durations", "bind_asset", "clear_asset_slot",
    "assign_character", "create_storyboard", "start_render", "create_character",
    "reroll_shot",
)
MODE_ALIASES = {"文生": "t2v", "文生视频": "t2v", "图生": "i2v", "首帧": "i2v",
                "首尾帧": "fl2v", "多参": "r2v", "参考素材": "r2v",
                "视频编辑": "v2v", "视频加参考": "rv2v"}
SLOTS = {"first_image": ("image", "first_image", 1),
         "last_image": ("image", "last_image", 1),
         "source_video": ("video", "source_video", 1),
         "ref_image": ("image", "ref_images", 9),
         "ref_video": ("video", "ref_videos", 3),
         "ref_audio": ("audio", "ref_audios", 3)}
SHOT_CHOICES = {
    "camera_move": {"自动/未指定", "固定镜头", "推近", "拉远", "横向移动", "跟随主体", "环绕主体", "手持"},
    "camera_speed": {"自动/未指定", "缓慢", "平稳", "快速"},
    "action_intensity": {"自然", "轻微", "明显", "夸张"},
    "emotion": {"自动/未指定", "轻松", "紧张", "悲伤", "温馨", "热烈", "悬疑", "浪漫"},
}
SHOT_FIELDS = {"name", "prompt", "duration", "mode", "selected", "continuity", *SHOT_CHOICES}


def _shot(project: dict, args: dict) -> tuple[dict, int]:
    shots = project.get("shots") or []
    shot_id = str(args.get("shot_id") or "")
    if shot_id:
        for index, shot in enumerate(shots):
            if shot.get("id") == shot_id:
                return shot, index
        raise ValueError("镜头 ID 不属于当前项目，请重新选择")
    try:
        index = int(args.get("shot_index")) - 1
    except (TypeError, ValueError):
        raise ValueError("请明确指定镜头编号") from None
    if not 0 <= index < len(shots):
        raise ValueError("镜头编号超出当前项目范围")
    return shots[index], index


def _duration(value) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError("时长必须是 2–15 秒的数字") from None
    if not 2 <= number <= 15 or round(number * 2) != number * 2:
        raise ValueError("时长须在 2–15 秒之间，且以 0.5 秒递增")
    return number


def _mode(value) -> str:
    mode = MODE_ALIASES.get(str(value or "").strip(), str(value or "").strip())
    if mode not in ALL_MODES:
        raise ValueError("生成模式无效")
    return mode


def _shot_field_value(field: str, value):
    if field not in SHOT_FIELDS:
        raise ValueError(f"此镜头字段暂不允许由智能体修改：{field}")
    if field == "duration":
        return _duration(value)
    if field == "mode":
        return _mode(value)
    if field in {"selected", "continuity"}:
        if not isinstance(value, bool):
            raise ValueError("开关值必须为 true 或 false")
        return value
    text = str(value or "").strip()
    if field in SHOT_CHOICES and text not in SHOT_CHOICES[field]:
        raise ValueError(f"{field} 的选项无效")
    if len(text) > (100 if field == "name" else 6000):
        raise ValueError("文本长度超出限制")
    if field == "name" and not text:
        raise ValueError("镜头名称不能为空")
    return text


def _apply(project: dict, catalog: dict, action: dict) -> tuple[str, str | None]:
    """Mutate only an in-memory project; caller decides whether to persist."""
    name = action.get("name")
    args = action.get("args")
    if name not in ACTION_NAMES or not isinstance(args, dict):
        raise ValueError("智能体提出了不支持的项目操作")
    shots = project["shots"]
    if name == "rename_project":
        value = str(args.get("name") or "").strip()
        if not 1 <= len(value) <= 100:
            raise ValueError("项目名称长度需在 1–100 字之间")
        previous = project["name"]
        project["name"] = value
        return f"项目改名：{previous} → {value}", None
    if name == "set_all_durations":
        value = _duration(args.get("duration"))
        for shot in shots:
            shot["duration"] = value
        return f"将全部 {len(shots)} 个镜头时长设为 {value:g} 秒", None
    if name == "add_shot":
        if len(shots) >= 100:
            raise ValueError("项目镜头数已达智能体单次操作上限")
        shot = new_shot()
        shot["id"] = str(action["new_id"])
        shot["name"] = str(args.get("name") or "新镜头").strip()[:100]
        shot["mode"] = _mode(args.get("mode") or "t2v")
        shot["prompt"] = str(args.get("prompt") or "")[:6000]
        shot["duration"] = _duration(args.get("duration", 5))
        after_id = str(args.get("after_shot_id") or "")
        if after_id:
            after_index = next((index for index, item in enumerate(shots)
                                if item.get("id") == after_id), None)
            if after_index is None:
                raise ValueError("插入位置的镜头不存在")
            position = after_index + 1
            shots.insert(position, shot)
        else:
            shots.append(shot)
            position = len(shots) - 1
        return (f"新增镜头 {position + 1}「{shot['name']}」· {shot['mode']} · {shot['duration']:g} 秒"
                + (f"\n提示词：{shot['prompt']}" if shot["prompt"] else "")), shot["id"]
    if name == "create_storyboard":
        brief = str(args.get("brief") or "").strip()
        if len(brief) < 5:
            raise ValueError("请提供至少 5 个字的故事或剧情描述")
        try:
            count = max(1, min(12, int(args.get("count", 4))))
        except (TypeError, ValueError):
            count = 4
        skill_id = str(args.get("skill_id") or project.get("skill_id") or "h3-prompt-writing")
        mode = str(args.get("mode") or "append").strip().lower()
        if mode not in {"append", "replace"}:
            mode = "append"

        generated_shots = action.get("generated_shots") or args.get("generated_shots")
        if not generated_shots:
            if len(brief) < 8:
                raise ValueError("请提供至少 8 个字的故事或剧情描述")
            from prompt_tools import storyboard as generate_storyboard
            generated_shots = generate_storyboard(brief, count, skill_id)
            action["generated_shots"] = generated_shots

        new_shots = []
        for i, item in enumerate(generated_shots):
            s = new_shot()
            s["id"] = uid()
            s["name"] = str(item.get("name") or f"镜头 {len(shots) + i + 1}").strip()[:100]
            s["prompt"] = str(item.get("prompt") or "")[:6000]
            s["duration"] = _duration(item.get("duration", 5))
            if item.get("dialogue_text"):
                s["dialogue_text"] = str(item["dialogue_text"])[:1000]
            if item.get("speaker_id"):
                s["speaker_id"] = str(item["speaker_id"])[:50]
            s["selected"] = True
            new_shots.append(s)

        if mode == "replace":
            shots.clear()
            shots.extend(new_shots)
            action_desc = "替换现有镜头"
        else:
            shots.extend(new_shots)
            action_desc = "追加到时间轴"

        summary_lines = [f"根据故事大纲生成了 {len(new_shots)} 个分镜（{action_desc}）："]
        for s in new_shots:
            p_snippet = (s["prompt"][:50] + "...") if len(s["prompt"]) > 50 else s["prompt"]
            summary_lines.append(f"- {s['name']} ({s['duration']:g}s): {p_snippet or '（无提示词）'}")

        first_new_id = new_shots[0]["id"] if new_shots else None
        return "\n".join(summary_lines), first_new_id
    if name == "create_character":
        char_name = str(args.get("name") or "新角色").strip()[:50]
        if not char_name:
            raise ValueError("角色姓名不能为空")
        description = str(args.get("description") or "").strip()[:1000]
        voice_desc = str(args.get("voice_description") or "").strip()[:500]
        char_id = uid()
        char_obj = {
            "id": char_id,
            "name": char_name,
            "description": description,
            "voice_description": voice_desc,
            "gender": str(args.get("gender") or "unknown"),
            "portrait_asset": None,
            "multiview_asset": None,
            "voice_asset": None
        }
        if "characters" not in project or not isinstance(project["characters"], list):
            project["characters"] = []
        project["characters"].append(char_obj)
        desc_note = f"：{description}" if description else ""
        return f"创建新角色「{char_name}」{desc_note}", None
    if name == "start_render":
        req_shot_ids = args.get("shot_ids")
        target_ids = []
        all_shot_ids = {s.get("id") for s in shots}
        if isinstance(req_shot_ids, list) and req_shot_ids:
            for sid in req_shot_ids:
                if str(sid) in all_shot_ids:
                    target_ids.append(str(sid))
        elif str(args.get("shot_id") or ""):
            sid = str(args.get("shot_id"))
            if sid in all_shot_ids:
                target_ids.append(sid)

        if not target_ids:
            target_ids = [s.get("id") for s in shots if s.get("selected", True)]
        if not target_ids:
            target_ids = [s.get("id") for s in shots]

        if not target_ids:
            raise ValueError("当前项目没有可渲染的镜头")

        action["render_shot_ids"] = target_ids
        for s in shots:
            if s.get("id") in target_ids:
                s["selected"] = True

        shot_names = [s.get("name") or f"镜头{i+1}" for i, s in enumerate(shots) if s.get("id") in target_ids]
        preview_names = "、".join(shot_names[:5]) + (f" 等共 {len(target_ids)} 个镜头" if len(target_ids) > 5 else "")
        return f"准备提交渲染任务（{preview_names}），进入 ComfyUI 生成队列", target_ids[0]
    shot, index = _shot(project, args)
    label = f"镜头 {index + 1}「{shot.get('name') or '未命名'}」"
    if name == "duplicate_shot":
        if len(shots) >= 100:
            raise ValueError("项目镜头数已达智能体单次操作上限")
        cloned = copy.deepcopy(shot)
        cloned["id"] = str(action["new_id"])
        cloned["name"] = (str(shot.get("name") or "新镜头") + " 副本")[:100]
        shots.insert(index + 1, cloned)
        return f"复制{label}，插入到它后面", cloned["id"]
    if name == "delete_shot":
        if len(shots) <= 1:
            raise ValueError("项目至少保留一个镜头")
        shots.pop(index)
        return f"删除{label}；素材和历史成片不会删除", shots[min(index, len(shots) - 1)]["id"]
    if name == "move_shot":
        try:
            position = int(args.get("target_position"))
        except (TypeError, ValueError):
            raise ValueError("目标位置必须是镜头编号") from None
        if not 1 <= position <= len(shots):
            raise ValueError("目标位置超出当前镜头数量")
        shots.pop(index)
        shots.insert(position - 1, shot)
        return f"将{label}移到第 {position} 位", shot["id"]
    if name in {"set_shot_field", "update_shot"}:
        fields = ({str(args.get("field") or ""): args.get("value")}
                  if name == "set_shot_field" else args.get("fields"))
        if not isinstance(fields, dict) or not 1 <= len(fields) <= 10:
            raise ValueError("请提供 1–10 个镜头字段")
        values = {field: _shot_field_value(field, value) for field, value in fields.items()}
        changes = []
        for field, value in values.items():
            old = shot.get(field)
            shot[field] = value
            if old != value:
                changes.append(f"{field}：{old} → {value}")
        return f"修改{label}：{'；'.join(changes) if changes else '参数已是目标值'}", shot["id"]
    if name == "reroll_shot":
        old_seed = shot.get("seed", 42)
        try:
            old_seed_int = int(old_seed)
        except (ValueError, TypeError):
            old_seed_int = 42
        new_seed = (old_seed_int + 1007) % 2147483647
        shot["seed"] = new_seed
        shot["selected"] = True
        action["render_shot_ids"] = [shot["id"]]
        notes = [f"重置种子：{old_seed} → {new_seed}"]
        new_prompt = args.get("new_prompt")
        if new_prompt:
            shot["prompt"] = str(new_prompt).strip()[:6000]
            notes.append("更新提示词")
        return f"{label} 将使用新种子重抽（{'，'.join(notes)}），确认后提交生成队列", shot["id"]
    if name == "assign_character":
        character_id = str(args.get("character_id") or "")
        character = next((item for item in project.get("characters") or []
                          if item.get("id") == character_id), None)
        if not character:
            raise ValueError("角色不属于当前项目")
        assigned = list(shot.get("character_ids") or [])
        enabled = args.get("enabled", True)
        if not isinstance(enabled, bool):
            raise ValueError("角色绑定开关无效")
        if enabled and character_id not in assigned:
            assigned.append(character_id)
        if not enabled:
            assigned = [item for item in assigned if item != character_id]
        shot["character_ids"] = assigned
        note = "；多参模式可能自动引用该角色的形象素材" if enabled and shot.get("mode") in {"r2v", "rv2v"} else ""
        return f"{label}{'绑定' if enabled else '解绑'}角色「{character.get('name') or character_id}」{note}", shot["id"]
    if name == "clear_asset_slot":
        slot = str(args.get("slot") or "")
        if slot not in {"first_image", "last_image", "source_video"}:
            raise ValueError("只能清空首帧、尾帧或源视频；参考素材请在镜头里手动核对标签后移除")
        shot[slot] = None
        return f"清空{label}的 {slot}", shot["id"]
    if name == "bind_asset":
        slot = str(args.get("slot") or "")
        asset_id = str(args.get("asset_id") or "")
        if slot not in SLOTS:
            raise ValueError("素材绑定位置无效")
        kind, field, maximum = SLOTS[slot]
        asset = catalog.get(asset_id)
        if not asset or asset.get("project_id") != project["id"] or asset.get("kind") != kind:
            raise ValueError("素材不存在、类型不符或不属于当前项目")
        if slot.startswith("ref_") and shot.get("mode") not in {"r2v", "rv2v"}:
            raise ValueError("参考素材只能绑定多参或视频加参考镜头；请先切换模式")
        if slot == "first_image" and shot.get("mode") not in {"i2v", "fl2v"}:
            raise ValueError("首帧只在图生或首尾帧模式生效；请先切换模式")
        if slot == "last_image" and shot.get("mode") != "fl2v":
            raise ValueError("尾帧只在首尾帧模式生效；请先切换模式")
        if slot == "source_video" and shot.get("mode") not in {"v2v", "rv2v"}:
            raise ValueError("源视频只在视频编辑模式生效；请先切换模式")
        if maximum == 1:
            shot[field] = asset_id
        else:
            items = list(shot.get(field) or [])
            if asset_id not in items:
                if len(items) >= maximum:
                    raise ValueError(f"此类参考素材最多绑定 {maximum} 个")
                items.append(asset_id)
            shot[field] = items
            if slot == "ref_image":
                manual = list(shot.get("manual_ref_images") or [])
                if asset_id not in manual:
                    manual.append(asset_id)
                shot["manual_ref_images"] = manual
        note = "；多参参考会进入模型输入，即使提示词没有 @ 标签" if maximum > 1 else ""
        return f"将素材「{asset.get('name') or asset_id}」绑定到{label}的 {slot}{note}", shot["id"]
    raise ValueError("智能体操作不受支持")


def catalog_for_planner(project_id: str) -> dict:
    state = read()
    project = state["projects"].get(project_id)
    if not project:
        raise ValueError("请先选择一个项目")
    return {
        "project": {"id": project_id, "name": project.get("name"),
                    "version": project.get("_data_version"),
                    "shots": [{"index": index + 1, "id": shot.get("id"),
                               "name": shot.get("name"), "mode": shot.get("mode"),
                               "duration": shot.get("duration")}
                              for index, shot in enumerate((project.get("shots") or [])[:100])],
                    "characters": [{"id": item.get("id"), "name": item.get("name")}
                                   for item in (project.get("characters") or [])[:80]]},
        "assets": [{"id": asset.get("id"), "name": str(asset.get("name") or "")[:100],
                    "kind": asset.get("kind")}
                   for asset in state["assets"].values()
                   if asset.get("project_id") == project_id][:120],
    }


def propose(project_id: str, action: dict) -> dict:
    state = read()
    project = state["projects"].get(project_id)
    if not project:
        raise ValueError("当前项目不存在")
    action = copy.deepcopy(action)
    if action["name"] == "reroll_shot":
        shot_id = str(action.get("args", {}).get("shot_id") or "")
        source = next((run for run in sorted(state["runs"].values(),
                          key=lambda run: run.get("created_at") or 0, reverse=True)
                       if run.get("project_id") == project_id and run.get("status") == "completed"
                       and not any(result.get("native_take") for result in run.get("results") or [])
                       and all(result.get("file") and (OUTPUTS / result["file"]).is_file()
                               for result in run.get("results") or [])
                       and any(result.get("shot_id") == shot_id
                               for result in run.get("results") or [])), None)
        if source:
            action["source_run_id"] = source["id"]
    if action["name"] in {"add_shot", "duplicate_shot"} and "new_id" not in action:
        action["new_id"] = uid()
    draft = copy.deepcopy(project)
    summary, selected_shot_id = _apply(draft, state["assets"], action)
    if draft == project and action["name"] != "start_render":
        raise ValueError("该操作不会改变项目，无需执行")
    token = secrets.token_urlsafe(24)
    with PENDING_LOCK:
        now = time.time()
        for key in list(PENDING):
            if now - PENDING[key]["created_at"] > TTL_SECONDS:
                PENDING.pop(key, None)
        if len(PENDING) >= 100:
            PENDING.pop(next(iter(PENDING)))
        PENDING[token] = {"project_id": project_id, "version": project.get("_data_version", 0),
                          "action": action, "created_at": now}
    return {"token": token, "summary": summary, "project_name": project.get("name"),
            "selected_shot_id": selected_shot_id, "expires_in": TTL_SECONDS}


def confirm(token: str, project_id: str) -> dict:
    with PENDING_LOCK:
        item = PENDING.pop(token, None)
    if not item or item["project_id"] != project_id or time.time() - item["created_at"] > TTL_SECONDS:
        raise ValueError("操作确认已失效，请重新向助手提出要求")

    def update(state: dict):
        project = state["projects"].get(project_id)
        if not project:
            raise ValueError("项目已不存在")
        if int(project.get("_data_version", 0)) != int(item["version"]):
            raise ValueError("项目在预览后已改变；为避免覆盖，请重新向助手提出要求")
        summary, selected_shot_id = _apply(project, state["assets"], item["action"])
        project["_data_version"] = int(project.get("_data_version", 0)) + 1

        render_ids = item["action"].get("render_shot_ids")
        run_id = None
        if item["action"].get("name") in {"start_render", "reroll_shot"} and render_ids:
            run_id = uid()
            run_data = {
                "id": run_id, "project_id": project_id, "shot_ids": render_ids,
                "status": "queued", "current": "", "results": [], "created_at": time.time()
            }
            if item["action"]["name"] == "reroll_shot" and item["action"].get("source_run_id"):
                run_data["reroll_shot_id"] = render_ids[0]
                run_data["source_run_id"] = item["action"]["source_run_id"]
            if "runs" not in state:
                state["runs"] = {}
            state["runs"][run_id] = run_data

        return {"summary": summary, "project": copy.deepcopy(project),
                "selected_shot_id": selected_shot_id, "run_id": run_id}

    result = change(update)
    if result["run_id"]:
        from runner import run_project
        threading.Thread(target=run_project, args=(result["run_id"],), daemon=True).start()
        result["summary"] += f"；生成任务已提交后台队列 [Run: {result['run_id'][:8]}]"
    return result

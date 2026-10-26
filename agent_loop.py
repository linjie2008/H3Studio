"""创作智能体的主循环（ReAct：想 → 做 → 看 → 再想）。

和原来的「一次规划 → 一次提案」不同，这里主脑可以连续调用多个工具：
读状态 → 查知识 → 改提示词 → 调参数 → 体检 → 看图自评 → 再改。

分级确认在这一层落地：
  read / write   直接执行（write 留痕可回滚）
  destructive / gpu  暂停，等用户点确认后从断点继续
"""

from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor

from agent_audit import create_run, get_run, save_run
from agent_tools import (RISK_DESTRUCTIVE, RISK_GPU, TOOLS, ToolContext,
                         image_numbers, ordered_assets, tool_catalog_prompt)
from prompt_tools import _deepseek_complete

MAX_STEPS = 14
MAX_SECONDS = 300
MODEL_TIMEOUT = 30
MAX_TOKENS = 1800
DECIDE_ATTEMPTS = 2

# 同一个工具在一轮里能问几次。智能体最容易在这里空转：把知识库翻来覆去查、
# 把每个镜头都体检一遍，然后迟迟不给结论。到达上限就让它用已有信息回答。
TOOL_CALL_CAPS = {
    "consult_playbook": 2, "lint_shot": 6, "get_shot": 6, "list_shots": 3,
    "list_assets": 3, "list_characters": 3, "list_runs": 2,
    "get_project_settings": 2, "list_skills": 2, "inspect_run_frames": 2,
}
GLOBAL_READ_CAP = 12
PROMPT_WRITES_PER_SHOT = 2

SYSTEM = (
    "你是 H3 Studio 的创作智能体（AI 导演主脑），不只是问答助手。\n"
    "你的目标是真正把用户的创作需求做出来：读状态、查知识、写提示词、调参数、"
    "体检、看图自评，然后交付结果。\n"
    "工作方式：\n"
    "0. 先回话，再动手。用户只是打招呼或问情况时，用 final 直接回答，"
    "不要一上来就全面体检整个项目；只有用户明确要求动手时才往下做。\n"
    "1. 看清现状要一次查全：把互相独立的只读查询放在同一批（tools 模式）一次问完，"
    "不要一个一个查；通常一批就够：list_shots + list_assets + get_project_settings。\n"
    "2. 用户要求优化、润色或增强提示词时，必须用 rewrite_prompt，让当前项目选定的 Skill 重写；"
    "只有用户给出明确的替换文本时才用 write_prompt 直接保存。其他创作用写入类工具："
    "set_shot_params、set_project_settings、bind_asset、insert_reference_tag、set_reference_role 等。\n"
    "2a. 用户只要求修改全片统一提示词时，只调用 set_shared_prompt；保留原有要求并加入新要求，"
    "成功后立刻收尾，不要改任何镜头提示词或素材。\n"
    "2b. 涉及换人时，先用 set_shot_params 把生成模式、参考图尺寸、底色策略配好，"
    "再用 bind_asset / insert_reference_tag 挂参考图并写提示词，最后 lint_shot 复检。\n"
    "2c. 用户抱怨「不清晰 / 糊 / AI 味重 / 发蜡 / 二采一开就崩」时，先 consult_playbook 查"
    "「清晰度与二采」手册，再按手册逐项核对并用 set_project_settings / set_shot_params 改参数；"
    "改完逐项说明「改了什么、为什么、预期提升多少」，并如实说明后期锐化只提升观感、不增加信息。\n"
    "3. 改完一定要用 lint_shot 体检；涉及成片效果判断时用 inspect_run_frames 看图自评。\n"
    "4. 删除类与渲染类工具会自动挂起等用户确认，这是正常流程，不是错误。\n"
    "5. 工具报错时先读错误原因，换参数或换工具再来一次，不要重复同一个失败调用；"
    "同一次尝试不要把所有改动塞在一个调用里，报错会连累已改的部分。\n"
    "6. 不要反复改同一个东西；不要凭空编造镜头 id、素材 id、Skill id，"
    "必须先用工具查出来再使用。\n"
    "7. 任务完成后用 final 给出中文报告：做了什么、为什么这么改、提示词改动的要点、"
    "还建议用户确认什么。不要说已经生成或已经看到画面，除非你真的调用过相关工具。\n"
    "8. 报告控制在 400 字以内，写重点，不要复述每一步。\n"
    "输出格式：每轮只输出一个 JSON 对象，不要 Markdown、不要解释文字。\n"
    "  单个调用：{\"type\":\"tool\",\"name\":\"...\",\"args\":{...}}\n"
    "  并行只读：{\"type\":\"tools\",\"calls\":[{\"name\":\"...\",\"args\":{...}}, ...]}\n"
    "  收尾：{\"type\":\"final\",\"answer\":\"...\"}"
)


class DecisionError(Exception):
    """The model answered, but not in the JSON shape the loop needs."""

    def __init__(self, message: str, raw: str = ""):
        super().__init__(message)
        self.raw = str(raw or "")


def _parse_json(raw: str) -> dict:
    text = str(raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"```$", "", text).strip()
    decoder = json.JSONDecoder()
    candidates = [0] + [match.start() for match in re.finditer(r"[\[{]", text)]
    for start in dict.fromkeys(candidates):
        try:
            value, _ = decoder.raw_decode(text[start:])
        except (TypeError, ValueError):
            continue
        if isinstance(value, dict):
            native_calls = value.get("tool_calls")
            if isinstance(native_calls, list) and native_calls:
                calls = [_canonical_call(call) for call in native_calls]
                calls = [call for call in calls if call]
                if calls:
                    return {"type": "tools", "calls": calls}
            if value.get("type") == "tools" and isinstance(value.get("calls"), list):
                calls = [_canonical_call(call) for call in value["calls"]]
                calls = [call for call in calls if call]
                if calls:
                    return {"type": "tools", "calls": calls}
            if value.get("type") == "tool" or value.get("name") or value.get("function"):
                call = _canonical_call(value)
                if call:
                    return {"type": "tool", **call}
            if value.get("type") == "final" and str(value.get("answer") or "").strip():
                return value
        elif isinstance(value, list) and value:
            # Some providers emit a bare tool-call array followed by DSML
            # delimiters instead of the requested JSON action envelope.
            calls = [_canonical_call(call) for call in value]
            if all(calls):
                return {"type": "tools", "calls": calls}
    raise ValueError("模型没有返回可解析的 JSON")


def _canonical_call(value: object) -> dict | None:
    """Accept the compact contract and common provider tool-call shapes."""
    if not isinstance(value, dict):
        return None
    function = value.get("function") if isinstance(value.get("function"), dict) else {}
    name = str(value.get("name") or value.get("tool") or function.get("name") or "").strip()
    args = value.get("args", value.get("arguments", value.get("parameters",
                                                          function.get("arguments", {}))))
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            return None
    return {"name": name, "args": args} if name and isinstance(args, dict) else None


def _plain_answer(raw: str) -> str:
    """Accept prose from a provider, never its internal tool-call markup."""
    text = str(raw or "").strip()
    if len(text) < 12 or len(text) > 4000:
        return ""
    if re.search(r"DSML|</?\|?invoke|</?\|?calls|\btool_calls\b", text, re.I):
        return ""
    if text.startswith(("{", "[", "```")):
        return ""
    return text


def project_snapshot(state: dict, project: dict, shot_id: str = "") -> dict:
    """A compact, factual view of the project for the planner."""
    numbers = image_numbers(state, project)
    assets = [{"id": asset["id"], "name": asset.get("name"), "kind": asset.get("kind"),
               "picture": numbers.get(asset["id"]),
               "description": (project.get("image_descriptions") or {}).get(asset["id"], "")}
              for asset in ordered_assets(state, project)]
    shots = []
    for index, shot in enumerate(project.get("shots") or []):
        prompt = str(shot.get("prompt") or "")
        shots.append({
            "index": index + 1, "id": shot.get("id"), "name": shot.get("name"),
            "mode": shot.get("mode"), "duration": shot.get("duration"),
            "selected": bool(shot.get("selected", True)),
            "ref_image_size": shot.get("ref_image_size") or "match",
            "ref_background": shot.get("ref_background") or "auto",
            "refine_mode": shot.get("refine_mode") if shot.get("refine_mode") is not None else project.get("refine_mode", "off"),
            "upscale": shot.get("upscale") if shot.get("upscale") is not None else bool(project.get("upscale")),
            "picture_tags": sorted({int(value) for value in re.findall(r"<Picture\s*(\d+)>", prompt)}),
            "has_source_video": bool(shot.get("source_video")),
            "ref_images": list(shot.get("ref_images") or []),
            "character_ids": list(shot.get("character_ids") or []),
            "prompt": prompt[:1000 if shot.get("id") == shot_id else 180]})
    characters = [{"id": item.get("id"), "name": item.get("name"),
                   "description": str(item.get("description") or "")[:200],
                   "portrait_asset": item.get("portrait_asset"),
                   "multiview_asset": item.get("multiview_asset")}
                  for item in project.get("characters") or []]
    settings = {key: project.get(key) for key in
                ("width", "height", "fps", "generation_speed", "prompt_skill",
                 "h3_lora_character_swap_strength", "h3_lora_mystic_strength",
                 "h3_lora_dmad_strength", "sage_attention", "selflift", "refine_mode",
                 "upscale", "continuous_take")
                if key in project}
    settings["shared_prompt"] = str(project.get("shared_prompt") or "")[:800]
    settings["script_text"] = str(project.get("script_text") or "")[:400]
    settings["sequence_brief"] = str(project.get("sequence_brief") or "")[:400]
    return {"project_id": project.get("id"), "name": project.get("name"),
            "current_shot_id": shot_id, "settings": settings,
            "shots": shots, "assets": assets, "characters": characters}


def _compact_observation(item: dict) -> dict:
    """Keep the model context small: a summary plus a short data digest."""
    data = item.get("data") or {}
    if item.get("tool") == "get_shot" and isinstance(data.get("shot"), dict):
        shot = data["shot"]
        data = {"id": shot.get("id"), "name": shot.get("name"),
                "mode": shot.get("mode"), "prompt": str(shot.get("prompt") or "")[:1500],
                "ref_images": shot.get("ref_images"), "ref_videos": shot.get("ref_videos"),
                "source_video": shot.get("source_video"),
                "bound_assets": data.get("bound_assets"), "lint": data.get("lint")}
    elif item.get("tool") == "list_assets" and isinstance(data.get("assets"), list):
        data = {"assets": [{"id": asset.get("id"), "name": asset.get("name"),
                            "kind": asset.get("kind"), "picture": asset.get("picture")}
                           for asset in data["assets"][:30]]}
    digest = ""
    if data:
        try:
            digest = json.dumps(data, ensure_ascii=False)[:2200 if item.get("tool") == "get_shot" else 1200]
        except (TypeError, ValueError):
            digest = str(data)[:700]
    return {"tool": item.get("tool"), "ok": item.get("ok"),
            "result": str(item.get("result") or "")[:400],
            "data": digest}


def _decide(run: dict, history: list[dict], force_final: bool = False) -> dict:
    """Ask the model for the next action."""
    from store import read
    state = read()
    project = state.get("projects", {}).get(run.get("project_id")) or {}
    snapshot = project_snapshot(state, project, run.get("shot_id") or "")
    catalog = tool_catalog_prompt()
    observations = [_compact_observation(item) for item in (run.get("observations") or [])][-8:]
    if force_final:
        note = ("现在必须收尾：只输出 {\"type\":\"final\",\"answer\":\"...\"}，"
                "用已经查到的信息给出结论，不要再调用任何工具。")
        if _read_only_request(run):
            note += " 本轮用户要求只读；不得声称修改或提交过项目。"
    else:
        note = ("输出 {\"type\":\"tool\",\"name\":...,\"args\":{...}}；"
                "互相独立的只读查询用 {\"type\":\"tools\",\"calls\":[{\"name\":...,\"args\":{...}},...]} 一次问完；"
                "或 {\"type\":\"final\",\"answer\":\"...\"}")
    payload = {
        "request": run.get("question"),
        "current_project": snapshot,
        "tool_results_so_far": observations[-8:],
        "tools": catalog,
        "budget": {
            "steps_left": max(0, MAX_STEPS - len(run.get("steps") or [])),
            "read_calls_left": max(0, GLOBAL_READ_CAP - _read_call_count(run)),
            "tool_limits": TOOL_CALL_CAPS,
        },
        "note": note,
    }
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]
    for item in (history or [])[-4:]:
        if item.get("role") in ("user", "assistant"):
            messages.append({"role": item["role"], "content": str(item.get("content") or "")[:800]})
    raw = _deepseek_complete(messages, max_tokens=MAX_TOKENS, timeout=MODEL_TIMEOUT, json_mode=True)
    try:
        return _parse_json(raw)
    except ValueError as error:
        raise DecisionError(str(error), raw) from None


def _tool_call_count(run: dict, name: str) -> int:
    return len([step for step in (run.get("steps") or []) if step.get("tool") == name])


def _read_call_count(run: dict) -> int:
    return len([step for step in (run.get("steps") or [])
                if (TOOLS.get(step.get("tool")) or {}).get("risk") == "read"])


def _simple_prompt_edit(run: dict) -> bool:
    """A single-shot prompt edit can finish as soon as its write succeeds."""
    question = str(run.get("question") or "")
    return bool(run.get("shot_id") and not _read_only_request(run)
                and re.search(r"提示词|prompt", question, re.I)
                and re.search(r"改|更新|优化|重写|调整|补|增加|删除|替换", question)
                and not re.search(r"全局|全片|统一|全部|所有|每个|多个|同时|分别|批量|然后|之后|再生成|再运行|参数|提交|渲染|和第|及第|第[一二三四五六七八九十\d]+(?:个)?镜头", question))


def _simple_shared_prompt_edit(run: dict) -> bool:
    """A request scoped solely to the project's shared prompt."""
    question = str(run.get("question") or "")
    return bool(not _read_only_request(run)
                and re.search(r"(?:全局|全片|统一)(?:.{0,4})?提示词|提示词(?:中的)?(?:全局|全片|统一)", question)
                and re.search(r"改|更新|优化|重写|调整|补|增加|添加|删除|替换|告诉|设置|要|是", question)
                and not re.search(r"(?:镜头|分镜|素材|角色)(?:提示词|参数|也|都)|同时|然后|之后|分别|批量|渲染|生成", question))


def _skill_optimization_request(run: dict) -> bool:
    """Prompt improvements must use the project's selected Skill."""
    question = str(run.get("question") or "")
    return bool(re.search(r"提示词|prompt", question, re.I)
                and re.search(r"优化|润色|增强|完善|按.{0,30}(?:Skill|skill).{0,20}(?:改|写)", question))


def _read_only_request(run: dict) -> bool:
    """An explicit no-write instruction always overrides model tool choices."""
    question = str(run.get("question") or "")
    return bool(re.search(r"只读|仅(?:查看|查询|核对|分析)|不要(?:修改|改动|写入|执行)|不做(?:修改|改动)|禁止(?:修改|改动|写入)|别(?:改|动)|不修改", question))


def _prompt_write_count(run: dict, shot_id: str) -> int:
    return sum(1 for step in (run.get("steps") or [])
               if step.get("tool") in ("write_prompt", "rewrite_prompt") and step.get("ok")
               and (step.get("args") or {}).get("shot_id", run.get("shot_id")) == shot_id
               and (step.get("data") or {}).get("prompt_after") is not None)


def _capped(run: dict, name: str) -> str:
    """Return the reason this tool may not be called again, or an empty string."""
    cap = TOOL_CALL_CAPS.get(name)
    if cap and _tool_call_count(run, name) >= cap:
        return f"{name} 已经用过 {cap} 次，不要再查了；用已经拿到的信息回答。"
    if (TOOLS.get(name) or {}).get("risk") == "read" and _read_call_count(run) >= GLOBAL_READ_CAP:
        return f"只读查询已经用满 {GLOBAL_READ_CAP} 次，请立刻用现有信息给出结论。"
    return ""


def _run_tool(run: dict, name: str, args: dict) -> dict:
    """Call one tool handler without touching the run bookkeeping.

    Kept separate so several read-only tools can run in parallel: each call
    gets its own context, and the bookkeeping happens later in one place.
    """
    spec = TOOLS.get(name)
    if not spec:
        return {"tool": name, "ok": False, "result": f"没有名为 {name} 的工具"}
    if spec["risk"] != "read" and _read_only_request(run):
        return {"tool": name, "args": args, "ok": False,
                "result": "本轮明确要求只读，已阻止修改或提交操作"}
    ctx = ToolContext(run.get("project_id", ""), run.get("shot_id", ""), run)
    started = time.time()
    try:
        result = spec["handler"](ctx, args if isinstance(args, dict) else {})
    except Exception as error:
        return {"tool": name, "args": args, "ok": False,
                "result": f"工具执行失败：{error}", "seconds": round(time.time() - started, 1)}
    return {"tool": name, "args": args, "ok": True,
            "result": result.get("summary", ""), "data": result.get("data") or {},
            "seconds": round(time.time() - started, 1)}


def _record_observation(run: dict, observation: dict) -> dict:
    """Remember one finished tool call so the loop can undo and report it."""
    run.setdefault("steps", []).append(observation)
    spec = TOOLS.get(observation.get("tool")) or {}
    if spec.get("risk") == "read":
        run.setdefault("read_cache", {})[observation.get("signature") or ""] = observation
        run["repeated_read_count"] = 0
    else:
        run.pop("read_cache", None)
        run["repeated_read_count"] = 0
    return observation


def _execute(run: dict, name: str, args: dict) -> dict:
    """Run one tool and record the outcome."""
    if name == "write_prompt" and _skill_optimization_request(run):
        name = "rewrite_prompt"
        args = {"shot_id": args.get("shot_id") or run.get("shot_id") or "",
                "instruction": str(run.get("question") or "").strip()}
    spec = TOOLS.get(name)
    if not spec:
        observation = {"tool": name, "ok": False, "result": f"没有名为 {name} 的工具"}
        run.setdefault("steps", []).append(observation)
        return observation
    if name in ("write_prompt", "rewrite_prompt"):
        shot_id = str(args.get("shot_id") or run.get("shot_id") or "")
        if shot_id and _prompt_write_count(run, shot_id) >= PROMPT_WRITES_PER_SHOT:
            observation = {"tool": name, "args": args, "ok": False,
                           "result": "这个镜头的提示词本轮已改写两次。请用当前版本收尾，不要继续反复改写。"}
            run.setdefault("steps", []).append(observation)
            return observation
    signature = f"{name}:{json.dumps(args, ensure_ascii=False, sort_keys=True)}"
    seen = run.setdefault("signatures", [])
    # 只有「会改项目」的工具才禁重复：只读工具在状态变化后重查是合理的。
    if spec["risk"] != "read" and signature in seen:
        observation = {"tool": name, "args": args, "ok": False,
                       "result": "这个改动刚才已经做过了，结果不会变化。换一件事做。"}
        run.setdefault("steps", []).append(observation)
        return observation
    # 只读工具同样的查询直接复用上次的观察，避免每次都要等模型再问一遍。
    # 任何写入都会清空缓存，所以不会拿到过期的状态。
    if spec["risk"] == "read":
        cached = (run.get("read_cache") or {}).get(signature)
        if cached:
            observation = {"tool": name, "args": args, "ok": True,
                           "result": str(cached.get("result") or "") + "（与刚才同样的查询，直接复用结果）",
                           "data": cached.get("data") or {}, "cached": True, "seconds": 0.0}
            run.setdefault("steps", []).append(observation)
            run["repeated_read_count"] = int(run.get("repeated_read_count") or 0) + 1
            return observation
    if name == "consult_playbook" and _tool_call_count(run, name) >= TOOL_CALL_CAPS["consult_playbook"]:
        observation = {"tool": name, "args": args, "ok": False,
                       "result": "知识库已经查过两遍了，请依据已查到的内容直接动手或作答，不要再查。"}
        run.setdefault("steps", []).append(observation)
        return observation
    blocked = _capped(run, name)
    if blocked:
        observation = {"tool": name, "args": args, "ok": False, "result": blocked}
        run.setdefault("steps", []).append(observation)
        return observation
    seen.append(signature)
    observation = _run_tool(run, name, args)
    observation["signature"] = signature
    return _record_observation(run, observation)


def _execute_batch(run: dict, calls: list[dict]) -> list[dict]:
    """Run several independent read-only tools at once.

    A read-only batch costs one model round trip instead of one per tool,
    which is the difference between a ten second task and a two minute one.
    """
    prepared: list[tuple[str, dict, str]] = []
    batch_signatures: set[str] = set()
    tool_counts: dict[str, int] = {}
    read_count = _read_call_count(run)
    for call in calls[:6]:
        name = str(call.get("name") or "")
        args = call.get("args") if isinstance(call.get("args"), dict) else {}
        spec = TOOLS.get(name)
        if not spec or spec["risk"] != "read":
            continue
        if _capped(run, name) or read_count + len(prepared) >= GLOBAL_READ_CAP:
            continue
        signature = f"{name}:{json.dumps(args, ensure_ascii=False, sort_keys=True)}"
        if (signature in batch_signatures or signature in (run.get("read_cache") or {})
                or _tool_call_count(run, name) + tool_counts.get(name, 0) >= TOOL_CALL_CAPS.get(name, GLOBAL_READ_CAP)):
            continue
        batch_signatures.add(signature)
        tool_counts[name] = tool_counts.get(name, 0) + 1
        prepared.append((name, args, signature))
    if not prepared:
        # The information was already read or the calls are invalid. Do not
        # show a synthetic failed "batch" step to the user; route the graph
        # to a final answer instead of asking the same tools indefinitely.
        run["read_batch_stalled"] = True
        return []
    results: list[dict] = []
    if len(prepared) == 1:
        name, args, signature = prepared[0]
        observation = _run_tool(run, name, args)
        observation["signature"] = signature
        results.append(observation)
    else:
        with ThreadPoolExecutor(max_workers=len(prepared)) as pool:
            futures = [pool.submit(_run_tool, run, name, args)
                       for name, args, _ in prepared]
            for (name, args, signature), future in zip(prepared, futures):
                observation = future.result()
                observation["signature"] = signature
                results.append(observation)
    for observation in results:
        _record_observation(run, observation)
    return results


def _pending_payload(run: dict, name: str, args: dict) -> dict:
    spec = TOOLS.get(name) or {}
    return {"name": name, "args": args, "risk": spec.get("risk", ""),
            "summary": spec.get("summary", "")}


def _project_state(run: dict) -> dict:
    from store import read
    return read().get("projects", {}).get(run.get("project_id")) or {}


def _final_report(run: dict) -> dict:
    steps = run.get("steps") or []
    applied = [step for step in steps if step.get("ok")]
    prompt_changes = []
    # 同一个镜头被反复改写时，只保留一条对比：第一次的「改前」+ 最后的「改后」。
    index_by_shot: dict[str, int] = {}
    for step in applied:
        data = step.get("data") or {}
        if data.get("prompt_before") is None or data.get("prompt_after") is None:
            continue
        args = step.get("args") if isinstance(step.get("args"), dict) else {}
        shot_id = str(args.get("shot_id") or step.get("shot_id") or "")
        if shot_id and shot_id in index_by_shot:
            prompt_changes[index_by_shot[shot_id]]["after"] = data.get("prompt_after")
            continue
        if shot_id:
            index_by_shot[shot_id] = len(prompt_changes)
        prompt_changes.append({"tool": step.get("tool"), "shot_id": shot_id,
                               "before": data.get("prompt_before"),
                               "after": data.get("prompt_after")})
    return {"steps": steps, "applied": len(applied), "prompt_changes": prompt_changes,
            "changes": len(run.get("changes") or [])}


def run_agent(body: dict, emit=None) -> dict:
    """Entry point for one creative-agent turn (driven by the LangGraph)."""
    question = str(body.get("message") or "").strip()
    if not question:
        raise ValueError("请先说说你想做什么")
    if len(question) > 4000:
        raise ValueError("指令太长，请拆成几次说")
    project_id = str(body.get("project_id") or "")
    shot_id = str(body.get("shot_id") or "")
    history = body.get("history") or []
    if not isinstance(history, list):
        history = []
    run = create_run(project_id, question, shot_id, history)
    # 编排交给 LangGraph：节点、条件路由、检查点都在 agent_graph 里。
    from agent_graph import start_run
    result = start_run(run, emit=emit)
    result["question"] = question
    save_run(run)
    return result


def resume_run(run_id: str, emit=None) -> dict:
    """Continue a run that stopped on its step budget."""
    run = get_run(run_id)
    if not run:
        raise ValueError("这次执行记录已失效，请重新提出要求")
    run["status"] = "running"
    save_run(run)
    from agent_graph import continue_run
    result = continue_run(run_id, extra_steps=8, emit=emit)
    result.setdefault("question", run.get("question") or "")
    return result


def confirm_run(run_id: str, approved: bool, emit=None) -> dict:
    """Resume a paused run after the user approved (or rejected) the pending action."""
    run = get_run(run_id)
    if not run:
        raise ValueError("这次执行记录已失效，请重新提出要求")
    pending = run.get("pending")
    if not pending:
        raise ValueError("当前没有等待确认的操作")
    run["pending"] = None
    run["status"] = "running"
    save_run(run)
    from agent_graph import resume_run as graph_resume
    result = graph_resume(run_id, approved, emit=emit)
    result.setdefault("question", run.get("question") or "")
    return result

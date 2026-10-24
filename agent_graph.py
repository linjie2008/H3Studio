"""创作智能体的 LangGraph 编排层。

为什么改成图，而不是让模型每一步自己决定下一个工具：

1. 控制流是代码，不是模型临场发挥 —— 读状态、动手、复检、收尾各有明确节点，
   不会再出现「查了 20 步还没给结论」的空转。
2. 断点续跑是内建能力 —— checkpointer 按 thread_id 存住中间状态，
   「让它继续」和服务重启后接着做都靠它，不用自己维护 pending。
3. 人在环内用 LangGraph 的 interrupt() —— 删除类与渲染类操作会真的挂起整个图，
   用户点确认后 Command(resume=...) 从原地继续，而不是我们自己拼状态。

节点：
    plan      模型一次决策：批量只读 / 单个工具 / 直接收尾
    act       执行动作；删除与渲染在这里 interrupt() 挂起等确认
    summarize 预算用完时强制收尾，保证一定给结论
    finalize  组装结果并同步给审计流水
"""

from __future__ import annotations

import time
from typing import Annotated, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

import agent_audit
import agent_loop
from agent_tools import RISK_DESTRUCTIVE, RISK_GPU, TOOLS


def _replace(left, right):
    """Reducer: the whole run dict is replaced as one unit."""
    return right if right is not None else left


class CreativeState(TypedDict, total=False):
    run: Annotated[dict, _replace]
    action: Annotated[dict, _replace]
    extra_steps: Annotated[int, _replace]


CHECKPOINTER = MemorySaver()
_GRAPH = None


def _graph():
    global _GRAPH
    if _GRAPH is not None:
        return _GRAPH
    builder = StateGraph(CreativeState)
    builder.add_node("plan", _plan_node)
    builder.add_node("act", _act_node)
    builder.add_node("summarize", _summarize_node)
    builder.add_node("finalize", _finalize_node)
    builder.add_edge(START, "plan")
    builder.add_conditional_edges("plan", _after_plan,
                                  {"act": "act", "finalize": "finalize"})
    builder.add_conditional_edges("act", _after_act,
                                  {"plan": "plan", "summarize": "summarize", "finalize": "finalize"})
    builder.add_edge("summarize", "finalize")
    builder.add_edge("finalize", END)
    _GRAPH = builder.compile(checkpointer=CHECKPOINTER)
    return _GRAPH


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _emit(context, event: dict) -> None:
    emit = (context or {}).get("emit") if isinstance(context, dict) else None
    if not emit:
        return
    try:
        emit(event)
    except Exception:  # a broken listener must never kill the run
        pass


def _mirror(run: dict) -> None:
    """Keep the HTTP-facing audit record in sync with the graph state."""
    record = agent_audit.get_run(run.get("id") or "")
    if not record:
        return
    record["steps"] = list(run.get("steps") or [])
    record["observations"] = list(run.get("observations") or [])
    record["changes"] = list(run.get("changes") or [])
    record["status"] = run.get("status") or record.get("status")
    agent_audit.save_run(record)


def _budget_left(run: dict, extra_steps: int = 0) -> bool:
    steps = len(run.get("steps") or [])
    limit = agent_loop.MAX_STEPS + int(extra_steps or 0)
    if steps >= limit:
        return False
    started = float(run.get("started_at") or time.time())
    return (time.time() - started) < agent_loop.MAX_SECONDS


def _copy_run(state: CreativeState) -> dict:
    """A private copy of the run.

    LangGraph rolls a node's writes back when it interrupts, so a node must
    never mutate the state objects in place — otherwise the rollback restores
    already-corrupted data.
    """
    run = dict(state.get("run") or {})
    for key in ("steps", "observations", "changes", "signatures"):
        run[key] = list(run.get(key) or [])
    run["read_cache"] = dict(run.get("read_cache") or {})
    return run


def _normalize(decision: dict) -> dict:
    """Turn the model's JSON into one of three kinds of action."""
    kind = str(decision.get("type") or "")
    if kind == "final":
        return {"kind": "final", "answer": str(decision.get("answer") or "").strip()}
    if kind == "tools":
        calls = [item for item in (decision.get("calls") or [])
                 if isinstance(item, dict)][:6]
        # Providers sometimes put a write call in a batch despite the read-only
        # contract. Execute it through the normal single-tool path so it is
        # audited (or interrupted for confirmation), never silently discarded.
        for call in calls:
            name = str(call.get("name") or "")
            risk = (TOOLS.get(name) or {}).get("risk")
            if risk and risk != "read":
                return {"kind": "tool", "name": name,
                        "args": call.get("args") if isinstance(call.get("args"), dict) else {}}
        return {"kind": "tools", "calls": calls}
    name = str(decision.get("name") or "")
    args = decision.get("args") if isinstance(decision.get("args"), dict) else {}
    return {"kind": "tool", "name": name, "args": args}


# --------------------------------------------------------------------------
# nodes
# --------------------------------------------------------------------------

def _plan_node(state: CreativeState, runtime=None) -> dict:
    context = getattr(runtime, "context", None)
    run = state.get("run") or {}
    if (not run.get("steps") and agent_loop._simple_prompt_edit(run)
            and agent_loop._skill_optimization_request(run)):
        # The selected shot and project Skill already identify the operation.
        # Asking the planner first adds a cloud round trip without information.
        return {"action": {"kind": "tool", "name": "rewrite_prompt",
                           "args": {"shot_id": run["shot_id"],
                                    "instruction": run["question"]}}}
    _emit(context, {"type": "thinking", "note": "正在思考下一步…"})
    last_error = ""
    decision = None
    for attempt in range(agent_loop.DECIDE_ATTEMPTS):
        try:
            decision = agent_loop._decide(run, run.get("history") or [])
            break
        except agent_loop.DecisionError as error:
            # A provider may return internal DSML/tool-call text as content.
            # Only actual prose is safe to show as an answer.
            text = agent_loop._plain_answer(error.raw)
            if text:
                return {"action": {"kind": "final", "answer": text}}
            last_error = "模型返回了无法识别的工具调用格式"
        except Exception as error:
            last_error = str(error)
        _emit(context, {"type": "retry",
                        "note": f"模型没接上话（{last_error[:60]}），正在重试…",
                        "attempt": attempt + 1, "total": agent_loop.DECIDE_ATTEMPTS})
        time.sleep(1.2 * (attempt + 1))
    if decision is None:
        return {"action": {"kind": "final",
                           "answer": f"这次没能完成：{last_error}。请查看下方执行流水确认已完成的操作，再重试。"}}
    if (decision.get("type") == "final" and agent_loop._simple_prompt_edit(run)
            and not run.get("prompt_fallback_attempted")
            and not any((step.get("data") or {}).get("prompt_after") is not None
                        for step in (run.get("steps") or []))
            and not any(word in str(decision.get("answer") or "")
                        for word in ("无法", "不能", "缺少", "未找到", "需要"))):
        # The model sometimes declares success after changing only a setting.
        # A prompt-edit request must actually write a prompt before finishing.
        next_run = _copy_run(state)
        next_run["prompt_fallback_attempted"] = True
        return {"run": next_run, "action": {"kind": "tool", "name": "rewrite_prompt",
                                          "args": {"shot_id": run["shot_id"],
                                                   "instruction": run["question"]}}}
    return {"action": _normalize(decision)}


def _act_node(state: CreativeState, runtime=None) -> dict:
    context = getattr(runtime, "context", None)
    run = _copy_run(state)
    action = state.get("action") or {}
    kind = action.get("kind")

    if kind == "tools":
        calls = action.get("calls") or []
        _emit(context, {"type": "tool_start", "tool": "batch",
                        "note": "正在并行查看：" + "、".join(
                            str(item.get("name") or "") for item in calls)})
        observations = agent_loop._execute_batch(run, calls)
        run.setdefault("observations", []).extend(observations)
        for observation in observations:
            _emit(context, {"type": "step", "step": observation})
        _mirror(run)
        return {"run": run}

    name = action.get("name") or ""
    args = action.get("args") if isinstance(action.get("args"), dict) else {}
    spec = TOOLS.get(name) or {}

    if spec.get("risk") != "read" and agent_loop._read_only_request(run):
        run["read_only_blocked"] = True
        _emit(context, {"type": "thinking", "note": "本轮要求只读，已阻止修改并整理已查到的信息…"})
        _mirror(run)
        return {"run": run}

    if spec.get("risk") in (RISK_DESTRUCTIVE, RISK_GPU):
        # 人机确认：整张图在这里挂起，用户点确认后从原地继续。
        decision = interrupt({"name": name, "args": args, "risk": spec.get("risk"),
                              "summary": spec.get("summary", "")})
        approved = bool(decision.get("approved")) if isinstance(decision, dict) else bool(decision)
        if not approved:
            observation = {"tool": name, "args": args, "ok": False,
                           "result": "用户拒绝了这次操作。请换一个方案，或询问用户希望怎么做。"}
            run.setdefault("steps", []).append(observation)
            run.setdefault("observations", []).append(observation)
            _emit(context, {"type": "step", "step": observation})
            _mirror(run)
            return {"run": run}
        run["pending"] = None

    _emit(context, {"type": "tool_start", "tool": name,
                    "note": f"正在执行：{spec.get('summary') or name}"})
    observation = agent_loop._execute(run, name, args)
    run.setdefault("observations", []).append(observation)
    _emit(context, {"type": "step", "step": observation})
    if (name in ("write_prompt", "rewrite_prompt") and observation.get("ok")
            and (observation.get("data") or {}).get("prompt_after") is not None
            and agent_loop._simple_prompt_edit(run)):
        run["answer"] = "已更新当前镜头的提示词，并保存到项目。改动内容见下方对比。"
        run["prompt_edit_complete"] = True
    if (name == "set_shared_prompt" and observation.get("ok")
            and agent_loop._simple_shared_prompt_edit(run)):
        run["answer"] = "已更新全片统一提示词，并保存到项目。改动内容见下方对比。"
        run["prompt_edit_complete"] = True
    _mirror(run)
    return {"run": run}


def _summarize_node(state: CreativeState, runtime=None) -> dict:
    """Budget is gone: ask the model once more for a conclusion."""
    context = getattr(runtime, "context", None)
    run = _copy_run(state)
    _emit(context, {"type": "thinking", "note": "时间到了，用现有信息收尾…"})
    answer = ""
    try:
        decision = agent_loop._decide(run, run.get("history") or [], force_final=True)
        if str(decision.get("type") or "") == "final":
            answer = str(decision.get("answer") or "").strip()
    except Exception:
        answer = ""
    if run.get("read_only_blocked") and any(
            phrase in answer for phrase in ("已修改", "已更新", "已写入", "已保存", "已提交")):
        answer = "本轮按只读要求未修改项目；模型未能给出可信的只读结论。"
    if not answer:
        answer = ("本轮按只读要求未修改项目；暂时无法形成可靠结论。" if run.get("read_only_blocked")
                  else "我已经做了 %d 步操作，但时间到了，先把已经查到的结论给你。"
                  % len(run.get("steps") or []))
    run["answer"] = answer
    run["status"] = "done" if (run.get("read_batch_stalled") or run.get("read_only_blocked") or
                               int(run.get("repeated_read_count") or 0) >= 2) else "budget"
    _mirror(run)
    return {"run": run}


def _finalize_node(state: CreativeState, runtime=None) -> dict:
    context = getattr(runtime, "context", None)
    run = _copy_run(state)
    action = state.get("action") or {}
    if action.get("kind") == "final" and not run.get("answer"):
        run["answer"] = action.get("answer") or ""
    if run.get("status") != "budget":
        run["status"] = "done"
    _mirror(run)
    _emit(context, {"type": "final", "answer": str(run.get("answer") or "")})
    return {"run": run}


# --------------------------------------------------------------------------
# routing
# --------------------------------------------------------------------------

def _after_plan(state: CreativeState) -> str:
    action = state.get("action") or {}
    return "finalize" if action.get("kind") == "final" else "act"


def _after_act(state: CreativeState) -> str:
    run = state.get("run") or {}
    if run.get("prompt_edit_complete"):
        return "finalize"
    if (run.get("read_batch_stalled") or run.get("read_only_blocked") or
            int(run.get("repeated_read_count") or 0) >= 2):
        return "summarize"
    if _budget_left(run, int(state.get("extra_steps") or 0)):
        return "plan"
    return "summarize"


# --------------------------------------------------------------------------
# public helpers
# --------------------------------------------------------------------------

def _config(run_id: str) -> dict:
    return {"configurable": {"thread_id": run_id}}


def _pending_from(output: dict) -> dict:
    """Pull the pending action out of a LangGraph interrupt."""
    interrupts = output.get("__interrupt__") or []
    if not interrupts:
        return {}
    value = getattr(interrupts[0], "value", None)
    return dict(value) if isinstance(value, dict) else {}


def start_run(run: dict, emit=None) -> dict:
    """Run the graph from the beginning; return the result dict."""
    run.setdefault("started_at", time.time())
    graph = _graph()
    output = graph.invoke({"run": run}, _config(run["id"]), context={"emit": emit})
    pending = _pending_from(output if isinstance(output, dict) else {})
    if pending:
        run["pending"] = pending
        run["status"] = "waiting"
        _mirror(run)
        return {"status": "needs_confirmation", "run_id": run["id"],
                "pending": pending, "report": agent_loop._final_report(run),
                "project": agent_loop._project_state(run)}
    result = output.get("run") if isinstance(output, dict) else None
    result = result or run
    result["run_id"] = result.get("id") or run["id"]
    result["answer"] = str(result.get("answer") or "")
    result["report"] = agent_loop._final_report(result)
    result["project"] = agent_loop._project_state(result)
    result["resumable"] = result.get("status") == "budget"
    return result


def resume_run(run_id: str, approved: bool, emit=None) -> dict:
    """Continue after the user approved or rejected a destructive action."""
    graph = _graph()
    output = graph.invoke(Command(resume={"approved": bool(approved)}),
                           _config(run_id), context={"emit": emit})
    return _finish(output, run_id, emit)


def continue_run(run_id: str, extra_steps: int = 8, emit=None) -> dict:
    """「让它继续」：从检查点接着跑，并额外给几步预算。"""
    graph = _graph()
    output = graph.invoke({"extra_steps": int(extra_steps)},
                           _config(run_id), context={"emit": emit})
    return _finish(output, run_id, emit)


def _finish(output: dict, run_id: str, emit=None) -> dict:
    pending = _pending_from(output if isinstance(output, dict) else {})
    result = output.get("run") if isinstance(output, dict) else None
    if pending:
        run = result or {}
        run["pending"] = pending
        run["status"] = "waiting"
        _mirror(run)
        return {"status": "needs_confirmation", "run_id": run_id,
                "pending": pending, "report": agent_loop._final_report(run),
                "project": agent_loop._project_state(run)}
    result = result or {}
    result["run_id"] = run_id
    result["answer"] = str(result.get("answer") or "")
    result["report"] = agent_loop._final_report(result)
    result["project"] = agent_loop._project_state(result)
    result["resumable"] = result.get("status") == "budget"
    return result

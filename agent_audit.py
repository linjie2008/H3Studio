"""创作智能体的执行流水、变更留痕与一键回滚。

创作类写入（write 风险）会自动执行，但每一步都要留下「改之前长什么样」，
用户可以在面板上一键回滚整轮改动。删除类与渲染类不会走到这里——
它们必须逐次确认，且删除素材本身不可撤销。
"""

from __future__ import annotations

import copy
import threading
import time

from store import change

RUNS: dict[str, dict] = {}
RUNS_LOCK = threading.Lock()
TTL_SECONDS = 7200
MAX_RUNS = 40


class _Missing(str):
    """Sentinel for a project key that did not exist before the change.

    It must survive copy.deepcopy() as the very same object, otherwise the
    rollback cannot tell "was missing" from "was set to this value".
    It subclasses str so it can also travel through JSON responses and
    LangGraph's msgpack checkpoints.
    """

    def __new__(cls):
        return super().__new__(cls, "__h3_missing__")

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return "<missing>"

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self


MISSING = _Missing()


def _now() -> float:
    return time.time()


def _prune_locked() -> None:
    now = _now()
    for key in [key for key, run in RUNS.items() if now - run.get("created_at", 0) > TTL_SECONDS]:
        RUNS.pop(key, None)
    while len(RUNS) >= MAX_RUNS:
        RUNS.pop(next(iter(RUNS)))


def create_run(project_id: str, question: str, shot_id: str = "",
               history: list | None = None) -> dict:
    with RUNS_LOCK:
        _prune_locked()
        run_id = f"arun_{int(_now() * 1000):x}_{abs(hash((project_id, question, _now()))) % 0xffffff:06x}"
        run = {
            "id": run_id,
            "project_id": project_id,
            "question": question,
            "shot_id": shot_id,
            "history": list(history or [])[-6:],
            "steps": [],
            "observations": [],
            "changes": [],
            "pending": None,
            "status": "running",
            "created_at": _now(),
            "updated_at": _now(),
        }
        RUNS[run_id] = run
        return run


def get_run(run_id: str) -> dict | None:
    with RUNS_LOCK:
        run = RUNS.get(str(run_id or ""))
        if not run:
            return None
        if _now() - run.get("created_at", 0) > TTL_SECONDS:
            RUNS.pop(run["id"], None)
            return None
        return run


def save_run(run: dict) -> dict:
    with RUNS_LOCK:
        run["updated_at"] = _now()
        RUNS[run["id"]] = run
        return run


def list_runs(project_id: str = "", limit: int = 20) -> list[dict]:
    """Recent runs, newest first, with a compact summary for the UI."""
    with RUNS_LOCK:
        _prune_locked()
        items = list(RUNS.values())
    if project_id:
        items = [run for run in items if run.get("project_id") == project_id]
    items.sort(key=lambda run: run.get("created_at", 0), reverse=True)
    summary = []
    for run in items[:max(1, int(limit or 20))]:
        steps = run.get("steps") or []
        summary.append({
            "id": run.get("id"),
            "project_id": run.get("project_id"),
            "question": str(run.get("question") or "")[:200],
            "status": run.get("status"),
            "steps": [{"tool": step.get("tool"), "ok": step.get("ok"),
                       "result": str(step.get("result") or "")[:200],
                       "seconds": step.get("seconds")}
                      for step in steps],
            "applied": len([step for step in steps if step.get("ok")]),
            "changes": len(run.get("changes") or []),
            "pending": (run.get("pending") or {}).get("name") if run.get("pending") else "",
            "created_at": run.get("created_at"),
            "updated_at": run.get("updated_at"),
        })
    return summary


def drop_run(run_id: str) -> None:
    with RUNS_LOCK:
        RUNS.pop(str(run_id or ""), None)


def record_change(run: dict, record: dict) -> None:
    """Store one reversible change so the whole run can be rolled back."""
    if not run:
        return
    run.setdefault("changes", []).append(record)


def _restore_project(project: dict, changes: list[dict]) -> None:
    shots = project.get("shots") or []
    for record in reversed(changes):
        kind = record.get("kind")
        if kind == "shot":
            shot_id = record.get("shot_id")
            index = next((i for i, shot in enumerate(shots) if shot.get("id") == shot_id), None)
            before = record.get("before")
            if before is None:
                if index is not None:
                    shots.pop(index)
                continue
            if index is None:
                position = min(int(record.get("index") or len(shots)), len(shots))
                shots.insert(position, copy.deepcopy(before))
            else:
                shots[index] = copy.deepcopy(before)
        elif kind == "order":
            wanted = list(record.get("before") or [])
            by_id = {shot.get("id"): shot for shot in shots}
            ordered = [by_id[shot_id] for shot_id in wanted if shot_id in by_id]
            ordered.extend(shot for shot in shots if shot.get("id") not in wanted)
            shots[:] = ordered
        elif kind == "project":
            for key, value in (record.get("before") or {}).items():
                if value is MISSING or isinstance(value, _Missing):
                    project.pop(key, None)
                else:
                    project[key] = copy.deepcopy(value)


def undo_run(run_id: str, project_id: str) -> dict:
    run = get_run(run_id)
    if not run or run.get("project_id") != project_id:
        raise ValueError("这次智能体执行记录已失效，无法回滚")
    changes = list(run.get("changes") or [])
    if not changes:
        return {"undone": 0, "summary": "这次执行没有产生可回滚的改动", "project": None}

    def restore(state):
        project = state["projects"].get(project_id)
        if not project:
            raise ValueError("项目不存在")
        _restore_project(project, changes)
        project["_data_version"] = int(project.get("_data_version", 0)) + 1
        return copy.deepcopy(project)

    project = change(restore)
    run["changes"] = []
    run["undone_at"] = _now()
    save_run(run)
    return {"undone": len(changes), "summary": f"已回滚本次智能体的 {len(changes)} 处改动",
            "project": project}

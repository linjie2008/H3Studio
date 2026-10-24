"""Remove run-history references whose media files were deleted from data/outputs.

The director reuses output files across runs (unchanged shots are copied from a
previous run), so deleting one intermediate ``shot_000_h3.mp4`` can leave
several run records pointing at a file that no longer exists. Those records used
to keep rendering broken preview/download links.

Usage::

    python3 repair_missing_media_refs.py            # dry run, prints the diff
    python3 repair_missing_media_refs.py --apply    # back up the DB, then rewrite

A result whose main ``file`` is gone is dropped (nothing playable is left); a
dangling secondary reference (``h3_file`` / ``pre_refine_file`` / ``merged``) is
cleared so the UI hides that button instead of 404ing. Each repaired run keeps a
``media_repairs`` entry recording what was removed and when.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATABASE = ROOT / "data" / "h3studio.sqlite3"
OUTPUTS = ROOT / "data" / "outputs"
SECONDARY_FIELDS = ("h3_file", "pre_refine_file")


def _exists(relative: str | None, outputs: Path) -> bool:
    return bool(relative) and (Path(outputs) / relative).is_file()


def repair_run(run: dict, outputs: Path = OUTPUTS) -> list[str]:
    """Drop vanished deliverables and clear dangling secondary references."""
    removed: list[str] = []
    kept: list[dict] = []
    for result in run.get("results") or []:
        deliverable = result.get("file")
        if deliverable and not _exists(deliverable, outputs):
            removed.append(deliverable)
            continue
        for field in SECONDARY_FIELDS:
            value = result.get(field)
            if value and not _exists(value, outputs):
                removed.append(value)
                result[field] = None
        kept.append(result)
    merged = run.get("merged")
    if isinstance(merged, str) and merged and not _exists(merged, outputs):
        removed.append(merged)
        run["merged"] = None
    if removed:
        run["results"] = kept
        run.setdefault("media_repairs", []).append({
            "at": time.time(), "removed": sorted(set(removed)),
        })
    return removed


def pending_repairs(database: Path = DATABASE, outputs: Path = OUTPUTS) -> dict[str, tuple[dict, list[str]]]:
    """Return ``{run_id: (repaired_run, removed_refs)}`` for runs that need work."""
    with sqlite3.connect(database) as con:
        rows = con.execute("SELECT id, data_json FROM runs").fetchall()
    repairs: dict[str, tuple[dict, list[str]]] = {}
    for run_id, raw in rows:
        run = json.loads(raw)
        removed = repair_run(run, outputs)
        if removed:
            repairs[run_id] = (run, removed)
    return repairs


def backup_database(database: Path = DATABASE) -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    target = database.with_name(f"h3studio.before-media-repair-{stamp}.sqlite3")
    with sqlite3.connect(database) as source, sqlite3.connect(target) as destination:
        source.backup(destination)
    return target


def apply_repairs(repairs: dict[str, tuple[dict, list[str]]],
                  database: Path = DATABASE) -> None:
    with sqlite3.connect(database, timeout=30) as con:
        con.execute("BEGIN IMMEDIATE")
        con.executemany(
            "UPDATE runs SET data_json=? WHERE id=?",
            [(json.dumps(run, ensure_ascii=False), run_id)
             for run_id, (run, _) in repairs.items()],
        )
        con.commit()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--apply", action="store_true", help="write the repairs")
    parser.add_argument("--database", type=Path, default=DATABASE)
    parser.add_argument("--outputs", type=Path, default=OUTPUTS)
    args = parser.parse_args(argv)

    repairs = pending_repairs(args.database, args.outputs)
    references = sum(len(set(removed)) for _, removed in repairs.values())
    for run_id, (_, removed) in sorted(repairs.items()):
        print(f"{run_id}: {', '.join(sorted(set(removed)))}")
    print(f"\n{len(repairs)} 个任务、{references} 条失效引用")
    if not repairs:
        return 0
    if not args.apply:
        print("dry run；加 --apply 才会写入（写入前会先备份数据库）")
        return 0

    backup = backup_database(args.database)
    print(f"已备份数据库：{backup}")
    apply_repairs(repairs, args.database)
    print("已写入修复")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

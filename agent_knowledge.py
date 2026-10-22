"""创作智能体的知识库检索。

主脑动手前可以调用 consult_playbook 查这里。来源有三类：
  1. 工作台自带的提示词 Skill（含参考图换人 Skill）——告诉它「怎么写」
  2. 产品说明文档——告诉它「界面在哪、能做什么」
  3. 智能体排障手册（data/agent_playbooks）——告诉它「参数怎么调、为什么失败」
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SKILL_ROOT = ROOT / "data" / "skills"
PLAYBOOK_ROOT = ROOT / "data" / "agent_playbooks"
DOCUMENTS = ("PRODUCT_GUIDE.md", "README.md", "OPEN_SOURCE_PARITY.md")


def terms(text: str) -> set[str]:
    words = set(re.findall(r"[a-zA-Z][a-zA-Z0-9_-]+", text.lower()))
    for phrase in re.findall(r"[\u4e00-\u9fff]+", text):
        words.update(phrase[index:index + 2] for index in range(len(phrase) - 1))
    return words


def _markdown_files() -> list[Path]:
    files: list[Path] = []
    for name in DOCUMENTS:
        path = ROOT / name
        if path.is_file():
            files.append(path)
    if SKILL_ROOT.is_dir():
        files.extend(sorted(path for path in SKILL_ROOT.rglob("*.md") if path.is_file()))
    if PLAYBOOK_ROOT.is_dir():
        files.extend(sorted(path for path in PLAYBOOK_ROOT.rglob("*.md") if path.is_file()))
    return files


def _blocks(path: Path) -> list[tuple[str, int, str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    heading = path.name
    blocks: list[tuple[str, int, str]] = []
    start = 1
    buffer: list[str] = []
    for number, line in enumerate(lines, 1):
        if line.startswith("#") and buffer:
            blocks.append((heading, start, "\n".join(buffer)))
            buffer = []
            start = number
        if line.startswith("#"):
            heading = line.lstrip("# ").strip() or path.name
        buffer.append(line)
    if buffer:
        blocks.append((heading, start, "\n".join(buffer)))
    return blocks


def search_knowledge(question: str, limit: int = 4) -> list[dict]:
    """Return the most relevant markdown blocks for a question."""
    query = terms(question)
    if not query:
        return []
    matches: list[tuple[int, dict]] = []
    for path in _markdown_files():
        try:
            source = str(path.relative_to(ROOT))
        except ValueError:
            source = path.name
        for title, line_no, content in _blocks(path):
            excerpt = content[:3500]
            score = len(query & terms(title + " " + excerpt)) + 2 * len(query & terms(title))
            if score:
                matches.append((score, {"title": title, "source": source,
                                        "line": line_no, "text": excerpt[:1700]}))
    matches.sort(key=lambda item: item[0], reverse=True)
    seen: set[str] = set()
    results: list[dict] = []
    for _, item in matches:
        key = f"{item['source']}::{item['title']}"
        if key in seen:
            continue
        seen.add(key)
        results.append(item)
        if len(results) >= limit:
            break
    return results

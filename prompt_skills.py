"""Local MiniMax H3 prompt skill library and safe skill package importer."""

from __future__ import annotations

import re
import shutil
import time
import zipfile
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parent
BUILTIN = ROOT / "skills"
CUSTOM = ROOT / "data" / "skills"
AICG3D = ROOT / "data" / "skill_libraries" / "aicg3d" / "skills"
DEFAULT_SKILL = "h3-prompt-writing"
MAX_SKILL_BYTES = 256_000
MAX_ARCHIVE_BYTES = 5_000_000
MAX_ARCHIVE_FILES = 80
TEXT_SUFFIXES = {".md", ".txt"}
MODE_NAMES = {
    "t2v": "T2VA", "i2v": "I2VA", "fl2v": "FL2VA",
    "r2v": "Ref2VA", "v2v": "Ref2VA video editing",
    "rv2v": "Ref2VA video editing with references",
}


def _section(text: str, start: str, end: str | None = None) -> str:
    begin = text.index(start)
    finish = text.index(end, begin + len(start)) if end else len(text)
    return text[begin:finish].strip()


def _base_mode_guide(text: str, mode: str) -> str:
    """Keep shared official rules, but include only the active mode's keyframe rules and example."""
    variants = {
        "t2v": ("**T2VA** has no image-alignment instruction", "**I2VA** always uses:", None, "### Case 1: T2VA", "### Case 2: I2VA"),
        "i2v": ("**I2VA** always uses:", "**FL2VA** always uses:", "### 3.1 I2VA:", "### Case 2: I2VA", "### Case 3: FL2VA"),
        "fl2v": ("**FL2VA** always uses:", "**L2VA** always uses:", "### 3.2 FL2VA:", "### Case 3: FL2VA", "### Case 4: L2VA"),
        "l2v": ("**L2VA** always uses:", "### 2.2 Part Two Contains the Three Core Fields", "### 3.3 L2VA:", "### Case 4: L2VA", None),
    }
    instruction, next_instruction, keyframe, example, next_example = variants[mode]
    parts = [f"# Official H3 guide — {mode.upper()}",
             _section(text, instruction, next_instruction),
             _section(text, "### 2.2 Part Two Contains the Three Core Fields", "## 3. How to Incorporate Keyframes")]
    if keyframe:
        next_heading = {"i2v": "### 3.2 FL2VA:", "fl2v": "### 3.3 L2VA:", "l2v": "## 4. How to Write"}[mode]
        parts.append(_section(text, keyframe, next_heading))
    parts.extend((_section(text, "## 4. How to Write the Three Shared Core Sections", "## 5. Cases"),
                  _section(text, example, next_example)))
    return "\n\n".join(parts)


def _frontmatter(markdown: str, key: str) -> str:
    match = re.search(rf"(?m)^{re.escape(key)}:\s*(.+?)\s*$", markdown.split("---", 2)[1] if markdown.startswith("---") and "---" in markdown[3:] else "")
    return match.group(1).strip().strip("\"'") if match else ""


def _describe(skill_id: str, folder: Path, builtin: bool) -> dict:
    text = (folder / "SKILL.md").read_text(encoding="utf-8", errors="replace")
    name = _frontmatter(text, "name") or folder.name
    description = _frontmatter(text, "description") or "导入的提示词 Skill"
    if skill_id.startswith("aicg3d:"):
        if folder.name == "minimax-h3-short-drama-director":
            name = "MiniMax H3 短剧提示词导演"
        interface = folder / "agents" / "openai.yaml"
        if interface.is_file():
            details = interface.read_text(encoding="utf-8", errors="replace")
            for key, field in (("display_name", "name"), ("short_description", "description")):
                match = re.search(rf"(?m)^\s*{key}:\s*(.+?)\s*$", details)
                if match:
                    if field == "name":
                        name = match.group(1).strip().strip("\"'")
                    else:
                        description = match.group(1).strip().strip("\"'")
    if builtin:
        source, source_label = "official", "官方（MiniMax 指南）"
    elif skill_id.startswith("aicg3d:"):
        source, source_label = "AICG3D", "AICG3D 社区库"
    else:
        source, source_label = "custom", "自定义导入"
    return {"id": skill_id, "name": name, "description": description,
            "builtin": builtin, "protected": builtin or skill_id.startswith("aicg3d:"),
            "source": source, "source_label": source_label, "official": builtin,
            "path": str(folder)}


def list_skills() -> list[dict]:
    result = []
    builtin_root = BUILTIN / DEFAULT_SKILL
    if (builtin_root / "SKILL.md").is_file():
        result.append(_describe(DEFAULT_SKILL, builtin_root, True))
    if CUSTOM.is_dir():
        for folder in sorted(CUSTOM.iterdir()):
            if folder.is_dir() and (folder / "SKILL.md").is_file():
                result.append(_describe(folder.name, folder, False))
    if AICG3D.is_dir():
        for folder in sorted(AICG3D.iterdir()):
            if folder.is_dir() and (folder / "SKILL.md").is_file() and re.fullmatch(r"[a-z0-9_-]+", folder.name):
                result.append(_describe("aicg3d:" + folder.name, folder, False))
    return result


def _folder_for(skill_id: str) -> Path:
    if skill_id == DEFAULT_SKILL:
        folder = BUILTIN / DEFAULT_SKILL
    elif re.fullmatch(r"aicg3d:[a-z0-9_-]{1,80}", skill_id or ""):
        folder = AICG3D / skill_id.split(":", 1)[1]
    elif re.fullmatch(r"custom-[a-z0-9][a-z0-9_-]{0,55}", skill_id or ""):
        folder = CUSTOM / skill_id
    else:
        raise ValueError("提示词 Skill 无效")
    if not (folder / "SKILL.md").is_file():
        raise ValueError("指定的提示词 Skill 不存在，请在项目设置中选择或导入 Skill")
    return folder


def load_skill_context(skill_id: str | None, mode: str, *, single_shot: bool = False) -> str:
    """Return official H3 rules plus the selected creative skill's local references."""
    selected = skill_id or DEFAULT_SKILL
    folder = _folder_for(selected)
    h3_folder = BUILTIN / DEFAULT_SKILL
    mode = mode if mode in MODE_NAMES else "t2v"
    is_reference = mode in {"r2v", "v2v", "rv2v"}
    mode_guide = "ref-en.txt" if is_reference else "base-en.txt"
    files: list[tuple[str, Path]] = [("Official MiniMax H3 skill", h3_folder / "SKILL.md"),
                                    ("Active mode rules", h3_folder / "mode-guides" / f"{mode}.md")]
    if folder != h3_folder:
        files.append(("Selected creative skill — apply its directing, staging, tone, camera vocabulary, "
                      "pacing and continuity rules inside the official H3 format; its output-format or "
                      "field-assembly rules are superseded by the application", folder / "SKILL.md"))
        if selected.startswith("aicg3d:"):
            # AICG3D's director package contains hundreds of KB of optional
            # references. Send only its compact entry point to the AI model.
            if folder.name == "minimax-h3-short-drama-director":
                extras = [folder / "references" / "minimax-h3-plugin-context.md"]
            else:
                extras = [folder / "references" / name for name in ("summary.md", "template.md")]
                if folder.name == "stage-startle-to-truce-encounter" and mode in {"t2v", "i2v", "fl2v", "r2v", "v2v", "rv2v"}:
                    extras.append(folder / "references" / "h3-template.md")
            extras = [path for path in extras if path.is_file()]
        else:
            extras = sorted(path for path in folder.rglob("*")
                            if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES
                            and path.name.lower() != "skill.md")
        files.extend((f"Selected skill reference: {path.relative_to(folder)}", path) for path in extras)
    chunks = []
    total = 0
    for label, path in files:
        if not path.is_file():
            continue
        content = path.read_text(encoding="utf-8", errors="replace")
        total += len(content.encode("utf-8"))
        if total > MAX_SKILL_BYTES:
            raise ValueError("已加载 Skill 的文字资料超过 256 KB，请精简 Skill 参考文档")
        chunks.append(f"## {label}\n{content}")
    official_path = h3_folder / "references" / mode_guide
    official_text = official_path.read_text(encoding="utf-8")
    if is_reference and single_shot:
        # Single-shot enhancement returns only the visual/action description.
        # The six-section schema, retention analysis, and long examples in the
        # full Ref2VA guide are irrelevant here and dominate request size.
        active_guide = "\n\n".join((
            "# Official H3 reference rules for one shot description",
            _section(official_text, "## 2. Reference Labels and Definitions", "### 2.1"),
            _section(official_text, "### 5.3 Using Reference Labels in Shots", "### 5.4"),
        ))
    else:
        active_guide = official_text if is_reference else _base_mode_guide(official_text, mode)
    total += len(active_guide.encode("utf-8"))
    if total > MAX_SKILL_BYTES:
        raise ValueError("已加载 Skill 的文字资料超过 256 KB，请精简 Skill 参考文档")
    chunks.insert(2, f"## Official MiniMax H3 mode guide\n{active_guide}")
    if not chunks:
        raise ValueError("提示词 Skill 内容为空")
    return "\n\n".join(chunks)


def preview_skill(skill_id: str) -> dict:
    """Return safe, bounded text files for the selected project Skill preview."""
    folder = _folder_for(skill_id).resolve()
    files = [folder / "SKILL.md"]
    if skill_id.startswith("aicg3d:"):
        names = ("minimax-h3-plugin-context.md", "h3-adapter.md") if folder.name == "minimax-h3-short-drama-director" else ("summary.md", "template.md", "h3-template.md")
        files.extend(path for name in names if (path := folder / "references" / name).is_file())
    else:
        files.extend(sorted(path for path in folder.rglob("*")
                            if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES
                            and path.name.lower() != "skill.md"))
    def role_for(relative: str, builtin: bool) -> tuple[str, bool]:
        """(role label, is_official_rules) for one previewed file."""
        if not builtin:
            return "Skill 正文" if relative.lower() == "skill.md" else "Skill 参考资料", False
        if relative == "references/base-en.txt":
            return "官方规则 · 基础模式 T2VA/I2VA/FL2VA/L2VA", True
        if relative == "references/ref-en.txt":
            return "官方规则 · 全参考模式 Ref2VA", True
        if relative.startswith("mode-guides/"):
            return "工作台模式要点", False
        return "包装层（官方规则摘要 + 工作台用法）", False

    builtin = skill_id == DEFAULT_SKILL
    entries, total, official_sources = [], 0, []
    for path in files:
        resolved = path.resolve()
        if not resolved.is_relative_to(folder) or not resolved.is_file():
            continue
        content = resolved.read_text(encoding="utf-8", errors="replace")
        total += len(content.encode("utf-8"))
        if total > MAX_SKILL_BYTES:
            raise ValueError("Skill 预览内容超过 256 KB")
        relative = path.relative_to(folder).as_posix()
        role, official_rules = role_for(relative, builtin)
        if official_rules:
            official_sources.append(relative)
        entries.append({"name": relative, "content": content, "role": role,
                        "official_rules": official_rules})
    if not entries:
        raise ValueError("Skill 没有可预览的文字内容")
    described = _describe(skill_id, folder, builtin)
    return {"id": skill_id, "name": described["name"],
            "source": described["source"], "source_label": described["source_label"],
            "official": described["official"], "official_sources": official_sources,
            "files": entries}


def _slug(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9_-]+", "-", value).strip("-_")
    return value[:48] or "video-style"


def _new_custom_id(name: str) -> str:
    CUSTOM.mkdir(parents=True, exist_ok=True)
    base = "custom-" + _slug(name)
    candidate = base
    suffix = 2
    while (CUSTOM / candidate).exists():
        candidate = f"{base}-{suffix}"
        suffix += 1
    return candidate


def import_skill(filename: str, data: bytes) -> dict:
    """Import a plain SKILL.md or ZIP skill package without executing package files."""
    if not data or len(data) > MAX_ARCHIVE_BYTES:
        raise ValueError("Skill 文件为空或超过 5 MB")
    CUSTOM.mkdir(parents=True, exist_ok=True)
    if filename.lower().endswith(".zip"):
        return _import_zip(data)
    if not filename.lower().endswith(".md"):
        raise ValueError("请导入 SKILL.md 或包含 SKILL.md 的 ZIP 包")
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("SKILL.md 必须是 UTF-8 文本") from error
    if not re.search(r"(?m)^#\s+.+", text):
        raise ValueError("文件内容不像有效的 SKILL.md")
    name = _frontmatter(text, "name") or Path(filename).stem
    skill_id = _new_custom_id(name)
    folder = CUSTOM / skill_id
    folder.mkdir()
    (folder / "SKILL.md").write_text(text, encoding="utf-8")
    return _describe(skill_id, folder, False)


def _import_zip(data: bytes) -> dict:
    import io

    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as error:
        raise ValueError("无法读取 ZIP Skill 包") from error
    with archive:
        infos = [info for info in archive.infolist() if not info.is_dir()]
        if len(infos) > MAX_ARCHIVE_FILES or sum(info.file_size for info in infos) > MAX_SKILL_BYTES:
            raise ValueError("Skill ZIP 包内文件过多或解压后超过 256 KB")
        safe_infos = []
        for info in infos:
            path = PurePosixPath(info.filename)
            if path.is_absolute() or ".." in path.parts or "\\" in info.filename:
                raise ValueError("Skill ZIP 包中包含不安全路径")
            if (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Skill ZIP 包不能包含符号链接")
            if path.suffix.lower() in TEXT_SUFFIXES:
                safe_infos.append((info, path))
        skill_entries = [(info, path) for info, path in safe_infos if path.name.lower() == "skill.md"]
        if len(skill_entries) != 1:
            raise ValueError("ZIP 包必须且只能包含一个 SKILL.md")
        skill_info, skill_path = skill_entries[0]
        text = archive.read(skill_info).decode("utf-8-sig", errors="strict")
        name = _frontmatter(text, "name") or skill_path.parent.name or "video-style"
        skill_id = _new_custom_id(name)
        folder = CUSTOM / skill_id
        folder.mkdir()
        root_parts = skill_path.parts[:-1]
        for info, path in safe_infos:
            relative = path.relative_to(PurePosixPath(*root_parts)) if root_parts and path.parts[:len(root_parts)] == root_parts else path
            if relative.is_absolute() or ".." in relative.parts:
                shutil.rmtree(folder, ignore_errors=True)
                raise ValueError("Skill ZIP 结构无效")
            target = folder.joinpath(*relative.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(info))
        if not (folder / "SKILL.md").is_file():
            shutil.rmtree(folder, ignore_errors=True)
            raise ValueError("没有找到根目录 SKILL.md")
    return _describe(skill_id, folder, False)


def delete_skill(skill_id: str) -> None:
    if not re.fullmatch(r"custom-[a-z0-9][a-z0-9_-]{0,55}", skill_id or ""):
        raise ValueError("只能删除导入的自定义 Skill")
    folder = CUSTOM / skill_id
    if not (folder / "SKILL.md").is_file():
        raise ValueError("提示词 Skill 不存在")
    shutil.rmtree(folder)


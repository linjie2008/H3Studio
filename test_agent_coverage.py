"""能力面覆盖测试：工作台能改的东西，创作智能体必须都能改。

这个测试是照着界面本身写的：
- 扫 static/index.html 里的 data-project / data-wb / data-character-field /
  data-entity-field，得出「系统真正暴露给用户的可改字段」；
- 再跟智能体工具的白名单逐条对照。

以前就是缺这条检查，智能体才会「不认识全片统一提示词」。
以后谁在界面上加了字段又忘了给智能体，这个测试立刻红。
"""

import re
import unittest
from pathlib import Path

import agent_tools

ROOT = Path(__file__).resolve().parent
PAGE = (ROOT / "static" / "index.html").read_text(encoding="utf-8")

# 这些字段有专门的工具负责，不在 set_project_settings / set_shot_params 里。
DEDICATED_TOOLS = {
    "shared_prompt": "set_shared_prompt",
    "prompt": "write_prompt",
    "character_ids": "assign_character",
    "entity_ids": "set_shot_params",
    "scene_id": "set_shot_scene",
    "portrait_asset": "bind_asset",
    "multiview_asset": "bind_asset",
    "voice_asset": "bind_asset",
    "generation_source": "bind_asset",
    "image_assets": "bind_asset",
}

# 界面上有、但它不是「内容字段」，由其它机制管（画布联动、模式联动等）。
IGNORED = {
    "aspect",            # 画面比例 = 项目 width/height 的另一个入口
    "resolution",        # 像素档位 = 项目 width/height 的另一个入口
    "duration",          # 已有
}


def _fields(pattern: str) -> set[str]:
    return set(re.findall(pattern, PAGE))


def _tool_allows(tool_name: str, key: str) -> bool:
    """Does the tool's whitelist accept this key? Read the source, don't guess."""
    source = Path(agent_tools.__file__).read_text(encoding="utf-8")
    marker = f'def _{tool_name}('
    start = source.find(marker)
    if start < 0:
        return False
    end = source.find("\n@tool(", start)
    body = source[start:end if end > 0 else len(source)]
    return f'"{key}"' in body


class CapabilityCoverageTests(unittest.TestCase):
    def test_project_settings_cover_every_project_field(self):
        fields = _fields(r'data-project="([a-z_]+)"')
        self.assertGreater(len(fields), 20, "没扫到项目字段，测试本身失效了")
        missing = []
        for field in sorted(fields):
            if field in DEDICATED_TOOLS or field in IGNORED:
                continue
            if not _tool_allows("set_project_settings", field):
                missing.append(field)
        self.assertEqual(missing, [],
                         f"这些项目字段智能体改不了：{'、'.join(missing)}")

    def test_shot_params_cover_every_shot_field(self):
        fields = _fields(r'data-wb="([a-z_]+)"')
        self.assertGreater(len(fields), 10, "没扫到镜头字段，测试本身失效了")
        missing = []
        for field in sorted(fields):
            if field in DEDICATED_TOOLS or field in IGNORED:
                continue
            if not _tool_allows("set_shot_params", field):
                missing.append(field)
        self.assertEqual(missing, [],
                         f"这些镜头字段智能体改不了：{'、'.join(missing)}")

    def test_character_profile_fields_are_reachable(self):
        fields = _fields(r'data-character-field="([a-z_]+)"')
        self.assertGreater(len(fields), 3, "没扫到角色字段，测试本身失效了")
        missing = [field for field in sorted(fields)
                   if field not in DEDICATED_TOOLS
                   and not _tool_allows("update_character", field)]
        self.assertEqual(missing, [],
                         f"这些角色档案字段智能体改不了：{'、'.join(missing)}")

    def test_entity_profile_has_its_own_tools(self):
        fields = _fields(r'data-entity-field="([a-z_]+)"')
        self.assertGreater(len(fields), 3, "没扫到实体字段，测试本身失效了")
        names = set(agent_tools.TOOLS)
        self.assertIn("update_entity", names, "实体档案没有可用的工具")
        self.assertIn("create_entity", names, "实体档案没有可用的工具")
        source = Path(agent_tools.__file__).read_text(encoding="utf-8")
        for field in sorted(fields):
            if field in ("image_assets", "generation_source"):
                continue
            self.assertIn(f'"{field}"', source,
                          f"实体档案字段 {field} 没有被任何工具处理")

    def test_every_write_tool_is_reversible(self):
        """所有自动执行的写工具都必须留痕，否则「一键回滚」就是假的。"""
        source = Path(agent_tools.__file__).read_text(encoding="utf-8")
        missing = []
        for name, spec in agent_tools.TOOLS.items():
            if spec["risk"] != "write":
                continue
            marker = f"def _{name}("
            start = source.find(marker)
            self.assertGreater(start, 0, name)
            end = source.find("\n@tool(", start)
            body = source[start:end if end > 0 else len(source)]
            if ("mutate_shot(" not in body and "mutate_project(" not in body
                    and "record_change(" not in body):
                missing.append(name)
        self.assertEqual(missing, [],
                         f"这些写工具没有走留痕通道，回滚不了：{'、'.join(missing)}")

    def test_destructive_and_gpu_tools_exist_for_the_high_risk_paths(self):
        risks = {}
        for name, spec in agent_tools.TOOLS.items():
            risks.setdefault(spec["risk"], []).append(name)
        self.assertTrue(risks.get("destructive"), "没有删除类工具")
        self.assertTrue(risks.get("gpu"), "没有渲染类工具")
        self.assertIn("submit_render", risks["gpu"])


if __name__ == "__main__":
    unittest.main()

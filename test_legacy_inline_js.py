"""旧版工作台页面的内联脚本必须保持语法正确。

这份界面是单文件内联 JS，浏览器只要解析失败就是整页白屏（用户看到
「打不开」）。历史上因为删控件时留下一个多余的加号导致过白屏，所以
这里把它固化成一条测试。
"""
import re
import shutil
import subprocess
import unittest
from pathlib import Path

INDEX = Path(__file__).resolve().parent / "static" / "index.html"


class LegacyInlineScriptTests(unittest.TestCase):
    def test_render_uses_existing_dom_ids(self):
        html = INDEX.read_text(encoding="utf-8")
        render = html.split("function render(){", 1)[1].split(
            "const renderBeforeCharacterLibrary", 1)[0]
        ids = set(re.findall(r'\bid=["\']([^"\']+)', html))
        lookups = set(re.findall(r"\$\(['\"]([\w-]+)['\"]\)", render))
        self.assertFalse(lookups - ids, f"render() 引用了不存在的控件：{sorted(lookups - ids)}")

    def test_inline_scripts_parse_as_javascript(self):
        if not INDEX.is_file():
            self.skipTest("static/index.html 不存在")
        html = INDEX.read_text(encoding="utf-8")
        scripts = [code for code in re.findall(
            r"<script(?![^>]*src=)[^>]*>(.*?)</script>", html, re.S) if len(code.strip()) >= 20]
        self.assertTrue(scripts, "没扫到内联脚本，测试本身失效了")
        node = shutil.which("node")
        if not node:
            self.skipTest("没有 node，跳过 JS 语法检查")
        for index, code in enumerate(scripts):
            with tempfile_path(code) as path:
                result = subprocess.run([node, "--check", str(path)],
                                        capture_output=True, text=True, timeout=60)
                self.assertEqual(result.returncode, 0,
                                 f"static/index.html 第 {index + 1} 段内联脚本有语法错误，页面会白屏：\n"
                                 f"{result.stderr[:800]}")


class tempfile_path:
    """把脚本文本写成临时文件（供 node --check 使用）。"""

    def __init__(self, code: str):
        self.code = code

    def __enter__(self):
        import tempfile
        self._tmp = tempfile.NamedTemporaryFile("w", suffix=".js", delete=False, encoding="utf-8")
        self._tmp.write(self.code)
        self._tmp.close()
        return Path(self._tmp.name)

    def __exit__(self, *args):
        Path(self._tmp.name).unlink(missing_ok=True)
        return False


if __name__ == "__main__":
    unittest.main()

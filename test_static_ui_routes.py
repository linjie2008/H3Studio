"""The legacy workbench exposes only its explicitly registered UI modules."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from server.router import resolve
from server.routes import ROUTES
from server.routes import system


class StaticUiRouteTests(unittest.TestCase):
    def test_allowed_assets_serve_the_exact_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ui = root / "static" / "ui"
            ui.mkdir(parents=True)
            for name in ("settings-hub.js", "settings-hub.css", "help.js"):
                (ui / name).write_text(f"asset:{name}", encoding="utf-8")

            class Request:
                def file(self, path):
                    self.body = path.read_text(encoding="utf-8")
                    self.path = path

            with patch.object(system, "ROOT", root):
                for name in ("settings-hub.js", "settings-hub.css", "help.js"):
                    with self.subTest(name=name):
                        route, path, parts, url = resolve(ROUTES, "GET", f"/ui/{name}?v=1")
                        self.assertEqual(route.name, "get_settings_hub_asset")
                        request = Request()
                        route.handler(request, path, parts, url)
                        self.assertEqual(request.path, ui / name)
                        self.assertEqual(request.body, f"asset:{name}")

    def test_other_ui_paths_cannot_reach_static_files(self):
        for path in (
            "/ui/secret.js", "/ui/../index.html", "/ui/%2e%2e/index.html",
            "/ui/settings-hub.js/extra", "/static/ui/help.js",
        ):
            with self.subTest(path=path):
                route, _, _, _ = resolve(ROUTES, "GET", path)
                self.assertIsNone(route)
        route, _, _, _ = resolve(ROUTES, "POST", "/ui/help.js")
        self.assertIsNone(route)

    def test_existing_frontend_entrypoints_are_unchanged(self):
        for path, name in (("/", "get_index"),
                           ("/index.html", "get_index"),
                           ("/app", "get_app_index"),
                           ("/app/", "get_app_index"),
                           ("/app/assets/main.js", "get_app_asset")):
            with self.subTest(path=path):
                route, _, _, _ = resolve(ROUTES, "GET", path)
                self.assertEqual(route.name, name)


if __name__ == "__main__":
    unittest.main()

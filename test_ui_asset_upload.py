import unittest
from pathlib import Path


class AssetUploadUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.html = Path(__file__).with_name("static").joinpath("index.html").read_text(encoding="utf-8")

    def test_shared_references_use_visual_selectors(self):
        self.assertIn('class="shared-ref-gallery"', self.html)
        self.assertIn('data-toggle-shared="${key}"', self.html)
        self.assertIn("绿色边框表示已提供给全部镜头", self.html)

    def test_character_portrait_can_be_uploaded_and_bound_directly(self):
        self.assertIn("data-character-portrait-upload", self.html)
        self.assertIn("从角色卡片上传并绑定原图", self.html)
        self.assertIn("character.portrait_asset=asset.id", self.html)
        self.assertIn("character.multiview_mode='i2i'", self.html)


if __name__ == "__main__":
    unittest.main()

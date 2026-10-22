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

    def test_v2v_and_rv2v_upload_preserves_assets_without_overwrite(self):
        # In v2v and rv2v, uploading an image upgrades/binds to ref_images and prompt without overwriting
        self.assertIn("shot.mode='rv2v';workTab='images';if(attachReference(shot,asset))", self.html)
        # In v2v, uploading a video when source_video already exists upgrades to rv2v and attaches as ref_video
        self.assertIn("else if(shot.source_video!==asset.id){shot.mode='rv2v';workTab='videos'", self.html)
        # In rv2v or r2v, on videos tab, video is bound to ref_videos
        self.assertIn("if(workTab==='videos'){shot.ref_videos||=[]", self.html)
        # toggleShotImageSelection retains both prompt references and manual references
        self.assertIn("shot.ref_images=[...new Set([...tagIds, ...manual])];", self.html)


if __name__ == "__main__":
    unittest.main()

import unittest

from app import parse_byte_range


class MediaRangeTests(unittest.TestCase):
    def test_video_player_byte_ranges(self):
        self.assertEqual(parse_byte_range("bytes=0-", 1000), (0, 999))
        self.assertEqual(parse_byte_range("bytes=500-9999", 1000), (500, 999))
        self.assertEqual(parse_byte_range("bytes=-128", 1000), (872, 999))
        self.assertEqual(parse_byte_range("bytes=-2000", 1000), (0, 999))
        self.assertIsNone(parse_byte_range("", 1000))

    def test_invalid_ranges_are_rejected(self):
        for header in ("bytes=1000-", "bytes=20-10", "bytes=-0",
                       "bytes=0-1,5-6", "bytes=a-b"):
            with self.subTest(header=header), self.assertRaises(ValueError):
                parse_byte_range(header, 1000)


if __name__ == "__main__":
    unittest.main()

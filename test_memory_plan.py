import unittest

from memory_plan import GIB, fit_canvas, plan_canvas


class MemoryPlanTests(unittest.TestCase):
    def test_native_request_is_preserved(self):
        self.assertEqual(fit_canvas(1344, 768, 1.04), (1344, 768))

    def test_large_canvas_is_scaled_to_budget_on_grid(self):
        width, height = fit_canvas(1920, 1088, 0.78)
        self.assertLessEqual(width * height, 780_000)
        self.assertEqual(width % 32, 0)
        self.assertEqual(height % 32, 0)
        self.assertGreater(width / height, 1.7)

    def test_low_free_memory_only_changes_idle_queue_plan(self):
        total = int(24 * GIB)
        idle = plan_canvas(1920, 1088, total, int(2 * GIB), queue_busy=False)
        busy = plan_canvas(1920, 1088, total, int(2 * GIB), queue_busy=True)
        self.assertEqual(idle["budget_megapixels"], 0.40)
        self.assertEqual(busy["budget_megapixels"], 0.78)
        self.assertTrue(busy["queue_busy"])

    def test_missing_stats_use_conservative_budget(self):
        plan = plan_canvas(1920, 1088)
        self.assertEqual(plan["budget_megapixels"], 0.40)
        self.assertIsNone(plan["free_vram_gib"])

    def test_native_cap_is_never_exceeded(self):
        plan = plan_canvas(3840, 2160, int(80 * GIB), int(70 * GIB))
        self.assertEqual(plan["budget_megapixels"], 1.04)
        self.assertLessEqual(plan["width"] * plan["height"], 1_040_000)


if __name__ == "__main__":
    unittest.main()

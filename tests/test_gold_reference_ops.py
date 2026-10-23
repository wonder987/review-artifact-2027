import importlib.util
import os
import unittest

import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "scripts", "gold_reference_ops.py")


def load_module():
    spec = importlib.util.spec_from_file_location("gold_reference_ops", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class GoldReferenceOpsTests(unittest.TestCase):
    def setUp(self):
        self.ops = load_module()

    def test_resource_guard_blocks_either_limit(self):
        self.assertFalse(
            self.ops.resource_decision(5000001, 10.0)["allow_full_read"]
        )
        self.assertFalse(
            self.ops.resource_decision(1000, 256.001)["allow_full_read"]
        )
        self.assertTrue(
            self.ops.resource_decision(5000000, 256.0)["allow_full_read"]
        )

    def test_index_window_is_inclusive_and_feasible(self):
        result = self.ops.index_window(-7.0, 1000.0, 1074000, 0.0, 10001)
        self.assertEqual(result["index_start"], 7000)
        self.assertEqual(result["index_end"], 17000)
        self.assertEqual(result["expected_points"], 10001)

    def test_index_window_rejects_out_of_bounds(self):
        with self.assertRaises(ValueError):
            self.ops.index_window(-1.0, 1000.0, 100, 0.0, 10001)

    def test_alignment_uses_intersection_without_extrapolation(self):
        t1 = np.array([0.0, 1.0, 2.0])
        y1 = np.array([0.0, 1.0, 2.0])
        t2 = np.array([1.0, 2.0, 3.0])
        y2 = np.array([10.0, 20.0, 30.0])
        grid, a, b = self.ops.align_on_intersection(t1, y1, t2, y2, 3)
        np.testing.assert_allclose(grid, [1.0, 1.5, 2.0])
        np.testing.assert_allclose(a, [1.0, 1.5, 2.0])
        np.testing.assert_allclose(b, [10.0, 15.0, 20.0])

    def test_alignment_rejects_nonmonotonic_time(self):
        with self.assertRaises(ValueError):
            self.ops.align_on_intersection(
                np.array([0.0, 2.0, 1.0]),
                np.arange(3.0),
                np.arange(3.0),
                np.arange(3.0),
                3,
            )

    def test_constant_array_correlation_is_null(self):
        result = self.ops.comparison_summary(np.ones(3), np.arange(3.0))
        self.assertIsNone(result["pearson_correlation"])

    def test_array_summary_hash_is_stable(self):
        a = self.ops.array_summary(np.array([1.0, 2.0], dtype=np.float32))
        b = self.ops.array_summary(np.array([1.0, 2.0], dtype=np.float32))
        self.assertEqual(a["sha256"], b["sha256"])
        self.assertEqual(a["finite_count"], 2)


if __name__ == "__main__":
    unittest.main()

import importlib.util
import os
import unittest

import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "scripts", "run_gold_reference_task.py")


def load_module():
    spec = importlib.util.spec_from_file_location("run_gold_reference", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def target(node="\\CCR1", signal="CCR1", points=100, mib=0.01):
    return {
        "server": "server",
        "tree": "east",
        "shot": 1,
        "subtree": "t2",
        "node": node,
        "signal_name": signal,
        "sample_num": points,
        "frequency": 10.0,
        "daqtime": 10.0,
        "trigtime": 0.0,
        "estimated_total_mib": mib,
    }


class FakeValue(object):
    def __init__(self, value):
        self.value = value

    def data(self):
        return self.value


class FakeConnection(object):
    def __init__(self, calls):
        self.calls = calls

    def openTree(self, tree, shot):
        self.calls.append(("open", tree, shot))

    def closeTree(self, tree, shot):
        self.calls.append(("close", tree, shot))

    def get(self, expression):
        self.calls.append(("get", expression))
        if expression.startswith("TCL"):
            return FakeValue(1)
        if expression.startswith("dim_of"):
            return FakeValue(np.linspace(0.0, 9.9, 100))
        return FakeValue(np.linspace(1.0, 2.0, 100).astype(np.float32))


class GoldReferenceTaskTests(unittest.TestCase):
    def setUp(self):
        self.runner = load_module()

    def test_full_read_is_refused_before_connection_when_unsafe(self):
        calls = []

        def factory(server):
            calls.append(("connect", server))
            return FakeConnection(calls)

        unsafe = target(points=6000000, mib=70.0)
        task = {
            "id": "unsafe",
            "subtype": "simple_retrieval",
            "targets": [unsafe],
            "operation": {"kind": "full_retrieval"},
            "safety_policy": {"max_points": 5000000, "max_total_mib": 256.0},
        }
        result = self.runner.execute_task(task, connection_factory=factory)
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["failure_type"], "unsafe_full_read")
        self.assertEqual(calls, [])

    def test_index_expression_uses_inclusive_bounds(self):
        expressions = self.runner.build_target_expressions(
            target(), {"index_start": 7, "index_end": 17}
        )
        self.assertEqual(expressions["signal"], "data(\\CCR1)[7:17]")
        self.assertEqual(expressions["time"], "dim_of(\\CCR1)[7:17]")

    def test_window_validation_checks_endpoints_and_step(self):
        signal = np.arange(11, dtype=np.float32)
        correct_time = np.linspace(0.0, 1.0, 11)
        window = {
            "time_start": 0.0,
            "time_end": 1.0,
            "time_step": 0.1,
        }
        valid = self.runner.validate_arrays(signal, correct_time, window)
        self.assertTrue(valid["window_start_match"])
        self.assertTrue(valid["window_end_match"])
        self.assertTrue(valid["time_step_match"])

        wrong_time = np.linspace(0.2, 1.2, 11)
        invalid = self.runner.validate_arrays(signal, wrong_time, window)
        self.assertFalse(invalid["window_start_match"])
        self.assertFalse(invalid["window_end_match"])

    def test_window_validation_accepts_nanosecond_time_quantization(self):
        signal = np.arange(1001, dtype=np.float32)
        expected_step = 2e-5
        time_axis = -5e-8 + np.arange(1001, dtype=np.float64) * expected_step
        time_axis[1::2] += 1e-9
        window = {
            "time_start": 0.0,
            "time_end": 0.02,
            "time_step": expected_step,
        }
        validation = self.runner.validate_arrays(signal, time_axis, window)
        self.assertTrue(validation["window_start_match"])
        self.assertTrue(validation["window_end_match"])
        self.assertTrue(validation["time_step_match"])

    def test_window_validation_rejects_material_step_error(self):
        signal = np.arange(101, dtype=np.float32)
        time_axis = np.arange(101, dtype=np.float64) * 1.1e-4
        window = {
            "time_start": 0.0,
            "time_end": 0.01,
            "time_step": 1e-4,
        }
        validation = self.runner.validate_arrays(signal, time_axis, window)
        self.assertFalse(validation["time_step_match"])

    def test_non_finite_signal_makes_execution_fail(self):
        calls = []

        class NonFiniteConnection(FakeConnection):
            def get(self, expression):
                value = super(NonFiniteConnection, self).get(expression)
                if not expression.startswith(("TCL", "dim_of")):
                    data = value.data()
                    data[0] = np.nan
                    return FakeValue(data)
                return value

        task = {
            "id": "non_finite",
            "subtype": "simple_retrieval",
            "targets": [target()],
            "operation": {"kind": "full_retrieval"},
            "safety_policy": {"max_points": 5000000, "max_total_mib": 256.0},
        }
        result = self.runner.execute_task(
            task, connection_factory=lambda server: NonFiniteConnection(calls)
        )
        self.assertEqual("error", result["status"])
        self.assertEqual("validation", result["failed_stage"])

    def test_simple_task_reads_signal_and_time(self):
        calls = []
        task = {
            "id": "simple",
            "subtype": "simple_retrieval",
            "targets": [target()],
            "operation": {"kind": "full_retrieval"},
            "safety_policy": {"max_points": 5000000, "max_total_mib": 256.0},
        }
        result = self.runner.execute_task(
            task, connection_factory=lambda server: FakeConnection(calls)
        )
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["targets"][0]["signal"]["size"], 100)
        self.assertTrue(result["targets"][0]["validation"]["same_size"])

    def test_same_shot_pair_reuses_one_connection_and_tree(self):
        calls = []
        factories = []

        def factory(server):
            factories.append(server)
            return FakeConnection(calls)

        second = target(node="\\CCR2", signal="CCR2")
        task = {
            "id": "pair",
            "subtype": "same_shot_alignment",
            "targets": [target(), second],
            "operation": {
                "kind": "same_shot_alignment",
                "per_target_windows": [
                    {"index_start": 0, "index_end": 99},
                    {"index_start": 0, "index_end": 99},
                ],
                "common_points": 100,
            },
            "safety_policy": {"max_points": 5000000, "max_total_mib": 256.0},
        }
        result = self.runner.execute_task(task, connection_factory=factory)
        self.assertEqual(result["status"], "success")
        self.assertEqual(len(factories), 1)
        self.assertEqual(sum(call[0] == "open" for call in calls), 1)
        self.assertEqual(sum(call[0] == "close" for call in calls), 1)


if __name__ == "__main__":
    unittest.main()

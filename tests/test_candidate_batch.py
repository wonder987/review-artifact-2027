import importlib.util
import json
import os
import tempfile
import unittest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "scripts", "batch_validate_candidates.py")


def load_batch_module():
    spec = importlib.util.spec_from_file_location("candidate_batch", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CandidateBatchTests(unittest.TestCase):
    def test_normalize_extended_json_number(self):
        module = load_batch_module()
        value = {"$numberDouble": "-0.0"}
        self.assertEqual(module.normalize_extended_json(value), -0.0)

    def test_build_plan_skips_existing_gold_and_never_full_reads_large(self):
        module = load_batch_module()
        candidates = [
            {
                "shot": 106915,
                "tree": "east",
                "subtree": "t2",
                "signal_name": "CCR1",
                "sample_num": 1074000,
                "frequency": 1000,
                "daqtime": 1074,
                "trigtime": -7,
                "size_band": "safe_full",
                "estimated_total_mib": 12.3,
            },
            {
                "shot": 70000,
                "tree": "east",
                "subtree": "t1",
                "signal_name": "CDG1",
                "sample_num": 770000,
                "frequency": 10000,
                "daqtime": 77,
                "trigtime": -7,
                "size_band": "safe_full",
                "estimated_total_mib": 8.9,
            },
            {
                "shot": 70000,
                "tree": "east",
                "subtree": "t6",
                "signal_name": "POINT0",
                "sample_num": 120000000,
                "frequency": 10000000,
                "daqtime": 12,
                "trigtime": -2,
                "size_band": "very_large",
                "estimated_total_mib": 1373.3,
            },
        ]
        plan = module.build_plan(candidates)
        self.assertEqual([task["signal_name"] for task in plan], ["CDG1", "POINT0"])
        self.assertEqual(plan[0]["planned_actions"], ["preflight", "full_retrieval"])
        self.assertEqual(plan[1]["planned_actions"], ["preflight"])

    def test_build_plan_rejects_safe_label_above_hard_limit(self):
        module = load_batch_module()
        candidates = [
            {
                "shot": 70000,
                "tree": "east",
                "subtree": "t2",
                "signal_name": "MISLABELED",
                "sample_num": 6000000,
                "frequency": 1000,
                "daqtime": 6000,
                "trigtime": 0,
                "size_band": "safe_full",
                "estimated_total_mib": 68.7,
            }
        ]
        plan = module.build_plan(candidates)
        self.assertEqual(plan[0]["planned_actions"], ["preflight"])

    def test_write_plan_is_valid_json(self):
        module = load_batch_module()
        with tempfile.TemporaryDirectory() as directory:
            output = os.path.join(directory, "plan.json")
            module.write_json(output, [{"task_id": "example"}])
            with open(output, "r") as handle:
                self.assertEqual(json.load(handle)[0]["task_id"], "example")


if __name__ == "__main__":
    unittest.main()

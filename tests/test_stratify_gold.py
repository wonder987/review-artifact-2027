import importlib.util
import json
import os
import unittest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "scripts", "stratify_gold_candidates.py")
POOL = os.path.join(ROOT, "benchmark", "candidate_pool_expanded.json")


def load_module():
    spec = importlib.util.spec_from_file_location("stratify_gold", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class StratifyGoldTests(unittest.TestCase):
    def setUp(self):
        with open(POOL, "r") as handle:
            self.candidates = json.load(handle)

    def test_builds_balanced_138_task_draft(self):
        module = load_module()
        manifest = module.build_manifest(self.candidates)
        tasks = manifest["tasks"]

        self.assertEqual(len(tasks), 138)
        self.assertEqual(len({task["id"] for task in tasks}), 138)
        self.assertEqual(
            module.count_by(tasks, "operation_kind"),
            {
                "full_retrieval": 62,
                "index_window": 38,
                "metadata_preflight": 38,
            },
        )
        self.assertEqual(
            module.count_by(tasks, "grounding_level"),
            {
                "explicit_target": 46,
                "subtree_hidden": 46,
                "semantic_request": 46,
            },
        )

    def test_excludes_metadata_mismatch_from_primary_tasks(self):
        module = load_module()
        manifest = module.build_manifest(self.candidates)
        keys = {
            (task["target"]["shot"], task["target"]["node"])
            for task in manifest["tasks"]
        }
        self.assertNotIn((90000, "\\RXC17A"), keys)
        self.assertEqual(
            manifest["excluded_diagnostics"][0]["reason"],
            "mongodb_sample_count_mismatch",
        )

    def test_window_tasks_have_safe_feasible_index_oracles(self):
        module = load_module()
        manifest = module.build_manifest(self.candidates)
        windows = [
            task
            for task in manifest["tasks"]
            if task["operation"]["kind"] == "index_window"
        ]
        for task in windows:
            operation = task["operation"]
            target = task["target"]
            self.assertEqual(operation["expected_points"], 10001)
            self.assertGreaterEqual(operation["index_start"], 0)
            self.assertLess(operation["index_end"], target["sample_num"])
            self.assertGreater(operation["time_end"], operation["time_start"])

    def test_known_oversized_seed_remains_preflight_only(self):
        module = load_module()
        manifest = module.build_manifest(self.candidates)
        task = next(
            task
            for task in manifest["tasks"]
            if task["target"]["shot"] == 106915
            and task["target"]["node"] == "\\AXUV_DIV1"
        )
        self.assertEqual(task["operation"]["kind"], "metadata_preflight")


if __name__ == "__main__":
    unittest.main()

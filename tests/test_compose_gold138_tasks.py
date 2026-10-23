import importlib.util
import json
import os
import unittest
from collections import Counter


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "scripts", "compose_gold138_tasks.py")
POOL = os.path.join(ROOT, "benchmark", "candidate_pool_expanded.json")
PILOT_SHOTS = {70000, 106915, 150000}


def load_module():
    spec = importlib.util.spec_from_file_location("compose_gold138", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_candidates():
    with open(POOL, "r") as handle:
        return json.load(handle)


class ComposeGold138Tests(unittest.TestCase):
    def setUp(self):
        self.module = load_module()
        self.manifest = self.module.compose_manifest(load_candidates())
        self.tasks = self.manifest["tasks"]

    def test_exact_level_and_subtype_counts(self):
        self.assertEqual(
            Counter(task["level"] for task in self.tasks),
            {"L1": 30, "L2": 48, "L3": 60},
        )
        self.assertEqual(
            Counter(
                task["subtype"]
                for task in self.tasks
                if task["level"] == "L2"
            ),
            {
                "subtree_discovery": 16,
                "exact_window": 16,
                "sampling_constrained": 16,
            },
        )
        self.assertEqual(
            Counter(
                task["subtype"]
                for task in self.tasks
                if task["level"] == "L3"
            ),
            {
                "same_shot_alignment": 20,
                "cross_shot_comparison": 20,
                "adaptive_access": 20,
            },
        )

    def test_split_counts_and_no_shot_leakage(self):
        self.assertEqual(
            Counter(task["split"] for task in self.tasks),
            {"pilot": 30, "held_out": 108},
        )
        for task in self.tasks:
            shots = {target["shot"] for target in task["targets"]}
            if task["split"] == "pilot":
                self.assertTrue(shots <= PILOT_SHOTS)
            else:
                self.assertFalse(shots & PILOT_SHOTS)
        splits = self.module.build_splits(self.tasks)
        self.assertEqual(60, len(splits["ablation"]))
        ablation_ids = set(splits["ablation"])
        ablation_tasks = [task for task in self.tasks if task["id"] in ablation_ids]
        self.assertEqual(
            Counter(task["level"] for task in ablation_tasks),
            {"L1": 13, "L2": 21, "L3": 26},
        )
        self.assertTrue(ablation_ids <= set(splits["held_out"]))

    def test_task_ids_specs_and_target_usage_are_bounded(self):
        ids = [task["id"] for task in self.tasks]
        specs = [task["spec_sha256"] for task in self.tasks]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(specs), len(set(specs)))

        usage = Counter()
        for task in self.tasks:
            for target in task["targets"]:
                usage[
                    (target["shot"], target["subtree"], target["signal_name"])
                ] += 1
        self.assertLessEqual(max(usage.values()), 2)

    def test_compound_pairs_are_valid(self):
        for task in self.tasks:
            if task["subtype"] == "same_shot_alignment":
                self.assertEqual(len(task["targets"]), 2)
                self.assertEqual(
                    task["targets"][0]["shot"], task["targets"][1]["shot"]
                )
                self.assertNotEqual(
                    task["targets"][0]["signal_name"],
                    task["targets"][1]["signal_name"],
                )
            if task["subtype"] == "cross_shot_comparison":
                self.assertEqual(len(task["targets"]), 2)
                self.assertNotEqual(
                    task["targets"][0]["shot"], task["targets"][1]["shot"]
                )
                self.assertEqual(
                    task["targets"][0]["signal_name"],
                    task["targets"][1]["signal_name"],
                )

    def test_adaptive_tasks_cover_both_policy_branches(self):
        adaptive = [
            task for task in self.tasks if task["subtype"] == "adaptive_access"
        ]
        self.assertEqual(
            Counter(task["operation"]["expected_decision"] for task in adaptive),
            {"read_full_signal": 10, "limit_time_window_or_downsample": 10},
        )

    def test_excludes_metadata_mismatch(self):
        self.assertTrue(
            all(
                not any(
                    target["shot"] == 90000
                    and target["node"] == "\\RXC17A"
                    for target in task["targets"]
                )
                for task in self.tasks
            )
        )

    def test_excludes_unreadable_bounded_access_targets(self):
        excluded = {
            (50000, "t7", "POINT0"),
            (106915, "t6", "LP1"),
        }
        selected = {
            (target["shot"], target["subtree"], target["signal_name"])
            for task in self.tasks
            for target in task["targets"]
        }
        self.assertTrue(excluded.isdisjoint(selected))
        self.assertEqual(138, len(self.tasks))

    def test_composition_is_deterministic(self):
        again = self.module.compose_manifest(load_candidates())
        self.assertEqual(self.manifest, again)


if __name__ == "__main__":
    unittest.main()

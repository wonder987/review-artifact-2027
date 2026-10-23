import json
import os
import tempfile
import unittest

from scripts.render_primary_zero_shot import render_task_request
from scripts.render_primary_zero_shot_v3 import (
    output_contract,
    render_manifest,
    render_prompt,
)


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(ROOT, "benchmark", "gold138_manifest.json")


def load_manifest():
    with open(MANIFEST, "r") as handle:
        return json.load(handle)


class PrimaryZeroShotV3RendererTests(unittest.TestCase):
    def test_reuses_task_request_exactly(self):
        for task in load_manifest()["tasks"]:
            rendered = render_prompt(task)
            request, _ = rendered["user_prompt"].split("\n\n", 1)
            self.assertEqual(render_task_request(task), request)

    def test_contract_makes_target_identity_unambiguous(self):
        task = load_manifest()["tasks"][0]
        contract = output_contract(task)
        self.assertIn(
            "targets[i].target itself is the identity object", contract
        )
        self.assertIn("Do not add a separate targets[i].identity sibling", contract)
        for token in ("server", "tree", "shot", "node", "subtree"):
            self.assertIn(token, contract)
        for token in (
            "non_empty",
            "same_size",
            "signal_all_finite",
            "time_all_finite",
            "time_monotonic_non_decreasing",
            "time_strictly_increasing",
        ):
            self.assertIn(token, contract)

    def test_contract_has_no_mdsplus_solution_api(self):
        task = load_manifest()["tasks"][0]
        contract = output_contract(task)
        for token in ("openTree", "getNode", "dim_of", "TCL(", ".get("):
            self.assertNotIn(token, contract)

    def test_pair_and_adaptive_sections_are_conditional(self):
        tasks = load_manifest()["tasks"]
        pair = next(t for t in tasks if t["subtype"] == "cross_shot_comparison")
        adaptive = next(t for t in tasks if t["subtype"] == "adaptive_access")
        simple = next(t for t in tasks if t["subtype"] == "simple_retrieval")
        pair_contract = output_contract(pair)
        adaptive_contract = output_contract(adaptive)
        simple_contract = output_contract(simple)

        self.assertIn("derived", pair_contract)
        self.assertIn("derived.comparison", pair_contract)
        for token in (
            "common_time_axis", "aligned_first", "aligned_second", "count",
            "a_mean", "a_std", "b_mean", "b_std", "pearson_correlation",
        ):
            self.assertIn(token, pair_contract)
        self.assertIn("resource_decision", adaptive_contract)
        self.assertNotIn("resource_decision", simple_contract)
        self.assertNotIn("derived", simple_contract)

    def test_manifest_is_deterministic_and_versioned(self):
        with tempfile.TemporaryDirectory() as directory:
            first_path = os.path.join(directory, "first.jsonl")
            second_path = os.path.join(directory, "second.jsonl")
            first = render_manifest(MANIFEST, first_path)
            second = render_manifest(MANIFEST, second_path)
            with open(first_path, "rb") as handle:
                first_bytes = handle.read()
            with open(second_path, "rb") as handle:
                second_bytes = handle.read()

        self.assertEqual(138, len(first))
        self.assertEqual(first, second)
        self.assertEqual(first_bytes, second_bytes)
        self.assertTrue(all(
            record["prompt_version"] == "primary-zero-shot-v3"
            for record in first
        ))


if __name__ == "__main__":
    unittest.main()

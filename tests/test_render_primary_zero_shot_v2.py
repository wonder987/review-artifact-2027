import json
import os
import tempfile
import unittest

from scripts.render_primary_zero_shot import render_task_request
from scripts.render_primary_zero_shot_v2 import render_manifest, render_prompt


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(ROOT, "benchmark", "gold138_manifest.json")


def load_manifest():
    with open(MANIFEST, "r") as handle:
        return json.load(handle)


class PrimaryZeroShotV2RendererTests(unittest.TestCase):
    def test_reuses_v1_task_request_exactly(self):
        for task in load_manifest()["tasks"]:
            rendered = render_prompt(task)
            request, contract = rendered["user_prompt"].split("\n\n", 1)

            self.assertEqual(render_task_request(task), request)
            self.assertIn("sha256", contract)

    def test_contract_requires_semantic_summaries_without_gold_values(self):
        task = load_manifest()["tasks"][0]
        rendered = render_prompt(task)
        contract = rendered["user_prompt"].split("\n\n", 1)[1]

        for token in (
            "targets", "signal", "time_axis", "size", "shape", "dtype",
            "sha256", "finite_count", "min", "max", "mean", "std",
            "validation", "elapsed_seconds",
        ):
            self.assertIn(token, contract)
        self.assertIn("targets[i].target", contract)
        self.assertIn("targets[i].signal", contract)
        self.assertIn("targets[i].time_axis", contract)
        for token in (
            "non_empty", "same_size", "signal_all_finite",
            "time_all_finite", "time_strictly_increasing",
        ):
            self.assertIn(token, contract)
        self.assertNotIn(task["spec_sha256"], rendered["prompt"])
        self.assertNotIn(str(task["operation"].get("index_start")), contract)

    def test_pair_and_adaptive_contract_sections_are_conditional(self):
        tasks = load_manifest()["tasks"]
        pair = next(t for t in tasks if t["subtype"] == "cross_shot_comparison")
        adaptive = next(t for t in tasks if t["subtype"] == "adaptive_access")
        simple = next(t for t in tasks if t["subtype"] == "simple_retrieval")

        self.assertIn("derived", render_prompt(pair)["user_prompt"])
        for token in (
            "a_mean", "a_std", "b_mean", "b_std",
            "pearson_correlation", "count",
        ):
            self.assertIn(token, render_prompt(pair)["user_prompt"])
        self.assertIn("resource_decision", render_prompt(adaptive)["user_prompt"])
        self.assertNotIn("resource_decision", render_prompt(simple)["user_prompt"])

    def test_manifest_is_deterministic_and_versioned(self):
        with tempfile.TemporaryDirectory() as directory:
            first_path = os.path.join(directory, "first.jsonl")
            second_path = os.path.join(directory, "second.jsonl")
            first = render_manifest(MANIFEST, first_path)
            second = render_manifest(MANIFEST, second_path)

            self.assertEqual(138, len(first))
            self.assertEqual(first, second)
            self.assertTrue(all(
                record["prompt_version"] == "primary-zero-shot-v2"
                for record in first
            ))
            with open(first_path, "rb") as handle:
                first_bytes = handle.read()
            with open(second_path, "rb") as handle:
                second_bytes = handle.read()
            self.assertEqual(first_bytes, second_bytes)


if __name__ == "__main__":
    unittest.main()

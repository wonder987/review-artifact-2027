import json
import os
import tempfile
import unittest

from scripts import render_primary_zero_shot
from scripts.render_primary_zero_shot import render_manifest, render_prompt


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(ROOT, "benchmark", "gold138_manifest.json")


def load_manifest():
    with open(MANIFEST, "r") as handle:
        return json.load(handle)


def task_by_subtype(subtype):
    return next(
        task for task in load_manifest()["tasks"] if task["subtype"] == subtype
    )


class PrimaryZeroShotRendererTests(unittest.TestCase):
    def test_exposes_task_request_without_output_contract(self):
        task = task_by_subtype("simple_retrieval")

        self.assertTrue(hasattr(render_primary_zero_shot, "render_task_request"))
        request = render_primary_zero_shot.render_task_request(task)

        self.assertIn(task["targets"][0]["node"], request)
        self.assertNotIn("JSON object", request)

    def test_simple_prompt_uses_request_without_internal_labels(self):
        task = task_by_subtype("simple_retrieval")
        rendered = render_prompt(task)

        self.assertNotIn(task["level"], rendered["prompt"])
        self.assertNotIn(task["subtype"], rendered["prompt"])
        self.assertNotIn(task["id"], rendered["prompt"])
        self.assertIn(str(task["targets"][0]["shot"]), rendered["user_prompt"])
        self.assertIn(task["targets"][0]["node"], rendered["user_prompt"])
        self.assertIn(task["targets"][0]["subtree"], rendered["user_prompt"])

    def test_subtree_discovery_does_not_disclose_hidden_subtree(self):
        task = task_by_subtype("subtree_discovery")
        rendered = render_prompt(task)

        self.assertNotIn(task["targets"][0]["subtree"], rendered["prompt"])
        self.assertIn(task["targets"][0]["signal_name"], rendered["user_prompt"])
        self.assertIn("subtree", rendered["user_prompt"].lower())

    def test_exact_window_discloses_physical_request_not_indices(self):
        task = task_by_subtype("exact_window")
        rendered = render_prompt(task)
        operation = task["operation"]

        self.assertIn("{:.12g}".format(operation["time_start"]), rendered["prompt"])
        self.assertIn("{:.12g}".format(operation["time_end"]), rendered["prompt"])
        self.assertIn(str(operation["expected_points"]), rendered["prompt"])
        self.assertNotIn(str(operation["index_start"]), rendered["prompt"])
        self.assertNotIn(str(operation["index_end"]), rendered["prompt"])

    def test_sampling_request_discloses_count_not_reference_indices(self):
        task = task_by_subtype("sampling_constrained")
        rendered = render_prompt(task)
        operation = task["operation"]

        self.assertIn(str(operation["expected_points"]), rendered["prompt"])
        self.assertIn("native", rendered["prompt"].lower())
        self.assertNotIn(str(operation["index_start"]), rendered["prompt"])
        self.assertNotIn(str(operation["index_end"]), rendered["prompt"])

    def test_pair_prompts_name_targets_and_outputs_but_hide_subtrees(self):
        for subtype in ("same_shot_alignment", "cross_shot_comparison"):
            task = task_by_subtype(subtype)
            rendered = render_prompt(task)
            for target in task["targets"]:
                self.assertIn(target["signal_name"], rendered["user_prompt"])
                self.assertNotIn(target["subtree"], rendered["prompt"])
            for window in task["operation"]["per_target_windows"]:
                self.assertNotIn(str(window["index_start"]), rendered["prompt"])
                self.assertNotIn(str(window["index_end"]), rendered["prompt"])
            self.assertIn("Pearson", rendered["user_prompt"])
            self.assertIn(
                str(task["operation"]["common_points"]),
                rendered["user_prompt"],
            )

    def test_adaptive_prompt_discloses_policy_not_expected_branch(self):
        task = task_by_subtype("adaptive_access")
        rendered = render_prompt(task)

        self.assertIn("5,000,000", rendered["prompt"])
        self.assertIn("256", rendered["prompt"])
        self.assertNotIn(task["operation"]["expected_decision"], rendered["prompt"])
        self.assertNotIn(task["targets"][0]["subtree"], rendered["prompt"])

    def test_common_output_protocol_is_code_only_and_machine_readable(self):
        rendered = render_prompt(task_by_subtype("simple_retrieval"))

        self.assertIn("Python 3.6", rendered["system_prompt"])
        self.assertIn("without Markdown fences", rendered["system_prompt"])
        self.assertIn("JSON object", rendered["user_prompt"])
        self.assertIn("Do not fabricate", rendered["user_prompt"])

    def test_manifest_renders_deterministic_independent_records(self):
        with tempfile.TemporaryDirectory() as directory:
            first_path = os.path.join(directory, "first.jsonl")
            second_path = os.path.join(directory, "second.jsonl")
            first = render_manifest(MANIFEST, first_path)
            second = render_manifest(MANIFEST, second_path)

            self.assertEqual(138, len(first))
            self.assertEqual(first, second)
            self.assertEqual(138, len(set(record["task_id"] for record in first)))
            self.assertEqual(138, len(set(record["prompt"] for record in first)))
            self.assertTrue(all(
                record["prompt_version"] == "primary-zero-shot-v1"
                for record in first
            ))
            with open(first_path, "rb") as first_file:
                first_bytes = first_file.read()
            with open(second_path, "rb") as second_file:
                second_bytes = second_file.read()
            self.assertEqual(first_bytes, second_bytes)


if __name__ == "__main__":
    unittest.main()

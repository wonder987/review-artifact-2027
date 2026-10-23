import json
import os
import tempfile
import unittest

from scripts.render_gold138_prompts import render_manifest, render_prompt


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(ROOT, "benchmark", "gold138_manifest.json")


def load_manifest():
    with open(MANIFEST, "r") as handle:
        return json.load(handle)


def task_by_subtype(subtype):
    return next(
        task for task in load_manifest()["tasks"] if task["subtype"] == subtype
    )


class GoldPromptRendererTests(unittest.TestCase):
    def test_hidden_subtree_prompt_does_not_disclose_subtree(self):
        task = task_by_subtype("subtree_discovery")
        prompt = render_prompt(task)
        self.assertNotIn(task["targets"][0]["subtree"], prompt)

    def test_prompt_never_contains_oracle_fields_or_values(self):
        forbidden_fields = (
            "sha256",
            "index_start",
            "index_end",
            "allow_full_read",
            "expected_decision",
        )
        for task in load_manifest()["tasks"]:
            prompt = render_prompt(task)
            for forbidden in forbidden_fields:
                self.assertNotIn(forbidden, prompt)
            if not task["prompt_spec"]["disclose_subtree"]:
                for target in task["targets"]:
                    self.assertNotIn(target["subtree"], prompt)

    def test_shared_instructions_require_python_and_json_summary(self):
        prompt = render_prompt(task_by_subtype("simple_retrieval"))
        self.assertIn("executable Python", prompt)
        self.assertIn("JSON result summary", prompt)
        self.assertIn("do not fabricate", prompt.lower())

    def test_cross_shot_prompt_states_required_outputs(self):
        prompt = render_prompt(task_by_subtype("cross_shot_comparison"))
        self.assertIn("two shots", prompt.lower())
        self.assertIn("common time grid", prompt.lower())
        self.assertIn("pearson", prompt.lower())

    def test_same_shot_prompt_names_both_signals(self):
        task = task_by_subtype("same_shot_alignment")
        prompt = render_prompt(task)
        for target in task["targets"]:
            self.assertIn(target["signal_name"], prompt)
        self.assertIn("common time grid", prompt.lower())

    def test_adaptive_prompt_discloses_policy_not_expected_branch(self):
        task = task_by_subtype("adaptive_access")
        prompt = render_prompt(task)
        self.assertIn("5,000,000", prompt)
        self.assertIn("256 MiB", prompt)
        self.assertNotIn(task["operation"]["expected_decision"], prompt)

    def test_manifest_renders_138_unique_jsonl_records(self):
        with tempfile.TemporaryDirectory() as directory:
            output = os.path.join(directory, "prompts.jsonl")
            records = render_manifest(MANIFEST, output)
            self.assertEqual(138, len(records))
            self.assertEqual(138, len(set(item["prompt"] for item in records)))
            self.assertEqual(
                138, len(set(item["task_id"] for item in records))
            )
            with open(output, "r") as handle:
                written = [json.loads(line) for line in handle if line.strip()]
            self.assertEqual(records, written)


if __name__ == "__main__":
    unittest.main()

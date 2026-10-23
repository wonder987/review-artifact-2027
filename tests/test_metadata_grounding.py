import json
import unittest

from scripts.metadata_grounding import build_grounding, render_grounded_prompt_record


def task(disclose_subtree=False):
    return {
        "id": "gold138_l2_001",
        "level": "L2",
        "operation": {
            "expected_points": 10001,
            "index_end": 4010000,
            "index_start": 4000000,
            "kind": "index_window",
            "time_end": 0.005,
            "time_start": 0.0,
            "time_step": 0.0000005,
        },
        "prompt_spec": {
            "disclose_indices": False,
            "disclose_subtree": disclose_subtree,
        },
        "safety_policy": {
            "max_points": 5000000,
            "max_total_mib": 256.0,
            "timeout_seconds": 90,
        },
        "spec_sha256": "a" * 64,
        "split": "pilot",
        "subtype": "subtree_discovery",
        "targets": [{
            "daqtime": 12.0,
            "estimated_total_mib": 274.0,
            "frequency": 2000000.0,
            "node": "\\CHP6N",
            "sample_num": 24000000,
            "server": "202.127.204.12",
            "shot": 165000,
            "signal_name": "CHP6N",
            "size_band": "large",
            "subtree": "t6",
            "tree": "east",
            "trigtime": -2.0,
            "unit": "V",
        }],
    }


class MetadataGroundingTests(unittest.TestCase):
    def test_full_grounding_discloses_complete_metadata_and_access_plan(self):
        grounding = build_grounding(task(False), enabled=True)
        serialized = json.dumps(grounding, sort_keys=True)

        self.assertEqual("metadata-grounding-v2", grounding["grounding_version"])
        self.assertEqual("t6", grounding["targets"][0]["identity"]["subtree"])
        self.assertEqual(task(False)["operation"], grounding["operation"])
        self.assertEqual(
            r"data(\CHP6N)[4000000:4010000]",
            grounding["access_plan"][0]["signal_expression"],
        )
        self.assertEqual(
            r"dim_of(\CHP6N)[4000000:4010000]",
            grounding["access_plan"][0]["time_expression"],
        )
        self.assertEqual(10001, grounding["access_plan"][0]["expected_points"])
        self.assertIn('"index_start": 4000000', serialized)
        self.assertIn('"index_end": 4010000', serialized)
        self.assertIn("Connection.get(expression).data()", serialized)
        self.assertIn("dim_of(<same expression>)", serialized)
        self.assertIn("data(<node>)[<start_index>:<end_index>]", serialized)
        self.assertIn('"trigtime": -2.0', serialized)
        self.assertIn(r"data(\CHP6N)[4000000:4010000]", grounding["prompt_block"])

    def test_disclosed_subtree_is_preserved(self):
        grounding = build_grounding(task(True), enabled=True)

        self.assertEqual("t6", grounding["targets"][0]["identity"]["subtree"])

    def test_disabled_grounding_contains_no_metadata_block(self):
        grounding = build_grounding(task(True), enabled=False)

        self.assertFalse(grounding["enabled"])
        self.assertEqual("", grounding["prompt_block"])
        self.assertNotIn("acquisition", grounding["targets"][0])

    def test_grounded_prompt_record_is_generation_runner_compatible(self):
        original = {
            "task_id": "gold138_l2_001",
            "split": "pilot",
            "level": "L2",
            "subtype": "subtree_discovery",
            "system_prompt": "Return Python only.",
            "user_prompt": "Find and retrieve CHP6N.",
        }

        record = render_grounded_prompt_record(task(False), original, enabled=True)

        self.assertEqual("pascore-grounded-v2", record["prompt_version"])
        self.assertEqual(
            record["system_prompt"] + "\n\n" + record["user_prompt"],
            record["prompt"],
        )
        self.assertEqual(64, len(record["prompt_sha256"]))
        self.assertIn("subtree=t6", record["user_prompt"].lower())
        self.assertEqual(64, len(record["grounding_sha256"]))


if __name__ == "__main__":
    unittest.main()

import json
import unittest

from scripts.structured_feedback import build_feedback


IDENTITY = {
    "condition": "full",
    "model_alias": "qwen",
    "round": 0,
    "run_id": "run-1",
    "schema_version": "1.0",
    "task_id": "task-1",
}


class StructuredFeedbackTests(unittest.TestCase):
    def test_structured_feedback_is_actionable_and_gold_free(self):
        diagnosis = {
            "major_category": "mdsplus_interaction",
            "failure_subtype": "hallucinated_api",
            "evidence": {"error_message": "no getNode"},
            "repair_constraints": ["Use Connection.get(expression).data()."],
        }

        feedback = build_feedback(
            IDENTITY,
            diagnosis,
            remaining_rounds=2,
            structured=True,
        )
        serialized = json.dumps(feedback, sort_keys=True).lower()

        self.assertTrue(feedback["required_changes"])
        self.assertTrue(feedback["must_preserve"])
        self.assertNotIn("gold_sha256", serialized)
        self.assertNotIn("reference_index", serialized)

    def test_raw_feedback_ablation_omits_diagnostic_labels(self):
        diagnosis = {
            "major_category": "mdsplus_interaction",
            "failure_subtype": "hallucinated_api",
            "evidence": {"error_message": "no getNode"},
        }

        feedback = build_feedback(
            IDENTITY,
            diagnosis,
            remaining_rounds=2,
            structured=False,
        )

        self.assertIn("raw_error", feedback)
        self.assertNotIn("failure_category", feedback)
        self.assertNotIn("required_changes", feedback)

    def test_structured_feedback_lists_sanitized_mismatch_paths(self):
        diagnosis = {
            "major_category": "semantic_result",
            "failure_subtype": "signal_mismatch",
            "evidence": {
                "mismatch_codes": ["signal_hash_mismatch"],
                "mismatch_paths": ["targets[0].signal.sha256"],
            },
            "repair_constraints": ["Avoid dtype-changing conversions."],
        }

        feedback = build_feedback(
            IDENTITY, diagnosis, remaining_rounds=1, structured=True
        )

        self.assertIn(
            "mismatch_code: signal_hash_mismatch", feedback["evidence"]
        )
        self.assertIn(
            "mismatch_path: targets[0].signal.sha256", feedback["evidence"]
        )

    def test_structured_feedback_includes_verified_access_plan(self):
        diagnosis = {
            "major_category": "mdsplus_interaction",
            "failure_subtype": "tdi_expression_error",
            "evidence": {"error_message": "bad expression"},
            "repair_constraints": ["Use one expression argument."],
        }
        grounding = {
            "enabled": True,
            "access_plan": [{
                "target_index": 0,
                "signal_expression": r"data(\CXRS1)[700000:710000]",
                "time_expression": r"dim_of(\CXRS1)[700000:710000]",
                "expected_points": 10001,
            }],
        }

        feedback = build_feedback(
            IDENTITY,
            diagnosis,
            remaining_rounds=1,
            structured=True,
            grounding=grounding,
        )

        self.assertEqual(grounding["access_plan"], feedback["verified_access_plan"])
        self.assertIn(
            "Use the verified access plan exactly; pass each expression as the sole argument to Connection.get.",
            feedback["required_changes"],
        )


if __name__ == "__main__":
    unittest.main()

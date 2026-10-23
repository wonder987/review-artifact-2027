import unittest

from scripts.pascore_contracts import (
    ContractError,
    assert_export_safe,
    build_identity,
    sha256_json,
    validate_condition,
    validate_identity,
)


class PascoreContractTests(unittest.TestCase):
    def test_condition_and_identity_are_canonical(self):
        condition = validate_condition({
            "condition_id": "full",
            "failure_diagnosis": "multi_grained",
            "max_repair_rounds": 2,
            "metadata_grounding": True,
            "resource_guard": True,
            "schema_version": "1.0",
            "structured_feedback": True,
        })
        identity = build_identity(
            "run-1",
            "task-1",
            "qwen",
            condition,
            0,
            "a" * 64,
            "b" * 64,
            "c" * 64,
        )

        self.assertEqual("full", identity["condition"])
        self.assertEqual(64, len(identity["condition_hash"]))
        self.assertEqual(identity, validate_identity(identity))
        self.assertEqual(sha256_json(condition), identity["condition_hash"])

    def test_export_safety_rejects_gold_material(self):
        with self.assertRaises(ContractError):
            assert_export_safe({"feedback": {"gold_sha256": "secret"}})

    def test_self_debugging_condition_has_raw_feedback_contract(self):
        condition = validate_condition({
            "condition_id": "self_debugging",
            "failure_diagnosis": "raw_only",
            "max_repair_rounds": 1,
            "metadata_grounding": True,
            "resource_guard": True,
            "schema_version": "1.0",
            "structured_feedback": False,
        })
        self.assertEqual("self_debugging", condition["condition_id"])

        invalid = dict(condition)
        invalid["structured_feedback"] = True
        with self.assertRaises(ContractError):
            validate_condition(invalid)


if __name__ == "__main__":
    unittest.main()


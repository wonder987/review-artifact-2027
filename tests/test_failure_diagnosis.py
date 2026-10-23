import unittest

from scripts.failure_diagnosis import diagnose_failure


class FailureDiagnosisTests(unittest.TestCase):
    def test_diagnoses_hallucinated_remote_api(self):
        execution = {
            "status": "program_failure",
            "reported_result": {
                "error_type": "AttributeError",
                "error_message": "'Connection' object has no attribute 'getNode'",
                "failed_stage": "retrieval",
            },
        }
        evaluation = {
            "category": "execution_failure",
            "code": "api_invocation_error",
        }

        diagnosis = diagnose_failure(
            execution,
            evaluation,
            program_source="c = Connection('s')\nc.getNode('x')\n",
        )

        self.assertEqual("mdsplus_interaction", diagnosis["major_category"])
        self.assertEqual("hallucinated_api", diagnosis["failure_subtype"])
        self.assertTrue(diagnosis["repair_constraints"])
        self.assertFalse(diagnosis["gold_disclosure"])

    def test_guard_rejection_has_resource_precedence(self):
        diagnosis = diagnose_failure(
            {"status": "policy_rejected"},
            {"category": "execution_failure", "code": "policy_rejected"},
            guard={
                "decision": "require_bounded_access",
                "violations": ["full_read_exceeds_point_limit"],
            },
        )

        self.assertEqual("resource_runtime", diagnosis["major_category"])
        self.assertEqual("resource_guard_rejection", diagnosis["failure_subtype"])

    def test_protocol_failures_have_actionable_schema_constraints(self):
        execution = {"status": "execution_success", "reported_result": {}}
        for code, expected_fragment in (
            ("invalid_json", "executable Python"),
            ("missing_derived_summary", "derived.comparison"),
        ):
            diagnosis = diagnose_failure(
                execution,
                {"category": "protocol_failure", "code": code},
            )
            self.assertTrue(diagnosis["repair_constraints"])
            self.assertTrue(any(
                expected_fragment in item
                for item in diagnosis["repair_constraints"]
            ))

    def test_tdi_constraint_documents_single_expression_argument(self):
        execution = {
            "status": "program_failure",
            "reported_result": {
                "error_type": "MdsIpException",
                "error_message": "%TDI-E-EXTRA_ARG, Too many arguments",
                "failed_stage": "retrieval",
            },
        }
        diagnosis = diagnose_failure(
            execution,
            {"category": "execution_failure", "code": "runtime_error"},
        )
        self.assertTrue(any(
            "exactly one TDI expression string" in item
            for item in diagnosis["repair_constraints"]
        ))

    def test_semantic_diagnosis_includes_secondary_mismatch_constraints(self):
        execution = {"status": "execution_success", "reported_result": {}}
        evaluation = {
            "category": "semantic_failure",
            "code": "signal_dtype_mismatch",
            "mismatches": [
                {"code": "signal_dtype_mismatch", "path": "targets[0].signal.dtype"},
                {"code": "aligned_first_hash_mismatch", "path": "derived.aligned_first.sha256"},
            ],
        }

        diagnosis = diagnose_failure(execution, evaluation)

        self.assertEqual(
            ["aligned_first_hash_mismatch", "signal_dtype_mismatch"],
            diagnosis["evidence"]["mismatch_codes"],
        )
        self.assertTrue(any(
            "dtype-changing" in item for item in diagnosis["repair_constraints"]
        ))
        self.assertTrue(any(
            "common grid" in item for item in diagnosis["repair_constraints"]
        ))

    def test_resource_decision_mismatch_uses_disclosed_metadata_values(self):
        execution = {"status": "execution_success", "reported_result": {}}
        evaluation = {
            "category": "semantic_failure",
            "code": "resource_decision_mismatch",
            "mismatches": [
                {
                    "code": "resource_decision_mismatch",
                    "path": "resource_decision.total_mib",
                },
            ],
        }

        diagnosis = diagnose_failure(execution, evaluation)

        self.assertEqual(
            "resource_decision_mismatch",
            diagnosis["failure_subtype"],
        )
        constraints = "\n".join(diagnosis["repair_constraints"])
        self.assertIn("sample_num", constraints)
        self.assertIn("estimated_total_mib", constraints)
        self.assertIn("Do not recompute", constraints)
        self.assertIn("max_points", constraints)
        self.assertIn("max_total_mib", constraints)

    def test_secondary_resource_mismatch_adds_resource_constraints(self):
        execution = {"status": "execution_success", "reported_result": {}}
        evaluation = {
            "category": "semantic_failure",
            "code": "signal_size_mismatch",
            "mismatches": [
                {
                    "code": "signal_size_mismatch",
                    "path": "targets[0].signal.size",
                },
                {
                    "code": "resource_decision_mismatch",
                    "path": "resource_decision.total_mib",
                },
            ],
        }

        diagnosis = diagnose_failure(execution, evaluation)

        constraints = "\n".join(diagnosis["repair_constraints"])
        self.assertIn("estimated_total_mib", constraints)
        self.assertIn("max_total_mib", constraints)

    def test_unknown_runtime_constraint_requires_open_tree_before_reads(self):
        execution = {
            "status": "program_failure",
            "reported_result": {
                "error_type": "MdsIpException",
                "error_message": "%NONAME-W-NOMSG, Message number 0x00000000",
                "failed_stage": "retrieval",
            },
        }

        diagnosis = diagnose_failure(
            execution,
            {"category": "execution_failure", "code": "runtime_error"},
            program_source=(
                "conn = Connection('server')\n"
                "conn.get('TCL(\"set def t4\")')\n"
                "conn.get(r'\\BOLO1').data()\n"
            ),
        )

        self.assertEqual("unknown_runtime_error", diagnosis["failure_subtype"])
        constraints = "\n".join(diagnosis["repair_constraints"])
        self.assertIn("Connection.openTree(tree, shot)", constraints)
        self.assertIn("before", constraints)

    def test_pearson_mismatch_documents_zero_variance_result(self):
        execution = {"status": "execution_success", "reported_result": {}}
        evaluation = {
            "category": "semantic_failure",
            "code": "derived_statistic_mismatch",
            "mismatches": [
                {
                    "code": "derived_statistic_mismatch",
                    "path": "derived.comparison.pearson_correlation",
                },
            ],
        }

        diagnosis = diagnose_failure(execution, evaluation)

        constraints = "\n".join(diagnosis["repair_constraints"])
        self.assertIn("pearson_correlation", constraints)
        self.assertIn("None", constraints)
        self.assertIn("zero", constraints)


if __name__ == "__main__":
    unittest.main()

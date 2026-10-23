import json
import os
import tempfile
import unittest

from scripts.failure_taxonomy import (
    classify_execution_failure,
    load_taxonomy,
    validate_taxonomy,
)


def program_failure(error_type, error_message, failed_stage=None):
    reported = {
        "status": "failure",
        "error_type": error_type,
        "error_message": error_message,
    }
    if failed_stage is not None:
        reported["failed_stage"] = failed_stage
    return {
        "status": "program_failure",
        "reported_result": reported,
    }


class FailureTaxonomyValidationTests(unittest.TestCase):
    def valid_taxonomy(self):
        return {
            "schema_version": "1.0",
            "taxonomy_version": "test",
            "major_categories": [
                {"id": "a", "subtypes": ["one"]},
                {"id": "b", "subtypes": ["two"]},
            ],
        }

    def test_taxonomy_requires_major_categories(self):
        value = self.valid_taxonomy()
        del value["major_categories"]
        with self.assertRaises(ValueError):
            validate_taxonomy(value)

    def test_taxonomy_requires_unique_major_category_ids(self):
        value = self.valid_taxonomy()
        value["major_categories"][1]["id"] = "a"
        with self.assertRaises(ValueError):
            validate_taxonomy(value)

    def test_taxonomy_requires_unique_subtype_ownership(self):
        value = self.valid_taxonomy()
        value["major_categories"][1]["subtypes"] = ["one"]
        with self.assertRaises(ValueError):
            validate_taxonomy(value)

    def test_load_taxonomy_validates_json(self):
        value = self.valid_taxonomy()
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "taxonomy.json")
            with open(path, "w") as handle:
                json.dump(value, handle)
            self.assertEqual(value, load_taxonomy(path))


class FailureClassificationTests(unittest.TestCase):
    def assert_classification(self, execution, major, subtype):
        result = classify_execution_failure(execution)
        self.assertEqual(major, result["major_category"])
        self.assertEqual(subtype, result["failure_subtype"])
        self.assertIn("matched_rule", result["evidence"])
        self.assertEqual(execution.get("status"), result["evidence"]["execution_status"])

    def test_generation_timeout(self):
        self.assert_classification(
            {
                "status": "generation_failure",
                "failure_code": "generation_timeout",
                "error_type": "ClientError",
                "error_message": "generation request timed out after 600 seconds",
            },
            "code_generation",
            "generation_timeout",
        )

    def test_syntax_policy_rejection_is_not_unsafe_operation(self):
        self.assert_classification(
            {
                "status": "policy_rejected",
                "policy_violations": [
                    "syntax_error:13:f-string expression part cannot include a backslash"
                ],
            },
            "code_generation",
            "syntax_error",
        )

    def test_forbidden_import_is_unsafe_operation(self):
        self.assert_classification(
            {
                "status": "policy_rejected",
                "policy_violations": ["forbidden_import:subprocess"],
            },
            "code_generation",
            "unsafe_operation",
        )

    def test_runtime_import_failure(self):
        self.assert_classification(
            program_failure("ModuleNotFoundError", "No module named 'MDSplus'", "import"),
            "code_generation",
            "import_error",
        )

    def test_attribute_error_under_node_wrapper_is_hallucinated_api(self):
        self.assert_classification(
            program_failure(
                "Exception",
                "Could not retrieve node: 'Connection' object has no attribute 'getNode'",
            ),
            "mdsplus_interaction",
            "hallucinated_api",
        )

    def test_module_attribute_error_is_hallucinated_api(self):
        self.assert_classification(
            program_failure(
                "AttributeError", "module 'MDSplus' has no attribute 'mdsopen'"
            ),
            "mdsplus_interaction",
            "hallucinated_api",
        )

    def test_wrong_argument_order_is_wrong_api_signature(self):
        self.assert_classification(
            program_failure("AttributeError", "'int' object has no attribute 'upper'"),
            "mdsplus_interaction",
            "wrong_api_signature",
        )

    def test_missing_argument_is_wrong_api_signature(self):
        self.assert_classification(
            program_failure(
                "TypeError", "closeTree() missing 1 required positional argument: 'shot'"
            ),
            "mdsplus_interaction",
            "wrong_api_signature",
        )

    def test_local_tree_api_is_local_remote_confusion(self):
        self.assert_classification(
            program_failure(
                "AttributeError", "remote task attempted MDSplus.Tree(...).getNode()"
            ),
            "mdsplus_interaction",
            "local_remote_api_confusion",
        )

    def test_connection_tree_attribute_is_local_remote_confusion(self):
        self.assert_classification(
            program_failure(
                "AttributeError", "'Connection' object has no attribute 'Tree'"
            ),
            "mdsplus_interaction",
            "local_remote_api_confusion",
        )

    def test_tree_nopath_is_node_lookup_error(self):
        for message in (
            "%TREE-E-NOPATH, No path found",
            "%TREE-W-NNF, Node Not Found",
            "TreeNOPATH: node lookup failed",
        ):
            self.assert_classification(
                program_failure("MdsIpException", message),
                "mdsplus_interaction",
                "node_lookup_error",
            )

    def test_tdi_status_is_tdi_expression_error(self):
        for message in (
            "%TDI-E-INVDTYDSC, Storage data type is not valid",
            "%TDI-E-SYNTAX, bad expression",
        ):
            self.assert_classification(
                program_failure("MdsIpException", message),
                "mdsplus_interaction",
                "tdi_expression_error",
            )

    def test_time_stage_failure_is_timebase_retrieval_error(self):
        self.assert_classification(
            program_failure("ValueError", "could not retrieve dimension", "get_time_axis"),
            "mdsplus_interaction",
            "timebase_retrieval_error",
        )

    def test_unqualified_time_node_source_is_timebase_retrieval_error(self):
        execution = program_failure(
            "MdsIpException", "%TREE-W-NNF, Node Not Found", "unknown"
        )
        execution["diagnostic_program_source"] = (
            "c = Connection('s')\n"
            "signal = c.get(r'\\IP').data()\n"
            "time_axis = c.get('T').data()\n"
        )
        self.assert_classification(
            execution,
            "mdsplus_interaction",
            "timebase_retrieval_error",
        )

    def test_memory_error(self):
        self.assert_classification(
            program_failure("MemoryError", "Unable to allocate array"),
            "resource_runtime",
            "memory_exhaustion",
        )

    def test_virtual_memory_message_in_diagnostic_stdout_is_memory_exhaustion(self):
        execution = program_failure(
            "RuntimeError",
            "%MDSPLUS-E-Error, Failure to complete operation",
            "processing",
        )
        execution["diagnostic_stdout"] = (
            '{"status":"failure"}\nInsufficient virtual memory\n'
        )
        self.assert_classification(
            execution,
            "resource_runtime",
            "memory_exhaustion",
        )

    def test_remote_connection_getnode_source_is_hallucinated_api(self):
        execution = program_failure(
            "RuntimeError",
            "Could not locate node \\CDG1 in any subtree",
            "retrieval",
        )
        execution["diagnostic_program_source"] = """
conn = MDSplus.Connection(SERVER)
conn.openTree(TREE, SHOT)
node = conn.getNode(candidate)
"""
        result = classify_execution_failure(execution)
        self.assertEqual("mdsplus_interaction", result["major_category"])
        self.assertEqual("hallucinated_api", result["failure_subtype"])
        self.assertEqual(
            "remote_connection_getNode",
            result["evidence"]["source_signature"],
        )

    def test_explicit_timebase_message_is_timebase_retrieval_error(self):
        self.assert_classification(
            program_failure(
                "RuntimeError",
                "Could not determine time base",
                "data_retrieval",
            ),
            "mdsplus_interaction",
            "timebase_retrieval_error",
        )

    def test_generic_locate_node_message_is_node_lookup_error(self):
        self.assert_classification(
            program_failure(
                "RuntimeError",
                "Could not locate node \\CDG1 in any subtree",
                "retrieval",
            ),
            "mdsplus_interaction",
            "node_lookup_error",
        )

    def test_execution_timeout(self):
        self.assert_classification(
            {"status": "timeout", "timed_out": True},
            "resource_runtime",
            "execution_timeout",
        )

    def test_ndarray_processing_error(self):
        self.assert_classification(
            program_failure("AttributeError", "'numpy.ndarray' object has no attribute 'dim'"),
            "resource_runtime",
            "array_processing_error",
        )

    def test_ndarray_error_outranks_generic_time_stage(self):
        self.assert_classification(
            program_failure(
                "AttributeError",
                "'numpy.ndarray' object has no attribute 'dim'",
                "get_time_axis",
            ),
            "resource_runtime",
            "array_processing_error",
        )

    def test_json_boolean_key_error(self):
        self.assert_classification(
            program_failure("TypeError", "keys must be str, int, float, bool or None, not tuple"),
            "resource_runtime",
            "serialization_error",
        )

    def test_bool_class_key_error_is_serialization_error(self):
        self.assert_classification(
            program_failure(
                "TypeError", "key <class 'bool'> is not a string"
            ),
            "resource_runtime",
            "serialization_error",
        )

    def test_unknown_runtime_error_preserves_source_evidence(self):
        execution = program_failure("RuntimeError", "unclassified failure")
        result = classify_execution_failure(execution)
        self.assertEqual("resource_runtime", result["major_category"])
        self.assertEqual("unknown_runtime_error", result["failure_subtype"])
        self.assertEqual("RuntimeError", result["evidence"]["error_type"])
        self.assertEqual("unclassified failure", result["evidence"]["error_message"])


if __name__ == "__main__":
    unittest.main()

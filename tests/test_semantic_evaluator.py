import copy
import json
import os
import unittest

from scripts.semantic_evaluator import evaluate_record


def summary(label="a", dtype="float32", size=3):
    return {
        "size": size,
        "shape": [size],
        "dtype": dtype,
        "sha256": label * 64,
        "finite_count": size,
        "min": 1.0,
        "max": 3.0,
        "mean": 2.0,
        "std": 0.5,
    }


def identity(node="\\IP", shot=1, subtree="t1"):
    return {
        "server": "202.127.204.12",
        "tree": "east",
        "shot": shot,
        "node": node,
        "subtree": subtree,
    }


def target(node="\\IP", shot=1, signal_hash="a", time_hash="b"):
    return {
        "target": identity(node=node, shot=shot),
        "signal": summary(signal_hash),
        "time_axis": summary(time_hash, dtype="float64"),
        "validation": {
            "non_empty": True,
            "same_size": True,
            "signal_all_finite": True,
            "time_all_finite": True,
            "time_strictly_increasing": True,
        },
    }


def gold(task_id="task", subtype="simple_retrieval"):
    return {
        "task_id": task_id,
        "status": "success",
        "subtype": subtype,
        "targets": [target()],
        "resource_decisions": [{
            "allow_full_read": True,
            "within_point_limit": True,
            "within_memory_limit": True,
            "points": 3,
            "total_mib": 0.1,
            "max_points": 5000000,
            "max_total_mib": 256.0,
            "recommended_action": "read_full_signal",
        }],
    }


def execution(reported=None, status="execution_success"):
    return {
        "model_alias": "model",
        "task_id": "task",
        "status": status,
        "exit_status": 0,
        "stdout_json_valid": reported is not None,
        "reported_result": reported,
        "timed_out": status == "timeout",
    }


def success_report(targets=None):
    return {
        "status": "SUCCESS",
        "accessed_objects": [],
        "elapsed_seconds": 1.0,
        "targets": targets if targets is not None else [target()],
    }


def derived():
    return {
        "common_time_axis": summary("c", dtype="float64"),
        "aligned_first": summary("d", dtype="float64"),
        "aligned_second": summary("e", dtype="float64"),
        "comparison": {
            "count": 3,
            "a_mean": 1.0,
            "a_std": 0.2,
            "b_mean": 2.0,
            "b_std": 0.3,
            "pearson_correlation": 0.5,
        },
    }


class SemanticEvaluatorTests(unittest.TestCase):
    def test_strict_mode_retains_exact_result_shape(self):
        result = evaluate_record(execution(success_report()), gold())
        self.assertEqual({
            "evaluator_version": "semantic-evaluator-v2",
            "model_alias": "model",
            "task_id": "task",
            "category": "success",
            "code": "semantic_success",
            "execution_success": True,
            "semantic_success": True,
            "mismatches": [],
            "diagnostics": [],
        }, result)

    def test_strict_mode_does_not_normalize_identity_sibling(self):
        report = success_report()
        report["targets"][0]["identity"] = report["targets"][0]["target"]
        report["targets"][0]["target"] = "\\IP"
        result = evaluate_record(execution(report), gold())
        self.assertEqual("protocol_failure", result["category"])
        self.assertEqual("missing_target_summary", result["code"])

    def test_diagnostic_mode_normalizes_identity_without_mutating_input(self):
        report = success_report()
        report["targets"][0]["identity"] = report["targets"][0]["target"]
        report["targets"][0]["target"] = "\\IP"
        before = copy.deepcopy(report)
        result = evaluate_record(
            execution(report), gold(), evaluation_mode="diagnostic-v2"
        )
        self.assertEqual(before, report)
        self.assertEqual([{
            "path": "targets[0].target",
            "normalization": "identity_to_target",
        }], result["normalizations"])
        self.assertTrue(result["semantic_success"])

    def test_diagnostic_mode_rejects_incomplete_identity(self):
        report = success_report()
        report["targets"][0]["identity"] = {"tree": "east", "shot": 1}
        report["targets"][0]["target"] = "\\IP"
        result = evaluate_record(
            execution(report), gold(), evaluation_mode="diagnostic-v2"
        )
        self.assertEqual("missing_target_summary", result["code"])
        self.assertEqual([], result["normalizations"])

    def test_diagnostic_mode_rejects_identity_with_empty_required_value(self):
        report = success_report()
        report["targets"][0]["identity"] = identity()
        report["targets"][0]["identity"]["node"] = None
        report["targets"][0]["target"] = "\\IP"
        result = evaluate_record(
            execution(report), gold(), evaluation_mode="diagnostic-v2"
        )
        self.assertEqual("missing_target_summary", result["code"])
        self.assertEqual([], result["normalizations"])

    def test_diagnostic_mode_does_not_overwrite_valid_target(self):
        report = success_report()
        report["targets"][0]["identity"] = identity(node="\\WRONG")
        result = evaluate_record(
            execution(report), gold(), evaluation_mode="diagnostic-v2"
        )
        self.assertTrue(result["semantic_success"])
        self.assertEqual([], result["normalizations"])

    def test_diagnostic_mode_uses_only_v2_visible_validation_fields(self):
        expected = gold()
        expected["targets"][0]["validation"][
            "time_monotonic_non_decreasing"
        ] = True
        report = success_report()
        strict = evaluate_record(execution(report), expected)
        diagnostic = evaluate_record(
            execution(report), expected, evaluation_mode="diagnostic-v2"
        )
        self.assertEqual("missing_validation", strict["code"])
        self.assertTrue(diagnostic["semantic_success"])

    def test_diagnostic_mode_does_not_repair_wrong_visible_boolean(self):
        report = success_report()
        report["targets"][0]["validation"]["time_strictly_increasing"] = False
        result = evaluate_record(
            execution(report), gold(), evaluation_mode="diagnostic-v2"
        )
        self.assertEqual("semantic_failure", result["category"])
        self.assertEqual("validation_mismatch", result["code"])

    def test_unknown_evaluation_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_record(
                execution(success_report()), gold(), evaluation_mode="lenient"
            )

    def test_missing_gold_is_infrastructure_failure(self):
        result = evaluate_record(execution(success_report()), None)
        self.assertEqual("infrastructure_failure", result["category"])
        self.assertEqual("missing_gold", result["code"])

    def test_executor_failures_are_classified_before_protocol(self):
        cases = (
            ("timeout", "timeout"),
            ("policy_rejected", "policy_rejected"),
            ("program_failure", "runtime_error"),
        )
        for status, code in cases:
            run = execution(None, status=status)
            if status == "program_failure":
                run["exit_status"] = 1
            result = evaluate_record(run, gold())
            self.assertEqual("execution_failure", result["category"])
            self.assertEqual(code, result["code"])

    def test_generation_timeout_is_model_evaluable_generation_failure(self):
        run = execution(None, status="generation_failure")
        run.update({
            "failure_code": "generation_timeout",
            "error_type": "ClientError",
            "error_message": "generation request timed out after 600 seconds",
        })

        result = evaluate_record(run, gold())

        self.assertEqual("generation_failure", result["category"])
        self.assertEqual("generation_timeout", result["code"])
        self.assertEqual("ClientError", result["error_type"])
        self.assertIn("600 seconds", result["error_message"])

    def test_program_failure_uses_reported_error_taxonomy(self):
        cases = (
            ("SyntaxError", "invalid syntax", "syntax_error"),
            ("ModuleNotFoundError", "No module named x", "import_error"),
            ("AttributeError", "Connection has no attribute getNode", "api_invocation_error"),
            (
                "RuntimeError",
                "Failed to retrieve signal: 'Connection' object has no attribute 'getNode'",
                "api_invocation_error",
            ),
            ("AttributeError", "numpy.ndarray has no attribute dim", "runtime_error"),
            ("MdsIpException", "Node not found", "node_lookup_error"),
            ("ValueError", "bad array", "runtime_error"),
        )
        for error_type, message, code in cases:
            reported = {
                "status": "failure",
                "error_type": error_type,
                "error_message": message,
            }
            run = execution(reported, status="program_failure")
            result = evaluate_record(run, gold())
            self.assertEqual(code, result["code"])
            self.assertEqual(error_type, result["error_type"])
            self.assertEqual(message, result["error_message"])

    def test_invalid_json_and_self_reported_failure_are_distinct(self):
        invalid = evaluate_record(execution(None), gold())
        self.assertEqual("protocol_failure", invalid["category"])
        self.assertEqual("invalid_json", invalid["code"])

        failed = success_report()
        failed["status"] = "failure"
        result = evaluate_record(execution(failed), gold())
        self.assertEqual("execution_failure", result["category"])
        self.assertEqual("self_reported_failure", result["code"])

        real_executor_shape = execution(None, status="program_failure")
        real_executor_shape["exit_status"] = 0
        result = evaluate_record(real_executor_shape, gold())
        self.assertEqual("protocol_failure", result["category"])
        self.assertEqual("invalid_json", result["code"])

    def test_exact_target_is_semantic_success(self):
        result = evaluate_record(execution(success_report()), gold())
        self.assertEqual("success", result["category"])
        self.assertEqual("semantic_success", result["code"])
        self.assertTrue(result["execution_success"])
        self.assertTrue(result["semantic_success"])
        self.assertEqual([], result["mismatches"])

    def test_missing_target_summary_is_protocol_failure(self):
        report = success_report()
        del report["targets"][0]["signal"]
        result = evaluate_record(execution(report), gold())
        self.assertEqual("protocol_failure", result["category"])
        self.assertEqual("missing_target_summary", result["code"])

    def test_wrong_array_hash_and_dtype_are_semantic_failures(self):
        report = success_report()
        report["targets"][0]["signal"]["sha256"] = "c" * 64
        result = evaluate_record(execution(report), gold())
        self.assertEqual("signal_hash_mismatch", result["code"])

        report = success_report()
        report["targets"][0]["signal"]["dtype"] = "float64"
        result = evaluate_record(execution(report), gold())
        self.assertEqual("signal_dtype_mismatch", result["code"])

        report = success_report()
        report["targets"][0]["time_axis"]["sha256"] = "c" * 64
        result = evaluate_record(execution(report), gold())
        self.assertEqual("time_axis_hash_mismatch", result["code"])

    def test_matching_hash_makes_array_statistics_diagnostic_only(self):
        report = success_report()
        report["targets"][0]["signal"]["mean"] = 2.000001

        result = evaluate_record(execution(report), gold())

        self.assertTrue(result["semantic_success"])
        self.assertEqual("signal_statistic_difference", result["diagnostics"][0]["code"])

    def test_missing_validation_field_is_protocol_failure(self):
        report = success_report()
        del report["targets"][0]["validation"]["same_size"]

        result = evaluate_record(execution(report), gold())

        self.assertEqual("protocol_failure", result["category"])
        self.assertEqual("missing_validation", result["code"])

    def test_target_order_and_identity_are_exact(self):
        expected = gold(subtype="cross_shot_comparison")
        expected["targets"] = [
            target(node="\\BOLO1", shot=1, signal_hash="a", time_hash="b"),
            target(node="\\BOLO1", shot=2, signal_hash="c", time_hash="d"),
        ]
        report = success_report(targets=copy.deepcopy(expected["targets"]))
        report["targets"].reverse()
        expected["derived"] = derived()
        report["derived"] = copy.deepcopy(expected["derived"])
        result = evaluate_record(execution(report), expected)
        self.assertEqual("semantic_failure", result["category"])
        self.assertEqual("target_identity_mismatch", result["code"])

    def test_qualified_node_is_equivalent_to_node_plus_subtree(self):
        report = success_report()
        report["targets"][0]["target"]["node"] = "\\t1::IP"

        result = evaluate_record(execution(report), gold())

        self.assertTrue(result["semantic_success"])

    def test_hidden_subtree_is_not_used_for_identity_scoring(self):
        expected = gold()
        report = success_report()
        report["targets"][0]["target"]["subtree"] = None
        expected_task = {
            "id": "task",
            "subtype": "simple_retrieval",
            "prompt_spec": {"disclose_subtree": False},
        }

        result = evaluate_record(
            execution(report), expected, expected_task=expected_task
        )

        self.assertTrue(result["semantic_success"])

    def test_pair_derived_exact_and_wrong_correlation(self):
        expected = gold(subtype="cross_shot_comparison")
        expected["targets"] = [target(), target(node="\\IP", shot=2)]
        expected["derived"] = derived()
        report = success_report(targets=copy.deepcopy(expected["targets"]))
        report["derived"] = copy.deepcopy(expected["derived"])
        self.assertTrue(
            evaluate_record(execution(report), expected)["semantic_success"]
        )

        report["derived"]["comparison"]["pearson_correlation"] = 0.6
        result = evaluate_record(execution(report), expected)
        self.assertEqual("derived_statistic_mismatch", result["code"])

    def test_pair_accepts_legacy_top_level_comparison(self):
        expected = gold(subtype="cross_shot_comparison")
        expected["targets"] = [target(), target(node="\\BOLO1", shot=2)]
        expected["derived"] = derived()
        report = success_report(targets=copy.deepcopy(expected["targets"]))
        report["derived"] = copy.deepcopy(expected["derived"])
        report["comparison"] = report["derived"].pop("comparison")

        result = evaluate_record(execution(report), expected)

        self.assertTrue(result["semantic_success"])

    def test_pair_ignores_unrequested_gold_comparison_metrics(self):
        expected = gold(subtype="cross_shot_comparison")
        expected["targets"] = [target(), target(node="\\BOLO1", shot=2)]
        expected["derived"] = derived()
        expected["derived"]["comparison"]["mean_absolute_difference"] = 9.0
        expected["derived"]["comparison"]["root_mean_square_difference"] = 10.0
        report = success_report(targets=copy.deepcopy(expected["targets"]))
        report["derived"] = derived()

        result = evaluate_record(execution(report), expected)

        self.assertTrue(result["semantic_success"])

    def test_adaptive_decision_must_match(self):
        expected = gold(subtype="adaptive_access")
        report = success_report()
        report["resource_decision"] = copy.deepcopy(
            expected["resource_decisions"][0]
        )
        self.assertTrue(
            evaluate_record(execution(report), expected)["semantic_success"]
        )

        report["resource_decision"]["allow_full_read"] = False
        result = evaluate_record(execution(report), expected)
        self.assertEqual("resource_decision_mismatch", result["code"])

    def test_recorded_smoke_regression_labels(self):
        fixture_path = os.path.join(
            os.path.dirname(__file__),
            "fixtures",
            "semantic_evaluation_smoke",
            "manifest.json",
        )
        with open(fixture_path, "r") as handle:
            cases = json.load(handle)["cases"]

        observed = {}
        for case in cases:
            expected = gold(task_id=case["task_id"])
            expected["subtype"] = case.get("subtype", "simple_retrieval")
            if case["kind"] == "execution_failure":
                error_message = (
                    "numpy.ndarray has no attribute dim"
                    if case["expected_code"] == "runtime_error"
                    else "MDSplus API method is unavailable"
                )
                run = execution({
                    "status": "failure",
                    "error_type": "AttributeError",
                    "error_message": error_message,
                }, status="program_failure")
                run["model_alias"] = case["model_alias"]
                run["task_id"] = case["task_id"]
            else:
                report = success_report()
                if case["kind"] == "signal_hash_mismatch":
                    report["targets"][0]["signal"]["sha256"] = "f" * 64
                elif case["kind"] == "time_axis_hash_mismatch":
                    report["targets"][0]["time_axis"]["sha256"] = "f" * 64
                elif case["kind"] == "pair_success":
                    expected["targets"] = [target(), target(shot=2)]
                    expected["derived"] = derived()
                    report["targets"] = copy.deepcopy(expected["targets"])
                    report["derived"] = copy.deepcopy(expected["derived"])
                run = execution(report)
                run["model_alias"] = case["model_alias"]
                run["task_id"] = case["task_id"]
            result = evaluate_record(run, expected)
            observed[(case["model_alias"], case["task_id"])] = (
                result["category"], result["code"]
            )

        wanted = dict(
            ((case["model_alias"], case["task_id"]),
             (case["expected_category"], case["expected_code"]))
            for case in cases
        )
        self.assertEqual(wanted, observed)


if __name__ == "__main__":
    unittest.main()

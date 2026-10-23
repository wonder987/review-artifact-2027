#!/usr/bin/env python3
"""Deterministic semantic comparison of generated execution and Gold data."""

from __future__ import print_function

import copy
import math


EVALUATOR_VERSION = "semantic-evaluator-v2"
REL_TOL = 1e-9
ABS_TOL = 1e-12

ARRAY_FIELDS = (
    "size", "shape", "dtype", "sha256", "finite_count",
    "min", "max", "mean", "std",
)
STRUCTURAL_ARRAY_FIELDS = ("size", "shape", "dtype", "sha256", "finite_count")
VALIDATION_FIELDS = (
    "non_empty",
    "same_size",
    "signal_all_finite",
    "time_all_finite",
    "time_monotonic_non_decreasing",
    "time_strictly_increasing",
)
IDENTITY_FIELDS = ("server", "tree", "shot", "node", "subtree")
DECISION_FIELDS = (
    "allow_full_read",
    "within_point_limit",
    "within_memory_limit",
    "points",
    "total_mib",
    "max_points",
    "max_total_mib",
    "recommended_action",
)
COMPARISON_FIELDS = (
    "count", "a_mean", "a_std", "b_mean", "b_std", "pearson_correlation",
)


def _base(execution):
    execution = execution or {}
    return {
        "evaluator_version": EVALUATOR_VERSION,
        "model_alias": execution.get("model_alias"),
        "task_id": execution.get("task_id"),
        "category": None,
        "code": None,
        "execution_success": False,
        "semantic_success": False,
        "mismatches": [],
        "diagnostics": [],
    }


def _finish(result, category, code):
    result["category"] = category
    result["code"] = code
    result["execution_success"] = category in ("success", "semantic_failure")
    result["semantic_success"] = category == "success"
    return result


def _mismatch(path, code, expected, observed):
    return {
        "path": path,
        "code": code,
        "expected": expected,
        "observed": observed,
    }


def _scalar_equal(observed, expected):
    if observed is None or expected is None:
        return observed is expected
    if isinstance(observed, bool) or isinstance(expected, bool):
        return type(observed) is bool and type(expected) is bool and observed == expected
    if isinstance(observed, (int, float)) and isinstance(expected, (int, float)):
        return math.isclose(
            float(observed),
            float(expected),
            rel_tol=REL_TOL,
            abs_tol=ABS_TOL,
        )
    return observed == expected


def _array_code(array_name, field):
    prefix = "time_axis" if array_name == "time_axis" else array_name
    names = {
        "size": "{}_size_mismatch".format(prefix),
        "shape": "{}_shape_mismatch".format(prefix),
        "dtype": "{}_dtype_mismatch".format(prefix),
        "sha256": "{}_hash_mismatch".format(prefix),
        "finite_count": "{}_finite_count_mismatch".format(prefix),
    }
    return names.get(field, "{}_statistic_mismatch".format(prefix))


def _compare_array(observed, expected, path, array_name):
    mismatches = []
    for field in STRUCTURAL_ARRAY_FIELDS:
        if not _scalar_equal(observed.get(field), expected.get(field)):
            mismatches.append(_mismatch(
                path + "." + field,
                _array_code(array_name, field),
                expected.get(field),
                observed.get(field),
            ))
    return mismatches


def _array_diagnostics(observed, expected, path, array_name):
    diagnostics = []
    for field in ("min", "max", "mean", "std"):
        if field in expected and not _scalar_equal(
            observed.get(field), expected.get(field)
        ):
            diagnostics.append(_mismatch(
                path + "." + field,
                "{}_statistic_difference".format(array_name),
                expected.get(field),
                observed.get(field),
            ))
    return diagnostics


def _required_target_shape(target):
    if not isinstance(target, dict):
        return False
    if not isinstance(target.get("target"), dict):
        return False
    if not isinstance(target.get("validation"), dict):
        return False
    for name in ("signal", "time_axis"):
        value = target.get(name)
        if not isinstance(value, dict):
            return False
        if any(field not in value for field in ARRAY_FIELDS):
            return False
    return True


def _compare_identity(observed, expected, path, compare_subtree=True):
    mismatches = []
    for field in ("server", "tree", "shot"):
        if field in expected and observed.get(field) != expected.get(field):
            mismatches.append(_mismatch(
                path + "." + field,
                "target_identity_mismatch",
                expected.get(field),
                observed.get(field),
            ))
    observed_node = _canonical_node(observed)
    expected_node = _canonical_node(expected)
    if not compare_subtree:
        observed_node = observed_node[1]
        expected_node = expected_node[1]
    if observed_node != expected_node:
        mismatches.append(_mismatch(
            path + ".node",
            "target_identity_mismatch",
            expected.get("node"),
            observed.get("node"),
        ))
    return mismatches


def _canonical_node(identity):
    node = str(identity.get("node", "")).strip().lstrip("\\")
    subtree = str(identity.get("subtree", "")).strip().lower()
    if "::" in node:
        qualified_subtree, node = node.split("::", 1)
        subtree = qualified_subtree.lower()
    return subtree, node.upper()


def _program_failure_code(execution):
    reported = execution.get("reported_result")
    if not isinstance(reported, dict):
        return "runtime_error"
    error_type = str(reported.get("error_type", "")).lower()
    message = str(reported.get("error_message", "")).lower()
    if "syntaxerror" in error_type:
        return "syntax_error"
    if error_type in ("importerror", "modulenotfounderror"):
        return "import_error"
    if "node" in message and (
        "mdsip" in error_type or "tree" in message or "not found" in message
    ):
        return "node_lookup_error"
    if (
        "mdsplus" in message or "connection" in message
    ) and (
        "attributeerror" in error_type or "has no attribute" in message
    ):
        return "api_invocation_error"
    return "runtime_error"


def _compare_validation(observed, expected, path, validation_fields=None):
    if validation_fields is None:
        validation_fields = VALIDATION_FIELDS
    mismatches = []
    for field in validation_fields:
        if field not in expected:
            continue
        if type(observed.get(field)) is not bool or observed.get(field) != expected[field]:
            mismatches.append(_mismatch(
                path + "." + field,
                "validation_mismatch",
                expected[field],
                observed.get(field),
            ))
    return mismatches


def _compare_target(
    observed,
    expected,
    index,
    validation_fields=None,
    compare_subtree=True,
):
    prefix = "targets[{}]".format(index)
    mismatches = _compare_identity(
        observed["target"], expected.get("target", {}), prefix + ".target",
        compare_subtree=compare_subtree,
    )
    mismatches.extend(_compare_array(
        observed["signal"], expected.get("signal", {}),
        prefix + ".signal", "signal",
    ))
    mismatches.extend(_compare_array(
        observed["time_axis"], expected.get("time_axis", {}),
        prefix + ".time_axis", "time_axis",
    ))
    mismatches.extend(_compare_validation(
        observed["validation"], expected.get("validation", {}),
        prefix + ".validation", validation_fields,
    ))
    return mismatches


def _derived_shape_valid(derived):
    if not isinstance(derived, dict) or not isinstance(
        derived.get("comparison"), dict
    ):
        return False
    for name in ("common_time_axis", "aligned_first", "aligned_second"):
        value = derived.get(name)
        if not isinstance(value, dict):
            return False
        if any(field not in value for field in ARRAY_FIELDS):
            return False
    return True


def _compare_derived(observed, expected):
    mismatches = []
    for name in ("common_time_axis", "aligned_first", "aligned_second"):
        mismatches.extend(_compare_array(
            observed[name], expected.get(name, {}),
            "derived." + name, name,
        ))
    observed_comparison = observed["comparison"]
    expected_comparison = expected.get("comparison", {})
    for field in COMPARISON_FIELDS:
        if field not in expected_comparison:
            continue
        expected_value = expected_comparison[field]
        observed_value = observed_comparison.get(field)
        if not _scalar_equal(observed_value, expected_value):
            mismatches.append(_mismatch(
                "derived.comparison." + field,
                "derived_statistic_mismatch",
                expected_value,
                observed_value,
            ))
    return mismatches


def _compare_decision(observed, expected):
    mismatches = []
    for field in DECISION_FIELDS:
        if field not in expected:
            continue
        if not _scalar_equal(observed.get(field), expected[field]):
            mismatches.append(_mismatch(
                "resource_decision." + field,
                "resource_decision_mismatch",
                expected[field],
                observed.get(field),
            ))
    return mismatches


def _normalize_v2_report(reported):
    normalized = copy.deepcopy(reported)
    normalizations = []
    targets = normalized.get("targets")
    if not isinstance(targets, list):
        return normalized, normalizations
    required_identity_fields = ("server", "tree", "shot", "node")
    for index, item in enumerate(targets):
        if not isinstance(item, dict) or isinstance(item.get("target"), dict):
            continue
        identity = item.get("identity")
        if not isinstance(identity, dict):
            continue
        if any(field not in identity for field in required_identity_fields):
            continue
        if any(
            not isinstance(identity.get(field), str) or not identity.get(field).strip()
            for field in ("server", "tree", "node")
        ):
            continue
        if (
            isinstance(identity.get("shot"), bool)
            or not isinstance(identity.get("shot"), (int, float))
        ):
            continue
        item["target"] = copy.deepcopy(identity)
        normalizations.append({
            "path": "targets[{}].target".format(index),
            "normalization": "identity_to_target",
        })
    return normalized, normalizations


def evaluate_record(execution, gold, expected_task=None, evaluation_mode="strict"):
    if evaluation_mode not in ("strict", "diagnostic-v2"):
        raise ValueError("unsupported evaluation mode: {}".format(evaluation_mode))
    result = _base(execution)
    if evaluation_mode == "diagnostic-v2":
        result["normalizations"] = []
    if not isinstance(gold, dict):
        return _finish(result, "infrastructure_failure", "missing_gold")
    if not isinstance(execution, dict):
        return _finish(result, "infrastructure_failure", "missing_execution")
    if execution.get("task_id") != gold.get("task_id"):
        return _finish(result, "infrastructure_failure", "task_binding_mismatch")

    execution_status = execution.get("status")
    if execution_status == "generation_failure":
        result["error_type"] = execution.get("error_type")
        result["error_message"] = execution.get("error_message")
        return _finish(
            result,
            "generation_failure",
            execution.get("failure_code") or "unknown_generation_failure",
        )
    if execution_status in ("timeout", "policy_rejected"):
        return _finish(result, "execution_failure", execution_status)
    if execution_status == "program_failure":
        if (
            execution.get("exit_status") == 0
            and execution.get("stdout_json_valid") is not True
        ):
            return _finish(result, "protocol_failure", "invalid_json")
        reported_error = execution.get("reported_result")
        if isinstance(reported_error, dict):
            result["error_type"] = reported_error.get("error_type")
            result["error_message"] = reported_error.get("error_message")
        return _finish(
            result, "execution_failure", _program_failure_code(execution)
        )
    if execution_status != "execution_success":
        return _finish(result, "execution_failure", "unknown_execution_status")

    reported = execution.get("reported_result")
    if not execution.get("stdout_json_valid") or not isinstance(reported, dict):
        return _finish(result, "protocol_failure", "invalid_json")
    if str(reported.get("status", "")).strip().lower() != "success":
        return _finish(result, "execution_failure", "self_reported_failure")

    if evaluation_mode == "diagnostic-v2":
        reported, normalizations = _normalize_v2_report(reported)
        result["normalizations"] = normalizations
        validation_fields = tuple(
            field for field in VALIDATION_FIELDS
            if field != "time_monotonic_non_decreasing"
        )
    else:
        validation_fields = VALIDATION_FIELDS

    expected_targets = gold.get("targets")
    observed_targets = reported.get("targets")
    if not isinstance(expected_targets, list):
        return _finish(result, "infrastructure_failure", "invalid_gold")
    if not isinstance(observed_targets, list):
        return _finish(result, "protocol_failure", "missing_target_summary")
    if len(observed_targets) != len(expected_targets):
        result["mismatches"].append(_mismatch(
            "targets", "target_count_mismatch",
            len(expected_targets), len(observed_targets),
        ))
        return _finish(result, "protocol_failure", "target_count_mismatch")
    if not all(_required_target_shape(item) for item in observed_targets):
        return _finish(result, "protocol_failure", "missing_target_summary")

    for observed, expected in zip(observed_targets, expected_targets):
        if any(
            field not in observed["target"]
            for field in ("server", "tree", "shot", "node")
        ):
            return _finish(result, "protocol_failure", "missing_target_identity")
        required_validation = [
            field for field in validation_fields
            if field in expected.get("validation", {})
        ]
        if any(field not in observed["validation"] for field in required_validation):
            return _finish(result, "protocol_failure", "missing_validation")

    compare_subtree = True
    if expected_task is not None:
        prompt_spec = expected_task.get("prompt_spec") or {}
        compare_subtree = bool(prompt_spec.get("disclose_subtree"))

    for index, (observed, expected) in enumerate(
        zip(observed_targets, expected_targets)
    ):
        result["mismatches"].extend(_compare_target(
            observed,
            expected,
            index,
            validation_fields,
            compare_subtree=compare_subtree,
        ))
        prefix = "targets[{}]".format(index)
        result["diagnostics"].extend(_array_diagnostics(
            observed["signal"], expected.get("signal", {}),
            prefix + ".signal", "signal",
        ))
        result["diagnostics"].extend(_array_diagnostics(
            observed["time_axis"], expected.get("time_axis", {}),
            prefix + ".time_axis", "time_axis",
        ))

    subtype = gold.get("subtype")
    if expected_task is not None:
        subtype = expected_task.get("subtype", subtype)
    if subtype in ("same_shot_alignment", "cross_shot_comparison"):
        observed_derived = reported.get("derived")
        if (
            isinstance(observed_derived, dict)
            and not isinstance(observed_derived.get("comparison"), dict)
            and isinstance(reported.get("comparison"), dict)
        ):
            # Pilot prompts described derived and comparison as sibling objects.
            # Normalize that legacy, prompt-compliant shape before evaluation.
            observed_derived = dict(observed_derived)
            observed_derived["comparison"] = reported["comparison"]
        expected_derived = gold.get("derived")
        if not _derived_shape_valid(observed_derived):
            return _finish(result, "protocol_failure", "missing_derived_summary")
        if not isinstance(expected_derived, dict):
            return _finish(result, "infrastructure_failure", "invalid_gold")
        result["mismatches"].extend(
            _compare_derived(observed_derived, expected_derived)
        )
        for name in ("common_time_axis", "aligned_first", "aligned_second"):
            result["diagnostics"].extend(_array_diagnostics(
                observed_derived[name], expected_derived.get(name, {}),
                "derived." + name, name,
            ))

    if subtype == "adaptive_access":
        observed_decision = reported.get("resource_decision")
        expected_decisions = gold.get("resource_decisions")
        if not isinstance(observed_decision, dict):
            return _finish(result, "protocol_failure", "missing_resource_decision")
        if not isinstance(expected_decisions, list) or len(expected_decisions) != 1:
            return _finish(result, "infrastructure_failure", "invalid_gold")
        result["mismatches"].extend(
            _compare_decision(observed_decision, expected_decisions[0])
        )

    if result["mismatches"]:
        return _finish(
            result, "semantic_failure", result["mismatches"][0]["code"]
        )
    return _finish(result, "success", "semantic_success")

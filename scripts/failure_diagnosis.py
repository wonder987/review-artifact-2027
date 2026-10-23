#!/usr/bin/env python3
"""Create export-safe multi-grained PA-SCoRe diagnoses."""

from __future__ import print_function

import copy

try:
    from scripts.failure_taxonomy import (
        classify_evaluation_outcome,
        classify_execution_failure,
    )
    from scripts.pascore_contracts import assert_export_safe, bounded_text
except ImportError:
    from failure_taxonomy import classify_evaluation_outcome, classify_execution_failure
    from pascore_contracts import assert_export_safe, bounded_text


REPAIR_CONSTRAINTS = {
    "hallucinated_api": [
        "Use the remote MDSplus Connection API only.",
        "Retrieve values with Connection.get(expression).data().",
        "Do not call getNode, mdsopen, open, close, or disconnect on Connection.",
    ],
    "local_remote_api_confusion": [
        "Do not construct a local MDSplus.Tree for the remote server.",
        "Use Connection.openTree followed by Connection.get.",
    ],
    "wrong_api_signature": [
        "Use Connection.openTree(tree, shot) with the documented argument order.",
        "Pass TDI expressions as strings to Connection.get.",
    ],
    "node_lookup_error": [
        "Preserve the requested signal identity.",
        "Use TCL set def only for a disclosed or programmatically discovered subtree.",
    ],
    "timebase_retrieval_error": [
        "Retrieve the time axis with Connection.get('dim_of(<same signal expression>)').data().",
        "Do not guess a separate TIME or T node.",
    ],
    "tdi_expression_error": [
        "Pass exactly one TDI expression string to Connection.get; do not pass time bounds or point counts as additional Python arguments.",
        "Use matching signal and dim_of expressions, and preserve the requested target and window.",
    ],
    "resource_guard_rejection": [
        "Replace the unbounded read with an index or time-bounded TDI expression.",
        "Keep the estimated retrieval within the task resource budget.",
    ],
    "memory_exhaustion": [
        "Avoid full materialization of oversized signal and time arrays.",
        "Use bounded server-side access before NumPy conversion.",
    ],
    "execution_timeout": [
        "Reduce remote data transfer and avoid repeated full reads.",
    ],
    "unknown_runtime_error": [
        "Initialize the remote context in this order: Connection(server), Connection.openTree(tree, shot), optional TCL set def, then Connection.get reads.",
        "Call Connection.openTree(tree, shot) before setting a subtree or retrieving signal and time expressions.",
    ],
    "syntax_error": [
        "Return a compact, complete, syntactically valid self-contained Python program.",
        "Do not include Markdown fences, repeated commentary, or a truncated program.",
    ],
    "import_error": [
        "Import the installed package with exact capitalization: `import MDSplus` or `from MDSplus import Connection`.",
        "Do not import `Mdsplus` or another spelling.",
    ],
    "invalid_json": [
        "Return executable Python code, not a JSON example or a copy of the feedback object.",
        "The program must print exactly one valid JSON object and no other stdout text.",
    ],
    "target_schema_error": [
        "Emit an ordered targets list with one complete target item per requested signal.",
        "Place identity under targets[i].target and summaries under targets[i].signal and targets[i].time_axis.",
    ],
    "validation_schema_error": [
        "Include every validation field required by the original output contract using ordinary Python booleans.",
    ],
    "derived_schema_error": [
        "Emit derived.common_time_axis, derived.aligned_first, and derived.aligned_second summary objects.",
        "Place count, a_mean, a_std, b_mean, b_std, and pearson_correlation inside derived.comparison.",
    ],
    "resource_decision_schema_error": [
        "Emit the complete resource_decision object required by the original output contract.",
    ],
    "resource_decision_mismatch": [
        "Set resource_decision.points to the disclosed target metadata sample_num.",
        "Set resource_decision.total_mib to the disclosed target metadata estimated_total_mib. Do not recompute it from NumPy dtypes or retrieved array sizes.",
        "Compare those disclosed values with safety_policy.max_points and safety_policy.max_total_mib; allow_full_read is true only when both limits are satisfied.",
        "Set recommended_action to read_full_signal when allow_full_read is true, otherwise limit_time_window_or_downsample.",
    ],
    "status_contract_error": [
        "Use status success only for a complete successful result; otherwise emit the required failure object and exit nonzero.",
    ],
    "target_identity_mismatch": [
        "Preserve server, tree, shot, node, and disclosed subtree exactly.",
    ],
    "signal_mismatch": [
        "Recheck the target expression and avoid dtype-changing conversions.",
        "Keep raw bounded arrays in targets; place interpolated or aligned arrays only in derived.",
    ],
    "time_axis_mismatch": [
        "Use dim_of on the same bounded signal expression.",
    ],
    "point_or_window_mismatch": [
        "Preserve the requested physical window and expected point count.",
    ],
    "alignment_or_statistic_mismatch": [
        "Align signals on the requested common grid before computing statistics.",
        "Build the float64 common grid over the exact intersection of the two retrieved time axes and interpolate both signals onto that same grid.",
        "Set pearson_correlation to None when either aligned standard deviation is zero; otherwise compute it from the two aligned arrays.",
    ],
}


def _constraint_subtype_for_mismatch(code):
    code = str(code or "")
    if code == "target_identity_mismatch":
        return "target_identity_mismatch"
    if code.startswith("signal_"):
        return "signal_mismatch"
    if code.startswith("time_axis_"):
        return "time_axis_mismatch"
    if code.startswith(("derived_", "aligned_", "common_time_axis_")):
        return "alignment_or_statistic_mismatch"
    if code == "resource_decision_mismatch":
        return "resource_decision_mismatch"
    if code in (
        "target_count_mismatch", "validation_mismatch",
        "point_count_mismatch", "window_mismatch",
    ):
        return "point_or_window_mismatch"
    return None


def _append_unique(values, additions):
    for item in additions:
        if item not in values:
            values.append(item)


def _raw_evidence(execution):
    reported = execution.get("reported_result")
    source = reported if isinstance(reported, dict) else execution
    return {
        "execution_status": execution.get("status"),
        "error_type": source.get("error_type"),
        "error_message": bounded_text(source.get("error_message")),
        "failed_stage": source.get("failed_stage"),
        "policy_violations": execution.get("policy_violations") or [],
    }


def diagnose_failure(
    execution,
    evaluation=None,
    guard=None,
    program_source="",
    enabled=True,
):
    if not enabled:
        value = {
            "schema_version": "1.0",
            "diagnosis_version": "failure-diagnosis-v1",
            "enabled": False,
            "major_category": None,
            "failure_subtype": None,
            "failure_stage": _raw_evidence(execution).get("failed_stage"),
            "evidence": _raw_evidence(execution),
            "repair_constraints": [],
            "gold_disclosure": False,
        }
        assert_export_safe(value)
        return value

    if guard and guard.get("decision") in ("reject", "require_bounded_access"):
        classified = {
            "major_category": "resource_runtime",
            "failure_subtype": "resource_guard_rejection",
            "evidence": {
                "guard_decision": guard.get("decision"),
                "policy_violations": guard.get("violations") or [],
                "recommended_action": guard.get("recommended_action"),
            },
        }
    else:
        enriched = copy.deepcopy(execution)
        if program_source:
            enriched["diagnostic_program_source"] = program_source
        if isinstance(evaluation, dict):
            classified = classify_evaluation_outcome(enriched, evaluation)
        else:
            classified = classify_execution_failure(enriched)

    evidence = classified.get("evidence") or {}
    mismatch_codes = []
    mismatch_paths = []
    if isinstance(evaluation, dict):
        for mismatch in evaluation.get("mismatches") or []:
            if not isinstance(mismatch, dict):
                continue
            code = mismatch.get("code")
            path = mismatch.get("path")
            if code and code not in mismatch_codes:
                mismatch_codes.append(code)
            if path and path not in mismatch_paths:
                mismatch_paths.append(path)
    if mismatch_codes:
        evidence["mismatch_codes"] = sorted(mismatch_codes)
    if mismatch_paths:
        evidence["mismatch_paths"] = sorted(mismatch_paths)
    if evidence.get("error_message"):
        evidence["error_message"] = bounded_text(evidence["error_message"])
    subtype = classified.get("failure_subtype")
    repair_constraints = list(REPAIR_CONSTRAINTS.get(subtype, ()))
    for code in mismatch_codes:
        secondary_subtype = _constraint_subtype_for_mismatch(code)
        if secondary_subtype:
            _append_unique(
                repair_constraints,
                REPAIR_CONSTRAINTS.get(secondary_subtype, ()),
            )
    value = {
        "schema_version": "1.0",
        "diagnosis_version": "failure-diagnosis-v1",
        "enabled": True,
        "major_category": classified.get("major_category"),
        "failure_subtype": subtype,
        "failure_stage": evidence.get("failed_stage"),
        "evidence": evidence,
        "repair_constraints": repair_constraints,
        "gold_disclosure": False,
    }
    assert_export_safe(value)
    return value

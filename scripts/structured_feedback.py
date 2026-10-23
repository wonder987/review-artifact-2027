#!/usr/bin/env python3
"""Render Gold-safe structured repair feedback."""

from __future__ import print_function

try:
    from scripts.pascore_contracts import assert_export_safe, bounded_text
except ImportError:
    from pascore_contracts import assert_export_safe, bounded_text


def _identity_subset(identity):
    fields = ("run_id", "task_id", "model_alias", "condition", "round")
    return dict((field, identity.get(field)) for field in fields)


def _evidence_lines(evidence):
    lines = []
    for key in (
        "error_type", "error_message", "failed_stage", "guard_decision",
        "matched_rule", "source_signature", "evaluation_category",
        "evaluation_code",
    ):
        if evidence.get(key) not in (None, "", []):
            lines.append("{}: {}".format(key, bounded_text(evidence[key], 1000)))
    for violation in evidence.get("policy_violations") or []:
        lines.append("policy_violation: {}".format(bounded_text(violation, 1000)))
    for code in evidence.get("mismatch_codes") or []:
        lines.append("mismatch_code: {}".format(bounded_text(code, 1000)))
    for path in evidence.get("mismatch_paths") or []:
        lines.append("mismatch_path: {}".format(bounded_text(path, 1000)))
    return lines


def build_feedback(
    identity,
    diagnosis,
    remaining_rounds,
    structured=True,
    timeout_seconds=90,
    max_memory_mib=8192,
    grounding=None,
):
    evidence = diagnosis.get("evidence") or {}
    if not structured:
        value = {
            "schema_version": "1.0",
            "feedback_version": "structured-feedback-v1",
            "structured": False,
            "identity": _identity_subset(identity),
            "raw_error": "\n".join(_evidence_lines(evidence)),
            "budget": {
                "remaining_repair_rounds": int(remaining_rounds),
                "timeout_seconds": int(timeout_seconds),
                "max_memory_mib": int(max_memory_mib),
            },
        }
        assert_export_safe(value)
        return value

    subtype = diagnosis.get("failure_subtype")
    access_plan = []
    if isinstance(grounding, dict) and grounding.get("enabled"):
        access_plan = list(grounding.get("access_plan") or [])
    required_changes = list(diagnosis.get("repair_constraints") or [])
    if access_plan:
        required_changes.append(
            "Use the verified access plan exactly; pass each expression as the sole argument to Connection.get."
        )
    summary = "Repair the program after {} / {}.".format(
        diagnosis.get("major_category") or "unclassified_failure",
        subtype or "unknown",
    )
    value = {
        "schema_version": "1.0",
        "feedback_version": "structured-feedback-v1",
        "structured": True,
        "identity": _identity_subset(identity),
        "summary": summary,
        "failure_category": diagnosis.get("major_category"),
        "failure_subtype": subtype,
        "evidence": _evidence_lines(evidence),
        "required_changes": required_changes,
        "verified_access_plan": access_plan,
        "must_preserve": [
            "Preserve the original task target and requested operation.",
            "Print exactly one JSON object to stdout.",
            "Keep success and failure output compatible with the task protocol.",
        ],
        "prohibited_actions": [
            "Do not hard-code expected signal values, statistics, hashes, or outcomes.",
            "Do not read Gold, reference, evaluator, or diagnostic files.",
            "Do not substitute a different signal, shot, tree, or requested window.",
        ],
        "budget": {
            "remaining_repair_rounds": int(remaining_rounds),
            "timeout_seconds": int(timeout_seconds),
            "max_memory_mib": int(max_memory_mib),
        },
    }
    assert_export_safe(value)
    return value

#!/usr/bin/env python3
"""Task-aware pre-execution resource guard for generated MDSplus programs."""

from __future__ import print_function

import ast


def _constant_string(node):
    if isinstance(node, ast.Str):
        return node.s
    if hasattr(ast, "Constant") and isinstance(node, ast.Constant):
        return node.value if isinstance(node.value, str) else None
    return None


def read_expressions(source):
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    expressions = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        if isinstance(node.func, ast.Attribute) and node.func.attr == "get":
            value = _constant_string(node.args[0])
            if value is not None:
                expressions.append(value)
    return expressions


def _target_in_expression(target, expression):
    value = str(expression).replace(" ", "").lower()
    candidates = (target.get("node"), target.get("signal_name"))
    return any(
        candidate and str(candidate).replace(" ", "").lower() in value
        for candidate in candidates
    )


def guard_program(source, task, enabled=True):
    expressions = read_expressions(source)
    policy = task.get("safety_policy") or {}
    targets = task.get("targets") or []
    oversized_targets = []
    for target in targets:
        points = target.get("sample_num")
        total_mib = target.get("estimated_total_mib")
        if (
            points is not None and policy.get("max_points") is not None
            and float(points) > float(policy["max_points"])
        ) or (
            total_mib is not None and policy.get("max_total_mib") is not None
            and float(total_mib) > float(policy["max_total_mib"])
        ):
            oversized_targets.append(target)

    full_read = False
    bounded_read = False
    matched_expressions = []
    for expression in expressions:
        matches = [
            target for target in targets if _target_in_expression(target, expression)
        ]
        if not matches:
            continue
        matched_expressions.append(expression)
        if "[" in expression and "]" in expression:
            bounded_read = True
        elif any(target in oversized_targets for target in matches):
            full_read = True

    decision = "allow"
    violations = []
    recommended_action = None
    if enabled and oversized_targets and full_read:
        decision = "require_bounded_access"
        violations.append("full_read_exceeds_task_resource_budget")
        recommended_action = (
            "Use an index or time-bounded TDI expression before retrieving data."
        )

    return {
        "schema_version": "1.0",
        "guard_version": "resource-aware-guard-v1",
        "enabled": bool(enabled),
        "task_id": task.get("id"),
        "decision": decision,
        "checks": {
            "bounded_read_detected": bool(bounded_read),
            "expression_count": len(expressions),
            "full_read_detected": bool(full_read),
            "matched_expressions": matched_expressions,
            "oversized_target_count": len(oversized_targets),
            "max_points": policy.get("max_points"),
            "max_total_mib": policy.get("max_total_mib"),
        },
        "violations": violations,
        "recommended_action": recommended_action,
    }


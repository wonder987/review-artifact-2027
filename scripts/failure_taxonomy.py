#!/usr/bin/env python3
"""Deterministic root-cause classification for PA-SCoRe failures."""

from __future__ import print_function

import json
import re


def validate_taxonomy(value):
    if not isinstance(value, dict):
        raise ValueError("taxonomy must be an object")
    if not value.get("schema_version"):
        raise ValueError("taxonomy requires schema_version")
    if not value.get("taxonomy_version"):
        raise ValueError("taxonomy requires taxonomy_version")
    categories = value.get("major_categories")
    if not isinstance(categories, list) or not categories:
        raise ValueError("taxonomy requires major_categories")

    category_ids = set()
    subtype_owners = {}
    for category in categories:
        if not isinstance(category, dict) or not category.get("id"):
            raise ValueError("each major category requires an id")
        category_id = category["id"]
        if category_id in category_ids:
            raise ValueError("duplicate major category: {}".format(category_id))
        category_ids.add(category_id)
        subtypes = category.get("subtypes")
        if not isinstance(subtypes, list) or not subtypes:
            raise ValueError("major category requires subtypes: {}".format(category_id))
        for subtype in subtypes:
            if not isinstance(subtype, str) or not subtype:
                raise ValueError("subtype ids must be non-empty strings")
            if subtype in subtype_owners:
                raise ValueError(
                    "subtype {} belongs to both {} and {}".format(
                        subtype, subtype_owners[subtype], category_id
                    )
                )
            subtype_owners[subtype] = category_id
    return value


def load_taxonomy(path):
    with open(path, "r") as handle:
        return validate_taxonomy(json.load(handle))


def _source(execution):
    reported = execution.get("reported_result")
    if isinstance(reported, dict):
        return reported
    return execution


def _evidence(execution, matched_rule, source_signature=None):
    source = _source(execution)
    evidence = {
        "execution_status": execution.get("status"),
        "error_type": source.get("error_type") or execution.get("error_type"),
        "error_message": source.get("error_message") or execution.get("error_message"),
        "failed_stage": source.get("failed_stage") or execution.get("failed_stage"),
        "failure_code": execution.get("failure_code"),
        "policy_violations": execution.get("policy_violations") or [],
        "matched_rule": matched_rule,
    }
    if source_signature:
        evidence["source_signature"] = source_signature
    return evidence


def _result(
    major_category,
    failure_subtype,
    execution,
    matched_rule,
    source_signature=None,
):
    return {
        "major_category": major_category,
        "failure_subtype": failure_subtype,
        "evidence": _evidence(execution, matched_rule, source_signature),
    }


def _contains_any(value, fragments):
    return any(fragment in value for fragment in fragments)


def _remote_connection_source_signature(program_source):
    if not isinstance(program_source, str) or not program_source:
        return None
    assignments = re.findall(
        r"\b([A-Za-z_]\w*)\s*=\s*(?:MDSplus\s*\.\s*)?Connection\s*\(",
        program_source,
    )
    for variable in assignments:
        if re.search(r"\b{}\s*\.\s*getNode\s*\(".format(re.escape(variable)), program_source):
            return "remote_connection_getNode"
    return None


def _unqualified_time_node_source_signature(program_source):
    if not isinstance(program_source, str) or not program_source:
        return None
    if re.search(
        r"\.\s*get\s*\(\s*['\"](?:T|TIME)['\"]\s*\)",
        program_source,
        re.IGNORECASE,
    ):
        return "unqualified_time_node"
    return None


def classify_execution_failure(execution):
    if not isinstance(execution, dict):
        raise ValueError("execution must be an object")

    status = str(execution.get("status") or "").lower()
    source = _source(execution)
    error_type = str(source.get("error_type") or execution.get("error_type") or "")
    error_message = str(
        source.get("error_message") or execution.get("error_message") or ""
    )
    failed_stage = str(
        source.get("failed_stage") or execution.get("failed_stage") or ""
    )
    error_type_lower = error_type.lower()
    message = error_message.lower()
    diagnostic_text = "\n".join(
        str(execution.get(field) or "")
        for field in ("diagnostic_stdout", "diagnostic_stderr")
    ).lower()
    combined_message = "{}\n{}".format(message, diagnostic_text)
    stage = failed_stage.lower()
    source_signature = _remote_connection_source_signature(
        execution.get("diagnostic_program_source")
    )
    time_node_signature = _unqualified_time_node_source_signature(
        execution.get("diagnostic_program_source")
    )

    infrastructure_code = str(execution.get("infrastructure_code") or "").lower()
    if status == "infrastructure_failure" or infrastructure_code:
        subtype = infrastructure_code if infrastructure_code in (
            "missing_artifact", "corrupt_artifact", "provider_outage",
            "network_failure", "evaluator_failure",
        ) else "evaluator_failure"
        return _result("infrastructure", subtype, execution, "infrastructure_provenance")

    if status == "generation_failure":
        if _contains_any(message, ("503", "high demand", "unavailable")):
            return _result("infrastructure", "provider_outage", execution, "provider_outage")
        if _contains_any(message, ("network", "connection reset", "tls", "ssl")):
            return _result("infrastructure", "network_failure", execution, "network_failure")
        return _result("code_generation", "generation_timeout", execution, "generation_timeout")

    if status == "policy_rejected":
        violations = execution.get("policy_violations") or []
        if violations and all(str(item).startswith("syntax_error:") for item in violations):
            return _result("code_generation", "syntax_error", execution, "syntax_policy")
        return _result("code_generation", "unsafe_operation", execution, "static_policy")

    if status == "timeout" or execution.get("timed_out"):
        return _result("resource_runtime", "execution_timeout", execution, "execution_timeout")

    if error_type_lower in ("importerror", "modulenotfounderror"):
        return _result("code_generation", "import_error", execution, "runtime_import")
    if "syntaxerror" in error_type_lower:
        return _result("code_generation", "syntax_error", execution, "runtime_syntax")
    if "memoryerror" in error_type_lower or _contains_any(
        combined_message,
        (
            "unable to allocate",
            "out of memory",
            "cannot allocate memory",
            "insufficient virtual memory",
        ),
    ):
        return _result("resource_runtime", "memory_exhaustion", execution, "memory_error")

    if source_signature == "remote_connection_getNode":
        return _result(
            "mdsplus_interaction",
            "hallucinated_api",
            execution,
            "generated_source_api_signature",
            source_signature,
        )

    if _contains_any(
        message,
        (
            "mdsplus.tree",
            "local tree",
            "tree object for remote",
            "'connection' object has no attribute 'tree'",
        ),
    ):
        return _result(
            "mdsplus_interaction", "local_remote_api_confusion", execution,
            "local_remote_api",
        )

    if _contains_any(
        message,
        (
            "missing 1 required positional argument",
            "missing required positional argument",
            "unexpected keyword argument",
            "takes exactly",
            "takes no arguments",
            "'int' object has no attribute 'upper'",
        ),
    ):
        return _result(
            "mdsplus_interaction", "wrong_api_signature", execution,
            "api_signature",
        )

    if "has no attribute" in message and _contains_any(
        message,
        (
            "connection", "mdsplus", "getnode", "mdsopen", "mdsserver",
            "disconnect", "execute", "get_data", "close", "open",
        ),
    ):
        return _result(
            "mdsplus_interaction", "hallucinated_api", execution,
            "nonexistent_api_attribute",
        )

    if _contains_any(message, ("%tdi-e-", "tdi-e-")):
        return _result(
            "mdsplus_interaction", "tdi_expression_error", execution,
            "tdi_status",
        )
    if time_node_signature and _contains_any(
        message, ("%tree-e-nopath", "%tree-w-nnf", "treenopath")
    ):
        return _result(
            "mdsplus_interaction",
            "timebase_retrieval_error",
            execution,
            "generated_source_time_node",
            time_node_signature,
        )
    if _contains_any(message, ("%tree-e-nopath", "%tree-w-nnf", "treenopath")):
        return _result(
            "mdsplus_interaction", "node_lookup_error", execution,
            "tree_node_status",
        )
    if "could not locate node" in message:
        return _result(
            "mdsplus_interaction", "node_lookup_error", execution,
            "node_lookup_wrapper",
        )
    if _contains_any(message, ("numpy.ndarray", "no overlap", "array", "broadcast")):
        return _result(
            "resource_runtime", "array_processing_error", execution,
            "array_processing",
        )
    if _contains_any(
        message,
        (
            "keys must be str",
            "not json serializable",
            "json serialization",
            "key <class 'bool'> is not a string",
        ),
    ):
        return _result(
            "resource_runtime", "serialization_error", execution,
            "serialization",
        )
    if "could not determine time base" in message:
        return _result(
            "mdsplus_interaction", "timebase_retrieval_error", execution,
            "timebase_message",
        )
    if _contains_any(stage, ("time", "dimension", "dim_of")):
        return _result(
            "mdsplus_interaction", "timebase_retrieval_error", execution,
            "timebase_stage",
        )
    return _result(
        "resource_runtime", "unknown_runtime_error", execution,
        "unknown_runtime",
    )


def classify_evaluation_outcome(execution, evaluation):
    category = str(evaluation.get("category") or "")
    code = str(evaluation.get("code") or "")
    if category == "success":
        return {
            "major_category": "success",
            "failure_subtype": "semantic_success",
            "evidence": {"evaluation_category": category, "evaluation_code": code},
        }
    if category in ("generation_failure", "execution_failure"):
        return classify_execution_failure(execution)
    if category == "infrastructure_failure":
        if code.startswith("missing_"):
            subtype = "missing_artifact"
        elif code.startswith("corrupt_") or code in ("invalid_gold",):
            subtype = "corrupt_artifact"
        else:
            subtype = "evaluator_failure"
        return {
            "major_category": "infrastructure",
            "failure_subtype": subtype,
            "evidence": {"evaluation_category": category, "evaluation_code": code},
        }
    if category == "protocol_failure":
        if code == "invalid_json":
            subtype = "invalid_json"
        elif code in (
            "missing_target_summary", "missing_target_identity",
            "target_count_mismatch",
        ):
            subtype = "target_schema_error"
        elif code == "missing_validation":
            subtype = "validation_schema_error"
        elif code == "missing_derived_summary":
            subtype = "derived_schema_error"
        elif code == "missing_resource_decision":
            subtype = "resource_decision_schema_error"
        else:
            subtype = "status_contract_error"
        return {
            "major_category": "output_protocol",
            "failure_subtype": subtype,
            "evidence": {"evaluation_category": category, "evaluation_code": code},
        }
    if category == "semantic_failure":
        if code == "target_identity_mismatch":
            subtype = "target_identity_mismatch"
        elif code.startswith("signal_"):
            subtype = "signal_mismatch"
        elif code.startswith("time_axis_"):
            subtype = "time_axis_mismatch"
        elif code in (
            "target_count_mismatch", "validation_mismatch",
            "point_count_mismatch", "window_mismatch",
        ):
            subtype = "point_or_window_mismatch"
        elif code.startswith("derived_") or code.startswith("alignment_"):
            subtype = "alignment_or_statistic_mismatch"
        elif code == "resource_decision_mismatch":
            subtype = "resource_decision_mismatch"
        else:
            subtype = "alignment_or_statistic_mismatch"
        return {
            "major_category": "semantic_result",
            "failure_subtype": subtype,
            "evidence": {"evaluation_category": category, "evaluation_code": code},
        }
    return {
        "major_category": "infrastructure",
        "failure_subtype": "evaluator_failure",
        "evidence": {"evaluation_category": category, "evaluation_code": code},
    }

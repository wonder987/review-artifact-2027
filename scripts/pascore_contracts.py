#!/usr/bin/env python3
"""Shared deterministic contracts for PA-SCoRe artifacts."""

from __future__ import print_function

import hashlib
import json
import os


class ContractError(ValueError):
    pass


CONDITION_IDS = frozenset((
    "full",
    "no_metadata_grounding",
    "no_failure_diagnosis",
    "no_structured_feedback",
    "no_resource_guard",
    "single_round",
    "self_debugging",
))

SENSITIVE_EXPORT_KEYS = frozenset((
    "gold",
    "gold_hash",
    "gold_sha256",
    "gold_value",
    "oracle",
    "oracle_status",
    "reference",
    "reference_hash",
    "reference_index",
    "reference_program",
    "reference_value",
    "repeat_files",
))


def canonical_json_bytes(value):
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha256_text(value):
    return sha256_bytes(value.encode("utf-8"))


def sha256_json(value):
    return sha256_bytes(canonical_json_bytes(value))


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    directory = os.path.dirname(os.path.abspath(path))
    if not os.path.isdir(directory):
        os.makedirs(directory)
    temporary = path + ".tmp"
    with open(temporary, "w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_json(path):
    with open(path, "r") as handle:
        return json.load(handle)


def bounded_text(value, limit=2000):
    text = str(value or "")
    if len(text) <= int(limit):
        return text
    return text[:int(limit)] + "...[truncated]"


def validate_condition(value):
    if not isinstance(value, dict):
        raise ContractError("condition must be an object")
    required = (
        "schema_version",
        "condition_id",
        "metadata_grounding",
        "resource_guard",
        "failure_diagnosis",
        "structured_feedback",
        "max_repair_rounds",
    )
    missing = [field for field in required if field not in value]
    if missing:
        raise ContractError("condition missing fields: {}".format(", ".join(missing)))
    if value["condition_id"] not in CONDITION_IDS:
        raise ContractError("unknown condition: {}".format(value["condition_id"]))
    for field in ("metadata_grounding", "resource_guard", "structured_feedback"):
        if not isinstance(value[field], bool):
            raise ContractError("{} must be boolean".format(field))
    if value["failure_diagnosis"] not in ("multi_grained", "raw_only"):
        raise ContractError("unsupported failure diagnosis mode")
    rounds = value["max_repair_rounds"]
    if not isinstance(rounds, int) or rounds < 0 or rounds > 2:
        raise ContractError("max_repair_rounds must be between 0 and 2")
    expected = {
        "no_metadata_grounding": ("metadata_grounding", False),
        "no_resource_guard": ("resource_guard", False),
        "no_failure_diagnosis": ("failure_diagnosis", "raw_only"),
        "no_structured_feedback": ("structured_feedback", False),
        "single_round": ("max_repair_rounds", 1),
    }.get(value["condition_id"])
    if expected and value.get(expected[0]) != expected[1]:
        raise ContractError(
            "{} requires {}={}".format(value["condition_id"], *expected)
        )
    if value["condition_id"] == "self_debugging":
        self_debugging_contract = {
            "failure_diagnosis": "raw_only",
            "max_repair_rounds": 1,
            "metadata_grounding": True,
            "resource_guard": True,
            "structured_feedback": False,
        }
        for field, expected_value in self_debugging_contract.items():
            if value.get(field) != expected_value:
                raise ContractError(
                    "self_debugging requires {}={}".format(
                        field, expected_value
                    )
                )
    return dict(value)


def _valid_hash(value):
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(character in "0123456789abcdef" for character in value.lower())


def build_identity(
    run_id,
    task_id,
    model_alias,
    condition,
    round_index,
    task_spec_hash,
    protocol_hash,
    program_hash,
    parent_program_hash=None,
):
    condition = validate_condition(condition)
    value = {
        "schema_version": "1.0",
        "run_id": str(run_id),
        "task_id": str(task_id),
        "model_alias": str(model_alias),
        "condition": condition["condition_id"],
        "condition_hash": sha256_json(condition),
        "round": int(round_index),
        "task_spec_hash": str(task_spec_hash),
        "protocol_hash": str(protocol_hash),
        "program_hash": str(program_hash),
        "parent_program_hash": parent_program_hash,
    }
    return validate_identity(value)


def validate_identity(value):
    if not isinstance(value, dict):
        raise ContractError("identity must be an object")
    required = (
        "schema_version", "run_id", "task_id", "model_alias", "condition",
        "condition_hash", "round", "task_spec_hash", "protocol_hash",
        "program_hash", "parent_program_hash",
    )
    missing = [field for field in required if field not in value]
    if missing:
        raise ContractError("identity missing fields: {}".format(", ".join(missing)))
    if value["condition"] not in CONDITION_IDS:
        raise ContractError("identity has unknown condition")
    if not isinstance(value["round"], int) or value["round"] < 0:
        raise ContractError("identity round must be nonnegative")
    for field in ("condition_hash", "task_spec_hash", "protocol_hash", "program_hash"):
        if not _valid_hash(value[field]):
            raise ContractError("identity has invalid {}".format(field))
    parent = value.get("parent_program_hash")
    if parent is not None and not _valid_hash(parent):
        raise ContractError("identity has invalid parent_program_hash")
    for field in ("run_id", "task_id", "model_alias"):
        if not isinstance(value[field], str) or not value[field]:
            raise ContractError("identity has empty {}".format(field))
    return dict(value)


def assert_export_safe(value, path="artifact"):
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = str(key).lower()
            if lowered in SENSITIVE_EXPORT_KEYS:
                raise ContractError("sensitive export field at {}.{}".format(path, key))
            assert_export_safe(child, "{}.{}".format(path, key))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            assert_export_safe(child, "{}[{}]".format(path, index))
    return value


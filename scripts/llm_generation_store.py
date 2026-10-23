#!/usr/bin/env python3
"""Extract generated programs and persist generation artifacts atomically."""

from __future__ import print_function

import ast
import hashlib
import json
import os
import re


class StoreError(ValueError):
    pass


def _sha256_text(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _valid_python(program):
    try:
        ast.parse(program)
    except (SyntaxError, ValueError, TypeError):
        return False
    return True


def extract_program(content):
    if not isinstance(content, str) or not content.strip():
        return {
            "error": "assistant content is empty",
            "program": None,
            "program_sha256": None,
            "status": "error",
            "strategy": "empty",
            "syntax_valid": None,
        }

    fence_pattern = re.compile(
        r"```(?:python|py)?\s*\n?(.*?)```",
        flags=re.IGNORECASE | re.DOTALL,
    )
    fenced = fence_pattern.findall(content)
    if len(fenced) > 1:
        return {
            "error": "multiple code fences are ambiguous",
            "program": None,
            "program_sha256": None,
            "status": "error",
            "strategy": "multiple_fences",
            "syntax_valid": None,
        }
    if len(fenced) == 1:
        program = fenced[0].strip()
        strategy = "single_fence"
    else:
        program = content.strip()
        strategy = "plain"

    if not program:
        return {
            "error": "assistant content is empty",
            "program": None,
            "program_sha256": None,
            "status": "error",
            "strategy": strategy,
            "syntax_valid": None,
        }
    return {
        "error": None,
        "program": program,
        "program_sha256": _sha256_text(program),
        "status": "success",
        "strategy": strategy,
        "syntax_valid": _valid_python(program),
    }


def result_key(
    protocol_hash,
    prompt_hash,
    model_alias,
    task_id,
    attempt,
    grounding_hash=None,
):
    identity = {
        "attempt": int(attempt),
        "model_alias": model_alias,
        "prompt_hash": prompt_hash,
        "protocol_hash": protocol_hash,
        "task_id": task_id,
    }
    if grounding_hash is not None:
        identity["grounding_hash"] = grounding_hash
    canonical = json.dumps(identity, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _safe_component(value):
    component = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value))
    if not component or component in (".", ".."):
        raise StoreError("unsafe result path component")
    return component


def result_directory(root, identity):
    return os.path.join(
        os.path.abspath(root),
        _safe_component(identity["model_alias"]),
        _safe_component(identity["task_id"]),
        "attempt-{}".format(int(identity["attempt"])),
    )


def _assert_no_credentials(value, path="request"):
    if isinstance(value, dict):
        for key, nested in value.items():
            if str(key).lower() in (
                "api_key",
                "authorization",
                "bearer",
                "secret",
            ):
                raise StoreError("credential field found at {}.{}".format(path, key))
            _assert_no_credentials(nested, "{}.{}".format(path, key))
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _assert_no_credentials(nested, "{}[{}]".format(path, index))
    elif isinstance(value, str):
        lowered = value.lower()
        if "bearer " in lowered or re.search(r"\bsk-[a-z0-9]", lowered):
            raise StoreError("credential-like value found at {}".format(path))


def _atomic_json(path, value):
    temporary = path + ".tmp"
    with open(temporary, "w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_text(path, value):
    temporary = path + ".tmp"
    with open(temporary, "wb") as handle:
        handle.write(value.encode("utf-8"))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_result(
    root,
    identity,
    request_snapshot,
    raw_response,
    normalized,
    extraction,
    transport,
):
    _assert_no_credentials(request_snapshot)
    directory = result_directory(root, identity)
    if not os.path.isdir(directory):
        os.makedirs(directory)

    _atomic_json(os.path.join(directory, "request.json"), request_snapshot)
    _atomic_json(os.path.join(directory, "response.raw.json"), raw_response)

    program_path = os.path.join(directory, "program.py")
    if extraction["status"] == "success":
        _atomic_text(program_path, extraction["program"])
    elif os.path.exists(program_path):
        os.remove(program_path)

    result = {
        "extraction": dict(extraction),
        "identity": dict(identity),
        "normalized": dict(normalized),
        "result_key": result_key(**identity),
        "status": "completed",
        "transport": dict(transport),
    }
    result["extraction"]["program"] = None
    _atomic_json(os.path.join(directory, "result.json"), result)
    failure_path = os.path.join(directory, "failure.json")
    if os.path.exists(failure_path):
        os.remove(failure_path)
    return directory


def is_complete(directory, expected_identity):
    path = os.path.join(directory, "result.json")
    if not os.path.isfile(path):
        return False
    try:
        with open(path, "r") as handle:
            result = json.load(handle)
    except (IOError, OSError, ValueError):
        return False
    return (
        result.get("status") == "completed"
        and result.get("identity") == expected_identity
        and result.get("result_key") == result_key(**expected_identity)
    )

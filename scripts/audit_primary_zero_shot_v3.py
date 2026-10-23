#!/usr/bin/env python3
"""Audit Primary Zero-shot V3 integrity, contract, and oracle isolation."""

from __future__ import print_function

import argparse
import hashlib
import json
import os
import re

try:
    from scripts.audit_primary_zero_shot import (
        AuditError,
        audit_records as audit_common_records,
        file_sha256,
        load_jsonl,
    )
    from scripts.render_primary_zero_shot_v3 import PROMPT_VERSION
except ImportError:
    from audit_primary_zero_shot import (
        AuditError,
        audit_records as audit_common_records,
        file_sha256,
        load_jsonl,
    )
    from render_primary_zero_shot_v3 import PROMPT_VERSION


HEX_SHA256_PATTERN = re.compile(
    r"(?i)(?<![0-9a-f])[0-9a-f]{64}(?![0-9a-f])"
)
VALIDATION_FIELDS = (
    "non_empty",
    "same_size",
    "signal_all_finite",
    "time_all_finite",
    "time_monotonic_non_decreasing",
    "time_strictly_increasing",
)
API_SOLUTION_TOKENS = ("opentree", "getnode", "dim_of", "tcl(", ".get(")


def _model_text(record):
    return "\n".join((
        record.get("system_prompt", ""),
        record.get("user_prompt", ""),
        record.get("prompt", ""),
    ))


def _contract(record):
    parts = record.get("user_prompt", "").split("\n\n", 1)
    if len(parts) != 2:
        raise AuditError(
            "{}: missing task/contract separator".format(
                record.get("task_id", "unknown")
            )
        )
    return parts[1]


def _audit_v3_contract(record):
    task_id = record.get("task_id", "unknown")
    contract = _contract(record)
    required_markers = (
        "targets[i].target itself is the identity object",
        "Do not add a separate targets[i].identity sibling",
    ) + VALIDATION_FIELDS
    for marker in required_markers:
        if marker not in contract:
            raise AuditError("{}: missing V3 contract marker {}".format(task_id, marker))
    lowered = contract.lower()
    for token in API_SOLUTION_TOKENS:
        if token in lowered:
            raise AuditError(
                "{}: output contract contains solution API token {}".format(
                    task_id, token
                )
            )


def audit_records(manifest, records, spec_rich_records=None):
    for record in records:
        if HEX_SHA256_PATTERN.search(_model_text(record)):
            raise AuditError(
                "{}: model-visible text contains a SHA-256 value".format(
                    record.get("task_id", "unknown")
                )
            )
        _audit_v3_contract(record)
    return audit_common_records(
        manifest,
        records,
        spec_rich_records=spec_rich_records,
        prompt_version=PROMPT_VERSION,
        allowed_contract_tokens=("sha256", "allow_full_read"),
    )


def audit_files(manifest_path, primary_path, spec_rich_path=None):
    with open(manifest_path, "r") as handle:
        manifest = json.load(handle)
    records = load_jsonl(primary_path)
    spec_records = load_jsonl(spec_rich_path) if spec_rich_path else None
    report = audit_records(manifest, records, spec_records)
    report["source_manifest_sha256"] = file_sha256(manifest_path)
    report["primary_jsonl_sha256"] = file_sha256(primary_path)
    report["spec_rich_jsonl_sha256"] = (
        file_sha256(spec_rich_path) if spec_rich_path else None
    )
    return report


def write_index(output_path, report, records):
    index = dict(report)
    index["prompts"] = [
        {"prompt_sha256": record["prompt_sha256"], "task_id": record["task_id"]}
        for record in records
    ]
    directory = os.path.dirname(os.path.abspath(output_path))
    if not os.path.isdir(directory):
        os.makedirs(directory)
    temporary = output_path + ".tmp"
    with open(temporary, "w") as handle:
        json.dump(index, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output_path)
    return index


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--primary", required=True)
    parser.add_argument("--spec-rich")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    report = audit_files(args.manifest, args.primary, args.spec_rich)
    write_index(args.output, report, load_jsonl(args.primary))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

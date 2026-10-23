#!/usr/bin/env python3
"""Audit Primary Zero-shot prompts for integrity and oracle leakage."""

from __future__ import print_function

import argparse
import datetime
import hashlib
import json
import os
import re

try:
    from scripts.render_primary_zero_shot import PROMPT_VERSION
except ImportError:
    from render_primary_zero_shot import PROMPT_VERSION


class AuditError(ValueError):
    pass


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path):
    records = []
    with open(path, "r") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except ValueError as error:
                raise AuditError(
                    "{}:{} invalid JSON: {}".format(path, line_number, error)
                )
            if not isinstance(record, dict):
                raise AuditError(
                    "{}:{} JSONL record must be an object".format(
                        path, line_number
                    )
                )
            records.append(record)
    return records


def contains_token(text, token):
    pattern = r"(?<![A-Za-z0-9_]){}(?![A-Za-z0-9_])".format(
        re.escape(str(token))
    )
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


def _fail(task_id, message):
    raise AuditError("{}: {}".format(task_id, message))


def _model_text(record):
    return "\n".join((
        record.get("system_prompt", ""),
        record.get("user_prompt", ""),
        record.get("prompt", ""),
    ))


def _reference_indices(task):
    operation = task["operation"]
    windows = operation.get("per_target_windows", [operation])
    indices = []
    for window in windows:
        for key in ("index_start", "index_end"):
            if key in window:
                indices.append(int(window[key]))
    return indices


def _allowed_numeric_tokens(task):
    allowed = set([10001])
    for target in task["targets"]:
        allowed.add(int(target["shot"]))
    operation = task["operation"]
    windows = operation.get("per_target_windows", [operation])
    for window in windows:
        for key in (
            "time_start",
            "time_end",
            "time_step",
            "expected_points",
            "common_points",
        ):
            if key in window:
                value = window[key]
                allowed.add(str(value))
                try:
                    allowed.add(int(value))
                except (TypeError, ValueError):
                    pass
    if "common_points" in operation:
        allowed.add(int(operation["common_points"]))
    policy = task["safety_policy"]
    allowed.add(int(policy["max_points"]))
    allowed.add(str(policy["max_total_mib"]))
    return set(str(value) for value in allowed)


def _audit_record(
    task,
    record,
    prompt_version=PROMPT_VERSION,
    allowed_contract_tokens=(),
):
    task_id = task["id"]
    required = (
        "task_id",
        "split",
        "level",
        "subtype",
        "prompt_version",
        "system_prompt",
        "user_prompt",
        "prompt",
        "prompt_sha256",
    )
    missing = [key for key in required if key not in record]
    if missing:
        _fail(task_id, "missing fields: {}".format(", ".join(missing)))

    for field in ("split", "level", "subtype"):
        if record[field] != task[field]:
            _fail(task_id, "{} does not match manifest".format(field))
    if record["prompt_version"] != prompt_version:
        _fail(task_id, "unexpected prompt version")

    expected_prompt = record["system_prompt"] + "\n\n" + record["user_prompt"]
    if record["prompt"] != expected_prompt:
        _fail(task_id, "prompt is not the system/user composition")
    expected_hash = hashlib.sha256(record["prompt"].encode("utf-8")).hexdigest()
    if record["prompt_sha256"] != expected_hash:
        _fail(task_id, "prompt SHA-256 mismatch")

    text = _model_text(record)
    forbidden_tokens = (
        task_id,
        task["level"],
        task["subtype"],
        "sha256",
        "spec_sha256",
        "oracle_status",
        "index_start",
        "index_end",
        "expected_decision",
        "allow_full_read",
        "estimated_total_mib",
        "size_band",
    )
    allowed_contract_tokens = set(allowed_contract_tokens)
    for token in forbidden_tokens:
        if token in allowed_contract_tokens:
            continue
        if contains_token(text, token):
            _fail(task_id, "model-visible text contains forbidden token {}".format(token))

    if not task["prompt_spec"]["disclose_subtree"]:
        for target in task["targets"]:
            if contains_token(text, target["subtree"]):
                _fail(
                    task_id,
                    "model-visible text discloses hidden subtree {}".format(
                        target["subtree"]
                    ),
                )

    allowed_numbers = _allowed_numeric_tokens(task)
    for index in _reference_indices(task):
        token = str(index)
        if token not in allowed_numbers and contains_token(text, token):
            _fail(task_id, "model-visible text discloses reference index")

    expected_decision = task["operation"].get("expected_decision")
    if expected_decision and contains_token(text, expected_decision):
        _fail(task_id, "model-visible text discloses expected decision")


def audit_records(
    manifest,
    records,
    spec_rich_records=None,
    prompt_version=PROMPT_VERSION,
    allowed_contract_tokens=(),
):
    tasks = manifest["tasks"]
    expected_ids = [task["id"] for task in tasks]
    actual_ids = [record.get("task_id") for record in records]
    if len(records) != len(tasks):
        raise AuditError(
            "record count {} does not match manifest count {}".format(
                len(records), len(tasks)
            )
        )
    if len(set(actual_ids)) != len(actual_ids):
        raise AuditError("duplicate task IDs in Primary prompt set")
    if actual_ids != expected_ids:
        raise AuditError("Primary task IDs or order do not match manifest")

    prompts = [record.get("prompt") for record in records]
    if len(set(prompts)) != len(prompts):
        raise AuditError("duplicate model-visible prompts in Primary prompt set")

    for task, record in zip(tasks, records):
        if record.get("task_id") != task["id"]:
            _fail(task["id"], "task binding mismatch")
        _audit_record(
            task,
            record,
            prompt_version=prompt_version,
            allowed_contract_tokens=allowed_contract_tokens,
        )

    if spec_rich_records is not None:
        spec_by_id = dict(
            (record.get("task_id"), record) for record in spec_rich_records
        )
        for record in records:
            spec_record = spec_by_id.get(record["task_id"])
            if spec_record and record["prompt"] == spec_record.get("prompt"):
                _fail(record["task_id"], "Primary prompt reuses Spec-rich text")

    split_counts = {}
    for record in records:
        split_counts[record["split"]] = split_counts.get(record["split"], 0) + 1
    if split_counts.get("pilot") != 30 or split_counts.get("held_out") != 108:
        raise AuditError("unexpected pilot/held-out split counts")

    return {
        "held_out_count": split_counts["held_out"],
        "pilot_count": split_counts["pilot"],
        "prompt_version": prompt_version,
        "record_count": len(records),
        "status": "passed",
    }


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
    index["created_utc"] = (
        datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    )
    index["prompts"] = [
        {
            "prompt_sha256": record["prompt_sha256"],
            "task_id": record["task_id"],
        }
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--primary", required=True)
    parser.add_argument("--spec-rich")
    parser.add_argument("--index-output")
    return parser.parse_args()


def main():
    args = parse_args()
    report = audit_files(args.manifest, args.primary, args.spec_rich)
    if args.index_output:
        write_index(args.index_output, report, load_jsonl(args.primary))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

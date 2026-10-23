#!/usr/bin/env python3
"""Prepare leakage-controlled prompts for a Self-Debugging baseline."""

from __future__ import print_function

import argparse
import hashlib
import json
import os


EXPLANATION_PROMPT_VERSION = "self-debugging-explanation-v1"
EXPLANATION_PROTOCOL_ID = "self-debugging-explanation-heldout-qwen-v1"
SYSTEM_PROMPT = (
    "You are performing self-debugging for a Python 3.6 scientific "
    "data-access program. Analyze the previous program against the original "
    "grounded task and the raw execution observation. Return exactly one JSON "
    "object with three keys: explanation (a string), suspected_faults (an "
    "array of strings), and repair_plan (an array of strings). Do not return "
    "a corrected program. Do not invent expected arrays, hashes, statistics, "
    "or numerical answers."
)


def _sha256_text(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path):
    with open(path, "r") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("JSON root must be an object: {}".format(path))
    return value


def _load_jsonl(path):
    records = []
    with open(path, "r") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(
                    "JSONL record must be an object: {}:{}".format(path, line_number)
                )
            records.append(value)
    return records


def _atomic_json(path, value):
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


def _atomic_jsonl(path, records):
    directory = os.path.dirname(os.path.abspath(path))
    if not os.path.isdir(directory):
        os.makedirs(directory)
    temporary = path + ".tmp"
    with open(temporary, "w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True))
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _read_required_text(path, label, limit=65536):
    if not os.path.isfile(path):
        raise ValueError("missing {}: {}".format(label, path))
    with open(path, "rb") as handle:
        data = handle.read(int(limit) + 1)
    truncated = len(data) > int(limit)
    text = data[:int(limit)].decode("utf-8", errors="replace")
    if truncated:
        text += "\n...[truncated]"
    return text


def _safe_package_path(package_root, relative_path):
    normalized = os.path.normpath(str(relative_path))
    if os.path.isabs(normalized) or normalized.startswith(".." + os.sep):
        raise ValueError("unsafe package path: {}".format(relative_path))
    absolute = os.path.abspath(os.path.join(package_root, normalized))
    if not absolute.startswith(os.path.abspath(package_root) + os.sep):
        raise ValueError("package path escapes root: {}".format(relative_path))
    return absolute


def _default_protocol(model_alias):
    return {
        "default_split": "pilot",
        "frozen": True,
        "frozen_model_aliases": [model_alias],
        "generation_attempts": 1,
        "generation_mode": "direct_zero_shot",
        "held_out_enabled": True,
        "max_network_retries": 3,
        "max_output_tokens": 4096,
        "pilot_gate_status": "passed",
        "prompt_version": EXPLANATION_PROMPT_VERSION,
        "protocol_id": EXPLANATION_PROTOCOL_ID,
        "repair_rounds": 0,
        "request_timeout_seconds": 600,
        "retry_backoff_seconds": [2, 4, 8],
        "schema_version": "2.0",
        "temperature": 0,
    }


def _build_protocol(base_protocol_path, model_alias):
    protocol = (
        _load_json(base_protocol_path)
        if base_protocol_path
        else _default_protocol(model_alias)
    )
    protocol.update({
        "frozen": True,
        "frozen_model_aliases": [model_alias],
        "generation_mode": "direct_zero_shot",
        "max_output_tokens": min(int(protocol.get("max_output_tokens", 4096)), 4096),
        "prompt_version": EXPLANATION_PROMPT_VERSION,
        "protocol_id": EXPLANATION_PROTOCOL_ID,
        "repair_rounds": 0,
        "temperature": 0,
    })
    return protocol


def _index_unique(records, key, label):
    result = {}
    for record in records:
        value = record.get(key)
        if not value:
            raise ValueError("{} record is missing {}".format(label, key))
        if value in result:
            raise ValueError("duplicate {} {}: {}".format(label, key, value))
        result[value] = record
    return result


def build_explanation_inputs(
    prompts_path,
    package_root,
    diagnostics_root,
    output_root,
    model_alias="qwen3.8-chat",
    expected_count=None,
    base_protocol_path=None,
):
    prompts = _index_unique(_load_jsonl(prompts_path), "task_id", "prompt")
    manifest = _load_json(os.path.join(package_root, "manifest.json"))
    entries = [
        entry for entry in manifest.get("entries", [])
        if entry.get("model_alias") == model_alias
    ]
    entries.sort(key=lambda item: item["task_id"])
    records = []
    plan_entries = []
    for entry in entries:
        task_id = entry["task_id"]
        evaluation_path = os.path.join(
            diagnostics_root,
            "evaluation",
            model_alias,
            task_id,
            "evaluation.json",
        )
        if not os.path.isfile(evaluation_path):
            raise ValueError("missing evaluation: {}".format(evaluation_path))
        evaluation = _load_json(evaluation_path)
        if evaluation.get("semantic_success") is True:
            continue
        if task_id not in prompts:
            raise ValueError("missing grounded prompt: {}".format(task_id))
        original = prompts[task_id]
        program_path = _safe_package_path(package_root, entry.get("program_path"))
        previous_program = _read_required_text(program_path, "previous program")
        expected_hash = entry.get("program_sha256")
        if expected_hash and _sha256_file(program_path) != expected_hash:
            raise ValueError("program hash mismatch: {}".format(task_id))
        execution_path = os.path.join(
            diagnostics_root,
            "execution",
            model_alias,
            task_id,
            "execution.json",
        )
        execution = _load_json(execution_path)
        log_root = os.path.join(diagnostics_root, "logs", model_alias, task_id)
        stdout_text = _read_required_text(
            os.path.join(log_root, "stdout.log"), "stdout log"
        )
        stderr_text = _read_required_text(
            os.path.join(log_root, "stderr.log"), "stderr log"
        )
        observation = {
            "accepted": False,
            "exit_status": execution.get("exit_status"),
            "policy_violations": list(execution.get("policy_violations") or []),
            "status": execution.get("status"),
            "stderr": stderr_text,
            "stdout": stdout_text,
            "timed_out": bool(execution.get("timed_out")),
        }
        user_prompt = (
            "Original grounded task:\n{task}\n\n"
            "Previous program:\n```python\n{program}\n```\n\n"
            "Raw execution observation:\n{observation}\n\n"
            "The candidate was rejected by the hidden acceptance test. The "
            "binary rejection does not reveal a semantic mismatch type, "
            "location, expected value, or expected hash. Explain what the "
            "program does, identify faults that can be inferred from the task, "
            "program, and raw observation, and propose a repair plan."
        ).format(
            task=original["user_prompt"].rstrip(),
            program=previous_program.rstrip(),
            observation=json.dumps(observation, indent=2, sort_keys=True),
        )
        prompt = SYSTEM_PROMPT + "\n\n" + user_prompt
        prompt_hash = _sha256_text(prompt)
        record = {
            "grounding_sha256": original.get("grounding_sha256"),
            "prompt": prompt,
            "prompt_sha256": prompt_hash,
            "prompt_version": EXPLANATION_PROMPT_VERSION,
            "self_debugging": {
                "parent_program_hash": _sha256_file(program_path),
                "source_round": 0,
                "stage": "explanation",
            },
            "split": original["split"],
            "system_prompt": SYSTEM_PROMPT,
            "task_id": task_id,
            "user_prompt": user_prompt,
        }
        if record["grounding_sha256"] is None:
            record.pop("grounding_sha256")
        records.append(record)
        plan_entries.append({
            "parent_program_hash": record["self_debugging"]["parent_program_hash"],
            "prompt_sha256": prompt_hash,
            "task_id": task_id,
        })
    if expected_count is not None and len(records) != int(expected_count):
        raise ValueError(
            "expected {} failed records, found {}".format(expected_count, len(records))
        )
    _atomic_jsonl(os.path.join(output_root, model_alias + ".jsonl"), records)
    protocol = _build_protocol(base_protocol_path, model_alias)
    _atomic_json(os.path.join(output_root, "protocol.json"), protocol)
    plan = {
        "entry_count": len(plan_entries),
        "entries": plan_entries,
        "model_alias": model_alias,
        "prompt_version": EXPLANATION_PROMPT_VERSION,
        "schema_version": "1.0",
        "stage": "explanation",
    }
    _atomic_json(os.path.join(output_root, "explanation_plan.json"), plan)
    return {
        "model_alias": model_alias,
        "output_root": os.path.abspath(output_root),
        "record_count": len(records),
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--round0-package", required=True)
    parser.add_argument("--diagnostics-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--base-protocol")
    parser.add_argument("--model-alias", default="qwen3.8-chat")
    parser.add_argument("--expected-count", type=int)
    args = parser.parse_args(argv)
    result = build_explanation_inputs(
        args.prompts,
        args.round0_package,
        args.diagnostics_root,
        args.output_root,
        model_alias=args.model_alias,
        expected_count=args.expected_count,
        base_protocol_path=args.base_protocol,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

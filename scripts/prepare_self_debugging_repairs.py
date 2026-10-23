#!/usr/bin/env python3
"""Build leakage-controlled repair prompts from Self-Debugging explanations."""

from __future__ import print_function

import argparse
import hashlib
import json
import os


PROMPT_VERSION = "self-debugging-repair-v1"
PROTOCOL_ID = "self-debugging-repair-heldout-qwen-v1"
SYSTEM_PROMPT = (
    "You are repairing a Python 3.6 scientific data-access program using "
    "Self-Debugging. Use only the original grounded task, previous program, "
    "raw execution observation, and the self-debugging analysis supplied "
    "below. Return one complete corrected Python 3.6 program and nothing "
    "else. Preserve the requested output contract. Do not invent expected "
    "arrays, hashes, statistics, or numerical answers, and do not rely on "
    "hidden acceptance-test details."
)
EXPLANATION_SUFFIX = (
    "\n\nThe candidate was rejected by the hidden acceptance test. The "
    "binary rejection does not reveal a semantic mismatch type, location, "
    "expected value, or expected hash. Explain what the program does, "
    "identify faults that can be inferred from the task, program, and raw "
    "observation, and propose a repair plan."
)


def _sha256_text(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


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
                    "JSONL record must be an object: {}:{}".format(
                        path, line_number
                    )
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


def _validate_explanation(value, task_id):
    expected_keys = set(("explanation", "suspected_faults", "repair_plan"))
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise ValueError("invalid explanation schema: {}".format(task_id))
    if not isinstance(value["explanation"], str):
        raise ValueError("invalid explanation schema: {}".format(task_id))
    for key in ("suspected_faults", "repair_plan"):
        if not isinstance(value[key], list) or not all(
            isinstance(item, str) for item in value[key]
        ):
            raise ValueError("invalid explanation schema: {}".format(task_id))
    return value


def _parse_explanation_content(content, task_id):
    if not isinstance(content, str):
        raise ValueError("invalid explanation schema: {}".format(task_id))
    candidate = content.strip()
    if candidate.startswith("```json\n") and candidate.endswith("\n```"):
        candidate = candidate[len("```json\n"):-len("\n```")]
    try:
        value = json.loads(candidate)
    except (TypeError, ValueError):
        raise ValueError("invalid explanation schema: {}".format(task_id))
    return _validate_explanation(value, task_id)


def _build_protocol(base_protocol_path, model_alias):
    if base_protocol_path:
        protocol = _load_json(base_protocol_path)
    else:
        protocol = {
            "default_split": "pilot",
            "frozen": True,
            "generation_attempts": 1,
            "held_out_enabled": True,
            "max_network_retries": 3,
            "pilot_gate_status": "passed",
            "request_timeout_seconds": 600,
            "retry_backoff_seconds": [2, 4, 8],
            "schema_version": "2.0",
        }
    protocol.update({
        "frozen": True,
        "frozen_model_aliases": [model_alias],
        "generation_mode": "direct_zero_shot",
        "max_output_tokens": 8192,
        "prompt_version": PROMPT_VERSION,
        "protocol_id": PROTOCOL_ID,
        "repair_rounds": 0,
        "temperature": 0,
    })
    return protocol


def build_repair_inputs(
    explanation_prompts_path,
    explanation_generation_root,
    output_root,
    model_alias="qwen3.8-chat",
    expected_count=None,
    base_protocol_path=None,
):
    source_records = _load_jsonl(explanation_prompts_path)
    records = []
    plan_entries = []
    seen = set()
    for source in source_records:
        task_id = source.get("task_id")
        if not task_id or task_id in seen:
            raise ValueError("missing or duplicate task_id: {}".format(task_id))
        seen.add(task_id)
        metadata = source.get("self_debugging") or {}
        if metadata.get("stage") != "explanation":
            raise ValueError("invalid explanation source stage: {}".format(task_id))
        result_path = os.path.join(
            explanation_generation_root,
            model_alias,
            task_id,
            "attempt-1",
            "result.json",
        )
        if not os.path.isfile(result_path):
            raise ValueError("missing explanation result: {}".format(result_path))
        result = _load_json(result_path)
        identity = result.get("identity") or {}
        if (
            result.get("status") != "completed"
            or identity.get("task_id") != task_id
            or identity.get("model_alias") != model_alias
            or identity.get("prompt_hash") != source.get("prompt_sha256")
        ):
            raise ValueError("explanation identity mismatch: {}".format(task_id))
        if (result.get("extraction") or {}).get("status") != "success":
            raise ValueError("explanation extraction failed: {}".format(task_id))
        content = (result.get("normalized") or {}).get("content")
        explanation = _parse_explanation_content(content, task_id)

        source_user_prompt = source.get("user_prompt") or ""
        if not source_user_prompt.endswith(EXPLANATION_SUFFIX):
            raise ValueError("unexpected explanation prompt format: {}".format(task_id))
        visible_context = source_user_prompt[:-len(EXPLANATION_SUFFIX)]
        user_prompt = (
            "{context}\n\nSelf-debugging analysis:\n{analysis}\n\n"
            "Revise the previous program according to this analysis. Return "
            "the complete corrected Python 3.6 program only."
        ).format(
            context=visible_context.rstrip(),
            analysis=json.dumps(explanation, indent=2, sort_keys=True),
        )
        prompt = SYSTEM_PROMPT + "\n\n" + user_prompt
        explanation_hash = _sha256_text(content)
        record = {
            "prompt": prompt,
            "prompt_sha256": _sha256_text(prompt),
            "prompt_version": PROMPT_VERSION,
            "self_debugging": {
                "explanation_prompt_hash": source.get("prompt_sha256"),
                "explanation_response_hash": explanation_hash,
                "parent_program_hash": metadata.get("parent_program_hash"),
                "source_round": 0,
                "stage": "repair",
            },
            "split": source.get("split"),
            "system_prompt": SYSTEM_PROMPT,
            "task_id": task_id,
            "user_prompt": user_prompt,
        }
        if source.get("grounding_sha256") is not None:
            record["grounding_sha256"] = source["grounding_sha256"]
        records.append(record)
        plan_entries.append({
            "explanation_response_hash": explanation_hash,
            "parent_program_hash": metadata.get("parent_program_hash"),
            "prompt_sha256": record["prompt_sha256"],
            "task_id": task_id,
        })

    records.sort(key=lambda item: item["task_id"])
    plan_entries.sort(key=lambda item: item["task_id"])
    if expected_count is not None and len(records) != int(expected_count):
        raise ValueError(
            "expected {} explanation records, found {}".format(
                expected_count, len(records)
            )
        )
    _atomic_jsonl(os.path.join(output_root, model_alias + ".jsonl"), records)
    _atomic_json(
        os.path.join(output_root, "protocol.json"),
        _build_protocol(base_protocol_path, model_alias),
    )
    _atomic_json(os.path.join(output_root, "repair_plan.json"), {
        "entry_count": len(plan_entries),
        "entries": plan_entries,
        "model_alias": model_alias,
        "prompt_version": PROMPT_VERSION,
        "schema_version": "1.0",
        "stage": "repair",
    })
    return {
        "model_alias": model_alias,
        "output_root": os.path.abspath(output_root),
        "record_count": len(records),
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--explanation-prompts", required=True)
    parser.add_argument("--explanation-generation-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--base-protocol")
    parser.add_argument("--model-alias", default="qwen3.8-chat")
    parser.add_argument("--expected-count", type=int)
    args = parser.parse_args(argv)
    result = build_repair_inputs(
        args.explanation_prompts,
        args.explanation_generation_root,
        args.output_root,
        model_alias=args.model_alias,
        expected_count=args.expected_count,
        base_protocol_path=args.base_protocol,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

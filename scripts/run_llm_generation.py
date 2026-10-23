#!/usr/bin/env python3
"""Run resumable Primary Zero-shot generation over an explicit task split."""

from __future__ import print_function

import argparse
import hashlib
import json
import os
import sys

try:
    from scripts.llm_generation_store import (
        extract_program,
        is_complete,
        result_directory,
        write_result,
    )
    from scripts.llm_runner_config import (
        load_models,
        load_protocol,
        protocol_sha256,
    )
    from scripts.openai_compatible_client import (
        build_request,
        normalize_response,
        request_chat_completion,
    )
except ImportError:
    from llm_generation_store import (
        extract_program,
        is_complete,
        result_directory,
        write_result,
    )
    from llm_runner_config import load_models, load_protocol, protocol_sha256
    from openai_compatible_client import (
        build_request,
        normalize_response,
        request_chat_completion,
    )


class RunnerError(ValueError):
    pass


def _sha256_text(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_prompt_records(path):
    records = []
    with open(path, "r") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except ValueError as error:
                raise RunnerError(
                    "{}:{} invalid JSON: {}".format(path, line_number, error)
                )
            if not isinstance(record, dict):
                raise RunnerError("{}:{} record must be an object".format(path, line_number))
            required = (
                "task_id",
                "split",
                "prompt_version",
                "system_prompt",
                "user_prompt",
                "prompt",
                "prompt_sha256",
            )
            missing = [key for key in required if key not in record]
            if missing:
                raise RunnerError(
                    "{}:{} missing fields: {}".format(
                        path, line_number, ", ".join(missing)
                    )
                )
            composed = record["system_prompt"] + "\n\n" + record["user_prompt"]
            if record["prompt"] != composed:
                raise RunnerError("{} prompt composition mismatch".format(record["task_id"]))
            if record["prompt_sha256"] != _sha256_text(record["prompt"]):
                raise RunnerError("{} prompt hash mismatch".format(record["task_id"]))
            records.append(record)
    task_ids = [record["task_id"] for record in records]
    if len(task_ids) != len(set(task_ids)):
        raise RunnerError("duplicate task IDs in prompt file")
    return records


def select_records(records, split, protocol):
    if split not in ("pilot", "held_out"):
        raise RunnerError("unsupported split: {}".format(split))
    if split == "held_out" and not protocol.get("held_out_enabled", False):
        raise RunnerError("held-out execution is disabled by the protocol")
    selected = [record for record in records if record["split"] == split]
    if not selected:
        raise RunnerError("no records found for split {}".format(split))
    for record in selected:
        if record["prompt_version"] != protocol["prompt_version"]:
            raise RunnerError(
                "{} prompt version does not match protocol".format(record["task_id"])
            )
    return selected


def build_plan(
    records,
    models,
    protocol,
    results_dir,
    split,
    model_aliases=None,
    limit=None,
    task_ids=None,
):
    selected_records = select_records(records, split, protocol)
    if task_ids is not None:
        if not task_ids:
            raise RunnerError("task_ids must not be empty")
        if len(task_ids) != len(set(task_ids)):
            raise RunnerError("duplicate task IDs")
        records_by_id = {record["task_id"]: record for record in records}
        unknown = [task_id for task_id in task_ids if task_id not in records_by_id]
        if unknown:
            raise RunnerError("unknown task IDs: {}".format(", ".join(unknown)))
        wrong_split = [
            task_id
            for task_id in task_ids
            if records_by_id[task_id]["split"] != split
        ]
        if wrong_split:
            raise RunnerError(
                "task IDs are not in split {}: {}".format(
                    split, ", ".join(wrong_split)
                )
            )
        selected_records = [records_by_id[task_id] for task_id in task_ids]
    if limit is not None:
        if limit <= 0:
            raise RunnerError("limit must be positive")
        selected_records = selected_records[:limit]

    aliases = list(model_aliases) if model_aliases else list(models.keys())
    unknown = [alias for alias in aliases if alias not in models]
    if unknown:
        raise RunnerError("unknown model aliases: {}".format(", ".join(unknown)))
    if len(aliases) != len(set(aliases)):
        raise RunnerError("duplicate model aliases")
    if split == "held_out":
        frozen_aliases = set(protocol.get("frozen_model_aliases", []))
        outside_roster = [alias for alias in aliases if alias not in frozen_aliases]
        if outside_roster:
            raise RunnerError(
                "models are outside the frozen held-out roster: {}".format(
                    ", ".join(outside_roster)
                )
            )

    protocol_hash = protocol_sha256(protocol)
    plan = []
    for record in selected_records:
        for alias in aliases:
            model_config = models[alias]
            payload = build_request(
                model_config,
                protocol,
                record["system_prompt"],
                record["user_prompt"],
            )
            identity = {
                "attempt": 1,
                "model_alias": alias,
                "prompt_hash": record["prompt_sha256"],
                "protocol_hash": protocol_hash,
                "task_id": record["task_id"],
            }
            if record.get("grounding_sha256"):
                identity["grounding_hash"] = record["grounding_sha256"]
            request_snapshot = {
                "endpoint": model_config["base_url"].rstrip("/") + "/chat/completions",
                "model_alias": alias,
                "payload": payload,
                "prompt_sha256": record["prompt_sha256"],
                "prompt_version": record["prompt_version"],
                "protocol_sha256": protocol_hash,
            }
            if record.get("grounding_sha256"):
                request_snapshot["grounding_sha256"] = record["grounding_sha256"]
            plan.append({
                "identity": identity,
                "model_alias": alias,
                "model_config": model_config,
                "output_directory": result_directory(results_dir, identity),
                "payload": payload,
                "request_snapshot": request_snapshot,
                "task_id": record["task_id"],
            })
    return plan


def _atomic_json(path, document):
    temporary = path + ".tmp"
    with open(temporary, "w") as handle:
        json.dump(document, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_failure(
    directory,
    item,
    error,
    raw_response=None,
    transport=None,
):
    if not os.path.isdir(directory):
        os.makedirs(directory)
    _atomic_json(os.path.join(directory, "request.json"), item["request_snapshot"])
    if raw_response is not None:
        _atomic_json(os.path.join(directory, "response.raw.json"), raw_response)
    if transport is not None:
        _atomic_json(os.path.join(directory, "transport.json"), transport)
    document = {
        "error_message": str(error),
        "error_type": type(error).__name__,
        "identity": item["identity"],
        "status": "failed",
    }
    _atomic_json(os.path.join(directory, "failure.json"), document)


def run_generation(
    records,
    models,
    protocol,
    results_dir,
    split=None,
    model_aliases=None,
    limit=None,
    task_ids=None,
    dry_run=False,
    rerun=False,
    environ=None,
    client_fn=request_chat_completion,
    progress_fn=None,
):
    if environ is None:
        environ = os.environ
    if split is None:
        split = protocol["default_split"]
    plan = build_plan(
        records,
        models,
        protocol,
        results_dir,
        split,
        model_aliases=model_aliases,
        limit=limit,
        task_ids=task_ids,
    )
    needed_env = sorted(set(
        item["model_config"]["api_key_env"] for item in plan
    ))
    missing = [name for name in needed_env if not environ.get(name)]
    summary = {
        "completed": 0,
        "dry_run": bool(dry_run),
        "extraction_failed": 0,
        "failed": 0,
        "failures": [],
        "missing_credentials": missing,
        "planned": len(plan),
        "skipped": 0,
        "split": split,
    }
    if dry_run:
        summary["planned_calls"] = [
            {
                "model_alias": item["model_alias"],
                "output_directory": item["output_directory"],
                "task_id": item["task_id"],
            }
            for item in plan
        ]
        return summary

    for item_index, item in enumerate(plan, 1):
        directory = item["output_directory"]
        if not rerun and is_complete(directory, item["identity"]):
            summary["skipped"] += 1
            if progress_fn:
                progress_fn({
                    "event": "request_skipped",
                    "index": item_index,
                    "model_alias": item["model_alias"],
                    "task_id": item["task_id"],
                    "total": len(plan),
                })
            continue
        key_env = item["model_config"]["api_key_env"]
        api_key = environ.get(key_env)
        if not api_key:
            error = RunnerError("missing credential environment: {}".format(key_env))
            _atomic_failure(directory, item, error)
            summary["failed"] += 1
            summary["failures"].append({
                "error": str(error),
                "model_alias": item["model_alias"],
                "task_id": item["task_id"],
            })
            continue
        raw_response = None
        transport = None
        try:
            if progress_fn:
                progress_fn({
                    "event": "request_start",
                    "index": item_index,
                    "model_alias": item["model_alias"],
                    "task_id": item["task_id"],
                    "timeout_seconds": protocol["request_timeout_seconds"],
                    "total": len(plan),
                    "transport": item["model_config"].get("transport", "urllib"),
                })
            raw_response, transport = client_fn(
                item["model_config"],
                protocol,
                item["payload"],
                api_key,
            )
            normalized = normalize_response(
                raw_response, item["model_config"]["model"]
            )
            extraction = extract_program(normalized["content"])
            write_result(
                results_dir,
                item["identity"],
                item["request_snapshot"],
                raw_response,
                normalized,
                extraction,
                transport,
            )
            summary["completed"] += 1
            if extraction["status"] != "success":
                summary["extraction_failed"] += 1
            if progress_fn:
                progress_fn({
                    "attempts": transport.get("attempts"),
                    "elapsed_seconds": transport.get("elapsed_seconds"),
                    "event": "request_complete",
                    "extraction_status": extraction["status"],
                    "index": item_index,
                    "model_alias": item["model_alias"],
                    "task_id": item["task_id"],
                    "total": len(plan),
                })
        except Exception as error:
            _atomic_failure(
                directory,
                item,
                error,
                raw_response=raw_response,
                transport=transport,
            )
            summary["failed"] += 1
            summary["failures"].append({
                "error": str(error),
                "model_alias": item["model_alias"],
                "task_id": item["task_id"],
            })
            if progress_fn:
                progress_fn({
                    "error": str(error),
                    "event": "request_failed",
                    "index": item_index,
                    "model_alias": item["model_alias"],
                    "task_id": item["task_id"],
                    "total": len(plan),
                })
    return summary


def _print_progress(event):
    print(json.dumps(event, sort_keys=True), file=sys.stderr, flush=True)


def _default_paths():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return {
        "models": os.path.join(root, "experiments", "config", "models.json"),
        "prompts": os.path.join(root, "benchmark", "gold138_primary_zero_shot.jsonl"),
        "protocol": os.path.join(root, "experiments", "config", "protocol_v1.json"),
        "results": os.path.join(
            root, "experiments", "results", "pilot", "primary-zero-shot-v1"
        ),
    }


def parse_args(argv=None):
    defaults = _default_paths()
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", default=defaults["prompts"])
    parser.add_argument("--models-config", default=defaults["models"])
    parser.add_argument("--protocol", default=defaults["protocol"])
    parser.add_argument("--results-dir", default=defaults["results"])
    parser.add_argument("--split")
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--task-ids", nargs="+")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--rerun", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    protocol = load_protocol(args.protocol)
    summary = run_generation(
        load_prompt_records(args.prompts),
        load_models(args.models_config),
        protocol,
        args.results_dir,
        split=args.split,
        model_aliases=args.models,
        limit=args.limit,
        task_ids=args.task_ids,
        dry_run=args.dry_run,
        rerun=args.rerun,
        progress_fn=_print_progress,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

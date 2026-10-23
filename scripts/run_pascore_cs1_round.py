#!/usr/bin/env python3
"""Run one PA-SCoRe execution, evaluation, diagnosis, and feedback round."""

from __future__ import print_function

import argparse
import json
import os
import sys

try:
    from scripts.evaluate_generated_batch import evaluate_batch
    from scripts.execute_generated_batch import execute_manifest
    from scripts.failure_diagnosis import diagnose_failure
    from scripts.pascore_contracts import (
        atomic_json,
        load_json,
        sha256_file,
        sha256_json,
        validate_condition,
    )
    from scripts.structured_feedback import build_feedback
except ImportError:
    from evaluate_generated_batch import evaluate_batch
    from execute_generated_batch import execute_manifest
    from failure_diagnosis import diagnose_failure
    from pascore_contracts import (
        atomic_json,
        load_json,
        sha256_file,
        sha256_json,
        validate_condition,
    )
    from structured_feedback import build_feedback


def _optional_json(path):
    return load_json(path) if os.path.isfile(path) else None


def _program_source(package_root, entry):
    path = entry.get("program_path")
    if not path:
        return ""
    absolute = os.path.abspath(os.path.join(package_root, path))
    root = os.path.abspath(package_root) + os.sep
    if not absolute.startswith(root):
        raise ValueError("program path escapes package")
    with open(absolute, "r") as handle:
        return handle.read()


def _grounding(package_root, task_id):
    path = os.path.join(package_root, "grounding", task_id, "grounding.json")
    if not os.path.isfile(path):
        raise ValueError("grounding is missing: {}".format(task_id))
    return load_json(path)


def _remove_stale_feedback(feedback_dir, model_alias, task_id):
    path = os.path.join(feedback_dir, model_alias, task_id, "feedback.json")
    try:
        os.remove(path)
    except OSError:
        pass


def verify_execution_package(package_root):
    manifest = load_json(os.path.join(package_root, "manifest.json"))
    if manifest.get("package_type") != "pascore_execution_round":
        raise ValueError("not a PA-SCoRe execution package")
    condition = validate_condition(load_json(os.path.join(package_root, "condition.json")))
    protocol = load_json(os.path.join(package_root, "protocol.json"))
    if sha256_json(condition) != manifest.get("condition_sha256"):
        raise ValueError("condition hash mismatch")
    if sha256_json(protocol) != manifest.get("protocol_sha256"):
        raise ValueError("protocol hash mismatch")
    if condition["condition_id"] != manifest.get("condition"):
        raise ValueError("condition binding mismatch")
    for relative, expected_hash in (manifest.get("runtime_sha256") or {}).items():
        path = os.path.abspath(os.path.join(package_root, relative))
        if not path.startswith(os.path.abspath(package_root) + os.sep):
            raise ValueError("runtime path escapes package")
        if not os.path.isfile(path) or sha256_file(path) != expected_hash:
            raise ValueError("runtime hash mismatch: {}".format(relative))
    task_manifest = load_json(os.path.join(package_root, "task_manifest.json"))
    task_by_id = dict((task["id"], task) for task in task_manifest.get("tasks", []))
    for entry in manifest.get("entries", []):
        task = task_by_id.get(entry["task_id"])
        identity = entry.get("pascore_identity") or {}
        if task is None or identity.get("task_spec_hash") != task.get("spec_sha256"):
            raise ValueError("task identity mismatch: {}".format(entry["task_id"]))
        if identity.get("condition") != manifest.get("condition"):
            raise ValueError("entry condition mismatch")
        if identity.get("round") != manifest.get("round"):
            raise ValueError("entry round mismatch")
        grounding = _grounding(package_root, entry["task_id"])
        if grounding.get("enabled") is not condition["metadata_grounding"]:
            raise ValueError("entry grounding mode mismatch: {}".format(entry["task_id"]))
        expected_grounding_hash = (
            grounding.get("grounding_sha256")
            if condition["metadata_grounding"] else None
        )
        if expected_grounding_hash != entry.get("grounding_sha256"):
            raise ValueError("entry grounding mismatch: {}".format(entry["task_id"]))
    return manifest, condition


def run_round(
    package_root,
    output_root,
    gold_dir,
    reference_index_path,
    python_executable=None,
    timeout_seconds=90,
    max_memory_mib=8192,
    max_output_mib=2,
    evaluation_mode="strict",
    resume=False,
    progress_fn=None,
):
    manifest_path = os.path.join(package_root, "manifest.json")
    task_manifest_path = os.path.join(package_root, "task_manifest.json")
    protocol_path = os.path.join(package_root, "protocol.json")
    manifest, condition = verify_execution_package(package_root)
    entries = manifest.get("entries") or []
    models = sorted(set(entry["model_alias"] for entry in entries))
    task_ids = []
    seen = set()
    for entry in entries:
        if entry["task_id"] not in seen:
            seen.add(entry["task_id"])
            task_ids.append(entry["task_id"])

    execution_dir = os.path.join(output_root, "execution")
    guard_dir = os.path.join(output_root, "guard")
    evaluation_dir = os.path.join(output_root, "evaluation")
    diagnosis_dir = os.path.join(output_root, "diagnosis")
    feedback_dir = os.path.join(output_root, "feedback")
    execution_summary = execute_manifest(
        manifest_path,
        execution_dir,
        timeout_seconds=timeout_seconds,
        python_executable=python_executable or sys.executable,
        max_memory_mib=max_memory_mib,
        max_output_mib=max_output_mib,
        resume=resume,
        progress_fn=progress_fn,
        task_manifest_path=task_manifest_path,
        guard_output_root=guard_dir,
        resource_guard_enabled=condition["resource_guard"],
    )
    evaluation_summary = evaluate_batch(
        execution_dir,
        gold_dir,
        reference_index_path,
        task_manifest_path,
        evaluation_dir,
        models,
        task_ids,
        protocol_path,
        package_manifest_path=manifest_path,
        evaluation_mode=evaluation_mode,
    )

    feedback_ready = 0
    semantic_success = 0
    infrastructure_retry = 0
    terminal_failure = 0
    for entry in entries:
        model_alias = entry["model_alias"]
        task_id = entry["task_id"]
        execution = load_json(os.path.join(
            execution_dir, model_alias, task_id, "execution.json"
        ))
        evaluation = load_json(os.path.join(
            evaluation_dir, model_alias, task_id, "evaluation.json"
        ))
        guard = _optional_json(os.path.join(
            guard_dir, model_alias, task_id, "guard.json"
        ))
        diagnosis = diagnose_failure(
            execution,
            evaluation,
            guard=guard,
            program_source=_program_source(package_root, entry),
            enabled=condition["failure_diagnosis"] == "multi_grained",
        )
        atomic_json(
            os.path.join(diagnosis_dir, model_alias, task_id, "diagnosis.json"),
            diagnosis,
        )
        if evaluation.get("semantic_success") is True:
            _remove_stale_feedback(feedback_dir, model_alias, task_id)
            semantic_success += 1
            continue
        if diagnosis.get("major_category") == "infrastructure":
            _remove_stale_feedback(feedback_dir, model_alias, task_id)
            infrastructure_retry += 1
            continue
        identity = entry.get("pascore_identity") or {
            "run_id": manifest.get("run_id"),
            "task_id": task_id,
            "model_alias": model_alias,
            "condition": condition["condition_id"],
            "round": manifest.get("round"),
        }
        remaining_rounds = max(
            0,
            int(condition["max_repair_rounds"]) - int(manifest.get("round", 0)),
        )
        if remaining_rounds <= 0:
            _remove_stale_feedback(feedback_dir, model_alias, task_id)
            terminal_failure += 1
            continue
        feedback = build_feedback(
            identity,
            diagnosis,
            remaining_rounds=remaining_rounds,
            structured=condition["structured_feedback"],
            timeout_seconds=timeout_seconds,
            max_memory_mib=max_memory_mib,
            grounding=_grounding(package_root, task_id),
        )
        atomic_json(
            os.path.join(feedback_dir, model_alias, task_id, "feedback.json"),
            feedback,
        )
        feedback_ready += 1

    summary = {
        "schema_version": "1.0",
        "run_id": manifest.get("run_id"),
        "round": manifest.get("round"),
        "condition": condition["condition_id"],
        "entry_count": len(entries),
        "execution": execution_summary,
        "evaluation": evaluation_summary,
        "feedback_ready": feedback_ready,
        "semantic_success": semantic_success,
        "infrastructure_retry": infrastructure_retry,
        "terminal_failure": terminal_failure,
    }
    atomic_json(os.path.join(output_root, "round_summary.json"), summary)
    return summary


def _print_progress(event):
    print(json.dumps(event, sort_keys=True), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", default=".")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gold-dir", required=True)
    parser.add_argument("--reference-index", required=True)
    parser.add_argument("--timeout-seconds", type=int, default=90)
    parser.add_argument("--max-memory-mib", type=int, default=8192)
    parser.add_argument("--max-output-mib", type=int, default=2)
    parser.add_argument(
        "--evaluation-mode", choices=("strict", "diagnostic-v2"), default="strict"
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    summary = run_round(
        os.path.abspath(args.package_root),
        args.output_dir,
        args.gold_dir,
        args.reference_index,
        timeout_seconds=args.timeout_seconds,
        max_memory_mib=args.max_memory_mib,
        max_output_mib=args.max_output_mib,
        evaluation_mode=args.evaluation_mode,
        resume=args.resume,
        progress_fn=_print_progress,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

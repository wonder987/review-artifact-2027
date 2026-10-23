#!/usr/bin/env python3
"""Build a deterministic PA-SCoRe execution-round package for cs1."""

from __future__ import print_function

import argparse
import json
import os
import zipfile

try:
    from scripts.metadata_grounding import build_grounding
    from scripts.package_generation_batch import (
        _canonical_json_bytes,
        _read_bytes,
        _zip_info,
        collect_entries,
    )
    from scripts.pascore_contracts import (
        build_identity,
        load_json,
        sha256_bytes,
        sha256_json,
        validate_condition,
    )
except ImportError:
    from metadata_grounding import build_grounding
    from package_generation_batch import (
        _canonical_json_bytes,
        _read_bytes,
        _zip_info,
        collect_entries,
    )
    from pascore_contracts import (
        build_identity,
        load_json,
        sha256_bytes,
        sha256_json,
        validate_condition,
    )


RUNTIME_NAMES = (
    "execute_generated_batch.py",
    "semantic_evaluator.py",
    "evaluate_generated_batch.py",
    "pascore_contracts.py",
    "resource_aware_guard.py",
    "failure_taxonomy.py",
    "failure_diagnosis.py",
    "structured_feedback.py",
    "run_pascore_cs1_round.py",
    "package_pascore_diagnostics.py",
)


def _runtime_paths():
    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    return dict(
        (name, os.path.join(scripts_dir, name)) for name in RUNTIME_NAMES
    )


def _task_subset(task_manifest, task_ids):
    task_ids = set(task_ids)
    tasks = [
        task for task in task_manifest.get("tasks", []) if task.get("id") in task_ids
    ]
    found = set(task["id"] for task in tasks)
    missing = sorted(task_ids - found)
    if missing:
        raise ValueError("task manifest is missing: {}".format(", ".join(missing)))
    return {
        "schema_version": task_manifest.get("schema_version"),
        "benchmark_name": task_manifest.get("benchmark_name"),
        "task_count": len(tasks),
        "tasks": tasks,
    }


def build_execution_package(
    results_root,
    output_zip,
    task_manifest_path,
    protocol_path,
    condition_path,
    run_id,
    round_index,
    allowed_models=None,
    allowed_task_ids=None,
    runtime_paths=None,
    parent_manifest_path=None,
    split=None,
    repair_plan_path=None,
):
    condition = validate_condition(load_json(condition_path))
    protocol = load_json(protocol_path)
    complete_task_manifest = load_json(task_manifest_path)
    repair_plan_by_key = {}
    if repair_plan_path:
        repair_plan = load_json(repair_plan_path)
        if repair_plan.get("run_id") != str(run_id):
            raise ValueError("repair plan run binding mismatch")
        if repair_plan.get("condition") != condition["condition_id"]:
            raise ValueError("repair plan condition binding mismatch")
        if int(repair_plan.get("round", -1)) != int(round_index):
            raise ValueError("repair plan round binding mismatch")
        repair_plan_by_key = dict(
            ((entry["model_alias"], entry["task_id"]), entry)
            for entry in repair_plan.get("entries", [])
        )
        expected = set(repair_plan_by_key)
        allowed_models = set(key[0] for key in expected)
        allowed_task_ids = set(key[1] for key in expected)
    else:
        if not allowed_models:
            raise ValueError("at least one model is required")
        allowed_models = set(allowed_models)
        if allowed_task_ids is None:
            if split is None:
                raise ValueError("task IDs or a split are required")
            allowed_task_ids = set(
                task["id"] for task in complete_task_manifest.get("tasks", [])
                if task.get("split") == split
            )
        else:
            allowed_task_ids = set(allowed_task_ids)
        expected = set(
            (model_alias, task_id)
            for model_alias in allowed_models
            for task_id in allowed_task_ids
        )
    if not expected:
        raise ValueError("package scope is empty")
    task_manifest = _task_subset(complete_task_manifest, allowed_task_ids)
    task_by_id = dict((task["id"], task) for task in task_manifest["tasks"])
    groundings = dict(
        (task_id, build_grounding(
            task_by_id[task_id], enabled=condition["metadata_grounding"]
        ))
        for task_id in sorted(allowed_task_ids)
    )
    parent_by_key = {}
    if int(round_index) > 0:
        if not parent_manifest_path:
            raise ValueError("repair rounds require a parent execution manifest")
        parent_manifest = load_json(parent_manifest_path)
        if parent_manifest.get("run_id") != str(run_id):
            raise ValueError("parent run binding mismatch")
        if parent_manifest.get("condition") != condition["condition_id"]:
            raise ValueError("parent condition binding mismatch")
        if int(parent_manifest.get("round", -1)) != int(round_index) - 1:
            raise ValueError("parent round binding mismatch")
        parent_by_key = dict(
            ((entry["model_alias"], entry["task_id"]), entry)
            for entry in parent_manifest.get("entries", [])
        )
    collected = collect_entries(
        results_root,
        allowed_models,
        allowed_task_ids=allowed_task_ids,
    )
    if repair_plan_by_key:
        collected = [
            entry for entry in collected
            if (entry["model_alias"], entry["task_id"]) in expected
        ]
    actual = set(
        (entry["model_alias"], entry["task_id"]) for entry in collected
    )
    missing = sorted(expected - actual)
    if missing:
        raise ValueError(
            "missing terminal generation artifacts: {}".format(
                ", ".join("{}/{}".format(*item) for item in missing)
            )
        )

    protocol_hash = sha256_json(protocol)
    manifest_entries = []
    for entry in collected:
        task = task_by_id[entry["task_id"]]
        expected_grounding_hash = (
            groundings[entry["task_id"]]["grounding_sha256"]
            if condition["metadata_grounding"]
            else None
        )
        if entry.get("grounding_sha256") != expected_grounding_hash:
            raise ValueError("grounding hash mismatch: {}/{}".format(
                entry["model_alias"], entry["task_id"]
            ))
        artifact_hash = entry.get("program_sha256") or entry.get("failure_sha256")
        parent_program_hash = None
        if int(round_index) > 0:
            parent = parent_by_key.get((entry["model_alias"], entry["task_id"]))
            if parent is None:
                raise ValueError("parent entry is missing: {}/{}".format(
                    entry["model_alias"], entry["task_id"]
                ))
            parent_program_hash = (
                (parent.get("pascore_identity") or {}).get("program_hash")
                or parent.get("program_sha256")
            )
            if not parent_program_hash:
                raise ValueError("parent program hash is missing")
            planned_parent = (
                repair_plan_by_key.get(
                    (entry["model_alias"], entry["task_id"]), {}
                ).get("parent_program_hash")
            )
            if planned_parent and parent_program_hash != planned_parent:
                raise ValueError("repair plan parent hash mismatch")
        manifest_entry = dict(
            (key, value) for key, value in entry.items() if not key.startswith("_")
        )
        manifest_entry["pascore_identity"] = build_identity(
            run_id,
            entry["task_id"],
            entry["model_alias"],
            condition,
            int(round_index),
            task["spec_sha256"],
            protocol_hash,
            artifact_hash,
            parent_program_hash=parent_program_hash,
        )
        manifest_entries.append(manifest_entry)

    runtime_paths = runtime_paths or _runtime_paths()
    missing_runtime = [
        name for name in RUNTIME_NAMES
        if name not in runtime_paths or not os.path.isfile(runtime_paths[name])
    ]
    if missing_runtime:
        raise ValueError("missing PA-SCoRe runtime: {}".format(", ".join(missing_runtime)))
    runtime_hashes = dict(
        (name, sha256_bytes(_read_bytes(runtime_paths[name])))
        for name in RUNTIME_NAMES
    )
    manifest = {
        "schema_version": "3.0",
        "package_type": "pascore_execution_round",
        "run_id": str(run_id),
        "round": int(round_index),
        "condition": condition["condition_id"],
        "condition_sha256": sha256_json(condition),
        "protocol_sha256": protocol_hash,
        "entry_count": len(manifest_entries),
        "entries": manifest_entries,
        "runtime_sha256": runtime_hashes,
    }
    output_directory = os.path.dirname(os.path.abspath(output_zip))
    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)
    temporary = output_zip + ".tmp"
    try:
        with zipfile.ZipFile(temporary, "w") as archive:
            archive.writestr(_zip_info("manifest.json"), _canonical_json_bytes(manifest))
            archive.writestr(
                _zip_info("task_manifest.json"), _canonical_json_bytes(task_manifest)
            )
            archive.writestr(_zip_info("protocol.json"), _canonical_json_bytes(protocol))
            archive.writestr(_zip_info("condition.json"), _canonical_json_bytes(condition))
            for task_id, grounding in groundings.items():
                archive.writestr(
                    _zip_info("grounding/{}/grounding.json".format(task_id)),
                    _canonical_json_bytes(grounding),
                )
            for name in RUNTIME_NAMES:
                archive.writestr(_zip_info(name), _read_bytes(runtime_paths[name]))
            for entry in collected:
                if entry.get("entry_type") == "generation_failure":
                    archive.writestr(
                        _zip_info(entry["failure_path"]),
                        _read_bytes(entry["_source_failure_path"]),
                    )
                else:
                    archive.writestr(
                        _zip_info(entry["program_path"]),
                        _read_bytes(entry["_source_program_path"]),
                    )
        os.replace(temporary, output_zip)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--task-manifest", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--task-ids", nargs="+")
    parser.add_argument("--split", choices=("pilot", "held_out"))
    parser.add_argument("--parent-manifest")
    parser.add_argument("--repair-plan")
    args = parser.parse_args(argv)
    manifest = build_execution_package(
        args.results_root,
        args.output,
        args.task_manifest,
        args.protocol,
        args.condition,
        args.run_id,
        args.round,
        set(args.models) if args.models else None,
        set(args.task_ids) if args.task_ids else None,
        parent_manifest_path=args.parent_manifest,
        split=args.split,
        repair_plan_path=args.repair_plan,
    )
    print(json.dumps({
        "entry_count": manifest["entry_count"],
        "output": os.path.abspath(args.output),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

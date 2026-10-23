#!/usr/bin/env python3
"""Evaluate controlled program executions against stable Gold references."""

from __future__ import print_function

import argparse
import collections
import hashlib
import json
import os

try:
    from scripts.semantic_evaluator import EVALUATOR_VERSION, evaluate_record
except ImportError:
    from semantic_evaluator import EVALUATOR_VERSION, evaluate_record


def _read_json(path):
    with open(path, "r") as handle:
        return json.load(handle)


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _infrastructure(model_alias, task_id, code, evaluation_mode="strict"):
    result = {
        "evaluator_version": EVALUATOR_VERSION,
        "model_alias": model_alias,
        "task_id": task_id,
        "category": "infrastructure_failure",
        "code": code,
        "execution_success": False,
        "semantic_success": False,
        "mismatches": [],
        "diagnostics": [],
    }
    if evaluation_mode == "diagnostic-v2":
        result["normalizations"] = []
    return result


def _gold_for_task(gold_dir, reference_index, task_id):
    entry = reference_index.get("tasks", {}).get(task_id)
    if not isinstance(entry, dict):
        return None, "missing_gold_index"
    stability = entry.get("stability", {})
    if stability.get("stable") is not True:
        return None, "unstable_gold"
    repeat_files = entry.get("repeat_files")
    if not isinstance(repeat_files, list) or not repeat_files:
        return None, "missing_gold"
    path = os.path.abspath(os.path.join(gold_dir, repeat_files[0]))
    gold_root = os.path.abspath(gold_dir) + os.sep
    if not path.startswith(gold_root) or not os.path.isfile(path):
        return None, "missing_gold"
    try:
        value = _read_json(path)
    except (IOError, OSError, ValueError):
        return None, "corrupt_gold"
    if value.get("task_id") != task_id or value.get("status") != "success":
        return None, "invalid_gold"
    return value, None


def _execution_for_task(execution_dir, model_alias, task_id):
    path = os.path.join(
        execution_dir, model_alias, task_id, "execution.json"
    )
    if not os.path.isfile(path):
        return None, "missing_execution"
    try:
        value = _read_json(path)
    except (IOError, OSError, ValueError):
        return None, "corrupt_execution"
    if value.get("model_alias") != model_alias or value.get("task_id") != task_id:
        return None, "execution_binding_mismatch"
    return value, None


def _increment(mapping, key):
    mapping[key] = mapping.get(key, 0) + 1


def _group_summary(records):
    summary = {
        "total": len(records),
        "semantic_success": sum(r["semantic_success"] for r in records),
        "infrastructure_failure": sum(
            r["category"] == "infrastructure_failure" for r in records
        ),
        "generation_failure": sum(
            r["category"] == "generation_failure" for r in records
        ),
        "execution_failure": sum(
            r["category"] == "execution_failure" for r in records
        ),
        "protocol_failure": sum(
            r["category"] == "protocol_failure" for r in records
        ),
        "semantic_failure": sum(
            r["category"] == "semantic_failure" for r in records
        ),
        "by_category": {},
        "by_code": {},
        "by_model": {},
        "by_level": {},
        "by_subtype": {},
    }
    summary["model_evaluable"] = sum(
        r["category"] != "infrastructure_failure" for r in records
    )
    summary["semantic_accuracy"] = (
        float(summary["semantic_success"]) / summary["model_evaluable"]
        if summary["model_evaluable"] else None
    )
    for record in records:
        _increment(summary["by_category"], record["category"])
        _increment(summary["by_code"], record["code"])
        for group_name, key in (
            ("by_model", record["model_alias"]),
            ("by_level", record.get("level")),
            ("by_subtype", record.get("subtype")),
        ):
            group = summary[group_name].setdefault(
                str(key), {"total": 0, "semantic_success": 0}
            )
            group["total"] += 1
            group["semantic_success"] += int(record["semantic_success"])
            group["model_evaluable"] = group.get("model_evaluable", 0) + int(
                record["category"] != "infrastructure_failure"
            )
    for grouping in ("by_model", "by_level", "by_subtype"):
        for group in summary[grouping].values():
            group["semantic_accuracy"] = (
                float(group["semantic_success"]) / group["model_evaluable"]
                if group["model_evaluable"] else None
            )
    return summary


def evaluate_batch(
    execution_dir,
    gold_dir,
    reference_index_path,
    manifest_path,
    output_dir,
    models,
    task_ids,
    protocol_path,
    package_manifest_path=None,
    evaluation_mode="strict",
):
    if evaluation_mode not in ("strict", "diagnostic-v2"):
        raise ValueError("unsupported evaluation mode: {}".format(evaluation_mode))
    if len(models) != len(set(models)):
        raise ValueError("duplicate model aliases")
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("duplicate task IDs")
    manifest = _read_json(manifest_path)
    reference_index = _read_json(reference_index_path)
    task_by_id = collections.OrderedDict(
        (task["id"], task) for task in manifest.get("tasks", [])
    )
    unknown = [task_id for task_id in task_ids if task_id not in task_by_id]
    if unknown:
        raise ValueError("unknown task IDs: {}".format(", ".join(unknown)))
    selected_ids = [task_id for task_id in task_by_id if task_id in set(task_ids)]
    requested_pairs = None
    if package_manifest_path:
        package_manifest = _read_json(package_manifest_path)
        package_entries = package_manifest.get("entries")
        if package_entries:
            requested_pairs = [
                (entry["model_alias"], entry["task_id"])
                for entry in package_entries
            ]
            if len(requested_pairs) != len(set(requested_pairs)):
                raise ValueError("duplicate model/task pairs in package manifest")
            allowed_models = set(models)
            allowed_tasks = set(selected_ids)
            outside_scope = [
                pair for pair in requested_pairs
                if pair[0] not in allowed_models or pair[1] not in allowed_tasks
            ]
            if outside_scope:
                raise ValueError("package manifest pair is outside evaluation scope")
    if requested_pairs is None:
        requested_pairs = [
            (model_alias, task_id)
            for model_alias in sorted(set(models))
            for task_id in selected_ids
        ]
    records = []
    for model_alias, task_id in requested_pairs:
        task = task_by_id[task_id]
        gold_value, gold_error = _gold_for_task(
            gold_dir, reference_index, task_id
        )
        execution_value, execution_error = _execution_for_task(
            execution_dir, model_alias, task_id
        )
        if gold_error:
            result = _infrastructure(
                model_alias, task_id, gold_error, evaluation_mode
            )
        elif execution_error:
            result = _infrastructure(
                model_alias, task_id, execution_error, evaluation_mode
            )
        else:
            result = evaluate_record(
                execution_value, gold_value, task, evaluation_mode
            )
        result["model_alias"] = model_alias
        result["task_id"] = task_id
        result["level"] = task.get("level")
        result["subtype"] = task.get("subtype")
        records.append(result)
        _atomic_json(
            os.path.join(
                output_dir, model_alias, task_id, "evaluation.json"
            ),
            result,
        )

    jsonl_path = os.path.join(output_dir, "evaluation.jsonl")
    summary_path = os.path.join(output_dir, "evaluation_summary.json")
    _atomic_jsonl(jsonl_path, records)
    summary = _group_summary(records)
    _atomic_json(summary_path, summary)
    evaluation_manifest = {
        "evaluator_version": EVALUATOR_VERSION,
        "evaluation_mode": evaluation_mode,
        "task_manifest_sha256": _file_sha256(manifest_path),
        "protocol_sha256": _file_sha256(protocol_path),
        "gold_reference_index_sha256": _file_sha256(reference_index_path),
        "package_manifest_sha256": (
            _file_sha256(package_manifest_path)
            if package_manifest_path else None
        ),
        "evaluation_jsonl_sha256": _file_sha256(jsonl_path),
        "record_count": len(records),
    }
    _atomic_json(
        os.path.join(output_dir, "evaluation_manifest.json"),
        evaluation_manifest,
    )
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution-dir", required=True)
    parser.add_argument("--gold-dir", required=True)
    parser.add_argument("--reference-index", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--task-ids", nargs="+", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--package-manifest", required=True)
    parser.add_argument(
        "--evaluation-mode",
        choices=("strict", "diagnostic-v2"),
        default="strict",
    )
    args = parser.parse_args(argv)
    summary = evaluate_batch(
        args.execution_dir,
        args.gold_dir,
        args.reference_index,
        args.manifest,
        args.output_dir,
        args.models,
        args.task_ids,
        args.protocol,
        args.package_manifest,
        args.evaluation_mode,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

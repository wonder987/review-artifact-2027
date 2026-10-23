#!/usr/bin/env python3
"""Join Pilot evidence and emit deterministic stage and failure summaries."""

from __future__ import print_function

import argparse
import collections
import hashlib
import json
import os

try:
    from scripts.failure_taxonomy import (
        classify_evaluation_outcome,
        load_taxonomy,
    )
except ImportError:
    from failure_taxonomy import classify_evaluation_outcome, load_taxonomy


ANALYZER_VERSION = "pilot-failure-analyzer-v1"
MAX_DIAGNOSTIC_BYTES = 2 * 1024 * 1024


def _read_json(path):
    with open(path, "r") as handle:
        return json.load(handle)


def _read_jsonl(path):
    values = []
    with open(path, "r") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                values.append(json.loads(line))
            except ValueError as error:
                raise ValueError("invalid JSONL line {}: {}".format(line_number, error))
    return values


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _directory_sha256(path):
    digest = hashlib.sha256()
    files = []
    for directory, _, names in os.walk(path):
        for name in names:
            absolute = os.path.join(directory, name)
            relative = os.path.relpath(absolute, path).replace(os.sep, "/")
            files.append((relative, absolute))
    for relative, absolute in sorted(files):
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        with open(absolute, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
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


def _atomic_jsonl(path, values):
    directory = os.path.dirname(os.path.abspath(path))
    if not os.path.isdir(directory):
        os.makedirs(directory)
    temporary = path + ".tmp"
    with open(temporary, "w") as handle:
        for value in values:
            handle.write(json.dumps(value, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _index_unique(values, label):
    indexed = {}
    for value in values:
        key = (value.get("model_alias"), value.get("task_id"))
        if not all(key):
            raise ValueError("{} record lacks model_alias or task_id".format(label))
        if key in indexed:
            raise ValueError("duplicate {} record: {} / {}".format(label, *key))
        indexed[key] = value
    return indexed


def _read_diagnostic_text(path):
    if not os.path.isfile(path):
        return ""
    with open(path, "rb") as handle:
        value = handle.read(MAX_DIAGNOSTIC_BYTES)
    return value.decode("utf-8", "replace")


def _load_executions(execution_dir):
    values = []
    for directory, _, names in os.walk(execution_dir):
        if "execution.json" not in names:
            continue
        value = _read_json(os.path.join(directory, "execution.json"))
        if not isinstance(value, dict):
            raise ValueError("execution record must be an object")
        value["diagnostic_stdout"] = _read_diagnostic_text(
            os.path.join(directory, "stdout.log")
        )
        value["diagnostic_stderr"] = _read_diagnostic_text(
            os.path.join(directory, "stderr.log")
        )
        value["diagnostic_program_source"] = _read_diagnostic_text(
            os.path.join(directory, "program.py")
        )
        values.append(value)
    return _index_unique(values, "execution")


def _group(records, field, model_evaluable_total):
    counts = collections.Counter(str(record.get(field)) for record in records)
    total = len(records)
    result = collections.OrderedDict()
    for key in sorted(counts):
        selected = [record for record in records if str(record.get(field)) == key]
        semantic_success = sum(bool(record["semantic_success"]) for record in selected)
        model_evaluable = sum(bool(record["model_evaluable"]) for record in selected)
        result[key] = {
            "count": counts[key],
            "fraction_of_all": float(counts[key]) / total if total else None,
            "model_evaluable": model_evaluable,
            "fraction": (
                float(model_evaluable) / model_evaluable_total
                if model_evaluable_total else None
            ),
            "fraction_of_model_evaluable": (
                float(model_evaluable) / model_evaluable_total
                if model_evaluable_total else None
            ),
            "semantic_success": semantic_success,
            "semantic_accuracy": (
                float(semantic_success) / model_evaluable
                if model_evaluable else None
            ),
        }
    return result


def _stage_summary(records):
    total = len(records)
    evaluable = [record for record in records if record["model_evaluable"]]
    model_evaluable = len(evaluable)
    stages = (
        "generation_completed", "syntax_valid", "policy_safe",
        "execution_success", "protocol_valid", "semantic_success",
    )
    summary = {
        "total": total,
        "model_evaluable": model_evaluable,
        "infrastructure_failure": total - model_evaluable,
    }
    for stage in stages:
        count = sum(bool(record[stage]) for record in evaluable)
        summary[stage] = count
        summary[stage + "_rate"] = (
            float(count) / model_evaluable if model_evaluable else None
        )
    summary["by_model"] = _group(records, "model_alias", model_evaluable)
    summary["by_level"] = _group(records, "level", model_evaluable)
    summary["by_task_subtype"] = _group(
        records, "task_subtype", model_evaluable
    )
    summary["by_major_category"] = _group(
        records, "major_category", model_evaluable
    )
    summary["by_failure_subtype"] = _group(
        records, "failure_subtype", model_evaluable
    )
    return summary


def analyze_pilot(
    execution_dir,
    evaluation_jsonl,
    task_manifest,
    taxonomy_path,
    output_dir,
):
    taxonomy = load_taxonomy(taxonomy_path)
    manifest = _read_json(task_manifest)
    tasks = manifest.get("tasks")
    if not isinstance(tasks, list):
        raise ValueError("task manifest requires tasks")
    task_by_id = collections.OrderedDict()
    for task in tasks:
        task_id = task.get("id")
        if not task_id or task_id in task_by_id:
            raise ValueError("task manifest contains missing or duplicate ID")
        task_by_id[task_id] = task

    executions = _load_executions(execution_dir)
    evaluations = _index_unique(_read_jsonl(evaluation_jsonl), "evaluation")
    if set(executions) != set(evaluations):
        missing_execution = sorted(set(evaluations) - set(executions))
        missing_evaluation = sorted(set(executions) - set(evaluations))
        raise ValueError(
            "execution/evaluation key mismatch; missing_execution={}, "
            "missing_evaluation={}".format(missing_execution, missing_evaluation)
        )

    task_order = dict((task_id, index) for index, task_id in enumerate(task_by_id))
    records = []
    for key in sorted(executions, key=lambda item: (item[0], task_order.get(item[1], 10 ** 9))):
        model_alias, task_id = key
        if task_id not in task_by_id:
            raise ValueError("unknown task ID in Pilot evidence: {}".format(task_id))
        execution = executions[key]
        evaluation = evaluations[key]
        outcome = classify_evaluation_outcome(execution, evaluation)
        status = execution.get("status")
        generation_completed = status != "generation_failure"
        syntax_valid = generation_completed and outcome["failure_subtype"] != "syntax_error"
        policy_safe = syntax_valid and outcome["failure_subtype"] != "unsafe_operation"
        execution_success = status == "execution_success"
        protocol_valid = evaluation.get("category") in ("success", "semantic_failure")
        semantic_success = evaluation.get("semantic_success") is True
        model_evaluable = outcome["major_category"] != "infrastructure"
        task = task_by_id[task_id]
        records.append({
            "model_alias": model_alias,
            "task_id": task_id,
            "level": task.get("level"),
            "task_subtype": task.get("subtype"),
            "legacy_category": evaluation.get("category"),
            "legacy_code": evaluation.get("code"),
            "major_category": outcome["major_category"],
            "failure_subtype": outcome["failure_subtype"],
            "evidence": outcome["evidence"],
            "normalizations": evaluation.get("normalizations") or [],
            "model_evaluable": model_evaluable,
            "generation_completed": generation_completed,
            "syntax_valid": syntax_valid,
            "policy_safe": policy_safe,
            "execution_success": execution_success,
            "protocol_valid": protocol_valid,
            "semantic_success": semantic_success,
        })

    records_path = os.path.join(output_dir, "failure_records.jsonl")
    summary_path = os.path.join(output_dir, "failure_summary.json")
    manifest_path = os.path.join(output_dir, "analysis_manifest.json")
    _atomic_jsonl(records_path, records)
    summary = _stage_summary(records)
    _atomic_json(summary_path, summary)
    analysis_manifest = {
        "analyzer_version": ANALYZER_VERSION,
        "taxonomy_version": taxonomy["taxonomy_version"],
        "record_count": len(records),
        "execution_directory_sha256": _directory_sha256(execution_dir),
        "evaluation_jsonl_sha256": _file_sha256(evaluation_jsonl),
        "task_manifest_sha256": _file_sha256(task_manifest),
        "taxonomy_sha256": _file_sha256(taxonomy_path),
        "failure_records_sha256": _file_sha256(records_path),
        "failure_summary_sha256": _file_sha256(summary_path),
    }
    _atomic_json(manifest_path, analysis_manifest)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution-dir", required=True)
    parser.add_argument("--evaluation-jsonl", required=True)
    parser.add_argument("--task-manifest", required=True)
    parser.add_argument("--taxonomy", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    summary = analyze_pilot(
        args.execution_dir,
        args.evaluation_jsonl,
        args.task_manifest,
        args.taxonomy,
        args.output_dir,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

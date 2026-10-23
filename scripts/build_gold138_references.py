#!/usr/bin/env python3
"""Build two-run Gold-138 references with timeout and resume support."""

from __future__ import print_function

import argparse
import collections
import json
import os
import subprocess
import sys


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TASK_RUNNER = os.path.join(SCRIPT_DIR, "run_gold_reference_task.py")


def read_json(path):
    with open(path, "r") as handle:
        return json.load(handle)


def atomic_write_json(path, value):
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


def _append_difference(differences, label, first, second):
    if first != second:
        differences.append(label)


def compare_repeats(first, second):
    differences = []
    _append_difference(
        differences, "task_id", first.get("task_id"), second.get("task_id")
    )
    _append_difference(
        differences, "status", first.get("status"), second.get("status")
    )
    targets_a = first.get("targets", [])
    targets_b = second.get("targets", [])
    _append_difference(
        differences, "target_count", len(targets_a), len(targets_b)
    )
    for index in range(min(len(targets_a), len(targets_b))):
        target_a = targets_a[index]
        target_b = targets_b[index]
        if not isinstance(target_a, dict) or not isinstance(target_b, dict):
            _append_difference(
                differences,
                "target_{}.missing".format(index),
                isinstance(target_a, dict),
                isinstance(target_b, dict),
            )
            continue
        _append_difference(
            differences,
            "target_{}.identity".format(index),
            target_a.get("target"),
            target_b.get("target"),
        )
        _append_difference(
            differences,
            "target_{}.expressions".format(index),
            target_a.get("expressions"),
            target_b.get("expressions"),
        )
        _append_difference(
            differences,
            "target_{}.window".format(index),
            target_a.get("window"),
            target_b.get("window"),
        )
        for array_name in ("signal", "time_axis"):
            a = target_a.get(array_name, {})
            b = target_b.get(array_name, {})
            for field in ("sha256", "shape", "size", "dtype"):
                _append_difference(
                    differences,
                    "target_{}.{}.{}".format(index, array_name, field),
                    a.get(field),
                    b.get(field),
                )
        for field in (
            "non_empty",
            "same_size",
            "signal_all_finite",
            "time_all_finite",
            "time_strictly_increasing",
            "expected_point_count_match",
            "window_start_match",
            "window_end_match",
            "time_step_match",
        ):
            _append_difference(
                differences,
                "target_{}.validation.{}".format(index, field),
                target_a.get("validation", {}).get(field),
                target_b.get("validation", {}).get(field),
            )

    _append_difference(
        differences,
        "resource_decisions",
        first.get("resource_decisions"),
        second.get("resource_decisions"),
    )

    derived_a = first.get("derived", {})
    derived_b = second.get("derived", {})
    for field in ("common_time_axis", "aligned_first", "aligned_second"):
        _append_difference(
            differences,
            "derived.{}.sha256".format(field),
            derived_a.get(field, {}).get("sha256"),
            derived_b.get(field, {}).get("sha256"),
        )
    _append_difference(
        differences,
        "derived.comparison",
        derived_a.get("comparison"),
        derived_b.get("comparison"),
    )
    stable = (
        first.get("status") == "success"
        and second.get("status") == "success"
        and not differences
    )
    return {"stable": bool(stable), "differences": differences}


def repeat_path(task_id, results_dir, repeat_id):
    return os.path.join(
        results_dir, "{}.repeat{}.json".format(task_id, int(repeat_id))
    )


def can_resume_task(task_id, results_dir, repeats=2):
    paths = [
        repeat_path(task_id, results_dir, repeat_id)
        for repeat_id in range(1, int(repeats) + 1)
    ]
    if not all(
        is_completed_repeat(path, task_id, repeat_id)
        for repeat_id, path in enumerate(paths, 1)
    ):
        return False
    values = [read_json(path) for path in paths]
    if not all(value.get("status") == "success" for value in values):
        return False
    return all(compare_repeats(values[0], value)["stable"] for value in values[1:])


def is_completed_repeat(path, task_id, repeat_id):
    if not os.path.isfile(path):
        return False
    try:
        value = read_json(path)
    except (IOError, OSError, ValueError):
        return False
    return (
        value.get("status") == "success"
        and value.get("task_id") == task_id
        and int(value.get("repeat_id", -1)) == int(repeat_id)
    )


def call_process(command, timeout_seconds, log_path=None):
    stream = None
    try:
        if log_path:
            stream = open(log_path, "w")
        return subprocess.call(
            command,
            timeout=timeout_seconds,
            stdout=stream,
            stderr=subprocess.STDOUT if stream is not None else None,
        )
    except subprocess.TimeoutExpired:
        return 124
    finally:
        if stream is not None:
            stream.close()


def run_reference_command(
    command,
    output_path,
    timeout_seconds,
    caller=None,
):
    if os.path.isfile(output_path):
        os.remove(output_path)
    if caller is None:
        exit_status = int(
            call_process(command, timeout_seconds, output_path + ".runner.log")
        )
    else:
        exit_status = int(caller(command, timeout_seconds))
    if exit_status != 0:
        if os.path.isfile(output_path):
            try:
                failure = read_json(output_path)
            except (IOError, OSError, ValueError):
                failure = {}
        else:
            failure = {}
        failure["status"] = "timeout" if exit_status == 124 else failure.get(
            "status", "error"
        )
        failure["exit_status"] = exit_status
        failure.setdefault(
            "failure_type",
            "execution_timeout" if exit_status == 124 else "runner_error",
        )
        atomic_write_json(output_path, failure)
    return exit_status


def balanced_limit(tasks, limit):
    if limit is None or int(limit) >= len(tasks):
        return list(tasks)
    by_subtype = collections.OrderedDict(
        (subtype, [])
        for subtype in (
            "simple_retrieval",
            "exact_window",
            "sampling_constrained",
            "subtree_discovery",
            "adaptive_access",
            "cross_shot_comparison",
            "same_shot_alignment",
        )
    )
    for task in tasks:
        by_subtype[task["subtype"]].append(task)
    order = (
        "simple_retrieval",
        "exact_window",
        "adaptive_access",
        "simple_retrieval",
        "sampling_constrained",
        "cross_shot_comparison",
        "simple_retrieval",
        "subtree_discovery",
        "same_shot_alignment",
    )
    selected = []
    while len(selected) < int(limit):
        progress = False
        for subtype in order:
            if by_subtype[subtype] and len(selected) < int(limit):
                selected.append(by_subtype[subtype].pop(0))
                progress = True
        if not progress:
            break
    return selected


def write_batch_summary(path, index, requested):
    atomic_write_json(
        path,
        {
            "completed": len(index["tasks"]),
            "requested": int(requested),
            "stable": sum(
                item["stability"]["stable"]
                for item in index["tasks"].values()
            ),
            "failed_or_unstable": sum(
                not item["stability"]["stable"]
                for item in index["tasks"].values()
            ),
        },
    )


def build_references(
    manifest_path,
    results_dir,
    repeats=2,
    timeout_seconds=90,
    limit=None,
    resume=False,
):
    if int(repeats) != 2:
        raise ValueError("Gold-138 requires exactly two reference runs")
    manifest = read_json(manifest_path)
    tasks = balanced_limit(manifest["tasks"], limit)
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    index = {
        "schema_version": "1.0",
        "manifest": os.path.abspath(manifest_path),
        "requested_task_count": len(tasks),
        "repeats": int(repeats),
        "timeout_seconds": int(timeout_seconds),
        "tasks": {},
    }
    index_path = os.path.join(results_dir, "reference_index.json")
    summary_path = os.path.join(results_dir, "batch_summary.json")

    for position, task in enumerate(tasks, 1):
        task_id = task["id"]
        if resume and can_resume_task(task_id, results_dir, repeats):
            repeat_files = [
                repeat_path(task_id, results_dir, repeat_id)
                for repeat_id in range(1, int(repeats) + 1)
            ]
            repeats_data = [read_json(path) for path in repeat_files]
            stability = compare_repeats(repeats_data[0], repeats_data[1])
            index["tasks"][task_id] = {
                "repeat_files": [
                    os.path.relpath(path, results_dir) for path in repeat_files
                ],
                "exit_statuses": [0] * int(repeats),
                "stability": stability,
                "resumed": True,
            }
            atomic_write_json(index_path, index)
            write_batch_summary(summary_path, index, len(tasks))
            continue

        repeat_files = []
        exit_statuses = []
        completed_repeats = set()
        if resume:
            completed_repeats = set(
                repeat_id
                for repeat_id in range(1, int(repeats) + 1)
                if is_completed_repeat(
                    repeat_path(task_id, results_dir, repeat_id),
                    task_id,
                    repeat_id,
                )
            )
            if len(completed_repeats) == int(repeats):
                completed_repeats = set()
        for repeat_id in range(1, int(repeats) + 1):
            output_path = repeat_path(task_id, results_dir, repeat_id)
            if repeat_id in completed_repeats:
                repeat_files.append(output_path)
                exit_statuses.append(0)
                continue
            command = [
                sys.executable,
                TASK_RUNNER,
                "--manifest",
                manifest_path,
                "--task-id",
                task_id,
                "--repeat-id",
                str(repeat_id),
                "--output",
                output_path,
            ]
            exit_status = run_reference_command(
                command, output_path, timeout_seconds
            )
            repeat_files.append(output_path)
            exit_statuses.append(exit_status)

        repeat_data = [read_json(path) for path in repeat_files]
        stability = compare_repeats(repeat_data[0], repeat_data[1])
        index["tasks"][task_id] = {
            "position": position,
            "repeat_files": [
                os.path.relpath(path, results_dir) for path in repeat_files
            ],
            "exit_statuses": exit_statuses,
            "stability": stability,
            "resumed": False,
        }
        atomic_write_json(index_path, index)
        write_batch_summary(summary_path, index, len(tasks))
        print(
            "completed={}/{} task={} stable={}".format(
                len(index["tasks"]),
                len(tasks),
                task_id,
                stability["stable"],
            )
        )
    return index


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=int, default=90)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    index = build_references(
        args.manifest,
        args.results_dir,
        args.repeats,
        args.timeout_seconds,
        args.limit,
        args.resume,
    )
    stable = sum(
        item["stability"]["stable"] for item in index["tasks"].values()
    )
    print("stable_tasks={}/{}".format(stable, len(index["tasks"])))
    return 0 if stable == len(index["tasks"]) else 1


if __name__ == "__main__":
    sys.exit(main())

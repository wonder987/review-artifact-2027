#!/usr/bin/env python3
"""Create a balanced draft task manifest from validated MDSplus targets."""

from __future__ import print_function

import argparse
import collections
import json
import os
import re
import sys


EXCLUDED_KEY = (90000, "t3", "RXC17A")
FORCED_PREFLIGHT = {(106915, "t2", "AXUV_DIV1")}
GROUNDING_LEVELS = (
    "explicit_target",
    "subtree_hidden",
    "semantic_request",
)


def candidate_key(candidate):
    return (
        int(candidate["shot"]),
        str(candidate["subtree"]),
        str(candidate["signal_name"]),
    )


def count_by(tasks, field):
    return dict(collections.Counter(task[field] for task in tasks))


def safe_id(value):
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")


def choose_window_keys(unsafe_candidates):
    groups = collections.defaultdict(list)
    for candidate in unsafe_candidates:
        groups[candidate["size_band"]].append(candidate)

    total_window = len(unsafe_candidates) // 2
    quotas = {band: len(items) // 2 for band, items in groups.items()}
    remainder = total_window - sum(quotas.values())
    for band in sorted(groups, key=lambda item: (-len(groups[item]), item)):
        if remainder <= 0:
            break
        quotas[band] += 1
        remainder -= 1

    selected = set()
    for band in sorted(groups):
        available = [
            item
            for item in groups[band]
            if candidate_key(item) not in FORCED_PREFLIGHT
        ]
        shot_counts = collections.Counter()
        subtree_counts = collections.Counter()
        remaining = sorted(
            available,
            key=lambda item: (
                int(item["shot"]),
                str(item["subtree"]),
                str(item["signal_name"]),
            ),
        )
        while len([key for key in selected if key[0] == band]) < quotas[band]:
            if not remaining:
                raise ValueError("not enough window candidates for {}".format(band))
            choice = min(
                remaining,
                key=lambda item: (
                    shot_counts[int(item["shot"])],
                    subtree_counts[str(item["subtree"])],
                    int(item["shot"]),
                    str(item["subtree"]),
                    str(item["signal_name"]),
                ),
            )
            remaining.remove(choice)
            key = candidate_key(choice)
            selected.add((band,) + key)
            shot_counts[int(choice["shot"])] += 1
            subtree_counts[str(choice["subtree"])] += 1

    return {key[1:] for key in selected}


def build_operation(candidate, operation_kind):
    if operation_kind == "full_retrieval":
        return {
            "kind": operation_kind,
            "requires_time_axis": True,
        }
    if operation_kind == "metadata_preflight":
        return {
            "kind": operation_kind,
            "must_not_load_full_signal": True,
            "max_points": 5000000,
            "max_total_mib": 256.0,
            "expected_decision": "limit_time_window_or_downsample",
        }

    frequency = float(candidate["frequency"])
    trigger_time = float(candidate["trigtime"])
    index_start = int(round((0.0 - trigger_time) * frequency))
    index_end = index_start + 10000
    if index_start < 0 or index_end >= int(candidate["sample_num"]):
        raise ValueError("infeasible zero-time window: {}".format(candidate_key(candidate)))
    return {
        "kind": "index_window",
        "requires_time_axis": True,
        "time_start": 0.0,
        "time_end": 10000.0 / frequency,
        "time_step": 1.0 / frequency,
        "index_start": index_start,
        "index_end": index_end,
        "expected_points": 10001,
    }


def assign_grounding_levels(tasks):
    global_counts = collections.Counter()
    operation_counts = collections.defaultdict(collections.Counter)
    shot_counts = collections.defaultdict(collections.Counter)
    ordered = sorted(
        tasks,
        key=lambda task: (
            task["operation_kind"],
            task["target"]["shot"],
            task["target"]["subtree"],
            task["target"]["node"],
        ),
    )
    for task in ordered:
        operation = task["operation_kind"]
        shot = task["target"]["shot"]
        level = min(
            GROUNDING_LEVELS,
            key=lambda item: (
                global_counts[item],
                operation_counts[operation][item],
                shot_counts[shot][item],
                GROUNDING_LEVELS.index(item),
            ),
        )
        task["grounding_level"] = level
        task["prompt_status"] = (
            "requires_domain_description"
            if level == "semantic_request"
            else "ready_for_template_generation"
        )
        global_counts[level] += 1
        operation_counts[operation][level] += 1
        shot_counts[shot][level] += 1


def build_manifest(candidates):
    reliable = [
        candidate
        for candidate in candidates
        if candidate_key(candidate) != EXCLUDED_KEY
    ]
    if len(reliable) != 138:
        raise ValueError("expected 138 reliable candidates, got {}".format(len(reliable)))

    unsafe = [
        candidate
        for candidate in reliable
        if candidate["size_band"] != "safe_full"
    ]
    window_keys = choose_window_keys(unsafe)
    tasks = []
    for candidate in reliable:
        key = candidate_key(candidate)
        if candidate["size_band"] == "safe_full":
            operation_kind = "full_retrieval"
        elif key in window_keys:
            operation_kind = "index_window"
        else:
            operation_kind = "metadata_preflight"

        target = {
            "server": "202.127.204.12",
            "tree": str(candidate["tree"]),
            "shot": int(candidate["shot"]),
            "subtree": str(candidate["subtree"]),
            "node": "\\" + str(candidate["signal_name"]),
            "signal_name": str(candidate["signal_name"]),
            "sample_num": int(candidate["sample_num"]),
            "frequency": float(candidate["frequency"]),
            "daqtime": float(candidate["daqtime"]),
            "trigtime": float(candidate["trigtime"]),
            "unit": candidate.get("unit"),
            "size_band": str(candidate["size_band"]),
            "estimated_total_mib": float(candidate["estimated_total_mib"]),
        }
        task = {
            "id": "east_{}_{}_{}_{}".format(
                safe_id(candidate["signal_name"]),
                int(candidate["shot"]),
                safe_id(candidate["subtree"]),
                operation_kind,
            ),
            "operation_kind": operation_kind,
            "target": target,
            "operation": build_operation(candidate, operation_kind),
            "oracle_stage": "target_validated_final_reference_pending",
        }
        tasks.append(task)

    assign_grounding_levels(tasks)
    tasks.sort(key=lambda task: task["id"])
    manifest = {
        "schema_version": "0.2-draft",
        "benchmark_name": "PA-SCoRe MDSplus Gold-138",
        "created_date": "2026-09-21",
        "status": "stratified_draft",
        "task_count": len(tasks),
        "operation_distribution": count_by(tasks, "operation_kind"),
        "grounding_distribution": count_by(tasks, "grounding_level"),
        "excluded_diagnostics": [
            {
                "shot": 90000,
                "subtree": "t3",
                "node": "\\RXC17A",
                "mongodb_points": 12000,
                "mdsplus_points": 10000,
                "reason": "mongodb_sample_count_mismatch",
            }
        ],
        "repair_set_policy": (
            "derive from observed model failures and a separately reported "
            "coverage-oriented seeded-failure subset"
        ),
        "tasks": tasks,
    }
    return manifest


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.candidates, "r") as handle:
        candidates = json.load(handle)
    manifest = build_manifest(candidates)
    output_dir = os.path.dirname(os.path.abspath(args.output))
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    with open(args.output, "w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print("task_count={}".format(manifest["task_count"]))
    print("operation_distribution={}".format(manifest["operation_distribution"]))
    print("grounding_distribution={}".format(manifest["grounding_distribution"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Compose the deterministic PA-SCoRe Gold-138 task manifest."""

from __future__ import print_function

import argparse
import collections
import hashlib
import itertools
import json
import os
import sys


PILOT_SHOTS = {70000, 106915, 150000}
METADATA_MISMATCH_KEY = (90000, "t3", "RXC17A")
UNREADABLE_REPLACEMENTS = {
    "gold138_l2_028": {
        "bad": (50000, "t7", "POINT0"),
        "replacement": (90000, "t3", "SXR10F"),
    },
    "gold138_l3_017": {
        "bad": (106915, "t6", "LP1"),
        "replacement": (106915, "t5", "BOLOV1"),
    },
}
MAX_POINTS = 5000000
MAX_TOTAL_MIB = 256.0
MAX_TARGET_USAGE = 2


def candidate_key(candidate):
    return (
        int(candidate["shot"]),
        str(candidate["subtree"]),
        str(candidate["signal_name"]),
    )


def stable_rank(value):
    rendered = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def is_safe_full(candidate):
    return (
        int(candidate["sample_num"]) <= MAX_POINTS
        and float(candidate["estimated_total_mib"]) <= MAX_TOTAL_MIB
    )


def to_target(candidate):
    return {
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


def window_operation(candidate, requested_points):
    frequency = float(candidate["frequency"])
    index_start = int(round((0.0 - float(candidate["trigtime"])) * frequency))
    index_end = index_start + int(requested_points) - 1
    if index_start < 0 or index_end >= int(candidate["sample_num"]):
        raise ValueError("infeasible index window: {}".format(candidate_key(candidate)))
    return {
        "time_start": 0.0,
        "time_end": float(requested_points - 1) / frequency,
        "time_step": 1.0 / frequency,
        "index_start": index_start,
        "index_end": index_end,
        "expected_points": int(requested_points),
    }


def window_feasible(candidate, requested_points):
    try:
        window_operation(candidate, requested_points)
        return True
    except ValueError:
        return False


def choose_singles(pool, count, usage, predicate, salt):
    available = [item for item in pool if predicate(item)]
    selected = []
    shot_counts = collections.Counter()
    subtree_counts = collections.Counter()
    while len(selected) < count:
        eligible = [
            item
            for item in available
            if usage[candidate_key(item)] < MAX_TARGET_USAGE
            and candidate_key(item) not in {candidate_key(x) for x in selected}
        ]
        if not eligible:
            raise ValueError("not enough candidates for {} (need {})".format(salt, count))
        choice = min(
            eligible,
            key=lambda item: (
                shot_counts[int(item["shot"])],
                subtree_counts[str(item["subtree"])],
                usage[candidate_key(item)],
                stable_rank([salt, candidate_key(item)]),
            ),
        )
        selected.append(choice)
        usage[candidate_key(choice)] += 1
        shot_counts[int(choice["shot"])] += 1
        subtree_counts[str(choice["subtree"])] += 1
    return selected


def choose_same_shot_pairs(pool, count, usage, salt):
    options = []
    by_shot = collections.defaultdict(list)
    for candidate in pool:
        if window_feasible(candidate, 1001):
            by_shot[int(candidate["shot"])].append(candidate)
    for shot in sorted(by_shot):
        for left, right in itertools.combinations(by_shot[shot], 2):
            if left["signal_name"] == right["signal_name"]:
                continue
            options.append((left, right))

    selected = []
    shot_counts = collections.Counter()
    used_specs = set()
    while len(selected) < count:
        eligible = []
        for left, right in options:
            keys = (candidate_key(left), candidate_key(right))
            spec = tuple(sorted(keys))
            if spec in used_specs:
                continue
            if any(usage[key] >= MAX_TARGET_USAGE for key in keys):
                continue
            eligible.append((left, right))
        if not eligible:
            raise ValueError("not enough same-shot pairs for {}".format(salt))
        left, right = min(
            eligible,
            key=lambda pair: (
                shot_counts[int(pair[0]["shot"])],
                max(usage[candidate_key(pair[0])], usage[candidate_key(pair[1])]),
                usage[candidate_key(pair[0])] + usage[candidate_key(pair[1])],
                stable_rank([salt, candidate_key(pair[0]), candidate_key(pair[1])]),
            ),
        )
        keys = (candidate_key(left), candidate_key(right))
        selected.append((left, right))
        used_specs.add(tuple(sorted(keys)))
        for key in keys:
            usage[key] += 1
        shot_counts[int(left["shot"])] += 1
    return selected


def choose_cross_shot_pairs(pool, count, usage, salt):
    options = []
    by_signal = collections.defaultdict(list)
    for candidate in pool:
        if window_feasible(candidate, 1001):
            by_signal[str(candidate["signal_name"])].append(candidate)
    for signal_name in sorted(by_signal):
        for left, right in itertools.combinations(by_signal[signal_name], 2):
            if int(left["shot"]) == int(right["shot"]):
                continue
            options.append((left, right))

    selected = []
    signal_counts = collections.Counter()
    shot_counts = collections.Counter()
    used_specs = set()
    while len(selected) < count:
        eligible = []
        for left, right in options:
            keys = (candidate_key(left), candidate_key(right))
            spec = tuple(sorted(keys))
            if spec in used_specs:
                continue
            if any(usage[key] >= MAX_TARGET_USAGE for key in keys):
                continue
            eligible.append((left, right))
        if not eligible:
            raise ValueError("not enough cross-shot pairs for {}".format(salt))
        left, right = min(
            eligible,
            key=lambda pair: (
                signal_counts[str(pair[0]["signal_name"])],
                shot_counts[int(pair[0]["shot"])]
                + shot_counts[int(pair[1]["shot"])],
                max(usage[candidate_key(pair[0])], usage[candidate_key(pair[1])]),
                usage[candidate_key(pair[0])] + usage[candidate_key(pair[1])],
                stable_rank([salt, candidate_key(pair[0]), candidate_key(pair[1])]),
            ),
        )
        keys = (candidate_key(left), candidate_key(right))
        selected.append((left, right))
        used_specs.add(tuple(sorted(keys)))
        for key in keys:
            usage[key] += 1
        signal_counts[str(left["signal_name"])] += 1
        shot_counts[int(left["shot"])] += 1
        shot_counts[int(right["shot"])] += 1
    return selected


def base_task(level, subtype, split, candidates, operation, disclosure):
    task = {
        "level": level,
        "subtype": subtype,
        "split": split,
        "targets": [to_target(item) for item in candidates],
        "operation": operation,
        "safety_policy": {
            "max_points": MAX_POINTS,
            "max_total_mib": MAX_TOTAL_MIB,
            "timeout_seconds": 90,
        },
        "prompt_spec": disclosure,
        "oracle_status": "reference_generation_pending",
    }
    canonical = json.dumps(task, sort_keys=True, separators=(",", ":"))
    task["spec_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return task


def full_operation():
    return {"kind": "full_retrieval", "requires_time_axis": True}


def make_tasks_for_split(pool, split, quotas, usage):
    tasks = []

    cross_pairs = choose_cross_shot_pairs(
        pool, quotas["cross_shot_comparison"], usage, split + ":cross"
    )
    for left, right in cross_pairs:
        tasks.append(
            base_task(
                "L3",
                "cross_shot_comparison",
                split,
                [left, right],
                {
                    "kind": "cross_shot_comparison",
                    "per_target_windows": [
                        window_operation(left, 1001),
                        window_operation(right, 1001),
                    ],
                    "common_points": 1001,
                    "derived_metrics": ["mean", "std", "pearson_correlation"],
                },
                {"disclose_subtree": False, "disclose_indices": False},
            )
        )

    same_pairs = choose_same_shot_pairs(
        pool, quotas["same_shot_alignment"], usage, split + ":same"
    )
    for left, right in same_pairs:
        tasks.append(
            base_task(
                "L3",
                "same_shot_alignment",
                split,
                [left, right],
                {
                    "kind": "same_shot_alignment",
                    "per_target_windows": [
                        window_operation(left, 1001),
                        window_operation(right, 1001),
                    ],
                    "common_points": 1001,
                    "derived_metrics": ["mean", "std", "pearson_correlation"],
                },
                {"disclose_subtree": False, "disclose_indices": False},
            )
        )

    l1 = choose_singles(
        pool,
        quotas["L1"],
        usage,
        is_safe_full,
        split + ":L1",
    )
    for candidate in l1:
        tasks.append(
            base_task(
                "L1",
                "simple_retrieval",
                split,
                [candidate],
                full_operation(),
                {"disclose_subtree": True, "disclose_indices": False},
            )
        )

    subtree = choose_singles(
        pool,
        quotas["subtree_discovery"],
        usage,
        is_safe_full,
        split + ":subtree",
    )
    for candidate in subtree:
        tasks.append(
            base_task(
                "L2",
                "subtree_discovery",
                split,
                [candidate],
                full_operation(),
                {"disclose_subtree": False, "disclose_indices": False},
            )
        )

    exact = choose_singles(
        pool,
        quotas["exact_window"],
        usage,
        lambda item: window_feasible(item, 10001),
        split + ":exact",
    )
    for candidate in exact:
        operation = window_operation(candidate, 10001)
        operation["kind"] = "index_window"
        tasks.append(
            base_task(
                "L2",
                "exact_window",
                split,
                [candidate],
                operation,
                {"disclose_subtree": True, "disclose_indices": False},
            )
        )

    sampled = choose_singles(
        pool,
        quotas["sampling_constrained"],
        usage,
        lambda item: window_feasible(item, 10000),
        split + ":sampled",
    )
    for candidate in sampled:
        operation = window_operation(candidate, 10000)
        operation["kind"] = "sampling_constrained"
        tasks.append(
            base_task(
                "L2",
                "sampling_constrained",
                split,
                [candidate],
                operation,
                {"disclose_subtree": True, "disclose_indices": False},
            )
        )

    safe_adaptive = choose_singles(
        pool,
        quotas["adaptive_safe"],
        usage,
        is_safe_full,
        split + ":adaptive_safe",
    )
    unsafe_adaptive = choose_singles(
        pool,
        quotas["adaptive_unsafe"],
        usage,
        lambda item: not is_safe_full(item) and window_feasible(item, 10001),
        split + ":adaptive_unsafe",
    )
    for candidate in safe_adaptive + unsafe_adaptive:
        safe = is_safe_full(candidate)
        operation = {
            "kind": "adaptive_access",
            "expected_decision": (
                "read_full_signal" if safe else "limit_time_window_or_downsample"
            ),
            "fallback_window": None if safe else window_operation(candidate, 10001),
        }
        tasks.append(
            base_task(
                "L3",
                "adaptive_access",
                split,
                [candidate],
                operation,
                {"disclose_subtree": False, "disclose_indices": False},
            )
        )
    return tasks


def assign_ids(tasks):
    counters = collections.Counter()
    ordered = sorted(
        tasks,
        key=lambda task: (
            task["level"],
            task["subtype"],
            task["split"],
            task["spec_sha256"],
        ),
    )
    for task in ordered:
        counters[task["level"]] += 1
        task["id"] = "gold138_{}_{:03d}".format(
            task["level"].lower(), counters[task["level"]]
        )
    return ordered


def refresh_spec_sha256(task):
    content = dict(task)
    content.pop("id", None)
    content.pop("spec_sha256", None)
    canonical = json.dumps(content, sort_keys=True, separators=(",", ":"))
    task["spec_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def replace_unreadable_targets(tasks, candidates):
    by_key = {candidate_key(candidate): candidate for candidate in candidates}
    by_id = {task["id"]: task for task in tasks}
    for task_id, replacement_spec in UNREADABLE_REPLACEMENTS.items():
        task = by_id[task_id]
        actual = candidate_key(task["targets"][0])
        if actual != replacement_spec["bad"]:
            raise ValueError(
                "unexpected target for {}: {}".format(task_id, actual)
            )
        candidate = by_key[replacement_spec["replacement"]]
        task["targets"] = [to_target(candidate)]
        if task["subtype"] == "sampling_constrained":
            task["operation"] = window_operation(candidate, 10000)
            task["operation"]["kind"] = "sampling_constrained"
        elif task["subtype"] == "adaptive_access":
            task["operation"] = {
                "kind": "adaptive_access",
                "expected_decision": "limit_time_window_or_downsample",
                "fallback_window": window_operation(candidate, 10001),
            }
        else:
            raise ValueError(
                "unsupported replacement subtype: {}".format(task["subtype"])
            )
        refresh_spec_sha256(task)


def build_splits(tasks):
    ablation_counts = {"L1": 13, "L2": 21, "L3": 26}
    ablation = []
    for level in ("L1", "L2", "L3"):
        eligible = sorted(
            (
                task
                for task in tasks
                if task["split"] == "held_out" and task["level"] == level
            ),
            key=lambda task: task["spec_sha256"],
        )
        ablation.extend(
            task["id"] for task in eligible[: ablation_counts[level]]
        )
    return {
        "schema_version": "1.0",
        "pilot_shots": sorted(PILOT_SHOTS),
        "pilot": [task["id"] for task in tasks if task["split"] == "pilot"],
        "held_out": [
            task["id"] for task in tasks if task["split"] == "held_out"
        ],
        "ablation": ablation,
        "ablation_selection_rule": {
            "counts": ablation_counts,
            "source_split": "held_out",
            "ordering": "spec_sha256_ascending_within_level",
        },
    }


def compose_manifest(candidates):
    reliable = [
        candidate
        for candidate in candidates
        if candidate_key(candidate) != METADATA_MISMATCH_KEY
    ]
    if len(reliable) != 138:
        raise ValueError("expected 138 candidate targets, got {}".format(len(reliable)))

    pilot_pool = [item for item in reliable if int(item["shot"]) in PILOT_SHOTS]
    held_pool = [item for item in reliable if int(item["shot"]) not in PILOT_SHOTS]
    usage = collections.Counter()
    pilot_quotas = {
        "L1": 6,
        "subtree_discovery": 3,
        "exact_window": 3,
        "sampling_constrained": 4,
        "same_shot_alignment": 5,
        "cross_shot_comparison": 4,
        "adaptive_safe": 2,
        "adaptive_unsafe": 3,
    }
    held_quotas = {
        "L1": 24,
        "subtree_discovery": 13,
        "exact_window": 13,
        "sampling_constrained": 12,
        "same_shot_alignment": 15,
        "cross_shot_comparison": 16,
        "adaptive_safe": 8,
        "adaptive_unsafe": 7,
    }
    tasks = make_tasks_for_split(pilot_pool, "pilot", pilot_quotas, usage)
    tasks.extend(
        make_tasks_for_split(held_pool, "held_out", held_quotas, usage)
    )
    tasks = assign_ids(tasks)
    splits = build_splits(tasks)
    replace_unreadable_targets(tasks, reliable)
    manifest = {
        "schema_version": "1.0-draft",
        "benchmark_name": "PA-SCoRe MDSplus Gold-138",
        "created_date": "2026-09-21",
        "task_count": len(tasks),
        "source_target_count": 136,
        "excluded_diagnostics": [
            {
                "shot": 90000,
                "subtree": "t3",
                "node": "\\RXC17A",
                "reason": "mongodb_sample_count_mismatch",
            },
            {
                "shot": 50000,
                "subtree": "t7",
                "node": "\\POINT0",
                "reason": "bounded_read_returns_invalid_descriptor",
            },
            {
                "shot": 106915,
                "subtree": "t6",
                "node": "\\LP1",
                "reason": "bounded_read_returns_invalid_descriptor",
            },
        ],
        "tasks": tasks,
    }
    manifest["splits"] = splits
    return manifest


def write_json(path, value):
    directory = os.path.dirname(os.path.abspath(path))
    if not os.path.isdir(directory):
        os.makedirs(directory)
    with open(path, "w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--splits", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.candidates, "r") as handle:
        candidates = json.load(handle)
    manifest = compose_manifest(candidates)
    write_json(args.manifest, manifest)
    write_json(args.splits, manifest["splits"])
    level_counts = collections.Counter(
        task["level"] for task in manifest["tasks"]
    )
    print("task_count={}".format(manifest["task_count"]))
    print("level_counts={}".format(dict(level_counts)))
    return 0


if __name__ == "__main__":
    sys.exit(main())

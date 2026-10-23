#!/usr/bin/env python3
"""Blocking integrity audit for the Gold-138 benchmark artifacts."""

from __future__ import print_function

import argparse
import collections
import hashlib
import json
import os
import statistics
import sys


EXPECTED_LEVELS = {"L1": 30, "L2": 48, "L3": 60}
EXPECTED_SUBTYPES = {
    "simple_retrieval": 30,
    "exact_window": 16,
    "sampling_constrained": 16,
    "subtree_discovery": 16,
    "adaptive_access": 20,
    "cross_shot_comparison": 20,
    "same_shot_alignment": 20,
}
EXPECTED_SPLITS = {"pilot": 30, "held_out": 108}
EXPECTED_PILOT_SHOTS = {70000, 106915, 150000}
EXPECTED_SPLIT_LEVELS = {
    "pilot": {"L1": 6, "L2": 10, "L3": 14},
    "held_out": {"L1": 24, "L2": 38, "L3": 46},
}
EXPECTED_SPLIT_SUBTYPES = {
    "pilot": {
        "simple_retrieval": 6,
        "subtree_discovery": 3,
        "exact_window": 3,
        "sampling_constrained": 4,
        "same_shot_alignment": 5,
        "cross_shot_comparison": 4,
        "adaptive_access": 5,
    },
    "held_out": {
        "simple_retrieval": 24,
        "subtree_discovery": 13,
        "exact_window": 13,
        "sampling_constrained": 12,
        "same_shot_alignment": 15,
        "cross_shot_comparison": 16,
        "adaptive_access": 15,
    },
}
EXPECTED_SUBTYPE_LEVEL = {
    "simple_retrieval": "L1",
    "subtree_discovery": "L2",
    "exact_window": "L2",
    "sampling_constrained": "L2",
    "same_shot_alignment": "L3",
    "cross_shot_comparison": "L3",
    "adaptive_access": "L3",
}
FORBIDDEN_PROMPT_FIELDS = (
    "sha256",
    "index_start",
    "index_end",
    "allow_full_read",
    "expected_decision",
)


def read_json(path):
    with open(path, "r") as handle:
        return json.load(handle)


def read_jsonl(path):
    with open(path, "r") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def issue(code, message, task_id=None):
    value = {"code": code, "message": message}
    if task_id is not None:
        value["task_id"] = task_id
    return value


def counter_dict(values):
    return dict(sorted(collections.Counter(values).items()))


def target_key(target):
    return (
        target.get("server"),
        target.get("tree"),
        int(target.get("shot")),
        target.get("subtree"),
        target.get("node"),
    )


def audit_manifest_structure(manifest, splits):
    issues = []
    tasks = manifest.get("tasks", [])
    ids = [task.get("id") for task in tasks]
    by_level = counter_dict(task.get("level") for task in tasks)
    by_subtype = counter_dict(task.get("subtype") for task in tasks)
    by_split = counter_dict(task.get("split") for task in tasks)

    if len(tasks) != 138:
        issues.append(issue("wrong_task_count", "expected 138 tasks"))
    if len(set(ids)) != len(ids):
        issues.append(issue("duplicate_task_id", "task ids are not unique"))
    if by_level != EXPECTED_LEVELS:
        issues.append(issue("wrong_level_counts", "level counts do not match"))
    if by_subtype != EXPECTED_SUBTYPES:
        issues.append(
            issue("wrong_subtype_counts", "subtype counts do not match")
        )
    if by_split != EXPECTED_SPLITS:
        issues.append(issue("wrong_split_counts", "split counts do not match"))
    for task in tasks:
        if EXPECTED_SUBTYPE_LEVEL.get(task.get("subtype")) != task.get("level"):
            issues.append(
                issue(
                    "subtype_level_mismatch",
                    "subtype is assigned to the wrong level",
                    task.get("id"),
                )
            )
    spec_hashes = [task.get("spec_sha256") for task in tasks]
    if None in spec_hashes or len(set(spec_hashes)) != len(spec_hashes):
        issues.append(
            issue("duplicate_task_spec", "exact task specifications are not unique")
        )
    for task in tasks:
        canonical_task = dict(task)
        canonical_task.pop("id", None)
        recorded_hash = canonical_task.pop("spec_sha256", None)
        canonical = json.dumps(
            canonical_task, sort_keys=True, separators=(",", ":")
        )
        calculated_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        if recorded_hash != calculated_hash:
            issues.append(
                issue(
                    "task_spec_hash_mismatch",
                    "task fields differ from the frozen specification hash",
                    task.get("id"),
                )
            )

    for split_name in EXPECTED_SPLITS:
        split_tasks = [task for task in tasks if task.get("split") == split_name]
        split_levels = counter_dict(task.get("level") for task in split_tasks)
        split_subtypes = counter_dict(task.get("subtype") for task in split_tasks)
        if split_levels != EXPECTED_SPLIT_LEVELS[split_name]:
            issues.append(
                issue(
                    "wrong_split_level_counts",
                    "{} level quotas do not match".format(split_name),
                )
            )
        if split_subtypes != EXPECTED_SPLIT_SUBTYPES[split_name]:
            issues.append(
                issue(
                    "wrong_split_subtype_counts",
                    "{} subtype quotas do not match".format(split_name),
                )
            )

    manifest_ids = set(ids)
    pilot_ids = set(splits.get("pilot", []))
    held_ids = set(splits.get("held_out", []))
    if pilot_ids & held_ids or pilot_ids | held_ids != manifest_ids:
        issues.append(
            issue("split_membership_mismatch", "split ids do not partition tasks")
        )
    ablation_ids = splits.get("ablation", [])
    if len(ablation_ids) != 60 or len(set(ablation_ids)) != 60:
        issues.append(
            issue("wrong_ablation_count", "ablation split must contain 60 unique tasks")
        )
    if not set(ablation_ids) <= held_ids:
        issues.append(
            issue("ablation_split_leakage", "ablation tasks must be held out")
        )
    ablation_tasks = [task for task in tasks if task.get("id") in set(ablation_ids)]
    if counter_dict(task.get("level") for task in ablation_tasks) != {
        "L1": 13,
        "L2": 21,
        "L3": 26,
    }:
        issues.append(
            issue("wrong_ablation_levels", "ablation level quotas do not match")
        )
    for task in tasks:
        expected_split = (
            "pilot" if task["id"] in pilot_ids else "held_out"
        )
        if task.get("split") != expected_split:
            issues.append(
                issue(
                    "split_label_mismatch",
                    "manifest and split file disagree",
                    task["id"],
                )
            )

    pilot_shots = set(int(value) for value in splits.get("pilot_shots", []))
    if pilot_shots != EXPECTED_PILOT_SHOTS:
        issues.append(
            issue("wrong_pilot_shots", "pilot shots differ from the frozen set")
        )
    for task in tasks:
        task_shots = set(int(target["shot"]) for target in task["targets"])
        if task.get("split") == "held_out" and task_shots & pilot_shots:
            issues.append(
                issue(
                    "split_shot_leakage",
                    "held-out task contains a pilot shot",
                    task["id"],
                )
            )
        if task.get("split") == "pilot" and not task_shots <= pilot_shots:
            issues.append(
                issue(
                    "split_shot_leakage",
                    "pilot task contains a non-pilot shot",
                    task["id"],
                )
            )

    target_usage = collections.Counter(
        target_key(target) for task in tasks for target in task["targets"]
    )
    for key, count in target_usage.items():
        if count > 2:
            issues.append(
                issue(
                    "target_usage_exceeded",
                    "target {} is used {} times".format(key, count),
                )
            )

    shots = [int(target["shot"]) for task in tasks for target in task["targets"]]
    subtrees = [target["subtree"] for task in tasks for target in task["targets"]]
    signals_by_split = collections.defaultdict(set)
    for task in tasks:
        for target in task["targets"]:
            signals_by_split[task["split"]].add(target["signal_name"])
    report = {
        "task_count": len(tasks),
        "by_level": by_level,
        "by_subtype": by_subtype,
        "by_split": by_split,
        "by_shot": counter_dict(shots),
        "by_subtree": counter_dict(subtrees),
        "maximum_target_usage": max(target_usage.values()) if target_usage else 0,
        "signal_name_overlap_across_splits": sorted(
            signals_by_split["pilot"] & signals_by_split["held_out"]
        ),
    }
    return issues, report


def audit_prompts(manifest, prompts):
    issues = []
    task_by_id = {task["id"]: task for task in manifest.get("tasks", [])}
    prompt_ids = [record.get("task_id") for record in prompts]
    prompt_by_id = {record.get("task_id"): record for record in prompts}
    if len(prompt_ids) != len(set(prompt_ids)):
        issues.append(issue("duplicate_prompt", "prompt task ids are not unique"))

    for task_id, task in task_by_id.items():
        record = prompt_by_id.get(task_id)
        if record is None:
            issues.append(issue("missing_prompt", "prompt is missing", task_id))
            continue
        prompt = record.get("prompt", "")
        hits = [field for field in FORBIDDEN_PROMPT_FIELDS if field in prompt]
        if hits:
            issues.append(
                issue(
                    "prompt_oracle_disclosure",
                    "forbidden fields disclosed: {}".format(", ".join(hits)),
                    task_id,
                )
            )
        operation = task.get("operation", {})
        windows = []
        if "index_start" in operation and "index_end" in operation:
            windows.append(operation)
        windows.extend(operation.get("per_target_windows", []))
        fallback = operation.get("fallback_window")
        if fallback:
            windows.append(fallback)
        disclosed_slices = [
            "{}:{}".format(window["index_start"], window["index_end"])
            for window in windows
            if "index_start" in window and "index_end" in window
            and "{}:{}".format(window["index_start"], window["index_end"])
            in prompt
        ]
        oracle_values = []
        if task.get("spec_sha256") and task["spec_sha256"] in prompt:
            oracle_values.append(task["spec_sha256"])
        expected_decision = operation.get("expected_decision")
        if expected_decision and expected_decision in prompt:
            oracle_values.append(expected_decision)
        oracle_values.extend(disclosed_slices)
        if oracle_values:
            issues.append(
                issue(
                    "prompt_oracle_value_disclosure",
                    "oracle values disclosed: {}".format(", ".join(oracle_values)),
                    task_id,
                )
            )
        if not task.get("prompt_spec", {}).get("disclose_subtree", False):
            leaked = [
                target["subtree"]
                for target in task["targets"]
                if target["subtree"] in prompt
            ]
            if leaked:
                issues.append(
                    issue(
                        "prompt_subtree_disclosure",
                        "hidden subtree disclosed: {}".format(", ".join(leaked)),
                        task_id,
                    )
                )
    unexpected = sorted(set(prompt_ids) - set(task_by_id))
    for task_id in unexpected:
        issues.append(
            issue("unexpected_prompt", "prompt has no manifest task", task_id)
        )
    report = {
        "prompt_count": len(prompts),
        "unique_prompt_task_ids": len(set(prompt_ids)),
        "unique_prompt_texts": len(
            set(record.get("prompt", "") for record in prompts)
        ),
    }
    if report["unique_prompt_texts"] != len(prompts):
        issues.append(issue("duplicate_prompt_text", "prompt texts are not unique"))
    return issues, report


def _reference_signature(value):
    targets = []
    for target in value.get("targets", []):
        targets.append(
            {
                "signal": target.get("signal", {}).get("sha256"),
                "time": target.get("time_axis", {}).get("sha256"),
                "validation": target.get("validation", {}),
                "target": target.get("target"),
                "expressions": target.get("expressions"),
                "window": target.get("window"),
            }
        )
    derived = value.get("derived", {})
    return {
        "status": value.get("status"),
        "task_id": value.get("task_id"),
        "resource_decisions": value.get("resource_decisions"),
        "targets": targets,
        "common_time": derived.get("common_time_axis", {}).get("sha256"),
        "aligned_first": derived.get("aligned_first", {}).get("sha256"),
        "aligned_second": derived.get("aligned_second", {}).get("sha256"),
        "comparison": derived.get("comparison"),
    }


def _resolved(path, base_dir):
    if os.path.isabs(path) or base_dir is None:
        return path
    return os.path.join(base_dir, path)


def expected_window(task, target_index, expected_allow):
    subtype = task.get("subtype")
    operation = task.get("operation", {})
    if subtype in ("exact_window", "sampling_constrained"):
        return operation
    if subtype in ("same_shot_alignment", "cross_shot_comparison"):
        return operation.get("per_target_windows", [])[target_index]
    if subtype == "adaptive_access" and not expected_allow:
        return operation.get("fallback_window")
    return None


def expected_expressions(target, window):
    node = str(target["node"])
    if window is None:
        return {"signal": node, "time": "dim_of({})".format(node)}
    return {
        "signal": "data({})[{}:{}]".format(
            node, int(window["index_start"]), int(window["index_end"])
        ),
        "time": "dim_of({})[{}:{}]".format(
            node, int(window["index_start"]), int(window["index_end"])
        ),
    }


def audit_references(manifest, reference_index, base_dir=None):
    issues = []
    timings = []
    successful = 0
    stable = 0
    failed_tasks = []
    indexed = reference_index.get("tasks", {})
    manifest_ids = [task["id"] for task in manifest.get("tasks", [])]
    task_by_id = {task["id"]: task for task in manifest.get("tasks", [])}
    if int(reference_index.get("timeout_seconds", -1)) != 90:
        issues.append(
            issue("wrong_timeout_policy", "reference timeout must be exactly 90 seconds")
        )

    for task_id in manifest_ids:
        entry = indexed.get(task_id)
        if entry is None:
            issues.append(
                issue("missing_reference", "reference entry is missing", task_id)
            )
            failed_tasks.append({"task_id": task_id, "reason": "missing"})
            continue
        if not entry.get("stability", {}).get("stable", False):
            issues.append(
                issue("unstable_reference", "repeat comparison is unstable", task_id)
            )
            failed_tasks.append({"task_id": task_id, "reason": "unstable"})
            continue
        repeat_files = entry.get("repeat_files", [])
        if len(repeat_files) != 2:
            issues.append(
                issue(
                    "wrong_repeat_count",
                    "exactly two repeat files are required",
                    task_id,
                )
            )
            continue
        if len(set(os.path.abspath(path) for path in repeat_files)) != 2:
            issues.append(
                issue(
                    "duplicate_repeat_file",
                    "repeat files must be distinct",
                    task_id,
                )
            )
        exit_statuses = entry.get("exit_statuses", [])
        if exit_statuses != [0, 0]:
            issues.append(
                issue(
                    "nonzero_reference_exit",
                    "repeat exit statuses must both be zero",
                    task_id,
                )
            )
        values = []
        missing_file = False
        for path in repeat_files:
            path = _resolved(path, base_dir)
            if not os.path.isfile(path):
                issues.append(
                    issue("missing_reference_file", "{} is missing".format(path), task_id)
                )
                missing_file = True
                continue
            values.append(read_json(path))
        if missing_file or len(values) != 2:
            continue
        if _reference_signature(values[0]) != _reference_signature(values[1]):
            issues.append(
                issue(
                    "unstable_reference",
                    "repeat files disagree despite stable index flag",
                    task_id,
                )
            )
            continue
        stable += 1
        task_success = True
        task = task_by_id[task_id]
        for repeat_position, value in enumerate(values, 1):
            if value.get("task_id") != task_id:
                issues.append(
                    issue(
                        "reference_task_mismatch",
                        "embedded task id does not match index entry",
                        task_id,
                    )
                )
            if int(value.get("repeat_id", -1)) != repeat_position:
                issues.append(
                    issue(
                        "reference_repeat_mismatch",
                        "embedded repeat id does not match file position",
                        task_id,
                    )
                )
            if value.get("status") != "success":
                task_success = False
                issues.append(
                    issue(
                        "reference_execution_failed",
                        "repeat status is {}".format(value.get("status")),
                        task_id,
                    )
                )
                continue
            timings.append(float(value.get("elapsed_seconds", 0.0)))
            decisions = value.get("resource_decisions", [])
            target_results = value.get("targets", [])
            if len(target_results) != len(task["targets"]):
                issues.append(
                    issue(
                        "reference_target_count_mismatch",
                        "reference target count differs from manifest",
                        task_id,
                    )
                )
            for position, target_result in enumerate(target_results):
                if position >= len(task["targets"]) or not isinstance(
                    target_result, dict
                ):
                    continue
                expected_target = task["targets"][position]
                if target_result.get("target") != expected_target:
                    issues.append(
                        issue(
                            "reference_target_mismatch",
                            "embedded target differs from manifest",
                            task_id,
                        )
                    )
                decision = decisions[position] if position < len(decisions) else {}
                policy = task.get("safety_policy", {})
                expected_allow = (
                    int(expected_target["sample_num"])
                    <= int(policy.get("max_points", 5000000))
                    and float(expected_target["estimated_total_mib"])
                    <= float(policy.get("max_total_mib", 256.0))
                )
                if decision.get("allow_full_read") is not expected_allow:
                    issues.append(
                        issue(
                            "resource_decision_mismatch",
                            "recorded decision differs from manifest policy",
                            task_id,
                        )
                    )
                required_window = expected_window(task, position, expected_allow)
                if target_result.get("window") != required_window:
                    issues.append(
                        issue(
                            "reference_window_mismatch",
                            "reference window differs from task operation",
                            task_id,
                        )
                    )
                if target_result.get("expressions") != expected_expressions(
                    expected_target, required_window
                ):
                    issues.append(
                        issue(
                            "reference_expression_mismatch",
                            "access expressions differ from task operation",
                            task_id,
                        )
                    )
                if (
                    not expected_allow
                    and target_result.get("window") is None
                ):
                    issues.append(
                        issue(
                            "unsafe_full_read",
                            "resource guard was bypassed",
                            task_id,
                        )
                    )
                validation = target_result.get("validation", {})
                if validation.get("signal_all_finite") is not True:
                    issues.append(
                        issue(
                            "invalid_signal_values",
                            "signal contains non-finite values",
                            task_id,
                        )
                    )
                time_required = (
                    "non_empty",
                    "same_size",
                    "time_all_finite",
                    "time_strictly_increasing",
                    "expected_point_count_match",
                )
                if not all(
                    validation.get(field) is True for field in time_required
                ):
                    issues.append(
                        issue(
                            "invalid_time_axis",
                            "target validation failed",
                            task_id,
                        )
                    )
                if target_result.get("window") is not None:
                    window_required = (
                        "window_start_match",
                        "window_end_match",
                        "time_step_match",
                    )
                    if not all(
                        validation.get(field) is True
                        for field in window_required
                    ):
                        issues.append(
                            issue(
                                "invalid_window",
                                "window endpoint or step validation failed",
                                task_id,
                            )
                        )
            if len(task["targets"]) == 2:
                derived = value.get("derived", {})
                common_points = int(task.get("operation", {}).get("common_points", 0))
                arrays = (
                    derived.get("common_time_axis", {}),
                    derived.get("aligned_first", {}),
                    derived.get("aligned_second", {}),
                )
                comparison = derived.get("comparison")
                required_metrics = (
                    "a_mean",
                    "a_std",
                    "b_mean",
                    "b_std",
                    "pearson_correlation",
                )
                if (
                    not common_points
                    or any(
                        array.get("size") != common_points
                        or not array.get("sha256")
                        for array in arrays
                    )
                    or not isinstance(comparison, dict)
                    or any(metric not in comparison for metric in required_metrics)
                ):
                    issues.append(
                        issue(
                            "missing_derived_result",
                            "pair task lacks complete aligned outputs or metrics",
                            task_id,
                        )
                    )
        if task_success:
            successful += 1

    unexpected = sorted(set(indexed) - set(manifest_ids))
    for task_id in unexpected:
        issues.append(
            issue("unexpected_reference", "reference has no manifest task", task_id)
        )
    timing_report = {}
    if timings:
        timing_report = {
            "minimum_seconds": min(timings),
            "median_seconds": statistics.median(timings),
            "maximum_seconds": max(timings),
        }
    report = {
        "indexed_task_count": len(indexed),
        "stable_task_count": stable,
        "successful_task_count": successful,
        "repeat_timing": timing_report,
        "failed_tasks": failed_tasks,
    }
    return issues, report


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


def run_audit(manifest_path, splits_path, prompts_path, references_path):
    manifest = read_json(manifest_path)
    splits = read_json(splits_path)
    prompts = read_jsonl(prompts_path)
    reference_index = read_json(references_path)
    manifest_issues, manifest_report = audit_manifest_structure(manifest, splits)
    prompt_issues, prompt_report = audit_prompts(manifest, prompts)
    reference_issues, reference_report = audit_references(
        manifest, reference_index, os.path.dirname(os.path.abspath(references_path))
    )
    issues = manifest_issues + prompt_issues + reference_issues
    return {
        "status": "pass" if not issues else "fail",
        "blocking_issue_count": len(issues),
        "issues": issues,
        "manifest": manifest_report,
        "prompts": prompt_report,
        "references": reference_report,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--references", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    report = run_audit(
        args.manifest, args.splits, args.prompts, args.references
    )
    atomic_write_json(args.output, report)
    print("audit_status={}".format(report["status"]))
    print("blocking_issue_count={}".format(report["blocking_issue_count"]))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())

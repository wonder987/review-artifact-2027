#!/usr/bin/env python3
"""Normalize MongoDB candidates and validate them with safety gates."""

from __future__ import print_function

import argparse
import json
import os
import subprocess
import sys


MAX_POINTS = 5000000
MAX_TOTAL_MIB = 256.0
EXISTING_GOLD = {
    (106915, "t2", "AXUV_DIV1"),
    (106915, "t2", "CCR1"),
    (106915, "t5", "NA1"),
}
REQUIRED_FIELDS = {
    "shot",
    "tree",
    "subtree",
    "signal_name",
    "sample_num",
    "frequency",
    "daqtime",
    "trigtime",
    "size_band",
    "estimated_total_mib",
}


def normalize_extended_json(value):
    if isinstance(value, list):
        return [normalize_extended_json(item) for item in value]
    if isinstance(value, dict):
        if set(value) == {"$numberDouble"}:
            return float(value["$numberDouble"])
        if set(value) in ({"$numberInt"}, {"$numberLong"}):
            return int(next(iter(value.values())))
        if set(value) == {"$numberDecimal"}:
            return float(value["$numberDecimal"])
        return {
            key: normalize_extended_json(item)
            for key, item in value.items()
        }
    return value


def candidate_key(candidate):
    return (
        int(candidate["shot"]),
        str(candidate["subtree"]),
        str(candidate["signal_name"]),
    )


def task_id(candidate):
    signal = str(candidate["signal_name"]).lower()
    return "east_{}_{}_{}".format(
        signal, int(candidate["shot"]), str(candidate["size_band"])
    )


def validate_candidates(candidates):
    seen = set()
    for index, candidate in enumerate(candidates):
        missing = sorted(REQUIRED_FIELDS - set(candidate))
        if missing:
            raise ValueError(
                "candidate {} is missing fields: {}".format(index, missing)
            )
        key = candidate_key(candidate)
        if key in seen:
            raise ValueError("duplicate candidate: {}".format(key))
        seen.add(key)


def build_plan(candidates):
    normalized = normalize_extended_json(candidates)
    validate_candidates(normalized)
    plan = []
    for candidate in normalized:
        if candidate_key(candidate) in EXISTING_GOLD:
            continue
        safe_by_numbers = (
            int(candidate["sample_num"]) <= MAX_POINTS
            and float(candidate["estimated_total_mib"]) <= MAX_TOTAL_MIB
        )
        actions = ["preflight"]
        if candidate["size_band"] == "safe_full" and safe_by_numbers:
            actions.append("full_retrieval")
        item = dict(candidate)
        item["task_id"] = task_id(candidate)
        item["node"] = "\\" + str(candidate["signal_name"])
        item["planned_actions"] = actions
        plan.append(item)
    return plan


def write_json(path, value):
    directory = os.path.dirname(os.path.abspath(path))
    if not os.path.isdir(directory):
        os.makedirs(directory)
    with open(path, "w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def read_json(path):
    with open(path, "r") as handle:
        return json.load(handle)


def run_command(command, timeout_seconds):
    try:
        return subprocess.call(command, timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        return 124


def execute_plan(plan, scripts_dir, results_dir, timeout_seconds):
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    summary = []
    for position, task in enumerate(plan, 1):
        prefix = "candidate_{:03d}_{}".format(position, task["task_id"])
        preflight_output = os.path.join(
            results_dir, prefix + "_preflight.json"
        )
        preflight_command = [
            sys.executable,
            os.path.join(scripts_dir, "preflight_mds_task.py"),
            "--tree", str(task["tree"]),
            "--shot", str(task["shot"]),
            "--node", str(task["node"]),
            "--default-path", str(task["subtree"]),
            "--max-points", str(MAX_POINTS),
            "--max-total-mib", str(MAX_TOTAL_MIB),
            "--output", preflight_output,
        ]
        preflight_status = run_command(preflight_command, timeout_seconds)
        record = {
            "task_id": task["task_id"],
            "preflight_exit_status": preflight_status,
            "preflight_output": preflight_output,
            "full_retrieval_attempted": False,
        }

        live_allows_full = False
        if preflight_status == 0 and os.path.isfile(preflight_output):
            result = read_json(preflight_output)
            live_allows_full = bool(
                result.get("decision", {}).get("allow_full_read", False)
            )

        if "full_retrieval" in task["planned_actions"] and live_allows_full:
            full_output = os.path.join(results_dir, prefix + "_full.json")
            full_command = [
                sys.executable,
                os.path.join(scripts_dir, "validate_mds_task.py"),
                "--tree", str(task["tree"]),
                "--shot", str(task["shot"]),
                "--node", str(task["node"]),
                "--default-path", str(task["subtree"]),
                "--output", full_output,
            ]
            record["full_retrieval_attempted"] = True
            record["full_retrieval_exit_status"] = run_command(
                full_command, timeout_seconds
            )
            record["full_retrieval_output"] = full_output
        summary.append(record)
        write_json(os.path.join(results_dir, "batch_summary.json"), summary)
    return summary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--plan-output", required=True)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--results-dir")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--timeout-seconds", type=int, default=90)
    return parser.parse_args()


def main():
    args = parse_args()
    candidates = read_json(args.candidates)
    normalized = normalize_extended_json(candidates)
    plan = build_plan(normalized)
    if args.limit is not None:
        plan = plan[:args.limit]
    write_json(args.candidates, normalized)
    write_json(args.plan_output, plan)
    print("normalized_candidates={}".format(len(normalized)))
    print("planned_candidates={}".format(len(plan)))

    if args.run:
        if not args.results_dir:
            raise ValueError("--results-dir is required with --run")
        scripts_dir = os.path.dirname(os.path.abspath(__file__))
        summary = execute_plan(
            plan, scripts_dir, args.results_dir, args.timeout_seconds
        )
        attempted = sum(
            1 for item in summary if item["full_retrieval_attempted"]
        )
        print("full_retrieval_attempted={}".format(attempted))
    return 0


if __name__ == "__main__":
    sys.exit(main())

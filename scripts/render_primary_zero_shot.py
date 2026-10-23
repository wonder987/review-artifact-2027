#!/usr/bin/env python3
"""Render the minimal Primary Zero-shot prompt condition for Gold-138."""

from __future__ import print_function

import argparse
import hashlib
import json
import os


PROMPT_VERSION = "primary-zero-shot-v1"

SYSTEM_PROMPT = (
    "You generate Python 3.6 programs for accessing EAST scientific signals "
    "through the MDSplus client. Use only Python, MDSplus, NumPy, and the "
    "standard library. Return only one self-contained executable Python "
    "program, without Markdown fences or explanatory text."
)

OUTPUT_CONTRACT = (
    "Print exactly one JSON object to stdout. It must summarize execution "
    "status, accessed objects, signal and time-axis sizes, validation results, "
    "and elapsed seconds. On failure, report the failed stage, error type, and "
    "error message, then exit with a nonzero status. Do not fabricate data or "
    "silently substitute another signal."
)


def format_number(value):
    return "{:.12g}".format(float(value))


def visible_target(target, disclose_subtree):
    parts = [
        "server {}".format(target["server"]),
        "tree {}".format(target["tree"]),
        "shot {}".format(target["shot"]),
        "signal {}".format(target["signal_name"]),
        "node {}".format(target["node"]),
    ]
    if disclose_subtree:
        parts.append("subtree {}".format(target["subtree"]))
    return ", ".join(parts)


def window_request(window):
    return (
        "the interval from {} to {} seconds with {} output points"
    ).format(
        format_number(window["time_start"]),
        format_number(window["time_end"]),
        int(window["expected_points"]),
    )


def render_pair_request(task, relationship):
    operation = task["operation"]
    descriptions = []
    for position, (target, window) in enumerate(
        zip(task["targets"], operation["per_target_windows"]), 1
    ):
        descriptions.append(
            "Input {} is {}; retrieve {}.".format(
                position,
                visible_target(target, False),
                window_request(window),
            )
        )
    descriptions.append(
        "{} Align the two results over their shared time range on a common "
        "grid of {} points without extrapolation. Return each aligned "
        "signal's mean and standard deviation and their Pearson correlation."
        .format(relationship, int(operation["common_points"]))
    )
    return " ".join(descriptions)


def render_task_request(task):
    subtype = task["subtype"]
    target = task["targets"][0]
    disclose_subtree = bool(task["prompt_spec"]["disclose_subtree"])

    if subtype == "simple_retrieval":
        request = (
            "Retrieve the complete signal and its time axis for {}."
        ).format(visible_target(target, disclose_subtree))
    elif subtype == "subtree_discovery":
        request = (
            "Retrieve the complete signal and its time axis for {}. The "
            "subtree is not specified; determine the applicable subtree at "
            "runtime rather than assuming one."
        ).format(visible_target(target, False))
    elif subtype == "exact_window":
        request = (
            "Retrieve {} for {} while preserving the signal's native time "
            "ordering."
        ).format(
            window_request(task["operation"]),
            visible_target(target, disclose_subtree),
        )
    elif subtype == "sampling_constrained":
        operation = task["operation"]
        request = (
            "Retrieve exactly {} samples beginning at {} seconds for {}, "
            "using the signal's native sampling interval and preserving time "
            "ordering."
        ).format(
            int(operation["expected_points"]),
            format_number(operation["time_start"]),
            visible_target(target, disclose_subtree),
        )
    elif subtype == "same_shot_alignment":
        request = render_pair_request(
            task,
            "Compare the two different signals from the same shot.",
        )
    elif subtype == "cross_shot_comparison":
        request = render_pair_request(
            task,
            "Compare the same signal across the two shots.",
        )
    elif subtype == "adaptive_access":
        policy = task["safety_policy"]
        request = (
            "Retrieve data for {} subject to hard access limits of {:,} "
            "points and {} MiB for the combined signal and time axis. Estimate "
            "the access size before reading samples. Return the complete "
            "signal only when both limits are satisfied; otherwise return a "
            "bounded preview of 10,001 samples beginning at 0 seconds with "
            "the native sampling interval."
        ).format(
            visible_target(target, False),
            int(policy["max_points"]),
            format_number(policy["max_total_mib"]),
        )
    else:
        raise ValueError("unsupported task subtype: {}".format(subtype))

    return request


def render_prompt(task):
    user_prompt = render_task_request(task) + "\n\n" + OUTPUT_CONTRACT
    return {
        "system_prompt": SYSTEM_PROMPT,
        "user_prompt": user_prompt,
        "prompt": SYSTEM_PROMPT + "\n\n" + user_prompt,
    }


def _prompt_hash(prompt):
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def render_manifest(manifest_path, output_path):
    with open(manifest_path, "r") as handle:
        manifest = json.load(handle)

    records = []
    for task in manifest["tasks"]:
        rendered = render_prompt(task)
        records.append({
            "level": task["level"],
            "prompt": rendered["prompt"],
            "prompt_sha256": _prompt_hash(rendered["prompt"]),
            "prompt_version": PROMPT_VERSION,
            "split": task["split"],
            "subtype": task["subtype"],
            "system_prompt": rendered["system_prompt"],
            "task_id": task["id"],
            "user_prompt": rendered["user_prompt"],
        })

    directory = os.path.dirname(os.path.abspath(output_path))
    if not os.path.isdir(directory):
        os.makedirs(directory)
    temporary = output_path + ".tmp"
    with open(temporary, "w") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output_path)
    return records


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    records = render_manifest(args.manifest, args.output)
    print("rendered_prompts={}".format(len(records)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Render model-facing Gold-138 prompts without exposing reference oracles."""

from __future__ import print_function

import argparse
import json
import os


SHARED_INSTRUCTIONS = (
    "Return one self-contained, executable Python program using the MDSplus "
    "client. Do not fabricate unavailable data or silently replace failed "
    "reads. The program must print a JSON result summary containing the task "
    "identifier, execution status, accessed objects, array sizes, validation "
    "checks, and elapsed time."
)


def _number(value):
    return "{:.12g}".format(float(value))


def _target_description(target, disclose_subtree):
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


def _window_description(window):
    return (
        "time interval [{}, {}] seconds with {} output points and nominal "
        "step {} seconds"
    ).format(
        _number(window["time_start"]),
        _number(window["time_end"]),
        int(window["expected_points"]),
        _number(window["time_step"]),
    )


def _render_pair_task(task, relationship):
    operation = task["operation"]
    lines = [relationship]
    for position, (target, window) in enumerate(
        zip(task["targets"], operation["per_target_windows"]), 1
    ):
        lines.append(
            "Input {}: {}; retrieve {}.".format(
                position,
                _target_description(target, False),
                _window_description(window),
            )
        )
    lines.append(
        "Align both signals on their overlapping interval using a common time "
        "grid of {} points without extrapolation.".format(
            int(operation["common_points"])
        )
    )
    lines.append(
        "Report the mean and standard deviation of each aligned signal and "
        "their Pearson correlation, together with alignment validation."
    )
    return "\n".join(lines)


def render_prompt(task):
    subtype = task["subtype"]
    target = task["targets"][0]
    disclose_subtree = bool(task["prompt_spec"]["disclose_subtree"])
    header = "Task {} ({} / {}).".format(
        task["id"], task["level"], subtype
    )

    if subtype == "simple_retrieval":
        body = (
            "Read the complete signal and its time axis from {}. Verify that "
            "both arrays are non-empty, equal in length, finite, and that the "
            "time axis is strictly increasing."
        ).format(_target_description(target, disclose_subtree))
    elif subtype == "subtree_discovery":
        body = (
            "Locate the correct subtree for the requested signal, then read "
            "the complete signal and time axis from {}. Verify non-empty "
            "equal-length arrays and a strictly increasing time axis."
        ).format(_target_description(target, False))
    elif subtype in ("exact_window", "sampling_constrained"):
        window = task["operation"]
        action = (
            "Retrieve exactly the requested time window"
            if subtype == "exact_window"
            else "Retrieve a sampling-constrained window"
        )
        body = (
            "{} from {}: {}. Do not read the complete signal before applying "
            "the requested bound. Validate the point count, endpoints, equal "
            "signal/time lengths, finiteness, and strictly increasing time."
        ).format(
            action,
            _target_description(target, disclose_subtree),
            _window_description(window),
        )
    elif subtype == "cross_shot_comparison":
        body = _render_pair_task(
            task,
            "Compare the same signal across two shots.",
        )
    elif subtype == "same_shot_alignment":
        body = _render_pair_task(
            task,
            "Retrieve and align two different signals from the same shot.",
        )
    elif subtype == "adaptive_access":
        policy = task["safety_policy"]
        body = (
            "Plan a resource-aware access to {}. Before reading signal "
            "samples, inspect available metadata to estimate point count and "
            "combined signal/time-axis memory. Read the complete signal only "
            "when the estimate is at most {:,} points and at most {} MiB. If "
            "either limit is exceeded, avoid the complete read and retrieve a "
            "bounded preview of 10,001 samples beginning at t=0 using the "
            "native sampling interval. Report the estimate, selected action, "
            "and validation results."
        ).format(
            _target_description(target, False),
            int(policy["max_points"]),
            _number(policy["max_total_mib"]),
        )
    else:
        raise ValueError("unsupported task subtype: {}".format(subtype))

    return "\n\n".join((header, body, SHARED_INSTRUCTIONS))


def render_manifest(manifest_path, output_path):
    with open(manifest_path, "r") as handle:
        manifest = json.load(handle)
    records = [
        {
            "task_id": task["id"],
            "level": task["level"],
            "subtype": task["subtype"],
            "split": task["split"],
            "prompt": render_prompt(task),
        }
        for task in manifest["tasks"]
    ]
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

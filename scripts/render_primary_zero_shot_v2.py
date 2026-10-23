#!/usr/bin/env python3
"""Render the semantic Primary Zero-shot V2 prompt condition."""

from __future__ import print_function

import argparse
import hashlib
import json
import os

try:
    from scripts.render_primary_zero_shot import SYSTEM_PROMPT, render_task_request
except ImportError:
    from render_primary_zero_shot import SYSTEM_PROMPT, render_task_request


PROMPT_VERSION = "primary-zero-shot-v2"

SUMMARY_FIELDS = (
    "size, shape, dtype, sha256, finite_count, min, max, mean, and std"
)

BASE_OUTPUT_CONTRACT = (
    "Print exactly one JSON object to stdout. On success, status must be "
    "success and the object must contain accessed_objects, elapsed_seconds, "
    "and an ordered targets list with one item per requested signal. Each "
    "targets[i] item must contain targets[i].target, an identity object with server, tree, "
    "shot, node, and any discovered subtree; targets[i].signal and "
    "targets[i].time_axis must be summary objects "
    "with {fields}. Compute each sha256 from "
    "np.ascontiguousarray(array).tobytes() without casting or rounding. "
    "Each targets[i].validation object must contain non_empty, same_size, "
    "signal_all_finite, time_all_finite, and time_strictly_increasing. "
    "Convert NumPy scalars and Booleans to ordinary "
    "Python JSON values and do not print raw arrays. On failure, print status, "
    "failed_stage, error_type, error_message, accessed_objects, and "
    "elapsed_seconds, then exit with a nonzero status. Do not fabricate data "
    "or silently substitute another signal."
).format(fields=SUMMARY_FIELDS)

PAIR_OUTPUT_CONTRACT = (
    " For this two-target request, also include derived with summaries for "
    "common_time_axis, aligned_first, and aligned_second using the same "
    "summary fields, plus comparison containing count, a_mean, a_std, "
    "b_mean, b_std, and pearson_correlation."
)

ADAPTIVE_OUTPUT_CONTRACT = (
    " For this bounded-access request, also include resource_decision with "
    "allow_full_read, within_point_limit, within_memory_limit, points, "
    "total_mib, max_points, max_total_mib, and recommended_action."
)


def output_contract(task):
    contract = BASE_OUTPUT_CONTRACT
    if task["subtype"] in ("same_shot_alignment", "cross_shot_comparison"):
        contract += PAIR_OUTPUT_CONTRACT
    if task["subtype"] == "adaptive_access":
        contract += ADAPTIVE_OUTPUT_CONTRACT
    return contract


def render_prompt(task):
    user_prompt = render_task_request(task) + "\n\n" + output_contract(task)
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


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    records = render_manifest(args.manifest, args.output)
    print("rendered_prompts={}".format(len(records)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

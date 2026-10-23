#!/usr/bin/env python3
"""Execute one Gold-138 reference task against MDSplus."""

from __future__ import print_function

import argparse
import json
import os
import sys
import time

import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import gold_reference_ops as ops


def default_connection_factory(server):
    from MDSplus import Connection

    return Connection(server)


def build_target_expressions(target, window=None):
    node = str(target["node"])
    if window is None:
        return {"signal": node, "time": "dim_of({})".format(node)}
    start = int(window["index_start"])
    end = int(window["index_end"])
    return {
        "signal": "data({})[{}:{}]".format(node, start, end),
        "time": "dim_of({})[{}:{}]".format(node, start, end),
    }


def target_decision(target, policy):
    return ops.resource_decision(
        target["sample_num"],
        target["estimated_total_mib"],
        policy.get("max_points", 5000000),
        policy.get("max_total_mib", 256.0),
    )


def operation_window(task, target_index):
    subtype = task["subtype"]
    operation = task["operation"]
    if subtype in ("exact_window", "sampling_constrained"):
        return operation
    if subtype in ("same_shot_alignment", "cross_shot_comparison"):
        return operation["per_target_windows"][target_index]
    if subtype == "adaptive_access":
        return operation.get("fallback_window")
    return None


def validate_arrays(signal, time_axis, window=None):
    signal = np.asarray(signal).reshape(-1)
    time_axis = np.asarray(time_axis).reshape(-1)
    deltas = np.diff(time_axis.astype(np.float64))
    result = {
        "non_empty": bool(signal.size > 0),
        "same_size": bool(signal.size == time_axis.size),
        "signal_all_finite": bool(
            np.issubdtype(signal.dtype, np.number)
            and np.all(np.isfinite(signal))
        ),
        "time_all_finite": bool(
            np.issubdtype(time_axis.dtype, np.number)
            and np.all(np.isfinite(time_axis))
        ),
        "time_monotonic_non_decreasing": bool(
            time_axis.size < 2 or np.all(deltas >= 0)
        ),
        "time_strictly_increasing": bool(
            time_axis.size < 2 or np.all(deltas > 0)
        ),
    }
    if window is not None and all(
        field in window for field in ("time_start", "time_end", "time_step")
    ):
        expected_step = float(window["time_step"])
        # EAST time dimensions can be quantized at approximately 1 ns. Keep
        # the tolerance far below one sample while accepting that encoding.
        endpoint_atol = max(abs(expected_step) * 1e-2, 2e-9)
        step_atol = max(abs(expected_step) * 1e-4, 2e-9)
        result.update(
            {
                "window_start_match": bool(
                    time_axis.size
                    and np.isclose(
                        float(time_axis[0]),
                        float(window["time_start"]),
                        rtol=1e-7,
                        atol=endpoint_atol,
                    )
                ),
                "window_end_match": bool(
                    time_axis.size
                    and np.isclose(
                        float(time_axis[-1]),
                        float(window["time_end"]),
                        rtol=1e-7,
                        atol=endpoint_atol,
                    )
                ),
                "time_step_match": bool(
                    deltas.size
                    and np.allclose(
                        deltas,
                        expected_step,
                        rtol=1e-5,
                        atol=step_atol,
                    )
                ),
                "minimum_time_step": (
                    float(deltas.min()) if deltas.size else None
                ),
                "maximum_time_step": (
                    float(deltas.max()) if deltas.size else None
                ),
                "median_time_step": (
                    float(np.median(deltas)) if deltas.size else None
                ),
                "endpoint_tolerance": endpoint_atol,
                "time_step_tolerance": step_atol,
                "maximum_time_step_error": (
                    float(np.max(np.abs(deltas - expected_step)))
                    if deltas.size
                    else None
                ),
            }
        )
    return result


def validation_passes(validation, window=None):
    required = [
        "non_empty",
        "same_size",
        "signal_all_finite",
        "time_all_finite",
        "time_strictly_increasing",
        "expected_point_count_match",
    ]
    if window is not None and all(
        field in window for field in ("time_start", "time_end", "time_step")
    ):
        required.extend(
            ("window_start_match", "window_end_match", "time_step_match")
        )
    return all(validation.get(field) is True for field in required)


def _blocked_result(task, decisions, started):
    return {
        "task_id": task["id"],
        "status": "blocked",
        "failure_category": "resource_awareness",
        "failure_type": "unsafe_full_read",
        "last_successful_stage": "pre_execution_guard",
        "resource_decisions": decisions,
        "elapsed_seconds": round(time.time() - started, 6),
    }


def execute_task(task, connection_factory=None, repeat_id=None):
    if connection_factory is None:
        connection_factory = default_connection_factory
    started = time.time()
    stage = "start"
    trace = []
    policy = task.get(
        "safety_policy",
        {"max_points": 5000000, "max_total_mib": 256.0},
    )
    decisions = [target_decision(target, policy) for target in task["targets"]]

    if task["operation"].get("kind") == "full_retrieval" and not all(
        decision["allow_full_read"] for decision in decisions
    ):
        return _blocked_result(task, decisions, started)

    result = {
        "task_id": task["id"],
        "repeat_id": repeat_id,
        "level": task.get("level"),
        "subtype": task["subtype"],
        "status": "error",
        "resource_decisions": decisions,
        "targets": [None] * len(task["targets"]),
        "stage_trace": trace,
    }

    groups = {}
    for index, target in enumerate(task["targets"]):
        key = (target["server"], target["tree"], int(target["shot"]))
        groups.setdefault(key, []).append((index, target))

    try:
        for group_key in sorted(groups):
            server, tree, shot = group_key
            stage = "connect"
            stage_started = time.time()
            connection = connection_factory(server)
            trace.append(
                {"stage": stage, "seconds": round(time.time() - stage_started, 6)}
            )
            opened = False
            try:
                stage = "open_tree"
                stage_started = time.time()
                connection.openTree(tree, shot)
                opened = True
                trace.append(
                    {
                        "stage": stage,
                        "seconds": round(time.time() - stage_started, 6),
                        "tree": tree,
                        "shot": shot,
                    }
                )
                for target_index, target in groups[group_key]:
                    stage = "set_default"
                    expression = 'TCL("set def {}")'.format(target["subtree"])
                    connection.get(expression)
                    trace.append(
                        {
                            "stage": stage,
                            "target_index": target_index,
                            "subtree": target["subtree"],
                        }
                    )

                    window = operation_window(task, target_index)
                    if task["subtype"] == "adaptive_access":
                        if decisions[target_index]["allow_full_read"]:
                            window = None
                        elif window is None:
                            raise ValueError("unsafe adaptive task lacks fallback window")
                    expressions = build_target_expressions(target, window)
                    stage = "read"
                    read_started = time.time()
                    signal = np.asarray(connection.get(expressions["signal"]).data())
                    time_axis = np.asarray(connection.get(expressions["time"]).data())
                    read_seconds = round(time.time() - read_started, 6)
                    validation = validate_arrays(signal, time_axis, window)
                    expected_points = (
                        int(window["expected_points"])
                        if window is not None and "expected_points" in window
                        else int(target["sample_num"])
                    )
                    validation["expected_points"] = expected_points
                    validation["expected_point_count_match"] = bool(
                        signal.size == expected_points
                    )
                    result["targets"][target_index] = {
                        "target": target,
                        "expressions": expressions,
                        "window": window,
                        "signal": ops.array_summary(signal),
                        "time_axis": ops.array_summary(time_axis),
                        "validation": validation,
                        "read_seconds": read_seconds,
                        "_signal_values": signal,
                        "_time_values": time_axis,
                    }
                    trace.append(
                        {
                            "stage": "read",
                            "target_index": target_index,
                            "seconds": read_seconds,
                        }
                    )
                    stage = "validation"
                    if not validation_passes(validation, window):
                        raise ValueError(
                            "reference validation failed for target {}".format(
                                target_index
                            )
                        )
                    trace.append({"stage": stage, "target_index": target_index})
            finally:
                if opened:
                    try:
                        connection.closeTree(tree, shot)
                    except Exception:
                        pass

        if len(task["targets"]) == 2:
            stage = "align"
            first = result["targets"][0]
            second = result["targets"][1]
            common_points = int(task["operation"]["common_points"])
            grid, aligned_a, aligned_b = ops.align_on_intersection(
                first["_time_values"],
                first["_signal_values"],
                second["_time_values"],
                second["_signal_values"],
                common_points,
            )
            result["derived"] = {
                "common_time_axis": ops.array_summary(grid),
                "aligned_first": ops.array_summary(aligned_a),
                "aligned_second": ops.array_summary(aligned_b),
                "comparison": ops.comparison_summary(aligned_a, aligned_b),
            }

        for target_result in result["targets"]:
            target_result.pop("_signal_values", None)
            target_result.pop("_time_values", None)
        result["status"] = "success"
        result["last_successful_stage"] = "summarize"
    except Exception as exc:
        for target_result in result["targets"]:
            if target_result:
                target_result.pop("_signal_values", None)
                target_result.pop("_time_values", None)
        result["error_type"] = type(exc).__name__
        result["error_message"] = str(exc)
        result["failed_stage"] = stage
    result["elapsed_seconds"] = round(time.time() - started, 6)
    return result


def write_json(path, value):
    directory = os.path.dirname(os.path.abspath(path))
    if not os.path.isdir(directory):
        os.makedirs(directory)
    with open(path, "w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--repeat-id", type=int, required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.manifest, "r") as handle:
        manifest = json.load(handle)
    matches = [task for task in manifest["tasks"] if task["id"] == args.task_id]
    if len(matches) != 1:
        raise ValueError("task id must match exactly one task")
    result = execute_task(matches[0], repeat_id=args.repeat_id)
    write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "success" else 1


if __name__ == "__main__":
    sys.exit(main())

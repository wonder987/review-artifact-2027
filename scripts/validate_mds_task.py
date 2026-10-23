#!/usr/bin/env python3
"""Run and validate one MDSplus signal retrieval task."""

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
from MDSplus import Connection


def array_summary(values):
    array = np.asarray(values)
    flat = array.reshape(-1)
    numeric = np.issubdtype(array.dtype, np.number)

    summary = {
        "shape": [int(value) for value in array.shape],
        "size": int(array.size),
        "dtype": str(array.dtype),
        "sha256": hashlib.sha256(
            np.ascontiguousarray(array).tobytes()
        ).hexdigest(),
    }

    if numeric:
        finite = np.isfinite(flat)
        summary["finite_count"] = int(finite.sum())
        summary["finite_ratio"] = (
            float(finite.mean()) if flat.size else None
        )
        finite_values = flat[finite]
        if finite_values.size:
            summary.update(
                {
                    "min": float(finite_values.min()),
                    "max": float(finite_values.max()),
                    "mean": float(finite_values.mean()),
                    "std": float(finite_values.std()),
                }
            )

    return summary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="202.127.204.12")
    parser.add_argument("--tree", required=True)
    parser.add_argument("--shot", required=True, type=int)
    parser.add_argument("--node", required=True)
    parser.add_argument("--default-path")
    parser.add_argument("--time-start", type=float)
    parser.add_argument("--time-end", type=float)
    parser.add_argument("--time-step", type=float)
    parser.add_argument("--index-start", type=int)
    parser.add_argument("--index-end", type=int)
    parser.add_argument("--output")
    return parser.parse_args()


def main():
    args = parse_args()
    window_values = (args.time_start, args.time_end, args.time_step)
    has_window = all(value is not None for value in window_values)
    if any(value is not None for value in window_values) and not has_window:
        raise ValueError(
            "--time-start, --time-end, and --time-step must be used together"
        )
    index_values = (args.index_start, args.index_end)
    has_indices = all(value is not None for value in index_values)
    if any(value is not None for value in index_values) and not has_indices:
        raise ValueError(
            "--index-start and --index-end must be used together"
        )

    started = time.time()
    connection = None
    tree_opened = False

    result = {
        "status": "error",
        "server": args.server,
        "tree": args.tree,
        "shot": args.shot,
        "node": args.node,
        "default_path": args.default_path,
    }

    signal_expression = args.node
    time_expression = "dim_of({})".format(args.node)
    access_mode = "full_signal"

    if has_indices:
        signal_expression = "data({})[{}:{}]".format(
            args.node, args.index_start, args.index_end
        )
        time_expression = "dim_of({})[{}:{}]".format(
            args.node, args.index_start, args.index_end
        )
        access_mode = "index_window"
        result["requested_indices"] = {
            "start": args.index_start,
            "end": args.index_end,
        }
    elif has_window:
        signal_expression = "{}[{}:{}:{}]".format(
            args.node, args.time_start, args.time_end, args.time_step
        )

        time_expression = "dim_of({})".format(signal_expression)
        access_mode = "dimension_window"

    if has_window:
        result["requested_window"] = {
            "start": args.time_start,
            "end": args.time_end,
            "step": args.time_step,
        }

    result["access_mode"] = access_mode
    result["signal_expression"] = signal_expression
    result["time_expression"] = time_expression

    try:
        connection = Connection(args.server)
        connected_at = time.time()
        connection.openTree(args.tree, args.shot)
        tree_opened = True

        if args.default_path:
            expression = 'TCL("set def {}")'.format(args.default_path)
            connection.get(expression)

        opened_at = time.time()

        signal = np.asarray(connection.get(signal_expression).data())
        time_axis = np.asarray(connection.get(time_expression).data())
        read_at = time.time()

        signal_summary = array_summary(signal)
        time_summary = array_summary(time_axis)
        same_size = signal.size == time_axis.size
        time_monotonic = bool(
            time_axis.size < 2 or np.all(np.diff(time_axis.reshape(-1)) >= 0)
        )
        validation = {
            "non_empty": bool(signal.size > 0),
            "same_size": bool(same_size),
            "time_monotonic_non_decreasing": time_monotonic,
        }

        if has_indices:
            expected_index_points = args.index_end - args.index_start + 1
            validation.update(
                {
                    "expected_index_points": expected_index_points,
                    "index_point_count_match": bool(
                        signal.size == expected_index_points
                    ),
                }
            )

        if has_window:
            flat_time = time_axis.reshape(-1)
            expected_points = int(
                round((args.time_end - args.time_start) / args.time_step)
            ) + 1
            deltas = np.diff(flat_time)
            median_step = float(np.median(deltas)) if deltas.size else None
            tolerance = max(abs(args.time_step) * 1e-5, 1e-12)
            step_matches = np.isclose(
                deltas, args.time_step, atol=tolerance, rtol=0.0
            )
            validation.update(
                {
                    "expected_points": expected_points,
                    "expected_point_count_match": bool(
                        signal.size == expected_points
                    ),
                    "window_start_match": bool(
                        flat_time.size
                        and np.isclose(
                            flat_time[0], args.time_start, atol=tolerance
                        )
                    ),
                    "window_end_match": bool(
                        flat_time.size
                        and np.isclose(
                            flat_time[-1], args.time_end, atol=tolerance
                        )
                    ),
                    "median_time_step": median_step,
                    "minimum_time_step": (
                        float(deltas.min()) if deltas.size else None
                    ),
                    "maximum_time_step": (
                        float(deltas.max()) if deltas.size else None
                    ),
                    "nonmatching_time_step_count": int(
                        deltas.size - step_matches.sum()
                    ),
                    "time_step_match": bool(
                        deltas.size > 0 and np.all(step_matches)
                    ),
                }
            )

        result.update(
            {
                "status": "success",
                "signal": signal_summary,
                "time_axis": time_summary,
                "validation": validation,
                "timing_seconds": {
                    "connect": round(connected_at - started, 6),
                    "open_tree": round(opened_at - connected_at, 6),
                    "read": round(read_at - opened_at, 6),
                    "total": round(read_at - started, 6),
                },
            }
        )
    except Exception as exc:
        result["error_type"] = type(exc).__name__
        result["error_message"] = str(exc)
        result["timing_seconds"] = {
            "total": round(time.time() - started, 6)
        }
    finally:
        if tree_opened:
            try:
                connection.closeTree(args.tree, args.shot)
            except Exception:
                pass

    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)

    if args.output:
        output_dir = os.path.dirname(os.path.abspath(args.output))
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        with open(args.output, "w") as handle:
            handle.write(rendered)
            handle.write("\n")

    return 0 if result["status"] == "success" else 1


if __name__ == "__main__":
    sys.exit(main())

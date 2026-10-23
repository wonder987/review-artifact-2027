#!/usr/bin/env python3
"""Estimate MDSplus retrieval cost without loading the signal array."""

import argparse
import json
import os
import sys
import time

from MDSplus import Connection


MIB = 1024 * 1024


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="202.127.204.12")
    parser.add_argument("--tree", required=True)
    parser.add_argument("--shot", required=True, type=int)
    parser.add_argument("--node", required=True)
    parser.add_argument("--default-path")
    parser.add_argument("--daqtime-suffix", default=":daqtime")
    parser.add_argument("--freq-suffix", default=":freq")
    parser.add_argument("--signal-bytes-per-point", type=int, default=4)
    parser.add_argument("--time-bytes-per-point", type=int, default=8)
    parser.add_argument("--max-points", type=int, default=5000000)
    parser.add_argument("--max-total-mib", type=float, default=256.0)
    parser.add_argument("--output")
    return parser.parse_args()


def scalar(value):
    data = value.data()
    return float(data)


def main():
    args = parse_args()
    started = time.time()
    connection = None
    tree_opened = False
    result = {
        "status": "error",
        "server": args.server,
        "tree": args.tree,
        "shot": args.shot,
        "node": args.node,
        "mode": "metadata_preflight",
    }

    try:
        connection = Connection(args.server)
        connection.openTree(args.tree, args.shot)
        tree_opened = True

        if args.default_path:
            expression = 'TCL("set def {}")'.format(args.default_path)
            connection.get(expression)

        daqtime = scalar(connection.get(args.node + args.daqtime_suffix))
        frequency = scalar(connection.get(args.node + args.freq_suffix))
        estimated_points = int(round(daqtime * frequency))

        bytes_per_point = (
            args.signal_bytes_per_point + args.time_bytes_per_point
        )
        estimated_signal_bytes = (
            estimated_points * args.signal_bytes_per_point
        )
        estimated_time_bytes = estimated_points * args.time_bytes_per_point
        estimated_total_bytes = estimated_points * bytes_per_point
        estimated_total_mib = estimated_total_bytes / float(MIB)

        within_point_limit = estimated_points <= args.max_points
        within_memory_limit = estimated_total_mib <= args.max_total_mib
        allow_full_read = within_point_limit and within_memory_limit

        result.update(
            {
                "status": "success",
                "metadata": {
                    "daqtime": daqtime,
                    "frequency": frequency,
                },
                "estimate": {
                    "points": estimated_points,
                    "signal_mib": round(estimated_signal_bytes / float(MIB), 3),
                    "time_axis_mib": round(estimated_time_bytes / float(MIB), 3),
                    "total_mib": round(estimated_total_mib, 3),
                },
                "policy": {
                    "max_points": args.max_points,
                    "max_total_mib": args.max_total_mib,
                    "within_point_limit": within_point_limit,
                    "within_memory_limit": within_memory_limit,
                },
                "decision": {
                    "allow_full_read": allow_full_read,
                    "require_reduction": not allow_full_read,
                    "recommended_action": (
                        "read_full_signal"
                        if allow_full_read
                        else "limit_time_window_or_downsample"
                    ),
                },
                "elapsed_seconds": round(time.time() - started, 6),
            }
        )
    except Exception as exc:
        result["error_type"] = type(exc).__name__
        result["error_message"] = str(exc)
        result["elapsed_seconds"] = round(time.time() - started, 6)
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

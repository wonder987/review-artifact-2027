#!/usr/bin/env python3
"""Pure numerical and safety operations for Gold-138 references."""

from __future__ import print_function

import hashlib

import numpy as np


MIB = 1024.0 * 1024.0


def array_summary(values):
    array = np.asarray(values)
    flat = array.reshape(-1)
    contiguous = np.ascontiguousarray(array)
    result = {
        "shape": [int(value) for value in array.shape],
        "size": int(array.size),
        "dtype": str(array.dtype),
        "sha256": hashlib.sha256(contiguous.tobytes()).hexdigest(),
    }
    if np.issubdtype(array.dtype, np.number):
        finite = np.isfinite(flat)
        finite_values = flat[finite]
        result["finite_count"] = int(finite.sum())
        result["finite_ratio"] = float(finite.mean()) if flat.size else None
        if finite_values.size:
            result.update(
                {
                    "min": float(finite_values.min()),
                    "max": float(finite_values.max()),
                    "mean": float(finite_values.mean()),
                    "std": float(finite_values.std()),
                }
            )
    return result


def resource_decision(
    points,
    total_mib,
    max_points=5000000,
    max_total_mib=256.0,
):
    points = int(points)
    total_mib = float(total_mib)
    within_points = points <= int(max_points)
    within_memory = total_mib <= float(max_total_mib)
    allow = within_points and within_memory
    return {
        "allow_full_read": bool(allow),
        "within_point_limit": bool(within_points),
        "within_memory_limit": bool(within_memory),
        "points": points,
        "total_mib": total_mib,
        "max_points": int(max_points),
        "max_total_mib": float(max_total_mib),
        "recommended_action": (
            "read_full_signal" if allow else "limit_time_window_or_downsample"
        ),
    }


def estimate_total_mib(points, signal_bytes=4, time_bytes=8):
    return int(points) * (int(signal_bytes) + int(time_bytes)) / MIB


def index_window(
    trigger_time,
    frequency,
    sample_num,
    start_time,
    requested_points,
):
    frequency = float(frequency)
    requested_points = int(requested_points)
    if frequency <= 0:
        raise ValueError("frequency must be positive")
    if requested_points <= 0:
        raise ValueError("requested_points must be positive")
    index_start = int(round((float(start_time) - float(trigger_time)) * frequency))
    index_end = index_start + requested_points - 1
    if index_start < 0 or index_end >= int(sample_num):
        raise ValueError("requested index window is outside the signal")
    return {
        "time_start": float(start_time),
        "time_end": float(start_time) + (requested_points - 1) / frequency,
        "time_step": 1.0 / frequency,
        "index_start": index_start,
        "index_end": index_end,
        "expected_points": requested_points,
    }


def _validated_series(time_values, signal_values, label):
    time_array = np.asarray(time_values, dtype=np.float64).reshape(-1)
    signal_array = np.asarray(signal_values).reshape(-1)
    if time_array.size != signal_array.size:
        raise ValueError("{} signal and time sizes differ".format(label))
    if time_array.size < 2:
        raise ValueError("{} needs at least two points".format(label))
    if not np.all(np.isfinite(time_array)):
        raise ValueError("{} time axis contains non-finite values".format(label))
    if np.any(np.diff(time_array) <= 0):
        raise ValueError("{} time axis must be strictly increasing".format(label))
    if not np.all(np.isfinite(signal_array)):
        raise ValueError("{} signal contains non-finite values".format(label))
    return time_array, signal_array


def align_on_intersection(time_a, signal_a, time_b, signal_b, points):
    time_a, signal_a = _validated_series(time_a, signal_a, "first")
    time_b, signal_b = _validated_series(time_b, signal_b, "second")
    points = int(points)
    if points < 2:
        raise ValueError("alignment needs at least two points")
    start = max(float(time_a[0]), float(time_b[0]))
    end = min(float(time_a[-1]), float(time_b[-1]))
    if end <= start:
        raise ValueError("time axes do not overlap")
    grid = np.linspace(start, end, points, dtype=np.float64)
    aligned_a = np.interp(grid, time_a, signal_a)
    aligned_b = np.interp(grid, time_b, signal_b)
    if not np.all(np.isfinite(aligned_a)) or not np.all(np.isfinite(aligned_b)):
        raise ValueError("alignment produced non-finite values")
    return grid, aligned_a, aligned_b


def comparison_summary(values_a, values_b):
    array_a = np.asarray(values_a, dtype=np.float64).reshape(-1)
    array_b = np.asarray(values_b, dtype=np.float64).reshape(-1)
    if array_a.size != array_b.size or array_a.size < 2:
        raise ValueError("comparison arrays must have equal size of at least two")
    if not np.all(np.isfinite(array_a)) or not np.all(np.isfinite(array_b)):
        raise ValueError("comparison arrays contain non-finite values")
    difference = array_a - array_b
    std_a = float(array_a.std())
    std_b = float(array_b.std())
    correlation = None
    if std_a > 0.0 and std_b > 0.0:
        correlation = float(np.corrcoef(array_a, array_b)[0, 1])
    return {
        "count": int(array_a.size),
        "a_mean": float(array_a.mean()),
        "a_std": std_a,
        "b_mean": float(array_b.mean()),
        "b_std": std_b,
        "mean_absolute_difference": float(np.abs(difference).mean()),
        "root_mean_square_difference": float(np.sqrt(np.mean(difference ** 2))),
        "pearson_correlation": correlation,
    }

#!/usr/bin/env python3
"""Merge PA-SCoRe repair rounds and compute paired comparison statistics."""

from __future__ import print_function

import argparse
import hashlib
import json
import math
import random
import zipfile
from collections import Counter


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_diagnostics_zip(path):
    outcomes = {}
    with zipfile.ZipFile(path) as archive:
        if "manifest.json" not in archive.namelist():
            manifest_path = "evaluation/evaluation_manifest.json"
            data_path = "evaluation/evaluation.jsonl"
            manifest = json.loads(archive.read(manifest_path).decode("utf-8"))
            payload = archive.read(data_path)
            actual_hash = hashlib.sha256(payload).hexdigest()
            if actual_hash != manifest.get("evaluation_jsonl_sha256"):
                raise ValueError("evaluation hash mismatch: {}".format(path))
            evaluations = [
                json.loads(line) for line in payload.decode("utf-8").splitlines()
                if line.strip()
            ]
            if manifest.get("record_count") != len(evaluations):
                raise ValueError("diagnostics entry count mismatch: {}".format(path))
            for evaluation in evaluations:
                key = (evaluation.get("model_alias"), evaluation.get("task_id"))
                if not all(key) or key in outcomes:
                    raise ValueError("invalid or duplicate diagnostics pair: {}".format(key))
                outcomes[key] = evaluation
            return outcomes

        manifest = json.loads(archive.read("manifest.json").decode("utf-8"))
        entries = manifest.get("entries", [])
        if manifest.get("entry_count") != len(entries):
            raise ValueError("diagnostics entry count mismatch: {}".format(path))
        for entry in entries:
            key = (entry.get("model_alias"), entry.get("task_id"))
            if not all(key) or key in outcomes:
                raise ValueError("invalid or duplicate diagnostics pair: {}".format(key))
            reference = (entry.get("files") or {}).get("evaluation")
            if not reference:
                raise ValueError("missing evaluation reference: {}".format(key))
            payload = archive.read(reference["path"])
            actual_hash = hashlib.sha256(payload).hexdigest()
            if actual_hash != reference.get("sha256"):
                raise ValueError("evaluation hash mismatch: {}".format(key))
            evaluation = json.loads(payload.decode("utf-8"))
            if (
                evaluation.get("model_alias") != key[0]
                or evaluation.get("task_id") != key[1]
            ):
                raise ValueError("evaluation identity mismatch: {}".format(key))
            outcomes[key] = evaluation
    return outcomes


def _success(value):
    if isinstance(value, dict):
        return bool(value.get("semantic_success"))
    return bool(value)


def _percentile(values, probability):
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile requires at least one value")
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def merge_round_outcomes(rounds):
    if not rounds:
        raise ValueError("at least one round is required")
    final = dict((key, dict(value)) for key, value in rounds[0].items())
    success_round = dict(
        (key, 0) for key, value in final.items() if _success(value)
    )
    for round_index, outcomes in enumerate(rounds[1:], 1):
        for key, value in outcomes.items():
            if key not in final:
                raise ValueError("repair round contains unknown pair: {}".format(key))
            if _success(final[key]):
                raise ValueError(
                    "repair round contains already-successful pair: {}".format(key)
                )
            final[key] = dict(value)
            if _success(value):
                success_round[key] = round_index
    return final, success_round


def exact_mcnemar(direct, method):
    if set(direct) != set(method):
        raise ValueError("paired outcomes must have identical keys")
    direct_only = 0
    method_only = 0
    for key in direct:
        first = _success(direct[key])
        second = _success(method[key])
        direct_only += int(first and not second)
        method_only += int(second and not first)
    discordant = direct_only + method_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = min(direct_only, method_only)
        numerator = sum(math.comb(discordant, k) for k in range(tail + 1))
        p_value = min(1.0, 2.0 * numerator / float(2 ** discordant))
    return {
        "direct_only": direct_only,
        "method_only": method_only,
        "discordant": discordant,
        "p_value": p_value,
        "test": "exact_two_sided_mcnemar",
    }


def _task_index(records):
    by_task = {}
    for key in records:
        by_task.setdefault(key[1], []).append(key)
    return by_task


def cluster_bootstrap_difference(direct, method, iterations=10000, seed=20260930):
    if set(direct) != set(method):
        raise ValueError("paired outcomes must have identical keys")
    by_task = _task_index(direct)
    task_ids = sorted(by_task)
    rng = random.Random(seed)

    def difference(sampled_tasks):
        direct_success = 0
        method_success = 0
        total = 0
        for task_id in sampled_tasks:
            for key in by_task[task_id]:
                direct_success += int(_success(direct[key]))
                method_success += int(_success(method[key]))
                total += 1
        return (method_success - direct_success) / float(total)

    estimate = difference(task_ids)
    draws = [
        difference([rng.choice(task_ids) for _ in task_ids])
        for _ in range(iterations)
    ]
    return {
        "estimate": estimate,
        "confidence_low": _percentile(draws, 0.025),
        "confidence_high": _percentile(draws, 0.975),
        "confidence_level": 0.95,
        "iterations": iterations,
        "seed": seed,
        "resampling_unit": "task",
        "interval": "percentile_cluster_bootstrap",
    }


def cluster_bootstrap_macro_level_difference(
    direct, method, iterations=10000, seed=20260930
):
    if set(direct) != set(method):
        raise ValueError("paired outcomes must have identical keys")
    by_task = _task_index(direct)
    tasks_by_level = {}
    for task_id, keys in by_task.items():
        levels = set(direct[key].get("level") for key in keys)
        if len(levels) != 1 or None in levels:
            raise ValueError("task has inconsistent level: {}".format(task_id))
        tasks_by_level.setdefault(next(iter(levels)), []).append(task_id)
    for level in tasks_by_level:
        tasks_by_level[level].sort()
    rng = random.Random(seed)

    def macro_difference(sampled_by_level):
        differences = []
        for level in sorted(sampled_by_level):
            direct_success = 0
            method_success = 0
            total = 0
            for task_id in sampled_by_level[level]:
                for key in by_task[task_id]:
                    direct_success += int(_success(direct[key]))
                    method_success += int(_success(method[key]))
                    total += 1
            differences.append((method_success - direct_success) / float(total))
        return sum(differences) / float(len(differences))

    estimate = macro_difference(tasks_by_level)
    draws = []
    for _ in range(iterations):
        sampled = dict(
            (level, [rng.choice(tasks) for _ in tasks])
            for level, tasks in tasks_by_level.items()
        )
        draws.append(macro_difference(sampled))
    return {
        "estimate": estimate,
        "confidence_low": _percentile(draws, 0.025),
        "confidence_high": _percentile(draws, 0.975),
        "confidence_level": 0.95,
        "iterations": iterations,
        "seed": seed,
        "resampling_unit": "task",
        "stratified_by": "level",
        "interval": "percentile_cluster_bootstrap",
    }


def summarize_outcomes(outcomes):
    total = len(outcomes)
    semantic_success = sum(int(_success(value)) for value in outcomes.values())
    summary = {
        "total": total,
        "semantic_success": semantic_success,
        "semantic_accuracy": semantic_success / float(total) if total else 0.0,
        "by_category": dict(sorted(Counter(
            value.get("category", "unknown") for value in outcomes.values()
        ).items())),
        "by_code": dict(sorted(Counter(
            value.get("code", "unknown") for value in outcomes.values()
        ).items())),
    }
    for field, output_name in (
        ("model_alias", "by_model"),
        ("level", "by_level"),
        ("subtype", "by_subtype"),
    ):
        groups = {}
        for key, value in outcomes.items():
            group = key[0] if field == "model_alias" else value.get(field)
            if group is None:
                raise ValueError("outcome is missing {}: {}".format(field, key))
            bucket = groups.setdefault(group, {"semantic_success": 0, "total": 0})
            bucket["total"] += 1
            bucket["semantic_success"] += int(_success(value))
        for bucket in groups.values():
            bucket["semantic_accuracy"] = (
                bucket["semantic_success"] / float(bucket["total"])
            )
        summary[output_name] = dict(sorted(groups.items()))
    return summary


def analyze_outcomes(direct, rounds, iterations=10000, seed=20260930):
    final, success_round = merge_round_outcomes(rounds)
    if set(direct) != set(final):
        raise ValueError("direct and PA-SCoRe outcomes must have identical keys")

    round_summaries = []
    for round_index, outcomes in enumerate(rounds):
        round_summaries.append({
            "round": round_index,
            "eligible": len(outcomes),
            "newly_successful": sum(
                int(value == round_index) for value in success_round.values()
            ),
        })

    models = sorted(set(key[0] for key in direct))
    by_model = {}
    for model in models:
        direct_subset = dict(
            (key, value) for key, value in direct.items() if key[0] == model
        )
        method_subset = dict((key, final[key]) for key in direct_subset)
        by_model[model] = exact_mcnemar(direct_subset, method_subset)

    return {
        "direct": summarize_outcomes(direct),
        "pascore": summarize_outcomes(final),
        "rounds": round_summaries,
        "terminal_failure_count": sum(
            int(not _success(value)) for value in final.values()
        ),
        "paired": {
            "mcnemar": exact_mcnemar(direct, final),
            "mcnemar_by_model": by_model,
            "micro_accuracy_difference": cluster_bootstrap_difference(
                direct, final, iterations=iterations, seed=seed
            ),
            "macro_level_accuracy_difference": (
                cluster_bootstrap_macro_level_difference(
                    direct, final, iterations=iterations, seed=seed
                )
            ),
        },
    }


def analyze_files(
    direct_path, round_paths, iterations=10000, seed=20260930
):
    direct = load_diagnostics_zip(direct_path)
    rounds = [load_diagnostics_zip(path) for path in round_paths]
    return {
        "schema_version": "1.0",
        "artifacts": {
            "direct": {
                "path": str(direct_path),
                "sha256": _file_sha256(direct_path),
            },
            "pascore_rounds": [
                {
                    "round": index,
                    "path": str(path),
                    "sha256": _file_sha256(path),
                }
                for index, path in enumerate(round_paths)
            ],
        },
        "analysis": analyze_outcomes(
            direct, rounds, iterations=iterations, seed=seed
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Compare Direct Zero-shot and terminal PA-SCoRe outcomes."
    )
    parser.add_argument("--direct-diagnostics", required=True)
    parser.add_argument("--pascore-rounds", nargs="+", required=True)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    result = analyze_files(
        args.direct_diagnostics,
        args.pascore_rounds,
        iterations=args.iterations,
        seed=args.seed,
    )
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

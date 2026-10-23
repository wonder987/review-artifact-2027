#!/usr/bin/env python3
"""Audit reusable fixed-model baseline and ablation outcomes."""

from __future__ import print_function

import argparse
import hashlib
import json

try:
    from scripts.analyze_pascore_comparison import (
        load_diagnostics_zip,
        merge_round_outcomes,
        summarize_outcomes,
    )
except ImportError:
    from analyze_pascore_comparison import (
        load_diagnostics_zip,
        merge_round_outcomes,
        summarize_outcomes,
    )


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_ablation_tasks(split_path):
    with open(split_path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    tasks = value.get("ablation")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("split file must contain a non-empty ablation list")
    if len(tasks) != len(set(tasks)):
        raise ValueError("ablation split contains duplicate task IDs")
    return set(tasks)


def audit_reuse(
    round_paths,
    split_path,
    model_alias,
    expected_heldout=None,
    expected_ablation=None,
):
    if not round_paths or len(round_paths) > 3:
        raise ValueError("one to three diagnostic rounds are required")
    rounds = [load_diagnostics_zip(path) for path in round_paths]
    model_rounds = [
        dict((key, value) for key, value in outcomes.items()
             if key[0] == model_alias)
        for outcomes in rounds
    ]
    if not model_rounds[0]:
        raise ValueError("model is absent from round zero: {}".format(model_alias))
    if expected_heldout is not None and len(model_rounds[0]) != expected_heldout:
        raise ValueError(
            "held-out denominator mismatch: expected {}, found {}".format(
                expected_heldout, len(model_rounds[0])
            )
        )

    ablation_tasks = _load_ablation_tasks(split_path)
    round_zero_tasks = set(key[1] for key in model_rounds[0])
    missing = sorted(ablation_tasks - round_zero_tasks)
    if missing:
        raise ValueError("missing ablation tasks: {}".format(", ".join(missing)))
    if expected_ablation is not None and len(ablation_tasks) != expected_ablation:
        raise ValueError(
            "ablation denominator mismatch: expected {}, found {}".format(
                expected_ablation, len(ablation_tasks)
            )
        )

    ablation_rounds = [
        dict((key, value) for key, value in outcomes.items()
             if key[1] in ablation_tasks)
        for outcomes in model_rounds
    ]
    prompt_only_ablation, _ = merge_round_outcomes(ablation_rounds[:1])
    single_round_ablation, _ = merge_round_outcomes(ablation_rounds[:2])
    full_ablation, _ = merge_round_outcomes(ablation_rounds)

    return {
        "schema_version": "1.0",
        "model_alias": model_alias,
        "eligible_entries_by_round": [len(value) for value in ablation_rounds],
        "prompt_only_heldout": summarize_outcomes(model_rounds[0]),
        "prompt_only_ablation": summarize_outcomes(prompt_only_ablation),
        "single_round_ablation": summarize_outcomes(single_round_ablation),
        "full_ablation": summarize_outcomes(full_ablation),
        "artifacts": {
            "split": {
                "path": str(split_path),
                "sha256": _file_sha256(split_path),
                "ablation_task_count": len(ablation_tasks),
            },
            "diagnostic_rounds": [
                {
                    "round": index,
                    "path": str(path),
                    "sha256": _file_sha256(path),
                }
                for index, path in enumerate(round_paths)
            ],
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Audit reusable Prompt-only, Single-round, and Full results."
    )
    parser.add_argument("--diagnostic-rounds", nargs="+", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--expected-heldout", type=int)
    parser.add_argument("--expected-ablation", type=int)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    result = audit_reuse(
        args.diagnostic_rounds,
        args.splits,
        args.model,
        expected_heldout=args.expected_heldout,
        expected_ablation=args.expected_ablation,
    )
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

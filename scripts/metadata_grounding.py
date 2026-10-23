#!/usr/bin/env python3
"""Build disclosure-safe metadata grounding for PA-SCoRe."""

from __future__ import print_function

import argparse
import copy
import json

try:
    from scripts.pascore_contracts import (
        atomic_json,
        load_json,
        sha256_json,
        sha256_text,
    )
except ImportError:
    from pascore_contracts import atomic_json, load_json, sha256_json, sha256_text


INTERFACE_CONTRACT = (
    "Import the client as `from MDSplus import Connection` or `import MDSplus` with exact capitalization",
    "Connection(server)",
    "Connection.openTree(tree, shot)",
    "Connection.get(expression).data(); pass exactly one TDI expression string per call",
    "Connection.get('dim_of(<same expression>)').data() for its time axis",
    "Connection.get('data(<node>)[<start_index>:<end_index>]').data() for an inclusive bounded signal slice",
    "Connection.get('dim_of(<node>)[<start_index>:<end_index>]').data() for the matching inclusive time slice",
    "Connection.get('TCL(\"set def <subtree>\")') when a subtree is permitted",
)


def _target_grounding(target, prompt_spec, enabled):
    identity = {
        "server": target.get("server"),
        "tree": target.get("tree"),
        "shot": target.get("shot"),
        "signal_name": target.get("signal_name"),
        "node": target.get("node"),
        "subtree": target.get("subtree"),
    }
    result = {"identity": identity}
    if enabled:
        result["acquisition"] = {
            "daqtime": target.get("daqtime"),
            "estimated_total_mib": target.get("estimated_total_mib"),
            "frequency": target.get("frequency"),
            "sample_num": target.get("sample_num"),
            "size_band": target.get("size_band"),
            "trigtime": target.get("trigtime"),
            "unit": target.get("unit"),
        }
    return result


def _window_for_target(operation, target_index):
    kind = operation.get("kind")
    if kind in ("index_window", "sampling_constrained"):
        return operation
    if kind in ("cross_shot_comparison", "same_shot_alignment"):
        windows = operation.get("per_target_windows") or []
        return windows[target_index] if target_index < len(windows) else None
    if kind == "adaptive_access":
        return operation.get("fallback_window")
    return None


def _access_plan(task):
    operation = task.get("operation") or {}
    plans = []
    for index, target in enumerate(task.get("targets", [])):
        node = target.get("node")
        window = _window_for_target(operation, index)
        if window:
            start = int(window["index_start"])
            end = int(window["index_end"])
            signal_expression = "data({})[{}:{}]".format(node, start, end)
            time_expression = "dim_of({})[{}:{}]".format(node, start, end)
            expected_points = window.get("expected_points")
            access_mode = "index_window"
        else:
            signal_expression = node
            time_expression = "dim_of({})".format(node)
            expected_points = target.get("sample_num")
            access_mode = "full_signal"
        subtree = target.get("subtree")
        plans.append({
            "target_index": index,
            "access_mode": access_mode,
            "server": target.get("server"),
            "tree": target.get("tree"),
            "shot": target.get("shot"),
            "node": node,
            "subtree": subtree,
            "set_default_expression": (
                'TCL("set def {}")'.format(subtree) if subtree else None
            ),
            "signal_expression": signal_expression,
            "time_expression": time_expression,
            "expected_points": expected_points,
        })
    return plans


def _prompt_block(grounding):
    if not grounding["enabled"]:
        return ""
    lines = [
        "Verified MDSplus environment facts:",
        "- Use the remote Connection API only: {}.".format(
            "; ".join(grounding["interface_contract"])
        ),
        "- Complete operation metadata: {}.".format(
            json.dumps(grounding["operation"], sort_keys=True)
        ),
    ]
    for index, target in enumerate(grounding["targets"], 1):
        identity = target["identity"]
        acquisition = target["acquisition"]
        lines.append(
            "- Target {} metadata: server={}, tree={}, shot={}, signal={}, node={}, "
            "subtree={}, sample_num={}, frequency={}, daqtime={}, trigtime={}, "
            "unit={}, estimated_total_mib={}, size_band={}.".format(
                index,
                identity.get("server"),
                identity.get("tree"),
                identity.get("shot"),
                identity.get("signal_name"),
                identity.get("node"),
                identity.get("subtree"),
                acquisition.get("sample_num"),
                acquisition.get("frequency"),
                acquisition.get("daqtime"),
                acquisition.get("trigtime"),
                acquisition.get("unit"),
                acquisition.get("estimated_total_mib"),
                acquisition.get("size_band"),
            )
        )
    for plan in grounding["access_plan"]:
        lines.append(
            "- Verified access plan for target {}: set_default_expression={}; "
            "signal_expression={}; time_expression={}; expected_points={}. "
            "Pass each expression as the sole argument to Connection.get.".format(
                plan["target_index"] + 1,
                plan.get("set_default_expression"),
                plan.get("signal_expression"),
                plan.get("time_expression"),
                plan.get("expected_points"),
            )
        )
    budget = grounding["safety_budget"]
    lines.append(
        "- Safety budget: max_points={}, max_total_mib={}, timeout_seconds={}.".format(
            budget.get("max_points"),
            budget.get("max_total_mib"),
            budget.get("timeout_seconds"),
        )
    )
    lines.append(
        "Do not use local Tree APIs or invent methods such as getNode, mdsopen, "
        "open, close, or disconnect on a remote Connection."
    )
    return "\n".join(lines)


def build_grounding(task, enabled=True):
    prompt_spec = task.get("prompt_spec") or {}
    grounding = {
        "schema_version": "1.0",
        "grounding_version": "metadata-grounding-v2",
        "enabled": bool(enabled),
        "task_id": task["id"],
        "task_spec_hash": task.get("spec_sha256"),
        "targets": [
            _target_grounding(target, prompt_spec, bool(enabled))
            for target in task.get("targets", [])
        ],
        "safety_budget": dict(task.get("safety_policy") or {}),
    }
    if enabled:
        grounding["interface_contract"] = list(INTERFACE_CONTRACT)
        grounding["operation"] = copy.deepcopy(task.get("operation") or {})
        grounding["access_plan"] = _access_plan(task)
    else:
        grounding["interface_contract"] = []
        grounding["operation"] = {}
        grounding["access_plan"] = []
    grounding["prompt_block"] = _prompt_block(grounding)
    grounding["grounding_sha256"] = sha256_json(dict(
        (key, value) for key, value in grounding.items()
        if key != "grounding_sha256"
    ))
    return grounding


def build_groundings(manifest, task_ids=None, enabled=True):
    allowed = set(task_ids) if task_ids is not None else None
    return [
        build_grounding(task, enabled=enabled)
        for task in manifest.get("tasks", [])
        if allowed is None or task["id"] in allowed
    ]


def render_grounded_prompt_record(task, original_record, enabled=True):
    grounding = build_grounding(task, enabled=enabled)
    system_prompt = original_record["system_prompt"]
    if enabled:
        user_prompt = (
            original_record["user_prompt"].rstrip()
            + "\n\n"
            + grounding["prompt_block"]
        )
        prompt_version = "pascore-grounded-v2"
    else:
        user_prompt = original_record["user_prompt"]
        prompt_version = original_record.get("prompt_version", "primary-zero-shot-v3")
    prompt = system_prompt + "\n\n" + user_prompt
    result = dict(original_record)
    result.update({
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "prompt": prompt,
        "prompt_version": prompt_version,
        "prompt_sha256": sha256_text(prompt),
        "grounding_sha256": grounding["grounding_sha256"],
    })
    return result


def _read_jsonl(path):
    records = []
    with open(path, "r") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def _write_jsonl(path, records):
    import os
    parent = os.path.dirname(os.path.abspath(path))
    if not os.path.isdir(parent):
        os.makedirs(parent)
    temporary = path + ".tmp"
    with open(temporary, "w") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def render_grounded_prompt_set(manifest, prompt_records, enabled=True):
    task_by_id = dict(
        (task["id"], task) for task in manifest.get("tasks", [])
    )
    rendered = []
    for record in prompt_records:
        task_id = record["task_id"]
        if task_id not in task_by_id:
            raise ValueError("prompt task is absent from manifest: {}".format(task_id))
        rendered.append(render_grounded_prompt_record(
            task_by_id[task_id], record, enabled=enabled
        ))
    return rendered


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--task-ids", nargs="+")
    parser.add_argument("--disabled", action="store_true")
    parser.add_argument("--prompts")
    args = parser.parse_args(argv)
    manifest = load_json(args.manifest)
    if args.prompts:
        records = render_grounded_prompt_set(
            manifest, _read_jsonl(args.prompts), enabled=not args.disabled
        )
        if args.task_ids:
            allowed = set(args.task_ids)
            records = [record for record in records if record["task_id"] in allowed]
        _write_jsonl(args.output, records)
    else:
        records = build_groundings(
            manifest,
            task_ids=args.task_ids,
            enabled=not args.disabled,
        )
        atomic_json(args.output, {
            "schema_version": "1.0",
            "records": records,
            "record_count": len(records),
        })
    print(json.dumps({"record_count": len(records), "output": args.output}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

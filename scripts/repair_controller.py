#!/usr/bin/env python3
"""Append-only controller and repair-prompt renderer for PA-SCoRe."""

from __future__ import print_function

import copy
import argparse
import json
import os
import zipfile

try:
    from scripts.pascore_contracts import (
        assert_export_safe,
        atomic_json,
        load_json,
        sha256_bytes,
        sha256_file,
        sha256_json,
        sha256_text,
        validate_condition,
    )
except ImportError:
    from pascore_contracts import (
        assert_export_safe,
        atomic_json,
        load_json,
        sha256_bytes,
        sha256_file,
        sha256_json,
        sha256_text,
        validate_condition,
    )


def initialize_state(run_id, task_id, model_alias, condition, max_repair_rounds):
    return {
        "schema_version": "1.0",
        "controller_version": "repair-controller-v1",
        "run_id": str(run_id),
        "task_id": str(task_id),
        "model_alias": str(model_alias),
        "condition": str(condition),
        "current_round": 0,
        "max_repair_rounds": int(max_repair_rounds),
        "status": "generated",
        "events": [],
        "imported_feedback_hashes": [],
    }


def import_feedback(state, feedback):
    assert_export_safe(feedback)
    result = copy.deepcopy(state)
    identity = feedback.get("identity") or {}
    for field in ("run_id", "task_id", "model_alias", "condition"):
        if identity.get(field) != result.get(field):
            raise ValueError("feedback {} binding mismatch".format(field))
    if identity.get("round") != result.get("current_round"):
        raise ValueError("feedback round binding mismatch")
    feedback_hash = sha256_json(feedback)
    if feedback_hash in result["imported_feedback_hashes"]:
        return result
    result["imported_feedback_hashes"].append(feedback_hash)
    result["events"].append({
        "event": "feedback_imported",
        "round": result["current_round"],
        "artifact_sha256": feedback_hash,
    })
    result["status"] = "feedback_ready"
    return result


def build_repair_prompt_record(
    original_record,
    previous_program,
    feedback,
    round_index,
    prompt_version,
):
    assert_export_safe(feedback)
    system_prompt = (
        str(original_record["system_prompt"]).rstrip()
        + " Repair the previous program using only the supplied execution feedback. "
          "Return the complete replacement program only."
    )
    user_prompt = (
        "Original task:\n{task}\n\n"
        "Previous program:\n```python\n{program}\n```\n\n"
        "Execution feedback:\n{feedback}\n\n"
        "Produce a complete corrected program. Do not explain the changes and do "
        "not hard-code expected outputs."
    ).format(
        task=original_record["user_prompt"],
        program=str(previous_program).rstrip(),
        feedback=json.dumps(feedback, indent=2, sort_keys=True),
    )
    prompt = system_prompt + "\n\n" + user_prompt
    result = {
        "task_id": original_record["task_id"],
        "split": original_record["split"],
        "prompt_version": prompt_version,
        "repair_round": int(round_index),
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "prompt": prompt,
        "prompt_sha256": sha256_text(prompt),
    }
    if original_record.get("grounding_sha256"):
        result["grounding_sha256"] = original_record["grounding_sha256"]
    return result


def _safe_archive_path(root, relative):
    normalized = os.path.normpath(relative)
    if os.path.isabs(normalized) or normalized.startswith(".." + os.sep):
        raise ValueError("unsafe diagnostic path: {}".format(relative))
    absolute = os.path.abspath(os.path.join(root, normalized))
    if not absolute.startswith(os.path.abspath(root) + os.sep):
        raise ValueError("diagnostic path escapes destination")
    return absolute


def import_diagnostics_zip(zip_path, destination):
    with zipfile.ZipFile(zip_path, "r") as archive:
        manifest = json.loads(archive.read("manifest.json").decode("utf-8"))
        if manifest.get("package_type") != "pascore_diagnostics_round":
            raise ValueError("not a PA-SCoRe diagnostics package")
        assert_export_safe(manifest)
        payloads = {}
        for entry in manifest.get("entries", []):
            for file_info in (entry.get("files") or {}).values():
                relative = file_info["path"]
                data = archive.read(relative)
                if sha256_bytes(data) != file_info["sha256"]:
                    raise ValueError("diagnostic hash mismatch: {}".format(relative))
                if relative.endswith(".json"):
                    assert_export_safe(json.loads(data.decode("utf-8")))
                payloads[relative] = data
    if not os.path.isdir(destination):
        os.makedirs(destination)
    manifest_path = os.path.join(destination, "manifest.json")
    atomic_json(manifest_path, manifest)
    for relative, data in payloads.items():
        target = _safe_archive_path(destination, relative)
        parent = os.path.dirname(target)
        if not os.path.isdir(parent):
            os.makedirs(parent)
        temporary = target + ".tmp"
        with open(temporary, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    return manifest


def _load_prompt_records(path):
    records = {}
    with open(path, "r") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            records[record["task_id"]] = record
    return records


def _write_jsonl(path, records):
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


def prepare_repair_prompts(
    diagnostics_root,
    previous_package_root,
    original_prompts_path,
    condition_path,
    base_protocol_path,
    state_root,
    output_root,
    next_round,
):
    diagnostics_manifest = load_json(os.path.join(diagnostics_root, "manifest.json"))
    condition = validate_condition(load_json(condition_path))
    previous_manifest_path = os.path.join(previous_package_root, "manifest.json")
    expected_source_hash = diagnostics_manifest.get("source_manifest_sha256")
    if not expected_source_hash:
        raise ValueError("diagnostics manifest is missing source manifest hash")
    if not os.path.isfile(previous_manifest_path):
        raise ValueError("previous package manifest is missing")
    if sha256_file(previous_manifest_path) != expected_source_hash:
        raise ValueError("diagnostics package does not match previous package")
    previous_manifest = load_json(previous_manifest_path)
    previous_entries = dict(
        ((entry["model_alias"], entry["task_id"]), entry)
        for entry in previous_manifest.get("entries", [])
    )
    if diagnostics_manifest.get("condition") != condition["condition_id"]:
        raise ValueError("diagnostic condition mismatch")
    if int(next_round) != int(diagnostics_manifest.get("round", -1)) + 1:
        raise ValueError("next repair round is not consecutive")
    if int(next_round) > int(condition["max_repair_rounds"]):
        raise ValueError("maximum repair rounds exhausted")
    prompts = _load_prompt_records(original_prompts_path)
    records_by_model = {}
    plan_entries = []
    for entry in diagnostics_manifest.get("entries", []):
        feedback_info = (entry.get("files") or {}).get("feedback")
        if not feedback_info:
            continue
        model_alias = entry["model_alias"]
        task_id = entry["task_id"]
        if task_id not in prompts:
            raise ValueError("original prompt is missing: {}".format(task_id))
        feedback = load_json(os.path.join(diagnostics_root, feedback_info["path"]))
        program_path = os.path.join(
            previous_package_root, "programs", model_alias, task_id, "program.py"
        )
        if os.path.isfile(program_path):
            with open(program_path, "r") as handle:
                previous_program = handle.read()
            parent_program_hash = sha256_file(program_path)
        else:
            previous_entry = previous_entries.get((model_alias, task_id)) or {}
            if previous_entry.get("entry_type") != "generation_failure":
                raise ValueError("previous program is missing: {}/{}".format(
                    model_alias, task_id
                ))
            parent_program_hash = (
                (previous_entry.get("pascore_identity") or {}).get("program_hash")
                or previous_entry.get("failure_sha256")
            )
            if not parent_program_hash:
                raise ValueError("generation failure hash is missing: {}/{}".format(
                    model_alias, task_id
                ))
            previous_program = (
                "# No previous program was produced because the preceding "
                "generation request failed.\n"
            )
        state_path = os.path.join(state_root, model_alias, task_id, "repair_state.json")
        if os.path.isfile(state_path):
            state = load_json(state_path)
        else:
            state = initialize_state(
                diagnostics_manifest["run_id"],
                task_id,
                model_alias,
                condition["condition_id"],
                condition["max_repair_rounds"],
            )
        if state["current_round"] == int(next_round) - 1:
            state = import_feedback(state, feedback)
            state["current_round"] = int(next_round)
            state["status"] = "repaired"
        elif state["current_round"] != int(next_round):
            raise ValueError("repair state round mismatch")

        prompt_version = "pascore-repair-v1-r{}".format(next_round)
        record = build_repair_prompt_record(
            prompts[task_id],
            previous_program,
            feedback,
            round_index=next_round,
            prompt_version=prompt_version,
        )
        record["pascore"] = {
            "run_id": diagnostics_manifest["run_id"],
            "condition": condition["condition_id"],
            "repair_round": int(next_round),
            "parent_program_hash": parent_program_hash,
        }
        plan_entries.append({
            "model_alias": model_alias,
            "task_id": task_id,
            "parent_program_hash": record["pascore"]["parent_program_hash"],
            "prompt_sha256": record["prompt_sha256"],
        })
        prompt_hash = record["prompt_sha256"]
        event = {
            "event": "repair_prompt_created",
            "round": int(next_round),
            "artifact_sha256": prompt_hash,
        }
        if event not in state["events"]:
            state["events"].append(event)
        atomic_json(state_path, state)
        records_by_model.setdefault(model_alias, []).append(record)

    for model_alias, records in records_by_model.items():
        records.sort(key=lambda item: item["task_id"])
        _write_jsonl(os.path.join(output_root, model_alias + ".jsonl"), records)
    base_protocol = load_json(base_protocol_path)
    repair_protocol = dict(base_protocol)
    repair_protocol.update({
        "schema_version": "1.0",
        "generation_mode": "repair",
        "protocol_id": "pascore-repair-v1-r{}".format(next_round),
        "prompt_version": "pascore-repair-v1-r{}".format(next_round),
        "repair_rounds": int(next_round),
    })
    atomic_json(os.path.join(output_root, "protocol.json"), repair_protocol)
    plan_entries.sort(key=lambda item: (item["model_alias"], item["task_id"]))
    atomic_json(os.path.join(output_root, "repair_plan.json"), {
        "schema_version": "1.0",
        "run_id": diagnostics_manifest["run_id"],
        "condition": condition["condition_id"],
        "round": int(next_round),
        "entry_count": len(plan_entries),
        "entries": plan_entries,
    })
    return {
        "model_count": len(records_by_model),
        "record_count": sum(len(records) for records in records_by_model.values()),
        "next_round": int(next_round),
        "output_root": os.path.abspath(output_root),
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostics-zip", required=True)
    parser.add_argument("--import-dir", required=True)
    parser.add_argument("--previous-package-root", required=True)
    parser.add_argument("--original-prompts", required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--base-protocol", required=True)
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--next-round", type=int, required=True)
    args = parser.parse_args(argv)
    import_diagnostics_zip(args.diagnostics_zip, args.import_dir)
    summary = prepare_repair_prompts(
        args.import_dir,
        args.previous_package_root,
        args.original_prompts,
        args.condition,
        args.base_protocol,
        args.state_dir,
        args.output_dir,
        args.next_round,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

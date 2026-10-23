#!/usr/bin/env python3
"""Package Gold-sanitized PA-SCoRe diagnostics for transfer to Windows."""

from __future__ import print_function

import argparse
import json
import os
import zipfile

try:
    from scripts.pascore_contracts import (
        assert_export_safe,
        load_json,
        sha256_bytes,
    )
except ImportError:
    from pascore_contracts import assert_export_safe, load_json, sha256_bytes


def _canonical_json_bytes(value):
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("utf-8")


def _zip_info(name):
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    return info


def sanitize_evaluation(value):
    result = {}
    for field in (
        "evaluator_version", "model_alias", "task_id", "level", "subtype",
        "category", "code", "execution_success", "semantic_success",
        "error_type", "error_message",
    ):
        if field in value:
            result[field] = value[field]
    for field in ("mismatches", "diagnostics"):
        result[field] = [
            {
                "path": item.get("path"),
                "code": item.get("code"),
            }
            for item in value.get(field, [])
            if isinstance(item, dict)
        ]
    assert_export_safe(result)
    return result


def _artifact(output_root, category, model_alias, task_id, filename):
    return os.path.join(output_root, category, model_alias, task_id, filename)


def _bounded_log(path, limit=65536):
    if not os.path.isfile(path):
        return None
    with open(path, "rb") as handle:
        return handle.read(int(limit))


def build_diagnostics_package(output_root, execution_manifest_path, output_zip):
    source_manifest = load_json(execution_manifest_path)
    payloads = {}
    package_entries = []
    for entry in source_manifest.get("entries", []):
        model_alias = entry["model_alias"]
        task_id = entry["task_id"]
        files = {}
        json_specs = (
            ("guard", "guard.json", False),
            ("execution", "execution.json", False),
            ("evaluation", "evaluation.json", True),
            ("diagnosis", "diagnosis.json", False),
            ("feedback", "feedback.json", False),
        )
        for category, filename, sanitize in json_specs:
            source = _artifact(
                output_root, category, model_alias, task_id, filename
            )
            if not os.path.isfile(source):
                continue
            value = load_json(source)
            if sanitize:
                value = sanitize_evaluation(value)
            assert_export_safe(value)
            relative = "{}/{}/{}/{}".format(
                category, model_alias, task_id, filename
            )
            data = _canonical_json_bytes(value)
            payloads[relative] = data
            files[category] = {
                "path": relative,
                "sha256": sha256_bytes(data),
            }
        for log_name in ("stdout.log", "stderr.log"):
            source = _artifact(
                output_root, "execution", model_alias, task_id, log_name
            )
            data = _bounded_log(source)
            if data is not None:
                relative = "logs/{}/{}/{}".format(
                    model_alias, task_id, log_name
                )
                payloads[relative] = data
                files[log_name] = {
                    "path": relative,
                    "sha256": sha256_bytes(data),
                }
        package_entries.append({
            "model_alias": model_alias,
            "task_id": task_id,
            "pascore_identity": entry.get("pascore_identity"),
            "files": files,
        })
    with open(execution_manifest_path, "rb") as handle:
        source_manifest_hash = sha256_bytes(handle.read())
    manifest = {
        "schema_version": "1.0",
        "package_type": "pascore_diagnostics_round",
        "run_id": source_manifest.get("run_id"),
        "round": source_manifest.get("round"),
        "condition": source_manifest.get("condition"),
        "source_manifest_sha256": source_manifest_hash,
        "entry_count": len(package_entries),
        "entries": package_entries,
    }
    assert_export_safe(manifest)
    output_directory = os.path.dirname(os.path.abspath(output_zip))
    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)
    temporary = output_zip + ".tmp"
    try:
        with zipfile.ZipFile(temporary, "w") as archive:
            archive.writestr(_zip_info("manifest.json"), _canonical_json_bytes(manifest))
            for relative in sorted(payloads):
                archive.writestr(_zip_info(relative), payloads[relative])
        os.replace(temporary, output_zip)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--round-output", required=True)
    parser.add_argument("--execution-manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    manifest = build_diagnostics_package(
        args.round_output, args.execution_manifest, args.output
    )
    print(json.dumps({
        "entry_count": manifest["entry_count"],
        "output": os.path.abspath(args.output),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

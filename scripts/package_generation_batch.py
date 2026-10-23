#!/usr/bin/env python3
"""Package hash-verified generated programs for controlled cs1 execution."""

from __future__ import print_function

import argparse
import hashlib
import json
import os
import zipfile

try:
    from scripts.llm_runner_config import load_models
except ImportError:
    from llm_runner_config import load_models


def _read_json(path):
    with open(path, "r") as handle:
        return json.load(handle)


def _sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def _read_bytes(path):
    with open(path, "rb") as handle:
        return handle.read()


def _generation_failure_code(failure):
    message = str(failure.get("error_message", "")).lower()
    if "generation request timed out after" in message:
        return "generation_timeout"
    return None


def collect_entries(results_root, allowed_models, allowed_task_ids=None):
    entries = []
    allowed_models = set(allowed_models)
    if allowed_task_ids is not None:
        allowed_task_ids = set(allowed_task_ids)
    for root, _, files in os.walk(results_root):
        completed = "result.json" in files and "program.py" in files
        failed = "failure.json" in files
        if not completed and not failed:
            continue
        artifact_name = "result.json" if completed else "failure.json"
        artifact = _read_json(os.path.join(root, artifact_name))
        identity = artifact.get("identity", {})
        model_alias = identity.get("model_alias")
        if model_alias not in allowed_models:
            continue
        task_id = identity.get("task_id")
        if allowed_task_ids is not None and task_id not in allowed_task_ids:
            continue
        attempt = identity.get("attempt")
        if not task_id or attempt != 1:
            raise ValueError("invalid generation artifact identity: {}".format(root))
        if completed:
            extraction = artifact.get("extraction", {})
            if (
                artifact.get("status") != "completed"
                or extraction.get("status") != "success"
            ):
                continue
            program_path = os.path.join(root, "program.py")
            program_bytes = _read_bytes(program_path)
            actual_hash = _sha256_bytes(program_bytes)
            expected_hash = extraction.get("program_sha256")
            if actual_hash != expected_hash:
                raise ValueError("program hash mismatch: {}".format(program_path))
            entries.append({
                "attempt": 1,
                "entry_type": "program",
                "grounding_sha256": identity.get("grounding_hash"),
                "model_alias": model_alias,
                "program_path": "programs/{}/{}/program.py".format(
                    model_alias, task_id
                ),
                "program_sha256": actual_hash,
                "prompt_sha256": identity.get("prompt_hash"),
                "protocol_sha256": identity.get("protocol_hash"),
                "task_id": task_id,
                "_source_program_path": program_path,
            })
            continue
        failure_code = _generation_failure_code(artifact)
        if artifact.get("status") != "failed" or failure_code is None:
            continue
        failure_path = os.path.join(root, "failure.json")
        failure_hash = _sha256_bytes(_read_bytes(failure_path))
        entries.append({
            "attempt": 1,
            "entry_type": "generation_failure",
            "error_message": artifact.get("error_message"),
            "error_type": artifact.get("error_type"),
            "failure_code": failure_code,
            "failure_path": "failures/{}/{}/failure.json".format(
                model_alias, task_id
            ),
            "failure_sha256": failure_hash,
            "grounding_sha256": identity.get("grounding_hash"),
            "model_alias": model_alias,
            "prompt_sha256": identity.get("prompt_hash"),
            "protocol_sha256": identity.get("protocol_hash"),
            "task_id": task_id,
            "_source_failure_path": failure_path,
        })
    entries.sort(key=lambda item: (item["model_alias"], item["task_id"]))
    return entries


def _zip_info(name):
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    return info


def _canonical_json_bytes(value):
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("utf-8")


def build_package(
    results_root,
    output_zip,
    executor_path,
    allowed_models,
    allowed_task_ids=None,
    semantic_evaluator_path=None,
    batch_evaluator_path=None,
    prompt_version=None,
    resource_guard_path=None,
):
    allowed_models = set(allowed_models)
    allowed_task_ids = (
        set(allowed_task_ids) if allowed_task_ids is not None else None
    )
    collected = collect_entries(
        results_root,
        allowed_models,
        allowed_task_ids=allowed_task_ids,
    )
    if not collected:
        raise ValueError("no generated artifacts matched the package scope")
    if allowed_task_ids is not None:
        expected = set(
            (model_alias, task_id)
            for model_alias in allowed_models
            for task_id in allowed_task_ids
        )
        actual = set(
            (entry["model_alias"], entry["task_id"])
            for entry in collected
        )
        missing = sorted(expected - actual)
        if missing:
            raise ValueError(
                "missing terminal generation artifacts: {}".format(
                    ", ".join("{}/{}".format(*item) for item in missing)
                )
            )
    manifest_entries = []
    for entry in collected:
        manifest_entries.append(dict(
            (key, value) for key, value in entry.items() if not key.startswith("_")
        ))
    if resource_guard_path is None:
        resource_guard_path = os.path.join(
            os.path.dirname(os.path.abspath(executor_path)),
            "resource_aware_guard.py",
        )
    if not os.path.isfile(resource_guard_path):
        raise ValueError(
            "resource guard runtime not found: {}".format(resource_guard_path)
        )
    runtime_sha256 = {
        "executor": _sha256_bytes(_read_bytes(executor_path)),
        "resource_guard": _sha256_bytes(_read_bytes(resource_guard_path)),
    }
    manifest = {
        "entries": manifest_entries,
        "entry_count": len(manifest_entries),
        "runtime_sha256": runtime_sha256,
        "schema_version": (
            "1.1" if any(
                entry.get("entry_type") == "generation_failure"
                for entry in collected
            ) else "1.0"
        ),
    }
    evaluator_paths = (semantic_evaluator_path, batch_evaluator_path)
    if any(evaluator_paths) and not all(evaluator_paths):
        raise ValueError("both evaluator runtime paths are required")
    if all(evaluator_paths):
        if not prompt_version:
            raise ValueError("prompt_version is required for evaluator packages")
        runtime_sha256.update({
            "semantic_evaluator": _sha256_bytes(
                _read_bytes(semantic_evaluator_path)
            ),
            "batch_evaluator": _sha256_bytes(
                _read_bytes(batch_evaluator_path)
            ),
        })
        manifest.update({
            "evaluator_version": "semantic-evaluator-v2",
            "prompt_version": prompt_version,
            "schema_version": (
                "2.1" if any(
                    entry.get("entry_type") == "generation_failure"
                    for entry in collected
                ) else "2.0"
            ),
        })
    output_directory = os.path.dirname(os.path.abspath(output_zip))
    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)
    temporary = output_zip + ".tmp"
    try:
        with zipfile.ZipFile(temporary, "w") as archive:
            archive.writestr(_zip_info("manifest.json"), _canonical_json_bytes(manifest))
            archive.writestr(
                _zip_info("execute_generated_batch.py"),
                _read_bytes(executor_path),
            )
            archive.writestr(
                _zip_info("resource_aware_guard.py"),
                _read_bytes(resource_guard_path),
            )
            if all(evaluator_paths):
                archive.writestr(
                    _zip_info("semantic_evaluator.py"),
                    _read_bytes(semantic_evaluator_path),
                )
                archive.writestr(
                    _zip_info("evaluate_generated_batch.py"),
                    _read_bytes(batch_evaluator_path),
                )
            for entry in collected:
                if entry.get("entry_type") == "generation_failure":
                    archive.writestr(
                        _zip_info(entry["failure_path"]),
                        _read_bytes(entry["_source_failure_path"]),
                    )
                else:
                    archive.writestr(
                        _zip_info(entry["program_path"]),
                        _read_bytes(entry["_source_program_path"]),
                    )
        os.replace(temporary, output_zip)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)
    return manifest


def _defaults():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return {
        "executor": os.path.join(root, "scripts", "execute_generated_batch.py"),
        "resource_guard": os.path.join(
            root, "scripts", "resource_aware_guard.py"
        ),
        "models": os.path.join(root, "experiments", "config", "models.json"),
        "output": os.path.join(root, "experiments", "packages", "pilot-smoke.zip"),
        "results": os.path.join(
            root, "experiments", "results", "pilot", "primary-zero-shot-v1"
        ),
    }


def main(argv=None):
    defaults = _defaults()
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default=defaults["results"])
    parser.add_argument("--models-config", default=defaults["models"])
    parser.add_argument("--output", default=defaults["output"])
    parser.add_argument("--executor", default=defaults["executor"])
    parser.add_argument("--resource-guard", default=defaults["resource_guard"])
    parser.add_argument("--semantic-evaluator")
    parser.add_argument("--batch-evaluator")
    parser.add_argument("--prompt-version")
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--task-ids", nargs="+")
    args = parser.parse_args(argv)
    models = load_models(args.models_config)
    aliases = args.models if args.models else list(models.keys())
    unknown = [alias for alias in aliases if alias not in models]
    if unknown:
        raise ValueError("unknown model aliases: {}".format(", ".join(unknown)))
    manifest = build_package(
        args.results_root,
        args.output,
        args.executor,
        set(aliases),
        allowed_task_ids=args.task_ids,
        semantic_evaluator_path=args.semantic_evaluator,
        batch_evaluator_path=args.batch_evaluator,
        prompt_version=args.prompt_version,
        resource_guard_path=args.resource_guard,
    )
    print(json.dumps({
        "entry_count": manifest["entry_count"],
        "output": os.path.abspath(args.output),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

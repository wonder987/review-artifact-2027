#!/usr/bin/env python3
"""Build a deterministic, code-only Pilot diagnostic package for cs1."""

from __future__ import print_function

import argparse
import hashlib
import json
import os
import zipfile


PACKAGE_VERSION = "pilot-taxonomy-v3-tools-v2"
RUNTIME_NAMES = (
    "analyze_pilot_failures.py",
    "evaluate_generated_batch.py",
    "failure_taxonomy.py",
    "semantic_evaluator.py",
)
DEFAULT_TAXONOMY_NAME = "failure_taxonomy_v1.json"


def _read_bytes(path):
    with open(path, "rb") as handle:
        return handle.read()


def _sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def _zip_info(name):
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    return info


def _canonical_json_bytes(value):
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("utf-8")


def build_package(output_zip, runtime_files, taxonomy_path):
    if set(runtime_files) != set(RUNTIME_NAMES):
        missing = sorted(set(RUNTIME_NAMES) - set(runtime_files))
        unexpected = sorted(set(runtime_files) - set(RUNTIME_NAMES))
        raise ValueError(
            "runtime names mismatch; missing={}, unexpected={}".format(
                missing, unexpected
            )
        )
    members = {}
    for name in RUNTIME_NAMES:
        path = runtime_files[name]
        if not os.path.isfile(path):
            raise ValueError("missing runtime file: {}".format(path))
        members[name] = _read_bytes(path)
    if not os.path.isfile(taxonomy_path):
        raise ValueError("missing taxonomy file: {}".format(taxonomy_path))
    taxonomy_name = os.path.basename(taxonomy_path)
    if (
        not taxonomy_name.startswith("failure_taxonomy_")
        or not taxonomy_name.endswith(".json")
        or taxonomy_name in (".", "..")
    ):
        raise ValueError("invalid taxonomy filename: {}".format(taxonomy_name))
    taxonomy_bytes = _read_bytes(taxonomy_path)
    try:
        taxonomy = json.loads(taxonomy_bytes.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as error:
        raise ValueError("invalid taxonomy JSON: {}".format(error))
    members[taxonomy_name] = taxonomy_bytes

    manifest = {
        "package_version": PACKAGE_VERSION,
        "runtime_file_count": len(members),
        "runtime_sha256": dict(
            (name, _sha256_bytes(value))
            for name, value in sorted(members.items())
        ),
        "schema_version": "1.0",
        "taxonomy_version": taxonomy.get("taxonomy_version"),
    }
    output_directory = os.path.dirname(os.path.abspath(output_zip))
    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)
    temporary = output_zip + ".tmp"
    try:
        with zipfile.ZipFile(temporary, "w") as archive:
            archive.writestr(
                _zip_info("manifest.json"), _canonical_json_bytes(manifest)
            )
            for name in sorted(members):
                archive.writestr(_zip_info(name), members[name])
        os.replace(temporary, output_zip)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)
    return manifest


def _defaults():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return {
        "semantic_evaluator.py": os.path.join(
            root, "scripts", "semantic_evaluator.py"
        ),
        "evaluate_generated_batch.py": os.path.join(
            root, "scripts", "evaluate_generated_batch.py"
        ),
        "failure_taxonomy.py": os.path.join(
            root, "scripts", "failure_taxonomy.py"
        ),
        "analyze_pilot_failures.py": os.path.join(
            root, "scripts", "analyze_pilot_failures.py"
        ),
        "taxonomy": os.path.join(
            root, "experiments", "config", DEFAULT_TAXONOMY_NAME
        ),
        "output": os.path.join(
            root,
            "experiments",
            "packages",
            "pilot-taxonomy-v3-tools-v1.zip",
        ),
    }


def main(argv=None):
    defaults = _defaults()
    parser = argparse.ArgumentParser()
    parser.add_argument("--semantic-evaluator", default=defaults["semantic_evaluator.py"])
    parser.add_argument("--batch-evaluator", default=defaults["evaluate_generated_batch.py"])
    parser.add_argument("--failure-taxonomy-module", default=defaults["failure_taxonomy.py"])
    parser.add_argument("--analyzer", default=defaults["analyze_pilot_failures.py"])
    parser.add_argument("--taxonomy", default=defaults["taxonomy"])
    parser.add_argument("--output", default=defaults["output"])
    args = parser.parse_args(argv)
    runtime_files = {
        "semantic_evaluator.py": args.semantic_evaluator,
        "evaluate_generated_batch.py": args.batch_evaluator,
        "failure_taxonomy.py": args.failure_taxonomy_module,
        "analyze_pilot_failures.py": args.analyzer,
    }
    manifest = build_package(args.output, runtime_files, args.taxonomy)
    print(json.dumps({
        "output": os.path.abspath(args.output),
        "runtime_file_count": manifest["runtime_file_count"],
        "taxonomy_version": manifest["taxonomy_version"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

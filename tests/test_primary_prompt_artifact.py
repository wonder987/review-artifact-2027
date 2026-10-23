import json
import os
import subprocess
import sys
import tempfile
import unittest

from scripts.audit_primary_zero_shot import (
    audit_files,
    file_sha256,
    load_jsonl,
    write_index,
)
from scripts.render_primary_zero_shot import render_manifest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(ROOT, "benchmark", "gold138_manifest.json")
SPEC_RICH = os.path.join(ROOT, "benchmark", "gold138_prompts.jsonl")
PRIMARY_V1 = os.path.join(
    ROOT, "benchmark", "gold138_primary_zero_shot.jsonl"
)
PRIMARY_INDEX_V1 = os.path.join(
    ROOT, "benchmark", "gold138_prompt_set_index.json"
)

PROTECTED_HASHES = {
    MANIFEST: "5c00943dd49ff5e7ff67d318e9dda3fcb394164b407c02ecaf261e8b6eff6fac",
    SPEC_RICH: "1b395f63e94c7225b1686322a436441c371bebaea3dd7966cf187535e7a172f4",
    PRIMARY_V1: "8a18fa1ea48b24a8f5a98d095187f55003460de7d8c0a8576d4717d4c99855cd",
    PRIMARY_INDEX_V1: "9fb12f977f9850779c6b0cca06cae4cc42a23325360e0b03c2cb6c64402f4ded",
}

PRIMARY_V1_SHA256 = PROTECTED_HASHES[PRIMARY_V1]


class PrimaryPromptArtifactTests(unittest.TestCase):
    def test_auditor_cli_runs_by_script_path(self):
        with tempfile.TemporaryDirectory() as directory:
            primary_path = os.path.join(directory, "primary.jsonl")
            index_path = os.path.join(directory, "index.json")
            render_manifest(MANIFEST, primary_path)
            script = os.path.join(ROOT, "scripts", "audit_primary_zero_shot.py")
            completed = subprocess.run(
                [
                    sys.executable,
                    script,
                    "--manifest",
                    MANIFEST,
                    "--primary",
                    primary_path,
                    "--spec-rich",
                    SPEC_RICH,
                    "--index-output",
                    index_path,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
            )

        self.assertEqual(0, completed.returncode, completed.stderr)

    def test_certified_artifact_has_expected_counts_and_hash_index(self):
        with tempfile.TemporaryDirectory() as directory:
            primary_path = os.path.join(directory, "primary.jsonl")
            index_path = os.path.join(directory, "index.json")
            records = render_manifest(MANIFEST, primary_path)
            report = audit_files(MANIFEST, primary_path, SPEC_RICH)
            write_index(index_path, report, records)

            with open(index_path, "r") as handle:
                index = json.load(handle)

        self.assertEqual("passed", index["status"])
        self.assertEqual(138, index["record_count"])
        self.assertEqual(30, index["pilot_count"])
        self.assertEqual(108, index["held_out_count"])
        self.assertEqual(138, len(index["prompts"]))
        self.assertEqual(
            [record["task_id"] for record in records],
            [record["task_id"] for record in index["prompts"]],
        )
        self.assertTrue(all(
            len(record["prompt_sha256"]) == 64 for record in index["prompts"]
        ))

    def test_protected_benchmark_inputs_are_byte_for_byte_unchanged(self):
        for path, expected_hash in PROTECTED_HASHES.items():
            self.assertEqual(expected_hash, file_sha256(path))

    def test_rendered_artifact_ids_match_manifest_and_are_unique(self):
        with tempfile.TemporaryDirectory() as directory:
            primary_path = os.path.join(directory, "primary.jsonl")
            render_manifest(MANIFEST, primary_path)
            records = load_jsonl(primary_path)
            with open(MANIFEST, "r") as handle:
                manifest = json.load(handle)

        expected_ids = [task["id"] for task in manifest["tasks"]]
        actual_ids = [record["task_id"] for record in records]
        self.assertEqual(expected_ids, actual_ids)
        self.assertEqual(138, len(set(actual_ids)))

    def test_rendered_v1_is_byte_identical_to_certified_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            rendered_path = os.path.join(directory, "primary.jsonl")
            render_manifest(MANIFEST, rendered_path)

            self.assertEqual(PRIMARY_V1_SHA256, file_sha256(rendered_path))


if __name__ == "__main__":
    unittest.main()

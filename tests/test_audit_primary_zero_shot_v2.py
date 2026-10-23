import copy
import hashlib
import json
import os
import tempfile
import unittest

from scripts.audit_primary_zero_shot import AuditError
from scripts.audit_primary_zero_shot_v2 import (
    audit_files,
    audit_records,
    write_index,
)
from scripts.render_primary_zero_shot_v2 import render_manifest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(ROOT, "benchmark", "gold138_manifest.json")
SPEC_RICH = os.path.join(ROOT, "benchmark", "gold138_prompts.jsonl")


def load_manifest():
    with open(MANIFEST, "r") as handle:
        return json.load(handle)


def rehash(record):
    record["prompt"] = record["system_prompt"] + "\n\n" + record["user_prompt"]
    record["prompt_sha256"] = hashlib.sha256(
        record["prompt"].encode("utf-8")
    ).hexdigest()


class PrimaryZeroShotV2AuditTests(unittest.TestCase):
    def test_accepts_sha256_field_name_and_all_rendered_records(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "v2.jsonl")
            records = render_manifest(MANIFEST, path)
            report = audit_records(load_manifest(), records)

        self.assertEqual("passed", report["status"])
        self.assertEqual("primary-zero-shot-v2", report["prompt_version"])
        self.assertEqual(138, report["record_count"])

    def test_rejects_actual_sha256_digest_in_model_visible_text(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "v2.jsonl")
            records = render_manifest(MANIFEST, path)
            leaked = copy.deepcopy(records)
            leaked[0]["user_prompt"] += "\nExpected digest: " + ("a" * 64)
            rehash(leaked[0])

            with self.assertRaises(AuditError):
                audit_records(load_manifest(), leaked)

    def test_rejects_hidden_subtree_and_expected_decision(self):
        manifest = load_manifest()
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "v2.jsonl")
            records = render_manifest(MANIFEST, path)

            hidden_task = next(
                task for task in manifest["tasks"]
                if task["subtype"] == "subtree_discovery"
            )
            leaked = copy.deepcopy(records)
            record = next(r for r in leaked if r["task_id"] == hidden_task["id"])
            record["user_prompt"] += "\nsubtree=" + hidden_task["targets"][0]["subtree"]
            rehash(record)
            with self.assertRaises(AuditError):
                audit_records(manifest, leaked)

            adaptive = next(
                task for task in manifest["tasks"]
                if task["subtype"] == "adaptive_access"
            )
            leaked = copy.deepcopy(records)
            record = next(r for r in leaked if r["task_id"] == adaptive["id"])
            record["user_prompt"] += "\n" + adaptive["operation"]["expected_decision"]
            rehash(record)
            with self.assertRaises(AuditError):
                audit_records(manifest, leaked)

    def test_file_audit_reports_v2_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "v2.jsonl")
            render_manifest(MANIFEST, path)
            report = audit_files(MANIFEST, path, SPEC_RICH)

        self.assertEqual("primary-zero-shot-v2", report["prompt_version"])
        self.assertEqual(64, len(report["primary_jsonl_sha256"]))

    def test_v2_index_is_byte_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            primary = os.path.join(directory, "v2.jsonl")
            records = render_manifest(MANIFEST, primary)
            report = audit_files(MANIFEST, primary, SPEC_RICH)
            first = os.path.join(directory, "first.json")
            second = os.path.join(directory, "second.json")
            write_index(first, report, records)
            write_index(second, report, records)
            with open(first, "rb") as handle:
                first_bytes = handle.read()
            with open(second, "rb") as handle:
                second_bytes = handle.read()

        self.assertEqual(first_bytes, second_bytes)
        self.assertNotIn(b"created_utc", first_bytes)


if __name__ == "__main__":
    unittest.main()

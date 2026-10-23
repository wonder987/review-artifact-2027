import copy
import hashlib
import json
import os
import tempfile
import unittest

from scripts.audit_primary_zero_shot import AuditError
from scripts.audit_primary_zero_shot_v3 import audit_files, audit_records, write_index
from scripts.render_primary_zero_shot_v3 import render_manifest


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


class PrimaryZeroShotV3AuditTests(unittest.TestCase):
    def render(self, directory):
        path = os.path.join(directory, "v3.jsonl")
        return path, render_manifest(MANIFEST, path)

    def test_accepts_all_138_records_and_expected_split(self):
        with tempfile.TemporaryDirectory() as directory:
            _, records = self.render(directory)
            report = audit_records(load_manifest(), records)
        self.assertEqual("passed", report["status"])
        self.assertEqual("primary-zero-shot-v3", report["prompt_version"])
        self.assertEqual(138, report["record_count"])
        self.assertEqual(30, report["pilot_count"])
        self.assertEqual(108, report["held_out_count"])

    def test_rejects_hidden_subtree_gold_hash_and_reference_index(self):
        manifest = load_manifest()
        with tempfile.TemporaryDirectory() as directory:
            _, records = self.render(directory)
            hidden_task = next(
                task for task in manifest["tasks"]
                if task["subtype"] == "subtree_discovery"
            )
            hidden = copy.deepcopy(records)
            record = next(r for r in hidden if r["task_id"] == hidden_task["id"])
            record["user_prompt"] += "\nsubtree=" + hidden_task["targets"][0]["subtree"]
            rehash(record)
            with self.assertRaises(AuditError):
                audit_records(manifest, hidden)

            digest = copy.deepcopy(records)
            digest[0]["user_prompt"] += "\nGold=" + ("a" * 64)
            rehash(digest[0])
            with self.assertRaises(AuditError):
                audit_records(manifest, digest)

            indexed_task = next(
                task for task in manifest["tasks"]
                if "index_start" in task["operation"]
            )
            leaked_index = copy.deepcopy(records)
            record = next(r for r in leaked_index if r["task_id"] == indexed_task["id"])
            record["user_prompt"] += "\nreference_index=" + str(
                indexed_task["operation"]["index_start"]
            )
            rehash(record)
            with self.assertRaises(AuditError):
                audit_records(manifest, leaked_index)

    def test_rejects_api_example_missing_validation_and_wrong_version(self):
        manifest = load_manifest()
        with tempfile.TemporaryDirectory() as directory:
            _, records = self.render(directory)

            api_example = copy.deepcopy(records)
            api_example[0]["user_prompt"] += "\nUse connection.openTree(tree, shot)."
            rehash(api_example[0])
            with self.assertRaises(AuditError):
                audit_records(manifest, api_example)

            missing_validation = copy.deepcopy(records)
            missing_validation[0]["user_prompt"] = missing_validation[0][
                "user_prompt"
            ].replace(", and time_strictly_increasing", "")
            rehash(missing_validation[0])
            with self.assertRaises(AuditError):
                audit_records(manifest, missing_validation)

            wrong_version = copy.deepcopy(records)
            wrong_version[0]["prompt_version"] = "primary-zero-shot-v2"
            with self.assertRaises(AuditError):
                audit_records(manifest, wrong_version)

    def test_file_audit_and_index_are_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            primary, records = self.render(directory)
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
        self.assertEqual(64, len(report["primary_jsonl_sha256"]))


if __name__ == "__main__":
    unittest.main()

import copy
import hashlib
import json
import os
import tempfile
import unittest

from scripts.audit_primary_zero_shot import (
    AuditError,
    audit_files,
    audit_records,
    contains_token,
    load_jsonl,
)
from scripts.render_primary_zero_shot import render_manifest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(ROOT, "benchmark", "gold138_manifest.json")
SPEC_RICH = os.path.join(ROOT, "benchmark", "gold138_prompts.jsonl")


def load_manifest():
    with open(MANIFEST, "r") as handle:
        return json.load(handle)


def build_records(directory):
    path = os.path.join(directory, "primary.jsonl")
    return path, render_manifest(MANIFEST, path)


def rehash(record):
    record["prompt"] = record["system_prompt"] + "\n\n" + record["user_prompt"]
    record["prompt_sha256"] = hashlib.sha256(
        record["prompt"].encode("utf-8")
    ).hexdigest()


class PrimaryZeroShotAuditTests(unittest.TestCase):
    def test_accepts_rendered_records_and_reports_split_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            _, records = build_records(directory)
            report = audit_records(load_manifest(), records, load_jsonl(SPEC_RICH))

        self.assertEqual("passed", report["status"])
        self.assertEqual(138, report["record_count"])
        self.assertEqual(30, report["pilot_count"])
        self.assertEqual(108, report["held_out_count"])

    def test_rejects_missing_duplicate_and_wrong_split_records(self):
        with tempfile.TemporaryDirectory() as directory:
            _, records = build_records(directory)
            manifest = load_manifest()
            with self.assertRaises(AuditError):
                audit_records(manifest, records[:-1])

            duplicated = copy.deepcopy(records)
            duplicated[-1]["task_id"] = duplicated[0]["task_id"]
            with self.assertRaises(AuditError):
                audit_records(manifest, duplicated)

            wrong_split = copy.deepcopy(records)
            wrong_split[0]["split"] = "pilot"
            with self.assertRaises(AuditError):
                audit_records(manifest, wrong_split)

    def test_rejects_prompt_composition_and_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            _, records = build_records(directory)
            manifest = load_manifest()

            wrong_composition = copy.deepcopy(records)
            wrong_composition[0]["prompt"] += "\nextra"
            wrong_composition[0]["prompt_sha256"] = hashlib.sha256(
                wrong_composition[0]["prompt"].encode("utf-8")
            ).hexdigest()
            with self.assertRaises(AuditError):
                audit_records(manifest, wrong_composition)

            wrong_hash = copy.deepcopy(records)
            wrong_hash[0]["prompt_sha256"] = "0" * 64
            with self.assertRaises(AuditError):
                audit_records(manifest, wrong_hash)

    def test_rejects_internal_labels_in_model_visible_text(self):
        with tempfile.TemporaryDirectory() as directory:
            _, records = build_records(directory)
            leaked = copy.deepcopy(records)
            leaked[0]["user_prompt"] += "\nInternal class: L1 simple_retrieval"
            rehash(leaked[0])

            with self.assertRaises(AuditError) as context:
                audit_records(load_manifest(), leaked)

        self.assertIn(leaked[0]["task_id"], str(context.exception))

    def test_rejects_hidden_subtree_and_reference_index_leakage(self):
        manifest = load_manifest()
        with tempfile.TemporaryDirectory() as directory:
            _, records = build_records(directory)
            by_id = dict((record["task_id"], record) for record in records)

            subtree_task = next(
                task for task in manifest["tasks"]
                if task["subtype"] == "subtree_discovery"
            )
            leaked_subtree = copy.deepcopy(records)
            leaked_record = next(
                record for record in leaked_subtree
                if record["task_id"] == subtree_task["id"]
            )
            leaked_record["user_prompt"] += "\nUse subtree {}.".format(
                subtree_task["targets"][0]["subtree"]
            )
            rehash(leaked_record)
            with self.assertRaises(AuditError):
                audit_records(manifest, leaked_subtree)

            window_task = next(
                task for task in manifest["tasks"]
                if task["subtype"] == "exact_window"
            )
            leaked_index = copy.deepcopy(records)
            leaked_record = next(
                record for record in leaked_index
                if record["task_id"] == window_task["id"]
            )
            leaked_record["user_prompt"] += "\nUse index_start {}.".format(
                window_task["operation"]["index_start"]
            )
            rehash(leaked_record)
            with self.assertRaises(AuditError):
                audit_records(manifest, leaked_index)

            self.assertIn(window_task["id"], by_id)

    def test_rejects_adaptive_expected_decision_and_spec_prompt_reuse(self):
        manifest = load_manifest()
        with tempfile.TemporaryDirectory() as directory:
            _, records = build_records(directory)
            adaptive = next(
                task for task in manifest["tasks"]
                if task["subtype"] == "adaptive_access"
            )
            leaked = copy.deepcopy(records)
            record = next(
                item for item in leaked if item["task_id"] == adaptive["id"]
            )
            record["user_prompt"] += "\nExpected: {}.".format(
                adaptive["operation"]["expected_decision"]
            )
            rehash(record)
            with self.assertRaises(AuditError):
                audit_records(manifest, leaked)

            spec_records = load_jsonl(SPEC_RICH)
            reused = copy.deepcopy(records)
            reused[0]["system_prompt"] = ""
            reused[0]["user_prompt"] = spec_records[0]["prompt"]
            reused[0]["prompt"] = spec_records[0]["prompt"]
            reused[0]["prompt_sha256"] = hashlib.sha256(
                reused[0]["prompt"].encode("utf-8")
            ).hexdigest()
            with self.assertRaises(AuditError):
                audit_records(manifest, reused, spec_records)

    def test_load_jsonl_rejects_malformed_or_non_object_lines(self):
        with tempfile.TemporaryDirectory() as directory:
            malformed = os.path.join(directory, "malformed.jsonl")
            with open(malformed, "w") as handle:
                handle.write('{"task_id": "ok"}\nnot-json\n')
            with self.assertRaises(AuditError):
                load_jsonl(malformed)

            non_object = os.path.join(directory, "non-object.jsonl")
            with open(non_object, "w") as handle:
                handle.write("[]\n")
            with self.assertRaises(AuditError):
                load_jsonl(non_object)

    def test_token_matching_uses_identifier_boundaries(self):
        self.assertTrue(contains_token("use subtree t4", "t4"))
        self.assertTrue(contains_token("class L2", "L2"))
        self.assertFalse(contains_token("status_t4_ok", "t4"))
        self.assertFalse(contains_token("MODEL2", "L2"))

    def test_file_audit_reports_artifact_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            primary_path, _ = build_records(directory)
            report = audit_files(MANIFEST, primary_path, SPEC_RICH)

        self.assertEqual(64, len(report["source_manifest_sha256"]))
        self.assertEqual(64, len(report["primary_jsonl_sha256"]))
        self.assertEqual(64, len(report["spec_rich_jsonl_sha256"]))


if __name__ == "__main__":
    unittest.main()

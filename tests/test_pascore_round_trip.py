import hashlib
import json
import os
import sys
import tempfile
import unittest
import zipfile

from scripts.package_pascore_diagnostics import (
    build_diagnostics_package,
    sanitize_evaluation,
)
from scripts.package_pascore_round import build_execution_package
from scripts.evaluate_generated_batch import evaluate_batch
from scripts.run_pascore_cs1_round import run_round, verify_execution_package
from scripts.metadata_grounding import build_grounding


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def write_json(path, value):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        json.dump(value, handle)


def sha256_text(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class PascoreRoundTripTests(unittest.TestCase):
    def test_sparse_repair_package_evaluates_only_manifest_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            execution_dir = os.path.join(directory, "execution")
            gold_dir = os.path.join(directory, "gold")
            output_dir = os.path.join(directory, "evaluation")
            task_manifest = os.path.join(directory, "tasks.json")
            reference_index = os.path.join(gold_dir, "reference_index.json")
            protocol = os.path.join(directory, "protocol.json")
            package_manifest = os.path.join(directory, "package.json")
            tasks = []
            references = {}
            for task_id in ("task-1", "task-2"):
                tasks.append({
                    "id": task_id,
                    "level": "L1",
                    "subtype": "simple_retrieval",
                })
                repeat_name = task_id + ".json"
                write_json(os.path.join(gold_dir, repeat_name), {
                    "status": "success",
                    "task_id": task_id,
                    "targets": [],
                })
                references[task_id] = {
                    "stability": {"stable": True},
                    "repeat_files": [repeat_name],
                }
            write_json(task_manifest, {"tasks": tasks})
            write_json(reference_index, {"tasks": references})
            write_json(protocol, {"protocol_id": "repair"})
            pairs = (("qwen", "task-1"), ("gemini", "task-2"))
            write_json(package_manifest, {
                "entries": [
                    {"model_alias": model, "task_id": task}
                    for model, task in pairs
                ]
            })
            for model_alias, task_id in pairs:
                write_json(os.path.join(
                    execution_dir, model_alias, task_id, "execution.json"
                ), {
                    "status": "program_failure",
                    "error_type": "RuntimeError",
                    "error_message": "failed",
                })

            summary = evaluate_batch(
                execution_dir,
                gold_dir,
                reference_index,
                task_manifest,
                output_dir,
                ["qwen", "gemini"],
                ["task-1", "task-2"],
                protocol,
                package_manifest_path=package_manifest,
            )

        self.assertEqual(2, summary["total"])
        self.assertEqual(1, summary["by_model"]["qwen"]["total"])
        self.assertEqual(1, summary["by_model"]["gemini"]["total"])

    def test_sanitizer_removes_expected_and_observed_values(self):
        sanitized = sanitize_evaluation({
            "task_id": "task",
            "model_alias": "qwen",
            "category": "semantic_failure",
            "code": "signal_hash_mismatch",
            "semantic_success": False,
            "mismatches": [{
                "path": "targets[0].signal.sha256",
                "code": "signal_hash_mismatch",
                "expected": "gold-secret",
                "observed": "model-value",
            }],
        })

        serialized = json.dumps(sanitized, sort_keys=True)
        self.assertNotIn("gold-secret", serialized)
        self.assertNotIn("model-value", serialized)
        self.assertEqual(
            "targets[0].signal.sha256", sanitized["mismatches"][0]["path"]
        )

    def test_build_run_and_export_failure_round(self):
        with tempfile.TemporaryDirectory() as directory:
            results = os.path.join(directory, "generation")
            program = (
                "import json, sys\n"
                "print(json.dumps({'status':'failure','failed_stage':'retrieval',"
                "'error_type':'AttributeError','error_message':\"'Connection' "
                "object has no attribute 'getNode'\",'accessed_objects':[],"
                "'elapsed_seconds':0.1}))\n"
                "sys.exit(1)\n"
            )
            generation_dir = os.path.join(
                results, "qwen", "task-1", "attempt-1"
            )
            os.makedirs(generation_dir)
            with open(os.path.join(generation_dir, "program.py"), "wb") as handle:
                handle.write(program.encode("utf-8"))
            write_json(os.path.join(generation_dir, "result.json"), {
                "status": "completed",
                "identity": {
                    "attempt": 1,
                    "model_alias": "qwen",
                    "prompt_hash": "1" * 64,
                    "protocol_hash": "2" * 64,
                    "task_id": "task-1",
                },
                "extraction": {
                    "status": "success",
                    "program_sha256": sha256_text(program),
                },
            })
            task = {
                "id": "task-1",
                "level": "L1",
                "operation": {"kind": "full_retrieval"},
                "prompt_spec": {
                    "disclose_indices": False,
                    "disclose_subtree": True,
                },
                "safety_policy": {
                    "max_points": 5000000,
                    "max_total_mib": 256.0,
                    "timeout_seconds": 90,
                },
                "spec_sha256": "3" * 64,
                "split": "pilot",
                "subtype": "simple_retrieval",
                "targets": [{
                    "estimated_total_mib": 1.0,
                    "node": "\\SIG",
                    "sample_num": 100,
                    "server": "server",
                    "shot": 1,
                    "signal_name": "SIG",
                    "subtree": "t1",
                    "tree": "east",
                }],
            }
            task_manifest = os.path.join(directory, "tasks.json")
            protocol_path = os.path.join(directory, "protocol.json")
            condition_path = os.path.join(directory, "condition.json")
            write_json(task_manifest, {"tasks": [task]})
            grounding_hash = build_grounding(task, enabled=True)["grounding_sha256"]
            result_path = os.path.join(generation_dir, "result.json")
            with open(result_path, "r") as handle:
                generation_result = json.load(handle)
            generation_result["identity"]["grounding_hash"] = grounding_hash
            write_json(result_path, generation_result)
            write_json(protocol_path, {"protocol_id": "test"})
            write_json(condition_path, {
                "condition_id": "full",
                "failure_diagnosis": "multi_grained",
                "max_repair_rounds": 2,
                "metadata_grounding": True,
                "resource_guard": True,
                "schema_version": "1.0",
                "structured_feedback": True,
            })
            package = os.path.join(directory, "round.zip")

            manifest = build_execution_package(
                results,
                package,
                task_manifest,
                protocol_path,
                condition_path,
                "run-1",
                0,
                {"qwen"},
                {"task-1"},
            )
            extracted = os.path.join(directory, "package")
            with zipfile.ZipFile(package, "r") as archive:
                archive.extractall(extracted)
            gold_dir = os.path.join(directory, "gold")
            os.makedirs(gold_dir)
            write_json(os.path.join(gold_dir, "task-1.json"), {
                "status": "success",
                "task_id": "task-1",
                "targets": [],
            })
            reference_index = os.path.join(gold_dir, "reference_index.json")
            write_json(reference_index, {
                "tasks": {
                    "task-1": {
                        "stability": {"stable": True},
                        "repeat_files": ["task-1.json"],
                    }
                }
            })
            output = os.path.join(directory, "cs1-output")

            summary = run_round(
                extracted,
                output,
                gold_dir,
                reference_index,
                python_executable=sys.executable,
                timeout_seconds=5,
            )
            diagnostic_zip = os.path.join(directory, "diagnostics.zip")
            diagnostic_manifest = build_diagnostics_package(
                output,
                os.path.join(extracted, "manifest.json"),
                diagnostic_zip,
            )
            with zipfile.ZipFile(diagnostic_zip, "r") as archive:
                feedback = json.loads(
                    archive.read("feedback/qwen/task-1/feedback.json").decode("utf-8")
                )
                names = archive.namelist()

        self.assertEqual(1, manifest["entry_count"])
        self.assertEqual(1, summary["feedback_ready"])
        self.assertEqual("hallucinated_api", feedback["failure_subtype"])
        self.assertIn("evaluation/qwen/task-1/evaluation.json", names)
        self.assertEqual(1, diagnostic_manifest["entry_count"])

    def test_full_package_rejects_stale_grounding_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            results = os.path.join(directory, "generation")
            generation_dir = os.path.join(results, "qwen", "task-1", "attempt-1")
            os.makedirs(generation_dir)
            program = "print('{}')\n"
            with open(os.path.join(generation_dir, "program.py"), "wb") as handle:
                handle.write(program.encode("utf-8"))
            write_json(os.path.join(generation_dir, "result.json"), {
                "status": "completed",
                "identity": {
                    "attempt": 1,
                    "grounding_hash": "0" * 64,
                    "model_alias": "qwen",
                    "prompt_hash": "1" * 64,
                    "protocol_hash": "2" * 64,
                    "task_id": "task-1",
                },
                "extraction": {
                    "status": "success",
                    "program_sha256": sha256_text(program),
                },
            })
            task = {
                "id": "task-1",
                "level": "L1",
                "operation": {"kind": "full_retrieval"},
                "prompt_spec": {},
                "safety_policy": {},
                "spec_sha256": "3" * 64,
                "split": "pilot",
                "subtype": "simple_retrieval",
                "targets": [{
                    "node": r"\SIG", "server": "server", "shot": 1,
                    "signal_name": "SIG", "subtree": "t1", "tree": "east",
                }],
            }
            task_manifest = os.path.join(directory, "tasks.json")
            protocol = os.path.join(directory, "protocol.json")
            condition = os.path.join(directory, "condition.json")
            write_json(task_manifest, {"tasks": [task]})
            write_json(protocol, {"protocol_id": "test"})
            write_json(condition, {
                "condition_id": "full",
                "failure_diagnosis": "multi_grained",
                "max_repair_rounds": 2,
                "metadata_grounding": True,
                "resource_guard": True,
                "schema_version": "1.0",
                "structured_feedback": True,
            })

            with self.assertRaisesRegex(ValueError, "grounding hash mismatch"):
                build_execution_package(
                    results, os.path.join(directory, "round.zip"), task_manifest,
                    protocol, condition, "run-1", 0, {"qwen"}, {"task-1"},
                )

    def test_no_metadata_package_accepts_generation_without_grounding_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            results = os.path.join(directory, "generation")
            generation_dir = os.path.join(results, "qwen", "task-1", "attempt-1")
            os.makedirs(generation_dir)
            program = "print('{}')\n"
            with open(os.path.join(generation_dir, "program.py"), "wb") as handle:
                handle.write(program.encode("utf-8"))
            write_json(os.path.join(generation_dir, "result.json"), {
                "status": "completed",
                "identity": {
                    "attempt": 1,
                    "model_alias": "qwen",
                    "prompt_hash": "1" * 64,
                    "protocol_hash": "2" * 64,
                    "task_id": "task-1",
                },
                "extraction": {
                    "status": "success",
                    "program_sha256": sha256_text(program),
                },
            })
            task = {
                "id": "task-1",
                "level": "L1",
                "operation": {"kind": "full_retrieval"},
                "prompt_spec": {},
                "safety_policy": {},
                "spec_sha256": "3" * 64,
                "split": "held_out",
                "subtype": "simple_retrieval",
                "targets": [{
                    "node": r"\SIG", "server": "server", "shot": 1,
                    "signal_name": "SIG", "subtree": "t1", "tree": "east",
                }],
            }
            task_manifest = os.path.join(directory, "tasks.json")
            protocol = os.path.join(directory, "protocol.json")
            condition = os.path.join(directory, "condition.json")
            write_json(task_manifest, {"tasks": [task]})
            write_json(protocol, {"protocol_id": "primary-zero-shot"})
            write_json(condition, {
                "condition_id": "no_metadata_grounding",
                "failure_diagnosis": "multi_grained",
                "max_repair_rounds": 2,
                "metadata_grounding": False,
                "resource_guard": True,
                "schema_version": "1.0",
                "structured_feedback": True,
            })

            manifest = build_execution_package(
                results, os.path.join(directory, "round.zip"), task_manifest,
                protocol, condition, "run-1", 0, {"qwen"}, {"task-1"},
            )
            extracted = os.path.join(directory, "package")
            with zipfile.ZipFile(os.path.join(directory, "round.zip")) as archive:
                archive.extractall(extracted)
            verified_manifest, verified_condition = verify_execution_package(extracted)
            self.assertEqual(manifest, verified_manifest)
            self.assertFalse(verified_condition["metadata_grounding"])

        self.assertEqual(1, manifest["entry_count"])
        self.assertIsNone(manifest["entries"][0]["grounding_sha256"])


if __name__ == "__main__":
    unittest.main()

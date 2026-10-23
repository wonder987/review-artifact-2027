import hashlib
import inspect
import json
import os
import sys
import tempfile
import unittest
import zipfile

from scripts.execute_generated_batch import (
    audit_program_source,
    execute_manifest,
)
from scripts.package_generation_batch import build_package, collect_entries


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXECUTOR = os.path.join(ROOT, "scripts", "execute_generated_batch.py")
SEMANTIC_EVALUATOR = os.path.join(ROOT, "scripts", "semantic_evaluator.py")
BATCH_EVALUATOR = os.path.join(ROOT, "scripts", "evaluate_generated_batch.py")


def sha256_text(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def write_completed_result(root, model, task_id, program):
    directory = os.path.join(root, model, task_id, "attempt-1")
    os.makedirs(directory)
    program_hash = sha256_text(program)
    with open(os.path.join(directory, "program.py"), "wb") as handle:
        handle.write(program.encode("utf-8"))
    result = {
        "status": "completed",
        "identity": {
            "attempt": 1,
            "model_alias": model,
            "prompt_hash": "prompt-" + task_id,
            "protocol_hash": "protocol-v1",
            "task_id": task_id,
        },
        "extraction": {
            "status": "success",
            "program_sha256": program_hash,
        },
    }
    with open(os.path.join(directory, "result.json"), "w") as handle:
        json.dump(result, handle)
    return directory


def write_generation_timeout(root, model, task_id):
    directory = os.path.join(root, model, task_id, "attempt-1")
    os.makedirs(directory)
    failure = {
        "status": "failed",
        "identity": {
            "attempt": 1,
            "model_alias": model,
            "prompt_hash": "prompt-" + task_id,
            "protocol_hash": "protocol-v1",
            "task_id": task_id,
        },
        "error_type": "ClientError",
        "error_message": (
            "generation request timed out after 600 seconds and was not "
            "retried because the provider may still have completed it"
        ),
    }
    with open(os.path.join(directory, "failure.json"), "w") as handle:
        json.dump(failure, handle)
    return directory


class PackageGenerationBatchTests(unittest.TestCase):
    def test_collects_only_allowed_complete_hash_verified_programs(self):
        with tempfile.TemporaryDirectory() as directory:
            write_completed_result(directory, "qwen", "task_b", "print('b')\n")
            write_completed_result(directory, "deepseek", "task_a", "print('a')\n")
            write_completed_result(directory, "excluded", "task_c", "print('c')\n")

            entries = collect_entries(directory, {"qwen", "deepseek"})

        self.assertEqual(
            [("deepseek", "task_a"), ("qwen", "task_b")],
            [(entry["model_alias"], entry["task_id"]) for entry in entries],
        )
        self.assertTrue(all(entry["program_sha256"] for entry in entries))

    def test_rejects_program_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            result_directory = write_completed_result(
                directory, "qwen", "task_a", "print('a')\n"
            )
            with open(os.path.join(result_directory, "program.py"), "wb") as handle:
                handle.write(b"print('changed')\n")

            with self.assertRaises(ValueError):
                collect_entries(directory, {"qwen"})

    def test_collects_only_requested_task_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            write_completed_result(directory, "qwen", "task_l1", "print('l1')\n")
            write_completed_result(directory, "qwen", "task_l2", "print('l2')\n")
            write_completed_result(directory, "qwen", "task_l3", "print('l3')\n")

            entries = collect_entries(
                directory,
                {"qwen"},
                allowed_task_ids={"task_l2", "task_l3"},
            )

        self.assertEqual(
            ["task_l2", "task_l3"],
            [entry["task_id"] for entry in entries],
        )

    def test_collects_terminal_generation_timeout_with_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            write_completed_result(directory, "qwen", "task_ok", "print('ok')\n")
            write_generation_timeout(directory, "qwen", "task_timeout")

            entries = collect_entries(directory, {"qwen"})
            timeout = next(
                entry for entry in entries if entry["task_id"] == "task_timeout"
            )

        self.assertEqual("generation_failure", timeout["entry_type"])
        self.assertEqual("generation_timeout", timeout["failure_code"])
        self.assertEqual("ClientError", timeout["error_type"])
        self.assertEqual(64, len(timeout["failure_sha256"]))
        self.assertEqual(
            "failures/qwen/task_timeout/failure.json",
            timeout["failure_path"],
        )

    def test_builds_deterministic_self_contained_execution_zip(self):
        with tempfile.TemporaryDirectory() as directory:
            results = os.path.join(directory, "results")
            os.makedirs(results)
            write_completed_result(results, "qwen", "task_a", "print('a')\n")
            first = os.path.join(directory, "first.zip")
            second = os.path.join(directory, "second.zip")

            first_manifest = build_package(results, first, EXECUTOR, {"qwen"})
            second_manifest = build_package(results, second, EXECUTOR, {"qwen"})

            with open(first, "rb") as handle:
                first_bytes = handle.read()
            with open(second, "rb") as handle:
                second_bytes = handle.read()
            with zipfile.ZipFile(first, "r") as archive:
                names = sorted(archive.namelist())

        self.assertEqual(first_manifest, second_manifest)
        self.assertEqual(first_bytes, second_bytes)
        self.assertEqual(
            [
                "execute_generated_batch.py",
                "manifest.json",
                "programs/qwen/task_a/program.py",
                "resource_aware_guard.py",
            ],
            names,
        )
        self.assertEqual(64, len(first_manifest["runtime_sha256"]["executor"]))
        self.assertEqual(
            64,
            len(first_manifest["runtime_sha256"]["resource_guard"]),
        )

    def test_scoped_package_rejects_missing_model_task_result(self):
        with tempfile.TemporaryDirectory() as directory:
            results = os.path.join(directory, "results")
            os.makedirs(results)
            write_completed_result(results, "qwen", "task_l2", "print('l2')\n")

            with self.assertRaises(ValueError):
                build_package(
                    results,
                    os.path.join(directory, "scoped.zip"),
                    EXECUTOR,
                    {"qwen"},
                    allowed_task_ids={"task_l2", "task_l3"},
                )

    def test_scoped_package_accepts_program_and_generation_timeout(self):
        with tempfile.TemporaryDirectory() as directory:
            results = os.path.join(directory, "results")
            os.makedirs(results)
            write_completed_result(results, "qwen", "task_ok", "print('ok')\n")
            write_generation_timeout(results, "qwen", "task_timeout")
            output = os.path.join(directory, "mixed.zip")

            manifest = build_package(
                results,
                output,
                EXECUTOR,
                {"qwen"},
                allowed_task_ids={"task_ok", "task_timeout"},
            )
            with zipfile.ZipFile(output, "r") as archive:
                names = sorted(archive.namelist())

        self.assertEqual(2, manifest["entry_count"])
        self.assertIn("failures/qwen/task_timeout/failure.json", names)

    def test_v2_package_contains_hashed_evaluator_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            results = os.path.join(directory, "results")
            os.makedirs(results)
            write_completed_result(results, "qwen", "task_a", "print('a')\n")
            output = os.path.join(directory, "v2.zip")

            manifest = build_package(
                results,
                output,
                EXECUTOR,
                {"qwen"},
                semantic_evaluator_path=SEMANTIC_EVALUATOR,
                batch_evaluator_path=BATCH_EVALUATOR,
                prompt_version="primary-zero-shot-v2",
            )
            with zipfile.ZipFile(output, "r") as archive:
                names = sorted(archive.namelist())

        self.assertIn("semantic_evaluator.py", names)
        self.assertIn("evaluate_generated_batch.py", names)
        self.assertEqual("primary-zero-shot-v2", manifest["prompt_version"])
        self.assertEqual("semantic-evaluator-v2", manifest["evaluator_version"])
        self.assertEqual(64, len(manifest["runtime_sha256"]["executor"]))
        self.assertEqual(64, len(manifest["runtime_sha256"]["resource_guard"]))
        self.assertEqual(64, len(manifest["runtime_sha256"]["semantic_evaluator"]))
        self.assertEqual(64, len(manifest["runtime_sha256"]["batch_evaluator"]))


class ExecuteGeneratedBatchTests(unittest.TestCase):
    def test_default_virtual_memory_limit_supports_mdsplus_runtime(self):
        parameter = inspect.signature(execute_manifest).parameters["max_memory_mib"]

        self.assertEqual(8192, parameter.default)

    def test_static_policy_rejects_process_and_network_escape(self):
        violations = audit_program_source(
            "import subprocess\nimport socket\nsubprocess.call(['whoami'])\n"
        )

        self.assertIn("forbidden_import:subprocess", violations)
        self.assertIn("forbidden_import:socket", violations)
        self.assertIn("forbidden_call:subprocess.call", violations)

    def test_accepts_case_insensitive_success_status_without_rewriting_output(self):
        with tempfile.TemporaryDirectory() as directory:
            program_path = os.path.join(
                directory, "programs", "gemini", "task_l2", "program.py"
            )
            os.makedirs(os.path.dirname(program_path))
            program = (
                "import json\n"
                "print(json.dumps({'status':'SUCCESS','signal_size':10001}))\n"
            )
            with open(program_path, "wb") as handle:
                handle.write(program.encode("utf-8"))
            manifest = {
                "schema_version": "1.0",
                "entries": [{
                    "model_alias": "gemini",
                    "task_id": "task_l2",
                    "program_path": "programs/gemini/task_l2/program.py",
                    "program_sha256": sha256_text(program),
                }],
            }
            manifest_path = os.path.join(directory, "manifest.json")
            with open(manifest_path, "w") as handle:
                json.dump(manifest, handle)

            summary = execute_manifest(
                manifest_path,
                os.path.join(directory, "execution"),
                timeout_seconds=5,
                python_executable=sys.executable,
            )
            result_path = os.path.join(
                directory,
                "execution",
                "gemini",
                "task_l2",
                "execution.json",
            )
            with open(result_path, "r") as handle:
                result = json.load(handle)

        self.assertEqual(1, summary["execution_success"])
        self.assertEqual("SUCCESS", result["reported_result"]["status"])

    def test_executes_success_and_failure_and_writes_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            programs = os.path.join(directory, "programs")
            success_path = os.path.join(programs, "qwen", "task_ok", "program.py")
            failure_path = os.path.join(programs, "qwen", "task_bad", "program.py")
            os.makedirs(os.path.dirname(success_path))
            os.makedirs(os.path.dirname(failure_path))
            success_program = (
                "import json\n"
                "print(json.dumps({'status':'success','signal_size':3}))\n"
            )
            failure_program = (
                "import json, sys\n"
                "print(json.dumps({'status':'failure','error_type':'Example'}))\n"
                "sys.exit(1)\n"
            )
            with open(success_path, "wb") as handle:
                handle.write(success_program.encode("utf-8"))
            with open(failure_path, "wb") as handle:
                handle.write(failure_program.encode("utf-8"))
            manifest = {
                "schema_version": "1.0",
                "entries": [
                    {
                        "model_alias": "qwen",
                        "task_id": "task_ok",
                        "program_path": "programs/qwen/task_ok/program.py",
                        "program_sha256": sha256_text(success_program),
                    },
                    {
                        "model_alias": "qwen",
                        "task_id": "task_bad",
                        "program_path": "programs/qwen/task_bad/program.py",
                        "program_sha256": sha256_text(failure_program),
                    },
                ],
            }
            manifest_path = os.path.join(directory, "manifest.json")
            with open(manifest_path, "w") as handle:
                json.dump(manifest, handle)
            output = os.path.join(directory, "execution")

            summary = execute_manifest(
                manifest_path,
                output,
                timeout_seconds=5,
                python_executable=sys.executable,
            )

            with open(os.path.join(output, "qwen", "task_ok", "execution.json")) as handle:
                success = json.load(handle)
            with open(os.path.join(output, "qwen", "task_bad", "execution.json")) as handle:
                failure = json.load(handle)

        self.assertEqual(2, summary["completed"])
        self.assertEqual(1, summary["execution_success"])
        self.assertEqual(1, summary["program_failure"])
        self.assertEqual("execution_success", success["status"])
        self.assertEqual("program_failure", failure["status"])
        self.assertEqual("success", success["reported_result"]["status"])

    def test_generation_failure_is_materialized_without_program_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            failure_path = os.path.join(
                directory, "failures", "deepseek", "task_timeout", "failure.json"
            )
            os.makedirs(os.path.dirname(failure_path))
            failure = {
                "status": "failed",
                "identity": {
                    "attempt": 1,
                    "model_alias": "deepseek",
                    "task_id": "task_timeout",
                },
                "error_type": "ClientError",
                "error_message": "generation request timed out after 600 seconds",
            }
            with open(failure_path, "w") as handle:
                json.dump(failure, handle)
            with open(failure_path, "rb") as handle:
                failure_hash = hashlib.sha256(handle.read()).hexdigest()
            manifest = {
                "schema_version": "2.1",
                "entries": [{
                    "entry_type": "generation_failure",
                    "model_alias": "deepseek",
                    "task_id": "task_timeout",
                    "failure_code": "generation_timeout",
                    "failure_path": "failures/deepseek/task_timeout/failure.json",
                    "failure_sha256": failure_hash,
                    "error_type": "ClientError",
                    "error_message": "generation request timed out after 600 seconds",
                }],
            }
            manifest_path = os.path.join(directory, "manifest.json")
            with open(manifest_path, "w") as handle:
                json.dump(manifest, handle)

            summary = execute_manifest(
                manifest_path,
                os.path.join(directory, "execution"),
                timeout_seconds=5,
                python_executable="definitely-not-a-python-executable",
            )
            with open(os.path.join(
                directory,
                "execution",
                "deepseek",
                "task_timeout",
                "execution.json",
            )) as handle:
                execution = json.load(handle)

        self.assertEqual(1, summary["generation_failure"])
        self.assertEqual("generation_failure", execution["status"])
        self.assertEqual("generation_timeout", execution["failure_code"])
        self.assertEqual("ClientError", execution["error_type"])

    def test_policy_rejection_never_executes_program(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "programs", "bad", "task", "program.py")
            os.makedirs(os.path.dirname(path))
            source = "import subprocess\nsubprocess.call(['not-real'])\n"
            with open(path, "wb") as handle:
                handle.write(source.encode("utf-8"))
            manifest = {
                "schema_version": "1.0",
                "entries": [{
                    "model_alias": "bad",
                    "task_id": "task",
                    "program_path": "programs/bad/task/program.py",
                    "program_sha256": sha256_text(source),
                }],
            }
            manifest_path = os.path.join(directory, "manifest.json")
            with open(manifest_path, "w") as handle:
                json.dump(manifest, handle)

            summary = execute_manifest(
                manifest_path,
                os.path.join(directory, "execution"),
                timeout_seconds=5,
                python_executable=sys.executable,
            )

        self.assertEqual(1, summary["policy_rejected"])
        self.assertEqual(0, summary["execution_success"])

    def test_task_aware_guard_blocks_oversized_unbounded_read(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "programs", "qwen", "task", "program.py")
            os.makedirs(os.path.dirname(path))
            source = (
                "c.get(r'\\\\BIG1').data()\n"
                "raise RuntimeError('must not execute')\n"
            )
            with open(path, "wb") as handle:
                handle.write(source.encode("utf-8"))
            manifest = {
                "schema_version": "3.0",
                "entries": [{
                    "model_alias": "qwen",
                    "task_id": "task",
                    "program_path": "programs/qwen/task/program.py",
                    "program_sha256": sha256_text(source),
                }],
            }
            task_manifest = {
                "tasks": [{
                    "id": "task",
                    "operation": {"kind": "exact_window"},
                    "safety_policy": {
                        "max_points": 5000000,
                        "max_total_mib": 256.0,
                        "timeout_seconds": 90,
                    },
                    "targets": [{
                        "estimated_total_mib": 600.0,
                        "node": "\\BIG1",
                        "sample_num": 50000000,
                        "signal_name": "BIG1",
                    }],
                }],
            }
            manifest_path = os.path.join(directory, "manifest.json")
            task_manifest_path = os.path.join(directory, "tasks.json")
            with open(manifest_path, "w") as handle:
                json.dump(manifest, handle)
            with open(task_manifest_path, "w") as handle:
                json.dump(task_manifest, handle)
            output = os.path.join(directory, "execution")
            guards = os.path.join(directory, "guard")

            summary = execute_manifest(
                manifest_path,
                output,
                timeout_seconds=5,
                python_executable=sys.executable,
                task_manifest_path=task_manifest_path,
                guard_output_root=guards,
            )
            with open(os.path.join(output, "qwen", "task", "execution.json")) as handle:
                execution = json.load(handle)
            with open(os.path.join(guards, "qwen", "task", "guard.json")) as handle:
                guard = json.load(handle)

        self.assertEqual(1, summary["policy_rejected"])
        self.assertEqual("require_bounded_access", guard["decision"])
        self.assertEqual("require_bounded_access", execution["guard_decision"])


if __name__ == "__main__":
    unittest.main()

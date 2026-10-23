import hashlib
import json
import os
import tempfile
import unittest
import zipfile

from scripts.repair_controller import (
    build_repair_prompt_record,
    import_diagnostics_zip,
    import_feedback,
    initialize_state,
    prepare_repair_prompts,
)


class RepairControllerTests(unittest.TestCase):
    def test_feedback_import_is_idempotent(self):
        state = initialize_state("run-1", "task-1", "qwen", "full", 2)
        feedback = {
            "identity": {
                "run_id": "run-1",
                "task_id": "task-1",
                "model_alias": "qwen",
                "condition": "full",
                "round": 0,
            },
            "summary": "Use the remote API.",
        }

        first = import_feedback(state, feedback)
        second = import_feedback(first, feedback)

        self.assertEqual(first, second)
        self.assertEqual(1, len(first["events"]))

    def test_repair_prompt_is_compatible_with_generation_runner(self):
        original = {
            "grounding_sha256": "g" * 64,
            "split": "pilot",
            "system_prompt": "Return Python only.",
            "task_id": "task-1",
            "user_prompt": "Retrieve the signal.",
        }
        feedback = {"summary": "Replace getNode.", "required_changes": ["Use get."]}

        record = build_repair_prompt_record(
            original,
            "print('old')\n",
            feedback,
            round_index=1,
            prompt_version="pascore-repair-v1-r1",
        )

        self.assertEqual(
            record["system_prompt"] + "\n\n" + record["user_prompt"],
            record["prompt"],
        )
        self.assertEqual(64, len(record["prompt_sha256"]))
        self.assertIn("print('old')", record["user_prompt"])
        self.assertEqual("g" * 64, record["grounding_sha256"])

    def test_imports_verified_diagnostics_and_prepares_per_model_prompt(self):
        with tempfile.TemporaryDirectory() as directory:
            feedback = {
                "identity": {
                    "run_id": "run-1",
                    "task_id": "task-1",
                    "model_alias": "qwen",
                    "condition": "full",
                    "round": 0,
                },
                "summary": "Replace getNode.",
                "required_changes": ["Use Connection.get."],
            }
            feedback_bytes = (
                json.dumps(feedback, indent=2, sort_keys=True) + "\n"
            ).encode("utf-8")
            feedback_path = "feedback/qwen/task-1/feedback.json"
            previous = os.path.join(directory, "previous")
            os.makedirs(previous)
            previous_manifest_path = os.path.join(previous, "manifest.json")
            with open(previous_manifest_path, "wb") as handle:
                handle.write(b'{"package_type":"pascore_execution_round"}\n')
            manifest = {
                "package_type": "pascore_diagnostics_round",
                "run_id": "run-1",
                "round": 0,
                "condition": "full",
                "source_manifest_sha256": hashlib.sha256(
                    b'{"package_type":"pascore_execution_round"}\n'
                ).hexdigest(),
                "entries": [{
                    "model_alias": "qwen",
                    "task_id": "task-1",
                    "files": {
                        "feedback": {
                            "path": feedback_path,
                            "sha256": hashlib.sha256(feedback_bytes).hexdigest(),
                        }
                    },
                }],
            }
            package = os.path.join(directory, "diagnostics.zip")
            with zipfile.ZipFile(package, "w") as archive:
                archive.writestr("manifest.json", json.dumps(manifest))
                archive.writestr(feedback_path, feedback_bytes)
            imported = os.path.join(directory, "imported")
            import_diagnostics_zip(package, imported)

            program_path = os.path.join(
                previous, "programs", "qwen", "task-1", "program.py"
            )
            os.makedirs(os.path.dirname(program_path))
            with open(program_path, "w") as handle:
                handle.write("print('old')\n")
            prompt = {
                "task_id": "task-1",
                "split": "pilot",
                "system_prompt": "Return Python only.",
                "user_prompt": "Retrieve signal.",
            }
            prompts_path = os.path.join(directory, "prompts.jsonl")
            with open(prompts_path, "w") as handle:
                handle.write(json.dumps(prompt) + "\n")
            condition_path = os.path.join(directory, "condition.json")
            with open(condition_path, "w") as handle:
                json.dump({
                    "condition_id": "full",
                    "failure_diagnosis": "multi_grained",
                    "max_repair_rounds": 2,
                    "metadata_grounding": True,
                    "resource_guard": True,
                    "schema_version": "1.0",
                    "structured_feedback": True,
                }, handle)
            base_protocol = os.path.join(directory, "base-protocol.json")
            with open(base_protocol, "w") as handle:
                json.dump({
                    "protocol_id": "base",
                    "prompt_version": "primary-zero-shot-v3",
                    "default_split": "pilot",
                    "held_out_enabled": False,
                    "temperature": 0,
                    "max_output_tokens": 8192,
                    "request_timeout_seconds": 600,
                    "max_network_retries": 3,
                    "retry_backoff_seconds": [2, 4, 8],
                    "generation_attempts": 1,
                    "repair_rounds": 0,
                }, handle)
            output = os.path.join(directory, "repair")
            state = os.path.join(directory, "state")

            summary = prepare_repair_prompts(
                imported,
                previous,
                prompts_path,
                condition_path,
                base_protocol,
                state,
                output,
                next_round=1,
            )
            with open(os.path.join(output, "qwen.jsonl")) as handle:
                record = json.loads(handle.readline())
            with open(os.path.join(output, "protocol.json")) as handle:
                protocol = json.load(handle)

        self.assertEqual(1, summary["record_count"])
        self.assertEqual("pascore-repair-v1-r1", record["prompt_version"])
        self.assertEqual("repair", protocol["generation_mode"])
        self.assertEqual(1, protocol["repair_rounds"])

    def test_generation_failure_uses_failure_artifact_as_repair_parent(self):
        with tempfile.TemporaryDirectory() as directory:
            feedback = {
                "identity": {
                    "run_id": "run-1",
                    "task_id": "task-1",
                    "model_alias": "deepseek",
                    "condition": "full",
                    "round": 0,
                },
                "summary": "Generate a replacement after generation timeout.",
                "required_changes": ["Use the verified access plan."],
            }
            feedback_bytes = (
                json.dumps(feedback, indent=2, sort_keys=True) + "\n"
            ).encode("utf-8")
            feedback_path = "feedback/deepseek/task-1/feedback.json"
            failure_hash = "f" * 64
            previous_manifest = {
                "package_type": "pascore_execution_round",
                "run_id": "run-1",
                "round": 0,
                "condition": "full",
                "entries": [{
                    "entry_type": "generation_failure",
                    "model_alias": "deepseek",
                    "task_id": "task-1",
                    "failure_sha256": failure_hash,
                    "pascore_identity": {"program_hash": failure_hash},
                }],
            }
            previous_bytes = (
                json.dumps(previous_manifest, indent=2, sort_keys=True) + "\n"
            ).encode("utf-8")
            previous = os.path.join(directory, "previous")
            os.makedirs(previous)
            with open(os.path.join(previous, "manifest.json"), "wb") as handle:
                handle.write(previous_bytes)
            diagnostics_manifest = {
                "package_type": "pascore_diagnostics_round",
                "run_id": "run-1",
                "round": 0,
                "condition": "full",
                "source_manifest_sha256": hashlib.sha256(
                    previous_bytes
                ).hexdigest(),
                "entries": [{
                    "model_alias": "deepseek",
                    "task_id": "task-1",
                    "files": {
                        "feedback": {
                            "path": feedback_path,
                            "sha256": hashlib.sha256(feedback_bytes).hexdigest(),
                        }
                    },
                }],
            }
            package = os.path.join(directory, "diagnostics.zip")
            with zipfile.ZipFile(package, "w") as archive:
                archive.writestr(
                    "manifest.json", json.dumps(diagnostics_manifest)
                )
                archive.writestr(feedback_path, feedback_bytes)
            imported = os.path.join(directory, "imported")
            import_diagnostics_zip(package, imported)
            prompts_path = os.path.join(directory, "prompts.jsonl")
            with open(prompts_path, "w") as handle:
                handle.write(json.dumps({
                    "task_id": "task-1",
                    "split": "held_out",
                    "system_prompt": "Return Python only.",
                    "user_prompt": "Retrieve signal.",
                }) + "\n")
            condition_path = os.path.join(directory, "condition.json")
            with open(condition_path, "w") as handle:
                json.dump({
                    "condition_id": "full",
                    "failure_diagnosis": "multi_grained",
                    "max_repair_rounds": 2,
                    "metadata_grounding": True,
                    "resource_guard": True,
                    "schema_version": "1.0",
                    "structured_feedback": True,
                }, handle)
            base_protocol = os.path.join(directory, "base-protocol.json")
            with open(base_protocol, "w") as handle:
                json.dump({
                    "protocol_id": "base",
                    "prompt_version": "pascore-grounded-v2",
                    "default_split": "pilot",
                    "held_out_enabled": True,
                    "temperature": 0,
                    "max_output_tokens": 8192,
                    "request_timeout_seconds": 600,
                    "max_network_retries": 3,
                    "retry_backoff_seconds": [2, 4, 8],
                    "generation_attempts": 1,
                    "repair_rounds": 0,
                }, handle)
            output = os.path.join(directory, "repair")

            summary = prepare_repair_prompts(
                imported,
                previous,
                prompts_path,
                condition_path,
                base_protocol,
                os.path.join(directory, "state"),
                output,
                next_round=1,
            )
            with open(os.path.join(output, "deepseek.jsonl")) as handle:
                record = json.loads(handle.readline())
            with open(os.path.join(output, "repair_plan.json")) as handle:
                plan = json.load(handle)

        self.assertEqual(1, summary["record_count"])
        self.assertIn("No previous program was produced", record["user_prompt"])
        self.assertEqual(failure_hash, record["pascore"]["parent_program_hash"])
        self.assertEqual(failure_hash, plan["entries"][0]["parent_program_hash"])


if __name__ == "__main__":
    unittest.main()

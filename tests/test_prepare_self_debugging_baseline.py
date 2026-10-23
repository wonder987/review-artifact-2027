from __future__ import print_function

import json
import os
import tempfile
import unittest

from scripts.prepare_self_debugging_baseline import build_explanation_inputs


def write_json(path, value):
    directory = os.path.dirname(path)
    if directory and not os.path.isdir(directory):
        os.makedirs(directory)
    with open(path, "w") as handle:
        json.dump(value, handle, sort_keys=True)
        handle.write("\n")


def write_text(path, value):
    directory = os.path.dirname(path)
    if directory and not os.path.isdir(directory):
        os.makedirs(directory)
    with open(path, "w") as handle:
        handle.write(value)


class SelfDebuggingExplanationInputTests(unittest.TestCase):
    def make_fixture(self, root):
        prompts_path = os.path.join(root, "prompts.jsonl")
        package_root = os.path.join(root, "package")
        diagnostics_root = os.path.join(root, "diagnostics")
        output_root = os.path.join(root, "output")
        records = [
            {
                "task_id": "task_success",
                "split": "held_out",
                "prompt_version": "grounded-v2",
                "grounding_sha256": "a" * 64,
                "system_prompt": "Generate Python only.",
                "user_prompt": "Successful grounded task.",
                "prompt": "Generate Python only.\n\nSuccessful grounded task.",
                "prompt_sha256": "b" * 64,
            },
            {
                "task_id": "task_failure",
                "split": "held_out",
                "prompt_version": "grounded-v2",
                "grounding_sha256": "c" * 64,
                "system_prompt": "Generate Python only.",
                "user_prompt": "Failed grounded task with access plan.",
                "prompt": "Generate Python only.\n\nFailed grounded task.",
                "prompt_sha256": "d" * 64,
            },
        ]
        write_text(
            prompts_path,
            "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        )
        manifest_entries = []
        for task_id in ("task_success", "task_failure"):
            program_path = os.path.join(
                package_root, "programs", "qwen3.8-chat", task_id, "program.py"
            )
            write_text(program_path, "print(%r)\n" % task_id)
            manifest_entries.append({
                "model_alias": "qwen3.8-chat",
                "task_id": task_id,
                "program_path": os.path.relpath(program_path, package_root).replace("\\", "/"),
            })
        write_json(os.path.join(package_root, "manifest.json"), {
            "entry_count": 2,
            "entries": manifest_entries,
        })
        for task_id, semantic_success in (
            ("task_success", True),
            ("task_failure", False),
        ):
            base = os.path.join(diagnostics_root, "evaluation", "qwen3.8-chat", task_id)
            write_json(os.path.join(base, "evaluation.json"), {
                "category": "semantic_failure",
                "code": "signal_dtype_mismatch",
                "diagnostics": [{"code": "secret", "path": "targets[0]"}],
                "mismatches": [{"code": "secret", "path": "targets[0]"}],
                "semantic_success": semantic_success,
                "task_id": task_id,
            })
            execution_base = os.path.join(
                diagnostics_root, "execution", "qwen3.8-chat", task_id
            )
            write_json(os.path.join(execution_base, "execution.json"), {
                "exit_status": 0,
                "policy_violations": [],
                "reported_result": {"status": "success", "sha256": "hidden"},
                "status": "execution_success",
                "timed_out": False,
            })
            log_base = os.path.join(diagnostics_root, "logs", "qwen3.8-chat", task_id)
            write_text(os.path.join(log_base, "stdout.log"), "raw stdout for %s" % task_id)
            write_text(os.path.join(log_base, "stderr.log"), "raw stderr for %s" % task_id)
        return prompts_path, package_root, diagnostics_root, output_root

    def test_builds_failed_only_leakage_controlled_prompt(self):
        with tempfile.TemporaryDirectory() as root:
            paths = self.make_fixture(root)
            summary = build_explanation_inputs(
                paths[0], paths[1], paths[2], paths[3],
                model_alias="qwen3.8-chat", expected_count=1,
            )
            self.assertEqual(1, summary["record_count"])
            with open(os.path.join(paths[3], "qwen3.8-chat.jsonl"), "r") as handle:
                record = json.loads(handle.readline())
            self.assertEqual("task_failure", record["task_id"])
            self.assertIn("raw stdout for task_failure", record["user_prompt"])
            self.assertIn("raw stderr for task_failure", record["user_prompt"])
            self.assertIn('"accepted": false', record["user_prompt"])
            self.assertIn("Failed grounded task with access plan.", record["user_prompt"])
            self.assertNotIn("signal_dtype_mismatch", record["prompt"])
            self.assertNotIn("targets[0]", record["prompt"])
            self.assertNotIn('"sha256": "hidden"', record["prompt"])
            self.assertIn("Do not return a corrected program", record["system_prompt"])

    def test_missing_log_fails_instead_of_changing_denominator(self):
        with tempfile.TemporaryDirectory() as root:
            paths = self.make_fixture(root)
            os.remove(os.path.join(
                paths[2], "logs", "qwen3.8-chat", "task_failure", "stderr.log"
            ))
            with self.assertRaisesRegex(ValueError, "missing stderr log"):
                build_explanation_inputs(
                    paths[0], paths[1], paths[2], paths[3],
                    model_alias="qwen3.8-chat", expected_count=1,
                )


if __name__ == "__main__":
    unittest.main()

from __future__ import print_function

import hashlib
import json
import os
import tempfile
import unittest

from scripts.prepare_self_debugging_repairs import build_repair_inputs


def write_json(path, value):
    directory = os.path.dirname(path)
    if directory and not os.path.isdir(directory):
        os.makedirs(directory)
    with open(path, "w") as handle:
        json.dump(value, handle, sort_keys=True)
        handle.write("\n")


def write_jsonl(path, records):
    directory = os.path.dirname(path)
    if directory and not os.path.isdir(directory):
        os.makedirs(directory)
    with open(path, "w") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


class SelfDebuggingRepairInputTests(unittest.TestCase):
    def make_fixture(self, root):
        prompt_hash = "a" * 64
        prompts_path = os.path.join(root, "explanation-input.jsonl")
        generation_root = os.path.join(root, "explanation-generation")
        output_root = os.path.join(root, "repair-input")
        explanation_request = (
            "Original grounded task:\nRetrieve the requested signal.\n\n"
            "Previous program:\n```python\nprint('old')\n```\n\n"
            "Raw execution observation:\n{\"status\": \"runtime_error\"}\n\n"
            "The candidate was rejected by the hidden acceptance test. The "
            "binary rejection does not reveal a semantic mismatch type, "
            "location, expected value, or expected hash. Explain what the "
            "program does, identify faults that can be inferred from the task, "
            "program, and raw observation, and propose a repair plan."
        )
        write_jsonl(prompts_path, [{
            "grounding_sha256": "b" * 64,
            "prompt": "system\n\n" + explanation_request,
            "prompt_sha256": prompt_hash,
            "prompt_version": "self-debugging-explanation-v1",
            "self_debugging": {
                "parent_program_hash": "c" * 64,
                "source_round": 0,
                "stage": "explanation",
            },
            "split": "held_out",
            "system_prompt": "explain",
            "task_id": "task_1",
            "user_prompt": explanation_request,
        }])
        analysis = {
            "explanation": "The old program prints a placeholder.",
            "suspected_faults": ["It does not access the requested signal."],
            "repair_plan": ["Use the grounded access plan."],
        }
        content = json.dumps(analysis, sort_keys=True)
        write_json(
            os.path.join(
                generation_root,
                "qwen3.8-chat",
                "task_1",
                "attempt-1",
                "result.json",
            ),
            {
                "status": "completed",
                "identity": {
                    "attempt": 1,
                    "model_alias": "qwen3.8-chat",
                    "prompt_hash": prompt_hash,
                    "task_id": "task_1",
                },
                "extraction": {"status": "success"},
                "normalized": {"content": content},
            },
        )
        return prompts_path, generation_root, output_root

    def test_builds_repair_prompt_from_visible_inputs_and_analysis(self):
        with tempfile.TemporaryDirectory() as root:
            prompts_path, generation_root, output_root = self.make_fixture(root)
            summary = build_repair_inputs(
                prompts_path,
                generation_root,
                output_root,
                model_alias="qwen3.8-chat",
                expected_count=1,
            )
            self.assertEqual(1, summary["record_count"])
            with open(os.path.join(output_root, "qwen3.8-chat.jsonl"), "r") as handle:
                record = json.loads(handle.readline())
            self.assertEqual("task_1", record["task_id"])
            self.assertIn("Original grounded task", record["user_prompt"])
            self.assertIn("The old program prints a placeholder", record["user_prompt"])
            self.assertIn("complete corrected Python 3.6 program", record["system_prompt"])
            self.assertNotIn("Explain what the program does", record["user_prompt"])
            self.assertNotIn("semantic mismatch path", record["prompt"])
            self.assertEqual("repair", record["self_debugging"]["stage"])
            self.assertEqual(
                hashlib.sha256(record["prompt"].encode("utf-8")).hexdigest(),
                record["prompt_sha256"],
            )
            with open(os.path.join(output_root, "protocol.json"), "r") as handle:
                protocol = json.load(handle)
            self.assertEqual("self-debugging-repair-v1", protocol["prompt_version"])
            self.assertEqual(8192, protocol["max_output_tokens"])

    def test_rejects_task_identity_mismatch(self):
        with tempfile.TemporaryDirectory() as root:
            prompts_path, generation_root, output_root = self.make_fixture(root)
            result_path = os.path.join(
                generation_root,
                "qwen3.8-chat",
                "task_1",
                "attempt-1",
                "result.json",
            )
            with open(result_path, "r") as handle:
                result = json.load(handle)
            result["identity"]["task_id"] = "different_task"
            write_json(result_path, result)
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                build_repair_inputs(
                    prompts_path,
                    generation_root,
                    output_root,
                    model_alias="qwen3.8-chat",
                    expected_count=1,
                )

    def test_accepts_single_json_fence(self):
        with tempfile.TemporaryDirectory() as root:
            prompts_path, generation_root, output_root = self.make_fixture(root)
            result_path = os.path.join(
                generation_root,
                "qwen3.8-chat",
                "task_1",
                "attempt-1",
                "result.json",
            )
            with open(result_path, "r") as handle:
                result = json.load(handle)
            content = result["normalized"]["content"]
            result["normalized"]["content"] = "```json\n{}\n```".format(content)
            write_json(result_path, result)
            summary = build_repair_inputs(
                prompts_path,
                generation_root,
                output_root,
                model_alias="qwen3.8-chat",
                expected_count=1,
            )
            self.assertEqual(1, summary["record_count"])

    def test_rejects_malformed_explanation(self):
        with tempfile.TemporaryDirectory() as root:
            prompts_path, generation_root, output_root = self.make_fixture(root)
            result_path = os.path.join(
                generation_root,
                "qwen3.8-chat",
                "task_1",
                "attempt-1",
                "result.json",
            )
            with open(result_path, "r") as handle:
                result = json.load(handle)
            result["normalized"]["content"] = '{"explanation": "missing fields"}'
            write_json(result_path, result)
            with self.assertRaisesRegex(ValueError, "invalid explanation schema"):
                build_repair_inputs(
                    prompts_path,
                    generation_root,
                    output_root,
                    model_alias="qwen3.8-chat",
                    expected_count=1,
                )


if __name__ == "__main__":
    unittest.main()

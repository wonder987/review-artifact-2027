import hashlib
import json
import os
import tempfile
import unittest

from scripts.llm_generation_store import (
    StoreError,
    extract_program,
    is_complete,
    result_directory,
    result_key,
    write_result,
)


IDENTITY = {
    "attempt": 1,
    "model_alias": "qwen3.8-reasoner",
    "prompt_hash": "a" * 64,
    "protocol_hash": "b" * 64,
    "task_id": "gold138_l1_025",
}


class LlmGenerationStoreTests(unittest.TestCase):
    def test_extracts_plain_python_without_semantic_rewrite(self):
        content = "import json\nprint(json.dumps({'status': 'success'}))\n"
        result = extract_program(content)

        self.assertEqual("success", result["status"])
        self.assertEqual("plain", result["strategy"])
        self.assertEqual(content.strip(), result["program"])
        self.assertEqual(
            hashlib.sha256(content.strip().encode("utf-8")).hexdigest(),
            result["program_sha256"],
        )

    def test_extracts_one_python_or_unlabeled_fence(self):
        for content in (
            "Explanation\n```python\nprint('ok')\n```\n",
            "```\nprint('ok')\n```",
        ):
            result = extract_program(content)
            self.assertEqual("success", result["status"])
            self.assertEqual("single_fence", result["strategy"])
            self.assertEqual("print('ok')", result["program"])

    def test_rejects_multiple_fences_and_empty_content(self):
        cases = (
            "```python\nprint(1)\n```\n```python\nprint(2)\n```",
            "",
        )
        for content in cases:
            result = extract_program(content)
            self.assertEqual("error", result["status"])
            self.assertIsNone(result["program"])
            self.assertIsNone(result["program_sha256"])

    def test_preserves_syntax_invalid_output_for_execution_classification(self):
        content = "value = f'{path.lstrip(\"\\\\\")}'"

        result = extract_program(content)

        self.assertEqual("success", result["status"])
        self.assertEqual(content, result["program"])
        self.assertFalse(result["syntax_valid"])

    def test_result_key_is_deterministic_and_changes_with_attempt(self):
        first = result_key(**IDENTITY)
        second = result_key(**IDENTITY)
        changed = dict(IDENTITY)
        changed["attempt"] = 2

        self.assertEqual(first, second)
        self.assertNotEqual(first, result_key(**changed))

    def test_grounding_hash_is_bound_into_result_key_and_resume_identity(self):
        grounded_identity = dict(IDENTITY)
        grounded_identity["grounding_hash"] = "c" * 64
        changed_grounding = dict(grounded_identity)
        changed_grounding["grounding_hash"] = "d" * 64
        extraction = extract_program("print('ok')")

        self.assertNotEqual(
            result_key(**grounded_identity),
            result_key(**changed_grounding),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = write_result(
                directory,
                grounded_identity,
                {"model": "qwen3.8-reasoner", "messages": []},
                {"choices": []},
                {"content": "print('ok')", "usage": {}},
                extraction,
                {"attempts": 1, "http_status": 200},
            )

            self.assertTrue(is_complete(path, grounded_identity))
            self.assertFalse(is_complete(path, changed_grounding))

    def test_write_result_is_atomic_complete_and_resumable(self):
        extraction = extract_program("print('ok')")
        with tempfile.TemporaryDirectory() as directory:
            path = write_result(
                directory,
                IDENTITY,
                {"model": "qwen3.8-reasoner", "messages": []},
                {"choices": []},
                {"content": "print('ok')", "usage": {}},
                extraction,
                {"attempts": 1, "http_status": 200},
            )

            self.assertTrue(is_complete(path, IDENTITY))
            self.assertTrue(os.path.isfile(os.path.join(path, "request.json")))
            self.assertTrue(os.path.isfile(os.path.join(path, "response.raw.json")))
            self.assertTrue(os.path.isfile(os.path.join(path, "result.json")))
            self.assertTrue(os.path.isfile(os.path.join(path, "program.py")))
            self.assertFalse(any(name.endswith(".tmp") for name in os.listdir(path)))
            with open(os.path.join(path, "program.py"), "rb") as handle:
                program_bytes = handle.read()
            self.assertEqual(b"print('ok')", program_bytes)
            self.assertEqual(
                extraction["program_sha256"],
                hashlib.sha256(program_bytes).hexdigest(),
            )

    def test_partial_directory_is_not_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            path = result_directory(directory, IDENTITY)
            os.makedirs(path)
            with open(os.path.join(path, "request.json"), "w") as handle:
                json.dump({}, handle)

            self.assertFalse(is_complete(path, IDENTITY))

    def test_successful_retry_removes_stale_failure_marker(self):
        extraction = extract_program("print('ok')")
        with tempfile.TemporaryDirectory() as directory:
            path = result_directory(directory, IDENTITY)
            os.makedirs(path)
            with open(os.path.join(path, "failure.json"), "w") as handle:
                json.dump({"status": "failed"}, handle)

            write_result(
                directory,
                IDENTITY,
                {"model": "qwen3.8-reasoner", "messages": []},
                {"choices": []},
                {"content": "print('ok')", "usage": {}},
                extraction,
                {"attempts": 1, "http_status": 200},
            )

            self.assertFalse(os.path.exists(os.path.join(path, "failure.json")))

    def test_completed_extraction_error_is_terminal_for_resume(self):
        extraction = extract_program("")
        with tempfile.TemporaryDirectory() as directory:
            path = write_result(
                directory,
                IDENTITY,
                {"model": "qwen3.8-reasoner", "messages": []},
                {"choices": []},
                {"content": "not executable prose", "usage": {}},
                extraction,
                {"attempts": 1, "http_status": 200},
            )

            self.assertTrue(is_complete(path, IDENTITY))
            self.assertFalse(os.path.exists(os.path.join(path, "program.py")))

    def test_store_rejects_authorization_or_literal_key_material(self):
        extraction = extract_program("print('ok')")
        unsafe_snapshots = (
            {"Authorization": "Bearer secret"},
            {"api_key": "sk-secret"},
            {"nested": {"header": "Bearer secret"}},
        )
        with tempfile.TemporaryDirectory() as directory:
            for snapshot in unsafe_snapshots:
                with self.assertRaises(StoreError):
                    write_result(
                        directory,
                        IDENTITY,
                        snapshot,
                        {},
                        {"content": "print('ok')"},
                        extraction,
                        {"attempts": 1},
                    )


if __name__ == "__main__":
    unittest.main()

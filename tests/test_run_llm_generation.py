import copy
import json
import os
import tempfile
import unittest

from scripts.llm_runner_config import load_models, load_protocol
from scripts.run_llm_generation import (
    RunnerError,
    build_plan,
    load_prompt_records,
    run_generation,
    select_records,
)


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROMPTS = os.path.join(ROOT, "benchmark", "gold138_primary_zero_shot.jsonl")
MODELS = os.path.join(ROOT, "experiments", "config", "models.json")
PROTOCOL = os.path.join(ROOT, "experiments", "config", "protocol_v1.json")


def fake_document(model):
    return {
        "id": "fake-response",
        "model": model,
        "choices": [{
            "finish_reason": "stop",
            "message": {
                "role": "assistant",
                "content": "print('ok')"
            }
        }],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15
        }
    }


class LlmGenerationRunnerTests(unittest.TestCase):
    def test_selects_exact_pilot_records_in_manifest_order(self):
        records = load_prompt_records(PROMPTS)
        protocol = load_protocol(PROTOCOL)
        selected = select_records(records, "pilot", protocol)

        self.assertEqual(30, len(selected))
        self.assertTrue(all(record["split"] == "pilot" for record in selected))
        self.assertEqual(
            [record["task_id"] for record in records if record["split"] == "pilot"],
            [record["task_id"] for record in selected],
        )

    def test_rejects_held_out_and_prompt_hash_mismatch(self):
        records = load_prompt_records(PROMPTS)
        protocol = load_protocol(PROTOCOL)
        with self.assertRaises(RunnerError):
            select_records(records, "held_out", protocol)

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "bad.jsonl")
            bad = copy.deepcopy(records[0])
            bad["user_prompt"] += " changed"
            with open(path, "w") as handle:
                handle.write(json.dumps(bad) + "\n")
            with self.assertRaises(RunnerError):
                load_prompt_records(path)

    def test_held_out_plan_rejects_model_outside_frozen_roster(self):
        records = load_prompt_records(PROMPTS)
        protocol = load_protocol(PROTOCOL)
        protocol.update({
            "held_out_enabled": True,
            "frozen": True,
            "frozen_model_aliases": ["qwen3.8-chat"],
            "pilot_gate_status": "passed",
        })
        models = load_models(MODELS)

        with self.assertRaises(RunnerError):
            build_plan(
                records,
                models,
                protocol,
                "results",
                "held_out",
                model_aliases=["gpt-5.6-sol"],
            )

    def test_limit_one_builds_exactly_four_deterministic_calls(self):
        records = load_prompt_records(PROMPTS)
        models = load_models(MODELS)
        protocol = load_protocol(PROTOCOL)
        with tempfile.TemporaryDirectory() as directory:
            first = build_plan(records, models, protocol, directory, "pilot", limit=1)
            second = build_plan(records, models, protocol, directory, "pilot", limit=1)

        self.assertEqual(4, len(first))
        self.assertEqual(first, second)
        self.assertEqual(1, len(set(item["task_id"] for item in first)))
        self.assertEqual(set(models), set(item["model_alias"] for item in first))
        for item in first:
            self.assertEqual(2, len(item["payload"]["messages"]))
            serialized = json.dumps(item["payload"])
            self.assertNotIn("subtype", serialized)
            self.assertNotIn("split", serialized)
            self.assertNotIn("level", serialized)

    def test_model_filter_is_explicit_and_rejects_unknown_alias(self):
        records = load_prompt_records(PROMPTS)
        models = load_models(MODELS)
        protocol = load_protocol(PROTOCOL)
        with tempfile.TemporaryDirectory() as directory:
            plan = build_plan(
                records,
                models,
                protocol,
                directory,
                "pilot",
                model_aliases=["qwen3.8-chat", "gemini-3.5-flash-lite"],
                limit=1,
            )
            self.assertEqual(
                ["qwen3.8-chat", "gemini-3.5-flash-lite"],
                [item["model_alias"] for item in plan],
            )
            with self.assertRaises(RunnerError):
                build_plan(
                    records,
                    models,
                    protocol,
                    directory,
                    "pilot",
                    model_aliases=["not-a-model"],
                    limit=1,
                )

    def test_task_filter_selects_only_requested_pilot_tasks_in_requested_order(self):
        records = load_prompt_records(PROMPTS)
        models = load_models(MODELS)
        protocol = load_protocol(PROTOCOL)
        task_ids = ["gold138_l3_037", "gold138_l2_014"]

        with tempfile.TemporaryDirectory() as directory:
            plan = build_plan(
                records,
                models,
                protocol,
                directory,
                "pilot",
                model_aliases=["qwen3.8-chat"],
                task_ids=task_ids,
            )

        self.assertEqual(task_ids, [item["task_id"] for item in plan])

    def test_plan_binds_grounding_hash_into_generation_identity(self):
        records = load_prompt_records(PROMPTS)
        record = dict(next(item for item in records if item["split"] == "pilot"))
        record["grounding_sha256"] = "g" * 64
        models = load_models(MODELS)
        protocol = load_protocol(PROTOCOL)

        with tempfile.TemporaryDirectory() as directory:
            plan = build_plan(
                [record],
                models,
                protocol,
                directory,
                record["split"],
                model_aliases=["qwen3.8-chat"],
            )

        self.assertEqual("g" * 64, plan[0]["identity"]["grounding_hash"])
        self.assertEqual(
            "g" * 64, plan[0]["request_snapshot"]["grounding_sha256"]
        )

    def test_task_filter_rejects_unknown_duplicate_and_wrong_split_ids(self):
        records = load_prompt_records(PROMPTS)
        models = load_models(MODELS)
        protocol = load_protocol(PROTOCOL)

        with tempfile.TemporaryDirectory() as directory:
            for task_ids in (
                ["not-a-task"],
                ["gold138_l2_014", "gold138_l2_014"],
                ["gold138_l2_001"],
            ):
                with self.assertRaises(RunnerError):
                    build_plan(
                        records,
                        models,
                        protocol,
                        directory,
                        "pilot",
                        model_aliases=["qwen3.8-chat"],
                        task_ids=task_ids,
                    )

    def test_fake_client_writes_results_then_resume_skips_all_calls(self):
        calls = []

        def fake_client(model_config, protocol, payload, api_key):
            calls.append((model_config, payload, api_key))
            return fake_document(model_config["model"]), {
                "attempts": 1,
                "elapsed_seconds": 0.01,
                "http_status": 200,
                "provider": model_config["provider"],
            }

        records = load_prompt_records(PROMPTS)
        models = load_models(MODELS)
        protocol = load_protocol(PROTOCOL)
        environment = {
            "GEMINI_API_KEY": "gemini-test-key",
            "GATEWAY_API_KEY": "gateway-test-key",
            "OPENAI_API_KEY": "openai-test-key",
        }
        with tempfile.TemporaryDirectory() as directory:
            first = run_generation(
                records,
                models,
                protocol,
                directory,
                split="pilot",
                limit=1,
                environ=environment,
                client_fn=fake_client,
            )
            first_call_count = len(calls)
            second = run_generation(
                records,
                models,
                protocol,
                directory,
                split="pilot",
                limit=1,
                environ=environment,
                client_fn=fake_client,
            )

            result_files = []
            for root, _, files in os.walk(directory):
                if "result.json" in files:
                    result_files.append(os.path.join(root, "result.json"))
            for path in result_files:
                with open(path, "r") as handle:
                    stored = json.load(handle)
                request_path = os.path.join(os.path.dirname(path), "request.json")
                with open(request_path, "r") as handle:
                    request_snapshot = json.load(handle)
                serialized = json.dumps(request_snapshot)
                self.assertNotIn("test-key", serialized)
                self.assertNotIn("level", serialized)
                self.assertNotIn("subtype", serialized)
                self.assertEqual("completed", stored["status"])

        self.assertEqual(4, first_call_count)
        self.assertEqual(4, first["completed"])
        self.assertEqual(0, first["failed"])
        self.assertEqual(4, second["skipped"])
        self.assertEqual(first_call_count, len(calls))
        self.assertEqual(4, len(result_files))

    def test_dry_run_never_calls_client_and_reports_missing_credentials(self):
        calls = []

        def forbidden_client(*args, **kwargs):
            calls.append((args, kwargs))
            raise AssertionError("dry-run must not call client")

        with tempfile.TemporaryDirectory() as directory:
            summary = run_generation(
                load_prompt_records(PROMPTS),
                load_models(MODELS),
                load_protocol(PROTOCOL),
                directory,
                split="pilot",
                limit=1,
                dry_run=True,
                environ={},
                client_fn=forbidden_client,
            )

        self.assertEqual([], calls)
        self.assertEqual(4, summary["planned"])
        self.assertEqual(
            ["GATEWAY_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY"],
            summary["missing_credentials"],
        )

    def test_live_run_reports_start_and_completion_progress(self):
        events = []

        def fake_client(model_config, protocol, payload, api_key):
            return fake_document(model_config["model"]), {
                "attempts": 1,
                "elapsed_seconds": 0.1,
                "http_status": 200,
                "provider": model_config["provider"],
                "transport": "fake",
            }

        with tempfile.TemporaryDirectory() as directory:
            summary = run_generation(
                load_prompt_records(PROMPTS),
                load_models(MODELS),
                load_protocol(PROTOCOL),
                directory,
                split="pilot",
                model_aliases=["qwen3.8-chat"],
                limit=1,
                environ={"GATEWAY_API_KEY": "secret-value"},
                client_fn=fake_client,
                progress_fn=events.append,
            )

        self.assertEqual(1, summary["completed"])
        self.assertEqual("request_start", events[0]["event"])
        self.assertEqual("request_complete", events[1]["event"])
        self.assertEqual("qwen3.8-chat", events[0]["model_alias"])

    def test_normalization_failure_preserves_raw_response_and_request_snapshot(self):
        raw = fake_document("qwen3.8-chat")
        raw["choices"][0]["finish_reason"] = "length"
        raw["choices"][0]["message"]["content"] = None

        def fake_client(model_config, protocol, payload, api_key):
            return raw, {
                "attempts": 1,
                "elapsed_seconds": 0.1,
                "http_status": 200,
                "provider": model_config["provider"],
                "transport": "fake",
            }

        with tempfile.TemporaryDirectory() as directory:
            summary = run_generation(
                load_prompt_records(PROMPTS),
                load_models(MODELS),
                load_protocol(PROTOCOL),
                directory,
                split="pilot",
                model_aliases=["qwen3.8-chat"],
                limit=1,
                environ={"GATEWAY_API_KEY": "secret-value"},
                client_fn=fake_client,
            )
            result_directories = []
            for root, _, files in os.walk(directory):
                if "failure.json" in files:
                    result_directories.append(root)

            self.assertEqual(1, len(result_directories))
            result_directory = result_directories[0]
            with open(os.path.join(result_directory, "response.raw.json"), "r") as handle:
                stored_raw = json.load(handle)
            with open(os.path.join(result_directory, "request.json"), "r") as handle:
                stored_request = json.load(handle)

        self.assertEqual(1, summary["failed"])
        self.assertEqual(raw, stored_raw)
        self.assertNotIn("secret-value", json.dumps(stored_request))


if __name__ == "__main__":
    unittest.main()

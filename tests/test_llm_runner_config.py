import json
import os
import tempfile
import unittest

from scripts.llm_runner_config import (
    ConfigError,
    load_models,
    load_protocol,
    protocol_sha256,
)


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, "experiments", "config", "models.json")
PROTOCOL = os.path.join(ROOT, "experiments", "config", "protocol_v1.json")
HELD_OUT_PROTOCOL = os.path.join(
    ROOT, "experiments", "config", "protocol_v2_heldout.json"
)
HELD_OUT_MODELS = os.path.join(
    ROOT, "experiments", "config", "models_direct_zero_shot_v2.json"
)
PASCORE_HELD_OUT_PROTOCOL = os.path.join(
    ROOT, "experiments", "config", "pascore_initial_protocol_v2_heldout.json"
)

EXPECTED_MODELS = {
    "qwen3.8-chat",
    "deepseek-v4-flash-ascend1",
    "gemini-3.5-flash-lite",
    "gpt-5.6-sol",
}


class LlmRunnerConfigTests(unittest.TestCase):
    def test_real_model_config_has_four_exact_aliases_and_no_secrets(self):
        models = load_models(MODELS)

        self.assertEqual(EXPECTED_MODELS, set(models))
        self.assertEqual("GATEWAY_API_KEY", models["qwen3.8-chat"]["api_key_env"])
        self.assertEqual(
            "GEMINI_API_KEY",
            models["gemini-3.5-flash-lite"]["api_key_env"],
        )
        self.assertEqual("OPENAI_API_KEY", models["gpt-5.6-sol"]["api_key_env"])
        self.assertEqual({"curl"}, {config["transport"] for config in models.values()})
        with open(MODELS, "r") as handle:
            raw = handle.read().lower()
        with open(MODELS, "r") as handle:
            document = json.load(handle)
        self.assertNotIn("bearer ", raw)
        self.assertNotIn("sk-", raw)
        for config in document["models"].values():
            self.assertNotIn("api_key", config)

    def test_real_protocol_fixes_pilot_budget(self):
        protocol = load_protocol(PROTOCOL)

        self.assertEqual("primary-zero-shot-v1", protocol["prompt_version"])
        self.assertEqual("pilot", protocol["default_split"])
        self.assertFalse(protocol["held_out_enabled"])
        self.assertEqual(8192, protocol["max_output_tokens"])
        self.assertEqual(600, protocol["request_timeout_seconds"])
        self.assertEqual(3, protocol["max_network_retries"])
        self.assertEqual(0, protocol["temperature"])
        self.assertEqual(1, protocol["generation_attempts"])
        self.assertEqual(0, protocol["repair_rounds"])
        self.assertEqual(64, len(protocol_sha256(protocol)))

    def test_frozen_held_out_configuration_has_three_exact_models(self):
        models = load_models(HELD_OUT_MODELS)
        protocol = load_protocol(HELD_OUT_PROTOCOL)

        expected = {
            "qwen3.8-chat",
            "deepseek-v4-flash-ascend1",
            "gemini-3.5-flash-lite",
        }
        self.assertEqual(expected, set(models))
        self.assertTrue(protocol["held_out_enabled"])
        self.assertTrue(protocol["frozen"])
        self.assertEqual(expected, set(protocol["frozen_model_aliases"]))
        self.assertEqual("passed", protocol["pilot_gate_status"])

    def test_held_out_protocol_requires_explicit_freeze_metadata(self):
        with open(PROTOCOL, "r") as handle:
            protocol = json.load(handle)
        protocol["held_out_enabled"] = True

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "protocol.json")
            with open(path, "w") as handle:
                json.dump(protocol, handle)
            with self.assertRaises(ConfigError):
                load_protocol(path)

    def test_frozen_pascore_held_out_protocol_preserves_pilot_budget(self):
        protocol = load_protocol(PASCORE_HELD_OUT_PROTOCOL)

        expected = {
            "qwen3.8-chat",
            "deepseek-v4-flash-ascend1",
            "gemini-3.5-flash-lite",
        }
        self.assertTrue(protocol["held_out_enabled"])
        self.assertTrue(protocol["frozen"])
        self.assertEqual("passed", protocol["pilot_gate_status"])
        self.assertEqual(expected, set(protocol["frozen_model_aliases"]))
        self.assertEqual("pascore-grounded-v2", protocol["prompt_version"])
        self.assertEqual("direct_zero_shot", protocol["generation_mode"])
        self.assertEqual(0, protocol["repair_rounds"])
        self.assertEqual(8192, protocol["max_output_tokens"])
        self.assertEqual(600, protocol["request_timeout_seconds"])
        self.assertEqual(0, protocol["temperature"])

    def test_loader_rejects_literal_key_and_non_https_endpoint(self):
        invalid = {
            "schema_version": "1.0",
            "models": {
                "bad": {
                    "provider": "gateway",
                    "base_url": "http://example.invalid/v1",
                    "model": "bad",
                    "api_key_env": "GATEWAY_API_KEY",
                    "api_key": "sk-secret",
                    "max_tokens_parameter": "max_tokens",
                    "supports_temperature": True,
                    "transport": "curl"
                }
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "models.json")
            with open(path, "w") as handle:
                json.dump(invalid, handle)
            with self.assertRaises(ConfigError):
                load_models(path)

    def test_loader_rejects_unknown_provider_and_missing_key_environment(self):
        invalid = {
            "schema_version": "1.0",
            "models": {
                "bad": {
                    "provider": "unknown",
                    "base_url": "https://example.invalid/v1",
                    "model": "bad",
                    "api_key_env": "",
                    "max_tokens_parameter": "max_tokens",
                    "supports_temperature": True,
                    "transport": "invalid"
                }
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "models.json")
            with open(path, "w") as handle:
                json.dump(invalid, handle)
            with self.assertRaises(ConfigError):
                load_models(path)

    def test_protocol_hash_is_independent_of_json_key_order(self):
        first = {"b": 2, "a": 1}
        second = {"a": 1, "b": 2}
        self.assertEqual(protocol_sha256(first), protocol_sha256(second))

    def test_protocol_loader_rejects_unsafe_or_nonpositive_values(self):
        invalid = {
            "schema_version": "1.0",
            "protocol_id": "bad",
            "prompt_version": "primary-zero-shot-v1",
            "default_split": "held_out",
            "held_out_enabled": True,
            "temperature": 0,
            "max_output_tokens": 0,
            "request_timeout_seconds": -1,
            "max_network_retries": -1,
            "retry_backoff_seconds": [],
            "generation_attempts": 1,
            "repair_rounds": 0
        }
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "protocol.json")
            with open(path, "w") as handle:
                json.dump(invalid, handle)
            with self.assertRaises(ConfigError):
                load_protocol(path)

    def test_protocol_loader_accepts_bounded_pascore_repair_round(self):
        with open(PROTOCOL, "r") as handle:
            protocol = json.load(handle)
        protocol.update({
            "generation_mode": "repair",
            "prompt_version": "pascore-repair-v1-r1",
            "protocol_id": "pascore-repair-v1-r1",
            "repair_rounds": 1,
        })
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "protocol.json")
            with open(path, "w") as handle:
                json.dump(protocol, handle)

            loaded = load_protocol(path)

        self.assertEqual("repair", loaded["generation_mode"])
        self.assertEqual(1, loaded["repair_rounds"])


if __name__ == "__main__":
    unittest.main()

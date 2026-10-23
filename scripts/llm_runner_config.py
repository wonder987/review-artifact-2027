#!/usr/bin/env python3
"""Load and validate local LLM experiment configuration."""

from __future__ import print_function

import hashlib
import json


class ConfigError(ValueError):
    pass


def _load_object(path):
    with open(path, "r") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ConfigError("configuration root must be an object: {}".format(path))
    return value


def load_models(path):
    document = _load_object(path)
    models = document.get("models")
    if not isinstance(models, dict) or not models:
        raise ConfigError("models must be a non-empty object")

    validated = {}
    required = (
        "provider",
        "base_url",
        "model",
        "api_key_env",
        "max_tokens_parameter",
        "supports_temperature",
        "transport",
    )
    for alias, config in models.items():
        if not isinstance(config, dict):
            raise ConfigError("model {} configuration must be an object".format(alias))
        forbidden = set(config).intersection((
            "api_key",
            "authorization",
            "bearer",
            "secret",
        ))
        if forbidden:
            raise ConfigError("model {} contains a literal secret field".format(alias))
        missing = [field for field in required if field not in config]
        if missing:
            raise ConfigError(
                "model {} missing fields: {}".format(alias, ", ".join(missing))
            )
        if config["provider"] not in ("google", "gateway", "openai"):
            raise ConfigError("model {} has unknown provider".format(alias))
        if not str(config["base_url"]).startswith("https://"):
            raise ConfigError("model {} base URL must use HTTPS".format(alias))
        if not config["model"] or not config["api_key_env"]:
            raise ConfigError("model {} has an empty model or key environment".format(alias))
        if config["max_tokens_parameter"] not in (
            "max_tokens",
            "max_completion_tokens",
        ):
            raise ConfigError("model {} has unsupported token parameter".format(alias))
        if not isinstance(config["supports_temperature"], bool):
            raise ConfigError("model {} has invalid temperature capability".format(alias))
        if config["transport"] not in ("curl", "urllib"):
            raise ConfigError("model {} has unsupported transport".format(alias))
        validated[alias] = dict(config)
    return validated


def load_protocol(path):
    protocol = _load_object(path)
    required = (
        "protocol_id",
        "prompt_version",
        "default_split",
        "held_out_enabled",
        "temperature",
        "max_output_tokens",
        "request_timeout_seconds",
        "max_network_retries",
        "retry_backoff_seconds",
        "generation_attempts",
        "repair_rounds",
    )
    missing = [field for field in required if field not in protocol]
    if missing:
        raise ConfigError("protocol missing fields: {}".format(", ".join(missing)))
    if protocol["default_split"] != "pilot":
        raise ConfigError("development protocol must default to pilot")
    if not isinstance(protocol["held_out_enabled"], bool):
        raise ConfigError("held_out_enabled must be boolean")
    if protocol["held_out_enabled"]:
        if protocol.get("frozen") is not True:
            raise ConfigError("held-out execution requires frozen=true")
        aliases = protocol.get("frozen_model_aliases")
        if (
            not isinstance(aliases, list)
            or not aliases
            or len(aliases) != len(set(aliases))
            or not all(isinstance(alias, str) and alias for alias in aliases)
        ):
            raise ConfigError("held-out execution requires unique frozen model aliases")
        if protocol.get("pilot_gate_status") != "passed":
            raise ConfigError("held-out execution requires a passed Pilot gate")
    for field in ("max_output_tokens", "request_timeout_seconds"):
        if not isinstance(protocol[field], int) or protocol[field] <= 0:
            raise ConfigError("{} must be a positive integer".format(field))
    if not isinstance(protocol["max_network_retries"], int):
        raise ConfigError("max_network_retries must be an integer")
    if protocol["max_network_retries"] < 0:
        raise ConfigError("max_network_retries cannot be negative")
    backoff = protocol["retry_backoff_seconds"]
    if not isinstance(backoff, list):
        raise ConfigError("retry_backoff_seconds must be a list")
    if len(backoff) < protocol["max_network_retries"]:
        raise ConfigError("retry backoff list is shorter than retry count")
    if any(not isinstance(value, (int, float)) or value < 0 for value in backoff):
        raise ConfigError("retry backoff values must be nonnegative numbers")
    if protocol["generation_attempts"] != 1:
        raise ConfigError("initial generation must use exactly one attempt")
    generation_mode = protocol.get("generation_mode", "direct_zero_shot")
    if generation_mode == "direct_zero_shot":
        if protocol["repair_rounds"] != 0:
            raise ConfigError("Direct Zero-shot must use zero repair rounds")
    elif generation_mode == "repair":
        if protocol["repair_rounds"] not in (1, 2):
            raise ConfigError("PA-SCoRe repair round must be 1 or 2")
    else:
        raise ConfigError("unsupported generation_mode: {}".format(generation_mode))
    return protocol


def protocol_sha256(protocol):
    canonical = json.dumps(
        protocol,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

#!/usr/bin/env python3
"""Minimal OpenAI-compatible chat-completion transport."""

from __future__ import print_function

import json
import os
import socket
import subprocess
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


RETRYABLE_STATUS_CODES = frozenset((408, 429, 500, 502, 503, 504))
NON_RETRYABLE_QUOTA_MARKERS = frozenset((
    "credit_balance_exhausted",
    "insufficient_quota",
    "no credits remaining",
))


class ClientError(RuntimeError):
    pass


class ResponseError(ClientError):
    pass


def build_request(model_config, protocol, system_prompt, user_prompt):
    payload = {
        "messages": [
            {"content": system_prompt, "role": "system"},
            {"content": user_prompt, "role": "user"},
        ],
        "model": model_config["model"],
        "stream": False,
    }
    payload[model_config["max_tokens_parameter"]] = protocol["max_output_tokens"]
    if model_config.get("supports_temperature", False):
        payload["temperature"] = protocol["temperature"]
    return payload


def _redact(message, api_key):
    text = str(message)
    if api_key:
        text = text.replace(api_key, "[REDACTED]")
    return text


def _error_body(error):
    try:
        body = error.read()
    except Exception:
        return ""
    if isinstance(body, bytes):
        return body.decode("utf-8", errors="replace")
    return str(body)


def _is_retryable_http_error(status, detail):
    if status not in RETRYABLE_STATUS_CODES:
        return False
    if status != 429:
        return True
    lowered = detail.lower()
    return not any(marker in lowered for marker in NON_RETRYABLE_QUOTA_MARKERS)


def _curl_quote(value):
    return str(value).replace("\\", "\\\\").replace('"', '\\"')


def _curl_request(url, body, api_key, timeout, runner):
    body_descriptor, body_path = tempfile.mkstemp(
        prefix="pascore-request-", suffix=".json"
    )
    response_descriptor, response_path = tempfile.mkstemp(
        prefix="pascore-response-", suffix=".json"
    )
    os.close(response_descriptor)
    try:
        with os.fdopen(body_descriptor, "wb") as handle:
            handle.write(body)
        config = (
            'url = "{}"\n'
            'request = "POST"\n'
            'header = "Authorization: Bearer {}"\n'
            'header = "Content-Type: application/json"\n'
        ).format(_curl_quote(url), _curl_quote(api_key))
        arguments = [
            "curl.exe",
            "--silent",
            "--show-error",
            "--config",
            "-",
            "--data-binary",
            "@" + body_path,
            "--output",
            response_path,
            "--write-out",
            "%{http_code}",
            "--connect-timeout",
            str(min(30, timeout)),
            "--max-time",
            str(timeout),
        ]
        if os.name == "nt":
            arguments.insert(3, "--ssl-revoke-best-effort")
        try:
            completed = runner(
                arguments,
                input=config.encode("utf-8"),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout + 15,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            raise ClientError(
                "curl process exceeded {} seconds and was not retried: {}".format(
                    timeout + 15, error
                )
            )
        stderr = completed.stderr.decode("utf-8", errors="replace").strip()
        if completed.returncode != 0:
            if completed.returncode == 28:
                lowered_stderr = stderr.lower()
                if any(marker in lowered_stderr for marker in (
                    "connection timed out",
                    "failed to connect",
                    "couldn't connect",
                )):
                    raise URLError(
                        "curl connection timeout: {}".format(
                            _redact(stderr or "no diagnostic", api_key)
                        )
                    )
                raise ClientError(
                    "generation request timed out after {} seconds and was not retried "
                    "because the provider may still have completed it".format(timeout)
                )
            raise URLError(
                "curl exited {}: {}".format(
                    completed.returncode,
                    _redact(stderr or "no diagnostic", api_key),
                )
            )
        status_text = completed.stdout.decode("ascii", errors="replace").strip()
        try:
            status = int(status_text[-3:])
        except ValueError:
            raise ResponseError("curl returned invalid HTTP status: {}".format(status_text))
        with open(response_path, "rb") as handle:
            response_body = handle.read()
        return status, response_body
    finally:
        for path in (body_path, response_path):
            try:
                os.remove(path)
            except OSError:
                pass


def _request_chat_completion_curl(
    model_config,
    protocol,
    body,
    api_key,
    sleep_fn,
    runner,
):
    url = model_config["base_url"].rstrip("/") + "/chat/completions"
    timeout = protocol["request_timeout_seconds"]
    max_retries = protocol["max_network_retries"]
    backoff = protocol["retry_backoff_seconds"]
    started = time.monotonic()

    for attempt_index in range(max_retries + 1):
        try:
            status, response_body = _curl_request(
                url, body, api_key, timeout, runner
            )
            detail = _redact(
                response_body.decode("utf-8", errors="replace"), api_key
            )
            if status >= 400:
                if _is_retryable_http_error(status, detail) and attempt_index < max_retries:
                    sleep_fn(backoff[attempt_index])
                    continue
                message = "HTTP {} from {}".format(status, model_config["provider"])
                if detail:
                    message += ": " + detail[:1000]
                raise ClientError(message)
            try:
                document = json.loads(response_body.decode("utf-8"))
            except (UnicodeDecodeError, ValueError) as error:
                raise ResponseError("invalid JSON response: {}".format(error))
            return document, {
                "attempts": attempt_index + 1,
                "elapsed_seconds": round(time.monotonic() - started, 6),
                "http_status": status,
                "provider": model_config["provider"],
                "transport": "curl",
            }
        except (URLError, socket.timeout, TimeoutError) as error:
            if attempt_index < max_retries:
                sleep_fn(backoff[attempt_index])
                continue
            raise ClientError(
                "network failure from {} after {} attempts: {}".format(
                    model_config["provider"],
                    attempt_index + 1,
                    _redact(error, api_key),
                )
            )

    raise ClientError("unreachable curl retry state")


def request_chat_completion(
    model_config,
    protocol,
    payload,
    api_key,
    opener=None,
    sleep_fn=time.sleep,
    curl_runner=None,
):
    if not api_key:
        raise ClientError("API key is empty")
    url = model_config["base_url"].rstrip("/") + "/chat/completions"
    body = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    if model_config.get("transport", "urllib") == "curl":
        return _request_chat_completion_curl(
            model_config,
            protocol,
            body,
            api_key,
            sleep_fn,
            curl_runner or subprocess.run,
        )
    if opener is None:
        opener = urlopen
    headers = {
        "Authorization": "Bearer {}".format(api_key),
        "Content-Type": "application/json",
    }
    timeout = protocol["request_timeout_seconds"]
    max_retries = protocol["max_network_retries"]
    backoff = protocol["retry_backoff_seconds"]
    started = time.monotonic()

    for attempt_index in range(max_retries + 1):
        request = Request(url, data=body, headers=headers, method="POST")
        try:
            response = opener(request, timeout)
            with response:
                response_body = response.read()
                status = response.getcode()
            try:
                document = json.loads(response_body.decode("utf-8"))
            except (UnicodeDecodeError, ValueError) as error:
                raise ResponseError("invalid JSON response: {}".format(error))
            return document, {
                "attempts": attempt_index + 1,
                "elapsed_seconds": round(time.monotonic() - started, 6),
                "http_status": status,
                "provider": model_config["provider"],
                "transport": "urllib",
            }
        except HTTPError as error:
            status = int(error.code)
            detail = _redact(_error_body(error), api_key)
            if _is_retryable_http_error(status, detail) and attempt_index < max_retries:
                sleep_fn(backoff[attempt_index])
                continue
            message = "HTTP {} from {}".format(status, model_config["provider"])
            if detail:
                message += ": " + detail[:1000]
            raise ClientError(message)
        except ResponseError:
            raise
        except (URLError, socket.timeout, TimeoutError) as error:
            if attempt_index < max_retries:
                sleep_fn(backoff[attempt_index])
                continue
            raise ClientError(
                "network failure from {} after {} attempts: {}".format(
                    model_config["provider"],
                    attempt_index + 1,
                    _redact(error, api_key),
                )
            )

    raise ClientError("unreachable retry state")


def normalize_response(response, requested_model):
    if not isinstance(response, dict):
        raise ResponseError("response root must be an object")
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ResponseError("response must contain at least one choice")
    choice = choices[0]
    if not isinstance(choice, dict):
        raise ResponseError("first choice must be an object")
    message = choice.get("message")
    if not isinstance(message, dict):
        raise ResponseError("first choice has no message object")
    content = message.get("content")
    reasoning = message.get("reasoning_content")
    if reasoning is not None and not isinstance(reasoning, str):
        reasoning = str(reasoning)
    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, str):
                text_parts.append(part)
            elif (
                isinstance(part, dict)
                and part.get("type") in ("text", "output_text")
                and isinstance(part.get("text"), str)
            ):
                text_parts.append(part["text"])
            else:
                raise ResponseError(
                    "assistant content contains a non-text block: {}".format(
                        type(part).__name__
                    )
                )
        content = "".join(text_parts)
    if not isinstance(content, str):
        raise ResponseError(
            "assistant final content is unavailable "
            "(content_type={}, finish_reason={}, reasoning_content_present={})".format(
                type(content).__name__,
                choice.get("finish_reason"),
                bool(reasoning),
            )
        )
    usage = response.get("usage", {})
    if not isinstance(usage, dict):
        usage = {}
    return {
        "content": content,
        "finish_reason": choice.get("finish_reason"),
        "reasoning_content": reasoning,
        "requested_model": requested_model,
        "response_id": response.get("id"),
        "returned_model": response.get("model"),
        "usage": usage,
    }

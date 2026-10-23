import io
import json
import os
import subprocess
import unittest
from urllib.error import HTTPError

from scripts.openai_compatible_client import (
    ClientError,
    ResponseError,
    build_request,
    normalize_response,
    request_chat_completion,
)


PROTOCOL = {
    "temperature": 0,
    "max_output_tokens": 8192,
    "request_timeout_seconds": 180,
    "max_network_retries": 3,
    "retry_backoff_seconds": [2, 4, 8],
}

GATEWAY_MODEL = {
    "provider": "gateway",
    "base_url": "https://gateway.example/v1",
    "model": "qwen3.8-reasoner",
    "api_key_env": "GATEWAY_API_KEY",
    "max_tokens_parameter": "max_tokens",
    "supports_temperature": True,
    "transport": "urllib",
}

OPENAI_MODEL = {
    "provider": "openai",
    "base_url": "https://api.openai.com/v1",
    "model": "gpt-5.6-sol",
    "api_key_env": "OPENAI_API_KEY",
    "max_tokens_parameter": "max_completion_tokens",
    "supports_temperature": True,
    "transport": "urllib",
}


class FakeResponse(object):
    def __init__(self, document, status=200):
        self.body = json.dumps(document).encode("utf-8")
        self.status = status

    def read(self):
        return self.body

    def getcode(self):
        return self.status

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


def success_document():
    return {
        "id": "chatcmpl-test",
        "model": "qwen3.8-reasoner-202609",
        "choices": [{
            "finish_reason": "stop",
            "message": {
                "role": "assistant",
                "content": "print('ok')",
                "reasoning_content": "I should return code only."
            }
        }],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 25,
            "total_tokens": 125
        }
    }


class OpenAiCompatibleClientTests(unittest.TestCase):
    def test_build_request_sends_only_two_messages_and_common_budget(self):
        payload = build_request(GATEWAY_MODEL, PROTOCOL, "system text", "user text")

        self.assertEqual("qwen3.8-reasoner", payload["model"])
        self.assertEqual(8192, payload["max_tokens"])
        self.assertEqual(0, payload["temperature"])
        self.assertFalse(payload["stream"])
        self.assertEqual(
            [
                {"role": "system", "content": "system text"},
                {"role": "user", "content": "user text"},
            ],
            payload["messages"],
        )
        serialized = json.dumps(payload)
        self.assertNotIn("level", serialized)
        self.assertNotIn("subtype", serialized)
        self.assertNotIn("split", serialized)

    def test_build_request_uses_openai_token_parameter(self):
        payload = build_request(OPENAI_MODEL, PROTOCOL, "system", "user")

        self.assertEqual(8192, payload["max_completion_tokens"])
        self.assertNotIn("max_tokens", payload)

    def test_build_request_omits_temperature_when_model_does_not_support_it(self):
        model = dict(GATEWAY_MODEL)
        model["supports_temperature"] = False
        payload = build_request(model, PROTOCOL, "system", "user")

        self.assertNotIn("temperature", payload)

    def test_normalize_response_keeps_final_and_reasoning_content_separate(self):
        normalized = normalize_response(success_document(), "qwen3.8-reasoner")

        self.assertEqual("qwen3.8-reasoner", normalized["requested_model"])
        self.assertEqual("qwen3.8-reasoner-202609", normalized["returned_model"])
        self.assertEqual("print('ok')", normalized["content"])
        self.assertEqual(
            "I should return code only.", normalized["reasoning_content"]
        )
        self.assertEqual(125, normalized["usage"]["total_tokens"])

    def test_normalize_response_rejects_empty_choices_and_wrong_shapes(self):
        with self.assertRaises(ResponseError):
            normalize_response({"choices": []}, "model")
        with self.assertRaises(ResponseError):
            normalize_response([], "model")
        with self.assertRaises(ResponseError):
            normalize_response({"choices": [{}]}, "model")

    def test_normalize_response_accepts_text_content_blocks(self):
        document = success_document()
        document["choices"][0]["message"]["content"] = [
            {"type": "text", "text": "print('"},
            {"type": "text", "text": "ok')"},
        ]

        normalized = normalize_response(document, "qwen3.8-reasoner")

        self.assertEqual("print('ok')", normalized["content"])

    def test_normalize_response_reports_null_final_content_without_using_reasoning(self):
        document = success_document()
        document["choices"][0]["finish_reason"] = "length"
        document["choices"][0]["message"]["content"] = None

        with self.assertRaises(ResponseError) as context:
            normalize_response(document, "qwen3.8-reasoner")

        message = str(context.exception)
        self.assertIn("content_type=NoneType", message)
        self.assertIn("finish_reason=length", message)
        self.assertIn("reasoning_content_present=True", message)

    def test_request_does_not_retry_401_or_expose_key(self):
        calls = []
        api_key = "sk-sensitive-value"

        def opener(request, timeout):
            calls.append((request, timeout))
            raise HTTPError(
                request.full_url,
                401,
                "Unauthorized",
                {},
                io.BytesIO(b'{"error":{"message":"bad key"}}'),
            )

        with self.assertRaises(ClientError) as context:
            request_chat_completion(
                GATEWAY_MODEL,
                PROTOCOL,
                build_request(GATEWAY_MODEL, PROTOCOL, "system", "user"),
                api_key,
                opener=opener,
                sleep_fn=lambda seconds: None,
            )

        self.assertEqual(1, len(calls))
        self.assertNotIn(api_key, str(context.exception))
        self.assertIn("401", str(context.exception))

    def test_request_retries_429_then_returns_json_and_transport_metadata(self):
        calls = []
        sleeps = []

        def opener(request, timeout):
            calls.append((request, timeout))
            if len(calls) == 1:
                raise HTTPError(
                    request.full_url,
                    429,
                    "Too Many Requests",
                    {},
                    io.BytesIO(b'{"error":{"message":"slow down"}}'),
                )
            return FakeResponse(success_document())

        document, transport = request_chat_completion(
            GATEWAY_MODEL,
            PROTOCOL,
            build_request(GATEWAY_MODEL, PROTOCOL, "system", "user"),
            "secret-value",
            opener=opener,
            sleep_fn=sleeps.append,
        )

        self.assertEqual(success_document(), document)
        self.assertEqual(2, len(calls))
        self.assertEqual([2], sleeps)
        self.assertEqual(2, transport["attempts"])
        self.assertEqual(200, transport["http_status"])
        request = calls[-1][0]
        self.assertEqual("Bearer secret-value", request.headers["Authorization"])
        self.assertEqual(180, calls[-1][1])

    def test_request_does_not_retry_429_when_quota_is_exhausted(self):
        calls = []
        sleeps = []

        def opener(request, timeout):
            calls.append((request, timeout))
            raise HTTPError(
                request.full_url,
                429,
                "Too Many Requests",
                {},
                io.BytesIO(
                    b'{"error":{"message":"You have no credits remaining.",'
                    b'"type":"insufficient_quota",'
                    b'"code":"credit_balance_exhausted"}}'
                ),
            )

        with self.assertRaises(ClientError) as context:
            request_chat_completion(
                OPENAI_MODEL,
                PROTOCOL,
                build_request(OPENAI_MODEL, PROTOCOL, "system", "user"),
                "secret-value",
                opener=opener,
                sleep_fn=sleeps.append,
            )

        self.assertEqual(1, len(calls))
        self.assertEqual([], sleeps)
        self.assertIn("credit_balance_exhausted", str(context.exception))

    def test_request_rejects_malformed_json_response(self):
        class MalformedResponse(FakeResponse):
            def __init__(self):
                self.body = b"not-json"
                self.status = 200

        with self.assertRaises(ResponseError):
            request_chat_completion(
                GATEWAY_MODEL,
                PROTOCOL,
                build_request(GATEWAY_MODEL, PROTOCOL, "system", "user"),
                "secret-value",
                opener=lambda request, timeout: MalformedResponse(),
                sleep_fn=lambda seconds: None,
            )

    def test_curl_transport_keeps_key_out_of_argv_and_returns_json(self):
        calls = []
        api_key = "secret-curl-value"
        model = dict(GATEWAY_MODEL)
        model["transport"] = "curl"

        def curl_runner(args, input, stdout, stderr, timeout, check):
            calls.append({
                "args": args,
                "input": input,
                "timeout": timeout,
            })
            output_path = args[args.index("--output") + 1]
            with open(output_path, "wb") as handle:
                handle.write(json.dumps(success_document()).encode("utf-8"))
            return subprocess.CompletedProcess(args, 0, stdout=b"200", stderr=b"")

        document, transport = request_chat_completion(
            model,
            PROTOCOL,
            build_request(model, PROTOCOL, "system", "user"),
            api_key,
            sleep_fn=lambda seconds: None,
            curl_runner=curl_runner,
        )

        self.assertEqual(success_document(), document)
        self.assertEqual(1, len(calls))
        self.assertNotIn(api_key, " ".join(calls[0]["args"]))
        self.assertIn(api_key.encode("utf-8"), calls[0]["input"])
        if os.name == "nt":
            self.assertIn("--ssl-revoke-best-effort", calls[0]["args"])
        self.assertEqual("curl", transport["transport"])
        self.assertEqual(200, transport["http_status"])
        body_path = calls[0]["args"][calls[0]["args"].index("--data-binary") + 1][1:]
        output_path = calls[0]["args"][calls[0]["args"].index("--output") + 1]
        self.assertFalse(os.path.exists(body_path))
        self.assertFalse(os.path.exists(output_path))

    def test_curl_generation_timeout_is_not_retried(self):
        calls = []
        sleeps = []
        model = dict(GATEWAY_MODEL)
        model["transport"] = "curl"

        def curl_runner(args, input, stdout, stderr, timeout, check):
            calls.append(args)
            return subprocess.CompletedProcess(
                args,
                28,
                stdout=b"000",
                stderr=b"curl: (28) Operation timed out",
            )

        with self.assertRaises(ClientError) as context:
            request_chat_completion(
                model,
                PROTOCOL,
                build_request(model, PROTOCOL, "system", "user"),
                "secret-value",
                sleep_fn=sleeps.append,
                curl_runner=curl_runner,
            )

        self.assertEqual(1, len(calls))
        self.assertEqual([], sleeps)
        self.assertIn("not retried", str(context.exception))

    def test_curl_connection_timeout_is_retried_as_network_failure(self):
        calls = []
        sleeps = []
        model = dict(GATEWAY_MODEL)
        model["transport"] = "curl"

        def curl_runner(args, input, stdout, stderr, timeout, check):
            calls.append(args)
            return subprocess.CompletedProcess(
                args,
                28,
                stdout=b"000",
                stderr=b"curl: (28) Connection timed out after 30002 milliseconds",
            )

        with self.assertRaises(ClientError) as context:
            request_chat_completion(
                model,
                PROTOCOL,
                build_request(model, PROTOCOL, "system", "user"),
                "secret-value",
                sleep_fn=sleeps.append,
                curl_runner=curl_runner,
            )

        self.assertEqual(4, len(calls))
        self.assertEqual(PROTOCOL["retry_backoff_seconds"], sleeps)
        self.assertIn("network failure", str(context.exception))
        self.assertIn("connection timeout", str(context.exception).lower())


if __name__ == "__main__":
    unittest.main()

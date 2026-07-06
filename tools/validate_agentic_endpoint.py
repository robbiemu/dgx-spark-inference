#!/usr/bin/env python3
"""Authenticated chat/tool/schema smoke test without printing the API key."""

from __future__ import annotations

import argparse
import json
import os
import urllib.request


def post(url: str, key: str, payload: dict) -> dict:
    request = urllib.request.Request(
        f"{url}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.load(response)


def message(response: dict) -> dict:
    return response["choices"][0]["message"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    key = os.environ.get("SGLANG_API_KEY")
    if not key:
        raise SystemExit("SGLANG_API_KEY is required")

    common = {"model": args.model, "temperature": 0}
    chat = post(
        args.url,
        key,
        {
            **common,
            "messages": [
                {
                    "role": "user",
                    "content": "Reply with exactly: MANAGED CHAT OK",
                }
            ],
            "max_tokens": 1024,
        },
    )
    content = message(chat).get("content") or ""
    assert "MANAGED CHAT OK" in content, f"chat content mismatch: {content!r}"
    print("PASS chat: visible content")

    tool = post(
        args.url,
        key,
        {
            **common,
            "messages": [
                {
                    "role": "user",
                    "content": "Use get_temperature for Boston. Do not answer directly.",
                }
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_temperature",
                        "description": "Get the current temperature for a city.",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                            "additionalProperties": False,
                        },
                    },
                }
            ],
            "tool_choice": "required",
            "max_tokens": 1024,
        },
    )
    calls = message(tool).get("tool_calls") or []
    assert calls, "tool parser returned no tool_calls"
    assert calls[0]["function"]["name"] == "get_temperature", calls
    arguments = json.loads(calls[0]["function"]["arguments"])
    assert arguments.get("city", "").lower() == "boston", arguments
    print("PASS tool: parsed get_temperature(city=Boston)")

    structured = post(
        args.url,
        key,
        {
            **common,
            "messages": [
                {
                    "role": "user",
                    "content": "Return the requested validation object.",
                }
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "validation",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "status": {
                                "type": "string",
                                "enum": ["STRUCTURED_OK"],
                            },
                            "count": {"type": "integer", "const": 7},
                        },
                        "required": ["status", "count"],
                        "additionalProperties": False,
                    },
                },
            },
            "max_tokens": 2048,
        },
    )
    document = json.loads(message(structured).get("content") or "")
    assert document == {"status": "STRUCTURED_OK", "count": 7}, document
    print("PASS structured: strict JSON schema")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

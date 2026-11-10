"""Deterministic Anthropic-format mock upstream.

The gate forwards request bytes unchanged, so tests steer the mock through the
request body: ``metadata.user_id = "mock:key=value,key=value"``. Recognised keys:

  status=529          respond with this HTTP status and an Anthropic error body
  error_type=...      error type for that body (default derived from status)
  retry_after=2       add a Retry-After header
  chunks=20           number of text deltas in a stream (default 5)
  delay_ms=10         pause before each stream chunk
  hold_ms=500         pause after message_start (keeps admitted streams open)
  input=100 output=50 usage counters reported
  cache_read=0 cache_5m=0 cache_1h=0   cache usage reported (split in message_start
                      and non-streaming bodies; message_delta omits it, as Anthropic does)
  truncate=1          end the stream before message_delta/message_stop
  sse_error=1         emit an ``error`` event under HTTP 200
  model=...           report this model instead of the requested one

``GET /_last`` returns the last request's body (base64) and headers, and
``GET /_last_response`` the exact bytes of the last response, for fidelity checks.

Run: ``python -m bench.mock_upstream --port 9999``
"""

import argparse
import asyncio
import base64
import json
from collections.abc import AsyncIterator
from typing import Any

import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

ERROR_TYPES = {
    400: "invalid_request_error",
    401: "authentication_error",
    403: "permission_error",
    429: "rate_limit_error",
    500: "api_error",
    529: "overloaded_error",
}

LAST: dict[str, Any] = {}


def scenario(body: dict[str, Any]) -> dict[str, str]:
    meta = body.get("metadata")
    user = meta.get("user_id") if isinstance(meta, dict) else None
    if not isinstance(user, str) or not user.startswith("mock:"):
        return {}
    pairs = (item.split("=", 1) for item in user[5:].split(",") if "=" in item)
    return {k.strip(): v.strip() for k, v in pairs}


def _usage(s: dict[str, str], *, output: int) -> dict[str, Any]:
    m5, h1 = int(s.get("cache_5m", 0)), int(s.get("cache_1h", 0))
    return {
        "input_tokens": int(s.get("input", 100)),
        "cache_creation_input_tokens": m5 + h1,
        "cache_read_input_tokens": int(s.get("cache_read", 0)),
        "cache_creation": {"ephemeral_5m_input_tokens": m5, "ephemeral_1h_input_tokens": h1},
        "output_tokens": output,
    }


def _sse(event: str, data: dict[str, Any]) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n".encode()


async def messages(request: Request) -> Response:
    raw = await request.body()
    LAST["body"] = raw
    LAST["headers"] = [(k.decode(), v.decode()) for k, v in request.headers.raw]
    body = json.loads(raw)
    s = scenario(body)
    model = s.get("model", body.get("model", "mock-model"))

    status = int(s.get("status", 200))
    if status >= 400:
        headers = {"retry-after": s["retry_after"]} if "retry_after" in s else {}
        payload = json.dumps(
            {
                "type": "error",
                "error": {
                    "type": s.get("error_type", ERROR_TYPES.get(status, "api_error")),
                    "message": "mock error",
                },
            }
        ).encode()
        LAST["response"] = payload
        return Response(payload, status_code=status, headers=headers, media_type="application/json")

    output = int(s.get("output", 50))
    if not body.get("stream"):
        payload = json.dumps(
            {
                "id": "msg_mock",
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": [{"type": "text", "text": "Hello from the mock."}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": _usage(s, output=output),
            }
        ).encode()
        LAST["response"] = payload
        return Response(payload, media_type="application/json")

    return StreamingResponse(_stream(s, model, output), media_type="text/event-stream")


async def _stream(s: dict[str, str], model: str, output: int) -> AsyncIterator[bytes]:
    chunks = int(s.get("chunks", 5))
    delay = int(s.get("delay_ms", 0)) / 1000
    sent = bytearray()

    async def emit(data: bytes) -> bytes:
        if delay:
            await asyncio.sleep(delay)
        sent.extend(data)
        LAST["response"] = bytes(sent)
        return data

    start_usage = _usage(s, output=1)
    yield await emit(
        _sse(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "msg_mock",
                    "type": "message",
                    "role": "assistant",
                    "model": model,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": start_usage,
                },
            },
        )
    )
    if "hold_ms" in s:
        await asyncio.sleep(int(s["hold_ms"]) / 1000)
    yield await emit(
        _sse(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
        )
    )
    for i in range(chunks):
        yield await emit(
            _sse(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": f"chunk {i} — ünïcödé "},
                },
            )
        )
    if s.get("sse_error"):
        yield await emit(
            _sse(
                "error",
                {
                    "type": "error",
                    "error": {"type": "overloaded_error", "message": "mock overload"},
                },
            )
        )
        return
    if s.get("truncate"):
        return
    yield await emit(_sse("content_block_stop", {"type": "content_block_stop", "index": 0}))
    yield await emit(
        _sse(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                # Like Anthropic: the final delta repeats the cumulative cache-write
                # aggregate but omits the cache_creation split from message_start.
                "usage": {
                    k: v for k, v in _usage(s, output=output).items() if k != "cache_creation"
                },
            },
        )
    )
    yield await emit(_sse("message_stop", {"type": "message_stop"}))


async def count_tokens(request: Request) -> Response:
    raw = await request.body()
    LAST["body"] = raw
    LAST["headers"] = [(k.decode(), v.decode()) for k, v in request.headers.raw]
    return JSONResponse({"input_tokens": max(1, len(raw) // 4)})


async def last(_: Request) -> Response:
    return JSONResponse(
        {
            "body_b64": base64.b64encode(LAST.get("body", b"")).decode(),
            "headers": LAST.get("headers", []),
        }
    )


async def last_response(_: Request) -> Response:
    return Response(LAST.get("response", b""), media_type="application/octet-stream")


app = Starlette(
    routes=[
        Route("/v1/messages", messages, methods=["POST"]),
        Route("/anthropic/v1/messages", messages, methods=["POST"]),
        Route("/v1/messages/count_tokens", count_tokens, methods=["POST"]),
        Route("/anthropic/v1/messages/count_tokens", count_tokens, methods=["POST"]),
        Route("/_last", last),
        Route("/_last_response", last_response),
    ]
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9999)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()

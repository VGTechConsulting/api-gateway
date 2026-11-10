"""HTTP adapter disconnect handling, driven over raw ASGI.

The client's ``http.disconnect`` is scripted precisely, which a real socket
client cannot do, so the gate's reaction before response headers is testable.
"""

import asyncio
import json
from collections.abc import Coroutine
from dataclasses import replace
from typing import Any

import pytest

from gate.adapters.http import create_app
from gate.domain.models import TerminationReason
from tests.app.fakes import (
    RAW_KEY,
    Gate,
    ScriptedResponse,
    error_response,
    make_gate,
    stream_chunks,
)


def run(coro: Coroutine[Any, Any, None]) -> None:
    asyncio.run(coro)


def scope() -> dict[str, Any]:
    return {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/messages",
        "raw_path": b"/v1/messages",
        "query_string": b"",
        "root_path": "",
        "headers": [
            (b"x-api-key", RAW_KEY.encode()),
            (b"anthropic-version", b"2023-06-01"),
            (b"content-type", b"application/json"),
        ],
        "client": ("127.0.0.1", 1),
        "server": ("127.0.0.1", 2),
    }


class Client:
    """Sends the body, then disconnects when ``leave`` is set."""

    def __init__(self) -> None:
        self.leave = asyncio.Event()
        self.body_sent = False
        self.sent: list[dict[str, Any]] = []

    async def receive(self) -> dict[str, Any]:
        if not self.body_sent:
            self.body_sent = True
            body = {
                "model": "model-a",
                "max_tokens": 100,
                "stream": True,
                "messages": [{"role": "user", "content": "hello"}],
            }
            return {"type": "http.request", "body": json.dumps(body).encode(), "more_body": False}
        await self.leave.wait()
        return {"type": "http.disconnect"}

    async def send(self, message: dict[str, Any]) -> None:
        self.sent.append(message)


def app_for(gate: Gate) -> Any:
    return create_app(
        gate.handler, max_request_bytes=1 << 20, healthy=lambda: True, cleanup_timeout_s=1
    )


def stalled_rejection(gate: Gate) -> asyncio.Event:
    """dep-a answers 529 only once released; dep-b would accept the fallback."""
    headers_hold = asyncio.Event()
    gate.upstream.add(
        "https://dep-a.example",
        replace(error_response(529, "overloaded_error"), headers_hold=headers_hold),
    )
    gate.upstream.add("https://dep-b.example", ScriptedResponse(chunks=stream_chunks()))
    return headers_hold


def test_disconnect_while_waiting_for_upstream_starts_no_fallback() -> None:
    gate = make_gate()
    client = Client()

    async def scenario() -> None:
        headers_hold = stalled_rejection(gate)
        call = asyncio.ensure_future(app_for(gate)(scope(), client.receive, client.send))
        await asyncio.sleep(0.01)  # dispatched to dep-a, waiting for its headers
        client.leave.set()
        await asyncio.sleep(0.01)
        headers_hold.set()  # a late 529 must not trigger dispatch to dep-b
        await asyncio.wait_for(call, timeout=2)

    run(scenario())
    assert len(gate.upstream.calls) == 1
    [attempt] = gate.usage.attempt_list()
    assert attempt.deployment == "dep-a"
    assert attempt.termination_reason is TerminationReason.CLIENT_DISCONNECT
    assert attempt.client_disconnected
    assert gate.admission.reserved_tokens("key-1") == 0
    parent = gate.usage.only_parent()
    assert parent.http_status == 499 and parent.client_disconnected
    assert client.sent == []  # nothing is sent to a client that has gone


def test_server_cancellation_before_headers_is_recorded_as_shutdown() -> None:
    gate = make_gate()
    client = Client()  # never leaves

    async def scenario() -> None:
        stalled_rejection(gate)
        call = asyncio.ensure_future(app_for(gate)(scope(), client.receive, client.send))
        await asyncio.sleep(0.01)
        call.cancel()
        with pytest.raises(asyncio.CancelledError):
            await call

    run(scenario())
    assert len(gate.upstream.calls) == 1
    [attempt] = gate.usage.attempt_list()
    assert attempt.termination_reason is TerminationReason.SHUTDOWN
    assert not attempt.client_disconnected
    assert not gate.usage.only_parent().client_disconnected


def test_reply_after_handle_completes_is_streamed_normally() -> None:
    gate = make_gate()
    gate.upstream.add("https://dep-a.example", ScriptedResponse(chunks=stream_chunks()))
    client = Client()

    async def scenario() -> None:
        await asyncio.wait_for(app_for(gate)(scope(), client.receive, client.send), timeout=2)

    run(scenario())
    assert client.sent[0]["type"] == "http.response.start" and client.sent[0]["status"] == 200
    body = b"".join(m.get("body", b"") for m in client.sent[1:])
    assert body == b"".join(stream_chunks())
    [attempt] = gate.usage.attempt_list()
    assert attempt.termination_reason is TerminationReason.NORMAL

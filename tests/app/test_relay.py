"""Relay robustness: parser faults and close failures never abort relay or skip settlement.

The parser here deliberately violates the ``UsageParser`` "never raises" contract, so
the relay's own guard is what is under test, independent of any provider parser.
"""

import asyncio
from collections.abc import Coroutine
from dataclasses import dataclass
from typing import Any, cast

import pytest

from gate.app.ports import UpstreamResponse
from gate.app.relay import Relay, RelayResult
from gate.domain.models import UsageSource
from gate.providers.base import ParseResult
from tests.app.fakes import FakeMono, ScriptedResponse

CHUNKS = [b"data: 1\n\n", b"data: 2\n\n", b"data: 3\n\n", b"data: 4\n\n"]
SSE_HEADERS = [("content-type", "text/event-stream")]


def run[T](coro: Coroutine[Any, Any, T]) -> T:
    return asyncio.run(coro)


def complete_result() -> ParseResult:
    return ParseResult(
        snapshot=None,
        usage_complete=True,
        usage_source=UsageSource.OK,
        streamed=True,
        stream_complete=True,
        stream_error=False,
        upstream_model="model-a",
        error_type=None,
        execution_evidence=True,
    )


@dataclass
class FaultyParser:
    fail_on_feed: int | None = None  # raise on this (0-based) feed call
    fail_result: bool = False
    feeds: int = 0

    def feed(self, chunk: bytes) -> None:
        index = self.feeds
        self.feeds += 1
        if index == self.fail_on_feed:
            raise RecursionError("maximum recursion depth exceeded")

    def result(self, *, eof: bool) -> ParseResult:
        if self.fail_result:
            raise RuntimeError("parser state corrupt")
        return complete_result()


class FailingClose(ScriptedResponse):
    async def aclose(self) -> None:
        self.closed = True
        raise RuntimeError("close failed")


def make_relay(
    parser: FaultyParser, response: ScriptedResponse | None = None
) -> tuple[Relay, ScriptedResponse, list[RelayResult]]:
    response = response or ScriptedResponse(chunks=list(CHUNKS))
    settled: list[RelayResult] = []

    async def finish(result: RelayResult) -> None:
        settled.append(result)

    relay = Relay(
        status=200,
        headers=SSE_HEADERS,
        response=cast(UpstreamResponse, response),
        parser=parser,
        finish=finish,
        mono=FakeMono(),
        cleanup_timeout_s=1.0,
    )
    return relay, response, settled


async def pump_all(relay: Relay) -> list[bytes]:
    sent: list[bytes] = []

    async def send(chunk: bytes) -> None:
        sent.append(chunk)

    await relay.pump(send)
    return sent


def test_feed_fault_keeps_relaying_and_settles_as_parse_error() -> None:
    parser = FaultyParser(fail_on_feed=1)
    relay, response, settled = make_relay(parser)

    assert run(pump_all(relay)) == CHUNKS
    assert parser.feeds == 2  # nothing more is fed to a parser in an unknown state
    assert response.closed
    [result] = settled
    assert result.eof
    assert result.parse.usage_source is UsageSource.PARSE_ERROR
    assert not result.parse.usage_complete
    assert "parse_error" in result.parse.reasons
    assert result.parse.upstream_model == "model-a"  # retained observations survive


def test_result_fault_still_settles_conservatively() -> None:
    relay, response, settled = make_relay(FaultyParser(fail_result=True))

    assert run(pump_all(relay)) == CHUNKS
    assert response.closed
    [result] = settled
    assert result.parse.usage_source is UsageSource.PARSE_ERROR
    assert not result.parse.usage_complete
    assert result.parse.execution_evidence  # never a zero-cost rejection
    assert result.parse.streamed and result.parse.stream_complete is False


def test_fault_on_pending_chunk_at_cancellation_still_settles() -> None:
    parser = FaultyParser(fail_on_feed=0)
    relay, response, settled = make_relay(parser)

    async def scenario() -> None:
        async def send(chunk: bytes) -> None:
            relay.mark_client_disconnected()
            raise asyncio.CancelledError  # the retained chunk is parsed in cleanup

        with pytest.raises(asyncio.CancelledError):
            await relay.pump(send)

    run(scenario())
    assert parser.feeds == 1
    assert response.closed
    [result] = settled
    assert result.client_disconnected
    assert result.parse.usage_source is UsageSource.PARSE_ERROR


def test_unexpected_close_failure_still_settles() -> None:
    response = FailingClose(chunks=list(CHUNKS))
    relay, _, settled = make_relay(FaultyParser(), response)

    with pytest.raises(RuntimeError, match="close failed"):
        run(pump_all(relay))
    [result] = settled
    assert result.eof and result.parse.usage_complete

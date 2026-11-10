"""Yield-then-parse relay with cancellation-safe settlement.

Each upstream chunk is sent to the client *before* it is parsed, so metering
can never delay or alter relayed bytes. The chunk being sent is retained until
parsed: if the client disconnects at that boundary, the already-received bytes
are still parsed before settlement, without reading more from upstream.

Cleanup (closing upstream, settling, finalising) is bounded and shielded from
request cancellation, so an ordinary disconnect still persists its debit.

Parsing can never abort the relay or skip cleanup: a parser that raises, despite
the ``UsageParser`` contract, is treated as a parse error and the
bytes keep flowing; settlement runs even if closing upstream fails.
"""

import asyncio
from collections.abc import Awaitable, Callable, Coroutine, Sequence
from dataclasses import dataclass, replace
from typing import Any

from gate.app.ports import FailureKind, MonotonicClock, UpstreamReadError, UpstreamResponse
from gate.domain.models import TerminationReason, UsageSource
from gate.providers.base import ParseResult, UsageParser


class ClientGone(Exception):
    """Raised by the HTTP adapter's send function when the client can no longer receive."""


@dataclass(frozen=True)
class RelayResult:
    parse: ParseResult
    eof: bool
    interrupted_by: TerminationReason | None  # disconnects, timeouts, shutdown
    client_disconnected: bool
    first_chunk_received: float | None
    first_chunk_sent: float | None
    ended: float


Finish = Callable[[RelayResult], Awaitable[None]]
Send = Callable[[bytes], Awaitable[None]]


async def run_shielded(coro: Coroutine[Any, Any, None], timeout_s: float) -> None:
    """Run cleanup to completion even if the caller is cancelled meanwhile."""
    task = asyncio.ensure_future(coro)
    try:
        await asyncio.shield(task)
    except asyncio.CancelledError:
        await asyncio.wait({task}, timeout=timeout_s)
        raise


class _GuardedParser:
    """Enforces "never raises" on a usage parser at the relay boundary.

    After the first fault the parser's state is unknown, so it is fed nothing more
    and its result is downgraded to ``parse_error``: never complete usage, and the
    debit falls back to the reservation or the observations retained.
    """

    def __init__(self, parser: UsageParser, *, streamed: bool) -> None:
        self._parser = parser
        self._streamed = streamed
        self._failed = False

    def feed(self, chunk: bytes) -> None:
        if self._failed:
            return
        try:
            self._parser.feed(chunk)
        except Exception:
            self._failed = True

    def result(self, *, eof: bool) -> ParseResult:
        try:
            result = self._parser.result(eof=eof)
        except Exception:
            return self._unparsed()
        if not self._failed:
            return result
        source = (
            result.usage_source
            if result.usage_source is UsageSource.TOO_LARGE
            else UsageSource.PARSE_ERROR
        )
        reasons = result.reasons
        if "parse_error" not in reasons:
            reasons = (*reasons, "parse_error")
        return replace(result, usage_complete=False, usage_source=source, reasons=reasons)

    def _unparsed(self) -> ParseResult:
        """Nothing usable survived: the request may have executed, so never zero-cost."""
        return ParseResult(
            snapshot=None,
            usage_complete=False,
            usage_source=UsageSource.PARSE_ERROR,
            streamed=self._streamed,
            stream_complete=False if self._streamed else None,
            stream_error=False,
            upstream_model=None,
            error_type=None,
            execution_evidence=True,
            reasons=("parse_error",),
        )


def _is_event_stream(headers: Sequence[tuple[str, str]]) -> bool:
    for name, value in headers:
        if name.lower() == "content-type":
            return value.split(";", 1)[0].strip().lower() == "text/event-stream"
    return False


class Relay:
    """One committed upstream response, streamed to the client byte for byte."""

    def __init__(
        self,
        *,
        status: int,
        headers: Sequence[tuple[str, str]],
        response: UpstreamResponse,
        parser: UsageParser,
        finish: Finish,
        mono: MonotonicClock,
        cleanup_timeout_s: float,
    ) -> None:
        self.status = status
        self.headers = headers
        self._response = response
        self._parser = _GuardedParser(parser, streamed=_is_event_stream(headers))
        self._finish = finish
        self._mono = mono
        self.cleanup_timeout_s = cleanup_timeout_s
        self._client_disconnected = False
        self._done = False

    def mark_client_disconnected(self) -> None:
        """The adapter calls this before cancelling ``pump`` on ``http.disconnect``."""
        self._client_disconnected = True

    async def pump(self, send: Send) -> None:
        pending: bytes | None = None
        eof = False
        interrupted: TerminationReason | None = None
        first_received: float | None = None
        first_sent: float | None = None
        try:
            async for chunk in self._response.iter_raw():
                if first_received is None:
                    first_received = self._mono.now()
                pending = chunk
                if first_sent is None:
                    first_sent = self._mono.now()
                await send(chunk)
                self._parser.feed(chunk)
                pending = None
            eof = True
        except UpstreamReadError as exc:
            interrupted = (
                TerminationReason.UPSTREAM_TIMEOUT
                if exc.kind is FailureKind.TIMEOUT
                else TerminationReason.UPSTREAM_DISCONNECT
            )
        except ClientGone:
            self._client_disconnected = True
            interrupted = TerminationReason.CLIENT_DISCONNECT
        except asyncio.CancelledError:
            interrupted = (
                TerminationReason.CLIENT_DISCONNECT
                if self._client_disconnected
                else TerminationReason.SHUTDOWN
            )
            raise
        finally:
            try:
                if pending is not None:
                    self._parser.feed(pending)  # received but unparsed at the yield boundary
            finally:
                await run_shielded(
                    self._cleanup(eof, interrupted, first_received, first_sent),
                    self.cleanup_timeout_s,
                )

    async def close_unstarted(self) -> None:
        """Settle a committed response that was never pumped (e.g. the send failed at start)."""
        await run_shielded(
            self._cleanup(False, TerminationReason.CLIENT_DISCONNECT, None, None),
            self.cleanup_timeout_s,
        )

    async def _cleanup(
        self,
        eof: bool,
        interrupted: TerminationReason | None,
        first_received: float | None,
        first_sent: float | None,
    ) -> None:
        if self._done:
            return
        self._done = True
        try:
            async with asyncio.timeout(self.cleanup_timeout_s):
                await self._response.aclose()
        except (TimeoutError, OSError, UpstreamReadError):
            pass
        finally:  # settle even if closing upstream fails unexpectedly
            result = RelayResult(
                parse=self._parser.result(eof=eof),
                eof=eof,
                interrupted_by=interrupted,
                client_disconnected=self._client_disconnected,
                first_chunk_received=first_received,
                first_chunk_sent=first_sent,
                ended=self._mono.now(),
            )
            await self._finish(result)

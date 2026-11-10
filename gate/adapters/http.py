"""FastAPI routes, relayed streaming responses and error rendering.

Client disconnects (``http.disconnect``) are watched for the whole request, not
just the relay. While the lifecycle is still dispatching (waiting for upstream
headers, draining a rejection, admitting a fallback) a disconnect cancels
``handle`` so no further attempt is started. Once a response is
committed, the relay is driven by a small ASGI response class rather than an
async generator, so a disconnect is turned into a deterministic cancellation of
the pump, whose cleanup settles the attempt.
"""

import asyncio
import contextlib
import time
from collections.abc import Callable
from dataclasses import replace

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from starlette.types import Receive, Scope, Send

from gate.app.lifecycle import ErrorReply, HandleRequest, IncomingRequest, Reply, StreamReply
from gate.app.relay import ClientGone, Relay
from gate.domain.errors import GateError, NotFound
from gate.domain.models import Route
from gate.providers.anthropic import AnthropicFormat

_ANTHROPIC = AnthropicFormat()


class RelayResponse(Response):
    """Streams a committed upstream response and settles it however the stream ends."""

    def __init__(self, relay: Relay) -> None:
        self.relay = relay
        self.status_code = relay.status
        self.background = None

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        raw_headers = [
            (k.lower().encode("latin-1"), v.encode("latin-1")) for k, v in self.relay.headers
        ]
        try:
            await send(
                {"type": "http.response.start", "status": self.status_code, "headers": raw_headers}
            )
        except Exception:
            await self.relay.close_unstarted()
            return

        async def send_chunk(chunk: bytes) -> None:
            try:
                await send({"type": "http.response.body", "body": chunk, "more_body": True})
            except Exception as exc:
                raise ClientGone() from exc

        pump = asyncio.ensure_future(self.relay.pump(send_chunk))
        listener = asyncio.ensure_future(_wait_for_disconnect(receive))
        try:
            await asyncio.wait({pump, listener}, return_when=asyncio.FIRST_COMPLETED)
            if not pump.done():
                # The client went away first: stop reading upstream and settle what we have.
                self.relay.mark_client_disconnected()
                pump.cancel()
                await asyncio.wait({pump})
                return
            pump.result()  # surface unexpected pump errors
        finally:
            listener.cancel()
            if not pump.done():
                # This task is being cancelled (server shutdown): the pump records it.
                pump.cancel()
                await asyncio.wait({pump}, timeout=self.relay.cleanup_timeout_s)
        with contextlib.suppress(Exception):
            await send({"type": "http.response.body", "body": b"", "more_body": False})


class ClientGoneResponse(Response):
    """Nothing to send: the client disconnected before a response was chosen."""

    def __init__(self) -> None:
        self.status_code = 499
        self.background = None

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        return


async def handle_watching_disconnect(
    handler: HandleRequest, incoming: IncomingRequest, receive: Receive, cleanup_timeout_s: float
) -> Reply | None:
    """Run ``handler.handle`` and cancel it if the client disconnects first.

    Returns ``None`` when the client went away; the lifecycle has then settled any
    dispatched attempt as a client disconnect and started no fallback.
    """
    gone = False
    incoming = replace(incoming, client_gone=lambda: gone)
    handling = asyncio.ensure_future(handler.handle(incoming))
    listener = asyncio.ensure_future(_wait_for_disconnect(receive))
    try:
        await asyncio.wait({handling, listener}, return_when=asyncio.FIRST_COMPLETED)
        if not (listener.done() and not listener.cancelled() and listener.exception() is None):
            return await handling
        gone = True
        if not handling.done():
            handling.cancel()
            await asyncio.wait({handling})  # handle's own cleanup is shielded and bounded
            if not handling.cancelled():
                handling.result()  # surface unexpected errors
            return None
        reply = handling.result()
        if isinstance(reply, StreamReply):
            # Committed in the same instant the client left: settle it unsent.
            reply.relay.mark_client_disconnected()
            await reply.relay.close_unstarted()
        return None
    finally:
        listener.cancel()
        if not handling.done():
            # This task is being cancelled (server shutdown): handle records it.
            handling.cancel()
            await asyncio.wait({handling}, timeout=cleanup_timeout_s)


async def _wait_for_disconnect(receive: Receive) -> None:
    while True:
        message = await receive()
        if message["type"] == "http.disconnect":
            return


def render_error(
    error: GateError, request_id: str | None, would_deny: tuple[str, ...] = ()
) -> Response:
    headers = {"x-api-gateway-error": error.gateway_code}
    if request_id:
        headers["x-api-gateway-request-id"] = request_id
    if error.retry_after_s is not None:
        headers["retry-after"] = str(error.retry_after_s)
    if would_deny:
        headers["x-api-gateway-would-deny"] = ",".join(would_deny)
    body = _ANTHROPIC.error_body(error.error_type, error.message, error.gateway_code)
    return Response(body, status_code=error.status, headers=headers, media_type="application/json")


def create_app(
    handler: HandleRequest,
    *,
    max_request_bytes: int,
    healthy: Callable[[], bool],
    cleanup_timeout_s: float,
    monotonic: Callable[[], float] = time.monotonic,
) -> FastAPI:
    app = FastAPI(title="api-gateway", docs_url=None, redoc_url=None, openapi_url=None)

    async def proxy(request: Request, route: Route) -> Response:
        received = monotonic()
        body, too_large = await _read_body(request, max_request_bytes)
        headers = tuple(
            (k.decode("latin-1").lower(), v.decode("latin-1")) for k, v in request.headers.raw
        )
        reply = await handle_watching_disconnect(
            handler,
            IncomingRequest(
                route=route,
                headers=headers,
                body=body,
                body_too_large=too_large,
                received_mono=received,
            ),
            request.receive,
            cleanup_timeout_s,
        )
        if reply is None:
            return ClientGoneResponse()
        if isinstance(reply, ErrorReply):
            return render_error(reply.error, reply.request_id, reply.would_deny)
        return RelayResponse(reply.relay)

    @app.post("/v1/messages")
    async def messages(request: Request) -> Response:
        return await proxy(request, Route.MESSAGES)

    @app.post("/v1/messages/count_tokens")
    async def count_tokens(request: Request) -> Response:
        return await proxy(request, Route.COUNT_TOKENS)

    @app.get("/healthz")
    async def healthz() -> Response:
        if healthy():
            return JSONResponse({"status": "ok"})
        return JSONResponse({"status": "unavailable"}, status_code=503)

    @app.api_route(
        "/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"]
    )
    async def not_found(path: str) -> Response:
        return render_error(NotFound("unknown endpoint", gateway_code="unknown_endpoint"), None)

    return app


async def _read_body(request: Request, limit: int) -> tuple[bytes, bool]:
    """Read at most ``limit`` bytes; past it, keep draining without buffering (413 later)."""
    chunks: list[bytes] = []
    size = 0
    too_large = False
    async for chunk in request.stream():
        size += len(chunk)
        if size > limit:
            too_large = True
            chunks.clear()
            continue
        chunks.append(chunk)
    return (b"" if too_large else b"".join(chunks)), too_large

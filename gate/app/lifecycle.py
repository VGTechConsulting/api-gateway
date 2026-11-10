"""The request lifecycle: identify -> parse -> check policy -> detect -> route -> relay -> settle.

This module reads top to bottom in that order. Each step is one
method named after it, and each failure response sits next to its check.
"""

import asyncio
import contextlib
import hashlib
import os
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any

from gate.app.admission import Admission, AdmissionController, AdmissionDenied
from gate.app.ports import (
    FailureKind,
    KeyRepository,
    MonotonicClock,
    ParentRow,
    RequestLog,
    Upstream,
    UpstreamFailure,
    UpstreamReadError,
    UpstreamResponse,
    UsageRepository,
    WallClock,
)
from gate.app.relay import Relay, RelayResult, run_shielded
from gate.app.routing import Candidate, Router, fallback_eligible_status, parse_retry_after
from gate.domain import detection
from gate.domain.accounting import (
    AttemptEvidence,
    AttemptSummaryInput,
    Settlement,
    settle_generation,
    settle_unmetered,
    summarize,
)
from gate.domain.config import GatewayConfig
from gate.domain.errors import (
    AuthenticationFailed,
    ClientGone,
    DetectorUnavailable,
    GateError,
    InternalError,
    InvalidRequest,
    NotFound,
    PayloadTooLarge,
    PermissionDenied,
    RateLimited,
    ShuttingDown,
    StorageUnavailable,
    UpstreamConnectionFailed,
    UpstreamTimedOut,
)
from gate.domain.models import (
    ApiFormat,
    AttemptRecord,
    Budget,
    DetectMode,
    PolicyEvent,
    Principal,
    RequestState,
    Route,
    TerminationReason,
)
from gate.domain.money import Micros
from gate.domain.policy import apply_modes, check_allowlist, event
from gate.domain.pricing import Estimate, estimate
from gate.domain.request import ParsedRequest, parse_request, rewrite_model
from gate.providers.base import Headers, ParseResult, ProviderFormat, UsageParser

KEY_PREFIX = "gk_"

RELAYED_RESPONSE_HEADERS = frozenset({"content-type", "retry-after", "request-id", "x-request-id"})
RELAYED_RESPONSE_PREFIXES = ("anthropic-ratelimit-", "x-ratelimit-")

FALLBACK_TOKEN_CODES = frozenset({"tpm_exceeded", "token_estimate_exceeds_limit"})
# Cancellation of ``handle``: the attempt's cleanup is shielded and the error re-raised.
_CANCELLED = (TerminationReason.SHUTDOWN, TerminationReason.CLIENT_DISCONNECT)


def hash_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode()).hexdigest()


# --- request and reply values ------------------------------------------------------------


@dataclass(frozen=True)
class IncomingRequest:
    route: Route
    headers: Headers  # lower-case names, in arrival order
    body: bytes
    body_too_large: bool
    received_mono: float
    # True once the client has gone (http.disconnect). The adapter sets it before cancelling
    # ``handle``, so the cancellation is recorded as a client disconnect, not a shutdown.
    client_gone: Callable[[], bool] = lambda: False


@dataclass(frozen=True)
class ErrorReply:
    error: GateError
    request_id: str
    would_deny: tuple[str, ...]


@dataclass(frozen=True)
class StreamReply:
    request_id: str
    relay: Relay


Reply = ErrorReply | StreamReply


@dataclass
class Pending:
    """An admitted attempt that has not been dispatched yet."""

    candidate: Candidate
    estimate: Estimate | None
    admission: Admission


@dataclass
class AttemptRun:
    """A dispatched attempt awaiting settlement."""

    attempt_no: int
    candidate: Candidate
    estimate: Estimate | None
    admission: Admission
    ts_start: datetime
    dispatched_mono: float


@dataclass
class RequestContext:
    request_id: str
    route: Route
    ts_start: datetime
    received_mono: float
    session_id: str | None
    client_gone: Callable[[], bool] = lambda: False
    principal: Principal | None = None
    parsed: ParsedRequest | None = None
    events: list[PolicyEvent] = field(default_factory=list)
    would_deny: list[str] = field(default_factory=list)
    detections: dict[str, int] | None = None
    attempts: list[AttemptRecord] = field(default_factory=list)
    pending: Pending | None = None
    opened: bool = False
    finalised: bool = False
    first_dispatch_mono: float | None = None

    def record(self, events: Sequence[PolicyEvent], would_deny: Sequence[str] = ()) -> None:
        self.events.extend(events)
        self.would_deny.extend(would_deny)

    @property
    def would_deny_codes(self) -> tuple[str, ...]:
        return tuple(sorted(set(self.would_deny)))


@dataclass(frozen=True)
class FinalResponse:
    """What the client finally received, for the parent row."""

    http_status: int
    run: AttemptRun | None = None
    upstream_status: int | None = None
    upstream_model: str | None = None
    stream_complete: bool | None = None
    termination: TerminationReason | None = None
    client_disconnected: bool = False
    ttfb_ms: float | None = None
    gate_overhead_ms: float | None = None


# --- the use case ------------------------------------------------------------------------


class HandleRequest:
    def __init__(
        self,
        *,
        config: GatewayConfig,
        keys: KeyRepository,
        usage: UsageRepository,
        upstream: Upstream,
        request_log: RequestLog,
        admission: AdmissionController,
        router: Router,
        providers: Mapping[ApiFormat, ProviderFormat],
        wall: WallClock,
        mono: MonotonicClock,
        env: Mapping[str, str] | None = None,
        shutting_down: Callable[[], bool] = lambda: False,
    ) -> None:
        self._config = config
        self._keys = keys
        self._usage = usage
        self._upstream = upstream
        self._log = request_log
        self._admission = admission
        self._router = router
        self._providers = providers
        self._wall = wall
        self._mono = mono
        self._env = env if env is not None else os.environ
        self._shutting_down = shutting_down

    async def handle(self, incoming: IncomingRequest) -> Reply:
        ctx = RequestContext(
            request_id=str(uuid.uuid4()),
            route=incoming.route,
            ts_start=self._wall.now(),
            received_mono=incoming.received_mono,
            session_id=self._session_id(incoming.headers),
            client_gone=incoming.client_gone,
        )
        try:
            ctx.principal = self.identify(incoming.headers)
            if incoming.body_too_large:
                raise PayloadTooLarge("request body exceeds the configured limit")
            ctx.parsed = self.parse(incoming)
            candidates = self.check_policy(ctx, ctx.principal, ctx.parsed, incoming.headers)
            first, *fallbacks = self.select(candidates)
            ctx.pending = await self.admit(ctx, first, first_admission=True)
            self.detect(ctx, ctx.parsed)
            return await self.dispatch(ctx, incoming, fallbacks)
        except GateError as err:
            ctx.events.append(event("gate_error", gateway_code=err.gateway_code, status=err.status))
            await self._release_pending(ctx)
            await self._finalise(
                ctx,
                FinalResponse(http_status=err.status, client_disconnected=ctx.client_gone()),
            )
            return ErrorReply(err, ctx.request_id, ctx.would_deny_codes)
        except asyncio.CancelledError:
            await run_shielded(self._abandon(ctx), self._config.limits.cleanup_timeout_s)
            raise
        except Exception as exc:
            ctx.events.append(event("internal_error", error=type(exc).__name__))
            await self._release_pending(ctx)
            await self._finalise(ctx, FinalResponse(http_status=500))
            return ErrorReply(InternalError("internal gateway error"), ctx.request_id, ())

    # --- 5.1 identify -------------------------------------------------------------------

    def identify(self, headers: Headers) -> Principal:
        credential = _header(headers, "x-api-key")
        if credential is None:
            auth = _header(headers, "authorization")
            if auth is not None and auth[:7].lower() == "bearer ":
                credential = auth[7:].strip()
        if not credential or not credential.startswith(KEY_PREFIX):
            raise AuthenticationFailed("missing or invalid gateway key")
        try:
            record = self._keys.find_active_by_hash(hash_key(credential))
        except Exception as exc:
            raise StorageUnavailable("key storage unavailable") from exc
        if record is None:
            raise AuthenticationFailed("missing or invalid gateway key")
        team = self._config.team(record.team)
        return Principal(
            key_id=record.id,
            prefix=record.prefix,
            owner=record.owner,
            team=record.team,
            app=record.app,
            env=record.env,
            allowed_models=frozenset(record.allowed_models),
            budget=(
                Budget(Micros(record.budget_micros), record.period)
                if record.budget_micros is not None
                else None
            ),
            team_budget=team.budget if team is not None else None,
            tpm=record.tpm if record.tpm is not None else self._config.default_tpm,
            rpm=record.rpm if record.rpm is not None else self._config.default_rpm,
        )

    # --- 5.2 parse ----------------------------------------------------------------------

    def parse(self, incoming: IncomingRequest) -> ParsedRequest:
        return parse_request(incoming.body, incoming.route)

    # --- 5.3 check policy (stateless checks) ---------------------------------------------

    def check_policy(
        self, ctx: RequestContext, principal: Principal, parsed: ParsedRequest, headers: Headers
    ) -> list[Candidate]:
        modes = self._config.enforcement_for(principal.team)
        verdict = apply_modes([check_allowlist(principal, parsed.model)], modes)
        ctx.record(verdict.events, verdict.would_deny)
        if verdict.denial is not None:
            raise verdict.denial

        if not self._router.routed(parsed.model):
            raise NotFound("model is not routed by this gateway", gateway_code="model_not_routed")
        candidates = self._router.candidates(parsed.model)
        if not ctx.route.metered:
            return candidates

        if any(c.price is None for c in candidates):
            raise PermissionDenied("model has no configured price", gateway_code="model_not_priced")
        features: list[str] = []
        for fmt in {c.deployment.format for c in candidates}:
            features += self._providers[fmt].unsupported_pricing_features(parsed.tree, headers)
        if features:
            for feature in sorted(set(features)):
                ctx.events.append(event("pricing_not_supported", feature=feature))
            raise PermissionDenied(
                "request selects pricing this gateway cannot meter",
                gateway_code="pricing_not_supported",
            )
        return candidates

    # --- 5.6 route: choose deployments not in cooldown ------------------------------------

    def select(self, candidates: list[Candidate]) -> list[Candidate]:
        now = self._mono.now()
        available = self._router.available(candidates, now)
        if not available:
            wait = self._router.cooldown_remaining(candidates, now)
            raise RateLimited(
                "all deployments for this model are cooling down",
                gateway_code="deployments_cooling_down",
                retry_after_s=max(1, int(wait + 0.999)),
            )
        return available

    # --- 5.3 check policy (atomic rate and budget admission) -----------------------------

    async def admit(
        self, ctx: RequestContext, candidate: Candidate, *, first_admission: bool
    ) -> Pending:
        assert ctx.principal is not None and ctx.parsed is not None
        est = None
        if ctx.route.metered:
            assert candidate.price is not None and ctx.parsed.output_limit is not None
            est = estimate(ctx.parsed.body_size, ctx.parsed.output_limit, candidate.price)
        modes = self._config.enforcement_for(ctx.principal.team)
        try:
            admission = await self._admission.admit(
                ctx.principal,
                modes,
                est,
                first=first_admission,
                stage="admission" if first_admission else "fallback",
            )
        except AdmissionDenied as denied:
            ctx.record(denied.events, denied.would_deny)
            raise denied.error from None
        ctx.record(admission.events, admission.would_deny)
        return Pending(candidate, est, admission)

    # --- 5.5 detect ---------------------------------------------------------------------

    def detect(self, ctx: RequestContext, parsed: ParsedRequest) -> None:
        modes = {
            name: mode
            for name, mode in self._config.detect.categories.items()
            if mode is not DetectMode.OFF
        }
        if not modes:
            return
        try:
            counts = detection.scan(parsed.tree, tuple(modes))
        except Exception as exc:
            ctx.events.append(event("detector_error", error=type(exc).__name__))
            if self._config.detect.on_error_open:
                return
            raise DetectorUnavailable("content inspection failed") from None
        ctx.detections = counts
        blocked = sorted(name for name in counts if modes[name] is DetectMode.BLOCK)
        if blocked:
            ctx.events.append(event("content_blocked", category=blocked[0]))
            raise InvalidRequest(
                f"request blocked by content policy: {blocked[0]}",
                gateway_code="content_blocked",
            )

    # --- 5.6 route and dispatch, with fallback -------------------------------------------

    async def dispatch(
        self, ctx: RequestContext, incoming: IncomingRequest, fallbacks: list[Candidate]
    ) -> StreamReply:
        self._open_parent(ctx)
        while True:
            assert ctx.pending is not None
            candidate = ctx.pending.candidate
            provider = self._providers[candidate.deployment.format]
            # Build the upstream request while the reservation is still pending: a failure
            # here dispatches nothing, so ``handle`` releases it as a local rejection (5.8).
            url = self._upstream_url(candidate, ctx.route)
            headers = provider.upstream_headers(incoming.headers, self._api_key(candidate))
            body = self._upstream_body(candidate, incoming.body)
            run = self._start_attempt(ctx)
            # From here the attempt is dispatched: every exit must settle it exactly once.
            try:
                response = await self._upstream.send(url, headers, body)
            except UpstreamFailure as failure:
                await run_shielded(
                    self._settle_failure(ctx, run, failure),
                    self._config.limits.cleanup_timeout_s,
                )
                next_candidate = self._next_fallback(ctx, fallbacks)
                if next_candidate is None:
                    if failure.kind is FailureKind.TIMEOUT:
                        raise UpstreamTimedOut("upstream timed out") from None
                    raise UpstreamConnectionFailed("could not reach upstream") from None
                await self._admit_fallback(ctx, next_candidate)
                continue
            except asyncio.CancelledError:
                await run_shielded(
                    self._settle_lost(ctx, run, self._cancel_reason(ctx)),
                    self._config.limits.cleanup_timeout_s,
                )
                raise
            except Exception as exc:
                await run_shielded(
                    self._settle_unexpected(ctx, run, exc, None),
                    self._config.limits.cleanup_timeout_s,
                )
                raise

            try:
                self._note_cooldown(run, response)
                next_candidate = (
                    self._next_fallback(ctx, fallbacks)
                    if fallback_eligible_status(response.status)
                    else None
                )
                if next_candidate is None:
                    return self._commit(ctx, run, response, provider)  # the relay now settles
            except Exception as exc:
                await run_shielded(
                    self._settle_unexpected(ctx, run, exc, response),
                    self._config.limits.cleanup_timeout_s,
                )
                raise
            # Nothing has been committed to the client: settle this attempt, then try the next.
            await self._drain_and_settle(ctx, run, response, provider)
            await self._admit_fallback(ctx, next_candidate)

    def _open_parent(self, ctx: RequestContext) -> None:
        try:
            self._usage.open_request(self._parent_row(ctx, RequestState.OPEN, None))
        except Exception as exc:
            raise StorageUnavailable("usage storage unavailable") from exc
        ctx.opened = True

    def _start_attempt(self, ctx: RequestContext) -> AttemptRun:
        assert ctx.pending is not None
        pending, ctx.pending = ctx.pending, None
        now = self._mono.now()
        if ctx.first_dispatch_mono is None:
            ctx.first_dispatch_mono = now
        return AttemptRun(
            attempt_no=len(ctx.attempts) + 1,
            candidate=pending.candidate,
            estimate=pending.estimate,
            admission=pending.admission,
            ts_start=self._wall.now(),
            dispatched_mono=now,
        )

    def _next_fallback(self, ctx: RequestContext, fallbacks: list[Candidate]) -> Candidate | None:
        """The next untried deployment not in cooldown; none after shutdown or disconnect."""
        if self._shutting_down() or ctx.client_gone():
            return None
        now = self._mono.now()
        while fallbacks:
            candidate = fallbacks.pop(0)
            if self._router.available([candidate], now):
                return candidate
        return None

    async def _admit_fallback(self, ctx: RequestContext, candidate: Candidate) -> None:
        """Fresh admission with the candidate's own estimate; the RPM entry is not repeated.

        The candidate was chosen before the failed attempt was drained and settled, and
        admission may wait on the shared lock: shutdown or a disconnect can begin during
        either, so both are rechecked before admitting and again before dispatch (5.6).
        A halt after admission leaves ``ctx.pending`` set for ``handle`` to release.
        """
        self._halt_fallback_if_stopping(ctx, candidate)
        try:
            ctx.pending = await self.admit(ctx, candidate, first_admission=False)
        except GateError as err:
            if err.gateway_code == "budget_exceeded":
                ctx.events.append(event("fallback_budget_blocked", deployment=candidate.name))
            elif err.gateway_code in FALLBACK_TOKEN_CODES:
                ctx.events.append(event("fallback_token_blocked", deployment=candidate.name))
            raise
        self._halt_fallback_if_stopping(ctx, candidate)
        ctx.events.append(event("fallback", deployment=candidate.name))

    def _halt_fallback_if_stopping(self, ctx: RequestContext, candidate: Candidate) -> None:
        if ctx.client_gone():
            ctx.events.append(
                event("fallback_halted", deployment=candidate.name, reason="client_disconnect")
            )
            raise ClientGone("client disconnected before fallback")
        if self._shutting_down():
            ctx.events.append(
                event("fallback_halted", deployment=candidate.name, reason="shutdown")
            )
            raise ShuttingDown("gateway is shutting down")

    def _note_cooldown(self, run: AttemptRun, response: UpstreamResponse) -> None:
        if response.status not in (429, 529):
            return
        seconds = parse_retry_after(_header(response.headers, "retry-after"), self._wall.now())
        if seconds is not None:
            self._router.cool_down(run.candidate.name, seconds, self._mono.now())

    async def _drain_and_settle(
        self,
        ctx: RequestContext,
        run: AttemptRun,
        response: UpstreamResponse,
        provider: ProviderFormat,
    ) -> None:
        """Read, close and settle an uncommitted failed response before any fallback.

        The attempt is dispatched, so it must settle exactly once whatever stops the read:
        on cancellation, close and settlement still run (shielded, bounded) from the bytes
        already received, as the relay does for committed responses (5.8).
        """
        parser = provider.new_parser(
            content_type=_header(response.headers, "content-type") or "",
            http_status=response.status,
            max_buffer_bytes=self._config.limits.max_buffer_bytes,
        )
        eof = False
        interrupted: TerminationReason | None = None
        try:
            async with asyncio.timeout(self._config.limits.connect_timeout_s):
                async for chunk in response.iter_raw():
                    parser.feed(chunk)
            eof = True
        except (UpstreamReadError, TimeoutError):
            interrupted = TerminationReason.UPSTREAM_DISCONNECT
        except asyncio.CancelledError:
            interrupted = self._cancel_reason(ctx)
            raise
        finally:
            await run_shielded(
                self._close_and_settle_drained(
                    ctx, run, response, provider, parser, eof, interrupted
                ),
                self._config.limits.cleanup_timeout_s,
            )

    async def _close_and_settle_drained(
        self,
        ctx: RequestContext,
        run: AttemptRun,
        response: UpstreamResponse,
        provider: ProviderFormat,
        parser: UsageParser,
        eof: bool,
        interrupted: TerminationReason | None,
    ) -> None:
        with contextlib.suppress(TimeoutError, OSError, UpstreamReadError):
            async with asyncio.timeout(self._config.limits.cleanup_timeout_s):
                await response.aclose()
        result = RelayResult(
            parse=parser.result(eof=eof),
            eof=eof,
            interrupted_by=interrupted,
            client_disconnected=interrupted is TerminationReason.CLIENT_DISCONNECT,
            first_chunk_received=None,
            first_chunk_sent=None,
            ended=self._mono.now(),
        )
        try:
            await self._settle_response(
                ctx, run, provider, response.status, result, _termination(response.status, result)
            )
        except StorageUnavailable:
            if interrupted not in _CANCELLED:
                raise  # fail closed: no fallback after a failed settlement
            # Cancelled: the CancelledError must keep propagating, not be replaced by this.
            ctx.events.append(event("settlement_failed", attempt_no=run.attempt_no))

    # --- 5.7 relay ----------------------------------------------------------------------

    def _commit(
        self,
        ctx: RequestContext,
        run: AttemptRun,
        response: UpstreamResponse,
        provider: ProviderFormat,
    ) -> StreamReply:
        parser = provider.new_parser(
            content_type=_header(response.headers, "content-type") or "",
            http_status=response.status,
            max_buffer_bytes=self._config.limits.max_buffer_bytes,
        )

        async def finish(result: RelayResult) -> None:
            await self._finish_committed(ctx, run, provider, response.status, result)

        relay = Relay(
            status=response.status,
            headers=self._response_headers(ctx, response.headers),
            response=response,
            parser=parser,
            finish=finish,
            mono=self._mono,
            cleanup_timeout_s=self._config.limits.cleanup_timeout_s,
        )
        return StreamReply(ctx.request_id, relay)

    def _response_headers(self, ctx: RequestContext, upstream: Headers) -> list[tuple[str, str]]:
        headers = [
            (name, value)
            for name, value in upstream
            if name.lower() in RELAYED_RESPONSE_HEADERS
            or name.lower().startswith(RELAYED_RESPONSE_PREFIXES)
        ]
        headers.append(("x-api-gateway-request-id", ctx.request_id))
        if ctx.would_deny_codes:
            headers.append(("x-api-gateway-would-deny", ",".join(ctx.would_deny_codes)))
        return headers

    async def _finish_committed(
        self,
        ctx: RequestContext,
        run: AttemptRun,
        provider: ProviderFormat,
        status: int,
        result: RelayResult,
    ) -> None:
        termination = _termination(status, result)
        try:
            await self._settle_response(ctx, run, provider, status, result, termination)
        except StorageUnavailable:
            ctx.events.append(event("settlement_failed", attempt_no=run.attempt_no))
        first_sent = result.first_chunk_sent
        overhead = None
        if first_sent is not None and result.first_chunk_received is not None:
            assert ctx.first_dispatch_mono is not None
            overhead = (ctx.first_dispatch_mono - ctx.received_mono) + (
                first_sent - result.first_chunk_received
            )
        await self._finalise(
            ctx,
            FinalResponse(
                http_status=status,
                run=run,
                upstream_status=status,
                upstream_model=result.parse.upstream_model,
                stream_complete=result.parse.stream_complete,
                termination=termination,
                client_disconnected=result.client_disconnected,
                ttfb_ms=_ms(first_sent - ctx.received_mono) if first_sent is not None else None,
                gate_overhead_ms=_ms(overhead) if overhead is not None else None,
            ),
        )

    # --- 5.8 settle ---------------------------------------------------------------------

    async def _settle_failure(
        self, ctx: RequestContext, run: AttemptRun, failure: UpstreamFailure
    ) -> None:
        termination = (
            TerminationReason.UPSTREAM_TIMEOUT
            if failure.kind is FailureKind.TIMEOUT
            else TerminationReason.CONNECTION_ERROR
        )
        evidence = AttemptEvidence(not_sent=failure.not_sent)
        await self._settle(
            ctx,
            run,
            evidence,
            termination=termination,
            upstream_status=None,
            parse=None,
            client_disconnected=False,
        )

    async def _settle_lost(
        self, ctx: RequestContext, run: AttemptRun, termination: TerminationReason
    ) -> None:
        """An attempt cut off before response headers arrived (cancelled): usage unknown."""
        try:
            await self._settle(
                ctx,
                run,
                AttemptEvidence(),
                termination=termination,
                upstream_status=None,
                parse=None,
                client_disconnected=termination is TerminationReason.CLIENT_DISCONNECT,
            )
        except StorageUnavailable:
            ctx.events.append(event("settlement_failed", attempt_no=run.attempt_no))

    async def _settle_unexpected(
        self,
        ctx: RequestContext,
        run: AttemptRun,
        exc: Exception,
        response: UpstreamResponse | None,
    ) -> None:
        """A dispatched attempt cut short by an unexpected gate-side error.

        A generic exception is not proof that nothing was sent, so the attempt settles
        as ``unknown`` on its reservation (5.8). The caller re-raises the original error;
        a settlement failure replaces it, failing closed with 503.
        """
        ctx.events.append(
            event("dispatch_error", error=type(exc).__name__, attempt_no=run.attempt_no)
        )
        if response is not None:
            with contextlib.suppress(Exception):
                async with asyncio.timeout(self._config.limits.cleanup_timeout_s):
                    await response.aclose()
        await self._settle(
            ctx,
            run,
            AttemptEvidence(),
            termination=TerminationReason.CONNECTION_ERROR,
            upstream_status=response.status if response is not None else None,
            parse=None,
            client_disconnected=False,
        )

    async def _settle_response(
        self,
        ctx: RequestContext,
        run: AttemptRun,
        provider: ProviderFormat,
        status: int,
        result: RelayResult,
        termination: TerminationReason,
    ) -> None:
        parse = result.parse
        extra: tuple[str, ...] = ()
        if parse.upstream_model is not None and 200 <= status < 300:
            extra = self._model_mismatch(ctx, run, parse.upstream_model)
        evidence = AttemptEvidence(
            rejection_basis=provider.non_billable_basis(status, parse) if status >= 400 else None,
            snapshot=parse.snapshot,
            usage_complete=parse.usage_complete,
            usage_source=parse.usage_source,
            extra_unsupported=extra,
        )
        for reason in parse.reasons:
            ctx.events.append(event("usage_diagnostic", reason=reason, attempt_no=run.attempt_no))
        await self._settle(
            ctx,
            run,
            evidence,
            termination=termination,
            upstream_status=status,
            parse=parse,
            client_disconnected=result.client_disconnected,
        )

    def _model_mismatch(
        self, ctx: RequestContext, run: AttemptRun, reported: str
    ) -> tuple[str, ...]:
        assert ctx.parsed is not None
        expected = run.candidate.deployment.upstream_model or ctx.parsed.model
        price = run.candidate.price
        verified = price.response_models if price is not None else frozenset()
        if reported == expected or reported in verified:
            return ()
        ctx.events.append(
            event(
                "upstream_model_mismatch", deployment=run.candidate.name, attempt_no=run.attempt_no
            )
        )
        return ("upstream_model_mismatch",)

    async def _settle(
        self,
        ctx: RequestContext,
        run: AttemptRun,
        evidence: AttemptEvidence,
        *,
        termination: TerminationReason,
        upstream_status: int | None,
        parse: ParseResult | None,
        client_disconnected: bool,
    ) -> None:
        """Classify, debit and persist one attempt; then swap reservations for debits."""
        assert ctx.principal is not None and ctx.parsed is not None
        if run.estimate is None or run.candidate.price is None:
            settlement = settle_unmetered(evidence)
        else:
            settlement = settle_generation(evidence, run.estimate, run.candidate.price)
        record = self._attempt_record(
            ctx, run, settlement, termination, upstream_status, parse, client_disconnected
        )

        def persist(overruns: tuple[PolicyEvent, ...]) -> bool:
            final = replace(record, policy_decisions=record.policy_decisions + overruns)
            parent = self._parent_row(ctx, RequestState.OPEN, None, extra=[final])
            inserted = self._usage.settle_attempt(final, parent)
            if inserted:
                ctx.attempts.append(final)
            return inserted

        inserted = await self._admission.settle(
            ctx.principal,
            run.admission,
            token_debit=settlement.token_debit,
            cost_micros=settlement.cost_micros,
            persist=persist,
        )
        if inserted:
            self._log.write(self._attempt_event(ctx, ctx.attempts[-1]))

    def _attempt_record(
        self,
        ctx: RequestContext,
        run: AttemptRun,
        s: Settlement,
        termination: TerminationReason,
        upstream_status: int | None,
        parse: ParseResult | None,
        client_disconnected: bool,
    ) -> AttemptRecord:
        assert ctx.parsed is not None
        deployment = run.candidate.deployment
        est = run.estimate
        return AttemptRecord(
            request_id=ctx.request_id,
            attempt_no=run.attempt_no,
            ts_admitted=run.admission.ts_admitted,
            ts_start=run.ts_start,
            ts_end=self._wall.now(),
            key_period_id=run.admission.key_period_id if est is not None else None,
            team_period_id=run.admission.team_period_id if est is not None else None,
            deployment=deployment.name,
            provider=deployment.provider,
            region=deployment.region,
            model=ctx.parsed.model,
            upstream_model=parse.upstream_model if parse is not None else None,
            upstream_status=upstream_status,
            outcome=s.outcome,
            billing_basis=s.billing_basis,
            accounting_kind=s.accounting_kind,
            pricing_mode=run.candidate.price.pricing_mode
            if est is not None and run.candidate.price
            else None,
            stream=ctx.parsed.stream,
            stream_complete=parse.stream_complete if parse is not None else None,
            usage_complete=s.usage_complete,
            termination_reason=termination,
            counts=s.counts,
            estimate_tokens=est.tokens if est is not None else 0,
            observed_total_tokens=s.observed_total_tokens,
            token_debit=s.token_debit,
            token_debit_source=s.token_debit_source,
            observed_cost_micros=s.observed_cost_micros,
            priced_cost_micros=s.priced_cost_micros,
            estimate_micros=est.micros if est is not None else Micros(0),
            cost_micros=s.cost_micros,
            usage_source=s.usage_source,
            pricing_status=s.pricing_status,
            client_disconnected=client_disconnected,
            policy_decisions=s.events,
        )

    # --- finalise the parent ------------------------------------------------------------

    async def _release_pending(self, ctx: RequestContext) -> None:
        if ctx.pending is not None:
            pending, ctx.pending = ctx.pending, None
            await self._admission.release(pending.admission)

    async def _abandon(self, ctx: RequestContext) -> None:
        await self._release_pending(ctx)
        await self._finalise(
            ctx, FinalResponse(http_status=499, client_disconnected=ctx.client_gone())
        )

    def _cancel_reason(self, ctx: RequestContext) -> TerminationReason:
        """Why ``handle`` was cancelled: the adapter saw the client go, or the server stopping."""
        if ctx.client_gone():
            return TerminationReason.CLIENT_DISCONNECT
        return TerminationReason.SHUTDOWN

    async def _finalise(self, ctx: RequestContext, final: FinalResponse) -> None:
        """Write the single parent row and request event. Adds no debit."""
        if ctx.finalised:
            return
        ctx.finalised = True
        row = self._parent_row(ctx, RequestState.FINISHED, final)
        try:
            self._usage.finish_request(row)
        except Exception as exc:
            ctx.events.append(event("finalise_failed", error=type(exc).__name__))
        self._log.write(self._request_event(row))

    def _parent_row(
        self,
        ctx: RequestContext,
        state: RequestState,
        final: FinalResponse | None,
        extra: Sequence[AttemptRecord] = (),
    ) -> ParentRow:
        principal, parsed = ctx.principal, ctx.parsed
        attempts = [*ctx.attempts, *extra]
        totals = summarize([_summary_input(a) for a in attempts]) if attempts or final else None
        run = final.run if final is not None else None
        last = attempts[-1] if attempts else None
        now = self._mono.now()
        return ParentRow(
            request_id=ctx.request_id,
            request_state=state,
            ts_start=ctx.ts_start,
            ts_end=self._wall.now() if final is not None else None,
            key_id=principal.key_id if principal else None,
            key_prefix=principal.prefix if principal else None,
            owner=principal.owner if principal else None,
            team=principal.team if principal else None,
            app=principal.app if principal else None,
            env=principal.env.value if principal else None,
            session_id=ctx.session_id,
            route=ctx.route.value,
            model=parsed.model if parsed else None,
            stream=parsed.stream if parsed else None,
            totals=totals,
            upstream_model=final.upstream_model if final else None,
            deployment=(run.candidate.name if run else (last.deployment if last else None)),
            provider=(
                run.candidate.deployment.provider if run else (last.provider if last else None)
            ),
            region=(run.candidate.deployment.region if run else (last.region if last else None)),
            http_status=final.http_status if final else None,
            upstream_status=(
                final.upstream_status if final and run else (last.upstream_status if last else None)
            ),
            stream_complete=final.stream_complete if final else None,
            termination_reason=(
                final.termination.value
                if final and final.termination
                else (last.termination_reason.value if last else None)
            ),
            ttfb_ms=final.ttfb_ms if final else None,
            latency_ms=_ms(now - ctx.received_mono) if final else None,
            gate_overhead_ms=final.gate_overhead_ms if final else None,
            client_disconnected=final.client_disconnected if final else False,
            policy_decisions=tuple(ctx.events),
            would_deny=ctx.would_deny_codes,
            detections=ctx.detections,
        )

    # --- log events ---------------------------------------------------------------------

    def _attempt_event(self, ctx: RequestContext, a: AttemptRecord) -> dict[str, Any]:
        p = ctx.principal
        return {
            "event_type": "attempt_settled",
            "request_id": a.request_id,
            "attempt_no": a.attempt_no,
            "owner": p.owner if p else None,
            "team": p.team if p else None,
            "app": p.app if p else None,
            "key_prefix": p.prefix if p else None,
            "deployment": a.deployment,
            "provider": a.provider,
            "model": a.model,
            "upstream_status": a.upstream_status,
            "outcome": a.outcome.value,
            "accounting_kind": a.accounting_kind.value,
            "usage_source": a.usage_source.value,
            "pricing_status": a.pricing_status.value,
            "termination_reason": a.termination_reason.value,
            "input_tokens": a.counts.input,
            "output_tokens": a.counts.output,
            "cache_read_tokens": a.counts.cache_read,
            "cache_write_5m_tokens": a.counts.cache_write_5m,
            "cache_write_1h_tokens": a.counts.cache_write_1h,
            "token_debit": a.token_debit,
            "cost_micros": a.cost_micros,
            "estimate_micros": a.estimate_micros,
            "key_period_id": a.key_period_id,
            "team_period_id": a.team_period_id,
        }

    def _request_event(self, row: ParentRow) -> dict[str, Any]:
        t = row.totals
        return {
            "event_type": "request_finished",
            "request_id": row.request_id,
            "ts_start": row.ts_start.isoformat(),
            "owner": row.owner,
            "team": row.team,
            "app": row.app,
            "key_prefix": row.key_prefix,
            "session_id": row.session_id,
            "route": row.route,
            "model": row.model,
            "http_status": row.http_status,
            "attempt_count": t.attempt_count if t else 0,
            "summary_cost_micros": t.cost_micros if t else 0,
            "accounting_kind": t.accounting_kind.value if t else None,
            "latency_ms": row.latency_ms,
            "gate_overhead_ms": row.gate_overhead_ms,
            "would_deny": list(row.would_deny),
            "detections": row.detections,
            "gate_error": next(
                (
                    e.get("gateway_code")
                    for e in row.policy_decisions
                    if e.get("event") == "gate_error"
                ),
                None,
            ),
        }

    # --- helpers ------------------------------------------------------------------------

    def _session_id(self, headers: Headers) -> str | None:
        for name in self._config.session_headers:
            value = _header(headers, name)
            if value:
                return value[:200]
        return None

    def _api_key(self, candidate: Candidate) -> str:
        return self._env.get(candidate.deployment.api_key_env, "")

    def _upstream_url(self, candidate: Candidate, route: Route) -> str:
        return candidate.deployment.base_url.rstrip("/") + route.value

    def _upstream_body(self, candidate: Candidate, body: bytes) -> bytes:
        upstream_model = candidate.deployment.upstream_model
        return rewrite_model(body, upstream_model) if upstream_model else body


def _header(headers: Headers, name: str) -> str | None:
    for key, value in headers:
        if key.lower() == name:
            return value
    return None


def _ms(seconds: float) -> float:
    return round(seconds * 1000, 3)


def _termination(status: int, result: RelayResult) -> TerminationReason:
    if result.interrupted_by is not None:
        return result.interrupted_by
    if status >= 400:
        return TerminationReason.PROVIDER_REJECTION
    if result.parse.stream_error:
        return TerminationReason.STREAM_ERROR
    if result.parse.streamed and not result.parse.stream_complete:
        return TerminationReason.UPSTREAM_DISCONNECT
    return TerminationReason.NORMAL


def _summary_input(a: AttemptRecord) -> AttemptSummaryInput:
    return AttemptSummaryInput(
        accounting_kind=a.accounting_kind,
        generation=a.generation,
        counts=a.counts,
        observed_cost_micros=a.observed_cost_micros,
        priced_cost_micros=a.priced_cost_micros,
        cost_micros=a.cost_micros,
        estimate_micros=a.estimate_micros,
        usage_source=a.usage_source,
        usage_complete=a.usage_complete,
        pricing_status=a.pricing_status,
    )

"""Request lifecycle against in-memory fakes: error paths, fallback, interrupted streams.

No network, database or real clock: upstream responses are scripted and the
relay is pumped by the test acting as the client.
"""

import asyncio
import hashlib
from collections.abc import AsyncIterator
from dataclasses import replace
from typing import Any, cast

import pytest

from gate.app.lifecycle import ErrorReply, StreamReply
from gate.app.ports import ParentRow, UsageRepository
from gate.app.reports import Reports
from gate.domain.ledger import BucketKey, ScopeType
from gate.domain.models import (
    AccountingKind,
    DetectMode,
    EnforcementMode,
    Outcome,
    Policy,
    RequestState,
    TerminationReason,
    TokenDebitSource,
    UsageSource,
)
from tests.app.fakes import (
    RAW_KEY,
    Gate,
    MemoryUsage,
    ScriptedResponse,
    collect,
    error_response,
    key_record,
    make_config,
    make_gate,
    not_sent,
    request,
    stream_chunks,
)


def run(coro):  # type: ignore[no-untyped-def]
    return asyncio.run(coro)


def body_size() -> int:
    return len(request().body)


# --- happy path --------------------------------------------------------------------------


def test_stream_is_relayed_byte_for_byte_and_settled_once() -> None:
    gate = make_gate()
    chunks = stream_chunks(
        input_tokens=100, output_tokens=50, cache_5m=10, cache_1h=20, cache_read=5
    )
    gate.upstream.add("https://dep-a.example", ScriptedResponse(chunks=chunks))

    async def scenario() -> bytes:
        reply = await gate.handler.handle(request())
        assert isinstance(reply, StreamReply)
        return await collect(reply.relay)

    relayed = run(scenario())
    assert hashlib.sha256(relayed).digest() == hashlib.sha256(b"".join(chunks)).digest()

    [attempt] = gate.usage.attempt_list()
    assert attempt.outcome is Outcome.COMPLETE
    assert attempt.accounting_kind is AccountingKind.CALCULATED
    # 100*3 + 50*15 + 5*0.3 + 10*3.75 + 20*6 = 300+750+1.5+37.5+120 = 1209 micro-dollars
    assert attempt.cost_micros == 1209
    assert attempt.token_debit == 185 and attempt.token_debit_source is TokenDebitSource.REPORTED
    parent = gate.usage.only_parent()
    assert parent.request_state is RequestState.FINISHED
    assert parent.totals is not None and parent.totals.cost_micros == 1209
    assert gate.admission.reserved_tokens("key-1") == 0
    assert [e["event_type"] for e in gate.log.events] == ["attempt_settled", "request_finished"]


def test_client_credentials_are_never_forwarded() -> None:
    gate = make_gate()
    gate.upstream.add("https://dep-a.example", ScriptedResponse(chunks=stream_chunks()))
    incoming = request(headers=[("authorization", "Bearer secret"), ("anthropic-beta", "b1")])

    async def scenario() -> None:
        reply = await gate.handler.handle(incoming)
        assert isinstance(reply, StreamReply)
        await collect(reply.relay)

    run(scenario())
    _, headers, body = gate.upstream.calls[0]
    assert headers["x-api-key"] == "sk-a"
    assert "authorization" not in headers and headers["anthropic-beta"] == "b1"
    assert body == incoming.body


# --- local denials -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("body", "key", "status", "code"),
    [
        ({"model": "model-a", "max_tokens": 1}, "gk_unknown", 401, "invalid_api_key"),
        ({"model": "model-a"}, RAW_KEY, 400, "invalid_output_limit"),
        ({"model": "other", "max_tokens": 1}, RAW_KEY, 403, "model_not_allowed"),
        (
            {"model": "model-a", "max_tokens": 1, "service_tier": "priority"},
            RAW_KEY,
            403,
            "pricing_not_supported",
        ),
    ],
)
def test_local_denials_never_reach_upstream(body: dict, key: str, status: int, code: str) -> None:  # type: ignore[type-arg]
    gate = make_gate()
    reply = run(gate.handler.handle(request(body, key=key)))
    assert isinstance(reply, ErrorReply)
    assert (reply.error.status, reply.error.gateway_code) == (status, code)
    assert gate.upstream.calls == []
    parent = gate.usage.only_parent()
    assert parent.totals is not None and parent.totals.attempt_count == 0
    assert any(
        e.get("event") == "gate_error" and e["gateway_code"] == code
        for e in parent.policy_decisions
    )


def test_detection_block_releases_reservation_but_keeps_rpm_entry() -> None:
    gate = make_gate(make_config(detect={"private_key": DetectMode.BLOCK}, rpm=1))
    body = {
        "model": "model-a",
        "max_tokens": 10,
        "messages": [{"role": "user", "content": "-----BEGIN " + "PRIVATE KEY-----"}],
    }

    async def scenario() -> None:
        reply = await gate.handler.handle(request(body))
        assert isinstance(reply, ErrorReply) and reply.error.gateway_code == "content_blocked"
        assert gate.admission.reserved_tokens("key-1") == 0
        again = await gate.handler.handle(request())
        assert isinstance(again, ErrorReply) and again.error.gateway_code == "rpm_exceeded"

    run(scenario())
    assert gate.upstream.calls == []
    assert gate.usage.attempts == {}


def test_flagged_detection_records_counts_only() -> None:
    gate = make_gate(make_config(detect={"card_number": DetectMode.FLAG}))
    gate.upstream.add("https://dep-a.example", ScriptedResponse(chunks=stream_chunks()))
    body = {
        "model": "model-a",
        "max_tokens": 10,
        "stream": True,
        "messages": [{"role": "user", "content": "pay 4111 1111 1111 1111"}],
    }

    async def scenario() -> None:
        reply = await gate.handler.handle(request(body))
        assert isinstance(reply, StreamReply)
        await collect(reply.relay)

    run(scenario())
    parent = gate.usage.only_parent()
    assert parent.detections == {"card_number": 1}
    assert "4111" not in repr(parent) and "4111" not in repr(gate.log.events)


# --- observe mode ------------------------------------------------------------------------


def test_observe_mode_admits_and_reports_would_deny() -> None:
    config = make_config(enforcement={p: EnforcementMode.OBSERVE for p in Policy})
    gate = make_gate(config, key_record(models=("something-else",), budget_usd="0.000001"))
    gate.upstream.add("https://dep-a.example", ScriptedResponse(chunks=stream_chunks()))

    async def scenario() -> StreamReply:
        reply = await gate.handler.handle(request())
        assert isinstance(reply, StreamReply)
        await collect(reply.relay)
        return reply

    reply = run(scenario())
    header = dict(reply.relay.headers)["x-api-gateway-would-deny"]
    assert header == "budget_exceeded,model_not_allowed"
    parent = gate.usage.only_parent()
    assert parent.would_deny == ("budget_exceeded", "model_not_allowed")
    [attempt] = gate.usage.attempt_list()
    assert any(e.get("event") == "budget_overrun" for e in attempt.policy_decisions)


def test_observe_mode_never_relaxes_parsing_or_detection() -> None:
    config = make_config(
        enforcement={p: EnforcementMode.OBSERVE for p in Policy},
        detect={"private_key": DetectMode.BLOCK},
    )
    gate = make_gate(config)
    blocked = {"model": "model-a", "max_tokens": 5, "messages": ["-----BEGIN PRIVATE KEY-----"]}
    assert run(gate.handler.handle(request(blocked))).error.gateway_code == "content_blocked"  # type: ignore[union-attr]
    assert run(gate.handler.handle(request({"model": "model-a"}))).error.gateway_code == (  # type: ignore[union-attr]
        "invalid_output_limit"
    )


# --- fallback ----------------------------------------------------------------------------


KEY_MONTH = BucketKey(ScopeType.KEY, "key-1", "month:2026-11")
TEAM_MONTH = BucketKey(ScopeType.TEAM, "platform", "month:2026-11")


def budgeted_gate() -> Gate:
    return make_gate(make_config(team_budget_usd="100"), key_record(budget_usd="10"))


def assert_nothing_reserved(gate: Gate) -> None:
    assert gate.admission.reserved_tokens("key-1") == 0
    assert gate.admission.reserved_money(KEY_MONTH) == 0
    assert gate.admission.reserved_money(TEAM_MONTH) == 0


def test_unforwardable_header_is_rejected_and_releases_reservations() -> None:
    gate = budgeted_gate()
    reply = run(gate.handler.handle(request(headers=[("anthropic-beta", "b1\xff")])))
    assert isinstance(reply, ErrorReply)
    assert (reply.error.status, reply.error.gateway_code) == (400, "invalid_header")
    assert "\xff" not in reply.error.message
    assert gate.upstream.calls == []
    assert gate.usage.attempts == {}
    assert_nothing_reserved(gate)


def test_unexpected_send_error_settles_the_attempt_and_releases_reservations() -> None:
    gate = budgeted_gate()
    gate.upstream.add("https://dep-a.example", RuntimeError("boom"))
    gate.upstream.add("https://dep-b.example", ScriptedResponse(chunks=stream_chunks()))

    reply = run(gate.handler.handle(request()))
    assert isinstance(reply, ErrorReply) and reply.error.status == 500
    assert len(gate.upstream.calls) == 1  # no fallback after a gate-side failure
    [attempt] = gate.usage.attempt_list()
    # A generic exception is not proof that nothing was sent: debit the reservation.
    assert attempt.outcome is Outcome.UNKNOWN
    assert attempt.accounting_kind is AccountingKind.ESTIMATED
    assert attempt.cost_micros == attempt.estimate_micros > 0
    assert attempt.token_debit == attempt.estimate_tokens
    assert attempt.termination_reason is TerminationReason.CONNECTION_ERROR
    assert_nothing_reserved(gate)
    assert gate.admission.spent(KEY_MONTH) == attempt.cost_micros
    parent = gate.usage.only_parent()
    assert parent.request_state is RequestState.FINISHED
    assert any(e.get("event") == "dispatch_error" for e in parent.policy_decisions)


def test_unexpected_error_after_headers_closes_and_settles_the_attempt() -> None:
    gate = budgeted_gate()
    response = ScriptedResponse(chunks=stream_chunks())
    gate.upstream.add("https://dep-a.example", response)

    def broken_commit(*args: object) -> StreamReply:
        raise RuntimeError("boom")

    gate.handler._commit = broken_commit  # type: ignore[method-assign,assignment]
    reply = run(gate.handler.handle(request()))
    assert isinstance(reply, ErrorReply) and reply.error.status == 500
    assert response.closed
    [attempt] = gate.usage.attempt_list()
    assert attempt.outcome is Outcome.UNKNOWN and attempt.upstream_status == 200
    assert_nothing_reserved(gate)


def test_settlement_failure_after_unexpected_send_error_fails_closed() -> None:
    gate = make_gate()
    gate.usage.fail_settle = True
    gate.upstream.add("https://dep-a.example", RuntimeError("boom"))

    reply = run(gate.handler.handle(request()))
    assert isinstance(reply, ErrorReply) and reply.error.status == 503
    assert gate.admission.disabled


def test_fallback_after_529_settles_both_attempts_and_cools_down() -> None:
    gate = make_gate()
    gate.upstream.add("https://dep-a.example", error_response(529, "overloaded_error", "30"))
    gate.upstream.add("https://dep-b.example", ScriptedResponse(chunks=stream_chunks()))

    async def scenario() -> None:
        reply = await gate.handler.handle(request())
        assert isinstance(reply, StreamReply)
        await collect(reply.relay)

    run(scenario())
    first, second = gate.usage.attempt_list()
    assert first.outcome is Outcome.UNKNOWN  # a 529 is not proof of zero cost
    assert first.cost_micros == first.estimate_micros > 0
    assert second.outcome is Outcome.COMPLETE and second.deployment == "dep-b"
    parent = gate.usage.only_parent()
    assert parent.totals is not None
    assert parent.totals.cost_micros == first.cost_micros + second.cost_micros
    assert parent.totals.accounting_kind is AccountingKind.MIXED
    assert parent.deployment == "dep-b"
    assert (
        gate.router.available(gate.router.candidates("model-a"), gate.mono.now())[0].name == "dep-b"
    )


def test_proven_pre_send_failure_debits_zero_then_falls_back() -> None:
    gate = make_gate()
    gate.upstream.add("https://dep-a.example", not_sent())
    gate.upstream.add("https://dep-b.example", ScriptedResponse(chunks=stream_chunks()))

    async def scenario() -> None:
        reply = await gate.handler.handle(request())
        assert isinstance(reply, StreamReply)
        await collect(reply.relay)

    run(scenario())
    first, second = gate.usage.attempt_list()
    assert first.outcome is Outcome.NOT_SENT and first.cost_micros == 0 and first.token_debit == 0
    assert second.outcome is Outcome.COMPLETE
    assert first.usage_source is UsageSource.NOT_APPLICABLE
    parent = gate.usage.only_parent()
    assert parent.totals is not None and parent.totals.accounting_kind is AccountingKind.CALCULATED


def test_fallback_budget_denial_dispatches_nothing_and_keeps_first_debit() -> None:
    # dep-a's estimate fits the budget; dep-b's (10x prices) does not, after dep-a's debit.
    est_a = (body_size() + 3) // 4 * 3 + 100 * 15
    gate = make_gate(None, key_record(budget_usd=f"{(est_a * 3) / 1_000_000:.6f}"))
    gate.upstream.add("https://dep-a.example", error_response(500, "api_error"))

    reply = run(gate.handler.handle(request()))
    assert isinstance(reply, ErrorReply)
    assert (reply.error.status, reply.error.gateway_code) == (403, "budget_exceeded")
    assert len(gate.upstream.calls) == 1
    [attempt] = gate.usage.attempt_list()
    assert attempt.cost_micros == est_a
    parent = gate.usage.only_parent()
    assert any(e.get("event") == "fallback_budget_blocked" for e in parent.policy_decisions)
    assert gate.admission.reserved_tokens("key-1") == 0
    [team] = parent_reports(gate.usage).by("team", since=None, until=None)
    assert team["blocked_requests_by_reason"] == {"budget_exceeded": 1}
    coverage = parent_reports(gate.usage).coverage(since=None, until=None)
    assert coverage["excluded_local_denials"] == 0


def test_upstream_failure_after_attempts_is_not_reported_as_blocked() -> None:
    gate = make_gate()
    gate.upstream.add("https://dep-a.example", not_sent())
    gate.upstream.add("https://dep-b.example", not_sent())

    reply = run(gate.handler.handle(request()))
    assert isinstance(reply, ErrorReply) and reply.error.status >= 500
    [team] = parent_reports(gate.usage).by("team", since=None, until=None)
    assert team["blocked_requests_by_reason"] == {}


class _ParentsOnly:
    """Report view of the in-memory usage store: parent rows only."""

    def __init__(self, usage: MemoryUsage) -> None:
        self._usage = usage

    def attempts(self, *, since: object, until: object) -> list[object]:
        return []

    def requests(self, *, since: object, until: object) -> list[ParentRow]:
        return list(self._usage.requests(since=None, until=None))


def parent_reports(usage: MemoryUsage) -> Reports:
    return Reports(cast(UsageRepository, _ParentsOnly(usage)))


def test_exhausted_candidates_return_the_final_upstream_failure_unchanged() -> None:
    gate = make_gate()
    gate.upstream.add("https://dep-a.example", error_response(529, "overloaded_error"))
    final = error_response(503, "api_error")
    gate.upstream.add("https://dep-b.example", final)

    async def scenario() -> tuple[int, bytes]:
        reply = await gate.handler.handle(request())
        assert isinstance(reply, StreamReply)
        return reply.relay.status, await collect(reply.relay)

    status, body = run(scenario())
    assert status == 503 and body == final.chunks[0]
    assert len(gate.usage.attempt_list()) == 2


def test_cancel_while_draining_a_fallback_rejection_settles_and_releases() -> None:
    gate = make_gate()
    hold = asyncio.Event()  # never set: upstream stalls mid-body
    body = error_response(529, "overloaded_error").chunks[0]
    rejection = ScriptedResponse(
        status=529,
        chunks=[body[:10], body[10:]],
        headers=[("content-type", "application/json")],
        hold=hold,
    )
    gate.upstream.add("https://dep-a.example", rejection)
    gate.upstream.add("https://dep-b.example", ScriptedResponse(chunks=stream_chunks()))

    async def scenario() -> None:
        task = asyncio.ensure_future(gate.handler.handle(request()))
        await asyncio.sleep(0.01)  # 529 received, first body chunk drained, upstream holding
        assert gate.admission.reserved_tokens("key-1") > 0
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    run(scenario())
    assert len(gate.upstream.calls) == 1
    assert getattr(rejection, "closed", False)
    [attempt] = gate.usage.attempt_list()
    assert attempt.deployment == "dep-a" and attempt.upstream_status == 529
    assert attempt.cost_micros == attempt.estimate_micros > 0  # not proof of zero cost
    assert gate.admission.reserved_tokens("key-1") == 0
    parent = gate.usage.only_parent()
    assert parent.request_state is RequestState.FINISHED and parent.http_status == 499
    assert parent.totals is not None and parent.totals.attempt_count == 1
    assert parent.totals.cost_micros == attempt.cost_micros


def test_client_disconnect_before_headers_settles_without_fallback() -> None:
    gate = make_gate()
    headers_hold = asyncio.Event()  # upstream has not answered yet
    gate.upstream.add(
        "https://dep-a.example",
        replace(error_response(529, "overloaded_error"), headers_hold=headers_hold),
    )
    gate.upstream.add("https://dep-b.example", ScriptedResponse(chunks=stream_chunks()))
    gone = [False]

    async def scenario() -> None:
        incoming = replace(request(), client_gone=lambda: gone[0])
        task = asyncio.ensure_future(gate.handler.handle(incoming))
        await asyncio.sleep(0.01)  # dispatched to dep-a, waiting for its headers
        gone[0] = True  # the adapter saw http.disconnect
        task.cancel()
        headers_hold.set()  # a late 529 must not start a fallback
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.sleep(0.01)

    run(scenario())
    assert [url for url, _, _ in gate.upstream.calls] == ["https://dep-a.example/v1/messages"]
    [attempt] = gate.usage.attempt_list()
    assert attempt.termination_reason is TerminationReason.CLIENT_DISCONNECT
    assert attempt.client_disconnected and attempt.upstream_status is None
    assert gate.admission.reserved_tokens("key-1") == 0
    parent = gate.usage.only_parent()
    assert parent.http_status == 499 and parent.client_disconnected


def test_client_disconnect_while_draining_a_rejection_is_not_recorded_as_shutdown() -> None:
    gate = make_gate()
    body = error_response(529, "overloaded_error").chunks[0]
    gate.upstream.add(
        "https://dep-a.example",
        ScriptedResponse(
            status=529,
            chunks=[body[:10], body[10:]],
            headers=[("content-type", "application/json")],
            hold=asyncio.Event(),  # never set: upstream stalls mid-body
        ),
    )
    gate.upstream.add("https://dep-b.example", ScriptedResponse(chunks=stream_chunks()))
    gone = [False]

    async def scenario() -> None:
        incoming = replace(request(), client_gone=lambda: gone[0])
        task = asyncio.ensure_future(gate.handler.handle(incoming))
        await asyncio.sleep(0.01)
        gone[0] = True
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    run(scenario())
    assert len(gate.upstream.calls) == 1
    [attempt] = gate.usage.attempt_list()
    assert attempt.termination_reason is TerminationReason.CLIENT_DISCONNECT
    assert attempt.client_disconnected and attempt.upstream_status == 529
    assert gate.usage.only_parent().client_disconnected


def test_no_fallback_once_the_client_is_gone() -> None:
    gate = make_gate()
    gate.upstream.add("https://dep-a.example", error_response(529, "overloaded_error"))
    gate.upstream.add("https://dep-b.example", ScriptedResponse(chunks=stream_chunks()))

    async def scenario() -> None:
        reply = await gate.handler.handle(replace(request(), client_gone=lambda: True))
        assert isinstance(reply, StreamReply) and reply.relay.status == 529
        await collect(reply.relay)

    run(scenario())
    assert len(gate.upstream.calls) == 1


def test_no_fallback_once_shutdown_has_started() -> None:
    gate = make_gate()
    gate.shutting_down[0] = True
    gate.upstream.add("https://dep-a.example", error_response(529, "overloaded_error"))

    async def scenario() -> int:
        reply = await gate.handler.handle(request())
        assert isinstance(reply, StreamReply)
        await collect(reply.relay)
        return reply.relay.status

    assert run(scenario()) == 529
    assert len(gate.upstream.calls) == 1


def test_no_fallback_when_shutdown_starts_while_draining_the_failed_response() -> None:
    gate = make_gate()
    failed = error_response(529, "overloaded_error")
    body = failed.chunks

    async def read_then_shut_down() -> AsyncIterator[bytes]:
        gate.shutting_down[0] = True  # SIGTERM arrives mid-drain, after dep-b was selected
        for chunk in body:
            yield chunk

    failed.iter_raw = read_then_shut_down  # type: ignore[method-assign]
    gate.upstream.add("https://dep-a.example", failed)
    gate.upstream.add("https://dep-b.example", ScriptedResponse(chunks=stream_chunks()))

    reply = run(gate.handler.handle(request()))
    assert isinstance(reply, ErrorReply) and reply.error.gateway_code == "shutting_down"
    assert reply.error.status == 503
    assert len(gate.upstream.calls) == 1
    [attempt] = gate.usage.attempt_list()
    assert attempt.upstream_status == 529
    assert gate.admission.reserved_tokens("key-1") == 0
    parent = gate.usage.only_parent()
    assert parent.request_state is RequestState.FINISHED
    assert {"event": "fallback_halted", "deployment": "dep-b", "reason": "shutdown"} in (
        parent.policy_decisions
    )


def test_no_dispatch_when_shutdown_starts_during_fallback_admission() -> None:
    gate = make_gate()
    gate.upstream.add("https://dep-a.example", error_response(529, "overloaded_error"))
    gate.upstream.add("https://dep-b.example", ScriptedResponse(chunks=stream_chunks()))
    admit = gate.admission.admit

    async def admit_then_shut_down(*args: Any, **kwargs: Any) -> Any:
        admission = await admit(*args, **kwargs)
        if kwargs.get("stage") == "fallback":
            gate.shutting_down[0] = True  # shutdown begins while waiting on the lock
        return admission

    gate.admission.admit = admit_then_shut_down  # type: ignore[method-assign]

    reply = run(gate.handler.handle(request()))
    assert isinstance(reply, ErrorReply) and reply.error.gateway_code == "shutting_down"
    assert len(gate.upstream.calls) == 1
    assert len(gate.usage.attempt_list()) == 1
    # The fallback's reservation was admitted but never dispatched: it is released.
    assert gate.admission.reserved_tokens("key-1") == 0


def test_no_fallback_when_the_client_leaves_while_draining() -> None:
    gate = make_gate()
    gone = [False]
    failed = error_response(529, "overloaded_error")
    body = failed.chunks

    async def read_then_leave() -> AsyncIterator[bytes]:
        gone[0] = True
        for chunk in body:
            yield chunk

    failed.iter_raw = read_then_leave  # type: ignore[method-assign]
    gate.upstream.add("https://dep-a.example", failed)
    gate.upstream.add("https://dep-b.example", ScriptedResponse(chunks=stream_chunks()))

    reply = run(gate.handler.handle(replace(request(), client_gone=lambda: gone[0])))
    assert isinstance(reply, ErrorReply) and reply.error.status == 499
    assert len(gate.upstream.calls) == 1
    assert gate.usage.only_parent().client_disconnected


def test_settlement_failure_stops_fallback_and_fails_closed() -> None:
    gate = make_gate()
    gate.usage.fail_settle = True
    gate.upstream.add("https://dep-a.example", error_response(529, "overloaded_error"))

    reply = run(gate.handler.handle(request()))
    assert isinstance(reply, ErrorReply) and reply.error.status == 503
    assert len(gate.upstream.calls) == 1
    assert gate.admission.disabled
    again = run(gate.handler.handle(request()))
    assert isinstance(again, ErrorReply) and again.error.status == 503


# --- interrupted streams (5.7, 5.8) ------------------------------------------------------


def test_upstream_disconnect_mid_stream_uses_estimated_debit() -> None:
    gate = make_gate()
    gate.upstream.add(
        "https://dep-a.example", ScriptedResponse(chunks=stream_chunks(), fail_after=2)
    )

    async def scenario() -> None:
        reply = await gate.handler.handle(request())
        assert isinstance(reply, StreamReply)
        await collect(reply.relay)

    run(scenario())
    [attempt] = gate.usage.attempt_list()
    assert attempt.termination_reason is TerminationReason.UPSTREAM_DISCONNECT
    assert attempt.outcome is Outcome.PARTIAL
    assert attempt.accounting_kind is AccountingKind.ESTIMATED
    assert attempt.cost_micros == max(attempt.estimate_micros, attempt.observed_cost_micros or 0)
    assert attempt.priced_cost_micros is None


def test_client_disconnect_mid_stream_settles_with_estimate() -> None:
    gate = make_gate()
    hold = asyncio.Event()
    gate.upstream.add("https://dep-a.example", ScriptedResponse(chunks=stream_chunks(), hold=hold))

    async def scenario() -> None:
        reply = await gate.handler.handle(request())
        assert isinstance(reply, StreamReply)
        pump = asyncio.ensure_future(collect(reply.relay))
        await asyncio.sleep(0.01)  # first chunk relayed; upstream is holding
        reply.relay.mark_client_disconnected()
        pump.cancel()
        await asyncio.wait({pump})

    run(scenario())
    [attempt] = gate.usage.attempt_list()
    assert attempt.termination_reason is TerminationReason.CLIENT_DISCONNECT
    assert attempt.client_disconnected and not attempt.usage_complete
    assert attempt.cost_micros == attempt.estimate_micros
    assert gate.usage.only_parent().request_state is RequestState.FINISHED
    assert gate.admission.reserved_tokens("key-1") == 0


def test_cancel_at_yield_boundary_still_parses_final_chunk_once() -> None:
    gate = make_gate()
    chunks = stream_chunks()
    gate.upstream.add("https://dep-a.example", ScriptedResponse(chunks=chunks))

    async def scenario() -> None:
        reply = await gate.handler.handle(request())
        assert isinstance(reply, StreamReply)
        sent = 0

        async def send(chunk: bytes) -> None:
            nonlocal sent
            sent += 1
            if sent == len(chunks):  # the client goes away while the last chunk is in flight
                reply.relay.mark_client_disconnected()
                raise asyncio.CancelledError

        with pytest.raises(asyncio.CancelledError):
            await reply.relay.pump(send)

    run(scenario())
    [attempt] = gate.usage.attempt_list()
    assert attempt.usage_complete and attempt.outcome is Outcome.COMPLETE
    assert attempt.client_disconnected
    assert len(gate.usage.attempts) == 1


def test_unparseable_event_mid_stream_is_relayed_and_settled_as_parse_error() -> None:
    gate = make_gate()
    start, _, delta, stop = stream_chunks()
    deep = b"event: content_block_delta\ndata: " + b"[" * 100_000 + b"\n\n"
    chunks = [start, deep, delta, stop]
    response = ScriptedResponse(chunks=chunks)
    gate.upstream.add("https://dep-a.example", response)

    async def scenario() -> bytes:
        reply = await gate.handler.handle(request())
        assert isinstance(reply, StreamReply)
        return await collect(reply.relay)

    assert run(scenario()) == b"".join(chunks)
    assert response.closed
    [attempt] = gate.usage.attempt_list()
    assert attempt.usage_source is UsageSource.PARSE_ERROR
    assert not attempt.usage_complete
    assert attempt.accounting_kind is AccountingKind.ESTIMATED
    assert gate.usage.only_parent().request_state is RequestState.FINISHED
    assert gate.admission.reserved_tokens("key-1") == 0


def test_count_tokens_is_unmetered_but_consumes_rpm() -> None:
    gate = make_gate(make_config(rpm=1))
    body = b'{"input_tokens": 12}'
    gate.upstream.add(
        "https://dep-a.example",
        ScriptedResponse(chunks=[body], headers=[("content-type", "application/json")]),
    )
    from gate.domain.models import Route

    async def scenario() -> None:
        reply = await gate.handler.handle(
            request({"model": "model-a", "messages": []}, route=Route.COUNT_TOKENS)
        )
        assert isinstance(reply, StreamReply)
        assert await collect(reply.relay) == body
        again = await gate.handler.handle(request())
        assert isinstance(again, ErrorReply) and again.error.gateway_code == "rpm_exceeded"

    run(scenario())
    [attempt] = gate.usage.attempt_list()
    assert attempt.outcome is Outcome.UNMETERED and attempt.cost_micros == 0
    assert attempt.key_period_id is None

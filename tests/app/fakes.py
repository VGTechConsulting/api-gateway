"""In-memory fakes for every application port.

These let the lifecycle, admission and relay be tested without a network,
a database or a real clock.
"""

import asyncio
import json
from collections.abc import AsyncIterator, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

from gate.app.admission import AdmissionController
from gate.app.lifecycle import HandleRequest, IncomingRequest, hash_key
from gate.app.ports import (
    AttemptWithParent,
    FailureKind,
    KeyRecord,
    ParentRow,
    RequestFilter,
    UpstreamFailure,
    UpstreamReadError,
)
from gate.app.routing import Router
from gate.domain.config import ALL_ENFORCED, DetectConfig, GatewayConfig, Limits, Team
from gate.domain.ledger import BucketKey, ScopeType
from gate.domain.models import (
    ApiFormat,
    AttemptRecord,
    Budget,
    Cadence,
    Deployment,
    DetectMode,
    EnforcementMode,
    Env,
    Policy,
    PriceEntry,
    PricingMode,
    RequestState,
    Route,
)
from gate.domain.money import Micros
from gate.providers.anthropic import AnthropicFormat

# --- clocks ------------------------------------------------------------------------------


class FakeWall:
    def __init__(self, start: datetime = datetime(2026, 11, 10, 12, 0, tzinfo=UTC)) -> None:
        self.current = start

    def now(self) -> datetime:
        return self.current

    def advance(self, seconds: float) -> None:
        self.current += timedelta(seconds=seconds)


class FakeMono:
    def __init__(self) -> None:
        self.current = 1000.0

    def now(self) -> float:
        return self.current

    def advance(self, seconds: float) -> None:
        self.current += seconds


# --- repositories ------------------------------------------------------------------------


class MemoryKeys:
    def __init__(self) -> None:
        self.records: dict[str, KeyRecord] = {}

    def find_active_by_hash(self, key_hash: str) -> KeyRecord | None:
        for record in self.records.values():
            if record.key_hash == key_hash and record.revoked_at is None:
                return record
        return None

    def insert(self, record: KeyRecord) -> None:
        self.records[record.id] = record

    def list(self, *, team: str | None, include_revoked: bool) -> Sequence[KeyRecord]:
        return list(self.records.values())

    def find_by_id_or_prefix(self, ident: str) -> Sequence[KeyRecord]:
        return [r for r in self.records.values() if ident in (r.id, r.prefix)]

    def revoke(self, key_id: str, at: datetime) -> bool:
        self.records[key_id] = replace(self.records[key_id], revoked_at=at)
        return True

    def get(self, key_id: str) -> KeyRecord | None:
        return self.records.get(key_id)


class MemoryUsage:
    def __init__(self) -> None:
        self.parents: dict[str, ParentRow] = {}
        self.attempts: dict[tuple[str, int], AttemptRecord] = {}
        self.fail_settle = False

    def open_request(self, row: ParentRow) -> None:
        self.parents[row.request_id] = row

    def settle_attempt(self, attempt: AttemptRecord, parent: ParentRow) -> bool:
        if self.fail_settle:
            raise OSError("disk full")
        key = (attempt.request_id, attempt.attempt_no)
        if key in self.attempts:
            return False
        self.attempts[key] = attempt
        self.parents[parent.request_id] = parent
        return True

    def finish_request(self, row: ParentRow) -> None:
        self.parents[row.request_id] = row

    def mark_interrupted(self) -> int:
        return 0

    def spend_by_bucket(self) -> Iterable[tuple[BucketKey, int]]:
        return []

    def attempts_for_period(self, period_id: str) -> Sequence[AttemptWithParent]:
        return []

    def requests(self, *, since: datetime | None, until: datetime | None) -> Sequence[ParentRow]:
        return list(self.parents.values())

    def latest_requests(self, where: RequestFilter) -> Iterable[ParentRow]:
        return sorted(self.parents.values(), key=lambda p: p.ts_start, reverse=True)

    def request(self, request_id: str) -> ParentRow | None:
        return self.parents.get(request_id)

    def attempts_for_requests(self, request_ids: Sequence[str]) -> Sequence[AttemptWithParent]:
        return []

    def writable(self) -> bool:
        return True

    def attempt_list(self) -> list[AttemptRecord]:
        return sorted(self.attempts.values(), key=lambda a: (a.request_id, a.attempt_no))

    def only_parent(self) -> ParentRow:
        assert len(self.parents) == 1, self.parents
        return next(iter(self.parents.values()))


class MemoryLog:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def write(self, event: Mapping[str, Any]) -> None:
        self.events.append(dict(event))


# --- upstream ----------------------------------------------------------------------------


@dataclass
class ScriptedResponse:
    status: int = 200
    chunks: list[bytes] = field(default_factory=list)
    headers: list[tuple[str, str]] = field(
        default_factory=lambda: [("content-type", "text/event-stream")]
    )
    fail_after: int | None = None  # raise a read error after this many chunks
    hold: asyncio.Event | None = None  # wait on this before the second chunk
    headers_hold: asyncio.Event | None = None  # send() waits on this before returning
    closed: bool = False

    async def iter_raw(self) -> AsyncIterator[bytes]:
        for index, chunk in enumerate(self.chunks):
            if self.fail_after is not None and index == self.fail_after:
                raise UpstreamReadError(FailureKind.CONNECTION)
            if index == 1 and self.hold is not None:
                await self.hold.wait()
            yield chunk

    async def aclose(self) -> None:
        self.closed = True


class ScriptedUpstream:
    """Returns scripted responses (or raises failures) per deployment base URL, in order."""

    def __init__(self) -> None:
        self.script: dict[str, list[ScriptedResponse | Exception]] = {}
        self.calls: list[tuple[str, dict[str, str], bytes]] = []

    def add(self, base_url: str, *items: ScriptedResponse | Exception) -> None:
        self.script.setdefault(base_url, []).extend(items)

    async def send(self, url: str, headers: Mapping[str, str], body: bytes) -> ScriptedResponse:
        self.calls.append((url, dict(headers), body))
        for base, items in self.script.items():
            if url.startswith(base) and items:
                item = items.pop(0)
                if isinstance(item, Exception):
                    raise item
                if item.headers_hold is not None:
                    await item.headers_hold.wait()
                return item
        raise AssertionError(f"no scripted response for {url}")


# --- SSE fixtures ------------------------------------------------------------------------


def sse(event: dict[str, Any]) -> bytes:
    return f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode()


def stream_chunks(
    *,
    input_tokens: int = 100,
    output_tokens: int = 50,
    model: str = "model-a",
    cache_5m: int = 0,
    cache_1h: int = 0,
    cache_read: int = 0,
    stop: bool = True,
) -> list[bytes]:
    usage = {
        "input_tokens": input_tokens,
        "cache_creation_input_tokens": cache_5m + cache_1h,
        "cache_read_input_tokens": cache_read,
        "cache_creation": {
            "ephemeral_5m_input_tokens": cache_5m,
            "ephemeral_1h_input_tokens": cache_1h,
        },
    }
    chunks = [
        sse(
            {
                "type": "message_start",
                "message": {"model": model, "usage": {**usage, "output_tokens": 1}},
            }
        ),
        sse(
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "hi"},
            }
        ),
    ]
    if stop:
        chunks += [
            sse(
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn"},
                    "usage": {**usage, "output_tokens": output_tokens},
                }
            ),
            sse({"type": "message_stop"}),
        ]
    return chunks


def error_response(
    status: int, error_type: str, retry_after: str | None = None
) -> ScriptedResponse:
    headers = [("content-type", "application/json")]
    if retry_after:
        headers.append(("retry-after", retry_after))
    body = json.dumps({"type": "error", "error": {"type": error_type, "message": "x"}}).encode()
    return ScriptedResponse(status=status, chunks=[body], headers=headers)


def not_sent() -> UpstreamFailure:
    return UpstreamFailure(FailureKind.CONNECT, not_sent=True)


# --- assembled gate ----------------------------------------------------------------------

RAW_KEY = "gk_" + "A" * 43
SECOND_KEY = "gk_" + "B" * 43

# USD per million tokens: model-a costs 3 in / 15 out on dep-a, more on dep-b.
PRICE_A = PriceEntry(
    "dep-a",
    "model-a",
    PricingMode.METERED,
    Decimal(3),
    Decimal(15),
    Decimal("0.3"),
    Decimal("3.75"),
    Decimal(6),
)
PRICE_B = PriceEntry(
    "dep-b",
    "model-a",
    PricingMode.METERED,
    Decimal(30),
    Decimal(150),
    Decimal(3),
    Decimal("37.5"),
    Decimal(60),
)


def make_config(
    *,
    aliases: Mapping[str, tuple[str, ...]] | None = None,
    team_budget_usd: str | None = None,
    enforcement: Mapping[Policy, EnforcementMode] | None = None,
    detect: Mapping[str, DetectMode] | None = None,
    tpm: int = 1_000_000,
    rpm: int = 1000,
) -> GatewayConfig:
    deployments = {
        name: Deployment(
            name,
            ApiFormat.ANTHROPIC,
            f"https://{name}.example",
            f"{name.upper()}_KEY",
            provider=name,
            region="us",
        )
        for name in ("dep-a", "dep-b")
    }
    from gate.domain.money import usd_to_micros

    team_budget = (
        Budget(usd_to_micros(Decimal(team_budget_usd)), Cadence.MONTH) if team_budget_usd else None
    )
    return GatewayConfig(
        host="127.0.0.1",
        port=0,
        sqlite_path=":memory:",
        request_log="stdout",
        default_tpm=tpm,
        default_rpm=rpm,
        limits=Limits(cleanup_timeout_s=1.0, connect_timeout_s=1.0),
        deployments=deployments,
        aliases=aliases or {"model-a": ("dep-a", "dep-b")},
        prices={("dep-a", "model-a"): PRICE_A, ("dep-b", "model-a"): PRICE_B},
        enforcement_default=enforcement or ALL_ENFORCED,
        teams={"platform": Team("platform", team_budget)},
        detect=DetectConfig(categories=detect or {}),
        session_headers=("x-session-id",),
    )


def key_record(
    raw: str = RAW_KEY,
    *,
    key_id: str = "key-1",
    models: tuple[str, ...] = ("model-a",),
    budget_usd: str | None = None,
    tpm: int | None = None,
    rpm: int | None = None,
    period: Cadence = Cadence.MONTH,
) -> KeyRecord:
    from gate.domain.money import usd_to_micros

    return KeyRecord(
        id=key_id,
        key_hash=hash_key(raw),
        prefix=raw[3:11],
        owner="alice",
        team="platform",
        app="demo",
        env=Env.DEV,
        allowed_models=models,
        budget_micros=usd_to_micros(Decimal(budget_usd)) if budget_usd else None,
        period=period,
        tpm=tpm,
        rpm=rpm,
        created_at=datetime(2026, 11, 1, tzinfo=UTC),
        revoked_at=None,
    )


@dataclass
class Gate:
    handler: HandleRequest
    admission: AdmissionController
    upstream: ScriptedUpstream
    usage: MemoryUsage
    keys: MemoryKeys
    log: MemoryLog
    wall: FakeWall
    mono: FakeMono
    router: Router
    shutting_down: list[bool]


def make_gate(config: GatewayConfig | None = None, *keys: KeyRecord) -> Gate:
    config = config or make_config()
    wall, mono = FakeWall(), FakeMono()
    usage, store, log = MemoryUsage(), MemoryKeys(), MemoryLog()
    for record in keys or (key_record(),):
        store.insert(record)
    upstream = ScriptedUpstream()
    admission = AdmissionController(wall, mono)
    router = Router(config.deployments, config.aliases, config.prices)
    flag = [False]
    handler = HandleRequest(
        config=config,
        keys=store,
        usage=usage,
        upstream=upstream,
        request_log=log,
        admission=admission,
        router=router,
        providers={ApiFormat.ANTHROPIC: AnthropicFormat()},
        wall=wall,
        mono=mono,
        env={"DEP-A_KEY": "sk-a", "DEP-B_KEY": "sk-b"},
        shutting_down=lambda: flag[0],
    )
    return Gate(handler, admission, upstream, usage, store, log, wall, mono, router, flag)


def request(
    body: dict[str, Any] | None = None,
    *,
    key: str = RAW_KEY,
    route: Route = Route.MESSAGES,
    headers: Sequence[tuple[str, str]] = (),
) -> IncomingRequest:
    payload = (
        body
        if body is not None
        else {
            "model": "model-a",
            "max_tokens": 100,
            "stream": True,
            "messages": [{"role": "user", "content": "hello"}],
        }
    )
    return IncomingRequest(
        route=route,
        headers=(("x-api-key", key), ("anthropic-version", "2023-06-01"), *headers),
        body=json.dumps(payload).encode(),
        body_too_large=False,
        received_mono=0.0,
    )


async def collect(relay: Any) -> bytes:
    received = bytearray()

    async def send(chunk: bytes) -> None:
        received.extend(chunk)

    await relay.pump(send)
    return bytes(received)


__all__ = [
    "PRICE_A",
    "PRICE_B",
    "RAW_KEY",
    "SECOND_KEY",
    "Gate",
    "Micros",
    "RequestState",
    "ScopeType",
    "collect",
    "error_response",
    "key_record",
    "make_config",
    "make_gate",
    "not_sent",
    "request",
    "sse",
    "stream_chunks",
]

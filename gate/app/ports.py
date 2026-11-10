"""Ports: the application's only boundaries.

Each port has a production adapter in ``gate/adapters`` and an in-memory fake
in the tests. No framework or driver type crosses these interfaces.
"""

from collections.abc import AsyncIterator, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any, Protocol

from gate.domain.accounting import ParentTotals
from gate.domain.ledger import BucketKey
from gate.domain.models import AttemptRecord, Cadence, Env, PolicyEvent, RequestState

# --- clocks ------------------------------------------------------------------------------


class WallClock(Protocol):
    def now(self) -> datetime:
        """Timezone-aware UTC now."""


class MonotonicClock(Protocol):
    def now(self) -> float:
        """Monotonic seconds."""


# --- keys --------------------------------------------------------------------------------


@dataclass(frozen=True)
class KeyRecord:
    id: str
    key_hash: str
    prefix: str
    owner: str
    team: str
    app: str
    env: Env
    allowed_models: tuple[str, ...]
    budget_micros: int | None
    period: Cadence
    tpm: int | None
    rpm: int | None
    created_at: datetime
    revoked_at: datetime | None


class KeyRepository(Protocol):
    def find_active_by_hash(self, key_hash: str) -> KeyRecord | None: ...

    def insert(self, record: KeyRecord) -> None: ...

    def list(self, *, team: str | None, include_revoked: bool) -> Sequence[KeyRecord]: ...

    def find_by_id_or_prefix(self, ident: str) -> Sequence[KeyRecord]: ...

    def revoke(self, key_id: str, at: datetime) -> bool: ...

    def get(self, key_id: str) -> KeyRecord | None: ...


# --- usage -------------------------------------------------------------------------------


@dataclass(frozen=True)
class ParentRow:
    """The parent ``usage`` row for one client request."""

    request_id: str
    request_state: RequestState
    ts_start: datetime
    ts_end: datetime | None
    key_id: str | None
    key_prefix: str | None
    owner: str | None
    team: str | None
    app: str | None
    env: str | None
    session_id: str | None
    route: str
    model: str | None
    stream: bool | None
    totals: ParentTotals | None
    upstream_model: str | None = None
    deployment: str | None = None
    provider: str | None = None
    region: str | None = None
    http_status: int | None = None
    upstream_status: int | None = None
    stream_complete: bool | None = None
    termination_reason: str | None = None
    ttfb_ms: float | None = None
    latency_ms: float | None = None
    gate_overhead_ms: float | None = None
    client_disconnected: bool = False
    policy_decisions: tuple[PolicyEvent, ...] = ()
    would_deny: tuple[str, ...] = ()
    detections: Mapping[str, int] | None = None


@dataclass(frozen=True)
class AttemptWithParent:
    """An attempt joined with its parent's attribution, for reports and balances."""

    attempt: AttemptRecord
    key_id: str | None
    owner: str | None
    team: str | None
    app: str | None
    request_finished: bool


@dataclass(frozen=True)
class RequestFilter:
    """Exact-match filters for the recent-requests feed.

    ``provider`` matches the parent's provider or the provider of any of its attempts,
    so a request that fell back across providers is found under each of them.
    """

    team: str | None = None
    app: str | None = None
    key_prefix: str | None = None
    model: str | None = None
    provider: str | None = None


class UsageRepository(Protocol):
    def open_request(self, row: ParentRow) -> None:
        """Insert the parent with ``request_state = open`` before first dispatch."""

    def settle_attempt(self, attempt: AttemptRecord, parent: ParentRow) -> bool:
        """Insert the attempt and update the parent's running totals in one transaction.

        Returns False, changing nothing, if ``(request_id, attempt_no)`` already exists.
        """

    def finish_request(self, row: ParentRow) -> None:
        """Insert or update the final parent row."""

    def mark_interrupted(self) -> int:
        """At startup: mark leftover open parents interrupted; returns how many."""

    def spend_by_bucket(self) -> Iterable[tuple[BucketKey, int]]:
        """Committed attempt debits grouped by stored key/team period IDs."""

    def attempts(
        self, *, since: datetime | None, until: datetime | None
    ) -> Sequence[AttemptWithParent]: ...

    def attempts_for_period(self, period_id: str) -> Sequence[AttemptWithParent]: ...

    def requests(
        self, *, since: datetime | None, until: datetime | None
    ) -> Sequence[ParentRow]: ...

    def latest_requests(self, where: RequestFilter) -> Iterable[ParentRow]:
        """Matching parents, newest ``ts_start`` first, read lazily so callers can stop."""

    def request(self, request_id: str) -> ParentRow | None: ...

    def attempts_for_requests(self, request_ids: Sequence[str]) -> Sequence[AttemptWithParent]:
        """Attempts of the given parents, ordered by request and attempt number."""

    def writable(self) -> bool: ...


# --- upstream ----------------------------------------------------------------------------


class FailureKind(StrEnum):
    CONNECT = "connect"
    TIMEOUT = "timeout"
    CONNECTION = "connection"


class UpstreamFailure(Exception):
    """A transport failure with no HTTP response.

    ``not_sent`` is True only with reliable evidence that no request bytes could
    have been sent (for example, connection establishment failed).
    """

    def __init__(self, kind: FailureKind, *, not_sent: bool) -> None:
        super().__init__(kind.value)
        self.kind = kind
        self.not_sent = not_sent


class UpstreamReadError(Exception):
    """The upstream body failed mid-read."""

    def __init__(self, kind: FailureKind) -> None:
        super().__init__(kind.value)
        self.kind = kind


class UpstreamResponse(Protocol):
    status: int
    headers: Sequence[tuple[str, str]]

    def iter_raw(self) -> AsyncIterator[bytes]:
        """Raw body bytes; raises UpstreamReadError on transport failure."""

    async def aclose(self) -> None: ...


class Upstream(Protocol):
    async def send(self, url: str, headers: Mapping[str, str], body: bytes) -> UpstreamResponse:
        """Dispatch one request, returning once response headers arrive.

        Raises UpstreamFailure. Never retries or follows redirects.
        """


# --- request log -------------------------------------------------------------------------


class RequestLog(Protocol):
    def write(self, event: Mapping[str, Any]) -> None: ...

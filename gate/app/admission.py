"""Atomic admission and settlement under one lock.

This is the only place the gate's concurrency rule lives. Under one
``asyncio.Lock`` the controller samples the admission instant, pins budget
periods, checks RPM, TPM and both budgets, and commits the RPM entry and the
token and money reservations together. Settlement takes the same lock, so a
settling attempt and a new admission can never interleave. No upstream I/O is
ever awaited while the lock is held.
"""

import asyncio
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime

from gate.app.ports import MonotonicClock, WallClock
from gate.domain.errors import GateError, StorageUnavailable
from gate.domain.ledger import BucketKey, Ledger, MoneyHold, Reservation, ScopeType, check_budgets
from gate.domain.models import EnforcementMode, Policy, PolicyEvent, Principal
from gate.domain.money import Micros
from gate.domain.periods import budget_period_id
from gate.domain.policy import apply_modes, event
from gate.domain.pricing import Estimate
from gate.domain.rate_window import RateLimits, RateWindow, check_rate


@dataclass(frozen=True)
class Admission:
    """A successful admission for one upstream attempt."""

    reservation: Reservation | None  # None for unmetered calls
    ts_admitted: datetime
    key_period_id: str | None
    team_period_id: str | None
    limits: RateLimits
    would_deny: tuple[str, ...] = ()
    events: tuple[PolicyEvent, ...] = field(default=())


class AdmissionDenied(Exception):
    def __init__(
        self, error: GateError, events: tuple[PolicyEvent, ...], would_deny: tuple[str, ...]
    ):
        super().__init__(error.gateway_code)
        self.error = error
        self.events = events
        self.would_deny = would_deny


# Called under the lock with post-settlement overrun events; returns False on a duplicate.
Persist = Callable[[tuple[PolicyEvent, ...]], bool]


class AdmissionController:
    def __init__(self, wall: WallClock, mono: MonotonicClock, *, naive_budgets: bool = False):
        self._wall = wall
        self._mono = mono
        self._naive = naive_budgets
        self._lock = asyncio.Lock()
        self._ledger = Ledger()
        self._windows: defaultdict[str, RateWindow] = defaultdict(RateWindow)
        self._disabled = False

    @property
    def disabled(self) -> bool:
        return self._disabled

    def load_spent(self, balances: Iterable[tuple[BucketKey, int]]) -> None:
        """Startup: rebuild committed spend from attempt rows."""
        for bucket, micros in balances:
            self._ledger.add_spent(bucket, micros)

    def spent(self, bucket: BucketKey) -> int:
        return self._ledger.spent(bucket)

    def reserved_tokens(self, key_id: str) -> int:
        return self._ledger.reserved_tokens(key_id)

    def reserved_money(self, bucket: BucketKey) -> int:
        return self._ledger.reserved_money(bucket)

    async def admit(
        self,
        principal: Principal,
        modes: Mapping[Policy, EnforcementMode],
        estimate: Estimate | None,
        *,
        first: bool,
        stage: str = "admission",
    ) -> Admission:
        """Admit one attempt, or raise AdmissionDenied having committed nothing.

        ``estimate`` is None for unmetered endpoints (RPM only). ``first`` is
        True only for a client request's first admission: fallback attempts
        neither check nor add an RPM entry.
        """
        limits = RateLimits(rpm=principal.rpm, tpm=principal.tpm)
        async with self._lock:
            if self._disabled:
                raise StorageUnavailable("usage storage failed; admission is disabled")
            now = self._wall.now()
            mono = self._mono.now()
            key_period = budget_period_id(principal.budget, now)
            team_period = budget_period_id(principal.team_budget, now)
            window = self._windows[principal.key_id]
            window.expire(mono)

            metered = estimate is not None
            holds = self._holds(principal, estimate, key_period, team_period)
            decisions = check_rate(
                window,
                now=mono,
                limits=limits,
                reserved_tokens=self._ledger.reserved_tokens(principal.key_id),
                estimate_tokens=estimate.tokens if estimate else 0,
                count_request=first,
                metered=metered,
            )
            decisions += check_budgets(self._ledger, holds, naive=self._naive)
            verdict = apply_modes(decisions, modes, stage=stage)
            if verdict.denial is not None:
                raise AdmissionDenied(verdict.denial, verdict.events, verdict.would_deny)

            if first:
                window.add_request(mono)
            reservation = (
                self._ledger.reserve(principal.key_id, estimate.tokens, holds) if estimate else None
            )
            return Admission(
                reservation=reservation,
                ts_admitted=now,
                key_period_id=key_period,
                team_period_id=team_period,
                limits=limits,
                would_deny=verdict.would_deny,
                events=verdict.events,
            )

    def _holds(
        self,
        principal: Principal,
        estimate: Estimate | None,
        key_period: str | None,
        team_period: str | None,
    ) -> tuple[MoneyHold, ...]:
        if estimate is None:
            return ()
        holds: list[MoneyHold] = []
        if principal.budget is not None and key_period is not None:
            bucket = BucketKey(ScopeType.KEY, principal.key_id, key_period)
            holds.append(MoneyHold(bucket, principal.budget.limit, estimate.micros))
        if principal.team_budget is not None and team_period is not None:
            bucket = BucketKey(ScopeType.TEAM, principal.team, team_period)
            holds.append(MoneyHold(bucket, principal.team_budget.limit, estimate.micros))
        return tuple(holds)

    async def release(self, admission: Admission) -> None:
        """Release an attempt's unused reservations from their pinned buckets. Idempotent."""
        if admission.reservation is None:
            return
        async with self._lock:
            self._ledger.release(admission.reservation)

    async def settle(
        self,
        principal: Principal,
        admission: Admission,
        *,
        token_debit: int,
        cost_micros: Micros,
        persist: Persist,
    ) -> bool:
        """Persist the attempt, then swap its reservations for debits, atomically.

        If persistence fails, the reservations are kept, new admission is
        disabled, and StorageUnavailable is raised.
        """
        async with self._lock:
            reservation = admission.reservation
            if reservation is not None and reservation.released:
                return False  # duplicate cleanup: debits were already applied
            overruns = self._overrun_events(principal, admission, token_debit, cost_micros)
            try:
                inserted = persist(overruns)
            except Exception as exc:
                self._disabled = True
                raise StorageUnavailable("usage storage failed; admission is disabled") from exc
            if reservation is not None:
                self._ledger.release(reservation)
            if inserted:
                window = self._windows[principal.key_id]
                window.expire(self._mono.now())
                window.add_tokens(self._mono.now(), token_debit)
                if reservation is not None:
                    for hold in reservation.holds:
                        self._ledger.add_spent(hold.bucket, cost_micros)
            return inserted

    def _overrun_events(
        self, principal: Principal, admission: Admission, token_debit: int, cost: Micros
    ) -> tuple[PolicyEvent, ...]:
        events: list[PolicyEvent] = []
        reservation = admission.reservation
        if reservation is None:
            return ()
        for hold in reservation.holds:
            after = self._ledger.spent(hold.bucket) + cost
            if after > hold.limit:
                events.append(
                    event(
                        "budget_overrun",
                        scope=hold.bucket.scope_type.value,
                        period_id=hold.bucket.period_id,
                        overrun_micros=after - hold.limit,
                    )
                )
        window = self._windows[principal.key_id]
        window.expire(self._mono.now())
        others = self._ledger.reserved_tokens(principal.key_id) - reservation.tokens
        used = window.settled_tokens + others + token_debit
        if used > admission.limits.tpm:
            events.append(event("tpm_overrun", overrun_tokens=used - admission.limits.tpm))
        return tuple(events)

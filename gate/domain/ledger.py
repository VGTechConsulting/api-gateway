"""Token and money reservations, and committed spend per budget bucket.

Money state is keyed by ``(scope_type, scope_id, period_id)``. A reservation
pins its buckets at admission; release and settlement always use those pinned
buckets, even after a period rolls over. Token reservations are per key and
have no period. The ledger holds no lock: the admission controller serialises
every call to it.
"""

import itertools
from collections import defaultdict
from dataclasses import dataclass
from enum import StrEnum

from gate.domain.errors import PermissionDenied
from gate.domain.models import Policy
from gate.domain.money import Micros
from gate.domain.policy import Decision


class ScopeType(StrEnum):
    KEY = "key"
    TEAM = "team"


@dataclass(frozen=True)
class BucketKey:
    scope_type: ScopeType
    scope_id: str
    period_id: str


@dataclass(frozen=True)
class MoneyHold:
    """Part of a reservation held against one budget bucket."""

    bucket: BucketKey
    limit: Micros
    amount: Micros


@dataclass
class Reservation:
    id: int
    key_id: str
    tokens: int
    holds: tuple[MoneyHold, ...]
    released: bool = False


class Ledger:
    def __init__(self) -> None:
        self._spent: defaultdict[BucketKey, int] = defaultdict(int)
        self._reserved_money: defaultdict[BucketKey, int] = defaultdict(int)
        self._reserved_tokens: defaultdict[str, int] = defaultdict(int)
        self._ids = itertools.count(1)

    def spent(self, bucket: BucketKey) -> int:
        return self._spent.get(bucket, 0)

    def reserved_money(self, bucket: BucketKey) -> int:
        return self._reserved_money.get(bucket, 0)

    def reserved_tokens(self, key_id: str) -> int:
        return self._reserved_tokens.get(key_id, 0)

    def reserve(self, key_id: str, tokens: int, holds: tuple[MoneyHold, ...]) -> Reservation:
        self._reserved_tokens[key_id] += tokens
        for hold in holds:
            self._reserved_money[hold.bucket] += hold.amount
        return Reservation(next(self._ids), key_id, tokens, holds)

    def release(self, reservation: Reservation) -> bool:
        """Release both reservations. Idempotent: returns False if already released."""
        if reservation.released:
            return False
        reservation.released = True
        self._reserved_tokens[reservation.key_id] -= reservation.tokens
        if self._reserved_tokens[reservation.key_id] == 0:
            del self._reserved_tokens[reservation.key_id]
        for hold in reservation.holds:
            self._reserved_money[hold.bucket] -= hold.amount
            if self._reserved_money[hold.bucket] == 0:
                del self._reserved_money[hold.bucket]
        return True

    def add_spent(self, bucket: BucketKey, micros: int) -> None:
        if micros:
            self._spent[bucket] += micros

    def overrun(self, hold: MoneyHold) -> int:
        """Balance snapshot ``max(0, spent - budget)`` for a pinned bucket."""
        return max(0, self.spent(hold.bucket) - hold.limit)


def check_budgets(
    ledger: Ledger,
    holds: tuple[MoneyHold, ...],
    *,
    naive: bool = False,
) -> list[Decision]:
    """Budget admission for each pinned bucket.

    Both ``spent < budget`` and ``spent + reserved + estimate <= budget`` must
    hold. ``naive`` checks only the first, as in the article's sample; it
    exists only for the budget-overshoot experiment.
    """
    decisions: list[Decision] = []
    for hold in holds:
        spent = ledger.spent(hold.bucket)
        reserved = ledger.reserved_money(hold.bucket)
        fits = spent < hold.limit and (naive or spent + reserved + hold.amount <= hold.limit)
        check = f"{hold.bucket.scope_type.value}_budget"
        error = (
            None
            if fits
            else PermissionDenied(
                f"{hold.bucket.scope_type.value} budget exhausted for {hold.bucket.period_id}",
                gateway_code="budget_exceeded",
            )
        )
        detail: dict[str, str | int | bool | None] = (
            {}
            if fits
            else {
                "scope": hold.bucket.scope_type.value,
                "period_id": hold.bucket.period_id,
                "spent_micros": spent,
                "reserved_micros": reserved,
                "estimate_micros": hold.amount,
                "budget_micros": hold.limit,
            }
        )
        decisions.append(Decision(Policy.BUDGETS, check, error, detail))
    return decisions

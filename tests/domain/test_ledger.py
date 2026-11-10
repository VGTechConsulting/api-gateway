"""Money and token reservations and committed spend per pinned bucket: atomic
admission, budget periods, settlement and rollover.

Budget admission needs both ``spent < budget`` and
``spent + reserved + estimate <= budget`` for each configured bucket.
"""

from gate.domain.errors import PermissionDenied
from gate.domain.ledger import BucketKey, Ledger, MoneyHold, ScopeType, check_budgets
from gate.domain.models import Policy
from gate.domain.money import Micros

KEY_NOV = BucketKey(ScopeType.KEY, "key-1", "day:2026-11-30")
KEY_DEC = BucketKey(ScopeType.KEY, "key-1", "day:2026-12-01")
TEAM_NOV = BucketKey(ScopeType.TEAM, "platform", "month:2026-11")
BUDGET = Micros(1_000)


def hold(bucket: BucketKey, amount: int, limit: int = BUDGET) -> MoneyHold:
    return MoneyHold(bucket, Micros(limit), Micros(amount))


def passed(ledger: Ledger, *holds: MoneyHold, naive: bool = False) -> list[bool]:
    return [d.passed for d in check_budgets(ledger, holds, naive=naive)]


# --- check_budgets -----------------------------------------------------------------------


def test_fresh_bucket_admits_an_estimate_up_to_the_budget() -> None:
    ledger = Ledger()
    assert passed(ledger, hold(KEY_NOV, 1_000)) == [True]
    assert passed(ledger, hold(KEY_NOV, 1_001)) == [False]


def test_reservations_count_against_the_budget() -> None:
    ledger = Ledger()
    ledger.reserve("key-1", 10, (hold(KEY_NOV, 600),))
    assert passed(ledger, hold(KEY_NOV, 400)) == [True]
    assert passed(ledger, hold(KEY_NOV, 401)) == [False]


def test_spent_plus_reserved_plus_estimate_must_fit() -> None:
    ledger = Ledger()
    ledger.add_spent(KEY_NOV, 500)
    ledger.reserve("key-1", 10, (hold(KEY_NOV, 300),))
    assert passed(ledger, hold(KEY_NOV, 200)) == [True]
    assert passed(ledger, hold(KEY_NOV, 201)) == [False]


def test_zero_estimate_is_blocked_once_spend_reaches_the_budget() -> None:
    # spent == budget: the second condition (1000 + 0 + 0 <= 1000) holds, but
    # spent < budget fails, so even a zero-estimate request is rejected.
    ledger = Ledger()
    ledger.add_spent(KEY_NOV, 1_000)
    decisions = check_budgets(ledger, (hold(KEY_NOV, 0),))
    assert not decisions[0].passed


def test_overrun_spend_blocks_subsequent_requests() -> None:
    ledger = Ledger()
    ledger.add_spent(KEY_NOV, 1_250)
    assert passed(ledger, hold(KEY_NOV, 0)) == [False]


def test_denial_is_a_403_budget_exceeded_with_content_free_detail() -> None:
    ledger = Ledger()
    ledger.add_spent(TEAM_NOV, 900)
    ledger.reserve("key-1", 10, (hold(TEAM_NOV, 50),))
    (decision,) = check_budgets(ledger, (hold(TEAM_NOV, 100),))
    assert decision.policy is Policy.BUDGETS
    assert decision.check == "team_budget"
    assert isinstance(decision.error, PermissionDenied)
    assert decision.error.status == 403
    assert decision.error.gateway_code == "budget_exceeded"
    assert dict(decision.detail) == {
        "scope": "team",
        "period_id": "month:2026-11",
        "spent_micros": 900,
        "reserved_micros": 50,
        "estimate_micros": 100,
        "budget_micros": 1_000,
    }


def test_key_and_team_buckets_are_checked_independently() -> None:
    ledger = Ledger()
    ledger.add_spent(TEAM_NOV, 1_000)  # the team month is exhausted
    decisions = check_budgets(ledger, (hold(KEY_NOV, 10), hold(TEAM_NOV, 10)))
    assert [d.check for d in decisions] == ["key_budget", "team_budget"]
    assert [d.passed for d in decisions] == [True, False]


def test_new_daily_key_period_does_not_bypass_an_exhausted_team_month() -> None:
    ledger = Ledger()
    ledger.add_spent(KEY_NOV, 1_000)
    ledger.add_spent(TEAM_NOV, 5_000)
    # After midnight the key has a fresh day, but the team month is still exhausted.
    assert passed(ledger, hold(KEY_DEC, 10), hold(TEAM_NOV, 10, limit=5_000)) == [True, False]


def test_old_period_spend_does_not_consume_new_period_capacity() -> None:
    ledger = Ledger()
    ledger.add_spent(KEY_NOV, 1_500)
    ledger.reserve("key-1", 10, (hold(KEY_NOV, 400),))
    assert passed(ledger, hold(KEY_DEC, 1_000)) == [True]


def test_no_holds_means_no_budget_decisions() -> None:
    assert check_budgets(Ledger(), ()) == []


def test_naive_mode_checks_only_spent_below_budget() -> None:
    ledger = Ledger()
    ledger.add_spent(KEY_NOV, 500)
    ledger.reserve("key-1", 10, (hold(KEY_NOV, 400),))
    # Reserved and incoming estimates are ignored, as in the article's sample (experiment only).
    assert passed(ledger, hold(KEY_NOV, 900), naive=True) == [True]
    assert passed(ledger, hold(KEY_NOV, 900), naive=False) == [False]
    ledger.add_spent(KEY_NOV, 500)
    assert passed(ledger, hold(KEY_NOV, 0), naive=True) == [False]


# --- reserve / release -------------------------------------------------------------------


def test_reserve_adds_tokens_per_key_and_money_per_bucket() -> None:
    ledger = Ledger()
    ledger.reserve("key-1", 300, (hold(KEY_NOV, 40), hold(TEAM_NOV, 40)))
    ledger.reserve("key-1", 200, (hold(KEY_NOV, 60),))
    assert ledger.reserved_tokens("key-1") == 500
    assert ledger.reserved_tokens("key-2") == 0
    assert ledger.reserved_money(KEY_NOV) == 100
    assert ledger.reserved_money(TEAM_NOV) == 40


def test_release_returns_both_reservations_to_their_pinned_buckets() -> None:
    ledger = Ledger()
    first = ledger.reserve("key-1", 300, (hold(KEY_NOV, 40), hold(TEAM_NOV, 40)))
    ledger.reserve("key-1", 200, (hold(KEY_NOV, 60),))
    assert ledger.release(first) is True
    assert ledger.reserved_tokens("key-1") == 200
    assert ledger.reserved_money(KEY_NOV) == 60
    assert ledger.reserved_money(TEAM_NOV) == 0


def test_release_is_idempotent() -> None:
    ledger = Ledger()
    keep = ledger.reserve("key-1", 100, (hold(KEY_NOV, 10),))
    reservation = ledger.reserve("key-1", 300, (hold(KEY_NOV, 40),))
    assert ledger.release(reservation) is True
    assert ledger.release(reservation) is False  # duplicate cleanup changes nothing
    assert ledger.reserved_tokens("key-1") == 100
    assert ledger.reserved_money(KEY_NOV) == 10
    assert keep.released is False


def test_reservations_get_distinct_ids() -> None:
    ledger = Ledger()
    a = ledger.reserve("key-1", 1, ())
    b = ledger.reserve("key-1", 1, ())
    assert a.id != b.id


def test_release_after_rollover_uses_the_pinned_old_bucket() -> None:
    ledger = Ledger()
    reservation = ledger.reserve("key-1", 100, (hold(KEY_NOV, 400),))
    ledger.reserve("key-1", 100, (hold(KEY_DEC, 300),))
    ledger.release(reservation)
    assert ledger.reserved_money(KEY_NOV) == 0
    assert ledger.reserved_money(KEY_DEC) == 300


def test_overestimate_release_frees_capacity_for_the_next_admission() -> None:
    ledger = Ledger()
    reservation = ledger.reserve("key-1", 100, (hold(KEY_NOV, 900),))
    assert passed(ledger, hold(KEY_NOV, 500)) == [False]
    ledger.release(reservation)
    ledger.add_spent(KEY_NOV, 200)  # actual cost was well below the reservation
    assert passed(ledger, hold(KEY_NOV, 500)) == [True]


# --- spend and overrun -------------------------------------------------------------------


def test_add_spent_records_the_full_cost_without_capping() -> None:
    ledger = Ledger()
    ledger.add_spent(KEY_NOV, 700)
    ledger.add_spent(KEY_NOV, 800)
    assert ledger.spent(KEY_NOV) == 1_500
    assert ledger.spent(KEY_DEC) == 0


def test_overrun_is_a_balance_snapshot_per_pinned_bucket() -> None:
    ledger = Ledger()
    nov = hold(KEY_NOV, 0)
    dec = hold(KEY_DEC, 0)
    assert ledger.overrun(nov) == 0
    ledger.add_spent(KEY_NOV, 1_000)
    assert ledger.overrun(nov) == 0  # exactly at budget is not an overrun
    ledger.add_spent(KEY_NOV, 250)
    assert ledger.overrun(nov) == 250
    ledger.add_spent(KEY_NOV, 50)
    # A snapshot, not an additive charge: the second reading replaces the first.
    assert ledger.overrun(nov) == 300
    assert ledger.overrun(dec) == 0  # an old-period overrun does not touch the new period

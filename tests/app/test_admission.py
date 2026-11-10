"""Atomic admission and settlement under one lock: concurrency, token admission and
budget rollover.
"""

import asyncio
from datetime import UTC, datetime

import pytest

from gate.app.admission import Admission, AdmissionController, AdmissionDenied
from gate.domain.config import ALL_ENFORCED
from gate.domain.errors import StorageUnavailable
from gate.domain.ledger import BucketKey, ScopeType
from gate.domain.models import Budget, Cadence, EnforcementMode, Env, Policy, Principal
from gate.domain.money import Micros
from gate.domain.pricing import Estimate
from tests.app.fakes import FakeMono, FakeWall

EST = Estimate(input_tokens=400, output_tokens=600, micros=Micros(10_000))  # 1000 tokens, $0.01


def principal(
    key_id: str = "k1",
    *,
    budget: int | None = None,
    team_budget: int | None = None,
    tpm: int = 10_000_000,
    rpm: int = 10_000,
    key_cadence: Cadence = Cadence.MONTH,
    team_cadence: Cadence = Cadence.MONTH,
) -> Principal:
    return Principal(
        key_id=key_id,
        prefix=key_id,
        owner="o",
        team="t",
        app="a",
        env=Env.DEV,
        allowed_models=frozenset({"m"}),
        budget=Budget(Micros(budget), key_cadence) if budget is not None else None,
        team_budget=Budget(Micros(team_budget), team_cadence) if team_budget is not None else None,
        tpm=tpm,
        rpm=rpm,
    )


def controller(start: datetime | None = None) -> tuple[AdmissionController, FakeWall, FakeMono]:
    wall, mono = FakeWall(start) if start else FakeWall(), FakeMono()
    return AdmissionController(wall, mono), wall, mono


async def admit_many(
    ctl: AdmissionController, principals: list[Principal]
) -> list[Admission | None]:
    async def one(p: Principal) -> Admission | None:
        try:
            return await ctl.admit(p, ALL_ENFORCED, EST, first=True)
        except AdmissionDenied:
            return None

    return await asyncio.gather(*(one(p) for p in principals))


@pytest.mark.parametrize("scope", ["key", "team"])
def test_100_parallel_requests_admit_exactly_the_10_that_fit(scope: str) -> None:
    ctl, _, _ = controller()
    if scope == "key":
        people = [principal(budget=100_000)] * 100
    else:  # one team budget shared by many keys
        people = [principal(f"k{i}", team_budget=100_000) for i in range(100)]
    admitted = asyncio.run(admit_many(ctl, people))
    assert sum(a is not None for a in admitted) == 10


def test_tpm_admits_exactly_the_reservations_that_fit() -> None:
    ctl, _, _ = controller()
    admitted = asyncio.run(admit_many(ctl, [principal(tpm=10_000)] * 100))
    assert sum(a is not None for a in admitted) == 10
    assert ctl.reserved_tokens("k1") == 10_000


@pytest.mark.parametrize(
    ("kwargs", "code"),
    [
        ({"rpm": 1}, "rpm_exceeded"),
        ({"tpm": 1500}, "tpm_exceeded"),
        ({"budget": 5_000}, "budget_exceeded"),
        ({"team_budget": 5_000}, "budget_exceeded"),
    ],
)
def test_failed_admission_commits_nothing(kwargs: dict[str, int], code: str) -> None:
    ctl, _, _ = controller()

    async def scenario() -> None:
        p = principal(**kwargs)
        if code in ("tpm_exceeded", "rpm_exceeded"):  # use up the capacity first
            await ctl.admit(p, ALL_ENFORCED, Estimate(1, 999, Micros(0)), first=True)
        before = ctl.reserved_tokens("k1")
        with pytest.raises(AdmissionDenied) as denied:
            await ctl.admit(p, ALL_ENFORCED, EST, first=True)
        assert denied.value.error.gateway_code == code
        assert ctl.reserved_tokens("k1") == before

    asyncio.run(scenario())


def test_estimate_larger_than_tpm_is_400_without_retry_after() -> None:
    ctl, _, _ = controller()
    with pytest.raises(AdmissionDenied) as denied:
        asyncio.run(ctl.admit(principal(tpm=999), ALL_ENFORCED, EST, first=True))
    assert denied.value.error.status == 400
    assert denied.value.error.gateway_code == "token_estimate_exceeds_limit"
    assert denied.value.error.retry_after_s is None


def test_settlement_swaps_reservation_for_full_debit_even_above_estimate() -> None:
    ctl, _, _ = controller()
    p = principal(budget=15_000)

    async def scenario() -> None:
        admission = await ctl.admit(p, ALL_ENFORCED, EST, first=True)
        recorded: list[tuple[object, ...]] = []

        def persist(overruns: tuple[object, ...]) -> bool:
            recorded.append(overruns)
            return True

        await ctl.settle(
            p, admission, token_debit=3000, cost_micros=Micros(20_000), persist=persist
        )
        bucket = BucketKey(ScopeType.KEY, "k1", "month:2026-11")
        assert ctl.spent(bucket) == 20_000  # never clamped to the estimate or the budget
        assert ctl.reserved_tokens("k1") == 0
        [events] = recorded
        assert {e["event"] for e in events} == {"budget_overrun"}  # type: ignore[index]
        # Exhausted: even a zero-estimate request is rejected.
        with pytest.raises(AdmissionDenied):
            await ctl.admit(p, ALL_ENFORCED, Estimate(0, 0, Micros(0)), first=True)
        # A duplicate settlement adds nothing.
        assert not await ctl.settle(
            p, admission, token_debit=3000, cost_micros=Micros(20_000), persist=persist
        )
        assert ctl.spent(bucket) == 20_000

    asyncio.run(scenario())


def test_persist_failure_keeps_reservations_and_disables_admission() -> None:
    ctl, _, _ = controller()
    p = principal()

    async def scenario() -> None:
        admission = await ctl.admit(p, ALL_ENFORCED, EST, first=True)

        def persist(_: tuple[object, ...]) -> bool:
            raise OSError("disk full")

        with pytest.raises(StorageUnavailable):
            await ctl.settle(p, admission, token_debit=1, cost_micros=Micros(1), persist=persist)
        assert ctl.reserved_tokens("k1") == EST.tokens
        with pytest.raises(StorageUnavailable):
            await ctl.admit(p, ALL_ENFORCED, EST, first=True)

    asyncio.run(scenario())


def test_observe_mode_commits_reservations_and_records_every_failure() -> None:
    ctl, _, _ = controller()
    observe = {p: EnforcementMode.OBSERVE for p in Policy}
    p = principal(rpm=1, tpm=500, budget=1, team_budget=1)
    asyncio.run(ctl.admit(p, observe, Estimate(0, 0, Micros(0)), first=True))
    admission = asyncio.run(ctl.admit(p, observe, Estimate(100, 100, Micros(10)), first=True))
    assert set(admission.would_deny) == {"rpm_exceeded", "budget_exceeded"}
    assert ctl.reserved_tokens("k1") == 200


def test_settlement_after_midnight_stays_in_the_pinned_period() -> None:
    start = datetime(2026, 11, 30, 23, 59, 59, tzinfo=UTC)
    ctl, wall, _ = controller(start)
    p = principal(budget=1_000_000, team_budget=1_000_000, key_cadence=Cadence.DAY)

    async def scenario() -> None:
        admission = await ctl.admit(p, ALL_ENFORCED, EST, first=True)
        assert (admission.key_period_id, admission.team_period_id) == (
            "day:2026-11-30",
            "month:2026-11",
        )
        wall.advance(2)  # now 2026-12-01T00:00:01Z
        await ctl.settle(p, admission, token_debit=1, cost_micros=Micros(7), persist=lambda _: True)
        assert ctl.spent(BucketKey(ScopeType.KEY, "k1", "day:2026-11-30")) == 7
        assert ctl.spent(BucketKey(ScopeType.TEAM, "t", "month:2026-11")) == 7
        assert ctl.spent(BucketKey(ScopeType.KEY, "k1", "day:2026-12-01")) == 0
        fresh = await ctl.admit(p, ALL_ENFORCED, EST, first=True)
        assert (fresh.key_period_id, fresh.team_period_id) == ("day:2026-12-01", "month:2026-12")

    asyncio.run(scenario())


def test_new_daily_key_period_does_not_bypass_exhausted_team_month() -> None:
    start = datetime(2026, 11, 15, 23, 0, tzinfo=UTC)
    ctl, wall, _ = controller(start)
    p = principal(budget=10**9, team_budget=10_000, key_cadence=Cadence.DAY)

    async def scenario() -> None:
        admission = await ctl.admit(p, ALL_ENFORCED, EST, first=True)
        await ctl.settle(
            p, admission, token_debit=1, cost_micros=Micros(10_000), persist=lambda _: True
        )
        wall.advance(3600 * 2)  # next UTC day, same month
        with pytest.raises(AdmissionDenied) as denied:
            await ctl.admit(p, ALL_ENFORCED, EST, first=True)
        assert denied.value.error.gateway_code == "budget_exceeded"

    asyncio.run(scenario())


def test_tpm_debit_enters_window_at_settlement_and_expires_60s_later() -> None:
    ctl, _, mono = controller()
    p = principal(tpm=1500)

    async def scenario() -> None:
        admission = await ctl.admit(p, ALL_ENFORCED, EST, first=True)
        mono.advance(120)  # a long stream: its reservation must not expire on its own
        with pytest.raises(AdmissionDenied):
            await ctl.admit(p, ALL_ENFORCED, EST, first=True)
        await ctl.settle(
            p, admission, token_debit=1000, cost_micros=Micros(0), persist=lambda _: True
        )
        mono.advance(59.9)
        with pytest.raises(AdmissionDenied) as denied:
            await ctl.admit(p, ALL_ENFORCED, EST, first=True)
        assert denied.value.error.retry_after_s == 1
        mono.advance(0.1)
        await ctl.admit(p, ALL_ENFORCED, EST, first=True)

    asyncio.run(scenario())


def test_load_spent_rebuilds_balances_after_restart() -> None:
    ctl, _, _ = controller()
    ctl.load_spent([(BucketKey(ScopeType.TEAM, "t", "month:2026-11"), 99_000)])
    with pytest.raises(AdmissionDenied):
        asyncio.run(ctl.admit(principal(team_budget=100_000), ALL_ENFORCED, EST, first=True))

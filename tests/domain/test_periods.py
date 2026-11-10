"""UTC calendar budget periods and rollover.

A day is ``[00:00 UTC, next 00:00 UTC)`` and a month is ``[00:00 UTC on the 1st,
00:00 UTC on the next 1st)``. IDs carry their cadence. An instant exactly on a
boundary belongs to the new period. Instants are passed in, never read from a clock.
"""

from datetime import UTC, datetime, timedelta, timezone

import pytest

from gate.domain.models import Budget, Cadence
from gate.domain.money import Micros
from gate.domain.periods import budget_period_id, is_period_id, period_bounds, period_id

ONE_US = timedelta(microseconds=1)


def utc(year: int, month: int, day: int, hour: int = 0, minute: int = 0, sec: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, sec, tzinfo=UTC)


def test_ids_carry_the_cadence() -> None:
    instant = utc(2026, 11, 30, 15, 4, 5)
    assert period_id(Cadence.DAY, instant) == "day:2026-11-30"
    assert period_id(Cadence.MONTH, instant) == "month:2026-11"


def test_instant_exactly_on_day_boundary_belongs_to_new_day() -> None:
    midnight = utc(2026, 12, 1)
    assert period_id(Cadence.DAY, midnight - ONE_US) == "day:2026-11-30"
    assert period_id(Cadence.DAY, midnight) == "day:2026-12-01"


def test_instant_exactly_on_month_boundary_belongs_to_new_month() -> None:
    first = utc(2026, 12, 1)
    assert period_id(Cadence.MONTH, first - ONE_US) == "month:2026-11"
    assert period_id(Cadence.MONTH, first) == "month:2026-12"


def test_year_rollover() -> None:
    new_year = utc(2027, 1, 1)
    assert period_id(Cadence.DAY, new_year - ONE_US) == "day:2026-12-31"
    assert period_id(Cadence.MONTH, new_year - ONE_US) == "month:2026-12"
    assert period_id(Cadence.DAY, new_year) == "day:2027-01-01"
    assert period_id(Cadence.MONTH, new_year) == "month:2027-01"


def test_leap_day_is_its_own_day_inside_february() -> None:
    leap_day = utc(2028, 2, 29, 12)
    assert period_id(Cadence.DAY, leap_day) == "day:2028-02-29"
    assert period_id(Cadence.MONTH, leap_day) == "month:2028-02"
    assert period_id(Cadence.DAY, utc(2028, 3, 1)) == "day:2028-03-01"
    assert period_id(Cadence.MONTH, utc(2028, 3, 1)) == "month:2028-03"
    assert period_bounds("day:2028-02-29") == (utc(2028, 2, 29), utc(2028, 3, 1))


@pytest.mark.parametrize(
    ("pid", "days"),
    [
        ("month:2026-01", 31),
        ("month:2026-02", 28),
        ("month:2028-02", 29),  # leap year
        ("month:2100-02", 28),  # century, not a leap year
        ("month:2000-02", 29),  # divisible by 400, a leap year
        ("month:2026-04", 30),
        ("month:2026-11", 30),
        ("month:2026-12", 31),
    ],
)
def test_month_lengths(pid: str, days: int) -> None:
    start, end = period_bounds(pid)
    assert end - start == timedelta(days=days)


def test_period_bounds_are_half_open_utc_intervals() -> None:
    assert period_bounds("day:2026-11-30") == (utc(2026, 11, 30), utc(2026, 12, 1))
    assert period_bounds("month:2026-11") == (utc(2026, 11, 1), utc(2026, 12, 1))
    assert period_bounds("month:2026-12") == (utc(2026, 12, 1), utc(2027, 1, 1))
    start, end = period_bounds("day:2026-11-30")
    assert start.tzinfo is UTC
    assert end.tzinfo is UTC


@pytest.mark.parametrize("cadence", list(Cadence))
def test_every_instant_lies_inside_the_bounds_of_its_own_period(cadence: Cadence) -> None:
    for instant in (utc(2026, 1, 31, 23, 59, 59), utc(2028, 2, 29), utc(2026, 12, 31, 23)):
        start, end = period_bounds(period_id(cadence, instant))
        assert start <= instant < end


@pytest.mark.parametrize("bad", ["week:2026-46", "day:2026-02-30", "month:2026-13", "", "2026-11"])
def test_unknown_or_invalid_period_ids(bad: str) -> None:
    with pytest.raises(ValueError):
        period_bounds(bad)
    assert not is_period_id(bad)


def test_valid_period_ids_are_recognised() -> None:
    assert is_period_id("day:2026-11-30")
    assert is_period_id("month:2026-11")


def test_naive_datetime_is_rejected() -> None:
    with pytest.raises(ValueError):
        period_id(Cadence.DAY, datetime(2026, 11, 30, 12))


def test_non_utc_instant_is_converted_to_utc_first() -> None:
    # 2026-12-01 01:30 in UTC+02:00 is still 2026-11-30 in UTC: the host or client
    # timezone never decides the budget period.
    plus_two = timezone(timedelta(hours=2))
    local = datetime(2026, 12, 1, 1, 30, tzinfo=plus_two)
    assert period_id(Cadence.DAY, local) == "day:2026-11-30"
    assert period_id(Cadence.MONTH, local) == "month:2026-11"

    # 2026-12-31 20:00 in UTC-05:00 is already 2027-01-01 01:00 UTC.
    minus_five = timezone(timedelta(hours=-5))
    late = datetime(2026, 12, 31, 20, tzinfo=minus_five)
    assert period_id(Cadence.DAY, late) == "day:2027-01-01"
    assert period_id(Cadence.MONTH, late) == "month:2027-01"


def test_key_and_team_periods_derive_independently_from_one_instant() -> None:
    instant = utc(2026, 11, 30, 23, 59, 59)
    key_budget = Budget(Micros(1_000_000), Cadence.DAY)
    team_budget = Budget(Micros(5_000_000), Cadence.MONTH)
    assert budget_period_id(key_budget, instant) == "day:2026-11-30"
    assert budget_period_id(team_budget, instant) == "month:2026-11"


def test_scope_without_a_budget_has_no_period() -> None:
    assert budget_period_id(None, utc(2026, 11, 30)) is None

"""UTC calendar budget periods.

Period IDs carry their cadence (``day:2026-11-30``, ``month:2026-11``) and are
derived from an admission instant passed in by the caller. An instant exactly
on a boundary belongs to the new period.
"""

from datetime import UTC, datetime, timedelta

from gate.domain.models import Budget, Cadence


def period_id(cadence: Cadence, instant: datetime) -> str:
    if instant.tzinfo is None:
        raise ValueError("budget periods require a timezone-aware instant")
    utc = instant.astimezone(UTC)
    if cadence is Cadence.DAY:
        return f"day:{utc:%Y-%m-%d}"
    return f"month:{utc:%Y-%m}"


def budget_period_id(budget: Budget | None, instant: datetime) -> str | None:
    """The pinned period for a scope, or None when that scope has no budget."""
    return None if budget is None else period_id(budget.cadence, instant)


def period_bounds(pid: str) -> tuple[datetime, datetime]:
    """Half-open UTC interval ``[start, end)`` for a period ID."""
    cadence, _, value = pid.partition(":")
    if cadence == Cadence.DAY:
        start = datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=UTC)
        return start, start + timedelta(days=1)
    if cadence == Cadence.MONTH:
        start = datetime.strptime(value, "%Y-%m").replace(tzinfo=UTC)
        if start.month == 12:
            end = start.replace(year=start.year + 1, month=1)
        else:
            end = start.replace(month=start.month + 1)
        return start, end
    raise ValueError(f"unknown period id: {pid}")


def is_period_id(value: str) -> bool:
    try:
        period_bounds(value)
    except ValueError:
        return False
    return True

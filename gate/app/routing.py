"""Alias resolution, deployment cooldown and fallback eligibility.

Aliases are exact model names mapping to an ordered list of deployments.
Cooldowns use monotonic time and come only from upstream ``Retry-After``.
"""

import email.utils
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime

from gate.domain.models import Deployment, PriceEntry

FALLBACK_STATUSES = frozenset({429, 529})


@dataclass(frozen=True)
class Candidate:
    deployment: Deployment
    price: PriceEntry | None  # None only for unmetered routes

    @property
    def name(self) -> str:
        return self.deployment.name


class Router:
    def __init__(
        self,
        deployments: Mapping[str, Deployment],
        aliases: Mapping[str, tuple[str, ...]],
        prices: Mapping[tuple[str, str], PriceEntry],
    ) -> None:
        self._deployments = deployments
        self._aliases = aliases
        self._prices = prices
        self._cooldown_until: dict[str, float] = {}

    def routed(self, model: str) -> bool:
        return model in self._aliases

    def candidates(self, model: str) -> list[Candidate]:
        """All deployments for a model, in configured order, with their pinned price entry."""
        return [
            Candidate(self._deployments[name], self._prices.get((name, model)))
            for name in self._aliases.get(model, ())
        ]

    def available(self, candidates: Sequence[Candidate], now: float) -> list[Candidate]:
        return [c for c in candidates if self._cooldown_until.get(c.name, 0.0) <= now]

    def cooldown_remaining(self, candidates: Sequence[Candidate], now: float) -> float:
        remaining = [self._cooldown_until.get(c.name, 0.0) - now for c in candidates]
        return max(0.0, min(remaining, default=0.0))

    def cool_down(self, deployment: str, seconds: float, now: float) -> None:
        if seconds > 0:
            self._cooldown_until[deployment] = max(
                self._cooldown_until.get(deployment, 0.0), now + seconds
            )


def fallback_eligible_status(status: int) -> bool:
    """Retry eligibility only: an eligible failure is not necessarily free."""
    return status >= 500 or status in FALLBACK_STATUSES


def parse_retry_after(value: str | None, now: datetime) -> float | None:
    """``Retry-After`` as delta-seconds or an HTTP date; None if absent or invalid."""
    if not value:
        return None
    value = value.strip()
    if value.isdigit():
        return float(value)
    try:
        when = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        return None
    return max(0.0, (when - now).total_seconds())

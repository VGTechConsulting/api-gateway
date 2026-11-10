"""Reservation estimates and usage pricing.

Token cost is calculated from five categories only: input, output, cache read,
5-minute cache write and 1-hour cache write. The provider's aggregate
cache-write counter is validation data and is never priced on top of its split.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from decimal import Decimal

from gate.domain.models import (
    PRICING_STATUS_PRIORITY,
    CacheBreakdown,
    PriceEntry,
    PricingMode,
    PricingStatus,
    UsageSnapshot,
)
from gate.domain.money import Micros, ceil_div, round_half_up, token_cost

BYTES_PER_TOKEN_ESTIMATE = 4


@dataclass(frozen=True)
class Estimate:
    input_tokens: int
    output_tokens: int
    micros: Micros

    @property
    def tokens(self) -> int:
        return self.input_tokens + self.output_tokens


def estimate(body_size: int, output_limit: int, price: PriceEntry) -> Estimate:
    """Heuristic reservation, not an upper bound."""
    est_input = ceil_div(body_size, BYTES_PER_TOKEN_ESTIMATE)
    if price.pricing_mode is PricingMode.FREE:
        return Estimate(est_input, output_limit, Micros(0))
    cost = token_cost(est_input, price.input) + token_cost(output_limit, price.output)
    return Estimate(est_input, output_limit, round_half_up(cost))


@dataclass(frozen=True)
class PricingResult:
    cost_micros: Micros | None
    status: PricingStatus
    reasons: tuple[str, ...]


def primary_status(statuses: Iterable[PricingStatus]) -> PricingStatus:
    present = set(statuses)
    for status in PRICING_STATUS_PRIORITY:
        if status in present:
            return status
    return PricingStatus.OK


def price_usage(
    snapshot: UsageSnapshot | None,
    price: PriceEntry,
    extra_unsupported: Iterable[str] = (),
) -> PricingResult:
    """Price the latest valid cumulative snapshot; cost is None when it cannot be priced."""
    reasons: list[tuple[PricingStatus, str]] = [
        (PricingStatus.UNSUPPORTED, reason) for reason in extra_unsupported
    ]
    if snapshot is None:
        reasons.append((PricingStatus.UNAVAILABLE, "no_usage"))
        return _result(None, reasons)

    reasons.extend((PricingStatus.UNSUPPORTED, reason) for reason in snapshot.unsupported)
    if snapshot.cache_breakdown is CacheBreakdown.MISSING:
        reasons.append((PricingStatus.MISSING_CACHE_BREAKDOWN, "cache_write_split_missing"))
    elif snapshot.cache_breakdown is CacheBreakdown.INCONSISTENT:
        reasons.append((PricingStatus.INCONSISTENT_CACHE_BREAKDOWN, "cache_write_split_invalid"))

    counts = snapshot.counts
    categories: tuple[tuple[str, int | None, Decimal | None], ...] = (
        ("input", counts.input, price.input),
        ("output", counts.output, price.output),
        ("cache_read", counts.cache_read, price.cache_read),
        ("cache_write_5m", counts.cache_write_5m, price.cache_write_5m),
        ("cache_write_1h", counts.cache_write_1h, price.cache_write_1h),
    )
    total = Decimal(0)
    for name, count, rate in categories:
        if count is None:
            if snapshot.cache_breakdown is CacheBreakdown.OK or not name.startswith("cache_write"):
                reasons.append((PricingStatus.UNAVAILABLE, f"{name}_unknown"))
            continue
        if count == 0:
            continue  # zero contributes zero, even in an unsupported category
        if rate is None:
            reasons.append((PricingStatus.UNSUPPORTED, f"{name}_not_priced"))
            continue
        total += token_cost(count, rate)

    if reasons:
        return _result(None, reasons)
    if price.pricing_mode is PricingMode.FREE:
        return PricingResult(Micros(0), PricingStatus.OK, ())
    return PricingResult(round_half_up(total), PricingStatus.OK, ())


def _result(cost: Micros | None, reasons: list[tuple[PricingStatus, str]]) -> PricingResult:
    return PricingResult(
        cost,
        primary_status(status for status, _ in reasons),
        tuple(reason for _, reason in reasons),
    )

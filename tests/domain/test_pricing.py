"""Reservation estimates and usage pricing: the pricing contract, cost calculation and
pricing modes.

Rates are USD per million tokens, so ``tokens x rate`` is micro-dollars. Five
categories are priced; the provider's cache-write aggregate is validation data and is
never priced on top of its 5-minute/1-hour split. Rounding happens once, half up.
"""

from decimal import Decimal

import pytest

from gate.domain.models import (
    CacheBreakdown,
    PriceEntry,
    PricingMode,
    PricingStatus,
    TokenCounts,
    UsageSnapshot,
)
from gate.domain.money import Micros
from gate.domain.pricing import Estimate, estimate, price_usage, primary_status


def price(
    *,
    input: str = "3",
    output: str = "15",
    cache_read: str | None = "0.30",
    cache_write_5m: str | None = "3.75",
    cache_write_1h: str | None = "6",
    mode: PricingMode = PricingMode.METERED,
) -> PriceEntry:
    def rate(value: str | None) -> Decimal | None:
        return None if value is None else Decimal(value)

    return PriceEntry(
        deployment="anthropic-direct",
        model="claude-sonnet-5-5",
        pricing_mode=mode,
        input=Decimal(input),
        output=Decimal(output),
        cache_read=rate(cache_read),
        cache_write_5m=rate(cache_write_5m),
        cache_write_1h=rate(cache_write_1h),
    )


def usage(
    *,
    input: int | None = 1_000,
    output: int | None = 500,
    cache_read: int | None = 0,
    m5: int | None = 0,
    h1: int | None = 0,
    aggregate: int | None = None,
    breakdown: CacheBreakdown = CacheBreakdown.OK,
    unsupported: tuple[str, ...] = (),
) -> UsageSnapshot:
    if aggregate is None and m5 is not None and h1 is not None:
        aggregate = m5 + h1
    counts = TokenCounts(
        input=input,
        output=output,
        cache_read=cache_read,
        cache_write=aggregate,
        cache_write_5m=m5,
        cache_write_1h=h1,
    )
    return UsageSnapshot(counts, breakdown, unsupported)


# --- estimate ----------------------------------------------------------------------------


def test_estimate_formula() -> None:
    # ceil(401 / 4) = 101 input tokens; output = the explicit limit.
    est = estimate(body_size=401, output_limit=100, price=price())
    assert est == Estimate(input_tokens=101, output_tokens=100, micros=Micros(101 * 3 + 100 * 15))
    assert est.tokens == 201


@pytest.mark.parametrize(("size", "tokens"), [(0, 0), (1, 1), (4, 1), (5, 2), (8, 2)])
def test_estimate_rounds_input_tokens_up(size: int, tokens: int) -> None:
    assert estimate(size, 0, price()).input_tokens == tokens


def test_estimate_ignores_cache_rates() -> None:
    cheap = estimate(4_000, 10, price(cache_read="0.01", cache_write_1h="100"))
    assert cheap.micros == 1_000 * 3 + 10 * 15


def test_estimate_rounds_once_half_up() -> None:
    # 1 input token x $0.25 + 1 output token x $0.25 = 0.5 micro-dollars -> 1.
    est = estimate(4, 1, price(input="0.25", output="0.25"))
    assert est.micros == 1


def test_free_mode_estimate_has_zero_money_but_full_tokens() -> None:
    free = price(
        input="0",
        output="0",
        cache_read="0",
        cache_write_5m="0",
        cache_write_1h="0",
        mode=PricingMode.FREE,
    )
    est = estimate(400, 1_000, free)
    assert est.micros == 0
    assert est.tokens == 100 + 1_000  # free mode still reserves TPM


# --- price_usage: cache write durations ---------------------------------------------------


def test_zero_cache_writes() -> None:
    result = price_usage(usage(), price())
    assert result.status is PricingStatus.OK
    assert result.cost_micros == 1_000 * 3 + 500 * 15  # 10,500
    assert result.reasons == ()


def test_five_minute_cache_writes() -> None:
    result = price_usage(usage(m5=200), price())
    assert result.cost_micros == 10_500 + 200 * Decimal("3.75")  # 11,250


def test_one_hour_cache_writes() -> None:
    result = price_usage(usage(h1=200), price())
    assert result.cost_micros == 10_500 + 200 * 6  # 11,700


def test_mixed_cache_writes_and_reads() -> None:
    result = price_usage(usage(cache_read=10_000, m5=100, h1=100), price())
    # 10,000 x 0.30 + 100 x 3.75 + 100 x 6 = 3,000 + 375 + 600
    assert result.cost_micros == 10_500 + 3_000 + 375 + 600


def test_aggregate_is_never_priced_on_top_of_its_split() -> None:
    with_split = price_usage(usage(m5=100, h1=100, aggregate=200), price())
    assert with_split.cost_micros == 10_500 + 375 + 600


def test_deployment_specific_rates_change_the_cost() -> None:
    snapshot = usage(m5=100)
    a = price_usage(snapshot, price(input="3"))
    b = price_usage(snapshot, price(input="0.27"))
    assert a.cost_micros == 3_000 + 7_500 + 375
    assert b.cost_micros == 270 + 7_500 + 375


# --- price_usage: uncertainty -------------------------------------------------------------


def test_missing_breakdown_cannot_be_priced() -> None:
    snapshot = usage(m5=None, h1=None, aggregate=200, breakdown=CacheBreakdown.MISSING)
    result = price_usage(snapshot, price())
    assert result.cost_micros is None
    assert result.status is PricingStatus.MISSING_CACHE_BREAKDOWN
    assert result.reasons == ("cache_write_split_missing",)


def test_inconsistent_breakdown_cannot_be_priced() -> None:
    snapshot = usage(m5=None, h1=None, aggregate=200, breakdown=CacheBreakdown.INCONSISTENT)
    result = price_usage(snapshot, price())
    assert result.cost_micros is None
    assert result.status is PricingStatus.INCONSISTENT_CACHE_BREAKDOWN


def test_null_rate_with_zero_count_is_priceable() -> None:
    deepseek = price(cache_write_5m=None, cache_write_1h=None)
    result = price_usage(usage(cache_read=100), deepseek)
    assert result.status is PricingStatus.OK
    assert result.cost_micros == 10_500 + 30


def test_null_rate_with_nonzero_count_is_unsupported() -> None:
    deepseek = price(cache_write_5m=None, cache_write_1h=None)
    result = price_usage(usage(m5=1), deepseek)
    assert result.cost_micros is None
    assert result.status is PricingStatus.UNSUPPORTED
    assert "cache_write_5m_not_priced" in result.reasons


def test_unknown_counter_makes_cost_unavailable() -> None:
    result = price_usage(usage(output=None), price())
    assert result.cost_micros is None
    assert result.status is PricingStatus.UNAVAILABLE


def test_no_snapshot_is_unavailable() -> None:
    result = price_usage(None, price())
    assert result.cost_micros is None
    assert result.status is PricingStatus.UNAVAILABLE


def test_unsupported_condition_reported_by_the_response() -> None:
    result = price_usage(usage(unsupported=("service_tier_priority",)), price())
    assert result.cost_micros is None
    assert result.status is PricingStatus.UNSUPPORTED
    assert result.reasons == ("service_tier_priority",)


def test_extra_unsupported_reason_such_as_a_model_mismatch() -> None:
    result = price_usage(usage(), price(), extra_unsupported=("upstream_model_mismatch",))
    assert result.cost_micros is None
    assert result.status is PricingStatus.UNSUPPORTED


def test_all_reasons_are_kept_and_the_primary_status_follows_priority() -> None:
    snapshot = usage(
        m5=None,
        h1=None,
        aggregate=200,
        breakdown=CacheBreakdown.MISSING,
        unsupported=("service_tier_priority",),
    )
    result = price_usage(snapshot, price())
    assert result.status is PricingStatus.UNSUPPORTED
    assert set(result.reasons) == {"service_tier_priority", "cache_write_split_missing"}


@pytest.mark.parametrize(
    ("statuses", "primary"),
    [
        ([PricingStatus.OK], PricingStatus.OK),
        ([], PricingStatus.OK),
        (
            [PricingStatus.MISSING_CACHE_BREAKDOWN, PricingStatus.INCONSISTENT_CACHE_BREAKDOWN],
            PricingStatus.INCONSISTENT_CACHE_BREAKDOWN,
        ),
        (
            [PricingStatus.INCONSISTENT_CACHE_BREAKDOWN, PricingStatus.UNAVAILABLE],
            PricingStatus.UNAVAILABLE,
        ),
        (
            [
                PricingStatus.MISSING_CACHE_BREAKDOWN,
                PricingStatus.UNAVAILABLE,
                PricingStatus.UNSUPPORTED,
                PricingStatus.OK,
            ],
            PricingStatus.UNSUPPORTED,
        ),
        (
            [PricingStatus.OK, PricingStatus.MISSING_CACHE_BREAKDOWN],
            PricingStatus.MISSING_CACHE_BREAKDOWN,
        ),
    ],
)
def test_primary_status_priority(statuses: list[PricingStatus], primary: PricingStatus) -> None:
    assert primary_status(statuses) is primary


# --- rounding and free mode ---------------------------------------------------------------


def test_cost_is_rounded_once_on_the_sum() -> None:
    # Each category costs 0.4 micro-dollars: per-category rounding would give 0.
    tiny = price(input="0.4", output="0.4", cache_read="0.4")
    result = price_usage(usage(input=1, output=1, cache_read=1), tiny)
    assert result.cost_micros == 1


def test_half_micro_dollar_rounds_up() -> None:
    result = price_usage(usage(input=1, output=0), price(input="2.5"))
    assert result.cost_micros == 3


def test_free_mode_prices_complete_usage_at_zero() -> None:
    free = price(
        input="0",
        output="0",
        cache_read="0",
        cache_write_5m="0",
        cache_write_1h=None,
        mode=PricingMode.FREE,
    )
    result = price_usage(usage(input=10_000, output=5_000, m5=10), free)
    assert result.status is PricingStatus.OK
    assert result.cost_micros == 0


def test_free_mode_null_rate_is_still_unsupported() -> None:
    free = price(
        input="0",
        output="0",
        cache_read="0",
        cache_write_5m="0",
        cache_write_1h=None,
        mode=PricingMode.FREE,
    )
    result = price_usage(usage(h1=10), free)
    assert result.cost_micros is None
    assert result.status is PricingStatus.UNSUPPORTED

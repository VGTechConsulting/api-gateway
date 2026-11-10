"""Attempt settlement and parent summaries: the outcome table, token settlement, parent
``usage`` rules, budget settlement, interrupted streams and failure accounting.

An attempt is classified from evidence first, then debited. Missing usage never proves
zero cost. Money and token debit sources are independent of each other.
"""

from decimal import Decimal

import pytest

from gate.domain.accounting import (
    AttemptEvidence,
    AttemptSummaryInput,
    Settlement,
    settle_generation,
    settle_unmetered,
    summarize,
)
from gate.domain.models import (
    AccountingKind,
    CacheBreakdown,
    Outcome,
    PriceEntry,
    PricingMode,
    PricingStatus,
    TokenCounts,
    TokenDebitSource,
    UsageSnapshot,
    UsageSource,
)
from gate.domain.money import Micros
from gate.domain.pricing import Estimate

METERED = PriceEntry(
    deployment="anthropic-direct",
    model="claude-sonnet-5-5",
    pricing_mode=PricingMode.METERED,
    input=Decimal("3"),
    output=Decimal("15"),
    cache_read=Decimal("0.30"),
    cache_write_5m=Decimal("3.75"),
    cache_write_1h=Decimal("6"),
)
FREE = PriceEntry(
    deployment="mock",
    model="mock-model",
    pricing_mode=PricingMode.FREE,
    input=Decimal(0),
    output=Decimal(0),
    cache_read=Decimal(0),
    cache_write_5m=Decimal(0),
    cache_write_1h=Decimal(0),
)

# 100 input x $3 + 50 output x $15 = 300 + 750 = 1,050 micro-dollars; 150 tokens.
SNAPSHOT = UsageSnapshot(
    TokenCounts(
        input=100, output=50, cache_read=0, cache_write=0, cache_write_5m=0, cache_write_1h=0
    )
)
SNAPSHOT_COST = 1_050
SNAPSHOT_TOKENS = 150


def est(micros: int, tokens: int = 1_000) -> Estimate:
    return Estimate(input_tokens=tokens - 100, output_tokens=100, micros=Micros(micros))


def complete(snapshot: UsageSnapshot = SNAPSHOT) -> AttemptEvidence:
    return AttemptEvidence(snapshot=snapshot, usage_complete=True, usage_source=UsageSource.OK)


def partial(snapshot: UsageSnapshot = SNAPSHOT) -> AttemptEvidence:
    return AttemptEvidence(snapshot=snapshot, usage_source=UsageSource.PARTIAL)


def events(settlement: Settlement, kind: str) -> list[dict[str, object]]:
    return [dict(e) for e in settlement.events if e.get("event") == kind]


# --- complete usage ----------------------------------------------------------------------


@pytest.mark.parametrize("reservation", [2_000, SNAPSHOT_COST], ids=["below", "equal"])
def test_complete_usage_at_or_below_reservation_records_the_calculated_cost(
    reservation: int,
) -> None:
    s = settle_generation(complete(), est(reservation), METERED)
    assert s.outcome is Outcome.COMPLETE
    assert s.accounting_kind is AccountingKind.CALCULATED
    assert s.usage_complete is True
    assert s.usage_source is UsageSource.OK
    assert s.pricing_status is PricingStatus.OK
    assert s.observed_cost_micros == SNAPSHOT_COST
    assert s.priced_cost_micros == SNAPSHOT_COST
    assert s.cost_micros == SNAPSHOT_COST  # not the reservation
    assert events(s, "reservation_underestimated") == []


def test_complete_usage_above_reservation_records_the_full_cost() -> None:
    s = settle_generation(complete(), est(1_000), METERED)
    assert s.accounting_kind is AccountingKind.CALCULATED
    assert s.cost_micros == SNAPSHOT_COST  # never clamped to the reservation
    assert events(s, "reservation_underestimated") == [
        {"event": "reservation_underestimated", "difference_micros": 50}
    ]


def test_complete_usage_token_debit_is_the_reported_total() -> None:
    s = settle_generation(complete(), est(2_000, tokens=1_000), METERED)
    assert s.observed_total_tokens == SNAPSHOT_TOKENS
    assert s.token_debit == SNAPSHOT_TOKENS  # below the 1,000-token reservation
    assert s.token_debit_source is TokenDebitSource.REPORTED


def test_complete_usage_above_token_reservation_is_not_capped() -> None:
    s = settle_generation(complete(), est(2_000, tokens=120), METERED)
    assert s.token_debit == SNAPSHOT_TOKENS
    assert s.token_debit_source is TokenDebitSource.REPORTED
    assert events(s, "token_underestimated") == [
        {"event": "token_underestimated", "difference_tokens": 30}
    ]


def test_token_total_counts_every_category_exactly_once() -> None:
    snapshot = UsageSnapshot(
        TokenCounts(
            input=10,
            output=20,
            cache_read=1_000,
            cache_write=300,
            cache_write_5m=100,
            cache_write_1h=200,
        )
    )
    s = settle_generation(complete(snapshot), est(1_000_000), METERED)
    # 10 + 20 + 1,000 + 100 + 200; the 300 aggregate is not added on top.
    assert s.observed_total_tokens == 1_330
    assert s.token_debit == 1_330


# --- partial and missing usage -----------------------------------------------------------


def test_partial_usage_below_estimate_debits_the_reservation() -> None:
    s = settle_generation(partial(), est(2_000), METERED)
    assert s.outcome is Outcome.PARTIAL
    assert s.accounting_kind is AccountingKind.ESTIMATED
    assert s.usage_complete is False
    assert s.observed_cost_micros == SNAPSHOT_COST  # an observation, not a charge
    assert s.priced_cost_micros is None
    assert s.cost_micros == 2_000
    (debit,) = events(s, "estimated_debit")
    assert debit["set_by"] == "reservation"
    assert debit["usage_source"] == "partial"


def test_partial_usage_above_estimate_debits_the_observed_cost() -> None:
    s = settle_generation(partial(), est(400), METERED)
    assert s.cost_micros == SNAPSHOT_COST
    assert s.priced_cost_micros is None
    (debit,) = events(s, "estimated_debit")
    assert debit["set_by"] == "observed_cost"
    assert events(s, "reservation_underestimated")[0]["difference_micros"] == 650


def test_unpriceable_partial_usage_debits_the_reservation() -> None:
    snapshot = UsageSnapshot(
        TokenCounts(input=100_000, output=50_000, cache_read=0, cache_write=200),
        CacheBreakdown.MISSING,
    )
    s = settle_generation(partial(snapshot), est(400), METERED)
    assert s.observed_cost_micros is None
    assert s.cost_micros == 400
    assert s.pricing_status is PricingStatus.MISSING_CACHE_BREAKDOWN


def test_missing_usage_debits_the_reservation() -> None:
    s = settle_generation(AttemptEvidence(), est(400, tokens=900), METERED)
    assert s.outcome is Outcome.UNKNOWN
    assert s.accounting_kind is AccountingKind.ESTIMATED
    assert s.cost_micros == 400
    assert s.observed_cost_micros is None
    assert s.pricing_status is PricingStatus.UNAVAILABLE
    assert s.observed_total_tokens is None
    assert s.token_debit == 900
    assert s.token_debit_source is TokenDebitSource.ESTIMATED


def test_partial_token_debit_is_the_greater_of_estimate_and_observed() -> None:
    low = settle_generation(partial(), est(2_000, tokens=1_000), METERED)
    assert low.token_debit == 1_000
    assert low.token_debit_source is TokenDebitSource.ESTIMATED
    high = settle_generation(partial(), est(2_000, tokens=120), METERED)
    assert high.token_debit == SNAPSHOT_TOKENS
    assert high.token_debit_source is TokenDebitSource.ESTIMATED


def test_parse_error_after_a_valid_snapshot_keeps_the_observed_cost() -> None:
    evidence = AttemptEvidence(snapshot=SNAPSHOT, usage_source=UsageSource.PARSE_ERROR)
    s = settle_generation(evidence, est(400), METERED)
    assert s.accounting_kind is AccountingKind.ESTIMATED
    assert s.usage_source is UsageSource.PARSE_ERROR
    assert s.cost_micros == SNAPSHOT_COST


def test_usage_complete_flag_without_ok_source_is_not_complete() -> None:
    evidence = AttemptEvidence(
        snapshot=SNAPSHOT, usage_complete=True, usage_source=UsageSource.TOO_LARGE
    )
    s = settle_generation(evidence, est(2_000), METERED)
    assert s.usage_complete is False
    assert s.accounting_kind is AccountingKind.ESTIMATED


# --- zero-cost outcomes ------------------------------------------------------------------


def assert_non_billable(s: Settlement) -> None:
    assert s.accounting_kind is AccountingKind.NON_BILLABLE
    assert s.cost_micros == 0
    assert s.priced_cost_micros == 0
    assert s.token_debit == 0
    assert s.token_debit_source is TokenDebitSource.NOT_APPLICABLE
    assert s.usage_source is UsageSource.NOT_APPLICABLE
    assert s.pricing_status is PricingStatus.NOT_APPLICABLE
    assert s.usage_complete is False
    assert s.observed_total_tokens is None
    assert s.counts == TokenCounts()  # no fabricated measurements: all unknown


def test_not_sent_is_non_billable() -> None:
    s = settle_generation(AttemptEvidence(not_sent=True), est(400), METERED)
    assert s.outcome is Outcome.NOT_SENT
    assert_non_billable(s)


def test_documented_rejection_is_non_billable() -> None:
    evidence = AttemptEvidence(rejection_basis="provider_auth_rejected")
    s = settle_generation(evidence, est(400), METERED)
    assert s.outcome is Outcome.REJECTED_NON_BILLABLE
    assert s.billing_basis == "provider_auth_rejected"
    assert_non_billable(s)


def test_rejection_basis_is_ignored_when_usage_shows_execution() -> None:
    evidence = AttemptEvidence(
        rejection_basis="provider_auth_rejected",
        snapshot=SNAPSHOT,
        usage_source=UsageSource.PARTIAL,
    )
    s = settle_generation(evidence, est(400), METERED)
    assert s.outcome is not Outcome.REJECTED_NON_BILLABLE
    assert s.accounting_kind is AccountingKind.ESTIMATED
    assert s.cost_micros == SNAPSHOT_COST


def test_unmetered_endpoint_has_no_debit() -> None:
    s = settle_unmetered(AttemptEvidence(snapshot=SNAPSHOT, usage_source=UsageSource.OK))
    assert s.outcome is Outcome.UNMETERED
    assert_non_billable(s)


# --- cache-split token rules -------------------------------------------------------------


def test_aggregate_with_missing_split_is_counted_once_and_reported() -> None:
    snapshot = UsageSnapshot(
        TokenCounts(input=100, output=50, cache_read=0, cache_write=200),
        CacheBreakdown.MISSING,
    )
    s = settle_generation(complete(snapshot), est(5_000, tokens=1_000), METERED)
    # Cost is uncertain ...
    assert s.accounting_kind is AccountingKind.ESTIMATED
    assert s.priced_cost_micros is None
    assert s.pricing_status is PricingStatus.MISSING_CACHE_BREAKDOWN
    assert s.cost_micros == 5_000
    # ... but the token total is complete: the aggregate stands in for the split once.
    assert s.usage_complete is True
    assert s.observed_total_tokens == 350
    assert s.token_debit == 350
    assert s.token_debit_source is TokenDebitSource.REPORTED


def test_inconsistent_split_cannot_establish_a_complete_token_total() -> None:
    snapshot = UsageSnapshot(
        TokenCounts(input=100, output=50, cache_read=0, cache_write=200),
        CacheBreakdown.INCONSISTENT,
    )
    s = settle_generation(complete(snapshot), est(5_000, tokens=120), METERED)
    assert s.pricing_status is PricingStatus.INCONSISTENT_CACHE_BREAKDOWN
    assert s.observed_total_tokens == 150  # cache writes excluded, not guessed
    assert s.token_debit == 150  # max(estimate 120, observed subtotal 150)
    assert s.token_debit_source is TokenDebitSource.ESTIMATED


def test_unsupported_pricing_does_not_make_a_complete_token_total_estimated() -> None:
    deepseek = PriceEntry(
        deployment="deepseek",
        model="deepseek-flash",
        pricing_mode=PricingMode.METERED,
        input=Decimal("0.27"),
        output=Decimal("1.10"),
        cache_read=Decimal("0.07"),
        cache_write_5m=None,
        cache_write_1h=None,
    )
    snapshot = UsageSnapshot(
        TokenCounts(
            input=100, output=50, cache_read=0, cache_write=10, cache_write_5m=10, cache_write_1h=0
        )
    )
    s = settle_generation(complete(snapshot), est(5_000, tokens=1_000), deepseek)
    assert s.pricing_status is PricingStatus.UNSUPPORTED
    assert s.accounting_kind is AccountingKind.ESTIMATED
    assert s.token_debit == 160
    assert s.token_debit_source is TokenDebitSource.REPORTED


def test_unsupported_outranks_unavailable_without_a_snapshot() -> None:
    evidence = AttemptEvidence(extra_unsupported=("upstream_model_mismatch",))
    s = settle_generation(evidence, est(400), METERED)
    assert s.pricing_status is PricingStatus.UNSUPPORTED


# --- free mode ---------------------------------------------------------------------------


def test_free_mode_complete_costs_zero_but_debits_reported_tokens() -> None:
    s = settle_generation(complete(), est(0, tokens=1_000), FREE)
    assert s.outcome is Outcome.COMPLETE
    assert s.accounting_kind is AccountingKind.CALCULATED  # never non_billable for free
    assert s.cost_micros == 0
    assert s.priced_cost_micros == 0
    assert s.token_debit == SNAPSHOT_TOKENS
    assert s.token_debit_source is TokenDebitSource.REPORTED


def test_free_mode_incomplete_stream_keeps_the_estimated_token_debit() -> None:
    s = settle_generation(partial(), est(0, tokens=1_000), FREE)
    assert s.accounting_kind is AccountingKind.ESTIMATED
    assert s.cost_micros == 0
    assert s.token_debit == 1_000
    assert s.token_debit_source is TokenDebitSource.ESTIMATED


# --- summarize ---------------------------------------------------------------------------

FULL_COUNTS = TokenCounts(
    input=10, output=5, cache_read=0, cache_write=0, cache_write_5m=0, cache_write_1h=0
)


def calculated(cost: int = 100, counts: TokenCounts = FULL_COUNTS) -> AttemptSummaryInput:
    return AttemptSummaryInput(
        accounting_kind=AccountingKind.CALCULATED,
        generation=True,
        counts=counts,
        observed_cost_micros=Micros(cost),
        priced_cost_micros=Micros(cost),
        cost_micros=Micros(cost),
        estimate_micros=Micros(cost + 50),
        usage_source=UsageSource.OK,
        usage_complete=True,
        pricing_status=PricingStatus.OK,
    )


def estimated(
    cost: int = 400,
    counts: TokenCounts = FULL_COUNTS,
    observed: int | None = 50,
    source: UsageSource = UsageSource.PARTIAL,
    status: PricingStatus = PricingStatus.OK,
) -> AttemptSummaryInput:
    return AttemptSummaryInput(
        accounting_kind=AccountingKind.ESTIMATED,
        generation=True,
        counts=counts,
        observed_cost_micros=None if observed is None else Micros(observed),
        priced_cost_micros=None,
        cost_micros=Micros(cost),
        estimate_micros=Micros(cost),
        usage_source=source,
        usage_complete=False,
        pricing_status=status,
    )


def non_billable(generation: bool = True) -> AttemptSummaryInput:
    return AttemptSummaryInput(
        accounting_kind=AccountingKind.NON_BILLABLE,
        generation=generation,
        counts=TokenCounts(),
        observed_cost_micros=Micros(0),
        priced_cost_micros=Micros(0),
        cost_micros=Micros(0),
        estimate_micros=Micros(300),
        usage_source=UsageSource.NOT_APPLICABLE,
        usage_complete=False,
        pricing_status=PricingStatus.NOT_APPLICABLE,
    )


def test_only_zero_cost_failures_is_non_billable() -> None:
    totals = summarize([non_billable(), non_billable()])
    assert totals.accounting_kind is AccountingKind.NON_BILLABLE
    assert totals.cost_micros == 0
    assert totals.priced_cost_micros == 0
    assert totals.usage_complete is False  # zero-cost rejections are not complete usage
    assert totals.usage_source is UsageSource.NOT_APPLICABLE
    assert totals.pricing_status is PricingStatus.NOT_APPLICABLE
    assert totals.attempt_count == 2


def test_calculated_plus_zero_cost_failure_is_calculated() -> None:
    totals = summarize([non_billable(), calculated(100)])
    assert totals.accounting_kind is AccountingKind.CALCULATED
    assert totals.priced_cost_micros == 100
    assert totals.cost_micros == 100
    assert totals.usage_complete is True
    assert totals.usage_source is UsageSource.OK
    # Counters come from billable attempts only; the failure's unknown counters don't null them.
    assert totals.counts == FULL_COUNTS


def test_estimated_plus_zero_cost_failure_is_estimated() -> None:
    totals = summarize([non_billable(), estimated(400)])
    assert totals.accounting_kind is AccountingKind.ESTIMATED
    assert totals.priced_cost_micros is None
    assert totals.cost_micros == 400


def test_calculated_and_estimated_attempts_are_mixed() -> None:
    totals = summarize([estimated(400), calculated(100)])
    assert totals.accounting_kind is AccountingKind.MIXED
    assert totals.priced_cost_micros is None
    assert totals.cost_micros == 500  # the earlier estimate survives the later success
    assert totals.estimate_micros == 400 + 150
    assert totals.usage_complete is False
    assert totals.usage_source is UsageSource.PARTIAL  # not overwritten by the fallback


def test_all_calculated_attempts_sum_their_priced_cost() -> None:
    totals = summarize([calculated(100), calculated(250)])
    assert totals.accounting_kind is AccountingKind.CALCULATED
    assert totals.priced_cost_micros == 350
    assert totals.observed_cost_micros == 350
    assert totals.counts.input == 20


def test_interrupted_parent_has_null_priced_cost_and_is_incomplete() -> None:
    totals = summarize([calculated(100)], interrupted=True)
    assert totals.priced_cost_micros is None
    assert totals.usage_complete is False
    assert totals.cost_micros == 100  # committed spend is preserved


def test_counter_is_null_if_any_billable_attempt_lacks_it() -> None:
    missing_output = TokenCounts(
        input=7, output=None, cache_read=0, cache_write=0, cache_write_5m=0, cache_write_1h=0
    )
    totals = summarize([calculated(100), estimated(400, counts=missing_output)])
    assert totals.counts.input == 17
    assert totals.counts.output is None


def test_observed_cost_is_null_if_any_billable_attempt_is_unpriceable() -> None:
    totals = summarize([calculated(100), estimated(400, observed=None)])
    assert totals.observed_cost_micros is None


def test_parent_pricing_status_keeps_attempt_uncertainty() -> None:
    totals = summarize([estimated(status=PricingStatus.MISSING_CACHE_BREAKDOWN), calculated()])
    assert totals.pricing_status is PricingStatus.MISSING_CACHE_BREAKDOWN


@pytest.mark.parametrize(
    ("sources", "primary"),
    [
        ([UsageSource.OK, UsageSource.PARTIAL], UsageSource.PARTIAL),
        ([UsageSource.PARTIAL, UsageSource.MISSING], UsageSource.MISSING),
        ([UsageSource.MISSING, UsageSource.PARSE_ERROR], UsageSource.PARSE_ERROR),
        ([UsageSource.PARSE_ERROR, UsageSource.TOO_LARGE], UsageSource.TOO_LARGE),
    ],
)
def test_parent_usage_source_keeps_the_most_serious_attempt_status(
    sources: list[UsageSource], primary: UsageSource
) -> None:
    attempts = [estimated(source=source) for source in sources]
    assert summarize(attempts).usage_source is primary


def test_unmetered_attempt_has_no_priced_generation_cost() -> None:
    totals = summarize([non_billable(generation=False)])
    assert totals.accounting_kind is AccountingKind.NON_BILLABLE
    assert totals.cost_micros == 0
    assert totals.priced_cost_micros is None  # no generation attempt to price

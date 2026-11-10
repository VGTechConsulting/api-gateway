"""Outcome classification, money and token debits, parent summaries.

An attempt is classified first, from transport and provider evidence, and only
then debited. Missing usage alone never proves zero cost.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field

from gate.domain.models import (
    AccountingKind,
    CacheBreakdown,
    Outcome,
    PolicyEvent,
    PriceEntry,
    PricingStatus,
    TokenCounts,
    TokenDebitSource,
    UsageSnapshot,
    UsageSource,
)
from gate.domain.money import Micros
from gate.domain.policy import event
from gate.domain.pricing import Estimate, price_usage, primary_status


@dataclass(frozen=True)
class AttemptEvidence:
    """What the transport and the provider adapter established about one attempt."""

    not_sent: bool = False  # reliable proof no request bytes could have been sent
    rejection_basis: str | None = None  # documented provider pre-execution rejection
    snapshot: UsageSnapshot | None = None
    usage_complete: bool = False
    usage_source: UsageSource = UsageSource.MISSING
    extra_unsupported: tuple[str, ...] = ()  # e.g. unverified upstream model mismatch


@dataclass(frozen=True)
class Settlement:
    outcome: Outcome
    accounting_kind: AccountingKind
    billing_basis: str
    counts: TokenCounts
    usage_source: UsageSource
    usage_complete: bool
    pricing_status: PricingStatus
    observed_cost_micros: Micros | None
    priced_cost_micros: Micros | None
    cost_micros: Micros
    observed_total_tokens: int | None
    token_debit: int
    token_debit_source: TokenDebitSource
    events: tuple[PolicyEvent, ...] = field(default=())


def observed_total(snapshot: UsageSnapshot | None) -> tuple[int | None, bool]:
    """Sum of validated, non-overlapping counters, and whether that sum is a full total.

    A valid aggregate stands in once for a missing duration split. Inconsistent
    cache-write counters are excluded rather than choosing between them.
    """
    if snapshot is None:
        return None, False
    c = snapshot.counts
    parts: list[int | None] = [c.input, c.output, c.cache_read]
    if snapshot.cache_breakdown is CacheBreakdown.OK:
        parts += [c.cache_write_5m, c.cache_write_1h]
    elif snapshot.cache_breakdown is CacheBreakdown.MISSING:
        parts.append(c.cache_write)
    else:
        parts.append(None)
    known = [p for p in parts if p is not None]
    if not known:
        return None, False
    return sum(known), len(known) == len(parts)


_ZERO_COUNTS = TokenCounts()


def _non_billable(outcome: Outcome, basis: str) -> Settlement:
    return Settlement(
        outcome=outcome,
        accounting_kind=AccountingKind.NON_BILLABLE,
        billing_basis=basis,
        counts=_ZERO_COUNTS,
        usage_source=UsageSource.NOT_APPLICABLE,
        usage_complete=False,
        pricing_status=PricingStatus.NOT_APPLICABLE,
        observed_cost_micros=Micros(0),
        priced_cost_micros=Micros(0),
        cost_micros=Micros(0),
        observed_total_tokens=None,
        token_debit=0,
        token_debit_source=TokenDebitSource.NOT_APPLICABLE,
    )


def settle_unmetered(evidence: AttemptEvidence) -> Settlement:
    """count_tokens and other authenticated proxy calls: no token or money debit."""
    basis = "not_sent" if evidence.not_sent else "unmetered_endpoint"
    return _non_billable(Outcome.UNMETERED, basis)


def settle_generation(
    evidence: AttemptEvidence,
    estimate: Estimate,
    price: PriceEntry,
) -> Settlement:
    if evidence.not_sent:
        return _non_billable(Outcome.NOT_SENT, "connect_failed_before_send")
    if evidence.rejection_basis is not None and evidence.snapshot is None:
        return _non_billable(Outcome.REJECTED_NON_BILLABLE, evidence.rejection_basis)

    events: list[PolicyEvent] = []
    snapshot = evidence.snapshot
    pricing = price_usage(snapshot, price, evidence.extra_unsupported)
    pricing_status = pricing.status  # already ranked: unsupported > unavailable > ...
    for reason in pricing.reasons:
        events.append(event("pricing_uncertain", reason=reason))

    complete = evidence.usage_complete and evidence.usage_source is UsageSource.OK
    calculated = complete and pricing.status is PricingStatus.OK and pricing.cost_micros is not None

    if calculated:
        assert pricing.cost_micros is not None
        outcome = Outcome.COMPLETE
        kind = AccountingKind.CALCULATED
        priced: Micros | None = pricing.cost_micros
        cost = pricing.cost_micros
        basis = "complete_usage"
    else:
        outcome = Outcome.PARTIAL if snapshot is not None and not complete else Outcome.UNKNOWN
        kind = AccountingKind.ESTIMATED
        priced = None
        observed = pricing.cost_micros
        if observed is not None and observed > estimate.micros:
            cost, set_by = observed, "observed_cost"
        else:
            cost, set_by = estimate.micros, "reservation"
        basis = f"estimated_{outcome.value}"
        events.append(
            event(
                "estimated_debit",
                usage_source=evidence.usage_source.value,
                pricing_status=pricing_status.value,
                set_by=set_by,
                debit_micros=cost,
            )
        )

    if cost > estimate.micros:
        events.append(event("reservation_underestimated", difference_micros=cost - estimate.micros))

    total, total_is_full = observed_total(snapshot)
    if complete and total is not None and total_is_full:
        token_debit, token_source = total, TokenDebitSource.REPORTED
    else:
        token_debit = estimate.tokens if total is None else max(estimate.tokens, total)
        token_source = TokenDebitSource.ESTIMATED
    if token_debit > estimate.tokens:
        events.append(
            event("token_underestimated", difference_tokens=token_debit - estimate.tokens)
        )

    return Settlement(
        outcome=outcome,
        accounting_kind=kind,
        billing_basis=basis,
        counts=snapshot.counts if snapshot is not None else _ZERO_COUNTS,
        usage_source=evidence.usage_source,
        usage_complete=complete,
        pricing_status=pricing_status,
        observed_cost_micros=pricing.cost_micros,
        priced_cost_micros=priced,
        cost_micros=cost,
        observed_total_tokens=total,
        token_debit=token_debit,
        token_debit_source=token_source,
        events=tuple(events),
    )


# --- Parent summary --------------------------------------------------------------------


@dataclass(frozen=True)
class AttemptSummaryInput:
    """The attempt fields a parent summary needs; built from ``AttemptRecord``."""

    accounting_kind: AccountingKind
    generation: bool
    counts: TokenCounts
    observed_cost_micros: Micros | None
    priced_cost_micros: Micros | None
    cost_micros: Micros
    estimate_micros: Micros
    usage_source: UsageSource
    usage_complete: bool
    pricing_status: PricingStatus


@dataclass(frozen=True)
class ParentTotals:
    attempt_count: int
    counts: TokenCounts
    observed_cost_micros: Micros | None
    priced_cost_micros: Micros | None
    cost_micros: Micros
    estimate_micros: Micros
    usage_source: UsageSource
    usage_complete: bool
    pricing_status: PricingStatus
    accounting_kind: AccountingKind


_USAGE_SOURCE_PRIORITY = (
    UsageSource.TOO_LARGE,
    UsageSource.PARSE_ERROR,
    UsageSource.MISSING,
    UsageSource.PARTIAL,
    UsageSource.OK,
)


def _sum_optional(values: Sequence[int | None]) -> int | None:
    if any(v is None for v in values):
        return None
    return sum(v for v in values if v is not None)


def summarize(
    attempts: Sequence[AttemptSummaryInput], *, interrupted: bool = False
) -> ParentTotals:
    """Fold attempt debits into one parent row; the parent is a summary, not a second charge."""
    billable = [a for a in attempts if a.accounting_kind is not AccountingKind.NON_BILLABLE]
    kinds = {a.accounting_kind for a in billable}
    if not billable:
        kind = AccountingKind.NON_BILLABLE
    elif kinds == {AccountingKind.CALCULATED}:
        kind = AccountingKind.CALCULATED
    elif kinds == {AccountingKind.ESTIMATED}:
        kind = AccountingKind.ESTIMATED
    else:
        kind = AccountingKind.MIXED

    def field_sum(name: str) -> int | None:
        return _sum_optional([getattr(a.counts, name) for a in billable]) if billable else None

    counts = TokenCounts(
        input=field_sum("input"),
        output=field_sum("output"),
        cache_read=field_sum("cache_read"),
        cache_write=field_sum("cache_write"),
        cache_write_5m=field_sum("cache_write_5m"),
        cache_write_1h=field_sum("cache_write_1h"),
    )

    generation = [a for a in attempts if a.generation]
    all_calculated = bool(generation) and all(
        a.accounting_kind in (AccountingKind.CALCULATED, AccountingKind.NON_BILLABLE)
        for a in generation
    )
    priced = _sum_optional([a.priced_cost_micros for a in generation]) if all_calculated else None
    if interrupted:
        priced = None
    observed = _sum_optional([a.observed_cost_micros for a in billable]) if billable else None

    if billable:
        usage_complete = all(a.usage_complete for a in billable) and not interrupted
        usage_source = next(
            s for s in _USAGE_SOURCE_PRIORITY if s in {a.usage_source for a in billable}
        )
        pricing_status = primary_status(a.pricing_status for a in billable)
    else:
        usage_complete = False
        usage_source = UsageSource.NOT_APPLICABLE
        pricing_status = PricingStatus.NOT_APPLICABLE

    return ParentTotals(
        attempt_count=len(attempts),
        counts=counts,
        observed_cost_micros=Micros(observed) if observed is not None else None,
        priced_cost_micros=Micros(priced) if priced is not None else None,
        cost_micros=Micros(sum(a.cost_micros for a in attempts)),
        estimate_micros=Micros(sum(a.estimate_micros for a in attempts)),
        usage_source=usage_source,
        usage_complete=usage_complete,
        pricing_status=pricing_status,
        accounting_kind=kind,
    )

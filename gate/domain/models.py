"""Core domain types and status enums shared across the gate.

Status values are enums, never bare strings. Every type here is a frozen
dataclass so a value can be passed between layers without being mutated.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import StrEnum

from gate.domain.money import Micros

# A policy decision as stored in ``policy_decisions`` JSON: flat, content-free fields.
PolicyEvent = Mapping[str, str | int | bool | None]


class ApiFormat(StrEnum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"


class Route(StrEnum):
    MESSAGES = "/v1/messages"
    COUNT_TOKENS = "/v1/messages/count_tokens"

    @property
    def metered(self) -> bool:
        return self is Route.MESSAGES


class Env(StrEnum):
    DEV = "dev"
    STAGING = "staging"
    PROD = "prod"


class Cadence(StrEnum):
    DAY = "day"
    MONTH = "month"


class Policy(StrEnum):
    ALLOWLIST = "allowlist"
    RATE_LIMITS = "rate_limits"
    BUDGETS = "budgets"


class EnforcementMode(StrEnum):
    OBSERVE = "observe"
    ENFORCE = "enforce"


class DetectMode(StrEnum):
    OFF = "off"
    FLAG = "flag"
    BLOCK = "block"


class PricingMode(StrEnum):
    METERED = "metered"
    FREE = "free"


class Outcome(StrEnum):
    NOT_SENT = "not_sent"
    REJECTED_NON_BILLABLE = "rejected_non_billable"
    COMPLETE = "complete"
    PARTIAL = "partial"
    UNKNOWN = "unknown"
    UNMETERED = "unmetered"


class UsageSource(StrEnum):
    OK = "ok"
    PARTIAL = "partial"
    MISSING = "missing"
    PARSE_ERROR = "parse_error"
    TOO_LARGE = "too_large"
    NOT_APPLICABLE = "not_applicable"


class PricingStatus(StrEnum):
    OK = "ok"
    MISSING_CACHE_BREAKDOWN = "missing_cache_breakdown"
    INCONSISTENT_CACHE_BREAKDOWN = "inconsistent_cache_breakdown"
    UNSUPPORTED = "unsupported"
    UNAVAILABLE = "unavailable"
    NOT_APPLICABLE = "not_applicable"


# Highest priority first.
PRICING_STATUS_PRIORITY: tuple[PricingStatus, ...] = (
    PricingStatus.UNSUPPORTED,
    PricingStatus.UNAVAILABLE,
    PricingStatus.INCONSISTENT_CACHE_BREAKDOWN,
    PricingStatus.MISSING_CACHE_BREAKDOWN,
    PricingStatus.OK,
)


class AccountingKind(StrEnum):
    CALCULATED = "calculated"
    ESTIMATED = "estimated"
    NON_BILLABLE = "non_billable"
    MIXED = "mixed"


class TokenDebitSource(StrEnum):
    REPORTED = "reported"
    ESTIMATED = "estimated"
    NOT_APPLICABLE = "not_applicable"


class TerminationReason(StrEnum):
    NORMAL = "normal"
    CLIENT_DISCONNECT = "client_disconnect"
    UPSTREAM_DISCONNECT = "upstream_disconnect"
    UPSTREAM_TIMEOUT = "upstream_timeout"
    CONNECTION_ERROR = "connection_error"
    PROVIDER_REJECTION = "provider_rejection"
    STREAM_ERROR = "stream_error"
    SHUTDOWN = "shutdown"


class RequestState(StrEnum):
    OPEN = "open"
    FINISHED = "finished"
    INTERRUPTED = "interrupted"


class CacheBreakdown(StrEnum):
    """Validity of a usage snapshot's cache-write duration split."""

    OK = "ok"
    MISSING = "missing"
    INCONSISTENT = "inconsistent"


@dataclass(frozen=True)
class Budget:
    limit: Micros
    cadence: Cadence


@dataclass(frozen=True)
class Principal:
    """Who is calling, resolved from a virtual key."""

    key_id: str
    prefix: str
    owner: str
    team: str
    app: str
    env: Env
    allowed_models: frozenset[str]
    budget: Budget | None
    team_budget: Budget | None
    tpm: int
    rpm: int


@dataclass(frozen=True)
class Deployment:
    name: str
    format: ApiFormat
    base_url: str
    api_key_env: str
    provider: str
    region: str
    upstream_model: str | None = None


@dataclass(frozen=True)
class PriceEntry:
    """Rates for one (deployment, requested model), USD per million tokens.

    ``None`` means the category is unsupported: a nonzero count in it makes the
    attempt's cost unpriceable. It never means free.
    """

    deployment: str
    model: str
    pricing_mode: PricingMode
    input: Decimal
    output: Decimal
    cache_read: Decimal | None
    cache_write_5m: Decimal | None
    cache_write_1h: Decimal | None
    response_models: frozenset[str] = frozenset()


@dataclass(frozen=True)
class TokenCounts:
    """Normalised token counters; ``None`` is unknown, never zero.

    ``input`` excludes cache reads and writes. ``cache_write`` is the provider's
    aggregate, kept for validation only and never priced on top of its split.
    """

    input: int | None = None
    output: int | None = None
    cache_read: int | None = None
    cache_write: int | None = None
    cache_write_5m: int | None = None
    cache_write_1h: int | None = None


@dataclass(frozen=True)
class UsageSnapshot:
    """The latest valid cumulative usage a response reported."""

    counts: TokenCounts
    cache_breakdown: CacheBreakdown = CacheBreakdown.OK
    unsupported: tuple[str, ...] = ()


@dataclass(frozen=True)
class AttemptRecord:
    """One settled upstream attempt: immutable once written."""

    request_id: str
    attempt_no: int
    ts_admitted: datetime
    ts_start: datetime
    ts_end: datetime
    key_period_id: str | None
    team_period_id: str | None
    deployment: str
    provider: str
    region: str
    model: str
    upstream_model: str | None
    upstream_status: int | None
    outcome: Outcome
    billing_basis: str
    accounting_kind: AccountingKind
    pricing_mode: PricingMode | None
    stream: bool
    stream_complete: bool | None
    usage_complete: bool
    termination_reason: TerminationReason
    counts: TokenCounts
    estimate_tokens: int
    observed_total_tokens: int | None
    token_debit: int
    token_debit_source: TokenDebitSource
    observed_cost_micros: Micros | None
    priced_cost_micros: Micros | None
    estimate_micros: Micros
    cost_micros: Micros
    usage_source: UsageSource
    pricing_status: PricingStatus
    client_disconnected: bool
    policy_decisions: tuple[PolicyEvent, ...] = field(default=())

    @property
    def generation(self) -> bool:
        """A metered generation attempt, as opposed to an unmetered proxy call."""
        return self.outcome is not Outcome.UNMETERED

    @property
    def potentially_billable(self) -> bool:
        return self.accounting_kind is not AccountingKind.NON_BILLABLE

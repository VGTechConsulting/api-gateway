"""Chargeback, budget, coverage and request reports as view models.

The CLI and the read-only report UI both render these rows; neither computes numbers
of its own, so the two always agree.

Rules that keep the numbers honest:

* Spend and tokens come from attempts; request counts, latency and denials from parents.
* Attempt metrics are filtered by attempt ``ts_admitted``; request metrics by parent
  ``ts_start``. A request can therefore count without its in-window attempts, or the
  reverse, near a ``--since``/``--until`` boundary.
* Calculated costs, estimated debits and partial observations are kept apart.
* The cache-write aggregate is never added to its duration split; unknown counters
  are counted as unknown, never shown as zero.
* Grouped distinct-request counts are not additive: one request can span providers.
"""

from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from gate.app.ports import (
    AttemptWithParent,
    KeyRepository,
    ParentRow,
    RequestFilter,
    UsageRepository,
)
from gate.domain.config import Team
from gate.domain.models import (
    AccountingKind,
    AttemptRecord,
    Outcome,
    PricingMode,
    PricingStatus,
    RequestState,
    TokenDebitSource,
    UsageSource,
)
from gate.domain.periods import period_id

ATTEMPT_GROUPS = ("model", "deployment", "provider")
PARENT_GROUPS = ("team", "app", "owner", "key", "all")  # "all": one row for everything
GROUPS = PARENT_GROUPS + ATTEMPT_GROUPS

# How a parent request ended, for the requests feed. "blocked" means the gate refused
# it (enforce mode); "error" is any other 4xx/5xx, including relayed upstream errors.
REQUEST_STATUSES = ("ok", "blocked", "error", "open", "interrupted")

Row = dict[str, Any]


def request_status(p: ParentRow) -> str:
    if p.request_state is RequestState.OPEN:
        return "open"
    if p.request_state is RequestState.INTERRUPTED:
        return "interrupted"
    if _blocked(p):
        return "blocked"
    return "error" if (p.http_status or 0) >= 400 else "ok"


def budget_limit_for(
    scope: str, teams: Mapping[str, Team], keys: KeyRepository
) -> Callable[[str], int | None]:
    """The configured budget, in micros, for a team name or key ID."""

    def limit(ident: str) -> int | None:
        if scope == "team":
            team = teams.get(ident)
            return team.budget.limit if team and team.budget else None
        key = keys.get(ident)
        return key.budget_micros if key else None

    return limit


def percentile(values: Sequence[float], pct: float) -> float | None:
    """Nearest-rank percentile; None for no values."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, min(len(ordered), int(-(-pct * len(ordered) // 100))))
    return ordered[rank - 1]


def _denied_code(row: ParentRow) -> str | None:
    """The gate-generated error a request ended with, if any."""
    for decision in row.policy_decisions:
        if decision.get("event") == "gate_error":
            code = decision.get("gateway_code")
            return str(code) if code is not None else None
    return None


def _local_denial(row: ParentRow) -> bool:
    """Ended with an error before any attempt was dispatched (coverage exclusion)."""
    return (row.totals is None or row.totals.attempt_count == 0) and (row.http_status or 0) >= 400


def _blocked(row: ParentRow) -> bool:
    """Refused by the gate: a local denial, or an enforced denial at any admission.

    A fallback admission can deny after earlier attempts ran, so an enforced ``deny``
    decision counts whether or not attempts exist; upstream failures relayed after an
    attempt carry no such decision and are not counted.
    """
    if (row.http_status or 0) < 400:
        return False
    return _local_denial(row) or any(d.get("result") == "deny" for d in row.policy_decisions)


def _executable(a: AttemptRecord) -> bool:
    """A generation attempt that may have executed (coverage denominator)."""
    return a.generation and a.accounting_kind is not AccountingKind.NON_BILLABLE


def _usage_ok(a: AttemptRecord) -> bool:
    return a.usage_source is UsageSource.OK and a.usage_complete


def _cost_ok(a: AttemptRecord) -> bool:
    return _usage_ok(a) and a.pricing_status is PricingStatus.OK


def _metered(a: AttemptRecord) -> bool:
    return a.pricing_mode is PricingMode.METERED


@dataclass
class _Group:
    requests: set[str]
    attempts: list[AttemptRecord]
    parents: list[ParentRow]


class Reports:
    def __init__(self, usage: UsageRepository) -> None:
        self._usage = usage

    # --- report --by ------------------------------------------------------------------

    def by(self, group: str, *, since: datetime | None, until: datetime | None) -> list[Row]:
        if group not in GROUPS:
            raise ValueError(f"unknown group {group!r}")
        attempts = self._usage.attempts(since=since, until=until)
        parents = self._usage.requests(since=since, until=until)

        groups: dict[str, _Group] = {}

        def bucket(name: str | None) -> _Group:
            return groups.setdefault(name or "-", _Group(set(), [], []))

        for item in attempts:
            bucket(_attempt_group(item, group)).attempts.append(item.attempt)
        for parent in parents:
            bucket(_parent_group(parent, group)).parents.append(parent)
        for request_id, names in self._request_groups(group, parents, since, until, attempts):
            for name in names:
                bucket(name).requests.add(request_id)

        return [self._group_row(group, name, g) for name, g in sorted(groups.items())]

    def _request_groups(
        self,
        group: str,
        parents: Sequence[ParentRow],
        since: datetime | None,
        until: datetime | None,
        attempts: Sequence[AttemptWithParent],
    ) -> Iterable[tuple[str, set[str | None]]]:
        """Groups each parent request counts in; requests are selected by ``ts_start``.

        A parent group is read off the parent itself. For an attempt group a request
        counts once in every group one of its attempts ran in, whenever those attempts
        were admitted; a request without attempts (a local denial) counts in its
        parent's group.
        """
        if group in PARENT_GROUPS:
            return ((p.request_id, {_parent_group(p, group)}) for p in parents)
        # Attempts are admitted after their parent started, so a parent's attempts are
        # all admitted at or after ``since``; only the upper bound needs lifting.
        candidates = attempts if until is None else self._usage.attempts(since=since, until=None)
        spanned: dict[str, set[str | None]] = {p.request_id: set() for p in parents}
        for item in candidates:
            names = spanned.get(item.attempt.request_id)
            if names is not None:
                names.add(_attempt_group(item, group))
        return ((p.request_id, spanned[p.request_id] or {_parent_group(p, group)}) for p in parents)

    def _group_row(self, group: str, name: str, g: _Group) -> Row:
        attempts = g.attempts
        complete = [a for a in attempts if _usage_ok(a)]
        partial = [a for a in attempts if a.potentially_billable and not _usage_ok(a)]
        free = [a for a in attempts if a.pricing_mode is PricingMode.FREE]

        def total(values: Iterable[int | None]) -> int:
            return sum(v for v in values if v is not None)

        unknown_counters = sum(
            1 for a in complete if None in (a.counts.input, a.counts.output, a.counts.cache_read)
        )
        unclassified_writes = total(
            a.counts.cache_write for a in complete if a.counts.cache_write_5m is None
        )
        overheads = [p.gate_overhead_ms for p in g.parents if p.gate_overhead_ms is not None]
        return {
            group: name,
            "requests": len(g.requests),
            "attempts": len(attempts),
            "input_tokens": total(a.counts.input for a in complete),
            "output_tokens": total(a.counts.output for a in complete),
            "cache_read_tokens": total(a.counts.cache_read for a in complete),
            "cache_write_5m_tokens": total(a.counts.cache_write_5m for a in complete),
            "cache_write_1h_tokens": total(a.counts.cache_write_1h for a in complete),
            "cache_write_unclassified_tokens": unclassified_writes,
            "attempts_with_unknown_counters": unknown_counters,
            "calculated_cost_usd": _usd(
                sum(
                    a.cost_micros
                    for a in attempts
                    if a.accounting_kind is AccountingKind.CALCULATED
                )
            ),
            "estimated_debit_usd": _usd(
                sum(
                    a.cost_micros for a in attempts if a.accounting_kind is AccountingKind.ESTIMATED
                )
            ),
            "total_debit_usd": _usd(sum(a.cost_micros for a in attempts)),
            "partial_attempts": len(partial),
            "partial_observed_tokens": total(a.observed_total_tokens for a in partial),
            "partial_observed_cost_usd": _usd(total(a.observed_cost_micros for a in partial)),
            "non_billable_by_reason": dict(
                Counter(
                    a.billing_basis
                    for a in attempts
                    if a.outcome in (Outcome.NOT_SENT, Outcome.REJECTED_NON_BILLABLE)
                )
            ),
            "pricing_uncertain_by_reason": dict(
                Counter(
                    a.pricing_status.value
                    for a in attempts
                    if a.potentially_billable and a.pricing_status is not PricingStatus.OK
                )
            ),
            "token_debit_reported": sum(
                a.token_debit for a in attempts if a.token_debit_source is TokenDebitSource.REPORTED
            ),
            "token_debit_estimated": sum(
                a.token_debit
                for a in attempts
                if a.token_debit_source is TokenDebitSource.ESTIMATED
            ),
            "free_mode_attempts": len(free),
            "free_mode_observed_tokens": total(a.observed_total_tokens for a in free),
            "blocked_requests_by_reason": dict(
                Counter(_denied_code(p) or "unknown" for p in g.parents if _blocked(p))
            ),
            "would_deny_by_code": dict(Counter(code for p in g.parents for code in p.would_deny)),
            "gate_overhead_p50_ms": percentile(overheads, 50),
            "gate_overhead_p95_ms": percentile(overheads, 95),
        }

    # --- report budgets ---------------------------------------------------------------

    def budgets(
        self,
        scope: str,
        period_id: str,
        limit_for: Callable[[str], int | None],
    ) -> list[Row]:
        """Debits in one stored period ID, by key or team; never sums the two views."""
        if scope not in ("key", "team"):
            raise ValueError("scope must be key or team")
        rows: dict[str, list[AttemptRecord]] = {}
        for item in self._usage.attempts_for_period(period_id):
            a = item.attempt
            stored = a.key_period_id if scope == "key" else a.team_period_id
            ident = item.key_id if scope == "key" else item.team
            if stored != period_id or ident is None:
                continue
            rows.setdefault(ident, []).append(a)
        out: list[Row] = []
        for ident, attempts in sorted(rows.items()):
            spent = sum(a.cost_micros for a in attempts)
            limit = limit_for(ident)
            out.append(
                {
                    scope: ident,
                    "period_id": period_id,
                    "attempts": len(attempts),
                    "calculated_usd": _usd(
                        sum(
                            a.cost_micros
                            for a in attempts
                            if a.accounting_kind is AccountingKind.CALCULATED
                        )
                    ),
                    "estimated_usd": _usd(
                        sum(
                            a.cost_micros
                            for a in attempts
                            if a.accounting_kind is AccountingKind.ESTIMATED
                        )
                    ),
                    "total_debit_usd": _usd(spent),
                    "budget_usd": _usd(limit) if limit is not None else None,
                    "overrun_usd": _usd(max(0, spent - limit)) if limit is not None else None,
                }
            )
        return out

    def team_budgets_now(self, teams: Mapping[str, Team], now: datetime) -> list[Row]:
        """Each budgeted team in its current period, including teams with no debits yet.

        Rows have the ``budgets`` shape. Teams can use different cadences, so each team
        is read in its own current period ID.
        """
        limits = {name: t.budget.limit for name, t in teams.items() if t.budget is not None}
        found: dict[str, dict[str, Row]] = {}
        out: list[Row] = []
        for name, team in sorted(teams.items()):
            if team.budget is None:
                continue
            pid = period_id(team.budget.cadence, now)
            if pid not in found:
                rows = self.budgets("team", pid, limits.get)
                found[pid] = {row["team"]: row for row in rows}
            out.append(
                found[pid].get(name)
                or {
                    "team": name,
                    "period_id": pid,
                    "attempts": 0,
                    "calculated_usd": _usd(0),
                    "estimated_usd": _usd(0),
                    "total_debit_usd": _usd(0),
                    "budget_usd": _usd(team.budget.limit),
                    "overrun_usd": _usd(0),
                }
            )
        return out

    # --- report requests / report request ---------------------------------------------

    def recent_requests(
        self, where: RequestFilter, *, limit: int, status: str | None = None
    ) -> list[Row]:
        """The newest matching parent requests; reads only as many rows as it needs."""
        if status is not None and status not in REQUEST_STATUSES:
            raise ValueError(f"unknown status {status!r}")
        parents: list[ParentRow] = []
        for parent in self._usage.latest_requests(where):
            if status is None or request_status(parent) == status:
                parents.append(parent)
                if len(parents) >= limit:
                    break
        attempts: dict[str, list[AttemptRecord]] = {}
        for item in self._usage.attempts_for_requests([p.request_id for p in parents]):
            attempts.setdefault(item.attempt.request_id, []).append(item.attempt)
        return [_request_row(p, attempts.get(p.request_id, [])) for p in parents]

    def request_detail(self, request_id: str) -> Row | None:
        """One request with every attempt; None if the ID is unknown."""
        parent = self._usage.request(request_id)
        if parent is None:
            return None
        attempts = [i.attempt for i in self._usage.attempts_for_requests([request_id])]
        row = _request_row(parent, attempts)
        row.update(
            {
                "request_state": parent.request_state.value,
                "ts_end": _iso(parent.ts_end),
                "session_id": parent.session_id,
                "route": parent.route,
                "stream": parent.stream,
                "stream_complete": parent.stream_complete,
                "termination_reason": parent.termination_reason,
                "upstream_status": parent.upstream_status,
                "ttfb_ms": parent.ttfb_ms,
                "client_disconnected": parent.client_disconnected,
                "policy_decisions": [dict(d) for d in parent.policy_decisions],
                "attempt_list": [_attempt_row(a) for a in attempts],
            }
        )
        return row

    # --- report coverage --------------------------------------------------------------

    def coverage(self, *, since: datetime | None, until: datetime | None) -> Row:
        attempts = self._usage.attempts(since=since, until=until)
        parents = self._usage.requests(since=since, until=until)
        records = [item.attempt for item in attempts]

        usage_pool = [a for a in records if _executable(a)]
        cost_pool = [a for a in usage_pool if _metered(a)]
        finished = {item.attempt.request_id: item.request_finished for item in attempts}

        def request_level(pool: list[AttemptRecord], rule: Callable[[AttemptRecord], bool]) -> Row:
            by_request: dict[str, list[AttemptRecord]] = {}
            for a in pool:
                by_request.setdefault(a.request_id, []).append(a)
            covered = sum(
                1
                for rid, items in by_request.items()
                if finished.get(rid) and all(rule(a) for a in items)
            )
            return _ratio(covered, len(by_request))

        interrupted = [p for p in parents if p.request_state is RequestState.INTERRUPTED]
        return {
            "attempt_usage_coverage": _ratio(sum(map(_usage_ok, usage_pool)), len(usage_pool)),
            "attempt_cost_coverage_metered": _ratio(sum(map(_cost_ok, cost_pool)), len(cost_pool)),
            "request_usage_coverage": request_level(usage_pool, _usage_ok),
            "request_cost_coverage_metered": request_level(cost_pool, _cost_ok),
            "free_mode_attempts": sum(1 for a in usage_pool if not _metered(a)),
            "excluded_non_billable_attempts": sum(
                1 for a in records if a.generation and not a.potentially_billable
            ),
            "excluded_unmetered_attempts": sum(1 for a in records if not a.generation),
            "excluded_local_denials": sum(1 for p in parents if _local_denial(p)),
            "interrupted_requests": len(interrupted),
            "interrupted_requests_without_attempts": sum(
                1 for p in interrupted if p.totals is None or p.totals.attempt_count == 0
            ),
            "calculated_debit_usd": _usd(
                sum(
                    a.cost_micros for a in records if a.accounting_kind is AccountingKind.CALCULATED
                )
            ),
            "estimated_debit_usd": _usd(
                sum(a.cost_micros for a in records if a.accounting_kind is AccountingKind.ESTIMATED)
            ),
        }


def _request_row(p: ParentRow, attempts: Sequence[AttemptRecord]) -> Row:
    """One parent request for the feed. Money comes from its attempts, split by kind.

    Token counts are the parent's summary of its billable attempts: ``None`` means
    unknown, and ``usage`` says whether they are complete, partial, or not applicable
    (no billable attempt ran).
    """
    totals = p.totals
    counts = totals.counts if totals else None
    providers = sorted({a.provider for a in attempts}) or ([p.provider] if p.provider else [])
    if totals is None or totals.accounting_kind is AccountingKind.NON_BILLABLE:
        usage = "n/a"
    else:
        usage = "complete" if totals.usage_complete else "partial"
    return {
        "request_id": p.request_id,
        "ts_start": _iso(p.ts_start),
        "status": request_status(p),
        "http_status": p.http_status,
        "key_prefix": p.key_prefix,
        "owner": p.owner,
        "team": p.team,
        "app": p.app,
        "env": p.env,
        "model": p.model,
        "providers": ",".join(providers),
        "attempts": len(attempts),
        "usage": usage,
        "input_tokens": counts.input if counts else None,
        "output_tokens": counts.output if counts else None,
        "cache_read_tokens": counts.cache_read if counts else None,
        "cache_write_5m_tokens": counts.cache_write_5m if counts else None,
        "cache_write_1h_tokens": counts.cache_write_1h if counts else None,
        "calculated_cost_usd": _usd(
            sum(a.cost_micros for a in attempts if a.accounting_kind is AccountingKind.CALCULATED)
        ),
        "estimated_debit_usd": _usd(
            sum(a.cost_micros for a in attempts if a.accounting_kind is AccountingKind.ESTIMATED)
        ),
        "total_debit_usd": _usd(sum(a.cost_micros for a in attempts)),
        "pricing_status": totals.pricing_status.value if totals else None,
        "denied_code": _denied_code(p),
        "would_deny": ",".join(p.would_deny),
        "detections": dict(p.detections or {}),
        "gate_overhead_ms": p.gate_overhead_ms,
        "latency_ms": p.latency_ms,
    }


def _attempt_row(a: AttemptRecord) -> Row:
    return {
        "attempt_no": a.attempt_no,
        "ts_admitted": _iso(a.ts_admitted),
        "ts_end": _iso(a.ts_end),
        "deployment": a.deployment,
        "provider": a.provider,
        "region": a.region,
        "model": a.model,
        "upstream_model": a.upstream_model,
        "upstream_status": a.upstream_status,
        "outcome": a.outcome.value,
        "billing_basis": a.billing_basis,
        "accounting_kind": a.accounting_kind.value,
        "pricing_mode": a.pricing_mode.value if a.pricing_mode else None,
        "pricing_status": a.pricing_status.value,
        "usage_source": a.usage_source.value,
        "usage_complete": a.usage_complete,
        "termination_reason": a.termination_reason.value,
        "input_tokens": a.counts.input,
        "output_tokens": a.counts.output,
        "cache_read_tokens": a.counts.cache_read,
        "cache_write_5m_tokens": a.counts.cache_write_5m,
        "cache_write_1h_tokens": a.counts.cache_write_1h,
        "token_debit": a.token_debit,
        "token_debit_source": a.token_debit_source.value,
        "estimate_usd": _usd(a.estimate_micros),
        "debit_usd": _usd(a.cost_micros),
        "observed_cost_usd": (
            _usd(a.observed_cost_micros) if a.observed_cost_micros is not None else None
        ),
        "key_period_id": a.key_period_id,
        "team_period_id": a.team_period_id,
        "policy_decisions": [dict(d) for d in a.policy_decisions],
    }


def _iso(value: datetime | None) -> str | None:
    return value.isoformat(timespec="milliseconds") if value is not None else None


def _attempt_group(item: AttemptWithParent, group: str) -> str | None:
    a = item.attempt
    values: Mapping[str, str | None] = {
        "all": "all",
        "model": a.model,
        "deployment": a.deployment,
        "provider": a.provider,
        "team": item.team,
        "app": item.app,
        "owner": item.owner,
        "key": item.key_id,
    }
    return values[group]


def _parent_group(p: ParentRow, group: str) -> str | None:
    values: Mapping[str, str | None] = {
        "all": "all",
        "model": p.model,
        "deployment": p.deployment,
        "provider": p.provider,
        "team": p.team,
        "app": p.app,
        "owner": p.owner,
        "key": p.key_id,
    }
    return values[group]


def _usd(micros: int) -> str:
    sign = "-" if micros < 0 else ""
    micros = abs(micros)
    return f"{sign}{micros // 1_000_000}.{micros % 1_000_000:06d}"


def _ratio(numerator: int, denominator: int) -> Row:
    return {
        "covered": numerator,
        "eligible": denominator,
        "pct": round(100 * numerator / denominator, 2) if denominator else None,
    }

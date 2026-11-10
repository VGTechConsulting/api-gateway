"""SQLite schema, migrations and repositories.

SQLite is the accounting source of truth. WAL mode and a busy timeout let the
CLI write keys while the server runs. Each settled attempt and its parent's
running totals are written in one transaction.
"""

import json
import sqlite3
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from gate.app.ports import AttemptWithParent, KeyRecord, ParentRow, RequestFilter
from gate.domain.accounting import ParentTotals
from gate.domain.ledger import BucketKey, ScopeType
from gate.domain.models import (
    AccountingKind,
    AttemptRecord,
    Cadence,
    Env,
    Outcome,
    PricingMode,
    PricingStatus,
    RequestState,
    TerminationReason,
    TokenCounts,
    TokenDebitSource,
    UsageSource,
)
from gate.domain.money import Micros

SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS keys (
    id TEXT PRIMARY KEY,
    key_hash TEXT NOT NULL UNIQUE,
    prefix TEXT NOT NULL,
    owner TEXT NOT NULL,
    team TEXT NOT NULL,
    app TEXT NOT NULL,
    env TEXT NOT NULL CHECK (env IN ('dev', 'staging', 'prod')),
    allowed_models TEXT NOT NULL,
    budget_micros INTEGER,
    period TEXT NOT NULL CHECK (period IN ('day', 'month')),
    tpm INTEGER CHECK (tpm IS NULL OR tpm > 0),
    rpm INTEGER CHECK (rpm IS NULL OR rpm > 0),
    created_at TEXT NOT NULL,
    revoked_at TEXT
);
CREATE INDEX IF NOT EXISTS keys_prefix ON keys (prefix);

CREATE TABLE IF NOT EXISTS usage (
    request_id TEXT PRIMARY KEY,
    request_state TEXT NOT NULL,
    ts_start TEXT NOT NULL,
    ts_end TEXT,
    key_id TEXT, key_prefix TEXT, owner TEXT, team TEXT, app TEXT, env TEXT,
    session_id TEXT,
    route TEXT NOT NULL,
    model TEXT, upstream_model TEXT, deployment TEXT, provider TEXT, region TEXT,
    http_status INTEGER, upstream_status INTEGER,
    stream INTEGER, stream_complete INTEGER, usage_complete INTEGER,
    termination_reason TEXT,
    input_tokens INTEGER, output_tokens INTEGER, cache_read_tokens INTEGER,
    cache_write_tokens INTEGER, cache_write_5m_tokens INTEGER, cache_write_1h_tokens INTEGER,
    observed_cost_micros INTEGER, priced_cost_micros INTEGER,
    cost_micros INTEGER NOT NULL DEFAULT 0, estimate_micros INTEGER NOT NULL DEFAULT 0,
    usage_source TEXT, pricing_status TEXT, accounting_kind TEXT,
    attempt_count INTEGER NOT NULL DEFAULT 0,
    ttfb_ms REAL, latency_ms REAL, gate_overhead_ms REAL,
    fallback_count INTEGER NOT NULL DEFAULT 0,
    client_disconnected INTEGER NOT NULL DEFAULT 0,
    policy_decisions TEXT NOT NULL DEFAULT '[]',
    would_deny TEXT NOT NULL DEFAULT '[]',
    detections TEXT
);
CREATE INDEX IF NOT EXISTS usage_team_ts ON usage (team, ts_start);
CREATE INDEX IF NOT EXISTS usage_key_ts ON usage (key_id, ts_start);

CREATE TABLE IF NOT EXISTS attempts (
    request_id TEXT NOT NULL REFERENCES usage (request_id),
    attempt_no INTEGER NOT NULL,
    ts_admitted TEXT NOT NULL, ts_start TEXT NOT NULL, ts_end TEXT NOT NULL,
    key_period_id TEXT, team_period_id TEXT,
    deployment TEXT NOT NULL, provider TEXT NOT NULL, region TEXT NOT NULL,
    model TEXT NOT NULL, upstream_model TEXT, upstream_status INTEGER,
    outcome TEXT NOT NULL, billing_basis TEXT NOT NULL, accounting_kind TEXT NOT NULL,
    pricing_mode TEXT,
    stream INTEGER NOT NULL, stream_complete INTEGER, usage_complete INTEGER NOT NULL,
    termination_reason TEXT NOT NULL,
    input_tokens INTEGER, output_tokens INTEGER, cache_read_tokens INTEGER,
    cache_write_tokens INTEGER, cache_write_5m_tokens INTEGER, cache_write_1h_tokens INTEGER,
    estimate_tokens INTEGER NOT NULL, observed_total_tokens INTEGER,
    token_debit INTEGER NOT NULL, token_debit_source TEXT NOT NULL,
    observed_cost_micros INTEGER, priced_cost_micros INTEGER,
    estimate_micros INTEGER NOT NULL, cost_micros INTEGER NOT NULL,
    usage_source TEXT NOT NULL, pricing_status TEXT NOT NULL,
    client_disconnected INTEGER NOT NULL,
    policy_decisions TEXT NOT NULL,
    PRIMARY KEY (request_id, attempt_no)
);
CREATE INDEX IF NOT EXISTS attempts_key_period ON attempts (key_period_id, request_id);
CREATE INDEX IF NOT EXISTS attempts_team_period ON attempts (team_period_id, request_id);
CREATE INDEX IF NOT EXISTS attempts_admitted ON attempts (ts_admitted);
"""


def ts(value: datetime | None) -> str | None:
    """Fixed-width UTC ISO 8601, so text comparison orders correctly."""
    if value is None:
        return None
    return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def parse_ts(value: str | None) -> datetime | None:
    if value is None:
        return None
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=UTC)


def _bool(value: bool | None) -> int | None:
    return None if value is None else int(value)


def _opt_bool(value: Any) -> bool | None:
    return None if value is None else bool(value)


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def connect(path: str) -> sqlite3.Connection:
    if path != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None, timeout=5.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def connect_readonly(path: str) -> sqlite3.Connection:
    """Open an existing database read-only for the report UI.

    SQLite itself rejects writes on this connection. It runs no migrations and sets no
    journal mode; WAL readers still need the ``-shm`` file next to the database, so the
    directory must be writable even though this connection never writes.
    """
    uri = Path(path).resolve().as_uri() + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True, check_same_thread=False, timeout=5.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA query_only=ON")
    return conn


def migrate(conn: sqlite3.Connection) -> None:
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    if version > SCHEMA_VERSION:
        raise RuntimeError(f"database schema {version} is newer than this build ({SCHEMA_VERSION})")
    conn.executescript(SCHEMA)
    conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")


class SqliteKeyRepository:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def find_active_by_hash(self, key_hash: str) -> KeyRecord | None:
        row = self._conn.execute(
            "SELECT * FROM keys WHERE key_hash = ? AND revoked_at IS NULL", (key_hash,)
        ).fetchone()
        return _key(row) if row else None

    def insert(self, record: KeyRecord) -> None:
        self._conn.execute(
            "INSERT INTO keys (id, key_hash, prefix, owner, team, app, env, allowed_models,"
            " budget_micros, period, tpm, rpm, created_at, revoked_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                record.id,
                record.key_hash,
                record.prefix,
                record.owner,
                record.team,
                record.app,
                record.env.value,
                _json(list(record.allowed_models)),
                record.budget_micros,
                record.period.value,
                record.tpm,
                record.rpm,
                ts(record.created_at),
                ts(record.revoked_at),
            ),
        )

    def list(self, *, team: str | None, include_revoked: bool) -> Sequence[KeyRecord]:
        sql = "SELECT * FROM keys WHERE 1=1"
        params: list[Any] = []
        if team is not None:
            sql += " AND team = ?"
            params.append(team)
        if not include_revoked:
            sql += " AND revoked_at IS NULL"
        sql += " ORDER BY created_at"
        return [_key(row) for row in self._conn.execute(sql, params)]

    def find_by_id_or_prefix(self, ident: str) -> Sequence[KeyRecord]:
        rows = self._conn.execute("SELECT * FROM keys WHERE id = ? OR prefix = ?", (ident, ident))
        return [_key(row) for row in rows]

    def revoke(self, key_id: str, at: datetime) -> bool:
        cur = self._conn.execute(
            "UPDATE keys SET revoked_at = ? WHERE id = ? AND revoked_at IS NULL", (ts(at), key_id)
        )
        return cur.rowcount == 1

    def get(self, key_id: str) -> KeyRecord | None:
        row = self._conn.execute("SELECT * FROM keys WHERE id = ?", (key_id,)).fetchone()
        return _key(row) if row else None


def _key(row: sqlite3.Row) -> KeyRecord:
    created = parse_ts(row["created_at"])
    assert created is not None
    return KeyRecord(
        id=row["id"],
        key_hash=row["key_hash"],
        prefix=row["prefix"],
        owner=row["owner"],
        team=row["team"],
        app=row["app"],
        env=Env(row["env"]),
        allowed_models=tuple(json.loads(row["allowed_models"])),
        budget_micros=row["budget_micros"],
        period=Cadence(row["period"]),
        tpm=row["tpm"],
        rpm=row["rpm"],
        created_at=created,
        revoked_at=parse_ts(row["revoked_at"]),
    )


# --- usage -------------------------------------------------------------------------------

PARENT_COLUMNS = (
    "request_id",
    "request_state",
    "ts_start",
    "ts_end",
    "key_id",
    "key_prefix",
    "owner",
    "team",
    "app",
    "env",
    "session_id",
    "route",
    "model",
    "upstream_model",
    "deployment",
    "provider",
    "region",
    "http_status",
    "upstream_status",
    "stream",
    "stream_complete",
    "usage_complete",
    "termination_reason",
    "input_tokens",
    "output_tokens",
    "cache_read_tokens",
    "cache_write_tokens",
    "cache_write_5m_tokens",
    "cache_write_1h_tokens",
    "observed_cost_micros",
    "priced_cost_micros",
    "cost_micros",
    "estimate_micros",
    "usage_source",
    "pricing_status",
    "accounting_kind",
    "attempt_count",
    "ttfb_ms",
    "latency_ms",
    "gate_overhead_ms",
    "fallback_count",
    "client_disconnected",
    "policy_decisions",
    "would_deny",
    "detections",
)

ATTEMPT_COLUMNS = (
    "request_id",
    "attempt_no",
    "ts_admitted",
    "ts_start",
    "ts_end",
    "key_period_id",
    "team_period_id",
    "deployment",
    "provider",
    "region",
    "model",
    "upstream_model",
    "upstream_status",
    "outcome",
    "billing_basis",
    "accounting_kind",
    "pricing_mode",
    "stream",
    "stream_complete",
    "usage_complete",
    "termination_reason",
    "input_tokens",
    "output_tokens",
    "cache_read_tokens",
    "cache_write_tokens",
    "cache_write_5m_tokens",
    "cache_write_1h_tokens",
    "estimate_tokens",
    "observed_total_tokens",
    "token_debit",
    "token_debit_source",
    "observed_cost_micros",
    "priced_cost_micros",
    "estimate_micros",
    "cost_micros",
    "usage_source",
    "pricing_status",
    "client_disconnected",
    "policy_decisions",
)

_UPSERT_PARENT = (
    f"INSERT INTO usage ({', '.join(PARENT_COLUMNS)})"
    f" VALUES ({', '.join('?' * len(PARENT_COLUMNS))})"
    " ON CONFLICT (request_id) DO UPDATE SET "
    + ", ".join(f"{c} = excluded.{c}" for c in PARENT_COLUMNS[1:])
)
_INSERT_ATTEMPT = (
    f"INSERT OR IGNORE INTO attempts ({', '.join(ATTEMPT_COLUMNS)})"
    f" VALUES ({', '.join('?' * len(ATTEMPT_COLUMNS))})"
)


def _parent_values(row: ParentRow) -> tuple[Any, ...]:
    t = row.totals
    counts = t.counts if t else TokenCounts()
    attempt_count = t.attempt_count if t else 0
    values: Mapping[str, Any] = {
        "request_id": row.request_id,
        "request_state": row.request_state.value,
        "ts_start": ts(row.ts_start),
        "ts_end": ts(row.ts_end),
        "key_id": row.key_id,
        "key_prefix": row.key_prefix,
        "owner": row.owner,
        "team": row.team,
        "app": row.app,
        "env": row.env,
        "session_id": row.session_id,
        "route": row.route,
        "model": row.model,
        "upstream_model": row.upstream_model,
        "deployment": row.deployment,
        "provider": row.provider,
        "region": row.region,
        "http_status": row.http_status,
        "upstream_status": row.upstream_status,
        "stream": _bool(row.stream),
        "stream_complete": _bool(row.stream_complete),
        "usage_complete": _bool(t.usage_complete) if t else 0,
        "termination_reason": row.termination_reason,
        "input_tokens": counts.input,
        "output_tokens": counts.output,
        "cache_read_tokens": counts.cache_read,
        "cache_write_tokens": counts.cache_write,
        "cache_write_5m_tokens": counts.cache_write_5m,
        "cache_write_1h_tokens": counts.cache_write_1h,
        "observed_cost_micros": t.observed_cost_micros if t else None,
        "priced_cost_micros": t.priced_cost_micros if t else None,
        "cost_micros": t.cost_micros if t else 0,
        "estimate_micros": t.estimate_micros if t else 0,
        "usage_source": t.usage_source.value if t else None,
        "pricing_status": t.pricing_status.value if t else None,
        "accounting_kind": t.accounting_kind.value if t else None,
        "attempt_count": attempt_count,
        "ttfb_ms": row.ttfb_ms,
        "latency_ms": row.latency_ms,
        "gate_overhead_ms": row.gate_overhead_ms,
        "fallback_count": max(0, attempt_count - 1),
        "client_disconnected": int(row.client_disconnected),
        "policy_decisions": _json(list(row.policy_decisions)),
        "would_deny": _json(list(row.would_deny)),
        "detections": _json(row.detections) if row.detections is not None else None,
    }
    return tuple(values[c] for c in PARENT_COLUMNS)


def _attempt_values(a: AttemptRecord) -> tuple[Any, ...]:
    values: Mapping[str, Any] = {
        "request_id": a.request_id,
        "attempt_no": a.attempt_no,
        "ts_admitted": ts(a.ts_admitted),
        "ts_start": ts(a.ts_start),
        "ts_end": ts(a.ts_end),
        "key_period_id": a.key_period_id,
        "team_period_id": a.team_period_id,
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
        "stream": int(a.stream),
        "stream_complete": _bool(a.stream_complete),
        "usage_complete": int(a.usage_complete),
        "termination_reason": a.termination_reason.value,
        "input_tokens": a.counts.input,
        "output_tokens": a.counts.output,
        "cache_read_tokens": a.counts.cache_read,
        "cache_write_tokens": a.counts.cache_write,
        "cache_write_5m_tokens": a.counts.cache_write_5m,
        "cache_write_1h_tokens": a.counts.cache_write_1h,
        "estimate_tokens": a.estimate_tokens,
        "observed_total_tokens": a.observed_total_tokens,
        "token_debit": a.token_debit,
        "token_debit_source": a.token_debit_source.value,
        "observed_cost_micros": a.observed_cost_micros,
        "priced_cost_micros": a.priced_cost_micros,
        "estimate_micros": a.estimate_micros,
        "cost_micros": a.cost_micros,
        "usage_source": a.usage_source.value,
        "pricing_status": a.pricing_status.value,
        "client_disconnected": int(a.client_disconnected),
        "policy_decisions": _json(list(a.policy_decisions)),
    }
    return tuple(values[c] for c in ATTEMPT_COLUMNS)


def _attempt(row: sqlite3.Row) -> AttemptRecord:
    def dt(name: str) -> datetime:
        value = parse_ts(row[name])
        assert value is not None
        return value

    def micros(name: str) -> Micros | None:
        return None if row[name] is None else Micros(row[name])

    return AttemptRecord(
        request_id=row["request_id"],
        attempt_no=row["attempt_no"],
        ts_admitted=dt("ts_admitted"),
        ts_start=dt("ts_start"),
        ts_end=dt("ts_end"),
        key_period_id=row["key_period_id"],
        team_period_id=row["team_period_id"],
        deployment=row["deployment"],
        provider=row["provider"],
        region=row["region"],
        model=row["model"],
        upstream_model=row["upstream_model"],
        upstream_status=row["upstream_status"],
        outcome=Outcome(row["outcome"]),
        billing_basis=row["billing_basis"],
        accounting_kind=AccountingKind(row["accounting_kind"]),
        pricing_mode=PricingMode(row["pricing_mode"]) if row["pricing_mode"] else None,
        stream=bool(row["stream"]),
        stream_complete=_opt_bool(row["stream_complete"]),
        usage_complete=bool(row["usage_complete"]),
        termination_reason=TerminationReason(row["termination_reason"]),
        counts=TokenCounts(
            input=row["input_tokens"],
            output=row["output_tokens"],
            cache_read=row["cache_read_tokens"],
            cache_write=row["cache_write_tokens"],
            cache_write_5m=row["cache_write_5m_tokens"],
            cache_write_1h=row["cache_write_1h_tokens"],
        ),
        estimate_tokens=row["estimate_tokens"],
        observed_total_tokens=row["observed_total_tokens"],
        token_debit=row["token_debit"],
        token_debit_source=TokenDebitSource(row["token_debit_source"]),
        observed_cost_micros=micros("observed_cost_micros"),
        priced_cost_micros=micros("priced_cost_micros"),
        estimate_micros=Micros(row["estimate_micros"]),
        cost_micros=Micros(row["cost_micros"]),
        usage_source=UsageSource(row["usage_source"]),
        pricing_status=PricingStatus(row["pricing_status"]),
        client_disconnected=bool(row["client_disconnected"]),
        policy_decisions=tuple(json.loads(row["policy_decisions"])),
    )


def _parent(row: sqlite3.Row) -> ParentRow:
    start = parse_ts(row["ts_start"])
    assert start is not None
    totals = None
    if row["accounting_kind"] is not None:
        totals = ParentTotals(
            attempt_count=row["attempt_count"],
            counts=TokenCounts(
                input=row["input_tokens"],
                output=row["output_tokens"],
                cache_read=row["cache_read_tokens"],
                cache_write=row["cache_write_tokens"],
                cache_write_5m=row["cache_write_5m_tokens"],
                cache_write_1h=row["cache_write_1h_tokens"],
            ),
            observed_cost_micros=row["observed_cost_micros"],
            priced_cost_micros=row["priced_cost_micros"],
            cost_micros=Micros(row["cost_micros"]),
            estimate_micros=Micros(row["estimate_micros"]),
            usage_source=UsageSource(row["usage_source"]),
            usage_complete=bool(row["usage_complete"]),
            pricing_status=PricingStatus(row["pricing_status"]),
            accounting_kind=AccountingKind(row["accounting_kind"]),
        )
    return ParentRow(
        request_id=row["request_id"],
        request_state=RequestState(row["request_state"]),
        ts_start=start,
        ts_end=parse_ts(row["ts_end"]),
        key_id=row["key_id"],
        key_prefix=row["key_prefix"],
        owner=row["owner"],
        team=row["team"],
        app=row["app"],
        env=row["env"],
        session_id=row["session_id"],
        route=row["route"],
        model=row["model"],
        stream=_opt_bool(row["stream"]),
        totals=totals,
        upstream_model=row["upstream_model"],
        deployment=row["deployment"],
        provider=row["provider"],
        region=row["region"],
        http_status=row["http_status"],
        upstream_status=row["upstream_status"],
        stream_complete=_opt_bool(row["stream_complete"]),
        termination_reason=row["termination_reason"],
        ttfb_ms=row["ttfb_ms"],
        latency_ms=row["latency_ms"],
        gate_overhead_ms=row["gate_overhead_ms"],
        client_disconnected=bool(row["client_disconnected"]),
        policy_decisions=tuple(json.loads(row["policy_decisions"])),
        would_deny=tuple(json.loads(row["would_deny"])),
        detections=json.loads(row["detections"]) if row["detections"] else None,
    )


class SqliteUsageRepository:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def open_request(self, row: ParentRow) -> None:
        self._conn.execute(_UPSERT_PARENT, _parent_values(row))

    def settle_attempt(self, attempt: AttemptRecord, parent: ParentRow) -> bool:
        conn = self._conn
        conn.execute("BEGIN IMMEDIATE")
        try:
            cur = conn.execute(_INSERT_ATTEMPT, _attempt_values(attempt))
            inserted = cur.rowcount == 1
            if inserted:
                conn.execute(_UPSERT_PARENT, _parent_values(parent))
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        return inserted

    def finish_request(self, row: ParentRow) -> None:
        self._conn.execute(_UPSERT_PARENT, _parent_values(row))

    def mark_interrupted(self) -> int:
        cur = self._conn.execute(
            "UPDATE usage SET request_state = 'interrupted', usage_complete = 0,"
            " priced_cost_micros = NULL WHERE request_state = 'open'"
        )
        return cur.rowcount

    def spend_by_bucket(self) -> Iterable[tuple[BucketKey, int]]:
        rows = self._conn.execute(
            "SELECT 'key' AS scope, u.key_id AS ident, a.key_period_id AS period,"
            "       SUM(a.cost_micros) AS spent"
            " FROM attempts a JOIN usage u ON u.request_id = a.request_id"
            " WHERE a.key_period_id IS NOT NULL GROUP BY u.key_id, a.key_period_id"
            " UNION ALL "
            "SELECT 'team', u.team, a.team_period_id, SUM(a.cost_micros)"
            " FROM attempts a JOIN usage u ON u.request_id = a.request_id"
            " WHERE a.team_period_id IS NOT NULL GROUP BY u.team, a.team_period_id"
        )
        for row in rows:
            if row["ident"] is not None:
                yield BucketKey(ScopeType(row["scope"]), row["ident"], row["period"]), row["spent"]

    def attempts(
        self, *, since: datetime | None, until: datetime | None
    ) -> Sequence[AttemptWithParent]:
        sql, params = self._attempt_query()
        if since is not None:
            sql += " AND a.ts_admitted >= ?"
            params.append(ts(since))
        if until is not None:
            sql += " AND a.ts_admitted < ?"
            params.append(ts(until))
        return self._attempts(sql + " ORDER BY a.ts_admitted", params)

    def attempts_for_period(self, period_id: str) -> Sequence[AttemptWithParent]:
        sql, params = self._attempt_query()
        sql += " AND (a.key_period_id = ? OR a.team_period_id = ?)"
        params += [period_id, period_id]
        return self._attempts(sql, params)

    def requests(self, *, since: datetime | None, until: datetime | None) -> Sequence[ParentRow]:
        sql = "SELECT * FROM usage WHERE 1=1"
        params: list[Any] = []
        if since is not None:
            sql += " AND ts_start >= ?"
            params.append(ts(since))
        if until is not None:
            sql += " AND ts_start < ?"
            params.append(ts(until))
        return [_parent(row) for row in self._conn.execute(sql + " ORDER BY ts_start", params)]

    def latest_requests(self, where: RequestFilter) -> Iterable[ParentRow]:
        sql = "SELECT * FROM usage WHERE 1=1"
        params: list[Any] = []
        for column, value in (
            ("team", where.team),
            ("app", where.app),
            ("key_prefix", where.key_prefix),
            ("model", where.model),
        ):
            if value is not None:
                sql += f" AND {column} = ?"
                params.append(value)
        if where.provider is not None:
            sql += (
                " AND (provider = ? OR request_id IN"
                " (SELECT request_id FROM attempts WHERE provider = ?))"
            )
            params += [where.provider, where.provider]
        cursor = self._conn.execute(sql + " ORDER BY ts_start DESC, request_id DESC", params)
        return (_parent(row) for row in cursor)

    def request(self, request_id: str) -> ParentRow | None:
        row = self._conn.execute(
            "SELECT * FROM usage WHERE request_id = ?", (request_id,)
        ).fetchone()
        return _parent(row) if row else None

    def attempts_for_requests(self, request_ids: Sequence[str]) -> Sequence[AttemptWithParent]:
        if not request_ids:
            return []
        sql, params = self._attempt_query()
        sql += f" AND a.request_id IN ({','.join('?' * len(request_ids))})"
        params += list(request_ids)
        return self._attempts(sql + " ORDER BY a.request_id, a.attempt_no", params)

    def writable(self) -> bool:
        try:
            self._conn.execute("BEGIN IMMEDIATE")
            self._conn.execute("ROLLBACK")
        except sqlite3.Error:
            return False
        return True

    def _attempt_query(self) -> tuple[str, list[Any]]:
        return (
            "SELECT a.*, u.key_id AS p_key_id, u.owner AS p_owner, u.team AS p_team,"
            " u.app AS p_app, u.request_state AS p_state"
            " FROM attempts a JOIN usage u ON u.request_id = a.request_id WHERE 1=1",
            [],
        )

    def _attempts(self, sql: str, params: list[Any]) -> list[AttemptWithParent]:
        return [
            AttemptWithParent(
                attempt=_attempt(row),
                key_id=row["p_key_id"],
                owner=row["p_owner"],
                team=row["p_team"],
                app=row["p_app"],
                request_finished=row["p_state"] == RequestState.FINISHED.value,
            )
            for row in self._conn.execute(sql, params)
        ]

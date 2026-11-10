"""argparse commands -> application use cases.

The CLI is the whole control plane: there is no admin HTTP API. It writes to
the same SQLite file as the server, which picks up key changes on the next
request. ``ui`` starts the read-only report UI, which never writes.
"""

import argparse
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from typing import Any

from gate.adapters.config_yaml import ConfigReport, config_path, load_config
from gate.adapters.formats import parse_when, render
from gate.adapters.sqlite_store import SqliteKeyRepository, SqliteUsageRepository, connect, migrate
from gate.app.keys import KeyInputError, KeyService, NewKey, key_row
from gate.app.ports import RequestFilter
from gate.app.reports import GROUPS, REQUEST_STATUSES, Reports, budget_limit_for
from gate.domain.config import GatewayConfig
from gate.domain.periods import is_period_id

Serve = Callable[[GatewayConfig], int]


class SystemClock:
    def now(self) -> datetime:
        return datetime.now(UTC)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="api-gateway", description="A lightweight API gate for LLM traffic."
    )
    parser.add_argument(
        "--config", help="path to config YAML (default: $API_GATEWAY_CONFIG or ./config.yaml)"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    keys = sub.add_parser("keys", help="manage virtual keys").add_subparsers(
        dest="keys_command", required=True
    )
    create = keys.add_parser("create", help="issue a key (printed once)")
    create.add_argument("--owner", required=True)
    create.add_argument("--team", required=True)
    create.add_argument("--app", required=True)
    create.add_argument("--env", required=True, choices=["dev", "staging", "prod"])
    create.add_argument(
        "--models", required=True, help="comma-separated exact model names or aliases"
    )
    create.add_argument("--budget", help="key budget in USD per period")
    create.add_argument("--period", default="month", choices=["day", "month"])
    create.add_argument("--tpm", type=int)
    create.add_argument("--rpm", type=int)
    listing = keys.add_parser("list", help="list keys")
    listing.add_argument("--team")
    listing.add_argument("--include-revoked", action="store_true")
    listing.add_argument("--format", default="table", choices=["table", "csv", "json"])
    revoke = keys.add_parser("revoke", help="revoke a key by id or prefix")
    revoke.add_argument("ident")

    report = sub.add_parser("report", help="chargeback, budget, coverage and request reports")
    report.add_argument("kind", nargs="?", choices=["budgets", "coverage", "requests", "request"])
    report.add_argument("request_id", nargs="?", help="request: the request ID to show")
    report.add_argument("--by", help=f"group: {', '.join(GROUPS)} (budgets: key or team)")
    dates = (
        " UTC date or datetime; attempt metrics filter on attempt admission,"
        " request counts, latency and denials on request start"
    )
    report.add_argument("--since", help="inclusive" + dates)
    report.add_argument("--until", help="exclusive" + dates)
    report.add_argument("--period", help="budgets: a period ID such as month:2026-11")
    report.add_argument("--limit", type=int, default=100, help="requests: rows (1-1000)")
    report.add_argument("--team", help="requests: exact team")
    report.add_argument("--app", help="requests: exact app")
    report.add_argument("--key", help="requests: exact key prefix")
    report.add_argument("--model", help="requests: exact requested model")
    report.add_argument("--provider", help="requests: provider of the request or any attempt")
    report.add_argument("--status", choices=REQUEST_STATUSES, help="requests: how it ended")
    report.add_argument("--format", default="table", choices=["table", "csv", "json"])

    check = sub.add_parser("check-config", help="validate the config file")
    check.add_argument("--experimental", action="store_true")
    serve = sub.add_parser("serve", help="run the gateway")
    serve.add_argument(
        "--experimental",
        action="store_true",
        help="allow budget.mode: naive (budget-overshoot experiment only)",
    )
    sub.add_parser("ui", help="run the read-only report UI")
    return parser


def run(
    argv: Sequence[str] | None,
    *,
    serve: Serve,
    ui: Serve | None = None,
    env: Mapping[str, str] | None = None,
) -> int:
    env = os.environ if env is None else env
    args = build_parser().parse_args(argv)
    path = config_path(args.config, env)

    if args.command == "check-config":
        report = load_config(path, env, experimental=args.experimental)
        _print_config_report(report)
        return 0 if report.ok else 1

    if args.command == "serve":
        report = load_config(path, env, experimental=args.experimental)
        if not report.ok:
            _print_config_report(report, stream=sys.stderr)
            return 1
        assert report.config is not None
        return serve(report.config)

    # keys, report and ui only need storage; a missing provider key must not block them.
    report = load_config(path, env, check_secrets=False, experimental=True)
    if report.config is None:
        _print_config_report(report, stream=sys.stderr)
        return 1
    config = report.config
    if args.command == "ui":
        if ui is None:
            print("error: the report UI is not available in this build", file=sys.stderr)
            return 1
        return ui(config)
    conn = connect(config.sqlite_path)
    migrate(conn)
    try:
        if args.command == "keys":
            return _keys(args, SqliteKeyRepository(conn))
        return _report(args, config, SqliteUsageRepository(conn), SqliteKeyRepository(conn))
    finally:
        conn.close()


def _print_config_report(report: ConfigReport, stream: Any = None) -> None:
    out = stream or sys.stdout
    for error in report.errors:
        print(f"error: {error}", file=out)
    for note in report.notes:
        print(f"note: {note}", file=out)
    if report.ok:
        print("config ok", file=out)


def _keys(args: argparse.Namespace, repo: SqliteKeyRepository) -> int:
    service = KeyService(repo, SystemClock())
    try:
        if args.keys_command == "create":
            budget = None
            if args.budget is not None:
                try:
                    budget = Decimal(args.budget)
                except InvalidOperation:
                    raise KeyInputError("--budget must be a number") from None
            record, raw = service.create(
                NewKey(
                    owner=args.owner,
                    team=args.team,
                    app=args.app,
                    env=args.env,
                    models=args.models.split(","),
                    budget_usd=budget,
                    period=args.period,
                    tpm=args.tpm,
                    rpm=args.rpm,
                )
            )
            print(f"key_id: {record.id}")
            print(f"prefix: {record.prefix}")
            print(f"key:    {raw}")
            print("Store this key now; it cannot be shown again.", file=sys.stderr)
            return 0
        if args.keys_command == "list":
            rows = [
                key_row(k)
                for k in service.list(team=args.team, include_revoked=args.include_revoked)
            ]
            _emit(rows, args.format)
            return 0
        key = service.revoke(args.ident)
        print(f"revoked {key.id} ({key.prefix})")
        return 0
    except KeyInputError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def _report(
    args: argparse.Namespace,
    config: GatewayConfig,
    usage: SqliteUsageRepository,
    keys: SqliteKeyRepository,
) -> int:
    reports = Reports(usage)
    try:
        since, until = parse_when(args.since), parse_when(args.until)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.kind == "budgets":
        if args.by not in ("key", "team") or not args.period or not is_period_id(args.period):
            print(
                "error: report budgets needs --by key|team and"
                " --period day:YYYY-MM-DD|month:YYYY-MM",
                file=sys.stderr,
            )
            return 2

        limit_for = budget_limit_for(args.by, config.teams, keys)
        _emit(reports.budgets(args.by, args.period, limit_for), args.format)
        return 0

    if args.kind == "coverage":
        _emit([reports.coverage(since=since, until=until)], args.format)
        return 0

    if args.kind == "requests":
        if not 1 <= args.limit <= 1000:
            print("error: --limit must be between 1 and 1000", file=sys.stderr)
            return 2
        where = RequestFilter(
            team=args.team,
            app=args.app,
            key_prefix=args.key,
            model=args.model,
            provider=args.provider,
        )
        _emit(reports.recent_requests(where, limit=args.limit, status=args.status), args.format)
        return 0

    if args.kind == "request":
        if not args.request_id:
            print("error: report request needs a request ID", file=sys.stderr)
            return 2
        detail = reports.request_detail(args.request_id)
        if detail is None:
            print(f"error: no request {args.request_id}", file=sys.stderr)
            return 1
        if args.format == "json":
            _emit([detail], "json")
        else:
            attempts = detail.pop("attempt_list")
            _emit([detail], args.format)
            print()
            _emit(attempts, args.format)
        return 0

    if args.by not in GROUPS:
        print(f"error: report needs --by one of {', '.join(GROUPS)}", file=sys.stderr)
        return 2
    _emit(reports.by(args.by, since=since, until=until), args.format)
    return 0


def _emit(rows: list[dict[str, Any]], fmt: str) -> None:
    print(render(rows, fmt), end="")

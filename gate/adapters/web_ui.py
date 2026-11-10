"""Read-only report UI: report view models -> HTML pages.

A view, not a second implementation. Every number comes from ``app/reports.py`` and
``app/keys.py``, the same use cases the CLI renders, and every table page offers the
CLI's ``--format json|csv`` output as a download.

Boundaries this module keeps:

* Read-only. Only GET routes exist, and each request opens its own read-only SQLite
  connection (``connect_readonly``), so a write would fail in SQLite itself.
* Separate process. ``api-gateway ui`` runs this app on its own listener; it shares
  nothing in memory with the gate, so live reservations are not visible here.
* No secrets. Config is loaded without provider keys; keys are shown by prefix only.
* Escaped by default. HTML is built with ``el``, which escapes every value that is
  not already ``Html``. Owner, team and app names are user input and are never
  trusted. Pages are self-contained: one local stylesheet, inline SVG, no scripts.
"""

import html
import sqlite3
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlencode

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from starlette.middleware.base import RequestResponseEndpoint

from gate.adapters.formats import parse_when, to_csv, to_json
from gate.adapters.sqlite_store import (
    SqliteKeyRepository,
    SqliteUsageRepository,
    connect_readonly,
)
from gate.app.keys import key_row
from gate.app.ports import RequestFilter
from gate.app.reports import GROUPS, REQUEST_STATUSES, Reports, budget_limit_for
from gate.domain.config import GatewayConfig
from gate.domain.models import Cadence
from gate.domain.periods import is_period_id, period_id

Row = Mapping[str, Any]
Clock = Callable[[], datetime]

STYLESHEET = Path(__file__).with_name("web_ui.css")

SECURITY_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'none'; style-src 'self'; img-src 'self'; form-action 'self';"
        " base-uri 'none'; frame-ancestors 'none'"
    ),
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "Cache-Control": "no-store",
}

RANGES = ("hour", "today", "month", "all")
DEFAULT_RANGE = "month"
NAV = (
    ("/", "Overview"),
    ("/requests", "Requests"),
    ("/reports", "Chargeback"),
    ("/budgets", "Budgets"),
    ("/coverage", "Coverage"),
    ("/keys", "Keys"),
)
# Token counts where None means "not reported", which must never read as zero.
TOKEN_COLUMNS = frozenset(
    {
        "input_tokens",
        "output_tokens",
        "cache_read_tokens",
        "cache_write_5m_tokens",
        "cache_write_1h_tokens",
    }
)
# Query parameters that carry over between pages.
STICKY = ("range", "since", "until", "refresh")


class BadInput(ValueError):
    """A query parameter the page cannot use; the message is safe to show."""


# --- escaping-by-default HTML --------------------------------------------------------------


class Html(str):
    """Markup that is already safe to emit. Every other value goes through ``esc``."""


def esc(value: object) -> Html:
    if isinstance(value, Html):
        return value
    return Html(html.escape("" if value is None else str(value), quote=True))


VOID = frozenset({"input", "meta", "link", "br"})


def el(tag: str, *children: object, **attrs: object) -> Html:
    """``<tag attrs>children</tag>``. ``class_`` becomes ``class``; False/None drop out."""
    parts = [tag]
    for name, value in attrs.items():
        if value is None or value is False:
            continue
        attr = name.rstrip("_").replace("_", "-")
        parts.append(attr if value is True else f'{attr}="{esc(value)}"')
    if tag in VOID:
        return Html(f"<{' '.join(parts)}>")
    return Html(f"<{' '.join(parts)}>{join(children)}</{tag}>")


def join(items: Iterable[object]) -> Html:
    return Html("".join(esc(item) for item in items))


# --- app -----------------------------------------------------------------------------------


def create_ui_app(config: GatewayConfig, *, now: Clock | None = None) -> FastAPI:
    clock: Clock = now or (lambda: datetime.now(UTC))
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    @contextmanager
    def store() -> Iterator[tuple[Reports, SqliteKeyRepository]]:
        conn = connect_readonly(config.sqlite_path)
        try:
            yield Reports(SqliteUsageRepository(conn)), SqliteKeyRepository(conn)
        finally:
            conn.close()

    @app.middleware("http")
    async def headers(request: Request, call_next: RequestResponseEndpoint) -> Response:
        response = await call_next(request)
        response.headers.update(SECURITY_HEADERS)
        return response

    @app.exception_handler(sqlite3.Error)
    async def database_unavailable(request: Request, exc: sqlite3.Error) -> Response:
        body = el(
            "p",
            "The gateway database cannot be opened yet. Start the gateway first;"
            " it creates the database on startup.",
        )
        return _page(request, "Database unavailable", body, status=503)

    @app.exception_handler(BadInput)
    async def bad_input(request: Request, exc: BadInput) -> Response:
        return _page(request, "Bad request", el("p", str(exc)), status=400)

    @app.get("/healthz")
    def healthz() -> Response:
        try:
            with store() as _:
                pass
        except sqlite3.Error:
            return JSONResponse({"status": "database unavailable"}, status_code=503)
        return JSONResponse({"status": "ok"})

    @app.get("/static/ui.css")
    def stylesheet() -> Response:
        return Response(STYLESHEET.read_text(encoding="utf-8"), media_type="text/css")

    @app.get("/")
    def overview(request: Request) -> Response:
        since, until, label = _window(request, clock())
        with store() as (reports, _):
            total = reports.by("all", since=since, until=until)
            by_group = {
                g: reports.by(g, since=since, until=until) for g in ("provider", "team", "model")
            }
            budgets = reports.team_budgets_now(config.teams, clock())
            recent = reports.recent_requests(RequestFilter(), limit=10)
        body = join(
            [
                _window_picker(request, label),
                _tiles(total[0] if total else None),
                el(
                    "section",
                    el("h2", "Team budgets, current period"),
                    _budget_meters(budgets, "team"),
                    el(
                        "p",
                        "Budgets are estimated admission controls, not billing ceilings."
                        " Reservations held by in-flight requests live in the gateway's"
                        " memory and are not shown here.",
                        class_="note",
                    ),
                ),
                el(
                    "div",
                    *(
                        el(
                            "section",
                            el("h2", f"Total debits by {g}"),
                            _debit_bars(rows, g),
                        )
                        for g, rows in by_group.items()
                    ),
                    class_="grid",
                ),
                el(
                    "p",
                    "Request counts per group are not additive: a request that fell back"
                    " across providers or models counts in each group it ran in.",
                    class_="note",
                ),
                el(
                    "section",
                    el("h2", "Latest requests"),
                    _requests_table(recent, request),
                    el("p", el("a", "All requests →", href=_href(request, "/requests"))),
                ),
            ]
        )
        return _page(request, "Overview", body)

    @app.get("/requests")
    def requests(request: Request) -> Response:
        status = _param(request, "status")
        if status is not None and status not in REQUEST_STATUSES:
            raise BadInput(f"status must be one of {', '.join(REQUEST_STATUSES)}")
        where = RequestFilter(
            team=_param(request, "team"),
            app=_param(request, "app"),
            key_prefix=_param(request, "key"),
            model=_param(request, "model"),
            provider=_param(request, "provider"),
        )
        with store() as (reports, _):
            rows = reports.recent_requests(where, limit=config.ui.recent_requests, status=status)
        download = _download(request, rows)
        if download is not None:
            return download
        body = join(
            [
                _request_filters(request),
                el("p", f"Newest {config.ui.recent_requests} matching requests.", class_="note"),
                _downloads(request),
                _requests_table(rows, request),
            ]
        )
        return _page(request, "Requests", body)

    @app.get("/requests/{request_id}")
    def request_detail(request: Request, request_id: str) -> Response:
        with store() as (reports, _):
            detail = reports.request_detail(request_id)
        if detail is None:
            return _page(request, "Not found", el("p", "No such request."), status=404)
        if _param(request, "format") == "json":
            return Response(to_json([detail]), media_type="application/json")
        fields = {k: v for k, v in detail.items() if k not in ("attempt_list", "policy_decisions")}
        attempts = detail["attempt_list"]
        attempt_blocks = [
            join([el("h3", f"Attempt {a['attempt_no']}"), _fields(a)]) for a in attempts
        ] or [el("p", "No upstream attempt was made.", class_="note")]
        body = join(
            [
                _downloads(request, formats=("json",)),
                el("section", el("h2", "Request"), _fields(fields)),
                el(
                    "section",
                    el("h2", "Policy decisions"),
                    _table(detail["policy_decisions"], empty="None recorded."),
                ),
                el("section", el("h2", f"Attempts ({len(attempts)})"), *attempt_blocks),
            ]
        )
        return _page(request, f"Request {request_id}", body)

    @app.get("/reports")
    def chargeback(request: Request) -> Response:
        group = _param(request, "by") or "team"
        if group not in GROUPS:
            raise BadInput(f"by must be one of {', '.join(GROUPS)}")
        since, until, label = _window(request, clock())
        with store() as (reports, _):
            rows = reports.by(group, since=since, until=until)
        download = _download(request, rows)
        if download is not None:
            return download
        body = join(
            [
                _choices(request, "by", GROUPS, group),
                _window_picker(request, label),
                _downloads(request),
                _transposed(rows, group),
                el(
                    "p",
                    "Calculated costs, estimated debits and partial observations are kept"
                    " apart; total debits are calculated plus estimated. Grouped request"
                    " counts are not additive.",
                    class_="note",
                ),
            ]
        )
        return _page(request, f"Chargeback by {group}", body)

    @app.get("/budgets")
    def budgets(request: Request) -> Response:
        scope = _param(request, "by") or "team"
        if scope not in ("key", "team"):
            raise BadInput("by must be key or team")
        period = _param(request, "period") or period_id(Cadence.MONTH, clock())
        if not is_period_id(period):
            raise BadInput("period must look like month:2026-11 or day:2026-11-30")
        with store() as (reports, keys):
            rows = reports.budgets(scope, period, budget_limit_for(scope, config.teams, keys))
        download = _download(request, rows)
        if download is not None:
            return download
        body = join(
            [
                _choices(request, "by", ("team", "key"), scope),
                el(
                    "form",
                    el("label", "Period ", el("input", name="period", value=period)),
                    _hidden(request, keep=("by", "refresh")),
                    el("button", "Show", type="submit"),
                    method="get",
                    class_="filters",
                ),
                _downloads(request),
                _budget_meters(rows, scope),
                _table(rows),
                el(
                    "p",
                    "Debits recorded in this stored period ID, whenever they settled."
                    " Budgets are estimated admission controls, not billing ceilings.",
                    class_="note",
                ),
            ]
        )
        return _page(request, f"Budgets by {scope}, {period}", body)

    @app.get("/coverage")
    def coverage(request: Request) -> Response:
        since, until, label = _window(request, clock())
        with store() as (reports, _):
            row = reports.coverage(since=since, until=until)
        download = _download(request, [row])
        if download is not None:
            return download
        body = join([_window_picker(request, label), _downloads(request), _fields(row)])
        return _page(request, "Coverage", body)

    @app.get("/keys")
    def keys_page(request: Request) -> Response:
        include_revoked = _param(request, "include_revoked") == "1"
        with store() as (_, keys):
            rows = [key_row(k) for k in keys.list(team=None, include_revoked=include_revoked)]
        download = _download(request, rows)
        if download is not None:
            return download
        toggle = (
            el("a", "Hide revoked", href=_href(request, "/keys"))
            if include_revoked
            else el(
                "a",
                "Show revoked",
                href=_href(request, "/keys", include_revoked="1"),  # gitleaks:allow
            )
        )
        body = join(
            [
                el("p", toggle, " · Keys are issued and revoked with the CLI only.", class_="note"),
                _downloads(request),
                _table(rows),
            ]
        )
        return _page(request, "Keys", body)

    return app


# --- query parameters ----------------------------------------------------------------------


def _param(request: Request, name: str) -> str | None:
    """A query parameter, with empty values treated as absent."""
    value = (request.query_params.get(name) or "").strip()
    return value or None


def _refresh(request: Request) -> int | None:
    """Seconds between reloads, or None; values outside 2-60 are ignored."""
    try:
        seconds = int(_param(request, "refresh") or "")
    except ValueError:
        return None
    return seconds if 2 <= seconds <= 60 else None


def _window(request: Request, now: datetime) -> tuple[datetime | None, datetime | None, str]:
    """The report window: a quick pick or explicit since/until (CLI semantics, UTC)."""
    try:
        since, until = parse_when(_param(request, "since")), parse_when(_param(request, "until"))
    except ValueError as exc:
        raise BadInput(str(exc)) from None
    if since is not None or until is not None:
        return since, until, "custom"
    pick = _param(request, "range") or DEFAULT_RANGE
    if pick not in RANGES:
        raise BadInput(f"range must be one of {', '.join(RANGES)}")
    utc = now.astimezone(UTC)
    starts = {
        "hour": utc - timedelta(hours=1),
        "today": utc.replace(hour=0, minute=0, second=0, microsecond=0),
        "month": utc.replace(day=1, hour=0, minute=0, second=0, microsecond=0),
        "all": None,
    }
    return starts[pick], None, pick


def _href(request: Request, path: str, **params: str | None) -> str:
    """A link to ``path`` that keeps the window and refresh, plus ``params``."""
    query = {k: v for k in STICKY if (v := _param(request, k)) is not None}
    for name, value in params.items():
        if value is None:
            query.pop(name, None)
        else:
            query[name] = value
    return path + ("?" + urlencode(query) if query else "")


def _here(request: Request, **params: str | None) -> str:
    """The current page with every current parameter, changed by ``params``."""
    query = {k: v for k, v in request.query_params.items() if v}
    for name, value in params.items():
        if value is None:
            query.pop(name, None)
        else:
            query[name] = value
    return request.url.path + ("?" + urlencode(query) if query else "")


def _download(request: Request, rows: Sequence[Row]) -> Response | None:
    fmt = _param(request, "format")
    if fmt == "json":
        return Response(to_json([dict(r) for r in rows]), media_type="application/json")
    if fmt == "csv":
        return Response(to_csv([dict(r) for r in rows]), media_type="text/csv; charset=utf-8")
    if fmt is not None:
        raise BadInput("format must be json or csv")
    return None


# --- page chrome ---------------------------------------------------------------------------


def _page(request: Request, title: str, body: Html, *, status: int = 200) -> Response:
    refresh = _refresh(request)
    path = request.url.path
    nav = el(
        "nav",
        *(
            el(
                "a",
                label,
                href=_href(request, target),
                aria_current="page" if path == target else None,
            )
            for target, label in NAV
        ),
    )
    live = (
        el("a", f"Live: every {refresh}s · stop", href=_here(request, refresh=None))
        if refresh
        else el("a", "Live: off · start", href=_here(request, refresh="5"))
    )
    head = join(
        [
            Html('<meta charset="utf-8">'),
            Html('<meta name="viewport" content="width=device-width, initial-scale=1">'),
            Html(f'<meta http-equiv="refresh" content="{refresh}">') if refresh else "",
            el("title", f"{title} · VGTech demo api gateway"),
            Html('<link rel="stylesheet" href="/static/ui.css">'),
        ]
    )
    document = join(
        [
            Html("<!doctype html>"),
            el(
                "html",
                el("head", head),
                el(
                    "body",
                    el(
                        "header",
                        el("strong", "VGTech demo api gateway"),
                        nav,
                        el("span", live, class_="live"),
                    ),
                    el(
                        "main",
                        el("h1", title),
                        body,
                        el(
                            "footer",
                            "Read-only view of committed gateway data. Times are UTC."
                            " No prompts or responses are stored, so none are shown.",
                        ),
                    ),
                ),
                lang="en",
            ),
        ]
    )
    return Response(document, status_code=status, media_type="text/html; charset=utf-8")


def _window_picker(request: Request, label: str) -> Html:
    picks = [
        el(
            "a",
            name,
            href=_here(request, range=name, since=None, until=None),
            aria_current="true" if label == name else None,
        )
        for name in RANGES
    ]
    form = el(
        "form",
        el(
            "label",
            "Since ",
            el("input", name="since", value=_param(request, "since"), placeholder="2026-11-01"),
        ),
        el(
            "label",
            "Until ",
            el("input", name="until", value=_param(request, "until"), placeholder="exclusive"),
        ),
        _hidden(request, keep=("by", "refresh")),
        el("button", "Apply", type="submit"),
        method="get",
    )
    return el("div", el("span", "Window: "), *picks, form, class_="filters")


def _choices(request: Request, name: str, options: Sequence[str], current: str) -> Html:
    return el(
        "div",
        el("span", f"{name.replace('_', ' ').capitalize()}: "),
        *(
            el(
                "a",
                option,
                href=_here(request, **{name: option}),
                aria_current="true" if option == current else None,
            )
            for option in options
        ),
        class_="filters",
    )


def _hidden(request: Request, *, keep: Sequence[str]) -> Html:
    return join(
        el("input", type="hidden", name=name, value=value)
        for name in (*keep, "range")
        if (value := _param(request, name)) is not None and name not in ("since", "until")
    )


def _downloads(request: Request, formats: Sequence[str] = ("csv", "json")) -> Html:
    links = [el("a", f.upper(), href=_here(request, format=f, refresh=None)) for f in formats]
    return el("p", "Download: ", *_interleave(links, " · "), class_="downloads")


def _request_filters(request: Request) -> Html:
    def field(name: str, label: str) -> Html:
        return el("label", f"{label} ", el("input", name=name, value=_param(request, name)))

    current = _param(request, "status")
    options = [el("option", "any", value="")] + [
        el("option", s, value=s, selected=s == current) for s in REQUEST_STATUSES
    ]
    return el(
        "form",
        field("team", "Team"),
        field("app", "App"),
        field("key", "Key prefix"),
        field("model", "Model"),
        field("provider", "Provider"),
        el("label", "Status ", el("select", *options, name="status")),
        _hidden(request, keep=("refresh",)),
        el("button", "Filter", type="submit"),
        method="get",
        class_="filters",
    )


# --- content blocks ------------------------------------------------------------------------


def _tiles(total: Row | None) -> Html:
    if total is None:
        return el("p", "No traffic in this window yet.", class_="note")

    def tile(label: str, value: object, hint: str = "") -> Html:
        return el("div", el("span", label), el("strong", value), el("small", hint), class_="tile")

    blocked = sum(total["blocked_requests_by_reason"].values())
    would_deny = sum(total["would_deny_by_code"].values())
    uncertain = sum(total["pricing_uncertain_by_reason"].values())
    return el(
        "div",
        tile("Client requests", _int(total["requests"])),
        tile("Upstream attempts", _int(total["attempts"])),
        tile("Calculated cost", _usd(total["calculated_cost_usd"]), "complete usage, priced"),
        tile("Estimated debits", _usd(total["estimated_debit_usd"]), "reservation-based"),
        tile("Total debits", _usd(total["total_debit_usd"]), "what budgets were charged"),
        tile("Blocked", _int(blocked), "enforced denials"),
        tile("Would deny", _int(would_deny), "observe mode, admitted"),
        tile(
            "Partial / uncertain",
            f"{_int(total['partial_attempts'])} / {_int(uncertain)}",
            "attempts",
        ),
        tile(
            "Gate overhead",
            f"{_num(total['gate_overhead_p50_ms'])} / {_num(total['gate_overhead_p95_ms'])}",
            "p50 / p95, ms",
        ),
        class_="tiles",
    )


def _debit_bars(rows: Sequence[Row], group: str) -> Html:
    if not rows:
        return el("p", "No attempts.", class_="note")
    # Reports name the group "-" when it has no value, such as the provider of a request
    # denied before any attempt, or the team of a request without a valid key.
    values = [
        ("(none)" if r[group] == "-" else str(r[group]), Decimal(r["total_debit_usd"]), r)
        for r in rows
    ]
    values.sort(key=lambda v: v[1], reverse=True)
    top = max((v for _, v, _ in values), default=Decimal(0))
    lines = [
        el(
            "tr",
            el("th", name, scope="row"),
            el("td", _bar(value, top)),
            el(
                "td",
                _usd(row["total_debit_usd"]),
                el(
                    "small",
                    f"calc {_usd(row['calculated_cost_usd'])} · est"
                    f" {_usd(row['estimated_debit_usd'])} · {_int(row['requests'])} req",
                ),
                class_="num",
            ),
        )
        for name, value, row in values
    ]
    return el("table", el("tbody", *lines), class_="bars")


def _budget_meters(rows: Sequence[Row], scope: str) -> Html:
    rows = [r for r in rows if r.get("budget_usd") is not None]
    if not rows:
        return el("p", "No budgets configured for this view.", class_="note")
    lines = []
    for row in rows:
        spent, budget = Decimal(row["total_debit_usd"]), Decimal(row["budget_usd"])
        over = spent > budget
        pct = (spent / budget * 100) if budget > 0 else Decimal(100 if spent > 0 else 0)
        lines.append(
            el(
                "tr",
                el("th", row[scope], el("small", row["period_id"]), scope="row"),
                el("td", _bar(min(spent, budget), budget, over=over)),
                el(
                    "td",
                    f"{_usd(row['total_debit_usd'])} of {_usd(row['budget_usd'])}",
                    el(
                        "small",
                        f"{pct:.0f}% · calc {_usd(row['calculated_usd'])} · est"
                        f" {_usd(row['estimated_usd'])}"
                        + (f" · overrun {_usd(row['overrun_usd'])}" if over else ""),
                    ),
                    class_="num over" if over else "num",
                ),
            )
        )
    return el("table", el("tbody", *lines), class_="bars")


def _bar(value: Decimal, top: Decimal, *, over: bool = False) -> Html:
    width = float(value / top * 100) if top > 0 else 0.0
    return el(
        "svg",
        el("rect", width="100", height="10", class_="track"),
        el("rect", width=f"{width:.2f}", height="10", class_="fill over" if over else "fill"),
        viewBox="0 0 100 10",
        preserveAspectRatio="none",
        role="img",
        aria_label=f"{width:.0f}%",
        class_="bar",
    )


def _requests_table(rows: Sequence[Row], request: Request) -> Html:
    if not rows:
        return el("p", "No requests match.", class_="note")
    head = el(
        "tr",
        *(
            el("th", h)
            for h in (
                "Time (UTC)",
                "Status",
                "Key",
                "Owner / team / app",
                "Model",
                "Providers",
                "Tokens in / out",
                "Debit",
                "Overhead",
                "Flags",
            )
        ),
    )
    lines = []
    for r in rows:
        flags = [f"denied: {r['denied_code']}"] if r["denied_code"] else []
        if r["would_deny"]:
            flags.append(f"would deny: {r['would_deny']}")
        if r["detections"]:
            flags.append("detected: " + _show("detections", r["detections"], r))
        if r["attempts"] > 1:
            flags.append(f"{r['attempts']} attempts")
        tokens = (
            f"{_show('input_tokens', r['input_tokens'], r)} /"
            f" {_show('output_tokens', r['output_tokens'], r)}"
        )
        if r["usage"] == "partial":
            tokens += " (partial)"
        lines.append(
            el(
                "tr",
                el(
                    "td",
                    el(
                        "a",
                        _short_time(r["ts_start"]),
                        href=_href(request, f"/requests/{quote(r['request_id'], safe='')}"),
                    ),
                ),
                el("td", el("span", r["status"], class_=f"status {r['status']}")),
                el("td", _show("key_prefix", r["key_prefix"], r), class_="mono"),
                el("td", " / ".join(_show(k, r[k], r) for k in ("owner", "team", "app"))),
                el("td", _show("model", r["model"], r)),
                el("td", _show("providers", r["providers"], r)),
                el("td", tokens, class_="num"),
                el("td", _money_split(r), class_="num"),
                el("td", _ms(r["gate_overhead_ms"]), class_="num"),
                el("td", "; ".join(flags)),
            )
        )
    return el("div", el("table", el("thead", head), el("tbody", *lines)), class_="scroll")


def _money_split(r: Row) -> Html:
    if r["attempts"] == 0:
        return Html("—")  # nothing was dispatched, so nothing was debited
    calc, est = Decimal(r["calculated_cost_usd"]), Decimal(r["estimated_debit_usd"])
    if est == 0:
        return join([_usd(r["calculated_cost_usd"]), el("small", "calculated")])
    if calc == 0:
        return join([_usd(r["estimated_debit_usd"]), el("small", "estimated")])
    return join(
        [
            _usd(r["total_debit_usd"]),
            el(
                "small",
                f"calc {_usd(r['calculated_cost_usd'])} + est {_usd(r['estimated_debit_usd'])}",
            ),
        ]
    )


def _table(rows: Sequence[Row], *, empty: str = "No rows.") -> Html:
    if not rows:
        return el("p", empty, class_="note")
    columns = list(dict.fromkeys(c for r in rows for c in r))
    head = el("tr", *(el("th", _label(c)) for c in columns))
    body = [
        el("tr", *(el("td", _show(c, r.get(c), r), class_=_align(c, r.get(c))) for c in columns))
        for r in rows
    ]
    return el("div", el("table", el("thead", head), el("tbody", *body)), class_="scroll")


def _fields(row: Row) -> Html:
    return el(
        "table",
        el(
            "tbody",
            *(
                el("tr", el("th", _label(k), scope="row"), el("td", _show(k, v, row)))
                for k, v in row.items()
            ),
        ),
        class_="fields",
    )


# --- value formatting ----------------------------------------------------------------------


def _show(column: str, value: object, row: Row) -> str:
    """Display text. Unknown token counts read "unknown", never 0."""
    if value is None:
        if column in TOKEN_COLUMNS and row.get("usage") != "n/a":
            return "unknown"
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, dict):
        if {"covered", "eligible", "pct"} <= value.keys():
            pct = value["pct"]
            share = f"{pct}%" if pct is not None else "no eligible traffic"
            return f"{value['covered']} of {value['eligible']} ({share})"
        return ", ".join(f"{k}: {v}" for k, v in sorted(value.items())) or "—"
    if isinstance(value, list):
        return "; ".join(_show(column, v, row) for v in value) or "—"
    if isinstance(value, float):
        return f"{value:,.1f}"
    if isinstance(value, int):
        return f"{value:,}" if column not in ("http_status", "upstream_status") else str(value)
    if column.endswith("_usd") and isinstance(value, str):
        return _usd(value)
    return str(value) if value != "" else "—"


def _transposed(rows: Sequence[Row], key: str) -> Html:
    """One column per group and one row per field: chargeback rows are long and few."""
    if not rows:
        return el("p", "No rows.", class_="note")
    names = ["(none)" if r[key] == "-" else r[key] for r in rows]
    head = el("tr", el("th", key.capitalize()), *(el("th", n, class_="num") for n in names))
    body = [
        el(
            "tr",
            el("th", _label(field), scope="row"),
            *(el("td", _show(field, r.get(field), r), class_="num") for r in rows),
        )
        for field in dict.fromkeys(f for r in rows for f in r)
        if field != key
    ]
    table = el("table", el("thead", head), el("tbody", *body), class_="fields")
    return el("div", table, class_="scroll")


LABELS = {
    "ts_start": "Started (UTC)",
    "ts_end": "Ended (UTC)",
    "ts_admitted": "Admitted (UTC)",
    "http_status": "HTTP status",
    "upstream_status": "Upstream HTTP status",
    "ttfb_ms": "Time to first byte (ms)",
    "request_id": "Request ID",
    "attempt_no": "Attempt",
}


def _label(column: str) -> str:
    if column in LABELS:
        return LABELS[column]
    text = column.replace("_usd", " (USD)").replace("_ms", " (ms)").replace("_", " ")
    return text[:1].upper() + text[1:]


def _align(column: str, value: object) -> str | None:
    number = isinstance(value, (int, float)) and not isinstance(value, bool)
    return "num" if number or column.endswith("_usd") else None


def _usd(value: str) -> str:
    return f"-${value[1:]}" if value.startswith("-") else f"${value}"


def _int(value: int) -> str:
    return f"{value:,}"


def _ms(value: float | None) -> str:
    return "—" if value is None else f"{value:.1f} ms"


def _num(value: float | None) -> str:
    return "—" if value is None else f"{value:.1f}"


def _short_time(value: str) -> str:
    return value.replace("T", " ")[:23].removesuffix("+00:00")


def _interleave(items: Sequence[Html], separator: str) -> list[object]:
    out: list[object] = []
    for i, item in enumerate(items):
        if i:
            out.append(separator)
        out.append(item)
    return out

"""Read-only report UI against a real gate, mock upstream and SQLite.

Traffic goes through the live gate first; the UI is then built from a config loaded
without provider keys and reads the same database file.
"""

import json
import re
import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from gate.adapters import cli
from gate.adapters.config_yaml import validate_config
from gate.adapters.sqlite_store import SqliteKeyRepository, connect, connect_readonly
from gate.adapters.web_ui import create_ui_app, el, esc
from gate.app.keys import KeyService, NewKey
from gate.domain.config import GatewayConfig
from tests.integration.test_http import Env, finished_parents, post, wait_for_attempts

HOSTILE_OWNER = '<script>alert("owner")</script>'
HOSTILE_TEAM = '"><img src=x onerror=alert(1)>'
PAGES = ("/", "/requests", "/reports", "/budgets", "/coverage", "/keys")


@dataclass
class Ui:
    client: TestClient
    config: GatewayConfig
    config_file: Path
    env: Env


def ui_config(raw: dict[str, object]) -> GatewayConfig:
    # No MOCK_KEY: the UI must run without any provider key in its environment.
    report = validate_config(json.loads(json.dumps(raw)), {}, check_secrets=False)
    assert report.config is not None, report.errors
    return report.config


@pytest.fixture
def ui(env: Env, tmp_path: Path) -> Iterator[Ui]:
    conn = connect(str(env.db))
    _, hostile_key = KeyService(SqliteKeyRepository(conn), cli.SystemClock()).create(
        NewKey(owner=HOSTILE_OWNER, team=HOSTILE_TEAM, app="x", env="dev", models=["mock-model"])
    )
    conn.close()

    def message(model: str, **meta: str) -> dict[str, object]:
        body: dict[str, object] = {"model": model, "max_tokens": 10, "messages": []}
        if meta:
            body["metadata"] = {"user_id": "mock:" + ",".join(f"{k}={v}" for k, v in meta.items())}
        return body

    assert post(env, message("mock-model", input="100", output="20")).status_code == 200
    assert post(env, message("seek-model")).status_code == 200
    streamed = {**message("mock-model", truncate="1"), "stream": True}
    assert post(env, streamed).status_code == 200  # partial usage: estimated debit
    assert post(env, message("nope")).status_code == 403  # blocked: model_not_allowed
    hostile = message("mock-model")
    assert post(env, hostile, **{"x-api-key": hostile_key}).status_code == 200
    wait_for_attempts(env, 4)
    finished_parents(env, 5)

    config_file = tmp_path / "config.yaml"
    config_file.write_text(json.dumps(env.raw_config))  # JSON is valid YAML
    config = ui_config(env.raw_config)
    with TestClient(create_ui_app(config)) as client:
        yield Ui(client, config, config_file, env)


def cli_out(ui: Ui, capsys: pytest.CaptureFixture[str], *args: str) -> str:
    capsys.readouterr()
    assert cli.run(["--config", str(ui.config_file), *args], serve=lambda _: 0, env={}) == 0
    return capsys.readouterr().out


def month() -> str:
    return datetime.now(UTC).strftime("month:%Y-%m")


# --- one source of numbers ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("path", "args"),
    [
        ("/reports?by=provider&range=all", ("report", "--by", "provider")),
        ("/reports?by=team&range=all", ("report", "--by", "team")),
        ("/reports?by=model&range=all", ("report", "--by", "model")),
        ("/reports?by=all&range=all", ("report", "--by", "all")),
        ("/coverage?range=all", ("report", "coverage")),
        ("/requests", ("report", "requests")),
        ("/requests?provider=mockseek", ("report", "requests", "--provider", "mockseek")),
        ("/requests?status=blocked", ("report", "requests", "--status", "blocked")),
        ("/keys", ("keys", "list")),
        ("/keys?include_revoked=1", ("keys", "list", "--include-revoked")),
    ],
)
@pytest.mark.parametrize("fmt", ["json", "csv"])
def test_downloads_equal_cli_output(
    ui: Ui, capsys: pytest.CaptureFixture[str], path: str, args: tuple[str, ...], fmt: str
) -> None:
    joiner = "&" if "?" in path else "?"
    response = ui.client.get(f"{path}{joiner}format={fmt}")
    assert response.status_code == 200
    assert response.text == cli_out(ui, capsys, *args, "--format", fmt)


def test_budget_download_equals_cli(ui: Ui, capsys: pytest.CaptureFixture[str]) -> None:
    for scope in ("team", "key"):
        response = ui.client.get(f"/budgets?by={scope}&period={month()}&format=json")
        expected = cli_out(
            ui, capsys, "report", "budgets", "--by", scope, "--period", month(), "--format", "json"
        )
        assert response.text == expected
        # Only the team has a budget, so only team-period debits exist.
        assert bool(json.loads(expected)) is (scope == "team")


def test_request_detail_download_equals_cli(ui: Ui, capsys: pytest.CaptureFixture[str]) -> None:
    [first] = json.loads(ui.client.get("/requests?status=ok&provider=mockseek&format=json").text)
    rid = first["request_id"]
    expected = cli_out(ui, capsys, "report", "request", rid, "--format", "json")
    assert ui.client.get(f"/requests/{rid}?format=json").text == expected
    [detail] = json.loads(expected)
    assert [a["provider"] for a in detail["attempt_list"]] == ["mockseek"]


def test_html_shows_the_report_values(ui: Ui) -> None:
    [total] = json.loads(ui.client.get("/reports?by=all&range=all&format=json").text)
    page = ui.client.get("/?range=all").text
    for column in ("calculated_cost_usd", "estimated_debit_usd", "total_debit_usd"):
        assert f"${total[column]}" in page
    assert total["estimated_debit_usd"] != "0.000000"  # the truncated stream
    assert total["blocked_requests_by_reason"] == {"model_not_allowed": 1}

    providers = json.loads(ui.client.get("/reports?by=provider&range=all&format=json").text)
    for row in providers:
        assert f"${row['total_debit_usd']}" in page

    [team_budget] = [
        b
        for b in json.loads(ui.client.get(f"/budgets?period={month()}&format=json").text)
        if b["team"] == "platform"
    ]
    assert f"${team_budget['total_debit_usd']} of ${team_budget['budget_usd']}" in page


def test_request_pages_show_status_and_attempts(ui: Ui) -> None:
    rows = json.loads(ui.client.get("/requests?format=json").text)
    assert {r["status"] for r in rows} == {"ok", "blocked"}
    partial = [r for r in rows if r["usage"] == "partial"]
    assert len(partial) == 1 and partial[0]["estimated_debit_usd"] != "0.000000"

    blocked = next(r for r in rows if r["status"] == "blocked")
    page = ui.client.get(f"/requests/{blocked['request_id']}").text
    assert "No upstream attempt was made." in page and "model_not_allowed" in page
    assert ui.client.get("/requests/does-not-exist").status_code == 404


# --- boundaries --------------------------------------------------------------------------


def test_database_connection_rejects_writes(ui: Ui) -> None:
    conn = connect_readonly(str(ui.env.db))
    try:
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("DELETE FROM usage")
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("UPDATE keys SET revoked_at = 'now'")
    finally:
        conn.close()


@pytest.mark.parametrize("path", [*PAGES, "/requests/x", "/healthz"])
@pytest.mark.parametrize("method", ["POST", "PUT", "PATCH", "DELETE"])
def test_no_route_accepts_writes(ui: Ui, path: str, method: str) -> None:
    assert ui.client.request(method, path).status_code == 405


def test_stored_names_are_escaped(ui: Ui) -> None:
    for path in ("/requests", "/keys", "/reports?by=owner&range=all", "/reports?by=team"):
        page = ui.client.get(path).text
        assert HOSTILE_OWNER not in page and HOSTILE_TEAM not in page
        assert "<script" not in page and "<img" not in page
    assert esc(HOSTILE_OWNER) in ui.client.get("/requests").text
    assert esc(HOSTILE_TEAM) in ui.client.get("/keys").text


def test_bad_input_is_escaped_too(ui: Ui) -> None:
    response = ui.client.get("/reports", params={"since": "<b>x</b>"})
    assert response.status_code == 400
    assert "<b>x</b>" not in response.text and esc("<b>x</b>") in response.text


def test_pages_carry_security_headers_and_nothing_external(ui: Ui) -> None:
    conn = connect(str(ui.env.db))
    hashes = [row["key_hash"] for row in conn.execute("SELECT key_hash FROM keys")]
    conn.close()
    assert hashes
    for path in PAGES:
        response = ui.client.get(path)
        assert response.status_code == 200, path
        assert "default-src 'none'" in response.headers["content-security-policy"]
        assert response.headers["x-content-type-options"] == "nosniff"
        assert response.headers["referrer-policy"] == "no-referrer"
        page = response.text
        assert not re.search(r"""(src|href|action)=["']?(https?:)?//""", page), path
        assert "<script" not in page and " style=" not in page
        assert not any(h in page for h in hashes)


def test_refresh_is_bounded(ui: Ui) -> None:
    assert '<meta http-equiv="refresh" content="5">' in ui.client.get("/?refresh=5").text
    for value in ("1", "61", "abc", ""):
        assert 'http-equiv="refresh"' not in ui.client.get(f"/?refresh={value}").text


def test_missing_database_is_unavailable_and_not_created(env: Env, tmp_path: Path) -> None:
    raw = {**env.raw_config, "storage": {"sqlite_path": str(tmp_path / "absent.db")}}
    with TestClient(create_ui_app(ui_config(raw))) as client:
        assert client.get("/healthz").status_code == 503
        assert client.get("/").status_code == 503
    assert not (tmp_path / "absent.db").exists()


# --- display rules -----------------------------------------------------------------------


def test_unknown_counters_read_unknown_not_zero(ui: Ui) -> None:
    from gate.adapters.web_ui import _show

    assert _show("input_tokens", None, {"usage": "partial"}) == "unknown"
    assert _show("input_tokens", None, {"usage": "complete"}) == "unknown"
    assert _show("input_tokens", None, {"usage": "n/a"}) == "—"
    assert _show("input_tokens", 0, {"usage": "complete"}) == "0"


def test_el_escapes_children_and_attributes() -> None:
    html = el("a", "<x>", href='"><script>', class_="c")
    assert html == '<a href="&quot;&gt;&lt;script&gt;" class="c">&lt;x&gt;</a>'
    assert el("input", name="q", value="<v>") == '<input name="q" value="&lt;v&gt;">'

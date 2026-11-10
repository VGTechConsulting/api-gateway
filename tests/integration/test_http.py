"""HTTP + mock upstream + SQLite: stream fidelity, no buffering, error paths.

Starts the real gate (uvicorn, FastAPI, httpx, SQLite) and the bench mock
upstream on loopback ports in background threads.
"""

import base64
import hashlib
import json
import socket
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest
import uvicorn

from bench import mock_upstream
from gate.adapters import cli
from gate.adapters.config_yaml import validate_config
from gate.adapters.jsonl_log import MemoryRequestLog
from gate.adapters.sqlite_store import SqliteKeyRepository, SqliteUsageRepository, connect
from gate.app.keys import KeyService, NewKey
from gate.app.reports import Reports
from gate.main import build_runtime


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class ThreadServer:
    def __init__(self, app: Any, port: int) -> None:
        self.server = uvicorn.Server(
            uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", lifespan="off")
        )
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    def __enter__(self) -> "ThreadServer":
        self.thread.start()
        deadline = time.time() + 5
        while not self.server.started:
            if time.time() > deadline:
                raise RuntimeError("server did not start")
            time.sleep(0.01)
        return self

    def __exit__(self, *_: object) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=5)


@dataclass
class Env:
    gate: str
    mock: str
    key: str
    db: Path
    log: MemoryRequestLog
    raw_config: dict[str, Any]


def config_dict(mock_url: str, db: Path) -> dict[str, Any]:
    return {
        "storage": {"sqlite_path": str(db), "request_log": "stdout"},
        "defaults": {"tpm": 1_000_000, "rpm": 1000},
        "limits": {"cleanup_timeout_s": 2},
        "deployments": [
            {
                "name": "mock",
                "format": "anthropic",
                "provider": "mock",
                "base_url": mock_url,
                "api_key_env": "MOCK_KEY",
                "region": "local",
            },
            {
                "name": "mockseek",
                "format": "anthropic",
                "provider": "mockseek",
                "base_url": mock_url + "/anthropic",
                "api_key_env": "MOCK_KEY",
                "region": "cn",
            },
        ],
        "aliases": {"mock-model": ["mock"], "seek-model": ["mockseek"]},
        "prices": {
            "mock": {
                "mock-model": {
                    "input": 3,
                    "output": 15,
                    "cache_read": 0.3,
                    "cache_write_5m": 3.75,
                    "cache_write_1h": 6,
                }
            },
            "mockseek": {
                "seek-model": {
                    "input": 0.28,
                    "output": 0.42,
                    "cache_read": 0.028,
                    "cache_write_5m": None,
                    "cache_write_1h": None,
                }
            },
        },
        "teams": {"platform": {"budget_usd": 5, "period": "month"}},
        "detect": {"categories": {"private_key": "block", "card_number": "flag"}},
    }


@pytest.fixture
def env(tmp_path: Path) -> Iterator[Env]:
    mock_port, gate_port = free_port(), free_port()
    mock_url = f"http://127.0.0.1:{mock_port}"
    db = tmp_path / "gate.db"
    raw = config_dict(mock_url, db)
    report = validate_config(json.loads(json.dumps(raw)), {"MOCK_KEY": "sk-mock"})
    assert report.ok, report.errors
    assert report.config is not None
    log = MemoryRequestLog()
    runtime = build_runtime(report.config, env={"MOCK_KEY": "sk-mock"}, request_log=log)

    conn = connect(str(db))
    _, key = KeyService(SqliteKeyRepository(conn), cli.SystemClock()).create(
        NewKey(
            owner="alice",
            team="platform",
            app="demo",
            env="dev",
            models=["mock-model", "seek-model"],
        )
    )
    conn.close()
    with ThreadServer(mock_upstream.app, mock_port), ThreadServer(runtime.app, gate_port):
        yield Env(f"http://127.0.0.1:{gate_port}", mock_url, key, db, log, raw)


def post(env: Env, body: dict[str, Any] | bytes, **headers: str) -> httpx.Response:
    raw = body if isinstance(body, bytes) else json.dumps(body).encode()
    return httpx.post(
        env.gate + "/v1/messages",
        content=raw,
        headers={
            "x-api-key": env.key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
            **headers,
        },
        timeout=10,
    )


def last_exchange(env: Env) -> tuple[bytes, dict[str, str], bytes]:
    last = httpx.get(env.mock + "/_last").json()
    response = httpx.get(env.mock + "/_last_response").content
    return base64.b64decode(last["body_b64"]), dict(last["headers"]), response


def attempts(env: Env) -> list[dict[str, Any]]:
    conn = connect(str(env.db))
    try:
        return [dict(r) for r in conn.execute("SELECT * FROM attempts ORDER BY ts_start")]
    finally:
        conn.close()


def wait_for_attempts(env: Env, n: int) -> list[dict[str, Any]]:
    deadline = time.time() + 5
    while time.time() < deadline:
        rows = attempts(env)
        if len(rows) >= n:
            return rows
        time.sleep(0.02)
    raise AssertionError(f"expected {n} attempts, have {attempts(env)}")


# --- transport fidelity ------------------------------------------------------------------


@pytest.mark.parametrize("model", ["mock-model", "seek-model"])
@pytest.mark.parametrize("stream", [True, False])
def test_request_and_response_bytes_are_unchanged(env: Env, model: str, stream: bool) -> None:
    raw = (
        b'{ "model" : "'
        + model.encode()
        + b'", "max_tokens":64,\n "stream": '
        + (b"true" if stream else b"false")
        + b', "messages":[{"role":"user","content":"caf\\u00e9 \xe2\x9c\x93"}],'
        b' "metadata":{"user_id":"mock:cache_5m=7,cache_1h=3,chunks=8"}}'
    )
    response = post(
        env,
        raw,
        **{"anthropic-beta": "prompt-caching-2024-07-31", "authorization": "Bearer client-secret"},
    )
    assert response.status_code == 200
    forwarded, headers, upstream_body = last_exchange(env)
    assert forwarded == raw
    assert hashlib.sha256(response.content).digest() == hashlib.sha256(upstream_body).digest()
    assert headers["x-api-key"] == "sk-mock"
    assert headers["anthropic-beta"] == "prompt-caching-2024-07-31"
    assert "authorization" not in headers
    assert response.headers["x-api-gateway-request-id"]

    [row] = wait_for_attempts(env, 1)
    assert row["usage_source"] == "ok"
    assert (row["cache_write_5m_tokens"], row["cache_write_1h_tokens"]) == (7, 3)
    if model == "mock-model":
        assert (row["outcome"], row["pricing_status"]) == ("complete", "ok")
    else:
        # seek-model prices cache writes as null (unsupported): nonzero writes are unpriceable.
        assert (row["outcome"], row["pricing_status"]) == ("unknown", "unsupported")
        assert row["cost_micros"] == row["estimate_micros"]


def test_first_chunk_arrives_before_upstream_finishes(env: Env) -> None:
    body = {
        "model": "mock-model",
        "max_tokens": 10,
        "stream": True,
        "messages": [],
        "metadata": {"user_id": "mock:chunks=4,delay_ms=150"},
    }
    started = time.monotonic()
    with httpx.stream(
        "POST", env.gate + "/v1/messages", json=body, headers={"x-api-key": env.key}, timeout=10
    ) as response:
        chunks = response.iter_raw()
        first = next(chunks)
        first_at = time.monotonic() - started
        rest = b"".join(chunks)
        total = time.monotonic() - started
    assert first.startswith(b"event: message_start")
    assert rest.endswith(b"\n\n")
    assert first_at < total / 2


# --- error paths -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("body", "headers", "status", "code"),
    [
        (
            {"model": "mock-model", "max_tokens": 1},
            {"x-api-key": "gk_wrong"},
            401,
            "invalid_api_key",
        ),
        (b'{"model":"mock-model","max_tokens":1,"max_tokens":2}', {}, 400, "duplicate_key"),
        ({"model": "mock-model", "max_tokens": -1}, {}, 400, "invalid_output_limit"),
        ({"model": "nope", "max_tokens": 1}, {}, 403, "model_not_allowed"),
        (
            {
                "model": "mock-model",
                "max_tokens": 1,
                "messages": [{"role": "user", "content": "-----BEGIN EC PRIVATE KEY-----"}],
            },
            {},
            400,
            "content_blocked",
        ),
    ],
)
def test_gate_errors_use_anthropic_shape_and_marker_header(
    env: Env, body: Any, headers: dict[str, str], status: int, code: str
) -> None:
    response = post(env, body, **headers)
    assert response.status_code == status
    payload = response.json()
    assert payload["type"] == "error" and payload["gateway_code"] == code
    assert payload["error"]["type"]
    assert response.headers["x-api-gateway-error"] == code


def test_unknown_path_is_404_in_provider_shape(env: Env) -> None:
    response = httpx.get(env.gate + "/v1/models", headers={"x-api-key": env.key})
    assert response.status_code == 404
    assert response.json()["error"]["type"] == "not_found_error"


def test_upstream_error_is_relayed_unchanged_and_debited_at_reservation(env: Env) -> None:
    body = {
        "model": "mock-model",
        "max_tokens": 10,
        "messages": [],
        "metadata": {"user_id": "mock:status=400"},
    }
    response = post(env, body)
    _, _, upstream_body = last_exchange(env)
    assert response.status_code == 400 and response.content == upstream_body
    assert "x-api-gateway-error" not in response.headers
    [row] = wait_for_attempts(env, 1)
    assert row["outcome"] == "unknown" and row["cost_micros"] == row["estimate_micros"]


def test_documented_rejection_is_non_billable(env: Env) -> None:
    body = {
        "model": "mock-model",
        "max_tokens": 10,
        "messages": [],
        "metadata": {"user_id": "mock:status=401"},
    }
    assert post(env, body).status_code == 401
    [row] = wait_for_attempts(env, 1)
    assert (row["outcome"], row["cost_micros"], row["accounting_kind"]) == (
        "rejected_non_billable",
        0,
        "non_billable",
    )


def test_revoked_key_is_rejected_on_the_next_request(env: Env) -> None:
    assert post(env, {"model": "mock-model", "max_tokens": 1, "messages": []}).status_code == 200
    conn = connect(str(env.db))
    KeyService(SqliteKeyRepository(conn), cli.SystemClock()).revoke(env.key[3:11])
    conn.close()
    assert post(env, {"model": "mock-model", "max_tokens": 1, "messages": []}).status_code == 401


# --- interrupted streams -----------------------------------------------------------------


def test_client_disconnect_mid_stream_is_settled(env: Env) -> None:
    body = {
        "model": "mock-model",
        "max_tokens": 10,
        "stream": True,
        "messages": [],
        "metadata": {"user_id": "mock:chunks=50,delay_ms=50"},
    }
    with httpx.stream(
        "POST", env.gate + "/v1/messages", json=body, headers={"x-api-key": env.key}, timeout=10
    ) as response:
        next(response.iter_raw())
    [row] = wait_for_attempts(env, 1)
    assert row["termination_reason"] == "client_disconnect"
    assert row["client_disconnected"] == 1 and row["usage_complete"] == 0
    assert row["accounting_kind"] == "estimated" and row["cost_micros"] >= row["estimate_micros"]


@pytest.mark.parametrize(
    ("scenario", "termination"),
    [("truncate=1", "upstream_disconnect"), ("sse_error=1", "stream_error")],
)
def test_incomplete_streams_use_estimated_debits(env: Env, scenario: str, termination: str) -> None:
    body = {
        "model": "mock-model",
        "max_tokens": 10,
        "stream": True,
        "messages": [],
        "metadata": {"user_id": f"mock:{scenario}"},
    }
    assert post(env, body).status_code == 200
    [row] = wait_for_attempts(env, 1)
    assert row["termination_reason"] == termination
    assert row["outcome"] == "partial" and row["priced_cost_micros"] is None


# --- persistence, restart, reports -------------------------------------------------------


def test_restart_rebuilds_spend_and_marks_open_requests_interrupted(env: Env) -> None:
    assert post(env, {"model": "mock-model", "max_tokens": 10, "messages": []}).status_code == 200
    wait_for_attempts(env, 1)
    conn = connect(str(env.db))
    conn.execute(
        "INSERT INTO usage (request_id, request_state, ts_start, route)"
        " VALUES ('orphan', 'open', '2026-11-10T00:00:00.000000Z', '/v1/messages')"
    )
    conn.close()

    report = validate_config(json.loads(json.dumps(env.raw_config)), {"MOCK_KEY": "x"})
    assert report.config is not None
    restarted = build_runtime(report.config, env={"MOCK_KEY": "x"}, request_log=MemoryRequestLog())
    usage = SqliteUsageRepository(connect(str(env.db)))
    balances = dict(usage.spend_by_bucket())
    [(bucket, spent)] = [(b, s) for b, s in balances.items() if b.scope_type.value == "team"]
    assert restarted.admission.spent(bucket) == spent > 0
    states = {p.request_id: p.request_state.value for p in usage.requests(since=None, until=None)}
    assert states["orphan"] == "interrupted"


def test_report_cli_separates_providers(
    env: Env, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    for model in ("mock-model", "seek-model"):
        assert post(env, {"model": model, "max_tokens": 10, "messages": []}).status_code == 200
    wait_for_attempts(env, 2)
    config_file = tmp_path / "config.yaml"
    config_file.write_text(json.dumps(env.raw_config))  # JSON is valid YAML
    assert (
        cli.run(
            ["--config", str(config_file), "report", "--by", "provider", "--format", "json"],
            serve=lambda _: 0,
            env={},
        )
        == 0
    )
    rows = {r["provider"]: r for r in json.loads(capsys.readouterr().out)}
    assert set(rows) == {"mock", "mockseek"}
    assert all(r["attempts"] == 1 and r["requests"] == 1 for r in rows.values())
    assert rows["mock"]["calculated_cost_usd"] != "0.000000"


def finished_parents(env: Env, n: int) -> None:
    deadline = time.time() + 5
    while time.time() < deadline:
        usage = SqliteUsageRepository(connect(str(env.db)))
        done = [p for p in usage.requests(since=None, until=None) if p.ts_end is not None]
        if len(done) >= n:
            return
        time.sleep(0.02)
    raise AssertionError(f"expected {n} finished requests")


def test_report_counts_locally_denied_requests(env: Env) -> None:
    assert post(env, {"model": "mock-model", "max_tokens": 10, "messages": []}).status_code == 200
    assert post(env, {"model": "nope", "max_tokens": 1, "messages": []}).status_code == 403
    wait_for_attempts(env, 1)
    finished_parents(env, 2)
    reports = Reports(SqliteUsageRepository(connect(str(env.db))))

    [team] = reports.by("team", since=None, until=None)
    assert team["requests"] == 2 and team["attempts"] == 1
    assert team["blocked_requests_by_reason"] == {"model_not_allowed": 1}

    providers = {r["provider"]: r for r in reports.by("provider", since=None, until=None)}
    assert providers["mock"]["requests"] == 1 and providers["mock"]["attempts"] == 1
    assert providers["-"]["requests"] == 1 and providers["-"]["attempts"] == 0


def test_report_filters_requests_by_start_and_attempts_by_admission(env: Env) -> None:
    assert post(env, {"model": "mock-model", "max_tokens": 10, "messages": []}).status_code == 200
    [attempt] = wait_for_attempts(env, 1)
    finished_parents(env, 1)
    conn = connect(str(env.db))
    conn.execute("UPDATE usage SET ts_start = '2026-01-01T00:00:00.000000Z'")
    conn.commit()
    reports = Reports(SqliteUsageRepository(conn))
    admitted = datetime.fromisoformat(attempt["ts_admitted"].replace("Z", "+00:00"))

    for group in ("team", "provider"):
        [row] = reports.by(group, since=admitted, until=None)
        assert row["attempts"] == 1 and row["requests"] == 0
    # A request that started in the window counts in its attempt's group even when
    # that attempt was admitted after ``--until``.
    early = datetime(2026, 1, 1, tzinfo=UTC)
    for group in ("team", "provider"):
        [row] = reports.by(group, since=early, until=admitted)
        assert row["attempts"] == 0 and row["requests"] == 1
    [row] = reports.by("provider", since=early, until=admitted)
    assert row["provider"] == "mock"

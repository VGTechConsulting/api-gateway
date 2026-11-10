"""Composition root: load config, build adapters, start the server or run the CLI.

Wiring only. Every dependency is built explicitly here; there is no DI framework.
"""

import logging
import os
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import FrameType

import uvicorn
from fastapi import FastAPI

from gate.adapters import cli
from gate.adapters.http import create_app
from gate.adapters.jsonl_log import JsonlRequestLog
from gate.adapters.sqlite_store import SqliteKeyRepository, SqliteUsageRepository, connect, migrate
from gate.adapters.upstream_httpx import HttpxUpstream
from gate.adapters.web_ui import create_ui_app
from gate.app.admission import AdmissionController
from gate.app.lifecycle import HandleRequest
from gate.app.ports import RequestLog, Upstream
from gate.app.routing import Router
from gate.domain.config import GatewayConfig
from gate.domain.models import ApiFormat
from gate.providers.anthropic import AnthropicFormat

log = logging.getLogger("api-gateway")


class MonotonicSystemClock:
    def now(self) -> float:
        return time.monotonic()


@dataclass
class Runtime:
    app: FastAPI
    handler: HandleRequest
    admission: AdmissionController
    upstream: Upstream
    shutdown: "ShutdownFlag"


class ShutdownFlag:
    def __init__(self) -> None:
        self.set = False

    def __call__(self) -> bool:
        return self.set


def build_runtime(
    config: GatewayConfig,
    *,
    env: Mapping[str, str] | None = None,
    upstream: Upstream | None = None,
    request_log: RequestLog | None = None,
) -> Runtime:
    env = os.environ if env is None else env
    conn = connect(config.sqlite_path)
    migrate(conn)
    keys = SqliteKeyRepository(conn)
    usage = SqliteUsageRepository(conn)

    interrupted = usage.mark_interrupted()
    if interrupted:
        log.warning("marked %d unfinished requests from a previous run as interrupted", interrupted)

    wall, mono = cli.SystemClock(), MonotonicSystemClock()
    admission = AdmissionController(wall, mono, naive_budgets=config.budget_mode_naive)
    admission.load_spent(usage.spend_by_bucket())

    upstream = upstream or HttpxUpstream(
        connect_timeout_s=config.limits.connect_timeout_s,
        read_timeout_s=config.limits.upstream_timeout_s,
    )
    shutdown = ShutdownFlag()
    handler = HandleRequest(
        config=config,
        keys=keys,
        usage=usage,
        upstream=upstream,
        request_log=request_log or JsonlRequestLog(config.request_log),
        admission=admission,
        router=Router(config.deployments, config.aliases, config.prices),
        providers={ApiFormat.ANTHROPIC: AnthropicFormat()},
        wall=wall,
        mono=mono,
        env=env,
        shutting_down=shutdown,
    )

    def healthy() -> bool:
        return not admission.disabled and usage.writable()

    app = create_app(
        handler,
        max_request_bytes=config.limits.max_request_bytes,
        healthy=healthy,
        cleanup_timeout_s=config.limits.cleanup_timeout_s,
    )
    return Runtime(app, handler, admission, upstream, shutdown)


class GatewayServer(uvicorn.Server):
    """uvicorn server that tells the lifecycle when shutdown starts (no new fallbacks)."""

    def __init__(self, config: uvicorn.Config, shutdown: ShutdownFlag) -> None:
        super().__init__(config)
        self._shutdown = shutdown

    def handle_exit(self, sig: int, frame: FrameType | None) -> None:
        self._shutdown.set = True
        super().handle_exit(sig, frame)


def serve(config: GatewayConfig) -> int:
    logging.basicConfig(
        level=logging.INFO, stream=sys.stderr, format="%(levelname)s %(name)s %(message)s"
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    runtime = build_runtime(config)
    server = GatewayServer(
        uvicorn.Config(
            runtime.app,
            host=config.host,
            port=config.port,
            log_level="info",
            access_log=False,
            timeout_graceful_shutdown=int(config.limits.shutdown_grace_s),
            log_config=None,
        ),
        runtime.shutdown,
    )
    server.run()
    return 0


def ui(config: GatewayConfig) -> int:
    """Run the read-only report UI on its own listener; it never serves the gate."""
    logging.basicConfig(
        level=logging.INFO, stream=sys.stderr, format="%(levelname)s %(name)s %(message)s"
    )
    uvicorn.run(
        create_ui_app(config),
        host=config.ui.host,
        port=config.ui.port,
        log_level="info",
        access_log=False,
        log_config=None,
    )
    return 0


def main(argv: Sequence[str] | None = None) -> None:
    sys.exit(cli.run(argv, serve=serve, ui=ui))


if __name__ == "__main__":
    main()

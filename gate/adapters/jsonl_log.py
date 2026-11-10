"""Metadata-only JSONL request log.

One ``attempt_settled`` event per settled attempt and one ``request_finished``
event per parent. Events never contain prompt or response bodies, keys, or
matched detection text: the lifecycle builds them from metadata fields only.
"""

import json
import sys
import threading
from collections.abc import Mapping
from typing import IO, Any


class JsonlRequestLog:
    def __init__(self, destination: str) -> None:
        self._lock = threading.Lock()
        self._owned = destination != "stdout"
        self._stream: IO[str] = (
            open(destination, "a", encoding="utf-8") if self._owned else sys.stdout  # noqa: SIM115
        )

    def write(self, event: Mapping[str, Any]) -> None:
        line = json.dumps(event, separators=(",", ":"), default=str)
        with self._lock:
            self._stream.write(line + "\n")
            self._stream.flush()

    def close(self) -> None:
        if self._owned:
            self._stream.close()


class MemoryRequestLog:
    """Collects events in memory (tests and benchmarks)."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def write(self, event: Mapping[str, Any]) -> None:
        self.events.append(dict(event))

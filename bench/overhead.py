"""Gate overhead benchmark: direct to the mock upstream vs through the gate.

Measures time to first byte (TTFB) and total time for streaming requests at a
given concurrency, once directly against the mock and once through the gate,
and reports p50/p95/p99 of the difference.

    # terminal 1: python -m bench.mock_upstream --port 9999
    # terminal 2: api-gateway serve   (config routes mock-model -> http://127.0.0.1:9999)
    GATE_KEY=gk_... python -m bench.overhead --concurrency 1 10 50 --requests 200

Gate rate limits must be non-limiting for the test key. Detection on/off is a
gate config setting; run once per setting and label the results.
"""

import argparse
import asyncio
import json
import os
import statistics
import sys
import time

import httpx

BODY = {
    "model": "mock-model",
    "max_tokens": 64,
    "stream": True,
    "messages": [{"role": "user", "content": "synthetic benchmark prompt " * 20}],
    "metadata": {"user_id": "mock:chunks=20,delay_ms=5"},
}


def pct(values: list[float], p: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, round(p / 100 * len(ordered)) - 1))]


async def one(client: httpx.AsyncClient, url: str, headers: dict[str, str]) -> tuple[float, float]:
    started = time.perf_counter()
    async with client.stream("POST", url, content=json.dumps(BODY), headers=headers) as response:
        response.raise_for_status()
        ttfb = None
        async for _ in response.aiter_raw():
            if ttfb is None:
                ttfb = time.perf_counter() - started
    total = time.perf_counter() - started
    return (ttfb or total) * 1000, total * 1000


async def run(
    url: str, headers: dict[str, str], concurrency: int, requests: int
) -> list[tuple[float, float]]:
    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    async with httpx.AsyncClient(timeout=60, limits=limits) as client:
        for _ in range(min(10, requests)):  # warm up connections
            await one(client, url, headers)
        semaphore = asyncio.Semaphore(concurrency)

        async def bounded() -> tuple[float, float]:
            async with semaphore:
                return await one(client, url, headers)

        return await asyncio.gather(*(bounded() for _ in range(requests)))


def summarize(
    label: str, direct: list[tuple[float, float]], gated: list[tuple[float, float]]
) -> None:
    d_ttfb = statistics.median(t for t, _ in direct)
    rows = {
        "ttfb_overhead_ms": [t - d_ttfb for t, _ in gated],
        "total_overhead_ms": [t - statistics.median(x for _, x in direct) for _, t in gated],
    }
    for name, values in rows.items():
        print(
            f"{label:>14} {name:<18} p50={pct(values, 50):7.2f}"
            f" p95={pct(values, 95):7.2f} p99={pct(values, 99):7.2f}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", default="http://127.0.0.1:9999")
    parser.add_argument("--gate", default="http://127.0.0.1:8787")
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 10, 50])
    parser.add_argument("--requests", type=int, default=200)
    args = parser.parse_args()
    key = os.environ.get("GATE_KEY")
    if not key:
        print("set GATE_KEY", file=sys.stderr)
        return 2
    for concurrency in args.concurrency:
        direct = asyncio.run(run(args.mock + "/v1/messages", {}, concurrency, args.requests))
        gated = asyncio.run(
            run(args.gate + "/v1/messages", {"x-api-key": key}, concurrency, args.requests)
        )
        summarize(f"c={concurrency}", direct, gated)
    return 0


if __name__ == "__main__":
    sys.exit(main())

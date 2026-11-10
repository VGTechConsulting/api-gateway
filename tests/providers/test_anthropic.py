"""Anthropic Messages format: usage parsing, completeness, headers, pricing features,
non-billable rejection rules and error shapes.

Fixtures are synthetic SSE streams shaped like recorded Anthropic responses, with
placeholder IDs. The parser sees the bytes the relay has already forwarded, in whatever
chunks the network delivered, so every result must be independent of chunk boundaries.
"""

import itertools
import json
from decimal import Decimal
from typing import Any

import pytest

from gate.domain.accounting import observed_total
from gate.domain.detection import scan
from gate.domain.models import (
    CacheBreakdown,
    PriceEntry,
    PricingMode,
    PricingStatus,
    TokenCounts,
    UsageSource,
)
from gate.domain.pricing import price_usage
from gate.providers.anthropic import AnthropicFormat, AnthropicUsageParser
from gate.providers.base import ParseResult

FORMAT = AnthropicFormat()
SSE = "text/event-stream; charset=utf-8"
MAX = 1 << 20

# --- fixtures ----------------------------------------------------------------------------

START_USAGE: dict[str, Any] = {
    "input_tokens": 25,
    "cache_creation_input_tokens": 300,
    "cache_read_input_tokens": 1_000,
    "cache_creation": {"ephemeral_5m_input_tokens": 100, "ephemeral_1h_input_tokens": 200},
    "output_tokens": 1,
    "service_tier": "standard",
}
EXPECTED_COUNTS = TokenCounts(
    input=25, output=42, cache_read=1_000, cache_write=300, cache_write_5m=100, cache_write_1h=200
)


def sse(kind: str, data: dict[str, Any]) -> str:
    return f"event: {kind}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def message_start(usage: dict[str, Any] | None = None, model: str = "claude-sonnet-5-5") -> str:
    message: dict[str, Any] = {
        "id": "msg_PLACEHOLDER",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [],
        "stop_reason": None,
        "usage": START_USAGE if usage is None else usage,
    }
    return sse("message_start", {"type": "message_start", "message": message})


CONTENT = (
    sse(
        "content_block_start",
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
    )
    + sse("ping", {"type": "ping"})
    + ": keep-alive comment\n\n"
    + sse(
        "content_block_delta",
        # Multi-byte UTF-8: 2-, 3- and 4-byte sequences, sent raw rather than escaped.
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "Héllo — 世界 🌍"},
        },
    )
    + sse("content_block_stop", {"type": "content_block_stop", "index": 0})
)


def message_delta(usage: dict[str, Any] | None = None) -> str:
    return sse(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": 42} if usage is None else usage,
        },
    )


MESSAGE_STOP = sse("message_stop", {"type": "message_stop"})
ERROR_EVENT = sse(
    "error", {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}
)

COMPLETE_STREAM = message_start() + CONTENT + message_delta() + MESSAGE_STOP

NON_STREAM_BODY = json.dumps(
    {
        "id": "msg_PLACEHOLDER",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-5-5",
        "content": [{"type": "text", "text": "Héllo — 世界 🌍"}],
        "stop_reason": "end_turn",
        "usage": {**START_USAGE, "output_tokens": 42},
    },
    ensure_ascii=False,
).encode()


def run_stream(
    text: str | bytes, *, chunks: list[int] | None = None, status: int = 200, max_buffer: int = MAX
) -> ParseResult:
    data = text.encode() if isinstance(text, str) else text
    parser = FORMAT.new_parser(content_type=SSE, http_status=status, max_buffer_bytes=max_buffer)
    feed(parser, data, chunks)
    return parser.result(eof=True)


def run_json(
    body: bytes,
    *,
    eof: bool = True,
    status: int = 200,
    max_buffer: int = MAX,
    chunks: list[int] | None = None,
) -> ParseResult:
    parser = FORMAT.new_parser(
        content_type="application/json", http_status=status, max_buffer_bytes=max_buffer
    )
    feed(parser, body, chunks)
    return parser.result(eof=eof)


def feed(parser: AnthropicUsageParser, data: bytes, cuts: list[int] | None) -> None:
    edges = [0, *(cuts or []), len(data)]
    for start, end in itertools.pairwise(edges):
        parser.feed(data[start:end])


def test_fixtures_contain_no_detector_patterns() -> None:
    # Scrubbed fixtures: provider fixtures must not match any detector.
    for text in (COMPLETE_STREAM, NON_STREAM_BODY.decode()):
        assert scan([text]) == {}


# --- complete stream ---------------------------------------------------------------------


def test_complete_stream() -> None:
    result = run_stream(COMPLETE_STREAM)
    assert result.usage_complete is True
    assert result.usage_source is UsageSource.OK
    assert result.streamed is True
    assert result.stream_complete is True
    assert result.stream_error is False
    assert result.upstream_model == "claude-sonnet-5-5"
    assert result.reasons == ()
    assert result.snapshot is not None
    assert result.snapshot.counts == EXPECTED_COUNTS
    assert result.snapshot.cache_breakdown is CacheBreakdown.OK
    assert result.snapshot.unsupported == ()


def test_every_two_way_split_gives_the_same_result() -> None:
    data = COMPLETE_STREAM.encode()
    expected = run_stream(data)
    for cut in range(1, len(data)):
        assert run_stream(data, chunks=[cut]) == expected, f"split at byte {cut}"


def test_byte_by_byte_feeding_gives_the_same_result() -> None:
    data = COMPLETE_STREAM.encode()
    assert run_stream(data, chunks=list(range(1, len(data)))) == run_stream(data)


def test_splits_inside_multibyte_utf8_are_exercised() -> None:
    data = COMPLETE_STREAM.encode()
    emoji = "🌍".encode()
    at = data.index(emoji)
    expected = run_stream(data)
    for cut in range(at + 1, at + len(emoji)):  # strictly inside the 4-byte sequence
        assert run_stream(data, chunks=[cut]) == expected


def test_crlf_line_endings() -> None:
    data = COMPLETE_STREAM.replace("\n", "\r\n").encode()
    expected = run_stream(COMPLETE_STREAM)
    assert run_stream(data) == expected
    # Every split, including each one between a \r and its \n.
    crlf_cuts = [i + 1 for i in range(len(data) - 1) if data[i : i + 2] == b"\r\n"]
    assert crlf_cuts
    for cut in range(1, len(data)):
        assert run_stream(data, chunks=[cut]) == expected, f"split at byte {cut}"


def test_data_lines_without_a_space_after_the_colon() -> None:
    text = COMPLETE_STREAM.replace("data: ", "data:")
    assert run_stream(text) == run_stream(COMPLETE_STREAM)


# --- incomplete streams ------------------------------------------------------------------


def test_missing_message_stop_is_partial() -> None:
    result = run_stream(message_start() + CONTENT + message_delta())
    assert result.usage_complete is False
    assert result.usage_source is UsageSource.PARTIAL
    assert result.stream_complete is False
    assert result.snapshot is not None
    assert result.snapshot.counts.output == 42  # observed, but not final


def test_message_start_only_is_partial() -> None:
    result = run_stream(message_start())
    assert result.usage_complete is False
    assert result.usage_source is UsageSource.PARTIAL
    assert result.snapshot is not None
    assert result.snapshot.counts.output == 1


def test_message_stop_without_final_delta_usage_is_not_complete() -> None:
    result = run_stream(message_start() + CONTENT + MESSAGE_STOP)
    assert result.stream_complete is True
    assert result.usage_complete is False
    assert result.usage_source is UsageSource.PARTIAL


def test_empty_stream_is_missing() -> None:
    result = run_stream(b"")
    assert result.snapshot is None
    assert result.usage_source is UsageSource.MISSING
    assert result.usage_complete is False


def test_error_event_under_http_200_is_a_stream_error() -> None:
    result = run_stream(message_start() + CONTENT + ERROR_EVENT)
    assert result.stream_error is True
    assert result.error_type == "overloaded_error"
    assert result.usage_complete is False
    assert result.usage_source is UsageSource.PARTIAL
    assert "stream_error" in result.reasons


def test_error_event_prevents_later_completion() -> None:
    result = run_stream(message_start() + ERROR_EVENT + message_delta() + MESSAGE_STOP)
    assert result.stream_error is True
    assert result.usage_complete is False


def test_malformed_data_line_is_a_parse_error_and_keeps_the_last_snapshot() -> None:
    text = message_start() + "event: content_block_delta\ndata: {not json\n\n" + MESSAGE_STOP
    result = run_stream(text)
    assert result.usage_source is UsageSource.PARSE_ERROR
    assert result.usage_complete is False
    assert "parse_error" in result.reasons
    assert result.snapshot is not None
    assert result.snapshot.counts.input == 25


def test_parse_error_before_completion_is_never_promoted() -> None:
    text = message_start() + "data: [1, 2\n\n" + message_delta() + MESSAGE_STOP
    result = run_stream(text)
    assert result.usage_source is UsageSource.PARSE_ERROR
    assert result.usage_complete is False


# Nesting deep enough to exhaust the JSON decoder's recursion limit (RecursionError).
DEEP_JSON = "[" * 100_000


def test_deeply_nested_data_line_is_a_parse_error_not_an_exception() -> None:
    text = message_start() + f"data: {DEEP_JSON}\n\n" + message_delta() + MESSAGE_STOP
    result = run_stream(text)
    assert result.usage_source is UsageSource.PARSE_ERROR
    assert result.usage_complete is False
    assert "parse_error" in result.reasons
    # Parsing continues after the bad line: later events are still observed.
    assert result.stream_complete is True
    assert result.snapshot is not None
    assert result.snapshot.counts.output == 42


def test_deeply_nested_non_stream_body_is_a_parse_error_not_an_exception() -> None:
    result = run_json(DEEP_JSON.encode())
    assert result.usage_source is UsageSource.PARSE_ERROR
    assert result.usage_complete is False


def test_non_2xx_stream_is_not_complete() -> None:
    result = run_stream(COMPLETE_STREAM, status=500)
    assert result.usage_complete is False


# --- cache-write snapshots ---------------------------------------------------------------


def test_changed_aggregate_without_a_fresh_split_is_missing_breakdown() -> None:
    delta = message_delta({"output_tokens": 42, "cache_creation_input_tokens": 500})
    result = run_stream(message_start() + delta + MESSAGE_STOP)
    snapshot = result.snapshot
    assert snapshot is not None
    assert snapshot.cache_breakdown is CacheBreakdown.MISSING
    assert snapshot.counts.cache_write == 500
    # Stale duration counters are not combined with the newer aggregate.
    assert snapshot.counts.cache_write_5m is None
    assert snapshot.counts.cache_write_1h is None


def test_missing_breakdown_survives_into_pricing_status() -> None:
    price = PriceEntry(
        "d",
        "m",
        PricingMode.METERED,
        Decimal(3),
        Decimal(15),
        Decimal("0.3"),
        Decimal("3.75"),
        Decimal(6),
    )
    delta = message_delta({"output_tokens": 42, "cache_creation_input_tokens": 500})
    result = run_stream(message_start() + delta + MESSAGE_STOP)
    assert result.usage_complete is True  # usage is complete; only the split is missing
    priced = price_usage(result.snapshot, price)
    assert priced.status is PricingStatus.MISSING_CACHE_BREAKDOWN
    assert priced.cost_micros is None


def test_repeated_aggregate_without_split_keeps_the_earlier_split() -> None:
    # Shape observed from Anthropic on 2026-11-10: message_delta repeats the cumulative
    # aggregate but omits the cache_creation object that message_start carried.
    delta = message_delta(
        {
            "input_tokens": 25,
            "cache_creation_input_tokens": 300,
            "cache_read_input_tokens": 1_000,
            "output_tokens": 42,
            "output_tokens_details": {"thinking_tokens": 0},
        }
    )
    result = run_stream(message_start() + CONTENT + delta + MESSAGE_STOP)
    snapshot = result.snapshot
    assert snapshot is not None
    assert snapshot.cache_breakdown is CacheBreakdown.OK
    assert snapshot.counts == EXPECTED_COUNTS


def test_repeated_aggregate_is_priced_from_the_earlier_split() -> None:
    price = PriceEntry(
        "d",
        "m",
        PricingMode.METERED,
        Decimal(3),
        Decimal(15),
        Decimal("0.3"),
        Decimal("3.75"),
        Decimal(6),
    )
    delta = message_delta({"output_tokens": 42, "cache_creation_input_tokens": 300})
    result = run_stream(message_start() + delta + MESSAGE_STOP)
    priced = price_usage(result.snapshot, price)
    assert priced.status is PricingStatus.OK
    # 25*3 + 42*15 + 1000*0.3 + 100*3.75 + 200*6 = 2,580 USD per million tokens -> 2,580 micros
    assert priced.cost_micros == 2_580


def test_repeated_aggregate_without_any_split_stays_missing() -> None:
    start = {**START_USAGE}
    del start["cache_creation"]
    delta = message_delta({"output_tokens": 42, "cache_creation_input_tokens": 300})
    result = run_stream(message_start(start) + delta + MESSAGE_STOP)
    snapshot = result.snapshot
    assert snapshot is not None
    assert snapshot.cache_breakdown is CacheBreakdown.MISSING
    assert snapshot.counts.cache_write_5m is None
    assert snapshot.counts.cache_write_1h is None


def test_repeated_aggregate_with_a_partial_split_is_not_filled_in() -> None:
    delta = message_delta(
        {
            "output_tokens": 42,
            "cache_creation_input_tokens": 300,
            "cache_creation": {"ephemeral_5m_input_tokens": 100},
        }
    )
    result = run_stream(message_start() + delta + MESSAGE_STOP)
    assert result.snapshot is not None
    assert result.snapshot.cache_breakdown is CacheBreakdown.MISSING


def test_fresh_split_with_a_changed_aggregate_replaces_the_old_one() -> None:
    delta = message_delta(
        {
            "output_tokens": 42,
            "cache_creation_input_tokens": 500,
            "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 500},
        }
    )
    result = run_stream(message_start() + delta + MESSAGE_STOP)
    assert result.snapshot is not None
    assert result.snapshot.cache_breakdown is CacheBreakdown.OK
    assert result.snapshot.counts.cache_write_5m == 0
    assert result.snapshot.counts.cache_write_1h == 500


def test_zero_aggregate_without_split_maps_to_zero_writes() -> None:
    usage = {
        "input_tokens": 25,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
        "output_tokens": 1,
    }
    result = run_stream(message_start(usage) + message_delta() + MESSAGE_STOP)
    assert result.snapshot is not None
    assert result.snapshot.cache_breakdown is CacheBreakdown.OK
    assert result.snapshot.counts.cache_write == 0
    assert result.snapshot.counts.cache_write_5m == 0
    assert result.snapshot.counts.cache_write_1h == 0


def test_zero_aggregate_with_nonzero_split_is_inconsistent() -> None:
    usage = {**START_USAGE, "cache_creation_input_tokens": 0}
    result = run_stream(message_start(usage) + message_delta() + MESSAGE_STOP)
    assert result.snapshot is not None
    assert result.snapshot.cache_breakdown is CacheBreakdown.INCONSISTENT


def test_split_that_does_not_sum_to_the_aggregate_is_inconsistent() -> None:
    usage = {
        **START_USAGE,
        "cache_creation": {"ephemeral_5m_input_tokens": 100, "ephemeral_1h_input_tokens": 100},
    }
    result = run_stream(message_start(usage) + message_delta() + MESSAGE_STOP)
    snapshot = result.snapshot
    assert snapshot is not None
    assert snapshot.cache_breakdown is CacheBreakdown.INCONSISTENT
    assert snapshot.counts.cache_write == 300
    assert snapshot.counts.cache_write_5m is None
    assert snapshot.counts.cache_write_1h is None


@pytest.mark.parametrize("bad", [-1, "100", 1.5, True])
def test_invalid_split_counter_is_inconsistent(bad: object) -> None:
    usage = {
        **START_USAGE,
        "cache_creation": {"ephemeral_5m_input_tokens": bad, "ephemeral_1h_input_tokens": 200},
    }
    result = run_stream(message_start(usage) + message_delta() + MESSAGE_STOP)
    assert result.snapshot is not None
    assert result.snapshot.cache_breakdown is CacheBreakdown.INCONSISTENT


@pytest.mark.parametrize(
    "split",
    [
        {"ephemeral_1h_input_tokens": 200},
        {"ephemeral_5m_input_tokens": None, "ephemeral_1h_input_tokens": 200},
        {"ephemeral_5m_input_tokens": 100},
        {},
    ],
)
def test_absent_duration_counter_is_missing_breakdown(split: dict[str, Any]) -> None:
    usage = {**START_USAGE, "cache_creation": split}
    result = run_stream(message_start(usage) + message_delta() + MESSAGE_STOP)
    snapshot = result.snapshot
    assert snapshot is not None
    assert snapshot.cache_breakdown is CacheBreakdown.MISSING
    assert snapshot.counts.cache_write == 300
    # The aggregate stands in once for the split; a lone duration is not kept beside it.
    assert snapshot.counts.cache_write_5m is None
    assert snapshot.counts.cache_write_1h is None


def test_absent_duration_counter_keeps_a_reported_token_total() -> None:
    usage = {**START_USAGE, "cache_creation": {"ephemeral_1h_input_tokens": 200}}
    result = run_stream(message_start(usage) + message_delta() + MESSAGE_STOP)
    assert result.usage_complete is True
    assert observed_total(result.snapshot) == (25 + 42 + 1_000 + 300, True)


def test_lone_duration_counter_above_the_aggregate_is_inconsistent() -> None:
    usage = {**START_USAGE, "cache_creation": {"ephemeral_1h_input_tokens": 301}}
    result = run_stream(message_start(usage) + message_delta() + MESSAGE_STOP)
    assert result.snapshot is not None
    assert result.snapshot.cache_breakdown is CacheBreakdown.INCONSISTENT


def test_zero_aggregate_with_partial_zero_split_maps_to_zero_writes() -> None:
    usage = {
        **START_USAGE,
        "cache_creation_input_tokens": 0,
        "cache_creation": {"ephemeral_5m_input_tokens": 0},
    }
    result = run_stream(message_start(usage) + message_delta() + MESSAGE_STOP)
    snapshot = result.snapshot
    assert snapshot is not None
    assert snapshot.cache_breakdown is CacheBreakdown.OK
    assert snapshot.counts.cache_write_5m == 0
    assert snapshot.counts.cache_write_1h == 0


def test_negative_counter_is_rejected_not_recorded() -> None:
    usage = {**START_USAGE, "input_tokens": -5}
    result = run_stream(message_start(usage) + message_delta() + MESSAGE_STOP)
    assert result.usage_complete is False
    assert result.usage_source is UsageSource.PARSE_ERROR


def test_priority_service_tier_in_usage_is_flagged_unsupported() -> None:
    usage = {**START_USAGE, "service_tier": "priority"}
    result = run_stream(message_start(usage) + message_delta() + MESSAGE_STOP)
    assert result.snapshot is not None
    assert result.snapshot.unsupported == ("service_tier_priority",)


def test_server_tool_use_in_usage_is_flagged_unsupported() -> None:
    delta = message_delta({"output_tokens": 42, "server_tool_use": {"web_search_requests": 2}})
    result = run_stream(message_start() + delta + MESSAGE_STOP)
    assert result.snapshot is not None
    assert "server_tool_use" in result.snapshot.unsupported


@pytest.mark.parametrize(
    ("selection", "reason"),
    [
        ({"speed": "fast"}, "speed_fast"),
        ({"inference_geo": "us"}, "inference_geo_us"),
        ({"speed": ["fast"]}, "speed"),
        ({"inference_geo": {"region": "us"}}, "inference_geo"),
    ],
)
def test_premium_speed_or_geography_in_usage_is_flagged_unsupported(
    selection: dict[str, Any], reason: str
) -> None:
    result = run_stream(
        message_start({**START_USAGE, **selection}) + message_delta() + MESSAGE_STOP
    )
    assert result.snapshot is not None
    assert result.snapshot.unsupported == (reason,)


@pytest.mark.parametrize(
    "selection",
    [
        {"speed": "standard"},
        {"inference_geo": "global"},
        {"inference_geo": "not_available"},
        {"speed": None, "inference_geo": None},
    ],
)
def test_standard_speed_and_geography_in_usage_are_priced(selection: dict[str, Any]) -> None:
    result = run_stream(
        message_start({**START_USAGE, **selection}) + message_delta() + MESSAGE_STOP
    )
    assert result.snapshot is not None
    assert result.snapshot.unsupported == ()


@pytest.mark.parametrize("selection", [{"speed": "fast"}, {"inference_geo": "us"}])
def test_complete_premium_usage_is_unsupported_but_keeps_reported_tokens(
    selection: dict[str, Any],
) -> None:
    # A response-side pricing condition must not be priced at the standard rate;
    # the relay keeps the reported token usage for the debit.
    price = PriceEntry(
        "d",
        "m",
        PricingMode.METERED,
        Decimal(3),
        Decimal(15),
        Decimal("0.3"),
        Decimal("3.75"),
        Decimal(6),
    )
    body = json.loads(NON_STREAM_BODY)
    body["usage"] = {**body["usage"], **selection}
    result = run_json(json.dumps(body).encode())
    assert result.usage_complete is True
    assert result.snapshot is not None
    assert result.snapshot.counts == EXPECTED_COUNTS
    priced = price_usage(result.snapshot, price)
    assert priced.status is PricingStatus.UNSUPPORTED
    assert priced.cost_micros is None


# --- non-stream responses ----------------------------------------------------------------


def test_non_stream_body_complete() -> None:
    result = run_json(NON_STREAM_BODY)
    assert result.usage_complete is True
    assert result.usage_source is UsageSource.OK
    assert result.streamed is False
    assert result.stream_complete is None
    assert result.upstream_model == "claude-sonnet-5-5"
    assert result.snapshot is not None
    assert result.snapshot.counts == EXPECTED_COUNTS


def test_non_stream_body_split_anywhere_gives_the_same_result() -> None:
    expected = run_json(NON_STREAM_BODY)
    for cut in range(1, len(NON_STREAM_BODY)):
        assert run_json(NON_STREAM_BODY, chunks=[cut]) == expected


def test_truncated_non_stream_body_without_eof_is_missing() -> None:
    result = run_json(NON_STREAM_BODY[:-10], eof=False)
    assert result.usage_complete is False
    assert result.usage_source is UsageSource.MISSING
    assert result.snapshot is None


def test_truncated_non_stream_body_at_eof_is_not_complete() -> None:
    result = run_json(NON_STREAM_BODY[:-10], eof=True)
    assert result.usage_complete is False
    assert result.usage_source is UsageSource.PARSE_ERROR


def test_non_stream_body_over_the_buffer_limit_is_too_large() -> None:
    result = run_json(NON_STREAM_BODY, max_buffer=64)
    assert result.usage_source is UsageSource.TOO_LARGE
    assert result.usage_complete is False
    assert "too_large" in result.reasons


def test_stream_buffer_limit_applies_to_an_unterminated_line_only() -> None:
    # Complete lines are parsed and dropped, so a long stream never hits the limit ...
    assert run_stream(COMPLETE_STREAM, max_buffer=512).usage_source is UsageSource.OK
    # ... but a single line that keeps growing past it stops buffering.
    runaway = message_start() + "data: " + "x" * 600
    result = run_stream(runaway, max_buffer=512, chunks=[len(message_start()) + 10])
    assert result.usage_source is UsageSource.TOO_LARGE
    assert result.usage_complete is False


def test_non_stream_error_body_records_the_error_type() -> None:
    body = b'{"type":"error","error":{"type":"authentication_error","message":"invalid x-api-key"}}'
    result = run_json(body, status=401)
    assert result.error_type == "authentication_error"
    assert result.snapshot is None
    assert result.usage_complete is False


# --- upstream headers --------------------------------------------------------------------

CLIENT_HEADERS = [
    ("Anthropic-Version", "2023-06-01"),
    ("anthropic-beta", "prompt-caching-2024-07-31"),
    ("Content-Type", "application/json"),
    ("x-api-key", "gk_CLIENTVIRTUALKEY"),
    ("Authorization", "Bearer gk_CLIENTBEARER"),
    ("Cookie", "session=abc"),
    ("X-Forwarded-For", "10.0.0.1"),
    ("x-session-id", "session-1"),
    ("accept-encoding", "gzip, br"),
]


def test_upstream_headers_forward_only_the_documented_headers() -> None:
    headers = FORMAT.upstream_headers(CLIENT_HEADERS, "provider-secret")
    assert headers == {
        "anthropic-version": "2023-06-01",
        "anthropic-beta": "prompt-caching-2024-07-31",
        "content-type": "application/json",
        "x-api-key": "provider-secret",
        "accept-encoding": "identity",  # set by the gate so relayed bytes stay unencoded
    }


def test_client_credentials_are_never_forwarded() -> None:
    headers = FORMAT.upstream_headers(CLIENT_HEADERS, "provider-secret")
    assert "authorization" not in {name.lower() for name in headers}
    values = " ".join(headers.values())
    assert "gk_CLIENTVIRTUALKEY" not in values
    assert "gk_CLIENTBEARER" not in values


def test_deployment_key_is_set_even_without_a_client_key() -> None:
    headers = FORMAT.upstream_headers([("authorization", "Bearer gk_X")], "provider-secret")
    assert headers["x-api-key"] == "provider-secret"
    assert "authorization" not in headers


def test_error_body_uses_the_anthropic_shape_with_a_gateway_code() -> None:
    body = json.loads(FORMAT.error_body("permission_error", "budget exhausted", "budget_exceeded"))
    assert body == {
        "type": "error",
        "error": {"type": "permission_error", "message": "budget exhausted"},
        "gateway_code": "budget_exceeded",
    }


# --- pricing features checked before forwarding -------------------------------------------


def features(tree: dict[str, Any], headers: list[tuple[str, str]] | None = None) -> list[str]:
    return FORMAT.unsupported_pricing_features(
        {"model": "m", "max_tokens": 1, **tree}, headers or []
    )


def test_plain_request_has_no_unsupported_features() -> None:
    assert features({}) == []


@pytest.mark.parametrize("tier", ["auto", "standard_only"])
def test_standard_service_tiers_are_allowed(tier: str) -> None:
    assert features({"service_tier": tier}) == []


def test_priority_service_tier_is_rejected() -> None:
    assert features({"service_tier": "priority"}) == ["service_tier"]


def test_server_tools_are_rejected() -> None:
    tools = [{"type": "web_search_20250305", "name": "web_search", "max_uses": 3}]
    assert features({"tools": tools}) == ["server_tool"]


@pytest.mark.parametrize(
    "tool",
    [
        {"name": "get_weather", "description": "d", "input_schema": {"type": "object"}},
        {"type": "custom", "name": "lookup", "input_schema": {"type": "object"}},
        {"type": "bash_20250124", "name": "bash"},
        {"type": "text_editor_20250124", "name": "str_replace_editor"},
    ],
)
def test_client_executed_tools_are_allowed(tool: dict[str, Any]) -> None:
    assert features({"tools": [tool]}) == []


def test_long_context_beta_header_is_rejected() -> None:
    headers = [("anthropic-beta", "prompt-caching-2024-07-31, context-1m-2025-08-07")]
    assert features({}, headers) == ["long_context_beta"]


def test_ordinary_beta_header_is_allowed() -> None:
    assert features({}, [("anthropic-beta", "prompt-caching-2024-07-31")]) == []


def test_several_unsupported_features_are_all_reported() -> None:
    found = features(
        {"service_tier": "priority", "tools": [{"type": "web_fetch_20250910", "name": "f"}]},
        [("Anthropic-Beta", "context-1m-2025-08-07")],
    )
    assert found == ["service_tier", "server_tool", "long_context_beta"]


# --- documented non-billable rejections -----------------------------------------------------


def error_result(status: int, error_type: str) -> ParseResult:
    body = json.dumps({"type": "error", "error": {"type": error_type, "message": "x"}}).encode()
    return run_json(body, status=status)


@pytest.mark.parametrize(
    ("status", "error_type", "basis"),
    [
        (401, "authentication_error", "provider_auth_rejected"),
        (403, "permission_error", "provider_permission_rejected"),
        (429, "rate_limit_error", "provider_rate_limited"),
    ],
)
def test_listed_rejections_are_non_billable(status: int, error_type: str, basis: str) -> None:
    assert FORMAT.non_billable_basis(status, error_result(status, error_type)) == basis


@pytest.mark.parametrize(
    ("status", "error_type"),
    [
        (400, "invalid_request_error"),
        (413, "request_too_large"),
        (500, "api_error"),
        (529, "overloaded_error"),
        (401, "permission_error"),  # status and type must match together
        (403, "authentication_error"),
        (429, "overloaded_error"),
    ],
)
def test_other_rejections_are_not_proven_free(status: int, error_type: str) -> None:
    assert FORMAT.non_billable_basis(status, error_result(status, error_type)) is None


def test_rejection_without_a_parseable_error_type_is_not_free() -> None:
    result = run_json(b"<html>Unauthorized</html>", status=401)
    assert FORMAT.non_billable_basis(401, result) is None


def test_usage_evidence_overrides_a_listed_rejection() -> None:
    body = json.dumps(
        {
            "type": "error",
            "error": {"type": "rate_limit_error", "message": "x"},
            "usage": {"input_tokens": 10, "output_tokens": 0},
        }
    ).encode()
    result = run_json(body, status=429)
    assert result.snapshot is not None
    assert FORMAT.non_billable_basis(429, result) is None


@pytest.mark.parametrize(
    "usage",
    [
        {"input_tokens": 100, "output_tokens": "bad"},  # one valid counter, one invalid
        {"input_tokens": -1},
        {"cache_creation_input_tokens": 1.5},
        "garbage",
        None,
    ],
)
def test_malformed_usage_still_blocks_a_listed_rejection(usage: object) -> None:
    body = json.dumps(
        {
            "type": "error",
            "error": {"type": "rate_limit_error", "message": "x"},
            "usage": usage,
        }
    ).encode()
    result = run_json(body, status=429)
    assert result.snapshot is None  # nothing validated ...
    assert result.execution_evidence  # ... but execution is still uncertain
    assert FORMAT.non_billable_basis(429, result) is None


def test_listed_rejection_without_usage_has_no_execution_evidence() -> None:
    assert not error_result(429, "rate_limit_error").execution_evidence


def test_a_started_stream_is_never_a_pre_execution_rejection() -> None:
    result = run_stream(
        sse("error", {"type": "error", "error": {"type": "rate_limit_error", "message": "x"}}),
        status=429,
    )
    assert result.error_type == "rate_limit_error"
    assert FORMAT.non_billable_basis(429, result) is None

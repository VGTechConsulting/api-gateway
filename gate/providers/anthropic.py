"""Anthropic Messages format: headers, errors, usage parsing, completeness.

DeepSeek's Anthropic-compatible endpoint uses this same adapter.

The parser is incremental and works on the bytes the relay has already sent
to the client; it never changes them. Each usage update is treated as a
snapshot: a newer cache-write aggregate is never combined with stale duration
counters.
"""

import json
from dataclasses import dataclass, replace
from typing import Any

from gate.domain.errors import InvalidRequest
from gate.domain.models import ApiFormat, CacheBreakdown, TokenCounts, UsageSnapshot, UsageSource
from gate.providers.base import Headers, ParseResult

FORWARDED_HEADERS = ("anthropic-version", "anthropic-beta", "content-type")
# RFC 9110 field-value characters without obs-text: what the upstream client can encode.
_HEADER_VALUE_CHARS = frozenset(chr(c) for c in range(0x20, 0x7F)) | {"\t"}

# Request fields and header values that select pricing this version cannot price.
STANDARD_SERVICE_TIERS = {"auto", "standard_only"}
SERVER_TOOL_PREFIXES = ("web_search_", "web_fetch_", "code_execution_")
LONG_CONTEXT_BETA_PREFIX = "context-1m"
# Response usage fields that report a pricing selection, and the values billed at the
# standard rate. Any other value (priority tier, fast mode, US-only inference, ...) is a
# pricing condition this version cannot price. "not_available" is what models
# without geography selection report; it carries no premium.
STANDARD_USAGE_SELECTIONS: dict[str, frozenset[str]] = {
    "service_tier": frozenset({"standard"}),
    "speed": frozenset({"standard"}),
    "inference_geo": frozenset({"global", "not_available"}),
}


@dataclass(frozen=True)
class RejectionRule:
    """A documented pre-execution rejection that is known not to be billed."""

    http_status: int
    error_type: str
    basis: str
    source: str


# VERIFY before reconciling costs: each rule must cite provider documentation stating that the
# rejection happens before execution and is not billed. Unrecognised rejections
# fall through to ``unknown`` and are debited at the reservation.
REJECTION_RULES: tuple[RejectionRule, ...] = (
    RejectionRule(
        401,
        "authentication_error",
        "provider_auth_rejected",
        "https://docs.anthropic.com/en/api/errors",
    ),
    RejectionRule(
        403,
        "permission_error",
        "provider_permission_rejected",
        "https://docs.anthropic.com/en/api/errors",
    ),
    RejectionRule(
        429,
        "rate_limit_error",
        "provider_rate_limited",
        "https://docs.anthropic.com/en/api/rate-limits",
    ),
)


# Decoding untrusted JSON: malformed text raises ValueError (UnicodeDecodeError is one),
# and nesting deeper than the interpreter's recursion limit raises RecursionError.
_JSON_FAILURES = (ValueError, RecursionError)


def _count(value: object) -> int | None:
    """A valid token counter, or None if the value is not a non-negative integer."""
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


class _UsageState:
    """Latest valid cumulative usage, merged update by update."""

    def __init__(self) -> None:
        self.counts = TokenCounts()
        self.breakdown = CacheBreakdown.OK
        self.seen = False
        self.unsupported: set[str] = set()

    def apply(self, usage: object) -> bool:
        """Merge one usage object. Returns False (state unchanged) if it is malformed."""
        if not isinstance(usage, dict):
            return False
        fields = {
            "input": "input_tokens",
            "output": "output_tokens",
            "cache_read": "cache_read_input_tokens",
        }
        updates: dict[str, int] = {}
        for attr, name in fields.items():
            raw = usage.get(name)
            if raw is None:
                continue
            value = _count(raw)
            if value is None:
                return False
            updates[attr] = value

        counts = replace(self.counts, **updates)
        breakdown = self.breakdown
        raw_aggregate = usage.get("cache_creation_input_tokens")
        if raw_aggregate is not None:
            aggregate = _count(raw_aggregate)
            if aggregate is None:
                return False
            counts, breakdown = self._cache_writes(counts, aggregate, usage.get("cache_creation"))

        self._note_unsupported(usage)
        self.counts, self.breakdown, self.seen = counts, breakdown, True
        return True

    def _cache_writes(
        self, counts: TokenCounts, aggregate: int, split: object
    ) -> tuple[TokenCounts, CacheBreakdown]:
        # Absent (missing or null) duration counters mean the split is missing;
        # present but invalid or conflicting counters make it inconsistent.
        raw = split if isinstance(split, dict) else {}
        durations = [raw.get("ephemeral_5m_input_tokens"), raw.get("ephemeral_1h_input_tokens")]
        present = [d for d in durations if d is not None]
        valid = [v for v in map(_count, present) if v is not None]
        unknown = replace(counts, cache_write=aggregate, cache_write_5m=None, cache_write_1h=None)
        if len(valid) < len(present) or sum(valid) > aggregate:
            return unknown, CacheBreakdown.INCONSISTENT
        if len(valid) == 2:
            m5, h1 = valid
            if m5 + h1 != aggregate:
                return unknown, CacheBreakdown.INCONSISTENT
            return replace(
                counts, cache_write=aggregate, cache_write_5m=m5, cache_write_1h=h1
            ), CacheBreakdown.OK
        if aggregate == 0:
            return replace(
                counts, cache_write=0, cache_write_5m=0, cache_write_1h=0
            ), CacheBreakdown.OK
        # Anthropic's final ``message_delta`` repeats the cumulative aggregate without
        # the ``cache_creation`` object that ``message_start`` carried. When this update
        # has no duration counters at all and its aggregate equals the one a valid
        # split already accounts for, that split still describes it: nothing is stale.
        previous = self.counts
        if (
            not present
            and self.breakdown is CacheBreakdown.OK
            and previous.cache_write == aggregate
            and previous.cache_write_5m is not None
            and previous.cache_write_1h is not None
        ):
            return replace(
                counts,
                cache_write=aggregate,
                cache_write_5m=previous.cache_write_5m,
                cache_write_1h=previous.cache_write_1h,
            ), CacheBreakdown.OK
        # Otherwise a positive aggregate without both of its own duration counters is
        # missing a breakdown: a changed aggregate is never combined with an older
        # split. Both counters stay NULL so the aggregate is counted once, as
        # unclassified, rather than alongside a partial split.
        return unknown, CacheBreakdown.MISSING

    def _note_unsupported(self, usage: dict[str, Any]) -> None:
        for field, standard in STANDARD_USAGE_SELECTIONS.items():
            value = usage.get(field)
            if value is None or (isinstance(value, str) and value in standard):
                continue
            # A non-string (possibly unhashable) value is still an unknown selection.
            self.unsupported.add(f"{field}_{value}" if isinstance(value, str) else field)
        server_tools = usage.get("server_tool_use")
        if isinstance(server_tools, dict) and any(
            (_count(v) or 0) > 0 for v in server_tools.values()
        ):
            self.unsupported.add("server_tool_use")

    def snapshot(self) -> UsageSnapshot | None:
        if not self.seen:
            return None
        return UsageSnapshot(self.counts, self.breakdown, tuple(sorted(self.unsupported)))


class AnthropicUsageParser:
    """Incremental parser for one Anthropic-format response, streaming or not."""

    def __init__(self, *, streamed: bool, http_status: int, max_buffer_bytes: int) -> None:
        self._streamed = streamed
        self._status = http_status
        self._max = max_buffer_bytes
        self._buffer = bytearray()
        self._usage = _UsageState()
        self._model: str | None = None
        self._error_type: str | None = None
        self._usage_present = False  # any usage field, even one that fails validation
        self._initial_valid = False
        self._final_delta = False
        self._stopped = False
        self._stream_error = False
        self._parse_error = False
        self._too_large = False

    # -- feeding --------------------------------------------------------------------

    def feed(self, chunk: bytes) -> None:
        if self._too_large:
            return
        self._buffer += chunk
        if self._streamed:
            self._drain_lines()
        if len(self._buffer) > self._max:
            self._too_large = True
            self._buffer.clear()

    def _drain_lines(self) -> None:
        while True:
            newline = self._buffer.find(b"\n")
            if newline < 0:
                return
            line = bytes(self._buffer[:newline]).rstrip(b"\r")
            del self._buffer[: newline + 1]
            self._line(line)

    def _line(self, line: bytes) -> None:
        if not line.startswith(b"data:"):
            return  # event:, id:, comments and blank separators carry no usage
        payload = line[5:]
        if payload.startswith(b" "):
            payload = payload[1:]
        try:
            event = json.loads(payload)
        except _JSON_FAILURES:
            self._parse_error = True
            return
        if not isinstance(event, dict):
            self._parse_error = True
            return
        self._event(event)

    def _event(self, event: dict[str, Any]) -> None:
        kind = event.get("type")
        if kind == "message_start":
            message = event.get("message")
            if not isinstance(message, dict):
                self._parse_error = True
                return
            self._note_model(message.get("model"))
            usage = message.get("usage")
            if self._usage.apply(usage) and isinstance(usage, dict):
                self._initial_valid = _count(usage.get("input_tokens")) is not None
            else:
                self._parse_error = True
        elif kind == "message_delta":
            usage = event.get("usage")
            if self._usage.apply(usage) and isinstance(usage, dict):
                if _count(usage.get("output_tokens")) is not None:
                    self._final_delta = True
            elif usage is not None:
                self._parse_error = True
        elif kind == "message_stop":
            self._stopped = True
        elif kind == "error":
            self._stream_error = True
            self._note_error(event)

    def _note_model(self, model: object) -> None:
        if isinstance(model, str) and self._model is None:
            self._model = model

    def _note_error(self, body: dict[str, Any]) -> None:
        error = body.get("error")
        if isinstance(error, dict) and isinstance(error.get("type"), str):
            self._error_type = error["type"]

    # -- result ---------------------------------------------------------------------

    def result(self, *, eof: bool) -> ParseResult:
        if self._streamed:
            return self._stream_result()
        return self._json_result(eof)

    def _stream_result(self) -> ParseResult:
        snapshot = self._usage.snapshot()
        complete = (
            self._initial_valid
            and self._final_delta
            and self._stopped
            and not self._stream_error
            and not self._parse_error
            and not self._too_large
            and 200 <= self._status < 300
        )
        return ParseResult(
            snapshot=snapshot,
            usage_complete=complete,
            usage_source=self._source(snapshot, complete),
            streamed=True,
            stream_complete=self._stopped,
            stream_error=self._stream_error,
            upstream_model=self._model,
            error_type=self._error_type,
            execution_evidence=True,  # a started stream is evidence of execution
            reasons=self._reasons(),
        )

    def _json_result(self, eof: bool) -> ParseResult:
        complete = False
        if eof and not self._too_large and self._buffer:
            try:
                body = json.loads(bytes(self._buffer))
            except _JSON_FAILURES:
                body = None
                if 200 <= self._status < 300:
                    self._parse_error = True
            if isinstance(body, dict):
                if body.get("type") == "error":
                    self._note_error(body)
                self._note_model(body.get("model"))
                self._usage_present = "usage" in body
                usage = body.get("usage")
                if usage is not None:
                    if self._usage.apply(usage) and isinstance(usage, dict):
                        complete = (
                            200 <= self._status < 300
                            and _count(usage.get("input_tokens")) is not None
                            and _count(usage.get("output_tokens")) is not None
                        )
                    else:
                        self._parse_error = True
        self._buffer.clear()
        snapshot = self._usage.snapshot()
        return ParseResult(
            snapshot=snapshot,
            usage_complete=complete,
            usage_source=self._source(snapshot, complete),
            streamed=False,
            stream_complete=None,
            stream_error=False,
            upstream_model=self._model,
            error_type=self._error_type,
            execution_evidence=self._usage_present or snapshot is not None,
            reasons=self._reasons(),
        )

    def _source(self, snapshot: UsageSnapshot | None, complete: bool) -> UsageSource:
        if self._too_large:
            return UsageSource.TOO_LARGE
        if self._parse_error:
            return UsageSource.PARSE_ERROR
        if complete:
            return UsageSource.OK
        return UsageSource.PARTIAL if snapshot is not None else UsageSource.MISSING

    def _reasons(self) -> tuple[str, ...]:
        reasons = []
        if self._too_large:
            reasons.append("too_large")
        if self._parse_error:
            reasons.append("parse_error")
        if self._stream_error:
            reasons.append("stream_error")
        return tuple(reasons)


class AnthropicFormat:
    format = ApiFormat.ANTHROPIC

    def upstream_headers(self, client_headers: Headers, api_key: str) -> dict[str, str]:
        """Forward only the documented headers; replace the client's credential."""
        forwarded: dict[str, list[str]] = {}
        for name, value in client_headers:
            lower = name.lower()
            if lower in FORWARDED_HEADERS:
                if not _HEADER_VALUE_CHARS.issuperset(value):
                    # Never echo the value: it is client-controlled.
                    raise InvalidRequest(
                        f"header {lower} contains characters that cannot be forwarded",
                        gateway_code="invalid_header",
                    )
                forwarded.setdefault(lower, []).append(value)
        headers = {name: ", ".join(values) for name, values in forwarded.items()}
        headers["x-api-key"] = api_key
        # Relay raw bytes unchanged: ask for no content encoding we would have to decode.
        headers["accept-encoding"] = "identity"
        return headers

    def error_body(self, error_type: str, message: str, gateway_code: str) -> bytes:
        body = {
            "type": "error",
            "error": {"type": error_type, "message": message},
            "gateway_code": gateway_code,
        }
        return json.dumps(body, separators=(",", ":")).encode()

    def new_parser(
        self, *, content_type: str, http_status: int, max_buffer_bytes: int
    ) -> AnthropicUsageParser:
        streamed = content_type.split(";", 1)[0].strip().lower() == "text/event-stream"
        return AnthropicUsageParser(
            streamed=streamed, http_status=http_status, max_buffer_bytes=max_buffer_bytes
        )

    def unsupported_pricing_features(self, tree: dict[str, Any], headers: Headers) -> list[str]:
        """Known request selections whose charges the v0.1 pricing contract cannot price."""
        found: list[str] = []
        tier = tree.get("service_tier")
        if tier is not None and tier not in STANDARD_SERVICE_TIERS:
            found.append("service_tier")
        speed = tree.get("speed")
        if speed is not None and speed != "standard":
            found.append("speed")
        geo = tree.get("inference_geo")
        if geo is not None and geo != "global":
            found.append("inference_geo")
        tools = tree.get("tools")
        if isinstance(tools, list):
            for tool in tools:
                tool_type = tool.get("type") if isinstance(tool, dict) else None
                if isinstance(tool_type, str) and tool_type.startswith(SERVER_TOOL_PREFIXES):
                    found.append("server_tool")
                    break
        for name, value in headers:
            if name.lower() == "anthropic-beta" and any(
                beta.strip().startswith(LONG_CONTEXT_BETA_PREFIX) for beta in value.split(",")
            ):
                found.append("long_context_beta")
                break
        return found

    def non_billable_basis(self, http_status: int, result: ParseResult) -> str | None:
        if result.execution_evidence:
            return None  # usage, even malformed, or a started stream: execution is uncertain
        for rule in REJECTION_RULES:
            if rule.http_status == http_status and rule.error_type == result.error_type:
                return rule.basis
        return None

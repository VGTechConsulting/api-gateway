"""The provider-format contract.

A provider format is pure: header policy, error shape, an incremental usage
parser, completeness rules, request pricing-feature checks and documented
non-billable rejection rules. It performs no I/O and reads no clock.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from gate.domain.models import ApiFormat, UsageSnapshot, UsageSource

Headers = Sequence[tuple[str, str]]


@dataclass(frozen=True)
class ParseResult:
    """What a usage parser established about one upstream response."""

    snapshot: UsageSnapshot | None
    usage_complete: bool
    usage_source: UsageSource
    streamed: bool
    stream_complete: bool | None  # None for non-stream responses
    stream_error: bool
    upstream_model: str | None
    error_type: str | None  # provider error type, from an error body or event
    # The response carried usage (valid or not) or started a stream: the request may
    # have executed, so it can never be a zero-cost pre-execution rejection.
    execution_evidence: bool
    reasons: tuple[str, ...] = ()  # diagnostics: parse_error, too_large, ...


class UsageParser(Protocol):
    def feed(self, chunk: bytes) -> None:
        """Consume relayed bytes. Never raises; failures become diagnostic status."""

    def result(self, *, eof: bool) -> ParseResult:
        """Final state. ``eof`` says whether the whole upstream body was read."""


class ProviderFormat(Protocol):
    format: ApiFormat

    def upstream_headers(self, client_headers: Headers, api_key: str) -> dict[str, str]: ...

    def error_body(self, error_type: str, message: str, gateway_code: str) -> bytes: ...

    def new_parser(
        self, *, content_type: str, http_status: int, max_buffer_bytes: int
    ) -> UsageParser: ...

    def unsupported_pricing_features(self, tree: dict[str, Any], headers: Headers) -> list[str]: ...

    def non_billable_basis(self, http_status: int, result: ParseResult) -> str | None: ...

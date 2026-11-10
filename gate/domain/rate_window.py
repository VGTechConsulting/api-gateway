"""Per-key RPM/TPM windows and Retry-After hints.

Time is monotonic seconds supplied by the caller. Entries live in
``(now - 60s, now]``: an entry stamped ``t`` expires exactly at ``t + 60``.
RPM entries are stamped at a request's first admission; token entries are
stamped when an attempt settles. Active token reservations live in the ledger
and never expire on their own.
"""

import math
from collections import deque
from dataclasses import dataclass

from gate.domain.errors import InvalidRequest, RateLimited
from gate.domain.models import Policy
from gate.domain.policy import Decision

WINDOW_S = 60.0
IN_FLIGHT_HINT_S = 1.0


class RateWindow:
    def __init__(self) -> None:
        self._requests: deque[float] = deque()
        self._tokens: deque[tuple[float, int]] = deque()
        self._token_sum = 0

    def expire(self, now: float) -> None:
        horizon = now - WINDOW_S
        while self._requests and self._requests[0] <= horizon:
            self._requests.popleft()
        while self._tokens and self._tokens[0][0] <= horizon:
            self._token_sum -= self._tokens.popleft()[1]

    @property
    def request_count(self) -> int:
        return len(self._requests)

    @property
    def settled_tokens(self) -> int:
        return self._token_sum

    def add_request(self, now: float) -> None:
        self._requests.append(now)

    def add_tokens(self, now: float, tokens: int) -> None:
        if tokens > 0:
            self._tokens.append((now, tokens))
            self._token_sum += tokens

    def rpm_delay(self, now: float, rpm: int) -> float:
        """Seconds until one more request fits under ``rpm``."""
        excess = len(self._requests) - rpm + 1
        if excess <= 0:
            return 0.0
        if excess > len(self._requests):  # rpm < 1: never fits (config forbids this)
            return WINDOW_S
        return self._requests[excess - 1] + WINDOW_S - now

    def tpm_delay(self, now: float, allowance: int) -> float | None:
        """Seconds until settled tokens drop to ``allowance``, or None if expiry alone can't."""
        if allowance < 0:
            return None
        remaining = self._token_sum
        for stamp, tokens in self._tokens:
            if remaining <= allowance:
                break
            remaining -= tokens
            if remaining <= allowance:
                return stamp + WINDOW_S - now
        return 0.0


@dataclass(frozen=True)
class RateLimits:
    rpm: int
    tpm: int


def retry_after_seconds(delay: float) -> int:
    return max(1, math.ceil(delay))


def check_rate(
    window: RateWindow,
    *,
    now: float,
    limits: RateLimits,
    reserved_tokens: int,
    estimate_tokens: int,
    count_request: bool,
    metered: bool,
) -> list[Decision]:
    """Evaluate RPM and TPM for one attempt. The window must already be expired to ``now``.

    Returns one decision per check, in order: estimate-exceeds-limit, RPM, TPM.
    """
    decisions: list[Decision] = []
    exceeds = metered and estimate_tokens > limits.tpm
    if exceeds:
        # Waiting cannot make this fit, so it is a 400 without Retry-After. RPM is still
        # evaluated so observe mode records every failing check.
        decisions.append(
            Decision(
                Policy.RATE_LIMITS,
                "token_estimate",
                InvalidRequest(
                    "the request's token estimate exceeds this key's TPM limit",
                    gateway_code="token_estimate_exceeds_limit",
                ),
                {"estimate_tokens": estimate_tokens, "tpm": limits.tpm},
            )
        )

    rpm_blocked = count_request and window.request_count >= limits.rpm
    tpm_used = window.settled_tokens + reserved_tokens
    tpm_blocked = metered and not exceeds and tpm_used + estimate_tokens > limits.tpm

    delays: list[float] = []
    estimated = False
    if rpm_blocked:
        delays.append(window.rpm_delay(now, limits.rpm))
    if tpm_blocked:
        allowance = limits.tpm - reserved_tokens - estimate_tokens
        delay = window.tpm_delay(now, allowance)
        if delay is None:
            # Expiry alone can't make room: some in-flight reservation must settle first, at
            # an unknown time. Even if every reservation settled with a zero debit, settled
            # tokens would still have to fall to ``tpm - estimate``; that expiry is known,
            # so combine it with the in-flight hint.
            estimated = True
            known = window.tpm_delay(now, limits.tpm - estimate_tokens) or 0.0
            delays.append(max(IN_FLIGHT_HINT_S, known))
        else:
            delays.append(delay)
    retry = retry_after_seconds(max(delays)) if delays else None
    detail: dict[str, str | int | bool | None] = (
        {"retry_after_s": retry, "retry_after_estimated": estimated} if delays else {}
    )

    def blocked(code: str, message: str) -> RateLimited:
        return RateLimited(message, gateway_code=code, retry_after_s=retry)

    return [
        *decisions,
        Decision(
            Policy.RATE_LIMITS,
            "rpm",
            blocked("rpm_exceeded", "request-per-minute limit reached") if rpm_blocked else None,
            detail if rpm_blocked else {},
        ),
        Decision(
            Policy.RATE_LIMITS,
            "tpm",
            blocked("tpm_exceeded", "token-per-minute capacity unavailable")
            if tpm_blocked
            else None,
            detail if tpm_blocked else {},
        ),
    ]

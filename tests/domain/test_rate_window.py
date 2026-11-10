"""Per-key RPM/TPM admission windows and Retry-After hints.

Time is an explicit monotonic float. Entries live in ``(now - 60s, now]``, so an
entry stamped ``t`` has expired at exactly ``t + 60``. Active token reservations
are passed in as ``reserved_tokens``: they never expire on their own.
"""

from gate.domain.errors import InvalidRequest, RateLimited
from gate.domain.models import Policy
from gate.domain.policy import Decision
from gate.domain.rate_window import RateLimits, RateWindow, check_rate, retry_after_seconds


def check(
    window: RateWindow,
    now: float,
    *,
    rpm: int = 1_000,
    tpm: int = 1_000_000,
    reserved: int = 0,
    estimate: int = 100,
    count_request: bool = True,
    metered: bool = True,
) -> dict[str, Decision]:
    window.expire(now)
    decisions = check_rate(
        window,
        now=now,
        limits=RateLimits(rpm=rpm, tpm=tpm),
        reserved_tokens=reserved,
        estimate_tokens=estimate,
        count_request=count_request,
        metered=metered,
    )
    assert all(d.policy is Policy.RATE_LIMITS for d in decisions)
    return {d.check: d for d in decisions}


# --- the window itself ------------------------------------------------------------------


def test_request_entry_expires_exactly_sixty_seconds_later() -> None:
    window = RateWindow()
    window.add_request(100.0)
    window.expire(159.999)
    assert window.request_count == 1
    window.expire(160.0)
    assert window.request_count == 0


def test_token_entry_expires_exactly_sixty_seconds_after_settlement() -> None:
    window = RateWindow()
    window.add_tokens(100.0, 500)
    window.add_tokens(130.0, 200)
    window.expire(159.999)
    assert window.settled_tokens == 700
    window.expire(160.0)
    assert window.settled_tokens == 200
    window.expire(190.0)
    assert window.settled_tokens == 0


def test_zero_token_debit_adds_no_entry() -> None:
    window = RateWindow()
    window.add_tokens(1.0, 0)
    assert window.settled_tokens == 0


def test_retry_after_rounds_up_with_a_one_second_minimum() -> None:
    assert retry_after_seconds(0.0) == 1
    assert retry_after_seconds(0.2) == 1
    assert retry_after_seconds(1.0) == 1
    assert retry_after_seconds(1.01) == 2
    assert retry_after_seconds(29.5) == 30


# --- RPM ---------------------------------------------------------------------------------


def test_rpm_allows_up_to_the_limit() -> None:
    window = RateWindow()
    window.add_request(0.0)
    decisions = check(window, 1.0, rpm=2)
    assert decisions["rpm"].passed
    assert decisions["tpm"].passed


def test_rpm_block_retry_after_is_the_earliest_expiry_that_makes_room() -> None:
    window = RateWindow()
    window.add_request(0.2)
    window.add_request(10.0)
    decisions = check(window, 30.0, rpm=2)
    rpm = decisions["rpm"]
    assert isinstance(rpm.error, RateLimited)
    assert rpm.error.status == 429
    assert rpm.error.gateway_code == "rpm_exceeded"
    # The 0.2 entry leaves at 60.2, i.e. 30.2 s from now, rounded up to 31.
    assert rpm.error.retry_after_s == 31
    assert rpm.detail["retry_after_estimated"] is False
    assert decisions["tpm"].passed


def test_rpm_retry_after_counts_only_the_entries_that_must_leave() -> None:
    # rpm=2 with three entries: two must expire before a fourth request fits.
    window = RateWindow()
    for t in (0.0, 5.0, 20.0):
        window.add_request(t)
    decisions = check(window, 30.0, rpm=2)
    assert decisions["rpm"].error is not None
    assert decisions["rpm"].error.retry_after_s == 35  # 5.0 + 60 - 30


def test_rpm_retry_after_has_a_one_second_minimum() -> None:
    window = RateWindow()
    window.add_request(0.0)
    decisions = check(window, 59.75, rpm=1)
    assert decisions["rpm"].error is not None
    assert decisions["rpm"].error.retry_after_s == 1


def test_rpm_frees_up_exactly_at_expiry() -> None:
    window = RateWindow()
    window.add_request(0.0)
    assert not check(window, 59.9, rpm=1)["rpm"].passed
    assert check(window, 60.0, rpm=1)["rpm"].passed


def test_fallback_admission_skips_the_rpm_check() -> None:
    window = RateWindow()
    window.add_request(0.0)
    decisions = check(window, 1.0, rpm=1, count_request=False)
    assert decisions["rpm"].passed


def test_unmetered_request_still_consumes_rpm() -> None:
    window = RateWindow()
    window.add_request(0.0)
    decisions = check(window, 1.0, rpm=1, metered=False, estimate=0)
    assert decisions["rpm"].error is not None
    assert decisions["rpm"].error.gateway_code == "rpm_exceeded"


# --- TPM ---------------------------------------------------------------------------------


def test_tpm_counts_settled_tokens_plus_reservations_plus_estimate() -> None:
    window = RateWindow()
    window.add_tokens(0.0, 400)
    # 400 settled + 300 reserved + 300 incoming = 1000: fits exactly.
    assert check(window, 1.0, tpm=1_000, reserved=300, estimate=300)["tpm"].passed
    # One more token does not.
    assert not check(window, 1.0, tpm=1_000, reserved=300, estimate=301)["tpm"].passed


def test_tpm_block_retry_after_is_the_earliest_expiry_that_makes_it_fit() -> None:
    window = RateWindow()
    window.add_tokens(0.0, 600)
    window.add_tokens(10.0, 300)
    decisions = check(window, 20.0, tpm=1_000, estimate=200)
    tpm = decisions["tpm"]
    assert isinstance(tpm.error, RateLimited)
    assert tpm.error.gateway_code == "tpm_exceeded"
    # Only the 600-token entry (expiring at 60) must leave: 40 s from now.
    assert tpm.error.retry_after_s == 40
    assert tpm.detail["retry_after_estimated"] is False


def test_tpm_retry_after_can_need_several_expiries() -> None:
    window = RateWindow()
    window.add_tokens(0.0, 300)
    window.add_tokens(10.0, 300)
    window.add_tokens(15.0, 300)
    # 900 settled + 500 incoming > 1000: 400 must leave, so the first two entries.
    decisions = check(window, 20.0, tpm=1_000, estimate=500)
    assert decisions["tpm"].error is not None
    assert decisions["tpm"].error.retry_after_s == 50  # 10.0 + 60 - 20


def test_tpm_retry_after_ignores_reservations_when_expiry_alone_suffices() -> None:
    window = RateWindow()
    window.add_tokens(0.0, 600)
    decisions = check(window, 30.0, tpm=1_000, reserved=300, estimate=200)
    assert decisions["tpm"].error is not None
    assert decisions["tpm"].error.retry_after_s == 30
    assert decisions["tpm"].detail["retry_after_estimated"] is False


def test_in_flight_reservation_gives_one_second_estimated_hint() -> None:
    # Capacity depends on a reservation whose settlement time is unknown.
    window = RateWindow()
    decisions = check(window, 5.0, tpm=1_000, reserved=700, estimate=400)
    tpm = decisions["tpm"]
    assert tpm.error is not None
    assert tpm.error.gateway_code == "tpm_exceeded"
    assert tpm.error.retry_after_s == 1
    assert tpm.detail["retry_after_estimated"] is True


def test_in_flight_hint_combines_with_a_known_settled_token_expiry() -> None:
    # Reservations block, but settled tokens alone also block until they expire at 60s:
    # releasing every reservation still leaves 100 + 50 > 100 until then.
    window = RateWindow()
    window.add_tokens(0.0, 100)
    decisions = check(window, 0.0, tpm=100, reserved=100, estimate=50)
    tpm = decisions["tpm"]
    assert tpm.error is not None
    assert tpm.error.retry_after_s == 60
    assert tpm.detail["retry_after_s"] == 60
    assert tpm.detail["retry_after_estimated"] is True


def test_in_flight_hint_uses_only_the_expiries_needed_without_reservations() -> None:
    # Settled 900 must only drop to tpm - estimate = 600, so the first entry (expires at 70)
    # suffices; the second (expires at 80) need not leave.
    window = RateWindow()
    window.add_tokens(10.0, 500)
    window.add_tokens(20.0, 400)
    decisions = check(window, 30.0, tpm=1_000, reserved=700, estimate=400)
    tpm = decisions["tpm"]
    assert tpm.error is not None
    assert tpm.error.retry_after_s == 40
    assert tpm.detail["retry_after_estimated"] is True


def test_in_flight_hint_combines_with_a_longer_known_rpm_delay() -> None:
    window = RateWindow()
    window.add_request(0.0)
    decisions = check(window, 30.0, rpm=1, tpm=1_000, reserved=700, estimate=400)
    assert decisions["rpm"].error is not None
    assert decisions["tpm"].error is not None
    # Both checks block; the hint is the larger delay, and it is flagged as estimated.
    assert decisions["rpm"].error.retry_after_s == 30
    assert decisions["tpm"].error.retry_after_s == 30
    assert decisions["tpm"].detail["retry_after_estimated"] is True


def test_retry_after_considers_all_blocking_checks() -> None:
    window = RateWindow()
    window.add_request(0.0)  # RPM frees at 60
    window.add_tokens(25.0, 900)  # TPM frees at 85
    decisions = check(window, 30.0, rpm=1, tpm=1_000, estimate=200)
    assert decisions["rpm"].error is not None
    assert decisions["tpm"].error is not None
    assert decisions["rpm"].error.retry_after_s == 55
    assert decisions["tpm"].error.retry_after_s == 55


def test_reservations_never_expire_with_the_window() -> None:
    # A stream held open for more than 60 s keeps its reservation: the caller still
    # passes it in, and it still blocks.
    window = RateWindow()
    window.add_request(0.0)
    assert not check(window, 600.0, tpm=1_000, reserved=900, estimate=200)["tpm"].passed


def test_settled_debit_is_retained_for_sixty_seconds_after_late_settlement() -> None:
    window = RateWindow()
    window.add_tokens(300.0, 900)  # settled long after admission
    assert not check(window, 359.0, tpm=1_000, estimate=200)["tpm"].passed
    assert check(window, 360.0, tpm=1_000, estimate=200)["tpm"].passed


def test_estimate_larger_than_tpm_is_a_400_without_retry_after() -> None:
    window = RateWindow()
    decisions = check(window, 0.0, tpm=1_000, estimate=1_001)
    error = decisions["token_estimate"].error
    assert isinstance(error, InvalidRequest)
    assert error.status == 400
    assert error.error_type == "invalid_request_error"
    assert error.gateway_code == "token_estimate_exceeds_limit"
    assert error.retry_after_s is None


def test_estimate_equal_to_tpm_fits_an_empty_window() -> None:
    decisions = check(RateWindow(), 0.0, tpm=1_000, estimate=1_000)
    assert all(d.passed for d in decisions.values())


def test_estimate_larger_than_tpm_is_reported_for_fallback_attempts_too() -> None:
    decisions = check(RateWindow(), 0.0, tpm=1_000, estimate=5_000, count_request=False)
    error = decisions["token_estimate"].error
    assert error is not None
    assert error.gateway_code == "token_estimate_exceeds_limit"


def test_estimate_exceeding_tpm_does_not_hide_an_rpm_failure() -> None:
    window = RateWindow()
    window.add_request(0.0)
    decisions = check(window, 1.0, rpm=1, tpm=1_000, estimate=5_000)
    assert decisions["token_estimate"].error is not None
    assert "rpm" in decisions
    assert decisions["rpm"].error is not None


def test_unmetered_attempt_skips_tpm() -> None:
    window = RateWindow()
    window.add_tokens(0.0, 1_000)
    decisions = check(window, 1.0, tpm=1_000, reserved=1_000, estimate=0, metered=False)
    assert decisions["tpm"].passed
    assert "token_estimate" not in decisions


def test_passing_checks_carry_no_retry_detail() -> None:
    decisions = check(RateWindow(), 0.0)
    assert decisions["rpm"].passed
    assert decisions["tpm"].passed
    assert dict(decisions["rpm"].detail) == {}
    assert dict(decisions["tpm"].detail) == {}

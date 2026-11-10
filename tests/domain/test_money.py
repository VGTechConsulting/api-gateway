"""Money arithmetic in integer micro-dollars, rounded once to the nearest micro-dollar
(half up).

Rates are USD per million tokens, so ``tokens x rate`` is already a micro-dollar
amount: the only conversion step is a single half-up rounding at the end.
"""

from decimal import Decimal

import pytest

from gate.domain.money import ceil_div, micros_to_usd, round_half_up, token_cost, usd_to_micros


@pytest.mark.parametrize(
    ("amount", "expected"),
    [
        ("0", 0),
        ("0.4999", 0),
        ("0.5", 1),  # a half rounds up, not to even
        ("1.5", 2),
        ("2.5", 3),  # banker's rounding would give 2
        ("2.4999999", 2),
        ("1234567.5", 1234568),
    ],
)
def test_round_half_up(amount: str, expected: int) -> None:
    assert round_half_up(Decimal(amount)) == expected


def test_round_half_up_returns_an_int() -> None:
    assert type(round_half_up(Decimal("7.5"))) is int


def test_token_cost_is_unrounded_micro_dollars() -> None:
    # 1,000 tokens at $3 per million = $0.003 = 3,000 micro-dollars.
    assert token_cost(1_000, Decimal("3")) == Decimal(3_000)
    # 1 token at $0.30 per million is 0.3 micro-dollars: kept exact until rounding.
    assert token_cost(1, Decimal("0.30")) == Decimal("0.30")


def test_token_cost_avoids_binary_float_error() -> None:
    # 0.1 + 0.2 style errors must not appear: decimal arithmetic throughout.
    total = token_cost(1, Decimal("0.1")) + token_cost(1, Decimal("0.2"))
    assert total == Decimal("0.3")


def test_rounding_happens_once_on_the_sum() -> None:
    # Three categories of 0.4 micro-dollars each: rounding per category would give 0,
    # rounding the sum once gives 1.
    parts = [token_cost(1, Decimal("0.4")) for _ in range(3)]
    assert sum(round_half_up(p) for p in parts) == 0
    assert round_half_up(sum(parts, Decimal(0))) == 1


@pytest.mark.parametrize(
    ("usd", "micros"),
    [
        ("0", 0),
        ("1", 1_000_000),
        ("500", 500_000_000),
        ("0.000001", 1),
        ("0.0000005", 1),  # half a micro-dollar rounds up
        ("0.0000004", 0),
        ("12.3456785", 12_345_679),
    ],
)
def test_usd_to_micros(usd: str, micros: int) -> None:
    assert usd_to_micros(Decimal(usd)) == micros


def test_micros_to_usd_round_trips() -> None:
    assert micros_to_usd(12_345_679) == Decimal("12.345679")


@pytest.mark.parametrize(
    ("numerator", "denominator", "expected"),
    [
        (0, 4, 0),
        (1, 4, 1),
        (4, 4, 1),
        (5, 4, 2),
        (8, 4, 2),
        (9, 4, 3),
        (33_554_432, 4, 8_388_608),
        (2**53 + 1, 4, 2**51 + 1),  # beyond float precision, still exact
    ],
)
def test_ceil_div(numerator: int, denominator: int, expected: int) -> None:
    assert ceil_div(numerator, denominator) == expected

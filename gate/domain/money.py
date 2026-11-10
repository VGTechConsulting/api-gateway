"""Money arithmetic: integer micro-dollars, decimal rates, one half-up rounding.

Prices are configured in USD per million tokens. Because one USD is one million
micro-dollars, ``tokens x usd_per_million`` is already a micro-dollar amount, so
cost needs no unit conversion beyond a single rounding step.
"""

from decimal import ROUND_CEILING, ROUND_HALF_UP, Decimal
from typing import NewType

Micros = NewType("Micros", int)
Tokens = NewType("Tokens", int)

MICROS_PER_USD = 1_000_000


def round_half_up(amount: Decimal) -> Micros:
    """Round a micro-dollar amount once, to the nearest integer, halves away from zero."""
    return Micros(int(amount.quantize(Decimal(1), rounding=ROUND_HALF_UP)))


def token_cost(tokens: int, usd_per_million: Decimal) -> Decimal:
    """Unrounded micro-dollar cost of ``tokens`` at a USD-per-million rate."""
    return Decimal(tokens) * usd_per_million


def usd_to_micros(usd: Decimal) -> Micros:
    """Convert a configured USD amount (a budget) to micro-dollars."""
    return round_half_up(usd * MICROS_PER_USD)


def micros_to_usd(micros: int) -> Decimal:
    return Decimal(micros) / MICROS_PER_USD


def ceil_div(numerator: int, denominator: int) -> int:
    return int((Decimal(numerator) / Decimal(denominator)).to_integral_value(ROUND_CEILING))

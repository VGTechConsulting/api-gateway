"""Secret and card-number detectors over a decoded JSON tree.

Every string value and object key is inspected on its own, after one JSON
decoding pass. Strings are never joined, numbers are never coerced, and nothing
is recursively decoded. Results are per-category counts only: matched text,
excerpts and field paths are never returned.

Out of scope by design: base64 or image contents, nested JSON inside a string,
double-encoded text, and secrets split across separate strings.
"""

import re
from collections import Counter
from collections.abc import Callable, Iterator, Mapping
from typing import Any

PRIVATE_KEY = "private_key"
CLOUD_KEY = "cloud_key"
API_TOKEN = "api_token"
CARD_NUMBER = "card_number"

CATEGORIES = (PRIVATE_KEY, CLOUD_KEY, API_TOKEN, CARD_NUMBER)

_PRIVATE_KEY = re.compile(r"-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY-----")
_AWS_ACCESS_KEY = re.compile(r"(?<![A-Z0-9])(?:AKIA|ASIA|ABIA|ACCA)[A-Z0-9]{16}(?![A-Z0-9])")
_API_TOKENS = re.compile(
    r"(?<![A-Za-z0-9_-])(?:"
    r"sk-ant-[A-Za-z0-9_-]{20,}"  # Anthropic
    r"|sk-(?:proj-|svcacct-|admin-)?[A-Za-z0-9_-]{20,}"  # OpenAI
    r"|gh[pousr]_[A-Za-z0-9]{36,}"  # GitHub classic tokens
    r"|github_pat_[A-Za-z0-9_]{22,}"  # GitHub fine-grained tokens
    r")"
)
# A run of digit groups separated by single spaces or dashes; cards are 13-19 digits within it.
_DIGIT_RUN = re.compile(r"(?<!\d)\d+(?:[ -]\d+)*(?!\d)")
_MAX_CARD_DIGITS = 19


def luhn_valid(digits: str) -> bool:
    total = 0
    for index, char in enumerate(reversed(digits)):
        n = ord(char) - 48
        if index % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


def _count_cards(text: str) -> int:
    """Count card numbers, including ones adjacent to other numbers (``4111... 12/28``).

    Within each run of digit groups, take the longest sequence of whole groups that
    starts at the current group, has 13-19 digits and passes Luhn; otherwise move on
    one group. Matches never overlap.
    """
    count = 0
    for match in _DIGIT_RUN.finditer(text):
        groups = re.split(r"[ -]", match.group())
        start = 0
        while start < len(groups):
            end = _card_end(groups, start)
            if end is None:
                start += 1
            else:
                count += 1
                start = end
    return count


def _card_end(groups: list[str], start: int) -> int | None:
    limit = min(len(groups), start + _MAX_CARD_DIGITS)
    for end in range(limit, start, -1):
        digits = "".join(groups[start:end])
        if 13 <= len(digits) <= _MAX_CARD_DIGITS and luhn_valid(digits):
            return end
    return None


_DETECTORS: Mapping[str, Callable[[str], int]] = {
    PRIVATE_KEY: lambda s: len(_PRIVATE_KEY.findall(s)),
    CLOUD_KEY: lambda s: len(_AWS_ACCESS_KEY.findall(s)),
    API_TOKEN: lambda s: len(_API_TOKENS.findall(s)),
    CARD_NUMBER: _count_cards,
}


def iter_strings(tree: Any) -> Iterator[str]:
    """Yield every string value and object key; never coerce other scalars."""
    stack = [tree]
    while stack:
        node = stack.pop()
        if isinstance(node, str):
            yield node
        elif isinstance(node, dict):
            for key, value in node.items():
                yield key
                stack.append(value)
        elif isinstance(node, list):
            stack.extend(node)


def scan(tree: Any, categories: tuple[str, ...] = CATEGORIES) -> dict[str, int]:
    """Count occurrences per enabled category across all decoded strings."""
    counts: Counter[str] = Counter()
    detectors = [(name, _DETECTORS[name]) for name in categories]
    for text in iter_strings(tree):
        for name, detect in detectors:
            hits = detect(text)
            if hits:
                counts[name] += hits
    return dict(counts)

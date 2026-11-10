"""Secret and card-number detection over the decoded JSON tree.

Every string value and object key is scanned on its own, after exactly one JSON
decoding pass. Strings are never joined and numbers are never coerced. The last
section labels cases that are *outside* detection coverage by design: those tests
assert that nothing is detected, so nobody mistakes them for protection.

All examples below are synthetic. Token-shaped values are assembled from pieces so
the repository's secret scanner does not flag this file.
"""

import base64
import json
from typing import Any

import pytest

from gate.domain.detection import (
    API_TOKEN,
    CARD_NUMBER,
    CATEGORIES,
    CLOUD_KEY,
    PRIVATE_KEY,
    luhn_valid,
    scan,
)
from gate.domain.request import decode_json

# --- synthetic corpus --------------------------------------------------------------------

PEM_HEADER = "-----BEGIN " + "RSA PRIVATE KEY-----"
PEM_HEADER_PLAIN = "-----BEGIN " + "PRIVATE KEY-----"
PEM_HEADER_OPENSSH = "-----BEGIN " + "OPENSSH PRIVATE KEY-----"
AWS_KEY_ID = "AKIA" + "IOSFODNN7EXAMPLE"  # AWS's documented example key ID
ANTHROPIC_TOKEN = "sk-ant-" + "api03-" + "EXAMPLE0" * 5
OPENAI_TOKEN = "sk-" + "proj-" + "EXAMPLE0" * 4
GITHUB_TOKEN = "gh" + "p_" + "EXAMPLE0" * 5
GITHUB_PAT = "github" + "_pat_" + "EXAMPLE0" * 4
CARD = "4111111111111111"  # the standard Visa test number
CARD_SPACED = "4111 1111 1111 1111"
CARD_DASHED = "5500-0000-0000-0004"
CARD_AMEX = "378282246310005"  # 15 digits

SAMPLES = {
    PRIVATE_KEY: [PEM_HEADER, PEM_HEADER_PLAIN, PEM_HEADER_OPENSSH],
    CLOUD_KEY: [AWS_KEY_ID],
    API_TOKEN: [ANTHROPIC_TOKEN, OPENAI_TOKEN, GITHUB_TOKEN, GITHUB_PAT],
    CARD_NUMBER: [CARD, CARD_SPACED, CARD_DASHED, CARD_AMEX],
}


def scan_json(text: str) -> dict[str, int]:
    """Decode exactly as the gate does (one pass, duplicates rejected), then scan."""
    return scan(decode_json(text.encode("utf-8")))


def escape_all(value: str) -> str:
    """A JSON string literal with every character written as a \\uXXXX escape."""
    return '"' + "".join(f"\\u{ord(ch):04x}" for ch in value) + '"'


def literal(value: str) -> str:
    return json.dumps(value)


# --- each category -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("category", "sample"),
    [(category, sample) for category, samples in SAMPLES.items() for sample in samples],
)
def test_each_category_detects_its_samples(category: str, sample: str) -> None:
    assert scan({"text": f"before {sample} after"}) == {category: 1}


@pytest.mark.parametrize(
    "harmless",
    [
        "-----BEGIN PUBLIC KEY-----",
        "-----BEGIN CERTIFICATE-----",
        "AKIA" + "short",
        "akia" + "iosfodnn7example",  # lower case is not an AWS key ID
        "sk-" + "tooshort",
        "gh" + "p_" + "short",
        "4111111111111112",  # fails Luhn
        "411111111111",  # 12 digits: too short
        "41111111111111111111",  # 20 digits: too long
        "ordinary prose with numbers 12 and 2026",
    ],
)
def test_near_misses_are_not_detected(harmless: str) -> None:
    assert scan({"text": harmless}) == {}


def test_counts_each_occurrence_once() -> None:
    text = f"{AWS_KEY_ID} and {AWS_KEY_ID}, cards {CARD} {CARD_AMEX}"
    assert scan({"text": text}) == {CLOUD_KEY: 2, CARD_NUMBER: 2}


def test_counts_aggregate_across_strings() -> None:
    tree = {"a": AWS_KEY_ID, "b": [AWS_KEY_ID, {"c": CARD}]}
    assert scan(tree) == {CLOUD_KEY: 2, CARD_NUMBER: 1}


def test_only_enabled_categories_are_scanned() -> None:
    tree = {"a": AWS_KEY_ID, "b": CARD}
    assert scan(tree, (CARD_NUMBER,)) == {CARD_NUMBER: 1}
    assert scan(tree, ()) == {}


def test_results_contain_category_counts_only() -> None:
    result = scan({AWS_KEY_ID: CARD})
    assert set(result) <= set(CATEGORIES)
    assert all(isinstance(v, int) for v in result.values())


@pytest.mark.parametrize("text", ["card 4111111111111111 12/28", "ref 1 4111111111111111"])
def test_card_next_to_a_short_number_is_still_detected(text: str) -> None:
    assert scan({"text": text}) == {CARD_NUMBER: 1}


# --- Luhn --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "digits", ["79927398713", "4111111111111111", "5500000000000004", "378282246310005", "0"]
)
def test_luhn_valid(digits: str) -> None:
    assert luhn_valid(digits)


@pytest.mark.parametrize("digits", ["79927398710", "4111111111111112", "1234567812345678"])
def test_luhn_invalid(digits: str) -> None:
    assert not luhn_valid(digits)


# --- literal versus escaped: identical counts --------------------------------------------


@pytest.mark.parametrize(
    "secret",
    [sample for samples in SAMPLES.values() for sample in samples],
)
def test_escaped_string_value_gives_identical_counts(secret: str) -> None:
    plain = scan_json('{"content":' + literal(secret) + "}")
    escaped = scan_json('{"content":' + escape_all(secret) + "}")
    assert plain == escaped
    assert sum(plain.values()) == 1


@pytest.mark.parametrize(
    "secret",
    [sample for samples in SAMPLES.values() for sample in samples],
)
def test_escaped_object_key_gives_identical_counts(secret: str) -> None:
    plain = scan_json("{" + literal(secret) + ':"v"}')
    escaped = scan_json("{" + escape_all(secret) + ':"v"}')
    assert plain == escaped
    assert sum(plain.values()) == 1


def test_escaped_newlines_and_quotes_decode_normally_without_double_counting() -> None:
    # A PEM block whose newlines are JSON escapes, next to an escaped quote.
    pem = PEM_HEADER + "\nMIIEexample\n-----END " + 'RSA PRIVATE KEY-----"'
    plain = scan_json('{"system":' + literal(pem) + "}")
    escaped = scan_json('{"system":' + escape_all(pem) + "}")
    assert plain == escaped == {PRIVATE_KEY: 1}


def test_mixed_partial_escaping_matches_too() -> None:
    # Only some characters escaped, as a client library might do.
    escaped_key = '"AKIA\\u0049OSFODNN7EXAMPLE"'
    assert scan_json('{"k":' + escaped_key + "}") == {CLOUD_KEY: 1}


def test_paired_corpus_in_nested_arrays_and_tool_schemas() -> None:
    template = (
        '{{"model":"m","max_tokens":1,'
        '"system":[{{"type":"text","text":{a}}}],'
        '"messages":[{{"role":"user","content":[[{{"type":"text","text":{b}}}]]}}],'
        '"tools":[{{"name":"lookup","description":{c},'
        '"input_schema":{{"type":"object","properties":{{{d}:{{"type":"string",'
        '"enum":[{e}]}}}}}}}}],'
        '"metadata":{{"user_id":{f}}}}}'
    )
    values = {
        "a": PEM_HEADER,
        "b": CARD_SPACED,
        "c": ANTHROPIC_TOKEN,
        "d": AWS_KEY_ID,  # an object key inside a tool schema
        "e": GITHUB_TOKEN,
        "f": CARD_AMEX,
    }
    plain = scan_json(template.format(**{k: literal(v) for k, v in values.items()}))
    escaped = scan_json(template.format(**{k: escape_all(v) for k, v in values.items()}))
    expected = {PRIVATE_KEY: 1, CARD_NUMBER: 2, API_TOKEN: 2, CLOUD_KEY: 1}
    assert plain == expected
    assert escaped == expected


# --- traversal rules ---------------------------------------------------------------------


def test_object_keys_are_scanned() -> None:
    assert scan({AWS_KEY_ID: "value"}) == {CLOUD_KEY: 1}
    assert scan({"outer": {CARD: 1}}) == {CARD_NUMBER: 1}


def test_card_number_as_a_json_integer_is_not_detected() -> None:
    # Numbers are never coerced to strings.
    assert scan_json('{"amount":' + CARD + "}") == {}
    assert scan({"amount": int(CARD)}) == {}


def test_booleans_and_null_are_not_coerced() -> None:
    assert scan({"a": True, "b": None, "c": 4111111111111111.0}) == {}


def test_adjacent_strings_are_not_joined() -> None:
    # Each half alone is too short to be a card number; joined they would pass Luhn.
    assert scan({"content": ["41111111", "11111111"]}) == {}
    assert scan({"first": "41111111", "second": "11111111"}) == {}
    # Same for a token split between a key and its value.
    assert scan({"AKIA": "IOSFODNN7EXAMPLE"}) == {}


def test_empty_and_scalar_trees() -> None:
    assert scan({}) == {}
    assert scan([]) == {}
    assert scan(AWS_KEY_ID) == {CLOUD_KEY: 1}


def test_deeply_nested_tree_does_not_recurse_out_of_stack() -> None:
    tree: Any = AWS_KEY_ID
    for _ in range(5_000):
        tree = [tree]
    assert scan(tree) == {CLOUD_KEY: 1}


# --- outside detection coverage ----------------------------------------------------------


def test_out_of_scope_base64_of_a_private_key_is_not_detected() -> None:
    encoded = base64.b64encode((PEM_HEADER + "\nMIIEexample\n").encode()).decode()
    assert scan({"source": {"type": "base64", "data": encoded}}) == {}


def test_out_of_scope_secret_split_across_two_strings_is_not_detected() -> None:
    # Split so that neither piece is token-shaped on its own.
    head, tail = ANTHROPIC_TOKEN[:10], ANTHROPIC_TOKEN[10:]
    assert scan({"content": [head, tail]}) == {}
    assert scan({"content": head + tail}) == {API_TOKEN: 1}  # the joined text would match


def test_out_of_scope_nested_json_string_is_scanned_only_as_literal_text() -> None:
    # A JSON document embedded in a string is not parsed again. When the secret survives
    # as literal text it still matches; when its characters are escaped inside the
    # inner document (double-encoded), it does not.
    inner_escaped = '{"key":' + escape_all(AWS_KEY_ID) + "}"
    outer = '{"content":' + json.dumps(inner_escaped) + "}"
    assert scan_json(outer) == {}


def test_out_of_scope_double_encoded_unicode_escapes_are_not_decoded_twice() -> None:
    # After one decoding pass the text is backslash-u sequences, not the secret.
    once_escaped = escape_all(AWS_KEY_ID)[1:-1]
    assert scan({"content": once_escaped}) == {}

"""Minimal request parsing, output limits, duplicate keys and the single permitted body
rewrite: the gate may change only the top-level ``model`` value.

Parse failures are 400 ``invalid_request_error``. The original bytes are what get
forwarded: the decoded tree is only read.
"""

import json

import pytest

from gate.domain.errors import InvalidRequest
from gate.domain.models import Route
from gate.domain.request import ParsedRequest, parse_request, rewrite_model


def parse(body: bytes, route: Route = Route.MESSAGES) -> ParsedRequest:
    return parse_request(body, route)


def rejected(body: bytes, route: Route = Route.MESSAGES) -> InvalidRequest:
    with pytest.raises(InvalidRequest) as info:
        parse_request(body, route)
    error = info.value
    assert error.status == 400
    assert error.error_type == "invalid_request_error"
    return error


# --- happy path --------------------------------------------------------------------------


def test_reads_model_stream_and_output_limit() -> None:
    body = b'{"model":"claude-sonnet-5-5","max_tokens":1024,"stream":true,"messages":[]}'
    parsed = parse(body)
    assert parsed.model == "claude-sonnet-5-5"
    assert parsed.stream is True
    assert parsed.output_limit == 1024
    assert parsed.body_size == len(body)
    assert parsed.tree["messages"] == []


@pytest.mark.parametrize("stream", [b"false", b'"true"', b"1", b"null"])
def test_only_literal_true_enables_streaming(stream: bytes) -> None:
    parsed = parse(b'{"model":"m","max_tokens":1,"stream":' + stream + b"}")
    assert parsed.stream is False


def test_stream_defaults_to_false() -> None:
    assert parse(b'{"model":"m","max_tokens":1}').stream is False


def test_zero_output_limit_is_allowed() -> None:
    assert parse(b'{"model":"m","max_tokens":0}').output_limit == 0


# --- invalid JSON ------------------------------------------------------------------------


@pytest.mark.parametrize(
    "body",
    [
        b"",
        b"{",
        b'{"model":"m","max_tokens":1,}',  # trailing comma
        b"{'model':'m'}",  # single quotes
        b'{"model":"m"} trailing',
        b'\xff{"model":"m"}',  # not UTF-8
    ],
)
def test_invalid_json_is_rejected(body: bytes) -> None:
    rejected(body)


@pytest.mark.parametrize("constant", [b"NaN", b"Infinity", b"-Infinity"])
def test_non_standard_constants_are_rejected(constant: bytes) -> None:
    rejected(b'{"model":"m","max_tokens":1,"temperature":' + constant + b"}")


@pytest.mark.parametrize("body", [b"[]", b'"model"', b"42", b"null"])
def test_body_must_be_an_object(body: bytes) -> None:
    rejected(body)


# --- duplicate keys ----------------------------------------------------------------------


def test_duplicate_top_level_key_is_rejected() -> None:
    rejected(b'{"model":"a","model":"b","max_tokens":1}')


def test_key_duplicated_through_a_unicode_escape_is_rejected() -> None:
    # "\u006dodel" decodes to "model": a byte-level comparison would miss it, while an
    # upstream parser would see two "model" keys.
    rejected(rb'{"model":"a","\u006dodel":"b","max_tokens":1}')


def test_duplicate_key_in_a_nested_object_is_rejected() -> None:
    rejected(b'{"model":"m","max_tokens":1,"metadata":{"user_id":"x","user_id":"y"}}')


def test_duplicate_key_error_does_not_echo_the_key_name() -> None:
    error = rejected(b'{"model":"m","max_tokens":1,"secret-ish-name":1,"secret-ish-name":2}')
    assert "secret-ish-name" not in error.message


def test_same_key_in_sibling_objects_is_not_a_duplicate() -> None:
    body = b'{"model":"m","max_tokens":1,"messages":[{"role":"user"},{"role":"assistant"}]}'
    assert parse(body).model == "m"


# --- model -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "body",
    [
        b'{"max_tokens":1}',
        b'{"model":null,"max_tokens":1}',
        b'{"model":"","max_tokens":1}',
        b'{"model":42,"max_tokens":1}',
        b'{"model":["a"],"max_tokens":1}',
    ],
)
def test_missing_or_invalid_model_is_rejected(body: bytes) -> None:
    rejected(body)


def test_model_is_read_after_escape_decoding() -> None:
    assert parse(rb'{"model":"claude-\u0073onnet","max_tokens":1}').model == "claude-sonnet"


# --- output limit ------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fragment",
    [
        b"",  # missing
        b',"max_tokens":null',
        b',"max_tokens":-1',
        b',"max_tokens":true',
        b',"max_tokens":false',
        b',"max_tokens":1.0',
        b',"max_tokens":1.5',
        b',"max_tokens":"100"',
        b',"max_tokens":[100]',
    ],
)
def test_invalid_output_limit(fragment: bytes) -> None:
    error = rejected(b'{"model":"m"' + fragment + b"}")
    assert error.gateway_code == "invalid_output_limit"


def test_count_tokens_route_needs_no_output_limit() -> None:
    parsed = parse(b'{"model":"m","messages":[]}', Route.COUNT_TOKENS)
    assert parsed.output_limit is None


def test_count_tokens_route_still_rejects_invalid_json_and_duplicates() -> None:
    rejected(b"{", Route.COUNT_TOKENS)
    rejected(b'{"model":"a","model":"b"}', Route.COUNT_TOKENS)
    rejected(b'{"messages":[]}', Route.COUNT_TOKENS)


# --- rewrite_model -----------------------------------------------------------------------

ODD_BODY = (
    b"  {\n"
    b'  "messages" : [ {"role":"user","content":"caf\xc3\xa9 \\u00e9 \\"q\\"",'
    b' "model": "nested-in-message"} ],\r\n'
    b'\t"model"\t:\t"alias-name" ,'
    b'   "metadata":{"model":"nested-in-metadata", "list":[{"model":"deep"}]},'
    b'"max_tokens":10 ,"tools":[{"name":"t","input_schema":{"properties":{"model":{}}}}]'
    b"\n}\n"
)


def test_rewrite_changes_only_the_top_level_model_value() -> None:
    rewritten = rewrite_model(ODD_BODY, "provider-name")
    assert rewritten == ODD_BODY.replace(b'"alias-name"', b'"provider-name"')


def test_rewrite_leaves_nested_model_keys_untouched() -> None:
    rewritten = rewrite_model(ODD_BODY, "provider-name")
    tree = json.loads(rewritten)
    assert tree["model"] == "provider-name"
    assert tree["messages"][0]["model"] == "nested-in-message"
    assert tree["metadata"]["model"] == "nested-in-metadata"
    assert tree["metadata"]["list"][0]["model"] == "deep"
    assert tree["tools"][0]["input_schema"]["properties"] == {"model": {}}


def test_rewrite_when_model_is_the_first_key() -> None:
    body = b'{"model":"a","max_tokens":5}'
    assert rewrite_model(body, "b") == b'{"model":"b","max_tokens":5}'


def test_rewrite_when_model_is_the_last_key() -> None:
    body = b'{"max_tokens":5,"model":"a"}'
    assert rewrite_model(body, "b") == b'{"max_tokens":5,"model":"b"}'


def test_rewrite_handles_an_escaped_key_and_value() -> None:
    body = rb'{"x":1, "\u006dodel" : "al\u0069as", "y":2}'
    assert rewrite_model(body, "real") == rb'{"x":1, "\u006dodel" : "real", "y":2}'


def test_rewrite_serialises_the_new_name_as_a_json_string() -> None:
    rewritten = rewrite_model(b'{"model":"a"}', 'we"ird')
    assert json.loads(rewritten) == {"model": 'we"ird'}


def test_rewritten_body_parses_identically_apart_from_model() -> None:
    before = parse(ODD_BODY).tree
    after = parse(rewrite_model(ODD_BODY, "provider-name")).tree
    assert {**before, "model": "provider-name"} == after

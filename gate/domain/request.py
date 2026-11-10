"""Minimal request parsing: model, stream flag, output limit, duplicate keys.

The JSON body is decoded exactly once. The decoded tree is reused for pricing
feature checks and detection, while the original bytes are what get forwarded.
"""

import json
from dataclasses import dataclass
from typing import Any

from gate.domain.errors import InvalidRequest
from gate.domain.models import ApiFormat, Route


@dataclass(frozen=True)
class ParsedRequest:
    model: str
    stream: bool
    output_limit: int | None  # None only for unmetered routes
    tree: dict[str, Any]
    body_size: int


class DuplicateKeyError(ValueError):
    pass


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    obj: dict[str, Any] = {}
    for key, value in pairs:
        if key in obj:
            # Never echo the key: object-key names can carry content.
            raise DuplicateKeyError("duplicate object key")
        obj[key] = value
    return obj


def _reject_constant(name: str) -> Any:
    raise ValueError(f"non-standard JSON constant {name}")


def decode_json(body: bytes) -> Any:
    """Decode strict JSON, rejecting duplicate keys (after escape decoding) and NaN/Infinity."""
    return json.loads(
        body,
        object_pairs_hook=_reject_duplicates,
        parse_constant=_reject_constant,
    )


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _output_limit(tree: dict[str, Any], fmt: ApiFormat) -> int:
    if fmt is ApiFormat.ANTHROPIC:
        candidates = ["max_tokens"]
    else:
        candidates = ["max_completion_tokens", "max_tokens"]
    present = [name for name in candidates if name in tree]
    if len(present) != 1:
        raise InvalidRequest(
            f"exactly one explicit output limit is required ({' or '.join(candidates)})",
            gateway_code="invalid_output_limit",
        )
    value = tree[present[0]]
    if not _is_int(value) or value < 0:
        raise InvalidRequest(
            f"{present[0]} must be a non-negative integer",
            gateway_code="invalid_output_limit",
        )
    return int(value)


def parse_request(body: bytes, route: Route, fmt: ApiFormat = ApiFormat.ANTHROPIC) -> ParsedRequest:
    try:
        tree = decode_json(body)
    except DuplicateKeyError:
        raise InvalidRequest(
            "request body contains a duplicate object key", gateway_code="duplicate_key"
        ) from None
    except (ValueError, UnicodeDecodeError, RecursionError):
        raise InvalidRequest(
            "request body is not valid JSON", gateway_code="invalid_json"
        ) from None

    if not isinstance(tree, dict):
        raise InvalidRequest("request body must be a JSON object", gateway_code="invalid_json")
    model = tree.get("model")
    if not isinstance(model, str) or not model:
        raise InvalidRequest("model is required", gateway_code="missing_model")

    output_limit = _output_limit(tree, fmt) if route.metered else None
    return ParsedRequest(
        model=model,
        stream=tree.get("stream") is True,
        output_limit=output_limit,
        tree=tree,
        body_size=len(body),
    )


def rewrite_model(body: bytes, new_model: str) -> bytes:
    """Replace only the top-level ``model`` value, keeping every other byte.

    This is the one body change the gate may make, and only for deployments
    that configure ``upstream_model``. Duplicate keys were already rejected.
    """
    text = body.decode("utf-8")
    decoder = json.JSONDecoder()
    pos = _skip_ws(text, 0)
    if text[pos : pos + 1] != "{":
        raise ValueError("body is not a JSON object")
    pos += 1
    while True:
        pos = _skip_ws(text, pos)
        if text[pos : pos + 1] == "}":
            raise ValueError("model key not found")
        key, pos = decoder.raw_decode(text, pos)
        pos = _skip_ws(text, pos)
        pos = _skip_ws(text, pos + 1)  # past ':'
        _, end = decoder.raw_decode(text, pos)
        if key == "model":
            replaced = text[:pos] + json.dumps(new_model) + text[end:]
            return replaced.encode("utf-8")
        pos = _skip_ws(text, end)
        if text[pos : pos + 1] == ",":
            pos += 1


def _skip_ws(text: str, pos: int) -> int:
    while pos < len(text) and text[pos] in " \t\r\n":
        pos += 1
    return pos

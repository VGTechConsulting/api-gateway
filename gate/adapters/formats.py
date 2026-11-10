"""Report rows as JSON, CSV or a text table, and report date parsing.

Shared by the CLI and the report UI's downloads, so ``report --format json`` and the
UI's JSON link return the same bytes for the same rows.
"""

import csv
import io
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

Rows = Sequence[dict[str, Any]]


def parse_when(value: str | None) -> datetime | None:
    """Date-only values mean midnight UTC; naive datetimes are UTC."""
    if value is None or value == "":
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        raise ValueError(f"invalid date: {value}") from None
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


def to_json(rows: Rows) -> str:
    return json.dumps(list(rows), indent=2, default=str) + "\n"


def to_csv(rows: Rows) -> str:
    buffer = io.StringIO()
    if rows:
        writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: cell(v) for k, v in row.items()})
    return buffer.getvalue()


def to_table(rows: Rows) -> str:
    if not rows:
        return "(no rows)\n"
    if len(rows) == 1 or len(rows[0]) > 12:
        # Wide reports read better transposed: one block per row.
        blocks = []
        for row in rows:
            width = max(len(k) for k in row)
            blocks.append("\n".join(f"{k.ljust(width)}  {cell(v)}" for k, v in row.items()))
        return "\n\n".join(blocks) + "\n"
    columns = list(rows[0])
    cells = [[cell(row.get(c)) for c in columns] for row in rows]
    widths = [max(len(c), *(len(r[i]) for r in cells)) for i, c in enumerate(columns)]
    lines = ["  ".join(c.ljust(w) for c, w in zip(columns, widths, strict=True))]
    lines += ["  ".join(v.ljust(w) for v, w in zip(r, widths, strict=True)) for r in cells]
    return "\n".join(lines) + "\n"


def render(rows: Rows, fmt: str) -> str:
    if fmt == "json":
        return to_json(rows)
    if fmt == "csv":
        return to_csv(rows)
    return to_table(rows)


def cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return ";".join(f"{k}={cell(v)}" for k, v in sorted(value.items()))
    if isinstance(value, list):
        return " | ".join(cell(v) for v in value)
    return str(value)

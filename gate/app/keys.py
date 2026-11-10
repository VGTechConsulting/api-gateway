"""Virtual key management: create, list, revoke.

Keys are ``gk_`` plus 32 random bytes in base62. Only the SHA-256 of the full
key is stored; the first 8 characters after ``gk_`` are kept as a display prefix.
The raw key is returned once, at creation.
"""

import secrets
import string
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any

from gate.app.lifecycle import KEY_PREFIX, hash_key
from gate.app.ports import KeyRecord, KeyRepository, WallClock
from gate.domain.models import Cadence, Env
from gate.domain.money import usd_to_micros

BASE62 = string.digits + string.ascii_uppercase + string.ascii_lowercase
KEY_BODY_LENGTH = 43  # ceil(256 / log2(62))
PREFIX_LENGTH = 8


class KeyInputError(ValueError):
    """Invalid key-management input; the message is safe to show."""


def base62(data: bytes, length: int) -> str:
    n = int.from_bytes(data, "big")
    chars = []
    while n:
        n, rem = divmod(n, 62)
        chars.append(BASE62[rem])
    return "".join(reversed(chars)).rjust(length, BASE62[0])


def generate_key() -> str:
    return KEY_PREFIX + base62(secrets.token_bytes(32), KEY_BODY_LENGTH)


@dataclass(frozen=True)
class NewKey:
    owner: str
    team: str
    app: str
    env: str
    models: Sequence[str]
    budget_usd: Decimal | None = None
    period: str = "month"
    tpm: int | None = None
    rpm: int | None = None


def _positive(name: str, value: int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise KeyInputError(f"{name} must be a positive integer")
    return value


class KeyService:
    def __init__(self, repo: KeyRepository, wall: WallClock) -> None:
        self._repo = repo
        self._wall = wall

    def create(self, spec: NewKey) -> tuple[KeyRecord, str]:
        for name in ("owner", "team", "app"):
            if not getattr(spec, name).strip():
                raise KeyInputError(f"--{name} is required")
        try:
            env = Env(spec.env)
        except ValueError:
            raise KeyInputError("--env must be one of dev, staging, prod") from None
        try:
            period = Cadence(spec.period)
        except ValueError:
            raise KeyInputError("--period must be day or month") from None
        models = tuple(m.strip() for m in spec.models if m.strip())
        if not models:
            raise KeyInputError("--models needs at least one model name")
        budget = None
        if spec.budget_usd is not None:
            if not spec.budget_usd.is_finite() or spec.budget_usd < 0:
                raise KeyInputError("--budget must be a non-negative amount in USD")
            budget = usd_to_micros(spec.budget_usd)

        raw = generate_key()
        record = KeyRecord(
            id=str(uuid.uuid4()),
            key_hash=hash_key(raw),
            prefix=raw[len(KEY_PREFIX) : len(KEY_PREFIX) + PREFIX_LENGTH],
            owner=spec.owner.strip(),
            team=spec.team.strip(),
            app=spec.app.strip(),
            env=env,
            allowed_models=models,
            budget_micros=budget,
            period=period,
            tpm=_positive("--tpm", spec.tpm),
            rpm=_positive("--rpm", spec.rpm),
            created_at=self._wall.now(),
            revoked_at=None,
        )
        self._repo.insert(record)
        return record, raw

    def list(
        self, *, team: str | None = None, include_revoked: bool = False
    ) -> Sequence[KeyRecord]:
        return self._repo.list(team=team, include_revoked=include_revoked)

    def revoke(self, ident: str) -> KeyRecord:
        matches = [k for k in self._repo.find_by_id_or_prefix(ident) if k.revoked_at is None]
        if not matches:
            raise KeyInputError("no active key matches that id or prefix")
        if len(matches) > 1:
            raise KeyInputError("prefix is ambiguous; use the key id")
        key = matches[0]
        at: datetime = self._wall.now()
        self._repo.revoke(key.id, at)
        return key


def key_row(k: KeyRecord) -> dict[str, Any]:
    """A key for display. Never includes the hash: keys are shown by prefix."""
    return {
        "id": k.id,
        "prefix": k.prefix,
        "owner": k.owner,
        "team": k.team,
        "app": k.app,
        "env": k.env.value,
        "models": ",".join(k.allowed_models),
        "budget_usd": (
            str(Decimal(k.budget_micros) / 1_000_000) if k.budget_micros is not None else None
        ),
        "period": k.period.value,
        "tpm": k.tpm,
        "rpm": k.rpm,
        "created_at": k.created_at.isoformat(timespec="seconds"),
        "revoked_at": k.revoked_at.isoformat(timespec="seconds") if k.revoked_at else None,
    }

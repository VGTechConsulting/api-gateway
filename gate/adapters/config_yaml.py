"""YAML file -> validated ``GatewayConfig``.

Pydantic stops at this boundary. Validation errors name the deployment, model
and field at fault and never print secrets: only environment variable *names*
are mentioned, never their values.
"""

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from gate.domain.config import DetectConfig, GatewayConfig, Limits, Team, UiConfig
from gate.domain.detection import CATEGORIES
from gate.domain.models import (
    ApiFormat,
    Budget,
    Cadence,
    Deployment,
    DetectMode,
    EnforcementMode,
    Policy,
    PriceEntry,
    PricingMode,
)
from gate.domain.money import usd_to_micros

RATE_FIELDS = ("input", "output", "cache_read", "cache_write_5m", "cache_write_1h")
DEFAULT_CONFIG_PATH = "config.yaml"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


def _positive_int(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("must be a positive integer")
    return value


class ServerModel(_Strict):
    host: str = "127.0.0.1"
    port: int = 8787


class UiModel(_Strict):
    host: str = "127.0.0.1"
    port: int = 8789
    recent_requests: Any = 100

    @field_validator("recent_requests")
    @classmethod
    def _rows(cls, value: Any) -> int:
        rows = _positive_int(value)
        if rows > 1000:
            raise ValueError("must be at most 1000")
        return rows


class StorageModel(_Strict):
    sqlite_path: str = "./api-gateway.db"
    request_log: str = "stdout"


class DefaultsModel(_Strict):
    tpm: Any = 200_000
    rpm: Any = 60

    @field_validator("tpm", "rpm")
    @classmethod
    def _limit(cls, value: Any) -> int:
        return _positive_int(value)


class LimitsModel(_Strict):
    max_request_bytes: int = 32 * 1024 * 1024
    max_buffer_bytes: int = 16 * 1024 * 1024
    upstream_timeout_s: float = 600.0
    connect_timeout_s: float = 10.0
    shutdown_grace_s: float = 25.0
    cleanup_timeout_s: float = 5.0

    model_config = ConfigDict(extra="forbid", strict=False)


class DeploymentModel(_Strict):
    name: str
    format: Literal["anthropic", "openai"]
    base_url: str
    api_key_env: str
    region: str
    provider: str | None = None
    upstream_model: str | None = None


class PriceModel(_Strict):
    pricing_mode: Literal["metered", "free"] = "metered"
    input: Any = Field(default=..., description="USD per million tokens")
    output: Any = ...
    cache_read: Any = ...
    cache_write_5m: Any = ...
    cache_write_1h: Any = ...
    response_models: list[str] = []

    @field_validator(*RATE_FIELDS)
    @classmethod
    def _rate(cls, value: Any) -> Decimal | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("must be a finite non-negative number or null (unsupported)")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("must be finite")
        try:
            rate = Decimal(str(value))
        except InvalidOperation:
            raise ValueError("must be a number") from None
        if rate < 0:
            raise ValueError("must not be negative")
        return rate


class TeamModel(_Strict):
    budget_usd: float | int | None = None
    period: Literal["day", "month"] = "month"
    enforcement: dict[
        Literal["allowlist", "rate_limits", "budgets"], Literal["observe", "enforce"]
    ] = {}


class EnforcementModel(_Strict):
    default: dict[
        Literal["allowlist", "rate_limits", "budgets"], Literal["observe", "enforce"]
    ] = {}


class DetectModel(_Strict):
    on_error: Literal["open", "closed"] = "open"
    categories: dict[str, Literal["off", "flag", "block"]] = {}


class LoggingModel(_Strict):
    session_headers: list[str] = ["x-session-id", "x-claude-code-session-id"]


class BudgetModeModel(_Strict):
    mode: Literal["reserve", "naive"] = "reserve"


class ConfigModel(_Strict):
    server: ServerModel = ServerModel()
    ui: UiModel = UiModel()
    storage: StorageModel = StorageModel()
    defaults: DefaultsModel = DefaultsModel()
    limits: LimitsModel = LimitsModel()
    deployments: list[DeploymentModel]
    aliases: dict[str, list[str]]
    prices: dict[str, dict[str, PriceModel]] = {}
    enforcement: EnforcementModel = EnforcementModel()
    teams: dict[str, TeamModel] = {}
    detect: DetectModel = DetectModel()
    logging: LoggingModel = LoggingModel()
    budget: BudgetModeModel = BudgetModeModel()


@dataclass
class ConfigReport:
    config: GatewayConfig | None
    errors: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.config is not None and not self.errors


def config_path(explicit: str | None, env: Mapping[str, str]) -> str:
    return explicit or env.get("API_GATEWAY_CONFIG") or DEFAULT_CONFIG_PATH


def load_config(
    path: str,
    env: Mapping[str, str] | None = None,
    *,
    check_secrets: bool = True,
    experimental: bool = False,
) -> ConfigReport:
    env = os.environ if env is None else env
    try:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except OSError as exc:
        return ConfigReport(None, [f"cannot read config file {path}: {exc.strerror}"])
    except yaml.YAMLError as exc:
        return ConfigReport(None, [f"config file is not valid YAML: {type(exc).__name__}"])
    if not isinstance(raw, dict):
        return ConfigReport(None, ["config file must contain a mapping"])
    _resolve_storage_paths(raw, Path(path).resolve().parent)
    return validate_config(raw, env, check_secrets=check_secrets, experimental=experimental)


def _resolve_storage_paths(raw: dict[str, Any], base: Path) -> None:
    """Relative storage paths in the file are relative to the file, so the CLI and the
    server agree on one database whatever their working directories."""
    storage = raw.get("storage")
    if not isinstance(storage, dict):
        return
    for name in ("sqlite_path", "request_log"):
        value = storage.get(name)
        if (
            isinstance(value, str)
            and value not in ("stdout", ":memory:")
            and not Path(value).is_absolute()
        ):
            storage[name] = str(base / value)


def validate_config(
    raw: dict[str, Any],
    env: Mapping[str, str],
    *,
    check_secrets: bool = True,
    experimental: bool = False,
) -> ConfigReport:
    _apply_env_overrides(raw, env)
    try:
        model = ConfigModel.model_validate(raw)
    except ValidationError as exc:
        return ConfigReport(None, [_format_error(err) for err in exc.errors()])

    errors: list[str] = []
    notes: list[str] = []
    deployments = {d.name: d for d in model.deployments}
    if len(deployments) != len(model.deployments):
        errors.append("deployments: names must be unique")
    for d in model.deployments:
        if d.format == "openai":
            errors.append(f"deployments.{d.name}: format openai is not implemented in this build")
        if check_secrets and not env.get(d.api_key_env):
            errors.append(f"deployments.{d.name}: environment variable {d.api_key_env} is not set")

    for alias, targets in model.aliases.items():
        if not targets:
            errors.append(f"aliases.{alias}: needs at least one deployment")
        for target in targets:
            if target not in deployments:
                errors.append(f"aliases.{alias}: unknown deployment {target}")
            elif alias not in model.prices.get(target, {}):
                errors.append(f"prices.{target}.{alias}: missing price entry")

    prices: dict[tuple[str, str], PriceEntry] = {}
    for dep_name, models in model.prices.items():
        if dep_name not in deployments:
            errors.append(f"prices.{dep_name}: unknown deployment")
            continue
        for model_name, entry in models.items():
            where = f"prices.{dep_name}.{model_name}"
            rates: dict[str, Decimal | None] = {f: getattr(entry, f) for f in RATE_FIELDS}
            for required in ("input", "output"):
                if rates[required] is None:
                    errors.append(f"{where}.{required}: cannot be null")
            supported = [r for r in rates.values() if r is not None]
            if entry.pricing_mode == "metered" and not any(r > 0 for r in supported):
                errors.append(f"{where}: metered entries need at least one positive rate")
            if entry.pricing_mode == "free":
                if any(r > 0 for r in supported):
                    errors.append(f"{where}: free entries must have zero or null rates")
                notes.append(
                    f"{where}: explicit free mode (zero monetary cost assigned by the gateway)"
                )
            if rates["input"] is None or rates["output"] is None:
                continue
            prices[(dep_name, model_name)] = PriceEntry(
                deployment=dep_name,
                model=model_name,
                pricing_mode=PricingMode(entry.pricing_mode),
                input=rates["input"],
                output=rates["output"],
                cache_read=rates["cache_read"],
                cache_write_5m=rates["cache_write_5m"],
                cache_write_1h=rates["cache_write_1h"],
                response_models=frozenset(entry.response_models),
            )

    for name in model.detect.categories:
        if name not in CATEGORIES:
            errors.append(
                f"detect.categories.{name}: unknown category (known: {', '.join(CATEGORIES)})"
            )

    if model.budget.mode == "naive" and not experimental:
        errors.append(
            "budget.mode: naive exists only for the budget-overshoot experiment"
            " and requires --experimental"
        )

    configured = {str(k): str(v) for k, v in model.enforcement.default.items()}
    enforcement_default = {
        policy: EnforcementMode(configured.get(policy.value, "enforce")) for policy in Policy
    }
    for policy, mode in enforcement_default.items():
        if mode is EnforcementMode.OBSERVE:
            notes.append(f"enforcement.default.{policy.value}: observe")
    teams: dict[str, Team] = {}
    for team_name, team in model.teams.items():
        if team.budget_usd is not None and (
            isinstance(team.budget_usd, bool)
            or team.budget_usd < 0
            or not math.isfinite(team.budget_usd)
        ):
            errors.append(f"teams.{team_name}.budget_usd: must be a non-negative amount")
            continue
        overrides = {Policy(p): EnforcementMode(m) for p, m in team.enforcement.items()}
        for policy, mode in overrides.items():
            if mode is EnforcementMode.OBSERVE:
                notes.append(f"teams.{team_name}.enforcement.{policy.value}: observe")
        teams[team_name] = Team(
            name=team_name,
            budget=(
                Budget(usd_to_micros(Decimal(str(team.budget_usd))), Cadence(team.period))
                if team.budget_usd is not None
                else None
            ),
            enforcement=overrides,
        )

    if errors:
        return ConfigReport(None, errors, notes)

    limits = model.limits
    config = GatewayConfig(
        host=model.server.host,
        port=model.server.port,
        sqlite_path=model.storage.sqlite_path,
        request_log=model.storage.request_log,
        default_tpm=model.defaults.tpm,
        default_rpm=model.defaults.rpm,
        limits=Limits(
            max_request_bytes=limits.max_request_bytes,
            max_buffer_bytes=limits.max_buffer_bytes,
            upstream_timeout_s=limits.upstream_timeout_s,
            connect_timeout_s=limits.connect_timeout_s,
            shutdown_grace_s=limits.shutdown_grace_s,
            cleanup_timeout_s=limits.cleanup_timeout_s,
        ),
        deployments={
            d.name: Deployment(
                name=d.name,
                format=ApiFormat(d.format),
                base_url=d.base_url,
                api_key_env=d.api_key_env,
                provider=d.provider or d.name,
                region=d.region,
                upstream_model=d.upstream_model,
            )
            for d in model.deployments
        },
        aliases={alias: tuple(targets) for alias, targets in model.aliases.items()},
        prices=prices,
        enforcement_default=enforcement_default,
        teams=teams,
        detect=DetectConfig(
            on_error_open=model.detect.on_error == "open",
            # Every category flags unless the file says otherwise.
            categories={
                name: DetectMode(model.detect.categories.get(name, "flag")) for name in CATEGORIES
            },
        ),
        session_headers=tuple(h.lower() for h in model.logging.session_headers),
        budget_mode_naive=model.budget.mode == "naive",
        ui=UiConfig(
            host=model.ui.host, port=model.ui.port, recent_requests=model.ui.recent_requests
        ),
    )
    return ConfigReport(config, [], notes)


def _apply_env_overrides(raw: dict[str, Any], env: Mapping[str, str]) -> None:
    server = raw.setdefault("server", {}) or {}
    storage = raw.setdefault("storage", {}) or {}
    ui = raw.setdefault("ui", {}) or {}
    raw["server"], raw["storage"], raw["ui"] = server, storage, ui
    for section, prefix in ((server, "API_GATEWAY_"), (ui, "API_GATEWAY_UI_")):
        if prefix + "HOST" in env:
            section["host"] = env[prefix + "HOST"]
        if prefix + "PORT" in env:
            try:
                section["port"] = int(env[prefix + "PORT"])
            except ValueError:
                section["port"] = env[prefix + "PORT"]  # rejected by validation
    if "API_GATEWAY_DB" in env:
        storage["sqlite_path"] = env["API_GATEWAY_DB"]
    if "API_GATEWAY_REQUEST_LOG" in env:
        storage["request_log"] = env["API_GATEWAY_REQUEST_LOG"]


def _format_error(err: Mapping[str, Any]) -> str:
    loc = ".".join(str(part) for part in err.get("loc", ()))
    message = str(err.get("msg", "invalid value"))
    if message.startswith("Value error, "):
        message = message[len("Value error, ") :]
    return f"{loc}: {message}"

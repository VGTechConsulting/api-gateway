"""Validated gateway configuration as plain domain values.

The YAML adapter parses and validates the file, then builds these frozen
dataclasses. Nothing outside the config adapter sees pydantic or YAML.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field

from gate.domain.models import (
    Budget,
    Deployment,
    DetectMode,
    EnforcementMode,
    Policy,
    PriceEntry,
)

EnforcementModes = Mapping[Policy, EnforcementMode]

ALL_ENFORCED: EnforcementModes = {policy: EnforcementMode.ENFORCE for policy in Policy}


@dataclass(frozen=True)
class Limits:
    max_request_bytes: int = 32 * 1024 * 1024
    max_buffer_bytes: int = 16 * 1024 * 1024
    upstream_timeout_s: float = 600.0
    connect_timeout_s: float = 10.0
    shutdown_grace_s: float = 25.0
    cleanup_timeout_s: float = 5.0


@dataclass(frozen=True)
class Team:
    name: str
    budget: Budget | None
    enforcement: Mapping[Policy, EnforcementMode] = field(default_factory=dict)


@dataclass(frozen=True)
class DetectConfig:
    on_error_open: bool = True
    categories: Mapping[str, DetectMode] = field(default_factory=dict)


@dataclass(frozen=True)
class UiConfig:
    """The read-only report UI's listener and page size."""

    host: str = "127.0.0.1"
    port: int = 8789
    recent_requests: int = 100


@dataclass(frozen=True)
class GatewayConfig:
    host: str
    port: int
    sqlite_path: str
    request_log: str
    default_tpm: int
    default_rpm: int
    limits: Limits
    deployments: Mapping[str, Deployment]
    aliases: Mapping[str, tuple[str, ...]]
    prices: Mapping[tuple[str, str], PriceEntry]  # (deployment, requested model)
    enforcement_default: EnforcementModes
    teams: Mapping[str, Team]
    detect: DetectConfig
    session_headers: tuple[str, ...]
    budget_mode_naive: bool = False  # budget-overshoot experiment only
    ui: UiConfig = field(default_factory=UiConfig)

    def team(self, name: str) -> Team | None:
        return self.teams.get(name)

    def enforcement_for(self, team: str) -> EnforcementModes:
        """Resolve per-policy modes: global default, overridden by the key's team."""
        modes = dict(self.enforcement_default)
        team_cfg = self.teams.get(team)
        if team_cfg is not None:
            modes.update(team_cfg.enforcement)
        return modes

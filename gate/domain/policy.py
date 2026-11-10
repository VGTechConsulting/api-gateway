"""Policy decisions and enforcement-mode resolution.

Each check returns a ``Decision`` describing what enforce mode would do. The
lifecycle applies the resolved mode with ``apply_modes``, so observe and
enforce share one implementation of every check.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from gate.domain.errors import GateError, PermissionDenied
from gate.domain.models import EnforcementMode, Policy, PolicyEvent, Principal

WILDCARD = "*"


@dataclass(frozen=True)
class Decision:
    policy: Policy
    check: str
    error: GateError | None = None  # the error enforce mode would return; None = passed
    detail: Mapping[str, str | int | bool | None] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.error is None


@dataclass(frozen=True)
class Verdict:
    denial: GateError | None
    would_deny: tuple[str, ...]
    events: tuple[PolicyEvent, ...]


def check_allowlist(principal: Principal, model: str) -> Decision:
    allowed = principal.allowed_models
    if model in allowed or WILDCARD in allowed:
        return Decision(Policy.ALLOWLIST, "allowlist")
    return Decision(
        Policy.ALLOWLIST,
        "allowlist",
        PermissionDenied("model is not allowed for this key", gateway_code="model_not_allowed"),
    )


def apply_modes(
    decisions: Iterable[Decision],
    modes: Mapping[Policy, EnforcementMode],
    *,
    stage: str = "admission",
) -> Verdict:
    """Apply per-policy modes to decisions in evaluation order.

    The first failure under an enforced policy denies the request. Failures
    under observed policies are recorded as would-deny and the request continues.
    """
    denial: GateError | None = None
    would_deny: list[str] = []
    events: list[PolicyEvent] = []
    for decision in decisions:
        mode = modes.get(decision.policy, EnforcementMode.ENFORCE)
        if decision.error is None:
            continue
        code = decision.error.gateway_code
        if mode is EnforcementMode.ENFORCE:
            if denial is None:
                denial = decision.error
                events.append(_event(decision, code, "deny", mode, stage))
            continue
        would_deny.append(code)
        events.append(_event(decision, code, "would_deny", mode, stage))
    return Verdict(denial, tuple(would_deny), tuple(events))


def _event(
    decision: Decision, code: str, result: str, mode: EnforcementMode, stage: str
) -> PolicyEvent:
    return {
        "policy": decision.policy.value,
        "check": decision.check,
        "gateway_code": code,
        "result": result,
        "mode": mode.value,
        "stage": stage,
        **decision.detail,
    }


def event(kind: str, **fields: str | int | bool | None) -> PolicyEvent:
    """A non-admission entry for ``policy_decisions`` (estimated debit, overrun, mismatch, ...)."""
    return {"event": kind, **fields}

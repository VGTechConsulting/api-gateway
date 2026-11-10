"""Configuration validation: price-entry rules, pricing modes, enforcement settings,
default limits and environment overrides.

``validate_config`` takes the already-parsed YAML mapping and an explicit environment,
so these tests never touch the filesystem or ``os.environ``. Errors must identify the
deployment, model and field at fault and must never print secret values.
"""

import copy
from decimal import Decimal
from typing import Any

import pytest

from gate.adapters.config_yaml import ConfigReport, validate_config
from gate.domain.detection import CATEGORIES
from gate.domain.models import Cadence, DetectMode, EnforcementMode, Policy, PricingMode

SECRET = "sk-super-secret-provider-value"
ENV = {"ANTHROPIC_API_KEY": SECRET, "DEEPSEEK_API_KEY": SECRET + "-2"}

BASE: dict[str, Any] = {
    "deployments": [
        {
            "name": "anthropic-direct",
            "format": "anthropic",
            "base_url": "https://api.anthropic.com",
            "api_key_env": "ANTHROPIC_API_KEY",
            "region": "us",
        },
        {
            "name": "deepseek",
            "format": "anthropic",
            "base_url": "https://api.deepseek.com/anthropic",
            "api_key_env": "DEEPSEEK_API_KEY",
            "region": "cn",
        },
    ],
    "aliases": {"claude-sonnet-5-5": ["anthropic-direct"], "deepseek-flash": ["deepseek"]},
    "prices": {
        "anthropic-direct": {
            "claude-sonnet-5-5": {
                "pricing_mode": "metered",
                "input": 3,
                "output": 15,
                "cache_read": 0.3,
                "cache_write_5m": 3.75,
                "cache_write_1h": 6,
            }
        },
        "deepseek": {
            "deepseek-flash": {
                "input": 0.27,
                "output": 1.1,
                "cache_read": 0.07,
                "cache_write_5m": None,
                "cache_write_1h": None,
            }
        },
    },
}


def config(**overrides: Any) -> dict[str, Any]:
    raw = copy.deepcopy(BASE)
    raw.update(overrides)
    return raw


def entry(
    raw: dict[str, Any], deployment: str = "anthropic-direct", model: str = "claude-sonnet-5-5"
) -> dict[str, Any]:
    result: dict[str, Any] = raw["prices"][deployment][model]
    return result


def validate(raw: dict[str, Any], env: dict[str, str] | None = None, **kwargs: Any) -> ConfigReport:
    return validate_config(raw, ENV if env is None else env, **kwargs)


def output_text(report: ConfigReport) -> str:
    return "\n".join(report.errors + report.notes)


def assert_rejected(report: ConfigReport, *fragments: str) -> str:
    assert not report.ok
    assert report.config is None
    text = output_text(report)
    for fragment in fragments:
        assert fragment in text, f"{fragment!r} not in {report.errors}"
    assert SECRET not in text
    return text


# --- valid configs -----------------------------------------------------------------------


def test_minimal_valid_config_passes() -> None:
    report = validate(config())
    assert report.ok, report.errors
    cfg = report.config
    assert cfg is not None
    assert cfg.aliases["claude-sonnet-5-5"] == ("anthropic-direct",)
    entry_ = cfg.prices[("anthropic-direct", "claude-sonnet-5-5")]
    assert entry_.pricing_mode is PricingMode.METERED
    assert entry_.input == Decimal("3")
    assert entry_.cache_read == Decimal("0.3")
    assert cfg.default_tpm == 200_000
    assert cfg.default_rpm == 60
    assert all(mode is EnforcementMode.ENFORCE for mode in cfg.enforcement_default.values())


def test_omitted_mode_behaves_as_metered_and_null_stays_unsupported() -> None:
    cfg = validate(config()).config
    assert cfg is not None
    deepseek = cfg.prices[("deepseek", "deepseek-flash")]
    assert deepseek.pricing_mode is PricingMode.METERED
    assert deepseek.cache_write_5m is None  # NULL is unsupported, never zero
    assert deepseek.cache_write_1h is None


def test_mixed_positive_and_zero_metered_rates_pass() -> None:
    raw = config()
    entry(raw)["cache_read"] = 0
    assert validate(raw).ok


def test_explicit_free_all_zero_entry_passes_and_is_listed_in_notes() -> None:
    raw = config()
    entry(raw).update(
        pricing_mode="free", input=0, output=0, cache_read=0, cache_write_5m=0, cache_write_1h=None
    )
    report = validate(raw)
    assert report.ok, report.errors
    assert report.config is not None
    assert report.config.prices[("anthropic-direct", "claude-sonnet-5-5")].pricing_mode is (
        PricingMode.FREE
    )
    assert any(
        "prices.anthropic-direct.claude-sonnet-5-5" in note and "free" in note
        for note in report.notes
    )


# --- price entries -----------------------------------------------------------------------


@pytest.mark.parametrize(
    "field", ["input", "output", "cache_read", "cache_write_5m", "cache_write_1h"]
)
def test_required_placeholder_is_rejected(field: str) -> None:
    raw = config()
    entry(raw)[field] = "REQUIRED"
    assert_rejected(validate(raw), "anthropic-direct", "claude-sonnet-5-5", field)


@pytest.mark.parametrize("field", ["cache_write_5m", "cache_write_1h", "cache_read", "input"])
def test_missing_category_is_rejected(field: str) -> None:
    raw = config()
    del entry(raw)[field]
    assert_rejected(validate(raw), "anthropic-direct", "claude-sonnet-5-5", field)


def test_old_single_cache_write_key_is_rejected() -> None:
    raw = config()
    e = entry(raw)
    del e["cache_write_5m"], e["cache_write_1h"]
    e["cache_write"] = 3.75
    assert_rejected(validate(raw), "anthropic-direct", "claude-sonnet-5-5", "cache_write")


def test_old_cache_write_key_is_rejected_even_alongside_the_split() -> None:
    raw = config()
    entry(raw)["cache_write"] = 3.75
    assert_rejected(validate(raw), "claude-sonnet-5-5", "cache_write")


@pytest.mark.parametrize(
    ("value", "label"),
    [
        (-1, "negative"),
        (-0.01, "negative float"),
        (True, "boolean"),
        (False, "boolean false"),
        (float("nan"), "NaN"),
        (float("inf"), "infinity"),
        ("3.00", "numeric string"),
        ([3], "list"),
    ],
)
def test_invalid_rate_is_rejected(value: object, label: str) -> None:
    raw = config()
    entry(raw)["output"] = value
    assert_rejected(validate(raw), "anthropic-direct", "claude-sonnet-5-5", "output")


@pytest.mark.parametrize("field", ["input", "output"])
def test_input_and_output_cannot_be_null(field: str) -> None:
    raw = config()
    entry(raw)[field] = None
    assert_rejected(validate(raw), "anthropic-direct", "claude-sonnet-5-5", field)


def test_unknown_pricing_mode_is_rejected() -> None:
    raw = config()
    entry(raw)["pricing_mode"] = "flat"
    assert_rejected(validate(raw), "anthropic-direct", "claude-sonnet-5-5", "pricing_mode")


@pytest.mark.parametrize("mode", [None, "metered"])
def test_all_zero_metered_entry_is_rejected(mode: str | None) -> None:
    raw = config()
    e = entry(raw)
    e.update(input=0, output=0, cache_read=0, cache_write_5m=0, cache_write_1h=None)
    if mode is None:
        del e["pricing_mode"]
    assert_rejected(validate(raw), "prices.anthropic-direct.claude-sonnet-5-5", "positive")


def test_free_entry_with_a_positive_rate_is_rejected() -> None:
    raw = config()
    entry(raw).update(
        pricing_mode="free", input=0, output=0, cache_read=0.3, cache_write_5m=0, cache_write_1h=0
    )
    assert_rejected(validate(raw), "prices.anthropic-direct.claude-sonnet-5-5", "free")


def test_price_for_an_unknown_deployment_is_rejected() -> None:
    raw = config()
    raw["prices"]["ghost"] = copy.deepcopy(raw["prices"]["deepseek"])
    assert_rejected(validate(raw), "ghost")


# --- aliases and secrets -----------------------------------------------------------------


def test_alias_to_an_unknown_deployment_is_rejected() -> None:
    raw = config()
    raw["aliases"]["claude-sonnet-5-5"] = ["anthropic-direct", "nowhere"]
    assert_rejected(validate(raw), "claude-sonnet-5-5", "nowhere")


def test_missing_price_for_a_deployment_alias_pair_is_rejected() -> None:
    raw = config()
    raw["aliases"]["claude-sonnet-5-5"] = ["anthropic-direct", "deepseek"]
    assert_rejected(validate(raw), "prices.deepseek.claude-sonnet-5-5")


def test_unset_api_key_env_names_the_variable_but_never_prints_values() -> None:
    env = {"DEEPSEEK_API_KEY": SECRET}
    text = assert_rejected(validate(config(), env), "anthropic-direct", "ANTHROPIC_API_KEY")
    assert SECRET not in text


def test_empty_api_key_env_counts_as_unset() -> None:
    env = {"ANTHROPIC_API_KEY": "", "DEEPSEEK_API_KEY": SECRET}
    assert_rejected(validate(config(), env), "ANTHROPIC_API_KEY")


def test_secret_check_can_be_skipped() -> None:
    assert validate(config(), {}, check_secrets=False).ok


def test_errors_never_echo_rejected_input_values() -> None:
    raw = config()
    entry(raw)["input"] = SECRET  # a secret pasted into the wrong place
    assert_rejected(validate(raw), "input")


# --- enforcement -------------------------------------------------------------------------


def test_invalid_enforcement_mode_is_rejected() -> None:
    raw = config(enforcement={"default": {"budgets": "warn"}})
    assert_rejected(validate(raw), "enforcement", "budgets")


def test_invalid_enforcement_policy_is_rejected() -> None:
    raw = config(enforcement={"default": {"detection": "observe"}})
    assert_rejected(validate(raw), "enforcement", "detection")


def test_invalid_team_enforcement_is_rejected() -> None:
    raw = config(teams={"research": {"budget_usd": 200, "enforcement": {"budgets": "off"}}})
    assert_rejected(validate(raw), "teams", "research")


def test_observe_modes_are_listed_in_notes() -> None:
    raw = config(
        enforcement={"default": {"allowlist": "observe"}},
        teams={
            "platform": {"budget_usd": 500, "period": "month"},
            "research": {"budget_usd": 200, "period": "day", "enforcement": {"budgets": "observe"}},
        },
    )
    report = validate(raw)
    assert report.ok, report.errors
    assert "enforcement.default.allowlist: observe" in report.notes
    assert "teams.research.enforcement.budgets: observe" in report.notes
    assert not any("platform" in note for note in report.notes)
    cfg = report.config
    assert cfg is not None
    assert cfg.enforcement_default[Policy.ALLOWLIST] is EnforcementMode.OBSERVE
    assert cfg.enforcement_default[Policy.BUDGETS] is EnforcementMode.ENFORCE  # omitted
    research = cfg.enforcement_for("research")
    assert research[Policy.BUDGETS] is EnforcementMode.OBSERVE
    assert cfg.enforcement_for("platform")[Policy.BUDGETS] is EnforcementMode.ENFORCE
    team = cfg.team("research")
    assert team is not None and team.budget is not None
    assert team.budget.limit == 200_000_000
    assert team.budget.cadence is Cadence.DAY


# --- budget mode, defaults, detection -----------------------------------------------------


def test_naive_budget_mode_requires_experimental() -> None:
    raw = config(budget={"mode": "naive"})
    assert_rejected(validate(raw), "budget.mode")
    report = validate(config(budget={"mode": "naive"}), experimental=True)
    assert report.ok
    assert report.config is not None
    assert report.config.budget_mode_naive is True


@pytest.mark.parametrize("field", ["tpm", "rpm"])
@pytest.mark.parametrize("value", [0, -1, True, 1.5, "60", None])
def test_default_limits_must_be_positive_integers(field: str, value: object) -> None:
    raw = config(defaults={field: value})
    assert_rejected(validate(raw), "defaults", field)


def test_explicit_default_limits_are_used() -> None:
    cfg = validate(config(defaults={"tpm": 50_000, "rpm": 10})).config
    assert cfg is not None
    assert (cfg.default_tpm, cfg.default_rpm) == (50_000, 10)


def test_detect_categories_default_to_flag() -> None:
    cfg = validate(config()).config
    assert cfg is not None
    assert cfg.detect.categories == {name: DetectMode.FLAG for name in CATEGORIES}


def test_partial_detect_categories_keep_flag_for_the_rest() -> None:
    raw = config(detect={"categories": {"private_key": "block", "card_number": "off"}})
    cfg = validate(raw).config
    assert cfg is not None
    expected = {name: DetectMode.FLAG for name in CATEGORIES}
    expected.update(private_key=DetectMode.BLOCK, card_number=DetectMode.OFF)
    assert cfg.detect.categories == expected


def test_unknown_detect_category_is_rejected() -> None:
    raw = config(detect={"categories": {"private_key": "block", "phone_number": "flag"}})
    assert_rejected(validate(raw), "detect.categories.phone_number")


def test_unknown_detect_mode_is_rejected() -> None:
    raw = config(detect={"categories": {"private_key": "redact"}})
    assert_rejected(validate(raw), "detect", "private_key")


def test_unknown_top_level_section_is_rejected() -> None:
    assert_rejected(validate(config(unexpected={"a": 1})), "unexpected")


# --- environment overrides ---------------------------------------------------------------


def test_environment_overrides_host_port_and_database() -> None:
    raw = config(
        server={"host": "127.0.0.1", "port": 8787}, storage={"sqlite_path": "./api-gateway.db"}
    )
    env = {
        **ENV,
        "API_GATEWAY_HOST": "0.0.0.0",
        "API_GATEWAY_PORT": "9000",
        "API_GATEWAY_DB": "/data/gate.db",
        "API_GATEWAY_REQUEST_LOG": "/data/log.jsonl",
    }
    cfg = validate(raw, env).config
    assert cfg is not None
    assert cfg.host == "0.0.0.0"
    assert cfg.port == 9000
    assert cfg.sqlite_path == "/data/gate.db"
    assert cfg.request_log == "/data/log.jsonl"


def test_file_values_apply_without_overrides() -> None:
    raw = config(server={"host": "10.0.0.5", "port": 1234})
    cfg = validate(raw).config
    assert cfg is not None
    assert (cfg.host, cfg.port) == ("10.0.0.5", 1234)


def test_non_numeric_port_override_is_rejected() -> None:
    env = {**ENV, "API_GATEWAY_PORT": "eighty"}
    assert_rejected(validate(config(), env), "port")

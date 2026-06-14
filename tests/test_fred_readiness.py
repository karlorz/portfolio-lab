"""Credential-policy matrix tests for src.monitor.fred_readiness."""

import json
import logging

import pytest

from src.monitor.fred_readiness import (
    FRED_READINESS_SCHEMA_VERSION,
    assess_fred_readiness,
    resolve_fred_operating_mode,
)


PUBLIC_READINESS_KEYS = {
    "schema_version",
    "status",
    "readiness",
    "ready",
    "blocking",
    "enforcement",
    "mode",
    "api_key_configured",
    "source_mode",
    "fred_cache_status",
    "reason",
    "message",
    "remediation",
}


def _health(*, api_key_configured: bool, status: str = "ok", source_mode: str = "cached"):
    return {
        "api_key_configured": api_key_configured,
        "status": status,
        "source_mode": source_mode,
    }


def test_resolve_fred_operating_mode_prefers_explicit_mode_over_environment():
    env = {
        "PORTFOLIO_LAB_MODE": "live",
        "ALPHALAB_MODE": "paper",
        "APP_MODE": "lab",
        "CRON_BACKEND": "tasker",
    }

    assert resolve_fred_operating_mode("dev", env=env) == "local"


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        ({"PORTFOLIO_LAB_MODE": "production"}, "live"),
        ({"ALPHALAB_MODE": "paper-trading"}, "paper"),
        ({"APP_MODE": "tests"}, "test"),
        ({"CRON_BACKEND": "tasker"}, "lab"),
        ({}, "local"),
    ],
)
def test_resolve_fred_operating_mode_uses_runtime_environment(env, expected):
    assert resolve_fred_operating_mode(env=env) == expected


@pytest.mark.parametrize("mode", ["local", "test", "lab", "paper", "staging", "live"])
def test_credential_present_passes_for_every_policy_mode(mode):
    readiness = assess_fred_readiness(
        _health(api_key_configured=True),
        mode=mode,
        env={"FRED_API_KEY": "real-key-must-not-leak"},
    )

    assert readiness["schema_version"] == FRED_READINESS_SCHEMA_VERSION
    assert set(readiness) == PUBLIC_READINESS_KEYS
    assert readiness["status"] == "ok"
    assert readiness["readiness"] == "pass"
    assert readiness["ready"] is True
    assert readiness["blocking"] is False
    assert readiness["mode"] == mode
    assert readiness["api_key_configured"] is True
    assert readiness["reason"] is None
    assert "real-key-must-not-leak" not in json.dumps(readiness)


@pytest.mark.parametrize("mode", ["local", "test"])
def test_credential_absent_permissive_modes_proceed_without_blocking(mode, caplog):
    readiness = assess_fred_readiness(
        _health(api_key_configured=False, status="unavailable", source_mode="synthetic"),
        mode=mode,
        env={},
    )

    assert readiness["status"] == "warning"
    assert readiness["readiness"] == "pass"
    assert readiness["ready"] is True
    assert readiness["blocking"] is False
    assert readiness["enforcement"] == "permissive"
    assert readiness["reason"] == "missing_fred_api_key"
    assert caplog.text == ""


@pytest.mark.parametrize("mode", ["lab", "paper", "staging"])
def test_credential_absent_warning_modes_proceed_and_log_warning(mode, caplog):
    with caplog.at_level(logging.WARNING, logger="src.monitor.fred_readiness"):
        readiness = assess_fred_readiness(
            _health(api_key_configured=False, status="ok", source_mode="cached"),
            mode=mode,
            env={},
        )

    assert readiness["status"] == "warning"
    assert readiness["readiness"] == "warn"
    assert readiness["ready"] is True
    assert readiness["blocking"] is False
    assert readiness["enforcement"] == "monitored"
    assert readiness["reason"] == "missing_fred_api_key"
    assert "FRED readiness warning" in caplog.text
    assert mode in caplog.text
    assert "missing_fred_api_key" in caplog.text


def test_credential_absent_live_mode_blocks_without_logging_secret(caplog):
    with caplog.at_level(logging.WARNING, logger="src.monitor.fred_readiness"):
        readiness = assess_fred_readiness(
            _health(api_key_configured=False, status="ok", source_mode="cached"),
            mode="live",
            env={"FRED_API_KEY": "real-key-must-not-leak"},
        )

    assert readiness["status"] == "critical"
    assert readiness["readiness"] == "fail"
    assert readiness["ready"] is False
    assert readiness["blocking"] is True
    assert readiness["enforcement"] == "required"
    assert readiness["reason"] == "missing_fred_api_key"
    assert "real-key-must-not-leak" not in json.dumps(readiness)
    assert "real-key-must-not-leak" not in caplog.text


def test_invalid_configured_key_blocks_live_without_leaking_secret():
    readiness = assess_fred_readiness(
        {
            "api_key_configured": True,
            "status": "unavailable",
            "source_mode": "synthetic",
            "reason": "unauthorized api_key",
        },
        mode="live",
        env={"FRED_API_KEY": "real-key-must-not-leak"},
    )

    assert readiness["status"] == "critical"
    assert readiness["readiness"] == "fail"
    assert readiness["blocking"] is True
    assert readiness["reason"] == "invalid_fred_api_key"
    assert "real-key-must-not-leak" not in json.dumps(readiness)


def test_public_return_shape_for_warning_payload():
    readiness = assess_fred_readiness(
        _health(api_key_configured=False, status="empty", source_mode="synthetic"),
        mode="paper",
        env={},
    )

    assert set(readiness) == PUBLIC_READINESS_KEYS
    assert readiness["schema_version"] == FRED_READINESS_SCHEMA_VERSION
    assert readiness["fred_cache_status"] == "empty"
    assert readiness["source_mode"] == "synthetic"

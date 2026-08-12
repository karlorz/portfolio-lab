#!/usr/bin/env python3
"""
Direct unit tests for ``src/strategy/health_gate_policy.py`` (64L governed
constants for advisory signal-health hard-zero decisions) — test file owed
by the TEST-GAP coverage gap (module has zero direct test references;
indirect coverage via test_generator only).

Pins the defaults (ADR-006 / 0.08 / 20), the env-override + invalid→fallback
behavior of ``unhealthy_min_ic`` and ``minimum_labeled_daily_cohorts``
(``max(1, int(...))`` floor), and the full ``disclosure()`` schema contract
(decision == "ADR-006", ``live_authoritative is False``).
"""
import pytest

from src.strategy.health_gate_policy import (
    DEFAULT_MIN_LABELED_DAILY_COHORTS,
    DEFAULT_UNHEALTHY_MIN_IC,
    HARD_ZERO_ADR_ID,
    disclosure,
    minimum_labeled_daily_cohorts,
    unhealthy_min_ic,
)


def test_defaults_pinned():
    """Governed constants — the public contract anchors."""
    assert HARD_ZERO_ADR_ID == "ADR-006"
    assert DEFAULT_UNHEALTHY_MIN_IC == 0.08
    assert DEFAULT_MIN_LABELED_DAILY_COHORTS == 20


def test_unhealthy_min_ic_default(monkeypatch):
    """No env override → default 0.08."""
    monkeypatch.delenv("ENSEMBLE_UNHEALTHY_MIN_IC", raising=False)
    assert unhealthy_min_ic() == 0.08


def test_unhealthy_min_ic_env_override(monkeypatch):
    """Valid env override wins."""
    monkeypatch.setenv("ENSEMBLE_UNHEALTHY_MIN_IC", "0.12")
    assert unhealthy_min_ic() == 0.12
    monkeypatch.setenv("ENSEMBLE_UNHEALTHY_MIN_IC", "0")
    assert unhealthy_min_ic() == 0.0


@pytest.mark.parametrize("bad", ["garbage", ""])
def test_unhealthy_min_ic_invalid_falls_back(monkeypatch, bad):
    """Invalid env value → default 0.08."""
    monkeypatch.setenv("ENSEMBLE_UNHEALTHY_MIN_IC", bad)
    assert unhealthy_min_ic() == 0.08


def test_minimum_labeled_daily_cohorts_default(monkeypatch):
    """No env override → default 20."""
    monkeypatch.delenv("ENSEMBLE_HARD_ZERO_MIN_COHORTS", raising=False)
    assert minimum_labeled_daily_cohorts() == 20


def test_minimum_labeled_daily_cohorts_env_override(monkeypatch):
    """Valid env override wins; floor is 1."""
    monkeypatch.setenv("ENSEMBLE_HARD_ZERO_MIN_COHORTS", "5")
    assert minimum_labeled_daily_cohorts() == 5
    monkeypatch.setenv("ENSEMBLE_HARD_ZERO_MIN_COHORTS", "0")
    assert minimum_labeled_daily_cohorts() == 1
    monkeypatch.setenv("ENSEMBLE_HARD_ZERO_MIN_COHORTS", "-3")
    assert minimum_labeled_daily_cohorts() == 1


def test_minimum_labeled_daily_cohorts_invalid_falls_back(monkeypatch):
    """Invalid env value → default 20."""
    monkeypatch.setenv("ENSEMBLE_HARD_ZERO_MIN_COHORTS", "garbage")
    assert minimum_labeled_daily_cohorts() == 20


def test_disclosure_schema_and_governance(monkeypatch):
    """Full disclosure contract: ADR-006, advisory-only, keys pinned."""
    monkeypatch.delenv("ENSEMBLE_UNHEALTHY_MIN_IC", raising=False)
    monkeypatch.delenv("ENSEMBLE_HARD_ZERO_MIN_COHORTS", raising=False)
    out = disclosure()

    assert set(out) == {
        "schema_version",
        "decision",
        "decision_path",
        "human_approved",
        "approved_at",
        "mode",
        "live_authoritative",
        "unhealthy_min_ic",
        "min_labeled_daily_cohorts",
        "shadow_collection",
        "target_allocations_unchanged",
        "reentry",
    }
    assert out["schema_version"] == "advisory-hard-zero-policy/v1"
    assert out["decision"] == "ADR-006"
    assert "2026-07-25-advisory-signal-hard-zero-policy.md" in out["decision_path"]
    assert out["human_approved"] is True
    assert out["approved_at"] == "2026-07-25"
    assert out["mode"] == "advisory_only"
    assert out["live_authoritative"] is False
    assert out["unhealthy_min_ic"] == 0.08
    assert out["min_labeled_daily_cohorts"] == 20
    assert out["shadow_collection"] is True
    assert out["target_allocations_unchanged"] is True
    assert "no auto-invert or live promotion" in out["reentry"]


def test_disclosure_reflects_env_overrides(monkeypatch):
    """Disclosure renders the live env-derived values, not just defaults."""
    monkeypatch.setenv("ENSEMBLE_UNHEALTHY_MIN_IC", "0.15")
    monkeypatch.setenv("ENSEMBLE_HARD_ZERO_MIN_COHORTS", "10")
    out = disclosure()
    assert out["unhealthy_min_ic"] == 0.15
    assert out["min_labeled_daily_cohorts"] == 10
    assert out["live_authoritative"] is False

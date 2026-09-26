"""Fail-closed ENABLE-style ensemble env flags: literal 1 only."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.strategy.ensemble_voter import EnsembleVoter, Regime, SignalSource, SignalReading


pytestmark = pytest.mark.usefixtures("_hermetic_health_db")

IC_ENV = "ENSEMBLE_USE_IC_WEIGHTS"
DISABLE_REGIME_ENV = "ENSEMBLE_DISABLE_REGIME_WEIGHTS"
TRUTHY_OFF = ("", "0", "true", "True", "yes", "on", "2", "1 ")


def _reading():
    return SignalReading(
        source=SignalSource.MULTI_SPEED_MOM,
        timestamp="2026-01-01",
        value=0.3,
        confidence=0.8,
        weight=0.0,
        regime_fit="all",
        asset_signals={"SPY": 0.3, "TLT": -0.1, "GLD": 0.0},
        explanation="test",
    )


@pytest.mark.parametrize("value", TRUTHY_OFF)
def test_ic_weights_off_unless_literal_one(tmp_path, monkeypatch, value):
    monkeypatch.setenv(IC_ENV, value)
    voter = EnsembleVoter(data_path=tmp_path)
    assert voter._use_ic_weights is False


def test_ic_weights_on_for_literal_one(tmp_path, monkeypatch):
    monkeypatch.setenv(IC_ENV, "1")
    voter = EnsembleVoter(data_path=tmp_path)
    assert voter._use_ic_weights is True


def test_ic_weights_default_off(tmp_path, monkeypatch):
    monkeypatch.delenv(IC_ENV, raising=False)
    voter = EnsembleVoter(data_path=tmp_path)
    assert voter._use_ic_weights is False


@pytest.mark.parametrize("value", TRUTHY_OFF)
def test_disable_regime_weights_off_unless_literal_one(tmp_path, monkeypatch, value):
    monkeypatch.setenv(DISABLE_REGIME_ENV, value)
    voter = EnsembleVoter(data_path=tmp_path)
    with patch.object(voter, "_apply_regime_weights", wraps=voter._apply_regime_weights) as mocked:
        voter.compute_vote(
            readings={SignalSource.MULTI_SPEED_MOM: _reading()},
            regime=Regime.NORMAL,
            regime_confidence=0.7,
        )
        assert mocked.called, value


def test_disable_regime_weights_skips_on_literal_one(tmp_path, monkeypatch):
    monkeypatch.setenv(DISABLE_REGIME_ENV, "1")
    voter = EnsembleVoter(data_path=tmp_path)
    with patch.object(voter, "_apply_regime_weights", wraps=voter._apply_regime_weights) as mocked:
        voter.compute_vote(
            readings={SignalSource.MULTI_SPEED_MOM: _reading()},
            regime=Regime.NORMAL,
            regime_confidence=0.7,
        )
        assert not mocked.called


def test_disable_regime_weights_default_applies(tmp_path, monkeypatch):
    monkeypatch.delenv(DISABLE_REGIME_ENV, raising=False)
    voter = EnsembleVoter(data_path=tmp_path)
    with patch.object(voter, "_apply_regime_weights", wraps=voter._apply_regime_weights) as mocked:
        voter.compute_vote(
            readings={SignalSource.MULTI_SPEED_MOM: _reading()},
            regime=Regime.NORMAL,
            regime_confidence=0.7,
        )
        assert mocked.called

#!/usr/bin/env python3
"""
Tests for vix_ensemble_adapter.py — VIXEnsembleStatus dataclass, constants,
defensive bias calculation, status flag logic, next action determination,
ensemble signal generation, and save/load.
"""
import sys
import os
import json
import importlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from pathlib import Path
from datetime import datetime
from unittest.mock import patch, MagicMock


@pytest.fixture(scope="module")
def adapter_module():
    """Import vix_ensemble_adapter with mocked vix_insurance_signal dependency."""
    mock_vix_signal = MagicMock()
    originals = {}
    mod_name = 'src.signals.vix_insurance_signal'
    originals[mod_name] = sys.modules.get(mod_name)
    sys.modules[mod_name] = mock_vix_signal
    sys.modules.pop('src.signals.vix_ensemble_adapter', None)

    import src.signals.vix_ensemble_adapter as mod
    yield mod, mock_vix_signal

    if originals[mod_name] is None:
        sys.modules.pop(mod_name, None)
    else:
        sys.modules[mod_name] = originals[mod_name]
    sys.modules.pop('src.signals.vix_ensemble_adapter', None)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_status(mod, **overrides):
    defaults = dict(
        timestamp=datetime.now().isoformat(),
        insurance_active=True,
        position_size_pct=0.01,
        days_to_expiry=30,
        unrealized_pnl_pct=25.0,
        cost_basis=1000.0,
        current_value=1250.0,
        budget_used_pct=0.40,
        defensive_bias=0.05,
        cash_buffer_increase=0.005,
        roll_pending=False,
        profit_opportunity=False,
        budget_exhausted=False,
        correlation_healthy=True,
        next_action="hold",
        action_urgency="routine",
    )
    defaults.update(overrides)
    return mod.VIXEnsembleStatus(**defaults)


def _make_mock_signal(**kwargs):
    mock = MagicMock()
    mock.position_active = kwargs.get('position_active', True)
    mock.position_cost_basis = kwargs.get('position_cost_basis', 1000)
    mock.position_current_value = kwargs.get('position_current_value', 1250)
    mock.portfolio_value = kwargs.get('portfolio_value', 100000)
    mock.days_to_position_expiry = kwargs.get('days_to_position_expiry', 30)
    mock.insurance_budget_ytd = kwargs.get('insurance_budget_ytd', 400)
    mock.spot_vix = kwargs.get('spot_vix', 18.0)
    mock.correlation_vix_spy = kwargs.get('correlation_vix_spy', -0.5)
    mock.signal_type = kwargs.get('signal_type', MagicMock())
    return mock


# ---------------------------------------------------------------------------
# VIXEnsembleStatus Tests
# ---------------------------------------------------------------------------

class TestVIXEnsembleStatus:

    def test_fields(self, adapter_module):
        mod, _ = adapter_module
        s = _make_status(mod)
        assert s.insurance_active is True
        assert s.position_size_pct == 0.01
        assert s.next_action == "hold"


# ---------------------------------------------------------------------------
# Constants Tests
# ---------------------------------------------------------------------------

class TestConstants:

    def test_defensive_bias_active(self, adapter_module):
        mod, _ = adapter_module
        assert mod.VIXEnsembleAdapter.DEFENSIVE_BIAS_ACTIVE == 0.05

    def test_defensive_bias_near_expiry(self, adapter_module):
        mod, _ = adapter_module
        assert mod.VIXEnsembleAdapter.DEFENSIVE_BIAS_NEAR_EXPIRY == 0.10

    def test_defensive_bias_profit(self, adapter_module):
        mod, _ = adapter_module
        assert mod.VIXEnsembleAdapter.DEFENSIVE_BIAS_PROFIT == -0.05

    def test_cash_buffer(self, adapter_module):
        mod, _ = adapter_module
        assert mod.VIXEnsembleAdapter.CASH_BUFFER_INCREASE == 0.005

    def test_thresholds(self, adapter_module):
        mod, _ = adapter_module
        assert mod.VIXEnsembleAdapter.BUDGET_WARNING_PCT == 0.80
        assert mod.VIXEnsembleAdapter.ROLL_WARNING_DTE == 7
        assert mod.VIXEnsembleAdapter.PROFIT_VIX_THRESHOLD == 35.0
        assert mod.VIXEnsembleAdapter.CORRELATION_BREAKDOWN_THRESHOLD == -0.3


# ---------------------------------------------------------------------------
# generate_status Tests
# ---------------------------------------------------------------------------

class TestGenerateStatus:

    def test_active_insurance(self, adapter_module):
        mod, _ = adapter_module
        adapter = mod.VIXEnsembleAdapter.__new__(mod.VIXEnsembleAdapter)
        adapter.signal_generator = MagicMock()
        adapter.signal_generator.generate_signal.return_value = _make_mock_signal()
        status = adapter.generate_status()
        assert status.insurance_active is True

    def test_inactive_insurance(self, adapter_module):
        mod, _ = adapter_module
        adapter = mod.VIXEnsembleAdapter.__new__(mod.VIXEnsembleAdapter)
        adapter.signal_generator = MagicMock()
        adapter.signal_generator.generate_signal.return_value = _make_mock_signal(
            position_active=False, position_cost_basis=0, position_current_value=0)
        status = adapter.generate_status()
        assert status.insurance_active is False
        assert status.defensive_bias == 0.0

    def test_near_expiry_higher_bias(self, adapter_module):
        mod, _ = adapter_module
        adapter = mod.VIXEnsembleAdapter.__new__(mod.VIXEnsembleAdapter)
        adapter.signal_generator = MagicMock()
        adapter.signal_generator.generate_signal.return_value = _make_mock_signal(
            days_to_position_expiry=15)
        status = adapter.generate_status()
        assert status.defensive_bias == mod.VIXEnsembleAdapter.DEFENSIVE_BIAS_NEAR_EXPIRY

    def test_roll_pending(self, adapter_module):
        mod, _ = adapter_module
        adapter = mod.VIXEnsembleAdapter.__new__(mod.VIXEnsembleAdapter)
        adapter.signal_generator = MagicMock()
        adapter.signal_generator.generate_signal.return_value = _make_mock_signal(
            days_to_position_expiry=5)
        status = adapter.generate_status()
        assert status.roll_pending is True

    def test_profit_opportunity(self, adapter_module):
        mod, _ = adapter_module
        adapter = mod.VIXEnsembleAdapter.__new__(mod.VIXEnsembleAdapter)
        adapter.signal_generator = MagicMock()
        adapter.signal_generator.generate_signal.return_value = _make_mock_signal(spot_vix=40.0)
        status = adapter.generate_status()
        assert status.profit_opportunity is True

    def test_budget_exhausted(self, adapter_module):
        mod, _ = adapter_module
        adapter = mod.VIXEnsembleAdapter.__new__(mod.VIXEnsembleAdapter)
        adapter.signal_generator = MagicMock()
        adapter.signal_generator.generate_signal.return_value = _make_mock_signal(insurance_budget_ytd=900)
        status = adapter.generate_status()
        assert status.budget_exhausted is True

    def test_next_action_profit_exit(self, adapter_module):
        mod, _ = adapter_module
        adapter = mod.VIXEnsembleAdapter.__new__(mod.VIXEnsembleAdapter)
        adapter.signal_generator = MagicMock()
        adapter.signal_generator.generate_signal.return_value = _make_mock_signal(spot_vix=40.0)
        status = adapter.generate_status()
        assert status.next_action == "exit"
        assert status.action_urgency == "immediate"

    def test_next_action_roll(self, adapter_module):
        mod, _ = adapter_module
        adapter = mod.VIXEnsembleAdapter.__new__(mod.VIXEnsembleAdapter)
        adapter.signal_generator = MagicMock()
        adapter.signal_generator.generate_signal.return_value = _make_mock_signal(
            days_to_position_expiry=5, spot_vix=18.0)
        status = adapter.generate_status()
        assert status.next_action == "roll"

    def test_next_action_hold(self, adapter_module):
        mod, _ = adapter_module
        adapter = mod.VIXEnsembleAdapter.__new__(mod.VIXEnsembleAdapter)
        adapter.signal_generator = MagicMock()
        adapter.signal_generator.generate_signal.return_value = _make_mock_signal(
            days_to_position_expiry=60, spot_vix=18.0)
        status = adapter.generate_status()
        assert status.next_action == "hold"


# ---------------------------------------------------------------------------
# get_ensemble_signal Tests
# ---------------------------------------------------------------------------

class TestGetEnsembleSignal:

    def test_returns_dict(self, adapter_module):
        mod, _ = adapter_module
        adapter = mod.VIXEnsembleAdapter.__new__(mod.VIXEnsembleAdapter)
        adapter.signal_generator = MagicMock()
        adapter.signal_generator.generate_signal.return_value = _make_mock_signal()
        signal = adapter.get_ensemble_signal()
        assert isinstance(signal, dict)
        assert "insurance_active" in signal
        assert "risk_score_adjustment" in signal

    def test_weight_adjustments_when_active(self, adapter_module):
        mod, _ = adapter_module
        adapter = mod.VIXEnsembleAdapter.__new__(mod.VIXEnsembleAdapter)
        adapter.signal_generator = MagicMock()
        adapter.signal_generator.generate_signal.return_value = _make_mock_signal(
            days_to_position_expiry=60)
        signal = adapter.get_ensemble_signal()
        assert 'hmm' in signal['weight_adjustments']
        assert 'cash' in signal['weight_adjustments']

    def test_alerts_roll_pending(self, adapter_module):
        mod, _ = adapter_module
        adapter = mod.VIXEnsembleAdapter.__new__(mod.VIXEnsembleAdapter)
        adapter.signal_generator = MagicMock()
        adapter.signal_generator.generate_signal.return_value = _make_mock_signal(
            days_to_position_expiry=5)
        signal = adapter.get_ensemble_signal()
        alert_types = [a['type'] for a in signal['alerts']]
        assert 'roll_needed' in alert_types


# ---------------------------------------------------------------------------
# Save/Load Tests
# ---------------------------------------------------------------------------

class TestSaveLoad:

    def test_save_creates_file(self, adapter_module, tmp_path):
        mod, _ = adapter_module
        adapter = mod.VIXEnsembleAdapter.__new__(mod.VIXEnsembleAdapter)
        adapter.status_file = tmp_path / "status.json"
        status = _make_status(mod)
        adapter.save_status(status)
        assert adapter.status_file.exists()

    def test_load_roundtrip(self, adapter_module, tmp_path):
        mod, _ = adapter_module
        adapter = mod.VIXEnsembleAdapter.__new__(mod.VIXEnsembleAdapter)
        adapter.status_file = tmp_path / "status.json"
        status = _make_status(mod)
        adapter.save_status(status)
        loaded = adapter.load_status()
        assert loaded is not None
        assert loaded.insurance_active == status.insurance_active

    def test_load_missing_file(self, adapter_module, tmp_path):
        mod, _ = adapter_module
        adapter = mod.VIXEnsembleAdapter.__new__(mod.VIXEnsembleAdapter)
        adapter.status_file = tmp_path / "nonexistent.json"
        assert adapter.load_status() is None

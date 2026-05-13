#!/usr/bin/env python3
"""
Tests for multi_strategy_adapters.py — adapter signal generation, portfolio signal
collection, signal clamping, and get_all_strategy_signals orchestrator.

Uses sys.modules mocking for heavy dependencies with proper cleanup.
"""
import sys
import os
import importlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# Module-scoped fixture that mocks heavy deps and imports the module
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def adapters_module():
    """Import multi_strategy_adapters with mocked dependencies."""
    mock_integrator = MagicMock()
    mock_ms = MagicMock()
    mock_rp = MagicMock()
    mock_nm = MagicMock()

    mock_ms.SPEED_TIERS = {"fast": {}, "medium": {}, "slow": {}}
    mock_ms.DEFAULT_BASE_ALLOCATION = {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16}
    mock_rp.DEFAULT_BASE = {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16, "CASH": 0.0}
    mock_nm.DEFAULT_BASE_ALLOCATION = {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16}

    originals = {}
    modules_to_mock = {
        'src.signals.integrator': mock_integrator,
        'src.signals.multi_speed_momentum': mock_ms,
        'src.strategy.risk_parity_weight_overlay': mock_rp,
        'src.strategy.network_momentum_leadlag': mock_nm,
    }

    for mod in modules_to_mock:
        originals[mod] = sys.modules.get(mod)

    for mod, mock in modules_to_mock.items():
        sys.modules[mod] = mock

    # Remove cached version if exists
    sys.modules.pop('src.signals.multi_strategy_adapters', None)

    import src.signals.multi_strategy_adapters as adapters
    yield adapters, mock_integrator

    # Cleanup: restore originals
    for mod, original in originals.items():
        if original is None:
            sys.modules.pop(mod, None)
        else:
            sys.modules[mod] = original
    sys.modules.pop('src.signals.multi_strategy_adapters', None)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ensemble_signal():
    mock = MagicMock()
    mock.fast_signal.signal = 0.5
    mock.medium_signal.signal = 0.6
    mock.slow_signal.signal = 0.4
    mock.ensemble_position = 0.5
    mock.ensemble_confidence = 0.8
    mock.target_weight = 0.55
    mock.timestamp = "2026-05-14"
    return mock


def _make_rp_allocation():
    mock = MagicMock()
    mock.rp_adjustments = {"SPY": 0.05, "GLD": -0.03, "TLT": -0.02}
    mock.risk_parity_score = 0.85
    mock.timestamp = "2026-05-14"
    mock.base_weights = {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16}
    mock.target_weights = {"SPY": 0.51, "GLD": 0.35, "TLT": 0.14}
    mock.asset_vols = {"SPY": 0.16, "GLD": 0.15, "TLT": 0.12}
    mock.raw_rp_weights = {"SPY": 0.40, "GLD": 0.42, "TLT": 0.18}
    mock.expected_vol = 0.14
    return mock


def _make_nm_ensemble():
    mock = MagicMock()
    mock.ensemble_momentum = 0.3
    mock.ensemble_confidence = 0.7
    mock.network_centrality = 0.5
    mock.leadership_score = 0.6
    mock.followership_score = 0.4
    mock.window_signals = [MagicMock(), MagicMock(), MagicMock()]
    mock.target_weight = 0.50
    mock.timestamp = "2026-05-14"
    return mock


# ---------------------------------------------------------------------------
# MultiSpeedSignalAdapter Tests
# ---------------------------------------------------------------------------

class TestMultiSpeedAdapter:

    def test_init(self, adapters_module):
        adapters, _ = adapters_module
        adapter = adapters.MultiSpeedSignalAdapter()
        assert adapter.source_type == "multi_speed"
        assert adapter.source_name == "manahl_multi_speed_ensemble"

    def test_generate_signal(self, adapters_module):
        adapters, mi = adapters_module
        adapter = adapters.MultiSpeedSignalAdapter()
        adapter.multi_speed = MagicMock()
        adapter.multi_speed.compute_ensemble_signal.return_value = _make_ensemble_signal()
        mi.SignalSourceResult = MagicMock
        signal = adapter.generate_signal("SPY")
        assert signal is not None

    def test_generate_signal_none(self, adapters_module):
        adapters, _ = adapters_module
        adapter = adapters.MultiSpeedSignalAdapter()
        adapter.multi_speed = MagicMock()
        adapter.multi_speed.compute_ensemble_signal.return_value = None
        signal = adapter.generate_signal("SPY")
        assert signal is None

    def test_get_portfolio_signals(self, adapters_module):
        adapters, mi = adapters_module
        adapter = adapters.MultiSpeedSignalAdapter()
        adapter.multi_speed = MagicMock()
        adapter.multi_speed.compute_ensemble_signal.return_value = _make_ensemble_signal()
        mi.SignalSourceResult = MagicMock
        signals = adapter.get_portfolio_signals(["SPY", "GLD"])
        assert isinstance(signals, dict)


# ---------------------------------------------------------------------------
# RiskParitySignalAdapter Tests
# ---------------------------------------------------------------------------

class TestRiskParityAdapter:

    def test_init(self, adapters_module):
        adapters, _ = adapters_module
        adapter = adapters.RiskParitySignalAdapter()
        assert adapter.source_type == "risk_parity"
        assert adapter.source_name == "bridgewater_rp_overlay"

    def test_generate_signal(self, adapters_module):
        adapters, mi = adapters_module
        adapter = adapters.RiskParitySignalAdapter()
        adapter.rp_overlay = MagicMock()
        adapter.rp_overlay.calculate_rp_overlay.return_value = _make_rp_allocation()
        mi.SignalSourceResult = MagicMock
        signal = adapter.generate_signal("SPY")
        assert signal is not None

    def test_generate_signal_none(self, adapters_module):
        adapters, _ = adapters_module
        adapter = adapters.RiskParitySignalAdapter()
        adapter.rp_overlay = MagicMock()
        adapter.rp_overlay.calculate_rp_overlay.return_value = None
        signal = adapter.generate_signal("SPY")
        assert signal is None

    def test_get_portfolio_signals(self, adapters_module):
        adapters, mi = adapters_module
        adapter = adapters.RiskParitySignalAdapter()
        adapter.rp_overlay = MagicMock()
        adapter.rp_overlay.calculate_rp_overlay.return_value = _make_rp_allocation()
        mi.SignalSourceResult = MagicMock
        signals = adapter.get_portfolio_signals(["SPY", "GLD", "TLT"])
        assert isinstance(signals, dict)


# ---------------------------------------------------------------------------
# NetworkMomentumSignalAdapter Tests
# ---------------------------------------------------------------------------

class TestNetworkMomentumAdapter:

    def test_init(self, adapters_module):
        adapters, _ = adapters_module
        adapter = adapters.NetworkMomentumSignalAdapter()
        assert adapter.source_type == "network_momentum"
        assert adapter.source_name == "imperial_network_momentum"

    def test_generate_signal(self, adapters_module):
        adapters, mi = adapters_module
        adapter = adapters.NetworkMomentumSignalAdapter()
        adapter.network_momentum = MagicMock()
        adapter.network_momentum.compute_ensemble_signal.return_value = _make_nm_ensemble()
        mock_leadlag = MagicMock()
        mock_leadlag.adjacency = {("SPY", "GLD"): 0.5}
        adapter.network_momentum.compute_leadlag_matrix.return_value = mock_leadlag
        mi.SignalSourceResult = MagicMock
        signal = adapter.generate_signal("SPY")
        assert signal is not None

    def test_generate_signal_none(self, adapters_module):
        adapters, _ = adapters_module
        adapter = adapters.NetworkMomentumSignalAdapter()
        adapter.network_momentum = MagicMock()
        adapter.network_momentum.compute_ensemble_signal.return_value = None
        signal = adapter.generate_signal("SPY")
        assert signal is None

    def test_dominant_leader(self, adapters_module):
        adapters, _ = adapters_module
        adapter = adapters.NetworkMomentumSignalAdapter()
        mock_leadlag = MagicMock()
        mock_leadlag.adjacency = {
            ("SPY", "GLD"): 0.5,
            ("SPY", "TLT"): 0.3,
            ("GLD", "TLT"): 0.1,
        }
        leader = adapter._get_dominant_leader(mock_leadlag)
        assert leader == "SPY"

    def test_get_portfolio_signals(self, adapters_module):
        adapters, mi = adapters_module
        adapter = adapters.NetworkMomentumSignalAdapter()
        adapter.network_momentum = MagicMock()
        adapter.network_momentum.compute_ensemble_signal.return_value = _make_nm_ensemble()
        mock_leadlag = MagicMock()
        mock_leadlag.adjacency = {}
        adapter.network_momentum.compute_leadlag_matrix.return_value = mock_leadlag
        mi.SignalSourceResult = MagicMock
        signals = adapter.get_portfolio_signals(["SPY", "GLD", "TLT"])
        assert isinstance(signals, dict)


# ---------------------------------------------------------------------------
# get_all_strategy_signals Tests
# ---------------------------------------------------------------------------

class TestGetAllSignals:

    def test_returns_three_strategies(self, adapters_module):
        adapters, _ = adapters_module
        with patch.object(adapters.MultiSpeedSignalAdapter, 'get_portfolio_signals', return_value={}):
            with patch.object(adapters.RiskParitySignalAdapter, 'get_portfolio_signals', return_value={}):
                with patch.object(adapters.NetworkMomentumSignalAdapter, 'get_portfolio_signals', return_value={}):
                    result = adapters.get_all_strategy_signals(["SPY"])
                    assert "multi_speed" in result
                    assert "risk_parity" in result
                    assert "network_momentum" in result

    def test_default_tickers(self, adapters_module):
        adapters, _ = adapters_module
        with patch.object(adapters.MultiSpeedSignalAdapter, 'get_portfolio_signals', return_value={}) as ms:
            with patch.object(adapters.RiskParitySignalAdapter, 'get_portfolio_signals', return_value={}):
                with patch.object(adapters.NetworkMomentumSignalAdapter, 'get_portfolio_signals', return_value={}):
                    adapters.get_all_strategy_signals()
                    ms.assert_called_once_with(["SPY", "GLD", "TLT"])

#!/usr/bin/env python3
"""
Tests for tsmom_integration.py — TSMOMSignalAdapter confidence calculation,
signal generation, portfolio signals, allocation deltas, and convenience function.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from unittest.mock import patch, MagicMock


@pytest.fixture(scope="module")
def tsmom_module():
    """Import tsmom_integration with mocked dependencies."""
    mock_tsmom = MagicMock()
    mock_integrator = MagicMock()
    mock_tsmom.DEFAULT_BASE_ALLOCATION = {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16}

    originals = {}
    for mod in ['src.signals.tsmom_overlay', 'src.signals.integrator']:
        originals[mod] = sys.modules.get(mod)
    sys.modules['src.signals.tsmom_overlay'] = mock_tsmom
    sys.modules['src.signals.integrator'] = mock_integrator
    sys.modules.pop('src.signals.tsmom_integration', None)

    import src.signals.tsmom_integration as mod
    yield mod, mock_tsmom, mock_integrator

    for mod_name, orig in originals.items():
        if orig is None:
            sys.modules.pop(mod_name, None)
        else:
            sys.modules[mod_name] = orig
    sys.modules.pop('src.signals.tsmom_integration', None)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tsmom_signal(**overrides):
    mock = MagicMock()
    mock.signal = overrides.get('signal', 1)
    mock.lookback_return = overrides.get('lookback_return', 0.15)
    mock.realized_vol = overrides.get('realized_vol', 0.18)
    mock.vol_scaled_position = overrides.get('vol_scaled_position', 0.55)
    mock.base_weight = overrides.get('base_weight', 0.46)
    mock.adjustment = overrides.get('adjustment', 0.05)
    mock.target_weight = overrides.get('target_weight', 0.51)
    mock.timestamp = overrides.get('timestamp', '2026-05-14')
    return mock


# ---------------------------------------------------------------------------
# _compute_confidence Tests
# ---------------------------------------------------------------------------

class TestComputeConfidence:

    def test_base_confidence(self, tsmom_module):
        mod, _, _ = tsmom_module
        adapter = mod.TSMOMSignalAdapter.__new__(mod.TSMOMSignalAdapter)
        signal = _make_tsmom_signal(lookback_return=0.0, realized_vol=0.18, signal=0)
        conf = adapter._compute_confidence(signal)
        assert conf >= 0.50

    def test_strong_trend_higher(self, tsmom_module):
        mod, _, _ = tsmom_module
        adapter = mod.TSMOMSignalAdapter.__new__(mod.TSMOMSignalAdapter)
        weak = _make_tsmom_signal(lookback_return=0.02, realized_vol=0.18, signal=1)
        strong = _make_tsmom_signal(lookback_return=0.20, realized_vol=0.18, signal=1)
        assert adapter._compute_confidence(strong) > adapter._compute_confidence(weak)

    def test_low_vol_higher(self, tsmom_module):
        mod, _, _ = tsmom_module
        adapter = mod.TSMOMSignalAdapter.__new__(mod.TSMOMSignalAdapter)
        high_vol = _make_tsmom_signal(lookback_return=0.10, realized_vol=0.25, signal=1)
        low_vol = _make_tsmom_signal(lookback_return=0.10, realized_vol=0.10, signal=1)
        assert adapter._compute_confidence(low_vol) > adapter._compute_confidence(high_vol)

    def test_signal_clarity_boost(self, tsmom_module):
        mod, _, _ = tsmom_module
        adapter = mod.TSMOMSignalAdapter.__new__(mod.TSMOMSignalAdapter)
        no_signal = _make_tsmom_signal(lookback_return=0.10, realized_vol=0.18, signal=0)
        has_signal = _make_tsmom_signal(lookback_return=0.10, realized_vol=0.18, signal=1)
        assert adapter._compute_confidence(has_signal) > adapter._compute_confidence(no_signal)

    def test_capped_at_one(self, tsmom_module):
        mod, _, _ = tsmom_module
        adapter = mod.TSMOMSignalAdapter.__new__(mod.TSMOMSignalAdapter)
        signal = _make_tsmom_signal(lookback_return=0.50, realized_vol=0.05, signal=1)
        assert adapter._compute_confidence(signal) <= 1.0

    def test_negative_return_same(self, tsmom_module):
        mod, _, _ = tsmom_module
        adapter = mod.TSMOMSignalAdapter.__new__(mod.TSMOMSignalAdapter)
        pos = _make_tsmom_signal(lookback_return=0.15, realized_vol=0.18, signal=1)
        neg = _make_tsmom_signal(lookback_return=-0.15, realized_vol=0.18, signal=-1)
        assert adapter._compute_confidence(pos) == adapter._compute_confidence(neg)


# ---------------------------------------------------------------------------
# get_signal Tests
# ---------------------------------------------------------------------------

class TestGetSignal:

    def test_returns_signal(self, tsmom_module):
        mod, _, mi = tsmom_module
        adapter = mod.TSMOMSignalAdapter.__new__(mod.TSMOMSignalAdapter)
        adapter.overlay = MagicMock()
        adapter.overlay.compute_signal.return_value = _make_tsmom_signal()
        mi.SignalSourceResult = MagicMock
        assert adapter.get_signal("SPY") is not None

    def test_none_when_no_signal(self, tsmom_module):
        mod, _, _ = tsmom_module
        adapter = mod.TSMOMSignalAdapter.__new__(mod.TSMOMSignalAdapter)
        adapter.overlay = MagicMock()
        adapter.overlay.compute_signal.return_value = None
        assert adapter.get_signal("SPY") is None


# ---------------------------------------------------------------------------
# get_portfolio_signals Tests
# ---------------------------------------------------------------------------

class TestGetPortfolioSignals:

    def test_returns_dict(self, tsmom_module):
        mod, _, mi = tsmom_module
        adapter = mod.TSMOMSignalAdapter.__new__(mod.TSMOMSignalAdapter)
        adapter.overlay = MagicMock()
        adapter.overlay.compute_signal.return_value = _make_tsmom_signal()
        mi.SignalSourceResult = MagicMock
        signals = adapter.get_portfolio_signals(["SPY", "GLD"])
        assert isinstance(signals, dict)

    def test_skips_none(self, tsmom_module):
        mod, _, mi = tsmom_module
        adapter = mod.TSMOMSignalAdapter.__new__(mod.TSMOMSignalAdapter)
        adapter.overlay = MagicMock()
        adapter.overlay.compute_signal.side_effect = [_make_tsmom_signal(), None]
        mi.SignalSourceResult = MagicMock
        signals = adapter.get_portfolio_signals(["SPY", "GLD"])
        assert "SPY" in signals
        assert "GLD" not in signals


# ---------------------------------------------------------------------------
# get_allocation_deltas Tests
# ---------------------------------------------------------------------------

class TestGetAllocationDeltas:

    def test_returns_dict(self, tsmom_module):
        mod, _, _ = tsmom_module
        adapter = mod.TSMOMSignalAdapter.__new__(mod.TSMOMSignalAdapter)
        adapter.overlay = MagicMock()
        adapter.overlay.compute_signal.return_value = _make_tsmom_signal(adjustment=0.05)
        deltas = adapter.get_allocation_deltas(["SPY"])
        assert isinstance(deltas, dict)

    def test_delta_from_signal(self, tsmom_module):
        mod, _, _ = tsmom_module
        adapter = mod.TSMOMSignalAdapter.__new__(mod.TSMOMSignalAdapter)
        adapter.overlay = MagicMock()
        adapter.overlay.compute_signal.return_value = _make_tsmom_signal(adjustment=0.07)
        assert adapter.get_allocation_deltas(["SPY"])["SPY"] == 0.07

    def test_zero_when_no_signal(self, tsmom_module):
        mod, _, _ = tsmom_module
        adapter = mod.TSMOMSignalAdapter.__new__(mod.TSMOMSignalAdapter)
        adapter.overlay = MagicMock()
        adapter.overlay.compute_signal.return_value = None
        assert adapter.get_allocation_deltas(["SPY"])["SPY"] == 0.0


# ---------------------------------------------------------------------------
# get_tsmom_integrator_result Tests
# ---------------------------------------------------------------------------

class TestGetIntegratorResult:

    def test_calls_adapter(self, tsmom_module):
        mod, _, _ = tsmom_module
        with patch.object(mod.TSMOMSignalAdapter, 'get_portfolio_signals', return_value={}) as mock:
            mod.get_tsmom_integrator_result(["SPY"])
            mock.assert_called_once_with(["SPY"])

    def test_default_tickers(self, tsmom_module):
        mod, _, _ = tsmom_module
        with patch.object(mod.TSMOMSignalAdapter, 'get_portfolio_signals', return_value={}) as mock:
            mod.get_tsmom_integrator_result()
            mock.assert_called_once_with(["SPY", "GLD", "TLT"])

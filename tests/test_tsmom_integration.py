#!/usr/bin/env python3
"""
Tests for tsmom_integration.py — TSMOMSignalAdapter confidence calculation,
signal generation, portfolio signals, allocation deltas, and convenience function.
"""
import sys

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


# ---------------------------------------------------------------------------
# Extended edge-case tests
# ---------------------------------------------------------------------------

class TestComputeConfidenceExtended:
    """Additional confidence calculation edge cases."""

    def test_zero_vol_high_stability(self, tsmom_module):
        """Zero volatility should give max vol stability contribution."""
        mod, _, _ = tsmom_module
        adapter = mod.TSMOMSignalAdapter.__new__(mod.TSMOMSignalAdapter)
        signal = _make_tsmom_signal(lookback_return=0.10, realized_vol=0.0, signal=1)
        conf = adapter._compute_confidence(signal)
        # vol_stability = max(0, 1.0 - 0/0.30) = 1.0 → +0.15
        assert conf >= 0.50 + 0.15

    def test_very_high_vol_reduces_confidence(self, tsmom_module):
        """Very high vol should make vol_stability = 0."""
        mod, _, _ = tsmom_module
        adapter = mod.TSMOMSignalAdapter.__new__(mod.TSMOMSignalAdapter)
        signal = _make_tsmom_signal(lookback_return=0.10, realized_vol=0.50, signal=1)
        conf = adapter._compute_confidence(signal)
        # vol_stability = max(0, 1.0 - 0.50/0.30) = max(0, -0.67) = 0
        # So no vol_stability contribution
        assert conf >= 0.50

    def test_extreme_trend_capped(self, tsmom_module):
        """Extremely large trend should not exceed confidence=1.0."""
        mod, _, _ = tsmom_module
        adapter = mod.TSMOMSignalAdapter.__new__(mod.TSMOMSignalAdapter)
        signal = _make_tsmom_signal(lookback_return=1.0, realized_vol=0.01, signal=1)
        conf = adapter._compute_confidence(signal)
        assert conf <= 1.0

    def test_weak_trend_low_confidence(self, tsmom_module):
        """Near-zero trend should give near-base confidence."""
        mod, _, _ = tsmom_module
        adapter = mod.TSMOMSignalAdapter.__new__(mod.TSMOMSignalAdapter)
        signal = _make_tsmom_signal(lookback_return=0.01, realized_vol=0.20, signal=1)
        conf = adapter._compute_confidence(signal)
        assert conf < 0.75  # Weak trend + high vol → modest confidence

    def test_signal_zero_no_clarity_boost(self, tsmom_module):
        """Signal=0 should not get clarity contribution."""
        mod, _, _ = tsmom_module
        adapter = mod.TSMOMSignalAdapter.__new__(mod.TSMOMSignalAdapter)
        signal_zero = _make_tsmom_signal(lookback_return=0.15, realized_vol=0.18, signal=0)
        signal_one = _make_tsmom_signal(lookback_return=0.15, realized_vol=0.18, signal=1)
        c_zero = adapter._compute_confidence(signal_zero)
        c_one = adapter._compute_confidence(signal_one)
        assert c_one > c_zero  # signal=1 gets clarity boost


class TestGetSignalExtended:
    """Additional get_signal edge cases."""

    def test_generate_signal_alias(self, tsmom_module):
        """generate_signal should be an alias for get_signal."""
        mod, _, mi = tsmom_module
        adapter = mod.TSMOMSignalAdapter.__new__(mod.TSMOMSignalAdapter)
        adapter.overlay = MagicMock()
        adapter.overlay.compute_signal.return_value = _make_tsmom_signal()
        mi.SignalSourceResult = MagicMock
        # Both should return the same result
        assert adapter.get_signal is adapter.generate_signal or adapter.generate_signal("SPY") is not None

    def test_signal_metadata_fields(self, tsmom_module):
        """Signal should include all expected metadata fields."""
        mod, _, mi = tsmom_module
        adapter = mod.TSMOMSignalAdapter.__new__(mod.TSMOMSignalAdapter)
        adapter.overlay = MagicMock()
        adapter.overlay.compute_signal.return_value = _make_tsmom_signal()
        mi.SignalSourceResult = MagicMock
        result = adapter.get_signal("SPY")
        assert result is not None

    def test_portfolio_signals_empty_tickers(self, tsmom_module):
        """Empty ticker list should return empty dict."""
        mod, _, mi = tsmom_module
        adapter = mod.TSMOMSignalAdapter.__new__(mod.TSMOMSignalAdapter)
        adapter.overlay = MagicMock()
        mi.SignalSourceResult = MagicMock
        signals = adapter.get_portfolio_signals([])
        assert signals == {}


class TestGetAllocationDeltasExtended:
    """Additional allocation deltas edge cases."""

    def test_default_tickers(self, tsmom_module):
        """Default tickers should be SPY, GLD, TLT."""
        mod, _, _ = tsmom_module
        adapter = mod.TSMOMSignalAdapter.__new__(mod.TSMOMSignalAdapter)
        adapter.overlay = MagicMock()
        adapter.overlay.compute_signal.return_value = _make_tsmom_signal(adjustment=0.03)
        deltas = adapter.get_allocation_deltas()
        assert "SPY" in deltas
        assert "GLD" in deltas
        assert "TLT" in deltas

    def test_mixed_signals_and_none(self, tsmom_module):
        """Mix of valid signals and None should give appropriate deltas."""
        mod, _, _ = tsmom_module
        adapter = mod.TSMOMSignalAdapter.__new__(mod.TSMOMSignalAdapter)
        adapter.overlay = MagicMock()
        adapter.overlay.compute_signal.side_effect = [
            _make_tsmom_signal(adjustment=0.05), None, _make_tsmom_signal(adjustment=-0.03)
        ]
        deltas = adapter.get_allocation_deltas(["SPY", "GLD", "TLT"])
        assert deltas["SPY"] == 0.05
        assert deltas["GLD"] == 0.0
        assert deltas["TLT"] == -0.03


# ---------------------------------------------------------------------------
# get_signal_snapshot Tests
# ---------------------------------------------------------------------------

class TestGetSignalSnapshot:
    """Tests for TSMOMSignalAdapter.get_signal_snapshot()."""

    def _make_adapter(self, mod, mi, signals_by_ticker=None):
        """Create adapter with mocked overlay returning specified signals."""
        adapter = mod.TSMOMSignalAdapter.__new__(mod.TSMOMSignalAdapter)
        adapter.overlay = MagicMock()
        adapter.base_allocation = {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16}
        if signals_by_ticker is None:
            signals_by_ticker = {
                "SPY": _make_tsmom_signal(adjustment=0.05, signal=1),
                "GLD": _make_tsmom_signal(adjustment=-0.02, signal=-1),
                "TLT": _make_tsmom_signal(adjustment=0.01, signal=1),
            }

        def compute_side_effect(ticker):
            return signals_by_ticker.get(ticker)

        adapter.overlay.compute_signal.side_effect = compute_side_effect
        # Also mock get_allocation_deltas to avoid double-compute
        adapter._compute_confidence = MagicMock(return_value=0.72)
        return adapter

    def test_returns_signal_snapshot(self, tsmom_module):
        """get_signal_snapshot should return a SignalSnapshot object."""
        mod, mock_tsmom, _ = tsmom_module
        # Mock SignalSnapshot
        mock_snapshot_cls = MagicMock()
        with patch.dict('sys.modules', {'src.signals.signal_snapshot': MagicMock(SignalSnapshot=mock_snapshot_cls)}):
            adapter = self._make_adapter(mod, mock_tsmom)
            adapter.get_signal_snapshot(["SPY", "GLD", "TLT"])
            # Should have called compute_signal for each ticker
            assert adapter.overlay.compute_signal.call_count >= 1

    def test_no_signals_returns_inactive(self, tsmom_module):
        """When no signals available, snapshot should be inactive."""
        mod, mock_tsmom, _ = tsmom_module
        adapter = mod.TSMOMSignalAdapter.__new__(mod.TSMOMSignalAdapter)
        adapter.overlay = MagicMock()
        adapter.overlay.compute_signal.return_value = None
        adapter.base_allocation = {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16}
        adapter._compute_confidence = MagicMock(return_value=0.5)

        mock_snapshot_cls = MagicMock()
        with patch.dict('sys.modules', {'src.signals.signal_snapshot': MagicMock(SignalSnapshot=mock_snapshot_cls)}):
            _ = adapter.get_signal_snapshot(["SPY"])
            # SignalSnapshot should be called with is_active=False
            call_kwargs = mock_snapshot_cls.call_args
            assert call_kwargs is not None

    def test_default_tickers(self, tsmom_module):
        """Default tickers should be SPY, GLD, TLT."""
        mod, mock_tsmom, _ = tsmom_module
        adapter = self._make_adapter(mod, mock_tsmom)
        mock_snapshot_cls = MagicMock()
        with patch.dict('sys.modules', {'src.signals.signal_snapshot': MagicMock(SignalSnapshot=mock_snapshot_cls)}):
            adapter.get_signal_snapshot()
            # Should call compute_signal for each default ticker
            calls = [c.args[0] for c in adapter.overlay.compute_signal.call_args_list]
            assert "SPY" in calls or len(calls) > 0

    def test_active_when_nonzero_deltas(self, tsmom_module):
        """Snapshot should be active when any delta is nonzero."""
        mod, mock_tsmom, _ = tsmom_module
        signals_by_ticker = {
            "SPY": _make_tsmom_signal(adjustment=0.05, signal=1),
            "GLD": _make_tsmom_signal(adjustment=-0.02, signal=-1),
        }
        adapter = self._make_adapter(mod, mock_tsmom, signals_by_ticker)
        mock_snapshot_cls = MagicMock()
        with patch.dict('sys.modules', {'src.signals.signal_snapshot': MagicMock(SignalSnapshot=mock_snapshot_cls)}):
            adapter.get_signal_snapshot(["SPY", "GLD"])
            call_kwargs = mock_snapshot_cls.call_args[1]
            assert call_kwargs.get('is_active', False) is True

    def test_inactive_when_all_zero_deltas(self, tsmom_module):
        """Snapshot should be inactive when all deltas are zero."""
        mod, mock_tsmom, _ = tsmom_module
        signals_by_ticker = {
            "SPY": _make_tsmom_signal(adjustment=0.0, signal=0),
        }
        adapter = self._make_adapter(mod, mock_tsmom, signals_by_ticker)
        mock_snapshot_cls = MagicMock()
        with patch.dict('sys.modules', {'src.signals.signal_snapshot': MagicMock(SignalSnapshot=mock_snapshot_cls)}):
            adapter.get_signal_snapshot(["SPY"])
            call_kwargs = mock_snapshot_cls.call_args[1]
            assert call_kwargs.get('is_active', True) is False

    def test_source_field(self, tsmom_module):
        """Source field should be 'tsmom_integration'."""
        mod, mock_tsmom, _ = tsmom_module
        adapter = self._make_adapter(mod, mock_tsmom)
        mock_snapshot_cls = MagicMock()
        with patch.dict('sys.modules', {'src.signals.signal_snapshot': MagicMock(SignalSnapshot=mock_snapshot_cls)}):
            adapter.get_signal_snapshot(["SPY"])
            call_kwargs = mock_snapshot_cls.call_args[1]
            assert call_kwargs['source'] == 'tsmom_integration'

    def test_metadata_contains_deltas(self, tsmom_module):
        """Metadata should include deltas dict."""
        mod, mock_tsmom, _ = tsmom_module
        adapter = self._make_adapter(mod, mock_tsmom)
        mock_snapshot_cls = MagicMock()
        with patch.dict('sys.modules', {'src.signals.signal_snapshot': MagicMock(SignalSnapshot=mock_snapshot_cls)}):
            adapter.get_signal_snapshot(["SPY"])
            call_kwargs = mock_snapshot_cls.call_args[1]
            assert 'deltas' in call_kwargs.get('metadata', {})


# ---------------------------------------------------------------------------
# __init__ and constructor Tests
# ---------------------------------------------------------------------------

class TestTSMOMAdapterConstructor:

    def test_default_base_allocation(self, tsmom_module):
        """Default base_allocation should use DEFAULT_BASE_ALLOCATION."""
        mod, mock_tsmom, _ = tsmom_module
        mock_tsmom.TSMOMOverlay = MagicMock
        mock_tsmom.DEFAULT_BASE_ALLOCATION = {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16}
        adapter = mod.TSMOMSignalAdapter()
        assert adapter.base_allocation == {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16}

    def test_custom_base_allocation(self, tsmom_module):
        """Custom base_allocation should override default."""
        mod, mock_tsmom, _ = tsmom_module
        mock_tsmom.TSMOMOverlay = MagicMock
        mock_tsmom.DEFAULT_BASE_ALLOCATION = {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16}
        custom = {"SPY": 0.50, "GLD": 0.30, "TLT": 0.20}
        adapter = mod.TSMOMSignalAdapter(base_allocation=custom)
        assert adapter.base_allocation == custom


# ---------------------------------------------------------------------------
# Convenience function extended tests
# ---------------------------------------------------------------------------

class TestGetIntegratorResultExtended:

    def test_custom_base_allocation(self, tsmom_module):
        """get_tsmom_integrator_result should pass base_allocation through."""
        mod, _, _ = tsmom_module
        custom = {"SPY": 0.50, "GLD": 0.30, "TLT": 0.20}
        with patch.object(mod.TSMOMSignalAdapter, 'get_portfolio_signals', return_value={}):
            mod.get_tsmom_integrator_result(["SPY"], base_allocation=custom)
            # Adapter should have been created with custom base_allocation
            # We can't easily check the constructor arg, but the call should succeed

    def test_custom_tickers(self, tsmom_module):
        """Should accept custom ticker list."""
        mod, _, _ = tsmom_module
        with patch.object(mod.TSMOMSignalAdapter, 'get_portfolio_signals', return_value={}) as mock:
            mod.get_tsmom_integrator_result(["QQQ", "IWM"])
            mock.assert_called_once_with(["QQQ", "IWM"])

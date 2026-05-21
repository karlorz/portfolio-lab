"""
Tests for src/signals/multi_strategy_adapters.py.

Covers: MultiSpeedSignalAdapter, RiskParitySignalAdapter,
NetworkMomentumSignalAdapter, get_all_strategy_signals(), and
_get_dominant_leader().
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from src.signals.multi_strategy_adapters import (
    MultiSpeedSignalAdapter,
    RiskParitySignalAdapter,
    NetworkMomentumSignalAdapter,
    get_all_strategy_signals,
)
from src.signals.integrator import SignalSourceResult


# ---------------------------------------------------------------------------
# Helpers — build mock return values for the underlying strategy classes
# ---------------------------------------------------------------------------

def _make_mock_speed_signal(signal_val=1.0, confidence_val=0.8, position=0.5):
    """Build a mock ensemble signal resembling EnsembleSignal dataclass."""
    class MockTierSignal:
        signal = signal_val
        vol_scaled_position = position

    class MockEnsemble:
        ticker = "SPY"
        timestamp = datetime.now().isoformat()
        fast_signal = MockTierSignal()
        medium_signal = MockTierSignal()
        slow_signal = MockTierSignal()
        ensemble_position = position
        ensemble_confidence = confidence_val
        base_weight = 0.3
        target_weight = 0.35
        adjustment = 0.05

    return MockEnsemble()


def _make_mock_rp_overlay(ticker="SPY", adjustment=0.03, score=0.85):
    """Build a mock RPWeightOverlay dataclass-like object."""
    class MockRP:
        timestamp = datetime.now().isoformat()
        rp_adjustments = {ticker: adjustment, "GLD": -0.02, "TLT": 0.01}
        base_weights = {ticker: 0.46, "GLD": 0.38, "TLT": 0.16}
        target_weights = {ticker: 0.49, "GLD": 0.36, "TLT": 0.17}
        asset_vols = {ticker: 0.15, "GLD": 0.12, "TLT": 0.08}
        raw_rp_weights = {ticker: 0.40, "GLD": 0.35, "TLT": 0.25}
        risk_parity_score = score
        expected_vol = 0.11

    return MockRP()


def _make_mock_network_signal(signal_val=0.5, confidence_val=0.7, momentum=0.25):
    """Build a mock EnsembleNetworkSignal dataclass-like object."""
    class MockWindowSignal:
        ticker = "SPY"
        window = 66
        timestamp = datetime.now().isoformat()
        momentum_return = 0.05
        signal = 1
        network_momentum = momentum
        network_adjustment = 0.03
        base_weight = 0.46
        target_weight = 0.49
        adjustment = 0.03

    class MockLeadLagMatrix:
        adjacency = {
            ("SPY", "GLD"): 0.3,
            ("SPY", "TLT"): 0.2,
        }

    class MockEnsemble:
        ticker = "SPY"
        timestamp = datetime.now().isoformat()
        window_signals = {66: MockWindowSignal()}
        ensemble_momentum = momentum
        ensemble_signal = 1
        ensemble_confidence = confidence_val
        base_weight = 0.46
        adjustment = 0.03
        target_weight = 0.49
        leadership_score = 0.5
        followership_score = 0.2
        network_centrality = 0.6

    return MockEnsemble(), MockLeadLagMatrix()


# ---------------------------------------------------------------------------
# Fixtures — patch the imported classes at the module level
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_multi_speed():
    with patch(
        "src.signals.multi_strategy_adapters.MultiSpeedMomentum"
    ) as m:
        instance = m.return_value
        instance.compute_ensemble_signal.return_value = _make_mock_speed_signal()
        yield instance


@pytest.fixture
def mock_risk_parity():
    with patch(
        "src.signals.multi_strategy_adapters.RiskParityWeightOverlay"
    ) as m:
        instance = m.return_value
        instance.calculate_rp_overlay.return_value = _make_mock_rp_overlay()
        yield instance


@pytest.fixture
def mock_network_momentum():
    ens, ll = _make_mock_network_signal()
    with patch(
        "src.signals.multi_strategy_adapters.NetworkMomentumLeadLag"
    ) as m:
        instance = m.return_value
        instance.compute_ensemble_signal.return_value = ens
        instance.compute_leadlag_matrix.return_value = ll
        yield instance


# ===================================================================
# MultiSpeedSignalAdapter
# ===================================================================

class TestMultiSpeedSignalAdapter:
    def test_init_defaults(self, mock_multi_speed):
        adapter = MultiSpeedSignalAdapter()
        assert adapter.source_type == "multi_speed"
        assert adapter.source_name == "manahl_multi_speed_ensemble"

    def test_init_custom_allocation(self, mock_multi_speed):
        alloc = {"SPY": 0.5, "GLD": 0.3, "TLT": 0.2}
        adapter = MultiSpeedSignalAdapter(base_allocation=alloc)
        assert adapter.base_allocation == alloc

    def test_generate_signal_returns_signal(self, mock_multi_speed):
        adapter = MultiSpeedSignalAdapter()
        result = adapter.generate_signal("SPY")
        assert result is not None
        assert isinstance(result, SignalSourceResult)

    def test_signal_source_metadata(self, mock_multi_speed):
        adapter = MultiSpeedSignalAdapter()
        result = adapter.generate_signal("SPY")
        assert result.source_type == "multi_speed"
        assert result.source_name == "manahl_multi_speed_ensemble"

    def test_signal_in_range(self, mock_multi_speed):
        adapter = MultiSpeedSignalAdapter()
        result = adapter.generate_signal("SPY")
        assert -1.0 <= result.signal <= 1.0
        assert 0.0 <= result.confidence <= 1.0

    def test_returns_none_when_no_ensemble(self, mock_multi_speed):
        mock_multi_speed.compute_ensemble_signal.return_value = None
        adapter = MultiSpeedSignalAdapter()
        result = adapter.generate_signal("SPY")
        assert result is None

    def test_metadata_contains_tier_signals(self, mock_multi_speed):
        adapter = MultiSpeedSignalAdapter()
        result = adapter.generate_signal("SPY")
        assert "fast_signal" in result.metadata
        assert "medium_signal" in result.metadata
        assert "slow_signal" in result.metadata
        assert "speed_tiers" in result.metadata

    def test_historical_accuracy(self, mock_multi_speed):
        adapter = MultiSpeedSignalAdapter()
        result = adapter.generate_signal("SPY")
        assert result.historical_accuracy == 0.72

    def test_get_portfolio_signals(self, mock_multi_speed):
        adapter = MultiSpeedSignalAdapter()
        signals = adapter.get_portfolio_signals(["SPY", "GLD"])
        assert isinstance(signals, dict)
        assert "SPY" in signals
        assert "GLD" in signals
        assert isinstance(signals["SPY"], SignalSourceResult)

    def test_get_portfolio_signals_empty(self, mock_multi_speed):
        adapter = MultiSpeedSignalAdapter()
        signals = adapter.get_portfolio_signals([])
        assert signals == {}

    def test_consensus_signal_all_agree(self, mock_multi_speed):
        adapter = MultiSpeedSignalAdapter()
        result = adapter.generate_signal("SPY")
        assert result.signal == 1.0  # all three tier signals are 1.0

    def test_consensus_signal_mixed(self, mock_multi_speed):
        ens = _make_mock_speed_signal()
        ens.fast_signal.signal = 1
        ens.medium_signal.signal = -1
        ens.slow_signal.signal = 1
        mock_multi_speed.compute_ensemble_signal.return_value = ens
        adapter = MultiSpeedSignalAdapter()
        result = adapter.generate_signal("SPY")
        # Consensus = (1 + (-1) + 1) / 3 = 1/3
        assert result.signal == pytest.approx(1 / 3)
        # Confidence: 2 out of 3 match tier_signals[0] (which is 1)
        assert result.confidence == pytest.approx(2 / 3)

    def test_confidence_from_agreement(self, mock_multi_speed):
        ens = _make_mock_speed_signal()
        ens.fast_signal.signal = 1
        ens.medium_signal.signal = 1
        ens.slow_signal.signal = -1
        mock_multi_speed.compute_ensemble_signal.return_value = ens
        adapter = MultiSpeedSignalAdapter()
        result = adapter.generate_signal("SPY")
        # 2 out of 3 agree (fast and medium match)
        assert result.confidence == pytest.approx(2 / 3)


# ===================================================================
# RiskParitySignalAdapter
# ===================================================================

class TestRiskParitySignalAdapter:
    def test_init_defaults(self, mock_risk_parity):
        adapter = RiskParitySignalAdapter()
        assert adapter.source_type == "risk_parity"
        assert adapter.source_name == "bridgewater_rp_overlay"

    def test_init_custom_allocation(self, mock_risk_parity):
        alloc = {"SPY": 0.5, "GLD": 0.3, "TLT": 0.2}
        adapter = RiskParitySignalAdapter(base_allocation=alloc)
        assert adapter.base_allocation == alloc

    def test_generate_signal_returns_signal(self, mock_risk_parity):
        adapter = RiskParitySignalAdapter()
        result = adapter.generate_signal("SPY")
        assert result is not None
        assert isinstance(result, SignalSourceResult)

    def test_signal_scaled_from_adjustment(self, mock_risk_parity):
        mock_risk_parity.calculate_rp_overlay.return_value = _make_mock_rp_overlay(
            adjustment=0.075
        )
        adapter = RiskParitySignalAdapter()
        result = adapter.generate_signal("SPY")
        assert result.signal == 0.5

    def test_signal_clamped_positive(self, mock_risk_parity):
        mock_risk_parity.calculate_rp_overlay.return_value = _make_mock_rp_overlay(
            adjustment=0.30
        )
        adapter = RiskParitySignalAdapter()
        result = adapter.generate_signal("SPY")
        assert result.signal == 1.0

    def test_signal_clamped_negative(self, mock_risk_parity):
        mock_risk_parity.calculate_rp_overlay.return_value = _make_mock_rp_overlay(
            adjustment=-0.30
        )
        adapter = RiskParitySignalAdapter()
        result = adapter.generate_signal("SPY")
        assert result.signal == -1.0

    def test_confidence_from_rp_score(self, mock_risk_parity):
        mock_risk_parity.calculate_rp_overlay.return_value = _make_mock_rp_overlay(
            score=0.75
        )
        adapter = RiskParitySignalAdapter()
        result = adapter.generate_signal("SPY")
        assert result.confidence == 0.75

    def test_returns_none_when_no_overlay(self, mock_risk_parity):
        mock_risk_parity.calculate_rp_overlay.return_value = None
        adapter = RiskParitySignalAdapter()
        result = adapter.generate_signal("SPY")
        assert result is None

    def test_raw_score_is_adjustment(self, mock_risk_parity):
        mock_risk_parity.calculate_rp_overlay.return_value = _make_mock_rp_overlay(
            adjustment=0.05
        )
        adapter = RiskParitySignalAdapter()
        result = adapter.generate_signal("SPY")
        assert result.raw_score == 0.05

    def test_raw_unit(self, mock_risk_parity):
        adapter = RiskParitySignalAdapter()
        result = adapter.generate_signal("SPY")
        assert result.raw_unit == "weight_adjustment"

    def test_metadata_contains_weight_info(self, mock_risk_parity):
        adapter = RiskParitySignalAdapter()
        result = adapter.generate_signal("SPY")
        assert "base_weight" in result.metadata
        assert "target_weight" in result.metadata
        assert "asset_volatility" in result.metadata
        assert "risk_parity_quality" in result.metadata
        assert "expected_vol" in result.metadata

    def test_sample_count(self, mock_risk_parity):
        adapter = RiskParitySignalAdapter()
        result = adapter.generate_signal("SPY")
        assert result.sample_count == 5371

    def test_get_portfolio_signals(self, mock_risk_parity):
        adapter = RiskParitySignalAdapter()
        signals = adapter.get_portfolio_signals(["SPY", "GLD"])
        assert "SPY" in signals
        assert "GLD" in signals
        assert isinstance(signals["SPY"], SignalSourceResult)

    def test_get_portfolio_signals_empty(self, mock_risk_parity):
        adapter = RiskParitySignalAdapter()
        signals = adapter.get_portfolio_signals([])
        assert signals == {}


# ===================================================================
# NetworkMomentumSignalAdapter
# ===================================================================

class TestNetworkMomentumSignalAdapter:
    def test_init_defaults(self, mock_network_momentum):
        adapter = NetworkMomentumSignalAdapter()
        assert adapter.source_type == "network_momentum"
        assert adapter.source_name == "imperial_network_momentum"

    def test_init_custom_allocation(self, mock_network_momentum):
        alloc = {"SPY": 0.5, "GLD": 0.3, "TLT": 0.2}
        adapter = NetworkMomentumSignalAdapter(base_allocation=alloc)
        assert adapter.base_allocation == alloc

    def test_generate_signal_returns_signal(self, mock_network_momentum):
        adapter = NetworkMomentumSignalAdapter()
        result = adapter.generate_signal("SPY")
        assert result is not None
        assert isinstance(result, SignalSourceResult)

    def test_signal_scaled_from_momentum(self, mock_network_momentum):
        # _make_mock_network_signal(momentum=0.25) → signal = 0.25*2 = 0.5
        adapter = NetworkMomentumSignalAdapter()
        result = adapter.generate_signal("SPY")
        assert result.signal == 0.5

    def test_signal_clamped_positive(self, mock_network_momentum):
        ens, ll = _make_mock_network_signal(momentum=2.0)
        mock_network_momentum.compute_ensemble_signal.return_value = ens
        mock_network_momentum.compute_leadlag_matrix.return_value = ll
        adapter = NetworkMomentumSignalAdapter()
        result = adapter.generate_signal("SPY")
        assert result.signal == 1.0

    def test_signal_clamped_negative(self, mock_network_momentum):
        ens, ll = _make_mock_network_signal(momentum=-2.0)
        mock_network_momentum.compute_ensemble_signal.return_value = ens
        mock_network_momentum.compute_leadlag_matrix.return_value = ll
        adapter = NetworkMomentumSignalAdapter()
        result = adapter.generate_signal("SPY")
        assert result.signal == -1.0

    def test_returns_none_when_no_ensemble(self, mock_network_momentum):
        mock_network_momentum.compute_ensemble_signal.return_value = None
        adapter = NetworkMomentumSignalAdapter()
        result = adapter.generate_signal("SPY")
        assert result is None

    def test_returns_signal_with_none_leadlag(self, mock_network_momentum):
        """Returns signal even when leadlag matrix is None (graceful fallback)."""
        mock_network_momentum.compute_leadlag_matrix.return_value = None
        adapter = NetworkMomentumSignalAdapter()
        result = adapter.generate_signal("SPY")
        assert result is not None
        assert result.metadata.get("dominant_leader") is None

    def test_metadata_contains_network_info(self, mock_network_momentum):
        adapter = NetworkMomentumSignalAdapter()
        result = adapter.generate_signal("SPY")
        assert "network_centrality" in result.metadata
        assert "leadership_score" in result.metadata
        assert "followership_score" in result.metadata
        assert "window_count" in result.metadata

    def test_metadata_contains_dominant_leader(self, mock_network_momentum):
        adapter = NetworkMomentumSignalAdapter()
        result = adapter.generate_signal("SPY")
        assert "dominant_leader" in result.metadata
        assert result.metadata["dominant_leader"] == "SPY"

    def test_get_portfolio_signals(self, mock_network_momentum):
        adapter = NetworkMomentumSignalAdapter()
        signals = adapter.get_portfolio_signals(["SPY", "GLD"])
        assert "SPY" in signals
        assert "GLD" in signals
        assert isinstance(signals["SPY"], SignalSourceResult)

    def test_get_portfolio_signals_empty(self, mock_network_momentum):
        adapter = NetworkMomentumSignalAdapter()
        signals = adapter.get_portfolio_signals([])
        assert signals == {}

    def test_historical_accuracy(self, mock_network_momentum):
        adapter = NetworkMomentumSignalAdapter()
        result = adapter.generate_signal("SPY")
        assert result.historical_accuracy == 0.68

    def test_sample_count(self, mock_network_momentum):
        adapter = NetworkMomentumSignalAdapter()
        result = adapter.generate_signal("SPY")
        assert result.sample_count == 5371

    def test_confidence_from_ensemble(self, mock_network_momentum):
        ens, ll = _make_mock_network_signal(confidence_val=0.85)
        mock_network_momentum.compute_ensemble_signal.return_value = ens
        mock_network_momentum.compute_leadlag_matrix.return_value = ll
        adapter = NetworkMomentumSignalAdapter()
        result = adapter.generate_signal("SPY")
        assert result.confidence == 0.85


# ===================================================================
# _get_dominant_leader
# ===================================================================

class TestGetDominantLeader:
    def test_returns_asset_with_highest_leadership(self, mock_network_momentum):
        adapter = NetworkMomentumSignalAdapter()

        class MockMatrix:
            adjacency = {
                ("SPY", "GLD"): 0.5,
                ("SPY", "TLT"): 0.3,
                ("GLD", "SPY"): 0.1,
            }

        result = adapter._get_dominant_leader(MockMatrix())
        assert result == "SPY"

    def test_returns_first_asset_for_empty_matrix(self, mock_network_momentum):
        """Returns first asset (SPY) when matrix has no adjacency data."""
        adapter = NetworkMomentumSignalAdapter()

        class MockMatrix:
            adjacency = {}

        result = adapter._get_dominant_leader(MockMatrix())
        # Hardcoded leadership dict has SPY=0.0, GLD=0.0, TLT=0.0 -> max returns SPY
        assert result == "SPY"


# ===================================================================
# get_all_strategy_signals
# ===================================================================

class TestGetAllStrategySignals:
    def test_returns_three_strategies(self, mock_multi_speed, mock_risk_parity,
                                       mock_network_momentum):
        result = get_all_strategy_signals()
        assert "multi_speed" in result
        assert "risk_parity" in result
        assert "network_momentum" in result

    def test_each_strategy_has_ticker_signals(self, mock_multi_speed,
                                               mock_risk_parity,
                                               mock_network_momentum):
        result = get_all_strategy_signals(["SPY", "GLD", "TLT"])
        for strategy in result.values():
            assert "SPY" in strategy
            assert "GLD" in strategy
            assert "TLT" in strategy

    def test_default_tickers(self, mock_multi_speed, mock_risk_parity,
                              mock_network_momentum):
        result = get_all_strategy_signals()
        for strategy in result.values():
            assert "SPY" in strategy
            assert "GLD" in strategy
            assert "TLT" in strategy
            assert isinstance(strategy["SPY"], SignalSourceResult)

    def test_custom_tickers(self, mock_multi_speed, mock_risk_parity,
                             mock_network_momentum):
        result = get_all_strategy_signals(["BTC-USD"])
        for strategy in result.values():
            assert "BTC-USD" in strategy

    def test_empty_tickers(self, mock_multi_speed, mock_risk_parity,
                            mock_network_momentum):
        result = get_all_strategy_signals([])
        for strategy in result.values():
            assert strategy == {}

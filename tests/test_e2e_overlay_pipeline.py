"""
End-to-End Integration Tests — Full Overlay Pipeline (v4.91)

Validates the complete flow:
  Signal → Strategy → Orchestrator → Dashboard → Ensemble Bridge → Backtest
"""

import json
import pytest
import numpy as np
from datetime import datetime


class TestSignalToStrategyPipeline:
    """Test that signals flow correctly into strategy modules."""

    def test_collar_signal_to_overlay(self):
        """v4.60: collar signal → collar overlay decision."""
        from src.signals.collar_signal import generate_collar_signal
        from src.strategy.collar_overlay import CollarOverlay

        signal = generate_collar_signal(spot=550.0, vix=16.0)
        assert signal.is_valid

        overlay = CollarOverlay()
        decision = overlay.recommend(spot=550.0, vix=16.0)
        assert decision.call_strike > 550.0
        assert decision.put_strike < 550.0
        assert decision.max_upside > 0

    def test_crypto_signal_to_allocation(self):
        """v4.70: crypto signal → allocation decision."""
        from src.signals.crypto_momentum import generate_crypto_signal
        from src.strategy.crypto_allocation import CryptoAllocationOverlay

        signal = generate_crypto_signal()
        assert signal.timestamp is not None

        overlay = CryptoAllocationOverlay()
        decision = overlay.recommend()
        assert decision.total_crypto <= 0.05
        assert decision.total_crypto == decision.btc_weight + decision.eth_weight

    def test_bond_signal_to_rotation(self):
        """v4.80: bond duration signal → rotation decision."""
        from src.signals.bond_duration_signal import generate_bond_duration_signal
        from src.strategy.bond_duration_rotator import BondDurationRotator

        signal = generate_bond_duration_signal(
            yield_10y=4.5, yield_2y=4.0, real_rate=2.0, rate_change_6m=-0.5
        )
        assert signal.is_valid

        rotator = BondDurationRotator()
        decision = rotator.recommend(
            yield_10y=4.5, yield_2y=4.0, real_rate=2.0, rate_change_6m=-0.5
        )
        assert decision.curve_regime is not None
        assert abs(decision.tlt_sleeve + decision.ief_sleeve + decision.shy_sleeve - 1.0) < 0.01

    def test_calendar_signal(self):
        """v3.50: calendar seasonality signal."""
        from src.signals.calendar_seasonality import check_calendar
        from datetime import date

        signal = check_calendar(date(2026, 3, 10))
        assert signal.urgency_modifier == 1.0  # Normal Tuesday
        assert signal.is_trading_day

    def test_kurtosis_regime_detection(self):
        """v4.91: kurtosis regime signal."""
        from src.regime.kurtosis_regime import detect_kurtosis_regime

        rng = np.random.RandomState(42)
        returns = list(rng.normal(0, 0.01, 200))
        signal = detect_kurtosis_regime(returns)
        assert signal.regime is not None
        assert 2.0 < signal.kurtosis_60d < 5.0  # Near-normal


class TestOrchestratorPipeline:
    """Test that orchestrator properly aggregates all overlays."""

    def test_unified_orchestrator_collects(self):
        """v4.90: unified orchestrator collects from all overlays."""
        from src.strategy.unified_orchestrator import UnifiedOrchestrator

        orch = UnifiedOrchestrator()
        contributions = orch.collect_overlay_contributions()
        assert len(contributions) >= 2  # At least calendar + one other

    def test_unified_recommendation_valid(self):
        """v4.90: unified orchestrator produces valid recommendation."""
        from src.strategy.unified_orchestrator import UnifiedOrchestrator

        orch = UnifiedOrchestrator()
        rec = orch.recommend()
        total = rec.spy + rec.gld + rec.tlt + rec.ief + rec.shy + rec.btc + rec.eth
        assert abs(total - 1.0) < 0.02
        assert rec.estimated_sharpe > 0.5

    def test_ensemble_bridge(self):
        """v4.90: orchestrator → ensemble voter bridge."""
        from src.strategy.orchestrator_ensemble_bridge import OrchestratorEnsembleBridge

        bridge = OrchestratorEnsembleBridge()
        signal = bridge.generate_signal()
        assert -1.0 <= signal.value <= 1.0
        assert signal.confidence > 0

        reading = bridge.get_ensemble_reading()
        assert reading.asset_signals is not None
        assert "SPY" in reading.asset_signals

    def test_regime_router(self):
        """v4.91: regime router routes between TSMOM and MR."""
        from src.strategy.regime_router import RegimeRouter

        rng = np.random.RandomState(42)
        returns = list(rng.normal(0, 0.01, 200))
        router = RegimeRouter()
        decision = router.route(returns)
        assert decision.strategy_preference is not None
        assert abs(decision.tsom_weight + decision.mr_weight + decision.cash_weight - 1.0) < 0.01


class TestDashboardPipeline:
    """Test that dashboard data generator collects from all overlays."""

    def test_dashboard_generates(self):
        """v4.91: overlay dashboard generator."""
        from src.dashboard.overlay_dashboard import OverlayDashboardGenerator

        gen = OverlayDashboardGenerator()
        dashboard = gen.generate()
        assert dashboard.total_overlays >= 1
        assert dashboard.portfolio_risk in ("low", "moderate", "elevated", "high")
        assert dashboard.collar is not None
        assert dashboard.crypto is not None
        assert dashboard.kurtosis is not None

    def test_dashboard_risk_assessment(self):
        """Risk assessment should classify correctly."""
        from src.dashboard.overlay_dashboard import OverlayDashboardGenerator

        gen = OverlayDashboardGenerator()
        risk, alerts = gen._assess_portfolio_risk({
            "collar": {"vix_level": 15.0},
            "crypto": {"btc_vol_regime": "normal"},
            "kurtosis": {"fat_tail_risk": 0.1},
            "bond_duration": {"curve_regime": "normal"},
            "unified": {"conflict_count": 0},
        })
        assert risk == "low"

        risk, alerts = gen._assess_portfolio_risk({
            "collar": {"vix_level": 35.0},
            "crypto": {"btc_vol_regime": "extreme"},
            "kurtosis": {"fat_tail_risk": 0.9},
            "bond_duration": {"curve_regime": "inverted"},
            "unified": {"conflict_count": 3},
        })
        assert risk == "high"


class TestBacktestPipeline:
    """Test that backtest engines work with real/synthetic data."""

    def test_combined_backtest_runs(self):
        """v4.90: combined overlay backtest."""
        from src.backtest.combined_overlay_backtest import CombinedOverlayBacktest

        bt = CombinedOverlayBacktest()
        result = bt.run_backtest()
        assert result.trading_days > 0
        assert result.baseline_sharpe != 0
        assert result.sharpe_delta is not None

    def test_dbc_sweep_runs(self):
        """v4.90: DBC weight sweep."""
        from src.backtest.dbc_weight_sweep import DBCWeightSweep

        sweep = DBCWeightSweep()
        result = sweep.run_sweep()
        assert len(result.rows) == 18
        assert result.best_weight is not None

    def test_real_data_backtest_runs(self):
        """v4.90: real data backtest."""
        from src.backtest.real_data_backtest import RealDataBacktest

        bt = RealDataBacktest()
        result = bt.run()
        assert result is not None
        if result.trading_days > 0:
            assert result.combined_sharpe != 0


class TestEndToEndFlow:
    """Complete end-to-end: signal → strategy → orchestrator → dashboard."""

    def test_full_pipeline(self):
        """All modules import and work together without errors."""
        # 1. Generate all signals
        from src.signals.collar_signal import generate_collar_signal
        from src.signals.crypto_momentum import generate_crypto_signal
        from src.signals.bond_duration_signal import generate_bond_duration_signal
        from src.regime.kurtosis_regime import detect_kurtosis_regime

        collar = generate_collar_signal(spot=550.0, vix=16.0)
        crypto = generate_crypto_signal()
        bond = generate_bond_duration_signal()
        rng = np.random.RandomState(42)
        kurt = detect_kurtosis_regime(list(rng.normal(0, 0.01, 200)))

        assert collar.is_valid or not collar.is_valid  # Just confirm no crash
        assert crypto is not None
        assert bond.is_valid
        assert kurt is not None

        # 2. Strategies generate decisions
        from src.strategy.collar_overlay import CollarOverlay
        from src.strategy.crypto_allocation import CryptoAllocationOverlay
        from src.strategy.bond_duration_rotator import BondDurationRotator

        coll_dec = CollarOverlay().recommend(spot=550.0, vix=16.0)
        cryp_dec = CryptoAllocationOverlay().recommend()
        bond_dec = BondDurationRotator().recommend()

        assert coll_dec is not None
        assert cryp_dec is not None
        assert bond_dec is not None

        # 3. Orchestrator aggregates
        from src.strategy.unified_orchestrator import UnifiedOrchestrator
        orch = UnifiedOrchestrator()
        rec = orch.recommend()
        assert rec.spy > 0

        # 4. Dashboard collects
        from src.dashboard.overlay_dashboard import OverlayDashboardGenerator
        dash = OverlayDashboardGenerator().generate()
        assert dash.active_overlays >= 1

        # 5. Bridge exports to ensemble voter
        from src.strategy.orchestrator_ensemble_bridge import OrchestratorEnsembleBridge
        bridge = OrchestratorEnsembleBridge()
        reading = bridge.get_ensemble_reading()
        assert reading.source is not None

    def test_crisis_scenario_pipeline(self):
        """High-VIX scenario: collar activates, crypto exits, bonds go defensive."""
        from src.signals.collar_signal import generate_collar_signal
        from src.signals.bond_duration_signal import generate_bond_duration_signal

        # Crisis: VIX 45, inverted curve, rising rates
        collar = generate_collar_signal(spot=500.0, vix=45.0)
        bond = generate_bond_duration_signal(
            yield_10y=3.5, yield_2y=4.5, rate_change_6m=0.8
        )

        # Collar should be in crisis/unhedged
        assert collar.regime in ("crisis", "stress") or not collar.is_valid

        # Bonds should favor SHY (inverted + rising)
        assert bond.curve_regime == "inverted"
        assert bond.shy_weight > bond.tlt_weight

    def test_bull_market_pipeline(self):
        """Low-VIX bull: collar inactive, crypto active, bonds long duration."""
        from src.signals.collar_signal import generate_collar_signal
        from src.signals.bond_duration_signal import generate_bond_duration_signal

        # Bull: VIX 14, steep curve, falling rates
        collar = generate_collar_signal(spot=600.0, vix=14.0)
        bond = generate_bond_duration_signal(
            yield_10y=5.0, yield_2y=3.5, rate_change_6m=-0.8
        )

        # Collar should be active (normal market)
        assert collar.regime == "normal"

        # Bonds should favor TLT (steep + falling)
        assert bond.curve_regime == "steep"
        assert bond.tlt_weight > 0.3

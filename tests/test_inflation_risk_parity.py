#!/usr/bin/env python3
"""
Tests for inflation_risk_parity.py — dataclasses, constants, volatility calculation,
inverse-vol weighting, regime tilt application, and CLI.
"""
import sys
import os
import json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.strategy.inflation_risk_parity import (
    InflationRegime,
    RiskParityAllocation,
    InflationRiskParityEngine,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_engine(tmp_path=None):
    """Create an engine with a mock db_path."""
    engine = InflationRiskParityEngine.__new__(InflationRiskParityEngine)
    engine.db_path = tmp_path / "test.db" if tmp_path else Path("/tmp/fake.db")
    engine.vol_lookback = 60
    return engine


def _make_prices(n=100, start=450.0, drift=0.0003, vol=0.012, seed=42):
    rng = np.random.RandomState(seed)
    prices = [start]
    for _ in range(n - 1):
        prices.append(prices[-1] * (1 + rng.normal(drift, vol)))
    return np.array(prices)


# ---------------------------------------------------------------------------
# Dataclass Tests
# ---------------------------------------------------------------------------

class TestInflationRegime:

    def test_fields(self):
        r = InflationRegime(regime="high_inflation", confidence=0.85, signals={"gold_trend": 5.0})
        assert r.regime == "high_inflation"
        assert r.confidence == 0.85
        assert r.signals["gold_trend"] == 5.0


class TestRiskParityAllocation:

    def test_fields(self):
        regime = InflationRegime(regime="low_inflation", confidence=0.7, signals={})
        r = RiskParityAllocation(
            timestamp="2026-05-14",
            base_weights={"SPY": 0.5},
            tilted_weights={"SPY": 0.55},
            regime=regime,
            volatilities={"SPY": 0.16},
            risk_contributions={"SPY": 1.0},
        )
        assert r.timestamp == "2026-05-14"
        assert r.regime.regime == "low_inflation"


# ---------------------------------------------------------------------------
# Constants Tests
# ---------------------------------------------------------------------------

class TestConstants:

    def test_assets(self):
        assert "SPY" in InflationRiskParityEngine.ASSETS
        assert "GLD" in InflationRiskParityEngine.ASSETS
        assert "TLT" in InflationRiskParityEngine.ASSETS
        assert "DBC" in InflationRiskParityEngine.ASSETS

    def test_asset_vol(self):
        assert InflationRiskParityEngine.ASSETS["SPY"]["vol"] == 0.16
        assert InflationRiskParityEngine.ASSETS["DBC"]["vol"] == 0.20

    def test_asset_inflation_beta(self):
        assert InflationRiskParityEngine.ASSETS["GLD"]["inflation_beta"] == 0.8
        assert InflationRiskParityEngine.ASSETS["TLT"]["inflation_beta"] == -0.4

    def test_regime_tilts(self):
        assert "low_inflation" in InflationRiskParityEngine.REGIME_TILTS
        assert "rising_inflation" in InflationRiskParityEngine.REGIME_TILTS
        assert "high_inflation" in InflationRiskParityEngine.REGIME_TILTS
        assert "disinflation" in InflationRiskParityEngine.REGIME_TILTS

    def test_init_default(self):
        engine = InflationRiskParityEngine()
        assert engine.vol_lookback == 60

    def test_init_custom(self, tmp_path):
        db = tmp_path / "custom.db"
        engine = InflationRiskParityEngine(db_path=db)
        assert engine.db_path == db


# ---------------------------------------------------------------------------
# _calculate_volatility Tests
# ---------------------------------------------------------------------------

class TestCalculateVolatility:

    def test_returns_float(self):
        engine = _make_engine()
        prices = _make_prices(100)
        vol = engine._calculate_volatility(prices, 60)
        assert isinstance(vol, float)
        assert vol > 0

    def test_short_data_returns_default(self):
        engine = _make_engine()
        prices = np.array([100, 101])
        vol = engine._calculate_volatility(prices, 60)
        assert vol == 0.15

    def test_floor_at_5pct(self):
        engine = _make_engine()
        prices = np.array([100.0] * 70)
        vol = engine._calculate_volatility(prices, 60)
        assert vol == pytest.approx(0.05)

    def test_annualized(self):
        engine = _make_engine()
        rng = np.random.RandomState(99)
        daily_vol = 0.015
        prices = 100 * np.cumprod(1 + rng.normal(0, daily_vol, 100))
        vol = engine._calculate_volatility(prices, 60)
        expected = daily_vol * np.sqrt(252)
        assert vol == pytest.approx(max(expected, 0.05), rel=0.2)


# ---------------------------------------------------------------------------
# _calculate_inverse_vol_weights Tests
# ---------------------------------------------------------------------------

class TestInverseVolWeights:

    def test_sum_to_one(self):
        engine = _make_engine()
        vols = {"SPY": 0.16, "GLD": 0.15, "TLT": 0.12}
        weights = engine._calculate_inverse_vol_weights(vols)
        assert sum(weights.values()) == pytest.approx(1.0)

    def test_lower_vol_higher_weight(self):
        engine = _make_engine()
        vols = {"SPY": 0.20, "GLD": 0.10}
        weights = engine._calculate_inverse_vol_weights(vols)
        assert weights["GLD"] > weights["SPY"]

    def test_equal_vols_equal_weights(self):
        engine = _make_engine()
        vols = {"A": 0.15, "B": 0.15, "C": 0.15}
        weights = engine._calculate_inverse_vol_weights(vols)
        for w in weights.values():
            assert w == pytest.approx(1 / 3)

    def test_zero_vol_handled(self):
        engine = _make_engine()
        vols = {"SPY": 0.0, "GLD": 0.15}
        weights = engine._calculate_inverse_vol_weights(vols)
        assert sum(weights.values()) == pytest.approx(1.0)

    def test_all_zero_vols(self):
        engine = _make_engine()
        vols = {"A": 0.0, "B": 0.0}
        weights = engine._calculate_inverse_vol_weights(vols)
        assert weights["A"] == pytest.approx(0.5)
        assert weights["B"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# _apply_regime_tilt Tests
# ---------------------------------------------------------------------------

class TestApplyRegimeTilt:

    def test_sum_to_one(self):
        engine = _make_engine()
        base = {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16}
        tilted = engine._apply_regime_tilt(base, "low_inflation", 1.0)
        assert sum(tilted.values()) == pytest.approx(1.0)

    def test_low_inflation_favors_bonds(self):
        engine = _make_engine()
        base = {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16}
        tilted = engine._apply_regime_tilt(base, "low_inflation", 1.0)
        assert tilted["TLT"] > base["TLT"]
        assert tilted["GLD"] < base["GLD"]

    def test_high_inflation_favors_commodities(self):
        engine = _make_engine()
        base = {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16}
        tilted = engine._apply_regime_tilt(base, "high_inflation", 1.0)
        assert tilted["GLD"] > base["GLD"]
        assert tilted["TLT"] < base["TLT"]

    def test_rising_inflation_favors_dbc(self):
        engine = _make_engine()
        base = {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16, "DBC": 0.0}
        tilted = engine._apply_regime_tilt(base, "rising_inflation", 1.0)
        assert tilted["DBC"] > 0

    def test_disinflation_favors_equity(self):
        engine = _make_engine()
        base = {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16}
        tilted = engine._apply_regime_tilt(base, "disinflation", 1.0)
        assert tilted["SPY"] > base["SPY"]

    def test_confidence_scales_tilt(self):
        engine = _make_engine()
        base = {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16}
        full_tilt = engine._apply_regime_tilt(base, "high_inflation", 1.0)
        half_tilt = engine._apply_regime_tilt(base, "high_inflation", 0.5)
        # Full tilt should move GLD more than half tilt
        assert full_tilt["GLD"] - base["GLD"] > half_tilt["GLD"] - base["GLD"]

    def test_min_weight_enforced_pre_normalize(self):
        engine = _make_engine()
        # The code enforces min 2% before renormalization,
        # so after normalization weights may dip slightly below 2%
        base = {"SPY": 0.90, "GLD": 0.05, "TLT": 0.05}
        tilted = engine._apply_regime_tilt(base, "high_inflation", 1.0)
        # All weights should still be positive and non-trivial
        for w in tilted.values():
            assert w > 0.01

    def test_unknown_regime_no_tilt(self):
        engine = _make_engine()
        base = {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16}
        tilted = engine._apply_regime_tilt(base, "unknown_regime", 1.0)
        # Should still sum to 1
        assert sum(tilted.values()) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# CLI Tests
# ---------------------------------------------------------------------------

class TestCLI:

    def test_main_no_args(self, capsys):
        from src.strategy.inflation_risk_parity import main
        with patch("sys.argv", ["inflation_rp.py"]):
            with patch.object(InflationRiskParityEngine, 'get_allocation_summary') as mock:
                mock.return_value = {
                    "timestamp": "2026-05-14",
                    "regime": {"name": "low_inflation", "confidence": 0.7, "signals": {"gold_trend": -1.0}},
                    "allocation": {"base_weights": {"SPY": 0.5}, "tilted_weights": {"SPY": 0.55}},
                    "volatilities": {"SPY": 0.16},
                    "risk_contributions": {"SPY": 1.0},
                    "asset_classes": {"SPY": "equity"},
                    "strategy": "inflation_risk_parity",
                    "version": "2.11",
                }
                main()
        captured = capsys.readouterr()
        assert "INFLATION" in captured.out or "inflation" in captured.out.lower()

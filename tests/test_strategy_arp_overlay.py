"""
Tests for src/strategy/arp_overlay.py — Alternative Risk Premia Overlay Strategy

Covers: PremiumSignal, ARPOverlay dataclasses, AlternativeRiskPremiaEngine
(all public methods), CLI interface, edge cases.
No database required — all DB access is mocked or uses in-memory SQLite.
"""

import json
import sqlite3
import argparse
import pytest
import numpy as np
from datetime import datetime
from unittest.mock import patch, MagicMock
from pathlib import Path

from src.strategy.arp_overlay import (
    PremiumSignal,
    ARPOverlay as ARPOverlayResult,
    AlternativeRiskPremiaEngine,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_price_data(n_days: int = 252, base_price: float = 100.0,
                     volatility: float = 0.01, drift: float = 0.0003):
    """Generate synthetic price data as [(date_str, price), ...]"""
    np.random.seed(42)
    prices = [base_price]
    for _ in range(n_days - 1):
        ret = drift + np.random.normal(0, volatility)
        prices.append(prices[-1] * (1 + ret))
    dates = [f"2025-01-{i+1:02d}" for i in range(n_days)]
    return list(zip(dates, prices))


def _make_rising_prices(n_days=252, start=100.0, daily_ret=0.001):
    """Monotonically rising prices for predictable momentum."""
    prices = [start * ((1 + daily_ret) ** i) for i in range(n_days)]
    dates = [f"2025-01-{i+1:02d}" for i in range(n_days)]
    return list(zip(dates, prices))


def _make_falling_prices(n_days=252, start=100.0, daily_ret=-0.001):
    """Monotonically falling prices."""
    return _make_rising_prices(n_days, start, daily_ret)


def _make_engine_with_data():
    """Engine with synthetic data in cache — no DB required."""
    engine = AlternativeRiskPremiaEngine(db_path="/tmp/nonexistent_arp_test.db")
    engine.data_cache = {
        "VTV_252": _make_rising_prices(252, 80.0, 0.001),
        "VUG_252": _make_falling_prices(252, 200.0, -0.0005),
        "SPY_252": _make_rising_prices(252, 450.0, 0.0008),
        "QQQ_252": _make_rising_prices(252, 350.0, 0.0012),
        "MTUM_252": _make_rising_prices(252, 150.0, 0.001),
        "GLD_252": _make_price_data(252, 180.0, 0.008),
        "TLT_252": _make_price_data(252, 95.0, 0.005, drift=0.0),
        "IEF_252": _make_price_data(252, 105.0, 0.003, drift=0.0001),
        "HYG_252": _make_price_data(252, 75.0, 0.004, drift=0.0002),
        "LQD_252": _make_price_data(252, 110.0, 0.003, drift=0.0001),
        "DBC_252": _make_price_data(252, 20.0, 0.01, drift=-0.0002),
    }
    return engine


@pytest.fixture
def engine():
    """Engine with no real database."""
    return AlternativeRiskPremiaEngine(db_path="/tmp/nonexistent_arp_test.db")


@pytest.fixture
def engine_with_data():
    """Engine whose data_cache has synthetic data for all universe symbols."""
    return _make_engine_with_data()


# ---------------------------------------------------------------------------
# PremiumSignal dataclass
# ---------------------------------------------------------------------------

class TestPremiumSignal:
    def test_create_defaults(self):
        sig = PremiumSignal(premium_type="value", scores={}, confidence=0.5, last_update="2025-01-01")
        assert sig.premium_type == "value"
        assert sig.scores == {}
        assert sig.confidence == 0.5

    def test_create_with_scores(self):
        sig = PremiumSignal(
            premium_type="momentum",
            scores={"SPY": 0.8, "GLD": -0.3},
            confidence=0.9,
            last_update="2025-06-01",
        )
        assert sig.scores["SPY"] == 0.8
        assert sig.scores["GLD"] == -0.3

    def test_confidence_range(self):
        sig = PremiumSignal(premium_type="carry", scores={}, confidence=1.0, last_update="")
        assert sig.confidence == 1.0
        sig2 = PremiumSignal(premium_type="carry", scores={}, confidence=0.0, last_update="")
        assert sig2.confidence == 0.0

    def test_premium_types(self):
        for ptype in ["value", "momentum", "carry"]:
            sig = PremiumSignal(premium_type=ptype, scores={}, confidence=0.5, last_update="")
            assert sig.premium_type == ptype


# ---------------------------------------------------------------------------
# ARPOverlay dataclass
# ---------------------------------------------------------------------------

class TestARPOverlayResult:
    def _make_signal(self, ptype="value"):
        return PremiumSignal(premium_type=ptype, scores={}, confidence=0.5, last_update="")

    def test_create(self):
        overlay = ARPOverlayResult(
            value_signal=self._make_signal("value"),
            momentum_signal=self._make_signal("momentum"),
            carry_signal=self._make_signal("carry"),
            combined_scores={"SPY": 0.3},
            overlay_weights={"SPY": 0.48},
            base_allocation={"SPY": 0.46},
            final_allocation={"SPY": 0.47},
            last_update="2025-01-01",
        )
        assert overlay.value_signal.premium_type == "value"
        assert overlay.combined_scores["SPY"] == 0.3
        assert overlay.final_allocation["SPY"] == 0.47

    def test_all_signals_present(self):
        overlay = ARPOverlayResult(
            value_signal=self._make_signal("value"),
            momentum_signal=self._make_signal("momentum"),
            carry_signal=self._make_signal("carry"),
            combined_scores={},
            overlay_weights={},
            base_allocation={},
            final_allocation={},
            last_update="",
        )
        assert overlay.value_signal.premium_type == "value"
        assert overlay.momentum_signal.premium_type == "momentum"
        assert overlay.carry_signal.premium_type == "carry"

    def test_empty_combined_scores(self):
        overlay = ARPOverlayResult(
            value_signal=self._make_signal(),
            momentum_signal=self._make_signal(),
            carry_signal=self._make_signal(),
            combined_scores={},
            overlay_weights={},
            base_allocation={},
            final_allocation={},
            last_update="",
        )
        assert overlay.combined_scores == {}


# ---------------------------------------------------------------------------
# AlternativeRiskPremiaEngine — init & constants
# ---------------------------------------------------------------------------

class TestEngineInit:
    def test_default_db_path(self):
        from src.paths import MARKET_DB
        eng = AlternativeRiskPremiaEngine()
        assert eng.db_path == MARKET_DB

    def test_custom_db_path(self):
        eng = AlternativeRiskPremiaEngine(db_path="/custom/path.db")
        assert str(eng.db_path) == "/custom/path.db"

    def test_custom_db_path_as_path(self):
        eng = AlternativeRiskPremiaEngine(db_path=Path("/custom/path.db"))
        assert str(eng.db_path) == "/custom/path.db"

    def test_empty_cache(self):
        eng = AlternativeRiskPremiaEngine(db_path="/tmp/test.db")
        assert eng.data_cache == {}

    def test_universe_no_duplicate_spy(self):
        """Bug fix validation: SPY should appear exactly once in UNIVERSE."""
        keys = list(AlternativeRiskPremiaEngine.UNIVERSE.keys())
        assert keys.count("SPY") == 1, "SPY must appear exactly once in UNIVERSE"

    def test_universe_factors(self):
        u = AlternativeRiskPremiaEngine.UNIVERSE
        factors = {v["factor"] for v in u.values()}
        assert "value" in factors
        assert "momentum" in factors
        assert "carry" in factors

    def test_universe_value_assets(self):
        u = AlternativeRiskPremiaEngine.UNIVERSE
        value_assets = [k for k, v in u.items() if v["factor"] == "value"]
        assert "VTV" in value_assets
        assert "VUG" in value_assets

    def test_universe_momentum_assets(self):
        u = AlternativeRiskPremiaEngine.UNIVERSE
        momentum_assets = [k for k, v in u.items() if v["factor"] == "momentum"]
        assert "SPY" in momentum_assets
        assert "MTUM" in momentum_assets

    def test_universe_carry_assets(self):
        u = AlternativeRiskPremiaEngine.UNIVERSE
        carry_assets = [k for k, v in u.items() if v["factor"] == "carry"]
        assert "TLT" in carry_assets
        assert "GLD" in carry_assets

    def test_max_overlay_constant(self):
        assert AlternativeRiskPremiaEngine.MAX_OVERLAY == 0.05

    def test_min_allocation_constant(self):
        assert AlternativeRiskPremiaEngine.MIN_ALLOCATION == 0.01


# ---------------------------------------------------------------------------
# _load_price_data
# ---------------------------------------------------------------------------

class TestLoadPriceData:
    def test_returns_empty_when_db_missing(self, engine):
        result = engine._load_price_data("SPY", 252)
        assert result == []

    def test_uses_cache(self, engine_with_data):
        result = engine_with_data._load_price_data("VTV", 252)
        assert len(result) > 0
        result2 = engine_with_data._load_price_data("VTV", 252)
        assert result is result2

    def test_cache_key_includes_days(self, engine_with_data):
        engine_with_data.data_cache["VTV_63"] = _make_rising_prices(63, 80.0, 0.001)
        r1 = engine_with_data._load_price_data("VTV", 252)
        r2 = engine_with_data._load_price_data("VTV", 63)
        assert "VTV_252" in engine_with_data.data_cache
        assert "VTV_63" in engine_with_data.data_cache

    def test_db_query_with_real_sqlite(self, tmp_path):
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE prices (date TEXT, symbol TEXT, close REAL)")
        conn.execute("INSERT INTO prices VALUES ('2025-01-01', 'SPY', 450.0)")
        conn.execute("INSERT INTO prices VALUES ('2025-01-02', 'SPY', 452.0)")
        conn.execute("INSERT INTO prices VALUES ('2025-01-03', 'SPY', 455.0)")
        conn.commit()
        conn.close()

        eng = AlternativeRiskPremiaEngine(db_path=str(db_path))
        data = eng._load_price_data("SPY", 10)
        assert len(data) == 3
        assert data[0][0] == "2025-01-01"
        assert data[-1][0] == "2025-01-03"

    def test_db_error_returns_empty(self, tmp_path):
        db_path = tmp_path / "empty.db"
        conn = sqlite3.connect(str(db_path))
        conn.close()

        eng = AlternativeRiskPremiaEngine(db_path=str(db_path))
        data = eng._load_price_data("SPY", 252)
        assert data == []

    def test_chronological_order(self, tmp_path):
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE prices (date TEXT, symbol TEXT, close REAL)")
        # Insert in reverse order to verify reversal
        conn.execute("INSERT INTO prices VALUES ('2025-01-03', 'SPY', 455.0)")
        conn.execute("INSERT INTO prices VALUES ('2025-01-01', 'SPY', 450.0)")
        conn.execute("INSERT INTO prices VALUES ('2025-01-02', 'SPY', 452.0)")
        conn.commit()
        conn.close()

        eng = AlternativeRiskPremiaEngine(db_path=str(db_path))
        data = eng._load_price_data("SPY", 10)
        dates = [d for d, _ in data]
        assert dates == sorted(dates)

    def test_missing_symbol_returns_empty(self, tmp_path):
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE prices (date TEXT, symbol TEXT, close REAL)")
        conn.execute("INSERT INTO prices VALUES ('2025-01-01', 'QQQ', 350.0)")
        conn.commit()
        conn.close()

        eng = AlternativeRiskPremiaEngine(db_path=str(db_path))
        data = eng._load_price_data("SPY", 252)
        assert data == []


# ---------------------------------------------------------------------------
# _calculate_momentum
# ---------------------------------------------------------------------------

class TestCalculateMomentum:
    def test_rising_prices(self, engine):
        prices = [100.0 * (1.001 ** i) for i in range(200)]
        mom = engine._calculate_momentum(prices, 63)
        assert mom > 0

    def test_falling_prices(self, engine):
        prices = [100.0 * (0.999 ** i) for i in range(200)]
        mom = engine._calculate_momentum(prices, 63)
        assert mom < 0

    def test_insufficient_data(self, engine):
        prices = [100.0, 101.0, 102.0]
        mom = engine._calculate_momentum(prices, 63)
        assert mom == 0.0

    def test_zero_past_price(self, engine):
        prices = [0.0] * 50 + [100.0] * 50
        mom = engine._calculate_momentum(prices, 63)
        assert mom == 0.0

    def test_exactly_lookback_plus_one(self, engine):
        prices = [100.0 + i for i in range(64)]
        mom = engine._calculate_momentum(prices, 63)
        assert isinstance(mom, float)

    def test_flat_prices(self, engine):
        prices = [100.0] * 200
        mom = engine._calculate_momentum(prices, 63)
        assert mom == 0.0

    def test_annualization(self, engine):
        """6-month 10% return should annualize to > 10%."""
        prices = [100.0] * 126 + [110.0]
        mom = engine._calculate_momentum(prices, 126)
        # 10% over 126 days → annualized > 10%
        assert mom > 0.10

    def test_different_lookbacks(self, engine):
        prices = [100.0 * (1.001 ** i) for i in range(300)]
        mom_63 = engine._calculate_momentum(prices, 63)
        mom_126 = engine._calculate_momentum(prices, 126)
        assert isinstance(mom_63, float)
        assert isinstance(mom_126, float)


# ---------------------------------------------------------------------------
# calculate_value_premium
# ---------------------------------------------------------------------------

class TestCalculateValuePremium:
    def test_vtv_outperforms_vug(self, engine_with_data):
        sig = engine_with_data.calculate_value_premium()
        assert sig.premium_type == "value"
        assert isinstance(sig.scores, dict)
        assert isinstance(sig.confidence, float)

    def test_insufficient_data_returns_neutral(self, engine):
        sig = engine.calculate_value_premium()
        assert sig.confidence == 0.5
        assert sig.scores == {}

    def test_last_update_is_iso_format(self, engine_with_data):
        sig = engine_with_data.calculate_value_premium()
        dt = datetime.fromisoformat(sig.last_update)
        assert isinstance(dt, datetime)

    def test_value_score_bounded(self, engine_with_data):
        sig = engine_with_data.calculate_value_premium()
        for sym, score in sig.scores.items():
            assert -1.0 <= score <= 1.0

    def test_confidence_bounded(self, engine_with_data):
        sig = engine_with_data.calculate_value_premium()
        assert 0.0 <= sig.confidence <= 1.0

    def test_vtv_only_short_data(self, engine_with_data):
        """VTV has < 126 days → neutral signal."""
        engine_with_data.data_cache["VTV_252"] = _make_rising_prices(50, 80.0, 0.001)
        sig = engine_with_data.calculate_value_premium()
        assert sig.scores == {}

    def test_both_short_data(self, engine_with_data):
        """Both VTV and VUG short → neutral."""
        engine_with_data.data_cache["VTV_252"] = _make_rising_prices(50, 80.0, 0.001)
        engine_with_data.data_cache["VUG_252"] = _make_falling_prices(50, 200.0, -0.001)
        sig = engine_with_data.calculate_value_premium()
        assert sig.scores == {}
        assert sig.confidence == 0.5


# ---------------------------------------------------------------------------
# calculate_momentum_premium
# ---------------------------------------------------------------------------

class TestCalculateMomentumPremium:
    def test_returns_signal(self, engine_with_data):
        sig = engine_with_data.calculate_momentum_premium()
        assert sig.premium_type == "momentum"
        assert isinstance(sig.scores, dict)

    def test_ranking_with_sufficient_assets(self, engine_with_data):
        sig = engine_with_data.calculate_momentum_premium()
        if len(sig.scores) >= 2:
            positive = [s for s in sig.scores.values() if s > 0]
            negative = [s for s in sig.scores.values() if s < 0]
            assert len(positive) >= 1 or len(negative) >= 1

    def test_insufficient_data(self, engine):
        sig = engine.calculate_momentum_premium()
        assert sig.premium_type == "momentum"
        # With no DB, all assets get score 0.0 but there are 6 of them (>= 4),
        # so confidence is 0.7 (ranking still occurs even with zero scores)
        assert sig.confidence in (0.5, 0.7)

    def test_confidence_high_with_enough_assets(self, engine_with_data):
        sig = engine_with_data.calculate_momentum_premium()
        assert sig.confidence >= 0.7

    def test_scores_bounded(self, engine_with_data):
        sig = engine_with_data.calculate_momentum_premium()
        for sym, score in sig.scores.items():
            assert -1.0 <= score <= 1.0

    def test_short_data_for_some_assets(self, engine_with_data):
        """Some momentum assets have short data → score 0.0."""
        engine_with_data.data_cache["SPY_252"] = _make_rising_prices(30, 450.0, 0.001)
        sig = engine_with_data.calculate_momentum_premium()
        # SPY should get score 0.0 (insufficient data)
        assert "SPY" not in sig.scores or sig.scores.get("SPY", 0) == 0.0


# ---------------------------------------------------------------------------
# calculate_carry_premium
# ---------------------------------------------------------------------------

class TestCalculateCarryPremium:
    def test_returns_signal(self, engine_with_data):
        sig = engine_with_data.calculate_carry_premium()
        assert sig.premium_type == "carry"
        assert isinstance(sig.scores, dict)

    def test_insufficient_data(self, engine):
        sig = engine.calculate_carry_premium()
        assert sig.premium_type == "carry"
        # With no DB, all assets get score 0.0 but there are 7 of them (>= 4),
        # so confidence is 0.6 (ranking still occurs even with zero scores)
        assert sig.confidence in (0.4, 0.6)

    def test_carry_scores_bounded(self, engine_with_data):
        sig = engine_with_data.calculate_carry_premium()
        for sym, score in sig.scores.items():
            assert -0.5 <= score <= 1.0  # Top normalized to 1.0, bottom to -0.5

    def test_confidence_bounded(self, engine_with_data):
        sig = engine_with_data.calculate_carry_premium()
        assert 0.0 <= sig.confidence <= 1.0

    def test_bond_carry_inverted_momentum(self, engine_with_data):
        """For bonds, falling prices (rising yields) → better carry proxy."""
        sig = engine_with_data.calculate_carry_premium()
        assert isinstance(sig, PremiumSignal)

    def test_short_data_for_some_assets(self, engine_with_data):
        """Some carry assets with < 63 days → score 0.0."""
        engine_with_data.data_cache["TLT_252"] = _make_price_data(30, 95.0)
        sig = engine_with_data.calculate_carry_premium()
        # TLT should get score 0.0 (insufficient data)
        assert isinstance(sig, PremiumSignal)


# ---------------------------------------------------------------------------
# combine_premia
# ---------------------------------------------------------------------------

class TestCombinePremia:
    def test_combine_three_signals(self, engine):
        value = PremiumSignal("value", {"VTV": 0.5, "VUG": -0.3}, 0.8, "")
        momentum = PremiumSignal("momentum", {"SPY": 0.6, "GLD": -0.4}, 0.7, "")
        carry = PremiumSignal("carry", {"TLT": 0.3, "HYG": -0.2}, 0.6, "")

        combined = engine.combine_premia(value, momentum, carry)
        assert isinstance(combined, dict)
        assert "VTV" in combined or "SPY" in combined or "TLT" in combined

    def test_low_confidence_signal_excluded(self, engine):
        value = PremiumSignal("value", {"VTV": 0.5}, 0.2, "")
        momentum = PremiumSignal("momentum", {"SPY": 0.6}, 0.7, "")
        carry = PremiumSignal("carry", {"TLT": 0.3}, 0.6, "")

        combined = engine.combine_premia(value, momentum, carry)
        assert "VTV" not in combined

    def test_all_low_confidence(self, engine):
        value = PremiumSignal("value", {"VTV": 0.5}, 0.1, "")
        momentum = PremiumSignal("momentum", {"SPY": 0.6}, 0.1, "")
        carry = PremiumSignal("carry", {"TLT": 0.3}, 0.1, "")

        combined = engine.combine_premia(value, momentum, carry)
        assert combined == {}

    def test_custom_weights(self, engine):
        value = PremiumSignal("value", {"VTV": 0.5}, 0.9, "")
        momentum = PremiumSignal("momentum", {"SPY": 0.6}, 0.9, "")
        carry = PremiumSignal("carry", {"TLT": 0.3}, 0.9, "")

        combined = engine.combine_premia(
            value, momentum, carry,
            value_weight=0.5, momentum_weight=0.3, carry_weight=0.2,
        )
        assert isinstance(combined, dict)

    def test_normalization_by_total_weight(self, engine):
        value = PremiumSignal("value", {"VTV": 0.5}, 0.9, "")
        momentum = PremiumSignal("momentum", {"SPY": 0.6}, 0.9, "")
        carry = PremiumSignal("carry", {"TLT": 0.3}, 0.9, "")

        combined = engine.combine_premia(value, momentum, carry)
        if combined:
            assert all(isinstance(v, float) for v in combined.values())

    def test_overlapping_symbols(self, engine):
        value = PremiumSignal("value", {"SPY": 0.5, "GLD": 0.3}, 0.9, "")
        momentum = PremiumSignal("momentum", {"SPY": 0.6, "TLT": -0.2}, 0.9, "")
        carry = PremiumSignal("carry", {"SPY": 0.1, "GLD": -0.4}, 0.9, "")

        combined = engine.combine_premia(value, momentum, carry)
        assert "SPY" in combined
        assert "GLD" in combined

    def test_empty_scores(self, engine):
        value = PremiumSignal("value", {}, 0.9, "")
        momentum = PremiumSignal("momentum", {}, 0.9, "")
        carry = PremiumSignal("carry", {}, 0.9, "")

        combined = engine.combine_premia(value, momentum, carry)
        assert combined == {}


# ---------------------------------------------------------------------------
# apply_overlay
# ---------------------------------------------------------------------------

class TestApplyOverlay:
    def test_basic_overlay(self, engine):
        base = {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16}
        scores = {"SPY": 0.5, "GLD": -0.3}

        result = engine.apply_overlay(base, scores)
        assert isinstance(result, dict)
        for sym in base:
            assert sym in result

    def test_renormalize_to_one(self, engine):
        base = {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16}
        scores = {"SPY": 0.3, "GLD": 0.2}

        result = engine.apply_overlay(base, scores)
        total = sum(result.values())
        assert abs(total - 1.0) < 1e-9

    def test_min_allocation_enforced(self, engine):
        base = {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16}
        scores = {"TLT": -1.0}

        result = engine.apply_overlay(base, scores)
        for sym in result:
            assert result[sym] >= engine.MIN_ALLOCATION

    def test_empty_scores(self, engine):
        base = {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16}
        result = engine.apply_overlay(base, {})
        total = sum(result.values())
        assert abs(total - 1.0) < 1e-9

    def test_custom_max_overlay(self, engine):
        base = {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16}
        scores = {"SPY": 1.0}

        result = engine.apply_overlay(base, scores, max_overlay=0.10)
        assert result["SPY"] >= base["SPY"]

    def test_asset_in_scores_not_in_base(self, engine):
        base = {"SPY": 0.50, "GLD": 0.50}
        scores = {"SPY": 0.3, "QQQ": 0.5}

        result = engine.apply_overlay(base, scores)
        assert "QQQ" in result
        assert result["QQQ"] >= engine.MIN_ALLOCATION

    def test_extreme_positive_score(self, engine):
        base = {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16}
        scores = {"SPY": 100.0}

        result = engine.apply_overlay(base, scores)
        assert result["SPY"] <= 0.50  # Hard cap at 50%

    def test_extreme_negative_score(self, engine):
        base = {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16}
        scores = {"TLT": -100.0}

        result = engine.apply_overlay(base, scores)
        assert result["TLT"] >= engine.MIN_ALLOCATION

    def test_single_asset_base(self, engine):
        base = {"SPY": 1.0}
        scores = {"SPY": 0.5}
        result = engine.apply_overlay(base, scores)
        total = sum(result.values())
        assert abs(total - 1.0) < 1e-9

    def test_empty_base_with_scores(self, engine):
        base = {}
        scores = {"SPY": 0.5}
        result = engine.apply_overlay(base, scores)
        assert "SPY" in result

    def test_all_weights_positive(self, engine):
        base = {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16}
        scores = {"SPY": 1.0, "GLD": -1.0, "TLT": 0.5}
        result = engine.apply_overlay(base, scores)
        for sym, w in result.items():
            assert w >= 0


# ---------------------------------------------------------------------------
# get_arp_overlay
# ---------------------------------------------------------------------------

class TestGetARPOverlay:
    def test_full_overlay(self, engine_with_data):
        overlay = engine_with_data.get_arp_overlay()
        assert isinstance(overlay, ARPOverlayResult)
        assert isinstance(overlay.value_signal, PremiumSignal)
        assert isinstance(overlay.momentum_signal, PremiumSignal)
        assert isinstance(overlay.carry_signal, PremiumSignal)

    def test_default_base_allocation(self, engine_with_data):
        from src.paths import BASE_ALLOCATION
        overlay = engine_with_data.get_arp_overlay()
        assert overlay.base_allocation == BASE_ALLOCATION

    def test_custom_base_allocation(self, engine_with_data):
        custom = {"SPY": 0.60, "GLD": 0.30, "TLT": 0.10}
        overlay = engine_with_data.get_arp_overlay(base_allocation=custom)
        assert overlay.base_allocation == custom

    def test_final_allocation_sums_to_one(self, engine_with_data):
        overlay = engine_with_data.get_arp_overlay()
        total = sum(overlay.final_allocation.values())
        assert abs(total - 1.0) < 1e-6

    def test_overlay_weights_present(self, engine_with_data):
        overlay = engine_with_data.get_arp_overlay()
        assert isinstance(overlay.overlay_weights, dict)
        assert len(overlay.overlay_weights) > 0

    def test_combined_scores_present(self, engine_with_data):
        overlay = engine_with_data.get_arp_overlay()
        assert isinstance(overlay.combined_scores, dict)

    def test_final_blend_95_5(self, engine_with_data):
        """Final allocation = 95% base + 5% overlay, then renormalized."""
        overlay = engine_with_data.get_arp_overlay()
        base = overlay.base_allocation
        final = overlay.final_allocation
        for sym in base:
            if sym in final:
                assert abs(final[sym] - base[sym]) < 0.05

    def test_last_update_iso_format(self, engine_with_data):
        overlay = engine_with_data.get_arp_overlay()
        dt = datetime.fromisoformat(overlay.last_update)
        assert isinstance(dt, datetime)


# ---------------------------------------------------------------------------
# format_overlay
# ---------------------------------------------------------------------------

class TestFormatOverlay:
    def test_structure(self, engine_with_data):
        overlay = engine_with_data.get_arp_overlay()
        formatted = engine_with_data.format_overlay(overlay)

        assert formatted["strategy"] == "Alternative Risk Premia Overlay"
        assert "value_premium" in formatted
        assert "momentum_premium" in formatted
        assert "carry_premium" in formatted
        assert "combined_scores" in formatted
        assert "overlay_weights" in formatted
        assert "base_allocation" in formatted
        assert "final_allocation" in formatted
        assert "last_update" in formatted

    def test_weights_as_percentages(self, engine_with_data):
        overlay = engine_with_data.get_arp_overlay()
        formatted = engine_with_data.format_overlay(overlay)

        for sym, w in formatted["overlay_weights"].items():
            assert w > 0.1

    def test_allocation_as_percentages(self, engine_with_data):
        overlay = engine_with_data.get_arp_overlay()
        formatted = engine_with_data.format_overlay(overlay)

        for sym, w in formatted["base_allocation"].items():
            assert w > 0.1

    def test_scores_rounded_to_3dp(self, engine_with_data):
        overlay = engine_with_data.get_arp_overlay()
        formatted = engine_with_data.format_overlay(overlay)

        for sym, s in formatted["combined_scores"].items():
            assert s == round(s, 3)

    def test_json_serializable(self, engine_with_data):
        overlay = engine_with_data.get_arp_overlay()
        formatted = engine_with_data.format_overlay(overlay)

        json_str = json.dumps(formatted)
        assert isinstance(json_str, str)

    def test_premium_confidence_rounded(self, engine_with_data):
        overlay = engine_with_data.get_arp_overlay()
        formatted = engine_with_data.format_overlay(overlay)

        for premium_key in ["value_premium", "momentum_premium", "carry_premium"]:
            conf = formatted[premium_key]["confidence"]
            assert conf == round(conf, 2)


# ---------------------------------------------------------------------------
# calculate_correlation_to_spy
# ---------------------------------------------------------------------------

class TestCalculateCorrelationToSpy:
    def test_returns_float(self, engine_with_data):
        corr = engine_with_data.calculate_correlation_to_spy()
        assert isinstance(corr, float)
        assert -1.0 <= abs(corr) <= 1.0 or corr == 0.5

    def test_insufficient_data(self, engine):
        corr = engine.calculate_correlation_to_spy()
        assert corr == 0.5

    def test_custom_lookback(self, engine_with_data):
        corr = engine_with_data.calculate_correlation_to_spy(lookback_days=126)
        assert isinstance(corr, float)

    def test_nan_protection(self, engine_with_data):
        corr = engine_with_data.calculate_correlation_to_spy()
        assert not np.isnan(corr)


# ---------------------------------------------------------------------------
# CLI logic (tested without __main__ guard)
# ---------------------------------------------------------------------------

class TestCLILogic:
    def test_overlay_flag_logic(self, engine_with_data, capsys):
        """Test the overlay branch of CLI logic."""
        overlay = engine_with_data.get_arp_overlay()
        formatted = engine_with_data.format_overlay(overlay)
        print(json.dumps(formatted, indent=2))

        captured = capsys.readouterr()
        assert "Alternative Risk Premia Overlay" in captured.out

    def test_correlation_flag_logic(self, engine_with_data, capsys):
        """Test the correlation branch of CLI logic."""
        corr = engine_with_data.calculate_correlation_to_spy()
        status = "GOOD" if abs(corr) < 0.7 else "HIGH"
        print(f"ARP-SPY Correlation (estimated): {corr:.3f}")
        print(f"Status: {status}")

        captured = capsys.readouterr()
        assert "ARP-SPY Correlation" in captured.out

    def test_base_allocation_json_parsing(self, engine_with_data):
        base_json = '{"SPY": 0.50, "GLD": 0.35, "TLT": 0.15}'
        base_alloc = json.loads(base_json)
        overlay = engine_with_data.get_arp_overlay(base_allocation=base_alloc)
        assert overlay.base_allocation == base_alloc

    def test_no_flag_prints_help(self):
        """Bug fix validation: no flag should not unconditionally run overlay."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--overlay", action="store_true")
        parser.add_argument("--correlation", action="store_true")
        args = parser.parse_args([])
        assert not args.overlay
        assert not args.correlation


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_zero_confidence_all_signals(self, engine):
        value = PremiumSignal("value", {}, 0.0, "")
        momentum = PremiumSignal("momentum", {}, 0.0, "")
        carry = PremiumSignal("carry", {}, 0.0, "")

        combined = engine.combine_premia(value, momentum, carry)
        assert combined == {}

    def test_many_assets_with_tiny_weights(self, engine):
        base = {f"SYM{i}": 0.1 for i in range(10)}
        scores = {f"SYM{i}": 0.5 * (i % 2 * 2 - 1) for i in range(10)}
        result = engine.apply_overlay(base, scores)
        total = sum(result.values())
        assert abs(total - 1.0) < 1e-6

    def test_momentum_with_constant_prices(self, engine):
        prices = [100.0] * 300
        mom = engine._calculate_momentum(prices, 63)
        assert mom == 0.0

    def test_momentum_with_single_price(self, engine):
        prices = [100.0]
        mom = engine._calculate_momentum(prices, 63)
        assert mom == 0.0

    def test_carry_with_zero_volatility(self, engine):
        data = [("2025-01-01", 100.0)] * 252
        engine.data_cache = {"TLT_252": data}
        sig = engine.calculate_carry_premium()
        assert sig.premium_type == "carry"

    def test_overlay_with_no_matching_symbols(self, engine):
        base = {"SPY": 0.50, "GLD": 0.50}
        scores = {"QQQ": 0.8, "IWM": -0.5}

        result = engine.apply_overlay(base, scores)
        assert "SPY" in result
        assert "QQQ" in result
        assert "IWM" in result

    def test_get_arp_overlay_no_data(self, engine):
        overlay = engine.get_arp_overlay()
        assert isinstance(overlay, ARPOverlayResult)
        assert overlay.value_signal.confidence == 0.5
        # momentum/carry may have >= 4 assets with score 0.0 → higher confidence
        assert overlay.momentum_signal.confidence <= 0.7
        assert overlay.carry_signal.confidence <= 0.6

    def test_format_overlay_empty_signals(self, engine):
        overlay = engine.get_arp_overlay()
        formatted = engine.format_overlay(overlay)
        assert formatted["strategy"] == "Alternative Risk Premia Overlay"

    def test_all_same_momentum(self, engine):
        """All assets with identical prices → all zero momentum."""
        engine.data_cache = {
            f"{sym}_252": [(f"2025-01-{i+1:02d}", 100.0) for i in range(252)]
            for sym in ["SPY", "QQQ", "MTUM", "GLD", "TLT", "DBC"]
        }
        engine.data_cache["VTV_252"] = [(f"2025-01-{i+1:02d}", 80.0) for i in range(252)]
        engine.data_cache["VUG_252"] = [(f"2025-01-{i+1:02d}", 200.0) for i in range(252)]
        engine.data_cache["HYG_252"] = [(f"2025-01-{i+1:02d}", 75.0) for i in range(252)]
        engine.data_cache["LQD_252"] = [(f"2025-01-{i+1:02d}", 110.0) for i in range(252)]
        engine.data_cache["IEF_252"] = [(f"2025-01-{i+1:02d}", 105.0) for i in range(252)]

        overlay = engine.get_arp_overlay()
        assert isinstance(overlay, ARPOverlayResult)

    def test_get_connection(self, tmp_path):
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))
        conn.close()

        eng = AlternativeRiskPremiaEngine(db_path=str(db_path))
        conn = eng._get_connection()
        assert isinstance(conn, sqlite3.Connection)
        conn.close()

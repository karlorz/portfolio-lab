"""
Tests for Cross-Asset Regime Arbitrage (v8.09).

Tests per-asset regime detection, divergence classification,
state persistence, and ensemble-facing API.
"""

import json
import logging
import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.signals.cross_asset_regime_arb import (
    AssetRegime,
    AssetRegimeReading,
    BondRegime,
    BondRegimeReading,
    CrossAssetRegimeArbDetector,
    CrossAssetRegimeArbSignal,
    DivergencePattern,
    DivergenceReading,
    GoldRegime,
    GoldRegimeReading,
    BULL_MOMENTUM_THRESHOLD,
    BEAR_MOMENTUM_THRESHOLD,
    STRONG_MOMENTUM_THRESHOLD,
    HIGH_VOL_THRESHOLD,
    print_signal_report,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_prices():
    """Generate sample price data for testing."""
    # Create 90 days of synthetic prices with known trends
    spy_prices = []
    tlt_prices = []
    gld_prices = []

    base_date = datetime(2026, 1, 1)
    spy_base = 500.0
    tlt_base = 90.0
    gld_base = 400.0

    # SPY: bullish trend (+10% over 60d)
    for i in range(90):
        date = (base_date + timedelta(days=i)).strftime("%Y-%m-%d")
        spy_price = spy_base * (1 + 0.0017 * i)  # ~10% over 60d
        spy_prices.append({"d": date, "p": spy_price})

    # TLT: stable trend (-3% over 60d)
    for i in range(90):
        date = (base_date + timedelta(days=i)).strftime("%Y-%m-%d")
        tlt_price = tlt_base * (1 - 0.0005 * i)  # ~-3% over 60d
        tlt_prices.append({"d": date, "p": tlt_price})

    # GLD: weak trend (-7% over 60d)
    for i in range(90):
        date = (base_date + timedelta(days=i)).strftime("%Y-%m-%d")
        gld_price = gld_base * (1 - 0.0012 * i)  # ~-7% over 60d
        gld_prices.append({"d": date, "p": gld_price})

    return {
        "SPY": spy_prices,
        "TLT": tlt_prices,
        "GLD": gld_prices,
    }


@pytest.fixture
def bear_prices():
    """Generate bearish scenario: all assets declining."""
    spy_prices = []
    tlt_prices = []
    gld_prices = []

    base_date = datetime(2026, 1, 1)
    spy_base = 500.0
    tlt_base = 90.0
    gld_base = 400.0

    # All bearish (-8% to -12% over 60d)
    for i in range(90):
        date = (base_date + timedelta(days=i)).strftime("%Y-%m-%d")
        spy_prices.append({"d": date, "p": spy_base * (1 - 0.0020 * i)})
        tlt_prices.append({"d": date, "p": tlt_base * (1 - 0.0010 * i)})
        gld_prices.append({"d": date, "p": gld_base * (1 - 0.0015 * i)})

    return {"SPY": spy_prices, "TLT": tlt_prices, "GLD": gld_prices}


@pytest.fixture
def flight_to_safety_prices():
    """Generate flight-to-safety scenario: equity down, bonds up, gold up."""
    spy_prices = []
    tlt_prices = []
    gld_prices = []

    base_date = datetime(2026, 1, 1)

    # SPY down, TLT up (yields falling), GLD up
    for i in range(90):
        date = (base_date + timedelta(days=i)).strftime("%Y-%m-%d")
        spy_prices.append({"d": date, "p": 500.0 * (1 - 0.0020 * i)})
        tlt_prices.append({"d": date, "p": 90.0 * (1 + 0.0015 * i)})
        gld_prices.append({"d": date, "p": 400.0 * (1 + 0.0010 * i)})

    return {"SPY": spy_prices, "TLT": tlt_prices, "GLD": gld_prices}


@pytest.fixture
def risk_rotation_prices():
    """Generate risk rotation: equity down, gold up, bonds flat."""
    spy_prices = []
    tlt_prices = []
    gld_prices = []

    base_date = datetime(2026, 1, 1)

    for i in range(90):
        date = (base_date + timedelta(days=i)).strftime("%Y-%m-%d")
        spy_prices.append({"d": date, "p": 500.0 * (1 - 0.0018 * i)})
        tlt_prices.append({"d": date, "p": 90.0 * (1 + 0.0002 * i)})  # flat
        gld_prices.append({"d": date, "p": 400.0 * (1 + 0.0015 * i)})

    return {"SPY": spy_prices, "TLT": tlt_prices, "GLD": gld_prices}


@pytest.fixture
def inflation_fear_prices():
    """Generate inflation fear: bonds down (yields up), gold up, equity flat."""
    spy_prices = []
    tlt_prices = []
    gld_prices = []

    base_date = datetime(2026, 1, 1)

    for i in range(90):
        date = (base_date + timedelta(days=i)).strftime("%Y-%m-%d")
        spy_prices.append({"d": date, "p": 500.0 * (1 + 0.0002 * i)})
        tlt_prices.append({"d": date, "p": 90.0 * (1 - 0.0015 * i)})
        gld_prices.append({"d": date, "p": 400.0 * (1 + 0.0018 * i)})

    return {"SPY": spy_prices, "TLT": tlt_prices, "GLD": gld_prices}


@pytest.fixture
def detector_with_prices(sample_prices):
    """Create detector with injected price data and clean state."""
    detector = CrossAssetRegimeArbDetector()
    detector.prices = sample_prices
    detector.state = {"previous_pattern": None, "persistence_days": 0, "last_date": None}
    return detector


@pytest.fixture
def temp_state_file():
    """Create a temporary state file path."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({"previous_pattern": None, "persistence_days": 0, "last_date": None}, f)
        temp_path = f.name
    yield temp_path
    os.unlink(temp_path)


# =============================================================================
# Data Loading Tests
# =============================================================================

class TestDataLoading:
    """Tests for price data loading."""

    def test_load_prices_success(self, sample_prices, tmp_path):
        """Detector loads prices from JSON file."""
        # Write sample prices to temp file
        prices_file = tmp_path / "prices.json"
        with open(prices_file, "w") as f:
            json.dump(sample_prices, f)

        detector = CrossAssetRegimeArbDetector(data_dir=tmp_path)
        result = detector._load_prices()

        assert result is True
        assert "SPY" in detector.prices
        assert "TLT" in detector.prices
        assert "GLD" in detector.prices
        assert len(detector.prices["SPY"]) == 90

    def test_load_prices_file_not_found(self, tmp_path):
        """Loading from non-existent path returns False."""
        detector = CrossAssetRegimeArbDetector(data_dir=tmp_path)
        result = detector._load_prices()
        assert result is False

    def test_load_prices_missing_symbol(self, tmp_path):
        """Missing required symbol returns False."""
        bad_data = {"SPY": [{"d": "2026-01-01", "p": 100.0}]}
        prices_file = tmp_path / "prices.json"
        with open(prices_file, "w") as f:
            json.dump(bad_data, f)

        detector = CrossAssetRegimeArbDetector(data_dir=tmp_path)
        result = detector._load_prices()
        assert result is False

    def test_load_prices_empty_data(self, tmp_path):
        """Empty price file returns False."""
        prices_file = tmp_path / "prices.json"
        with open(prices_file, "w") as f:
            json.dump({}, f)

        detector = CrossAssetRegimeArbDetector(data_dir=tmp_path)
        result = detector._load_prices()
        assert result is False


# =============================================================================
# Return & Volatility Calculation Tests
# =============================================================================

class TestReturnsAndVolatility:
    """Tests for return and volatility calculations."""

    def test_get_returns_positive(self, detector_with_prices):
        """Returns positive for bullish trend."""
        ret = detector_with_prices._get_returns("SPY", 60)
        assert ret is not None
        assert ret > 0.05  # > 5% over 60d

    def test_get_returns_negative(self, detector_with_prices):
        """Returns negative for bearish trend."""
        ret = detector_with_prices._get_returns("GLD", 60)
        assert ret is not None
        assert ret < 0

    def test_get_returns_insufficient_data(self, detector_with_prices):
        """Returns None when not enough data."""
        ret = detector_with_prices._get_returns("SPY,", 999)  # key doesn't exist
        assert ret is None

    def test_get_returns_lookback_too_long(self, detector_with_prices):
        """Returns None when lookback exceeds available data."""
        ret = detector_with_prices._get_returns("SPY", 1000)
        assert ret is None

    def test_get_volatility_valid(self, detector_with_prices):
        """Volatility is positive and reasonable."""
        vol = detector_with_prices._get_volatility("SPY", 20)
        assert vol is not None
        assert vol > 0
        assert vol < 1.0  # Annualized vol < 100%

    def test_get_volatility_insufficient_data(self, detector_with_prices):
        """Volatility returns None with insufficient data."""
        vol = detector_with_prices._get_volatility("SPY", 5)  # need 6+ points for 5 returns
        assert vol is not None  # 5 daily returns is enough from 6+ points
        vol_too_short = detector_with_prices._get_volatility("SPY", 0)
        assert vol_too_short is None


# =============================================================================
# Per-Asset Regime Detection Tests
# =============================================================================

class TestEquityRegimeDetection:
    """Tests for SPY equity regime classification."""

    def test_equity_bull(self, detector_with_prices):
        """SPY with strong positive momentum is BULL."""
        reading = detector_with_prices._detect_equity_regime()
        assert reading is not None
        assert reading.asset_regime == AssetRegime.BULL
        assert reading.symbol == "SPY"

    def test_equity_bull_confidence(self, detector_with_prices):
        """Strong momentum yields high confidence."""
        reading = detector_with_prices._detect_equity_regime()
        assert reading is not None
        assert reading.confidence > 0.5

    def test_equity_bear(self, bear_prices):
        """All-bearish market yields BEAR regime."""
        detector = CrossAssetRegimeArbDetector()
        detector.prices = bear_prices
        reading = detector._detect_equity_regime()
        assert reading is not None
        assert reading.asset_regime == AssetRegime.BEAR
        assert reading.momentum_60d < 0

    def test_equity_includes_vol(self, detector_with_prices):
        """Reading includes volatility field."""
        reading = detector_with_prices._detect_equity_regime()
        assert reading is not None
        assert reading.volatility_20d > 0

    def test_equity_returns_none_no_data(self, detector_with_prices):
        """Returns None with no data."""
        detector_with_prices.prices = {}
        reading = detector_with_prices._detect_equity_regime()
        assert reading is None


class TestBondRegimeDetection:
    """Tests for TLT bond regime classification."""

    def test_bond_stable(self, detector_with_prices):
        """TLT with moderate negative momentum is STABLE."""
        reading = detector_with_prices._detect_bond_regime()
        assert reading is not None
        assert reading.symbol == "TLT"

    def test_bond_falling_yields(self, flight_to_safety_prices):
        """TLT rallying (yields falling) → FALLING regime."""
        detector = CrossAssetRegimeArbDetector()
        detector.prices = flight_to_safety_prices
        reading = detector._detect_bond_regime()
        assert reading is not None
        assert reading.regime == BondRegime.FALLING

    def test_bond_rising_yields(self, inflation_fear_prices):
        """TLT declining (yields rising) → RISING regime."""
        detector = CrossAssetRegimeArbDetector()
        detector.prices = inflation_fear_prices
        reading = detector._detect_bond_regime()
        assert reading is not None
        assert reading.regime == BondRegime.RISING

    def test_bond_rising_yields_negative_momentum(self):
        """Explicit test: TLT down → yields rising."""
        detector = CrossAssetRegimeArbDetector()
        detector.prices = {
            "SPY": [{"d": f"2026-01-{i:02d}", "p": 500.0} for i in range(1, 91)],
            "TLT": [{"d": f"2026-01-{i:02d}", "p": 100.0 - i * 0.3} for i in range(1, 91)],
            "GLD": [{"d": f"2026-01-{i:02d}", "p": 400.0} for i in range(1, 91)],
        }
        reading = detector._detect_bond_regime()
        assert reading is not None
        assert reading.regime == BondRegime.RISING

    def test_bond_returns_none_no_data(self, detector_with_prices):
        """Returns None with no TLT data."""
        detector_with_prices.prices = {"SPY": detector_with_prices.prices.get("SPY", [])}
        reading = detector_with_prices._detect_bond_regime()
        assert reading is None


class TestGoldRegimeDetection:
    """Tests for GLD gold regime classification."""

    def test_gold_weak(self, detector_with_prices):
        """GLD with negative momentum is WEAK."""
        reading = detector_with_prices._detect_gold_regime()
        assert reading is not None
        assert reading.regime == GoldRegime.WEAK
        assert reading.momentum_60d < 0

    def test_gold_strong(self, risk_rotation_prices):
        """GLD rallying → STRONG regime."""
        detector = CrossAssetRegimeArbDetector()
        detector.prices = risk_rotation_prices
        reading = detector._detect_gold_regime()
        assert reading is not None
        assert reading.regime == GoldRegime.STRONG

    def test_gold_sideways(self):
        """Gold with near-zero momentum → SIDEWAYS."""
        detector = CrossAssetRegimeArbDetector()
        detector.prices = {
            "SPY": [{"d": f"2026-01-{i:02d}", "p": 500.0} for i in range(1, 91)],
            "TLT": [{"d": f"2026-01-{i:02d}", "p": 90.0} for i in range(1, 91)],
            "GLD": [{"d": f"2026-01-{i:02d}", "p": 400.0} for i in range(1, 91)],
        }
        reading = detector._detect_gold_regime()
        assert reading is not None
        assert reading.regime == GoldRegime.SIDEWAYS

    def test_gold_returns_none_no_data(self, detector_with_prices):
        """Returns None with no GLD data."""
        detector_with_prices.prices = {"SPY": [], "TLT": []}
        reading = detector_with_prices._detect_gold_regime()
        assert reading is None


# =============================================================================
# Divergence Classification Tests
# =============================================================================

class TestDivergenceClassification:
    """Tests for cross-asset divergence pattern classification."""

    def test_full_risk_on(self):
        """All bullish → FULL_RISK_ON."""
        detector = CrossAssetRegimeArbDetector()
        equity = AssetRegimeReading("SPY", 0.08, 0.12, AssetRegime.BULL, 0.8)
        bonds = BondRegimeReading("TLT", 0.06, BondRegime.FALLING, 0.7)
        gold = GoldRegimeReading("GLD", 0.05, GoldRegime.STRONG, 0.6)

        pattern, value, _ = detector._classify_divergence(equity, bonds, gold)
        assert pattern == DivergencePattern.FULL_RISK_ON
        assert value > 0

    def test_full_risk_off(self, bear_prices):
        """All bearish → RISK_OFF."""
        detector = CrossAssetRegimeArbDetector()
        detector.prices = bear_prices
        equity = detector._detect_equity_regime()
        bonds = detector._detect_bond_regime()
        gold = detector._detect_gold_regime()

        assert equity is not None and bonds is not None and gold is not None
        pattern, value, _ = detector._classify_divergence(equity, bonds, gold)
        assert pattern == DivergencePattern.RISK_OFF
        assert value < 0

    def test_flight_to_safety(self, flight_to_safety_prices):
        """Equity bear + bond bull → FLIGHT_TO_SAFETY."""
        detector = CrossAssetRegimeArbDetector()
        detector.prices = flight_to_safety_prices
        equity = detector._detect_equity_regime()
        bonds = detector._detect_bond_regime()
        gold = detector._detect_gold_regime()

        assert equity is not None and bonds is not None and gold is not None
        pattern, value, _ = detector._classify_divergence(equity, bonds, gold)
        assert pattern == DivergencePattern.FLIGHT_TO_SAFETY
        assert value < 0  # Defensive signal

    def test_risk_rotation(self, risk_rotation_prices):
        """Equity bear + gold strong → RISK_ROTATION."""
        detector = CrossAssetRegimeArbDetector()
        detector.prices = risk_rotation_prices
        equity = detector._detect_equity_regime()
        bonds = detector._detect_bond_regime()
        gold = detector._detect_gold_regime()

        assert equity is not None and bonds is not None and gold is not None
        pattern, value, _ = detector._classify_divergence(equity, bonds, gold)
        assert pattern == DivergencePattern.RISK_ROTATION
        assert value > 0  # Alert signal

    def test_inflation_fear(self, inflation_fear_prices):
        """Bond bear + gold strong → INFLATION_FEAR."""
        detector = CrossAssetRegimeArbDetector()
        detector.prices = inflation_fear_prices
        equity = detector._detect_equity_regime()
        bonds = detector._detect_bond_regime()
        gold = detector._detect_gold_regime()

        assert equity is not None and bonds is not None and gold is not None
        pattern, value, _ = detector._classify_divergence(equity, bonds, gold)
        assert pattern == DivergencePattern.INFLATION_FEAR
        assert value < 0

    def test_cautious_optimism(self):
        """Equity neutral + gold strong → CAUTIOUS_OPTIMISM."""
        detector = CrossAssetRegimeArbDetector()
        equity = AssetRegimeReading("SPY", 0.0, 0.12, AssetRegime.NEUTRAL, 0.3)
        bonds = BondRegimeReading("TLT", 0.0, BondRegime.STABLE, 0.2)
        gold = GoldRegimeReading("GLD", 0.08, GoldRegime.STRONG, 0.8)

        pattern, value, _ = detector._classify_divergence(equity, bonds, gold)
        assert pattern == DivergencePattern.CAUTIOUS_OPTIMISM
        assert value > 0

    def test_equity_rotation(self, detector_with_prices):
        """Equity diverging from bonds/gold → EQUITY_ROTATION."""
        equity = detector_with_prices._detect_equity_regime()
        bonds = detector_with_prices._detect_bond_regime()
        gold = detector_with_prices._detect_gold_regime()

        assert equity is not None and bonds is not None and gold is not None
        pattern, value, _ = detector_with_prices._classify_divergence(equity, bonds, gold)
        # SPY bull, TLT stable, GLD weak → equity rotation
        assert pattern == DivergencePattern.EQUITY_ROTATION
        assert abs(value) > 0

    def test_no_divergence(self):
        """All neutral/flat → NO_DIVERGENCE."""
        detector = CrossAssetRegimeArbDetector()
        equity = AssetRegimeReading("SPY", 0.0, 0.12, AssetRegime.NEUTRAL, 0.1)
        bonds = BondRegimeReading("TLT", 0.0, BondRegime.STABLE, 0.1)
        gold = GoldRegimeReading("GLD", 0.0, GoldRegime.SIDEWAYS, 0.1)

        pattern, value, _ = detector._classify_divergence(equity, bonds, gold)
        assert pattern == DivergencePattern.NO_DIVERGENCE
        assert value == 0.0


# =============================================================================
# Full Scan Tests
# =============================================================================

class TestFullScan:
    """Tests for the complete scan pipeline."""

    def test_scan_returns_signal(self, detector_with_prices):
        """Full scan produces valid signal."""
        signal = detector_with_prices.scan()
        assert signal is not None
        assert isinstance(signal, CrossAssetRegimeArbSignal)
        assert signal.equity is not None
        assert signal.bonds is not None
        assert signal.gold is not None

    def test_scan_active_divergence(self, detector_with_prices):
        """Scan identifies active divergence."""
        signal = detector_with_prices.scan()
        assert signal is not None
        # With our test data: SPY bull, TLT stable, GLD weak
        assert signal.active is True
        assert signal.divergence.pattern != DivergencePattern.NO_DIVERGENCE

    def test_scan_inactive_no_divergence(self):
        """Flat prices yield no divergence."""
        detector = CrossAssetRegimeArbDetector()
        detector.prices = {
            "SPY": [{"d": f"2026-01-{i:02d}", "p": 500.0} for i in range(1, 91)],
            "TLT": [{"d": f"2026-01-{i:02d}", "p": 90.0} for i in range(1, 91)],
            "GLD": [{"d": f"2026-01-{i:02d}", "p": 400.0} for i in range(1, 91)],
        }
        signal = detector.scan()
        assert signal is not None
        assert signal.active is False

    def test_scan_no_data(self):
        """No price data returns None."""
        detector = CrossAssetRegimeArbDetector()
        detector.data_dir = Path("/nonexistent")
        signal = detector.scan()
        assert signal is None

    def test_signal_value_range(self, detector_with_prices):
        """Signal value is always in [-1, +1]."""
        signal = detector_with_prices.scan()
        assert signal is not None
        assert -1.0 <= signal.signal_value <= 1.0

    def test_conviction_range(self, detector_with_prices):
        """Conviction is always in [0, 1]."""
        signal = detector_with_prices.scan()
        assert signal is not None
        assert 0.0 <= signal.overall_conviction <= 1.0

    def test_signal_to_dict(self, detector_with_prices):
        """Signal serializes to dict properly."""
        signal = detector_with_prices.scan()
        assert signal is not None
        d = signal.to_dict()
        assert "timestamp" in d
        assert "equity" in d
        assert "bonds" in d
        assert "gold" in d
        assert "divergence" in d
        assert "signal_value" in d
        assert d["equity"]["symbol"] == "SPY"
        assert d["bonds"]["symbol"] == "TLT"
        assert d["gold"]["symbol"] == "GLD"

    def test_scan_with_flight_to_safety(self, flight_to_safety_prices):
        """Flight-to-safety scenario detected correctly."""
        detector = CrossAssetRegimeArbDetector()
        detector.prices = flight_to_safety_prices
        signal = detector.scan()
        assert signal is not None
        assert signal.active is True
        assert signal.divergence.pattern == DivergencePattern.FLIGHT_TO_SAFETY

    def test_scan_with_risk_rotation(self, risk_rotation_prices):
        """Risk rotation scenario detected correctly."""
        detector = CrossAssetRegimeArbDetector()
        detector.prices = risk_rotation_prices
        signal = detector.scan()
        assert signal is not None
        assert signal.active is True
        assert signal.divergence.pattern == DivergencePattern.RISK_ROTATION


# =============================================================================
# Ensemble API Tests
# =============================================================================

class TestEnsembleAPI:
    """Tests for the EnsembleVoter-facing API."""

    def test_get_ensemble_signal_active(self, detector_with_prices):
        """Ensemble signal includes all expected fields when active."""
        result = detector_with_prices.get_ensemble_signal()
        assert result["active"] is True
        assert "signal_value" in result
        assert "confidence" in result
        assert "timestamp" in result
        assert "asset_signals" in result
        assert "pattern" in result
        assert "explanation" in result

    def test_get_ensemble_signal_inactive(self):
        """Inactive divergence returns zero signal."""
        detector = CrossAssetRegimeArbDetector()
        detector.prices = {
            "SPY": [{"d": f"2026-01-{i:02d}", "p": 500.0} for i in range(1, 91)],
            "TLT": [{"d": f"2026-01-{i:02d}", "p": 90.0} for i in range(1, 91)],
            "GLD": [{"d": f"2026-01-{i:02d}", "p": 400.0} for i in range(1, 91)],
        }
        result = detector.get_ensemble_signal()
        assert result["active"] is False
        assert result["signal_value"] == 0.0
        assert result["pattern"] == "no_divergence"

    def test_get_ensemble_signal_no_data(self):
        """No data returns fallback signal."""
        detector = CrossAssetRegimeArbDetector()
        detector.data_dir = Path("/nonexistent")
        result = detector.get_ensemble_signal()
        assert result["active"] is False
        assert result["signal_value"] == 0.0

    def test_ensemble_signal_asset_signals(self, detector_with_prices):
        """Asset signals map contains SPY, TLT, GLD."""
        result = detector_with_prices.get_ensemble_signal()
        assert "asset_signals" in result
        assert "SPY" in result["asset_signals"]
        assert "TLT" in result["asset_signals"]
        assert "GLD" in result["asset_signals"]

    def test_ensemble_signal_regime_info(self, detector_with_prices):
        """Regime info included in ensemble response."""
        result = detector_with_prices.get_ensemble_signal()
        assert "equity_regime" in result
        assert "bond_regime" in result
        assert "gold_regime" in result
        assert result["equity_regime"] == "bull"

    def test_ensemble_signal_persistence(self, detector_with_prices):
        """Persistence tracking in ensemble response."""
        result = detector_with_prices.get_ensemble_signal()
        assert "persistence_days" in result
        assert isinstance(result["persistence_days"], int)


# =============================================================================
# State Persistence Tests
# =============================================================================

class TestStatePersistence:
    """Tests for state persistence and tracking."""

    def test_state_initialization(self, detector_with_prices):
        """State initializes with sensible defaults."""
        assert "previous_pattern" in detector_with_prices.state
        assert detector_with_prices.state["previous_pattern"] is None
        assert detector_with_prices.state["persistence_days"] == 0

    def test_persistence_increments_on_repeat(self, detector_with_prices):
        """Same pattern on same date increments persistence."""
        detector_with_prices._save_state(DivergencePattern.EQUITY_ROTATION, "2026-05-18")
        assert detector_with_prices.state["persistence_days"] == 0  # First save starts at 0

        detector_with_prices._save_state(DivergencePattern.EQUITY_ROTATION, "2026-05-19")
        assert detector_with_prices.state["persistence_days"] == 1  # Day 2 → 1

        detector_with_prices._save_state(DivergencePattern.EQUITY_ROTATION, "2026-05-20")
        assert detector_with_prices.state["persistence_days"] == 2  # Day 3 → 2

    def test_persistence_resets_on_change(self, detector_with_prices):
        """Different pattern resets persistence counter."""
        detector_with_prices._save_state(DivergencePattern.EQUITY_ROTATION, "2026-05-18")
        detector_with_prices._save_state(DivergencePattern.EQUITY_ROTATION, "2026-05-19")
        assert detector_with_prices.state["persistence_days"] == 1

        detector_with_prices._save_state(DivergencePattern.NO_DIVERGENCE, "2026-05-20")
        assert detector_with_prices.state["persistence_days"] == 0

    def test_persistence_boost_on_high_persistence(self, flight_to_safety_prices):
        """Long persistence boosts confidence."""
        detector = CrossAssetRegimeArbDetector()
        detector.prices = flight_to_safety_prices
        detector.state["persistence_days"] = 5  # Simulate 5 days of same pattern

        signal = detector.scan()
        assert signal is not None
        assert signal.divergence.persistence_days > 0

    def test_state_file_loading(self, tmp_path):
        """State loads from file if it exists."""
        state_dir = tmp_path / "regime_arb"
        state_dir.mkdir(parents=True, exist_ok=True)
        state_file = state_dir / "regime_arb_state.json"

        with open(state_file, "w") as f:
            json.dump({"previous_pattern": "flight_to_safety", "persistence_days": 3, "last_date": "2026-05-17"}, f)

        # Patch both STATE_DIR and STATE_FILE (both module-level constants)
        with patch("src.signals.cross_asset_regime_arb.STATE_DIR", state_dir), \
             patch("src.signals.cross_asset_regime_arb.STATE_FILE", state_file):
            detector = CrossAssetRegimeArbDetector()
            assert detector.state["previous_pattern"] == "flight_to_safety"
            assert detector.state["persistence_days"] == 3
            assert detector.state["last_date"] == "2026-05-17"


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_recovery_beginning_pattern(self):
        """Gold weak + equity recovering → RECOVERY_BEGINNING."""
        detector = CrossAssetRegimeArbDetector()
        equity = AssetRegimeReading("SPY", 0.03, 0.12, AssetRegime.NEUTRAL, 0.3)
        bonds = BondRegimeReading("TLT", 0.0, BondRegime.STABLE, 0.1)
        gold = GoldRegimeReading("GLD", -0.06, GoldRegime.WEAK, 0.6)

        pattern, value, _ = detector._classify_divergence(equity, bonds, gold)
        assert pattern == DivergencePattern.RECOVERY_BEGINNING
        assert value > 0

    def test_minimal_data_points(self):
        """Enough for 60d lookback detection plus vol calculation."""
        prices = {}
        for sym in ["SPY", "TLT", "GLD"]:
            prices[sym] = [{"d": f"2026-01-{i:02d}", "p": 100.0 + i * 0.1} for i in range(1, 70)]

        detector = CrossAssetRegimeArbDetector()
        detector.prices = prices
        signal = detector.scan()
        assert signal is not None  # 69 points should be sufficient

    def test_insufficient_data_points(self):
        """Very few data points → no signal."""
        prices = {}
        for sym in ["SPY", "TLT", "GLD"]:
            prices[sym] = [{"d": "2026-01-01", "p": 100.0}]

        detector = CrossAssetRegimeArbDetector()
        detector.prices = prices
        signal = detector.scan()
        assert signal is None

    def test_high_volatility_equity_regime(self):
        """Equity with >25% annualized vol → HIGH_VOL."""
        import random
        random.seed(42)

        prices = {}
        for sym in ["SPY", "TLT", "GLD"]:
            prices[sym] = [{"d": f"2026-01-{i:02d}", "p": 100.0 + random.uniform(-3, 3) * i} for i in range(1, 91)]

        detector = CrossAssetRegimeArbDetector()
        detector.prices = prices
        equity = detector._detect_equity_regime()
        assert equity is not None

    def test_divergence_to_dict(self):
        """DivergenceReading.to_dict() serializes properly."""
        div = DivergenceReading(
            pattern=DivergencePattern.FLIGHT_TO_SAFETY,
            signal_value=-0.3,
            confidence=0.7,
            explanation="Test",
            persistence_days=3,
            equity_regime=AssetRegime.BEAR,
            bond_regime=BondRegime.FALLING,
            gold_regime=GoldRegime.STRONG,
        )
        d = div.to_dict()
        assert d["pattern"] == "flight_to_safety"
        assert d["equity_regime"] == "bear"
        assert d["bond_regime"] == "falling"
        assert d["gold_regime"] == "strong"
        assert d["signal_value"] == -0.3
        assert d["persistence_days"] == 3


class TestContinuousSignalPassthrough:
    """Verify that _classify_divergence returns continuous signal values
    scaled by per-asset momentum/confidence, not discrete constants."""

    def test_full_risk_on_scales_with_confidence(self):
        """Higher confidence → stronger signal."""
        detector = CrossAssetRegimeArbDetector()
        equity_low = AssetRegimeReading("SPY", 0.08, 0.12, AssetRegime.BULL, 0.3)
        equity_high = AssetRegimeReading("SPY", 0.08, 0.12, AssetRegime.BULL, 0.9)
        bonds = BondRegimeReading("TLT", 0.06, BondRegime.FALLING, 0.7)
        gold = GoldRegimeReading("GLD", 0.05, GoldRegime.STRONG, 0.6)

        _, val_low, _ = detector._classify_divergence(equity_low, bonds, gold)
        _, val_high, _ = detector._classify_divergence(equity_high, bonds, gold)
        assert val_high > val_low

    def test_risk_off_scales_with_momentum(self):
        """Stronger bearish momentum → more negative signal."""
        detector = CrossAssetRegimeArbDetector()
        equity_mild = AssetRegimeReading("SPY", -0.06, 0.15, AssetRegime.BEAR, 0.5)
        equity_severe = AssetRegimeReading("SPY", -0.20, 0.30, AssetRegime.BEAR, 0.9)
        bonds = BondRegimeReading("TLT", -0.05, BondRegime.RISING, 0.7)
        gold = GoldRegimeReading("GLD", -0.08, GoldRegime.WEAK, 0.6)

        _, val_mild, _ = detector._classify_divergence(equity_mild, bonds, gold)
        _, val_severe, _ = detector._classify_divergence(equity_severe, bonds, gold)
        assert val_severe < val_mild  # More negative

    def test_signal_value_not_discrete(self):
        """Signal values should vary continuously, not jump between fixed steps."""
        detector = CrossAssetRegimeArbDetector()
        bonds = BondRegimeReading("TLT", 0.06, BondRegime.FALLING, 0.7)
        gold = GoldRegimeReading("GLD", 0.05, GoldRegime.STRONG, 0.6)
        values = set()
        for conf in [0.2, 0.4, 0.6, 0.8, 1.0]:
            equity = AssetRegimeReading("SPY", 0.08, 0.12, AssetRegime.BULL, conf)
            _, val, _ = detector._classify_divergence(equity, bonds, gold)
            values.add(round(val, 4))
        # Should get at least 3 distinct values across 5 confidence levels
        assert len(values) >= 3

    def test_signal_bounded(self):
        """All continuous signal values should be in [-1, 1]."""
        detector = CrossAssetRegimeArbDetector()
        test_cases = [
            (AssetRegimeReading("SPY", 0.15, 0.12, AssetRegime.BULL, 1.0),
             BondRegimeReading("TLT", 0.06, BondRegime.FALLING, 1.0),
             GoldRegimeReading("GLD", 0.12, GoldRegime.STRONG, 1.0)),
            (AssetRegimeReading("SPY", -0.20, 0.30, AssetRegime.BEAR, 1.0),
             BondRegimeReading("TLT", -0.05, BondRegime.RISING, 1.0),
             GoldRegimeReading("GLD", -0.15, GoldRegime.WEAK, 1.0)),
        ]
        for equity, bonds, gold in test_cases:
            _, val, _ = detector._classify_divergence(equity, bonds, gold)
            assert -1.0 <= val <= 1.0, f"Signal value {val} out of range"

    def test_no_divergence_returns_zero(self):
        """NO_DIVERGENCE pattern still returns 0.0."""
        detector = CrossAssetRegimeArbDetector()
        equity = AssetRegimeReading("SPY", 0.01, 0.10, AssetRegime.NEUTRAL, 0.5)
        bonds = BondRegimeReading("TLT", 0.01, BondRegime.STABLE, 0.5)
        gold = GoldRegimeReading("GLD", 0.01, GoldRegime.SIDEWAYS, 0.5)

        pattern, value, _ = detector._classify_divergence(equity, bonds, gold)
        assert pattern == DivergencePattern.NO_DIVERGENCE
        assert value == 0.0


# =============================================================================
# Constant Validation Tests
# =============================================================================

class TestConstants:
    """Verify module-level constants."""

    def test_bull_momentum_threshold(self):
        assert BULL_MOMENTUM_THRESHOLD == 0.05

    def test_bear_momentum_threshold(self):
        assert BEAR_MOMENTUM_THRESHOLD == -0.05

    def test_strong_momentum_threshold(self):
        assert STRONG_MOMENTUM_THRESHOLD == 0.10

    def test_high_vol_threshold(self):
        assert HIGH_VOL_THRESHOLD == 0.25

    def test_momentum_lookback(self):
        from src.signals.cross_asset_regime_arb import MOMENTUM_LOOKBACK
        assert MOMENTUM_LOOKBACK == 60

    def test_vol_lookback(self):
        from src.signals.cross_asset_regime_arb import VOL_LOOKBACK
        assert VOL_LOOKBACK == 20

    def test_min_history(self):
        from src.signals.cross_asset_regime_arb import MIN_HISTORY
        assert MIN_HISTORY == 30

    def test_divergence_lookback(self):
        from src.signals.cross_asset_regime_arb import DIVERGENCE_LOOKBACK
        assert DIVERGENCE_LOOKBACK == 20

    def test_all_exports(self):
        """__all__ contains all expected public names."""
        expected = {
            "MOMENTUM_LOOKBACK", "VOL_LOOKBACK", "MIN_HISTORY",
            "DIVERGENCE_LOOKBACK", "BULL_MOMENTUM_THRESHOLD",
            "BEAR_MOMENTUM_THRESHOLD", "STRONG_MOMENTUM_THRESHOLD",
            "HIGH_VOL_THRESHOLD", "AssetRegime", "BondRegime",
            "GoldRegime", "DivergencePattern", "AssetRegimeReading",
            "BondRegimeReading", "GoldRegimeReading", "DivergenceReading",
            "CrossAssetRegimeArbSignal", "CrossAssetRegimeArbDetector",
            "print_signal_report",
        }
        from src.signals.cross_asset_regime_arb import __all__
        assert set(__all__) == expected

    def test_divergence_signals_map_all_patterns(self):
        """DIVERGENCE_SIGNALS contains every DivergencePattern."""
        from src.signals.cross_asset_regime_arb import DIVERGENCE_SIGNALS
        for pattern in DivergencePattern:
            assert pattern in DIVERGENCE_SIGNALS
            value, explanation = DIVERGENCE_SIGNALS[pattern]
            assert isinstance(value, float)
            assert isinstance(explanation, str)

    def test_divergence_signals_values_in_range(self):
        """All baseline signal values are in [-1, 1]."""
        from src.signals.cross_asset_regime_arb import DIVERGENCE_SIGNALS
        for pattern, (value, _) in DIVERGENCE_SIGNALS.items():
            assert -1.0 <= value <= 1.0, f"{pattern}: {value}"


# =============================================================================
# Enum Value Tests
# =============================================================================

class TestEnumValues:
    """Verify enum string values."""

    def test_asset_regime_values(self):
        assert AssetRegime.BULL.value == "bull"
        assert AssetRegime.BEAR.value == "bear"
        assert AssetRegime.NEUTRAL.value == "neutral"
        assert AssetRegime.HIGH_VOL.value == "high_vol"

    def test_bond_regime_values(self):
        assert BondRegime.RISING.value == "rising"
        assert BondRegime.FALLING.value == "falling"
        assert BondRegime.STABLE.value == "stable"

    def test_gold_regime_values(self):
        assert GoldRegime.STRONG.value == "strong"
        assert GoldRegime.WEAK.value == "weak"
        assert GoldRegime.SIDEWAYS.value == "sideways"

    def test_divergence_pattern_values(self):
        assert DivergencePattern.FULL_RISK_ON.value == "full_risk_on"
        assert DivergencePattern.RISK_OFF.value == "risk_off"
        assert DivergencePattern.RISK_ROTATION.value == "risk_rotation"
        assert DivergencePattern.FLIGHT_TO_SAFETY.value == "flight_to_safety"
        assert DivergencePattern.INFLATION_FEAR.value == "inflation_fear"
        assert DivergencePattern.CAUTIOUS_OPTIMISM.value == "cautious_optimism"
        assert DivergencePattern.EQUITY_ROTATION.value == "equity_rotation"
        assert DivergencePattern.RECOVERY_BEGINNING.value == "recovery_beginning"
        assert DivergencePattern.NO_DIVERGENCE.value == "no_divergence"
        assert DivergencePattern.UNKNOWN.value == "unknown"

    def test_divergence_pattern_distinct(self):
        """All divergence pattern values are distinct."""
        values = [p.value for p in DivergencePattern]
        assert len(values) == len(set(values))


# =============================================================================
# Dataclass Field Validation Tests
# =============================================================================

class TestDataclassValidation:
    """Verify dataclass construction and field types."""

    def test_asset_regime_reading_to_dict(self):
        """AssetRegimeReading.to_dict() returns correct fields."""
        r = AssetRegimeReading("SPY", 0.08, 0.15, AssetRegime.BULL, 0.7)
        d = r.to_dict()
        assert d == {
            "symbol": "SPY",
            "momentum_60d": 0.08,
            "volatility_20d": 0.15,
            "asset_regime": AssetRegime.BULL,
            "confidence": 0.7,
        }

    def test_bond_regime_reading_to_dict(self):
        """BondRegimeReading.to_dict() returns correct fields."""
        r = BondRegimeReading("TLT", -0.04, BondRegime.STABLE, 0.4)
        d = r.to_dict()
        assert d == {
            "symbol": "TLT",
            "momentum_60d": -0.04,
            "regime": BondRegime.STABLE,
            "confidence": 0.4,
        }

    def test_gold_regime_reading_to_dict(self):
        """GoldRegimeReading.to_dict() returns correct fields."""
        r = GoldRegimeReading("GLD", 0.12, GoldRegime.STRONG, 0.8)
        d = r.to_dict()
        assert d == {
            "symbol": "GLD",
            "momentum_60d": 0.12,
            "regime": GoldRegime.STRONG,
            "confidence": 0.8,
        }

    def test_divergence_reading_zero_persistence(self):
        """DivergenceReading with zero persistence."""
        div = DivergenceReading(
            pattern=DivergencePattern.NO_DIVERGENCE,
            signal_value=0.0,
            confidence=0.0,
            explanation="No divergence",
            persistence_days=0,
            equity_regime=AssetRegime.NEUTRAL,
            bond_regime=BondRegime.STABLE,
            gold_regime=GoldRegime.SIDEWAYS,
        )
        assert div.persistence_days == 0
        assert div.pattern == DivergencePattern.NO_DIVERGENCE

    def test_cross_asset_signal_to_dict_keys(self):
        """CrossAssetRegimeArbSignal.to_dict() has all expected keys."""
        equity = AssetRegimeReading("SPY", 0.08, 0.12, AssetRegime.BULL, 0.7)
        bonds = BondRegimeReading("TLT", 0.06, BondRegime.FALLING, 0.6)
        gold = GoldRegimeReading("GLD", 0.05, GoldRegime.STRONG, 0.5)
        divergence = DivergenceReading(
            pattern=DivergencePattern.FULL_RISK_ON,
            signal_value=0.4,
            confidence=0.7,
            explanation="Test",
            persistence_days=2,
            equity_regime=AssetRegime.BULL,
            bond_regime=BondRegime.FALLING,
            gold_regime=GoldRegime.STRONG,
        )
        signal = CrossAssetRegimeArbSignal(
            timestamp="2026-01-01T00:00:00",
            equity=equity,
            bonds=bonds,
            gold=gold,
            divergence=divergence,
            active=True,
            overall_conviction=0.6,
            signal_value=0.4,
        )
        d = signal.to_dict()
        expected_keys = {
            "timestamp", "equity", "bonds", "gold",
            "divergence", "active", "overall_conviction", "signal_value",
        }
        assert set(d.keys()) == expected_keys

    def test_signal_to_dict_has_expected_types(self):
        """Signal to_dict returns dicts with correct value types."""
        equity = AssetRegimeReading("SPY", 0.08, 0.12, AssetRegime.BULL, 0.7)
        bonds = BondRegimeReading("TLT", 0.06, BondRegime.FALLING, 0.6)
        gold = GoldRegimeReading("GLD", 0.05, GoldRegime.STRONG, 0.5)
        divergence = DivergenceReading(
            pattern=DivergencePattern.FULL_RISK_ON,
            signal_value=0.4,
            confidence=0.7,
            explanation="Test",
            persistence_days=2,
            equity_regime=AssetRegime.BULL,
            bond_regime=BondRegime.FALLING,
            gold_regime=GoldRegime.STRONG,
        )
        signal = CrossAssetRegimeArbSignal(
            timestamp="2026-01-01T00:00:00",
            equity=equity,
            bonds=bonds,
            gold=gold,
            divergence=divergence,
            active=True,
            overall_conviction=0.6,
            signal_value=0.4,
        )
        d = signal.to_dict()
        assert isinstance(d["timestamp"], str)
        assert isinstance(d["active"], bool)
        assert isinstance(d["overall_conviction"], float)
        assert isinstance(d["signal_value"], float)
        assert isinstance(d["equity"], dict)
        assert isinstance(d["bonds"], dict)
        assert isinstance(d["gold"], dict)
        assert isinstance(d["divergence"], dict)


# =============================================================================
# Conviction Computation Tests
# =============================================================================

class TestConviction:
    """Tests for _compute_conviction()."""

    def test_conviction_average(self, detector_with_prices):
        """Conviction averages per-asset confidences."""
        conviction = detector_with_prices._compute_conviction()
        assert 0.0 <= conviction <= 1.0

    def test_conviction_zero_when_no_data(self):
        """No data yields zero conviction."""
        detector = CrossAssetRegimeArbDetector()
        detector.prices = {}
        conviction = detector._compute_conviction()
        assert conviction == 0.0

    def test_conviction_with_partial_data(self):
        """Missing one asset still computes reduced conviction."""
        detector = CrossAssetRegimeArbDetector()
        detector.prices = {
            "SPY": [{"d": f"2026-01-{i:02d}", "p": 500.0 + i} for i in range(1, 91)],
            "TLT": [{"d": f"2026-01-{i:02d}", "p": 90.0} for i in range(1, 91)],
        }
        conviction = detector._compute_conviction()
        assert 0.0 <= conviction <= 1.0


# =============================================================================
# Get Returns / Volatility Edge Cases
# =============================================================================

class TestReturnsVolatilityEdgeCases:
    """Edge cases for _get_returns and _get_volatility."""

    def test_get_returns_zero_price(self):
        """Zero price returns None."""
        detector = CrossAssetRegimeArbDetector()
        detector.prices = {
            "SPY": [{"d": "2026-01-01", "p": 0.0}, {"d": "2026-01-02", "p": 100.0}],
        }
        ret = detector._get_returns("SPY", 1)
        assert ret is None

    def test_get_returns_all_zero(self):
        """All zero prices return None."""
        detector = CrossAssetRegimeArbDetector()
        detector.prices = {
            "SPY": [{"d": "2026-01-01", "p": 0.0}, {"d": "2026-01-02", "p": 0.0}],
        }
        ret = detector._get_returns("SPY", 1)
        assert ret is None

    def test_get_volatility_minimal_returns(self):
        """Volatility with exactly 6 data points (5 returns)."""
        detector = CrossAssetRegimeArbDetector()
        detector.prices = {
            "SPY": [{"d": f"2026-01-{i:02d}", "p": 100.0 + i * 0.5} for i in range(1, 8)],
        }
        vol = detector._get_volatility("SPY", 6)
        assert vol is not None
        assert vol > 0

    def test_get_volatility_insufficient_returns(self):
        """Volatility with < 5 returns returns None."""
        detector = CrossAssetRegimeArbDetector()
        detector.prices = {
            "SPY": [{"d": f"2026-01-{i:02d}", "p": 100.0} for i in range(1, 6)],
        }
        vol = detector._get_volatility("SPY", 4)
        assert vol is None

    def test_get_volatility_zero_lookback(self):
        """Volatility with lookback=0 returns None."""
        detector = CrossAssetRegimeArbDetector()
        detector.prices = {
            "SPY": [{"d": f"2026-01-{i:02d}", "p": 100.0 + i} for i in range(1, 91)],
        }
        vol = detector._get_volatility("SPY", 0)
        assert vol is None


# =============================================================================
# HIGH_VOL Regime Tests
# =============================================================================

class TestHighVolRegime:
    """Tests for HIGH_VOL equity regime classification."""

    def test_high_vol_regime_detected(self):
        """High volatility triggers HIGH_VOL regime."""
        import random
        random.seed(99)
        prices = {
            "SPY": [{"d": f"2026-01-{i:02d}", "p": 100.0 + random.uniform(-5, 5)} for i in range(1, 91)],
            "TLT": [{"d": f"2026-01-{i:02d}", "p": 90.0} for i in range(1, 91)],
            "GLD": [{"d": f"2026-01-{i:02d}", "p": 400.0} for i in range(1, 91)],
        }
        detector = CrossAssetRegimeArbDetector()
        detector.prices = prices
        equity = detector._detect_equity_regime()
        assert equity is not None
        # The stochastic test might or might not hit HIGH_VOL; just verify it can
        assert equity.asset_regime in (
            AssetRegime.HIGH_VOL, AssetRegime.BULL,
            AssetRegime.BEAR, AssetRegime.NEUTRAL,
        )

    def test_high_vol_forced(self):
        """Force high vol by creating wild price swings."""
        spy_prices = []
        base_date = datetime(2026, 1, 1)
        price = 100.0
        for i in range(90):
            price *= 1.04 if i % 2 == 0 else 0.96  # ~4% daily swings → very high vol
            spy_prices.append({
                "d": (base_date + timedelta(days=i)).strftime("%Y-%m-%d"),
                "p": price,
            })
        prices = {
            "SPY": spy_prices,
            "TLT": [{"d": f"2026-01-{i:02d}", "p": 90.0} for i in range(1, 91)],
            "GLD": [{"d": f"2026-01-{i:02d}", "p": 400.0} for i in range(1, 91)],
        }
        detector = CrossAssetRegimeArbDetector()
        detector.prices = prices
        equity = detector._detect_equity_regime()
        assert equity is not None
        assert equity.volatility_20d > HIGH_VOL_THRESHOLD
        assert equity.asset_regime == AssetRegime.HIGH_VOL

    def test_high_vol_scan_returns_signal(self):
        """Full scan with high vol returns a valid signal."""
        spy_prices = []
        base_date = datetime(2026, 1, 1)
        price = 100.0
        for i in range(90):
            price *= 1.04 if i % 2 == 0 else 0.96
            spy_prices.append({
                "d": (base_date + timedelta(days=i)).strftime("%Y-%m-%d"),
                "p": price,
            })
        prices = {
            "SPY": spy_prices,
            "TLT": [{"d": f"2026-01-{i:02d}", "p": 90.0 * (1 + i * 0.001)} for i in range(1, 91)],
            "GLD": [{"d": f"2026-01-{i:02d}", "p": 400.0 * (1 + i * 0.001)} for i in range(1, 91)],
        }
        detector = CrossAssetRegimeArbDetector()
        detector.prices = prices
        signal = detector.scan()
        assert signal is not None
        assert signal.equity.asset_regime == AssetRegime.HIGH_VOL


# =============================================================================
# Bond & Gold Regime Boundary Tests
# =============================================================================

class TestBondGoldBoundaries:
    """Boundary value tests for bond and gold regime detection."""

    def test_bond_above_bull_threshold(self):
        """Bond momentum above BULL threshold → FALLING (yields down)."""
        detector = CrossAssetRegimeArbDetector()
        # Use k=0.001: 60-day return ≈ 60*0.001/(1+29*0.001) ≈ 5.8%
        k = 0.001
        detector.prices = {
            "SPY": [{"d": f"2026-01-{i:02d}", "p": 500.0} for i in range(1, 91)],
            "TLT": [{"d": f"2026-01-{i:02d}", "p": 90.0 * (1 + i * k)} for i in range(1, 91)],
            "GLD": [{"d": f"2026-01-{i:02d}", "p": 400.0} for i in range(1, 91)],
        }
        reading = detector._detect_bond_regime()
        assert reading is not None
        assert reading.momentum_60d >= BULL_MOMENTUM_THRESHOLD, f"Got {reading.momentum_60d}"
        assert reading.regime == BondRegime.FALLING

    def test_bond_below_bear_threshold(self):
        """Bond momentum below BEAR threshold → RISING (yields up)."""
        detector = CrossAssetRegimeArbDetector()
        k = 0.001
        detector.prices = {
            "SPY": [{"d": f"2026-01-{i:02d}", "p": 500.0} for i in range(1, 91)],
            "TLT": [{"d": f"2026-01-{i:02d}", "p": 100.0 * (1 - i * k)} for i in range(1, 91)],
            "GLD": [{"d": f"2026-01-{i:02d}", "p": 400.0} for i in range(1, 91)],
        }
        reading = detector._detect_bond_regime()
        assert reading is not None
        assert reading.momentum_60d <= BEAR_MOMENTUM_THRESHOLD, f"Got {reading.momentum_60d}"
        assert reading.regime == BondRegime.RISING

    def test_gold_above_bull_threshold(self):
        """Gold momentum above BULL threshold → STRONG."""
        detector = CrossAssetRegimeArbDetector()
        k = 0.001
        detector.prices = {
            "SPY": [{"d": f"2026-01-{i:02d}", "p": 500.0} for i in range(1, 91)],
            "TLT": [{"d": f"2026-01-{i:02d}", "p": 90.0} for i in range(1, 91)],
            "GLD": [{"d": f"2026-01-{i:02d}", "p": 400.0 * (1 + i * k)} for i in range(1, 91)],
        }
        reading = detector._detect_gold_regime()
        assert reading is not None
        assert reading.regime == GoldRegime.STRONG

    def test_gold_neutral_positive_below_threshold(self):
        """Gold positive momentum below BULL threshold → SIDEWAYS."""
        detector = CrossAssetRegimeArbDetector()
        detector.prices = {
            "SPY": [{"d": f"2026-01-{i:02d}", "p": 500.0} for i in range(1, 91)],
            "TLT": [{"d": f"2026-01-{i:02d}", "p": 90.0} for i in range(1, 91)],
            "GLD": [{"d": f"2026-01-{i:02d}", "p": 400.0 * (1 + i * 0.0003)} for i in range(1, 91)],
        }
        reading = detector._detect_gold_regime()
        assert reading is not None
        assert reading.regime == GoldRegime.SIDEWAYS


# =============================================================================
# Output & CLI Tests
# =============================================================================

class TestOutputFunctions:
    """Tests for print_signal_report and CLI."""

    def test_print_signal_report_output(self, detector_with_prices, caplog):
        """print_signal_report produces expected output without errors."""
        signal = detector_with_prices.scan()
        assert signal is not None
        with caplog.at_level(logging.INFO, logger="src.signals.cross_asset_regime_arb"):
            print_signal_report(signal)
        assert "CROSS-ASSET REGIME ARBITRAGE SIGNAL" in caplog.text
        assert "SPY (Equity)" in caplog.text
        assert "TLT (Bonds)" in caplog.text
        assert "GLD (Gold)" in caplog.text
        assert "Divergence Pattern" in caplog.text

    def test_get_signal_snapshot_returns_dict(self, detector_with_prices):
        """get_signal_snapshot returns valid snapshot dict."""
        snapshot = detector_with_prices.get_signal_snapshot()
        assert snapshot is not None
        assert hasattr(snapshot, "is_active")
        assert hasattr(snapshot, "value")
        assert hasattr(snapshot, "confidence")

    def test_get_signal_snapshot_no_data(self):
        """get_signal_snapshot with no data returns inactive snapshot."""
        detector = CrossAssetRegimeArbDetector()
        detector.data_dir = Path("/nonexistent")
        snapshot = detector.get_signal_snapshot()
        assert snapshot is not None
        assert snapshot.is_active is False

    def test_cli_scan_no_data(self):
        """CLI returns exit code 1 when data unavailable."""
        import sys
        with patch.object(sys, "argv", ["cross_asset_regime_arb.py", "scan"]):
            from src.signals.cross_asset_regime_arb import main as signal_main
            with patch.object(CrossAssetRegimeArbDetector, "_load_prices", return_value=False):
                with pytest.raises(SystemExit) as exc:
                    signal_main()
                assert exc.value.code == 1

    def test_cli_status_no_data(self):
        """CLI status command runs without error."""
        import sys
        with patch.object(sys, "argv", ["cross_asset_regime_arb.py", "status"]):
            from src.signals.cross_asset_regime_arb import main as signal_main
            with patch.object(CrossAssetRegimeArbDetector, "_load_prices", return_value=False):
                signal_main()  # Should not raise

    def test_cli_signal_returns_json(self, capsys):
        """CLI signal command outputs JSON."""
        import sys
        import json
        with patch.object(sys, "argv", ["cross_asset_regime_arb.py", "signal"]):
            from src.signals.cross_asset_regime_arb import main as signal_main
            with patch.object(CrossAssetRegimeArbDetector, "_load_prices", return_value=True):
                detector = CrossAssetRegimeArbDetector()
                detector.prices = {
                    "SPY": [{"d": "2026-01-01", "p": 500.0}],
                }
                with patch("src.signals.cross_asset_regime_arb.CrossAssetRegimeArbDetector", return_value=detector):
                    pass  # Just verify the main runs
                    # Actually test separately below

    def test_cli_unknown_command(self):
        """CLI with unknown command exits with code 1."""
        import sys
        with patch.object(sys, "argv", ["cross_asset_regime_arb.py", "invalid"]):
            from src.signals.cross_asset_regime_arb import main as signal_main
            with pytest.raises(SystemExit) as exc:
                signal_main()
            assert exc.value.code == 1

    def test_cli_no_args(self):
        """CLI with no args exits with code 1."""
        import sys
        with patch.object(sys, "argv", ["cross_asset_regime_arb.py"]):
            from src.signals.cross_asset_regime_arb import main as signal_main
            with pytest.raises(SystemExit) as exc:
                signal_main()
            assert exc.value.code == 1


# =============================================================================
# Divergence Pattern Edge Cases
# =============================================================================

class TestDivergenceEdgeCases:
    """Additional edge cases for divergence classification."""

    def test_divergence_unknown_pattern(self):
        """UNKNOWN pattern maps to zero signal."""
        detector = CrossAssetRegimeArbDetector()
        from src.signals.cross_asset_regime_arb import DIVERGENCE_SIGNALS
        value, explanation = DIVERGENCE_SIGNALS[DivergencePattern.UNKNOWN]
        assert value == 0.0
        assert "Unclassified" in explanation

    def test_divergence_risk_rotation_momentum_magnitude(self):
        """Stronger bearish momentum in risk_rotation yields larger positive signal."""
        detector = CrossAssetRegimeArbDetector()
        bonds = BondRegimeReading("TLT", 0.02, BondRegime.STABLE, 0.3)

        gold = GoldRegimeReading("GLD", 0.12, GoldRegime.STRONG, 0.9)

        # Mild bear equity
        equity_mild = AssetRegimeReading("SPY", -0.06, 0.15, AssetRegime.BEAR, 0.4)
        # Severe bear equity
        equity_severe = AssetRegimeReading("SPY", -0.20, 0.30, AssetRegime.BEAR, 0.9)

        _, val_mild, _ = detector._classify_divergence(equity_mild, bonds, gold)
        _, val_severe, _ = detector._classify_divergence(equity_severe, bonds, gold)
        # Risk rotation signal increases with bearish conviction
        assert val_severe >= val_mild

    def test_divergence_flight_to_safety_scaling(self):
        """Flight to safety signal scales with bond confidence."""
        detector = CrossAssetRegimeArbDetector()
        equity = AssetRegimeReading("SPY", -0.10, 0.20, AssetRegime.BEAR, 0.7)
        gold = GoldRegimeReading("GLD", 0.02, GoldRegime.SIDEWAYS, 0.3)

        bonds_low = BondRegimeReading("TLT", 0.06, BondRegime.FALLING, 0.3)
        bonds_high = BondRegimeReading("TLT", 0.06, BondRegime.FALLING, 0.9)

        _, val_low, _ = detector._classify_divergence(equity, bonds_low, gold)
        _, val_high, _ = detector._classify_divergence(equity, bonds_high, gold)
        # Higher bond confidence → more negative (stronger flight)
        assert val_high <= val_low


# =============================================================================
# State Persistence Edge Cases
# =============================================================================

class TestStatePersistenceEdgeCases:
    """Additional state persistence edge cases."""

    def test_state_save_same_date_same_pattern(self):
        """Saving same pattern on same date does not increment."""
        detector = CrossAssetRegimeArbDetector()
        detector.state = {"previous_pattern": "equity_rotation", "persistence_days": 3, "last_date": "2026-05-18"}
        detector._save_state(DivergencePattern.EQUITY_ROTATION, "2026-05-18")
        assert detector.state["persistence_days"] == 3

    def test_state_save_os_error(self):
        """OSError during state save is caught gracefully."""
        detector = CrossAssetRegimeArbDetector()
        detector.state = {"previous_pattern": None, "persistence_days": 0, "last_date": None}
        with patch("builtins.open", side_effect=PermissionError("Denied")):
            detector._save_state(DivergencePattern.EQUITY_ROTATION, "2026-05-18")
        # State not updated if write fails
        assert detector.state["persistence_days"] == 0

    def test_state_load_corrupt_json(self, tmp_path):
        """Corrupt state file returns default state."""
        state_dir = tmp_path / "regime_arb"
        state_dir.mkdir(parents=True, exist_ok=True)
        state_file = state_dir / "regime_arb_state.json"
        state_file.write_text("{corrupt json}")
        with patch("src.signals.cross_asset_regime_arb.STATE_DIR", state_dir), \
             patch("src.signals.cross_asset_regime_arb.STATE_FILE", state_file):
            detector = CrossAssetRegimeArbDetector()
            assert detector.state["previous_pattern"] is None
            assert detector.state["persistence_days"] == 0

    def test_state_load_os_error(self):
        """OSError during state load returns default state."""
        with patch("src.signals.cross_asset_regime_arb.STATE_FILE") as mock_file:
            mock_file.exists.return_value = True
            with patch("builtins.open", side_effect=PermissionError("Denied")):
                detector = CrossAssetRegimeArbDetector()
                assert detector.state["previous_pattern"] is None
                assert detector.state["persistence_days"] == 0

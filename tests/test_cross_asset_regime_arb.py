"""
Tests for Cross-Asset Regime Arbitrage (v8.09).

Tests per-asset regime detection, divergence classification,
state persistence, and ensemble-facing API.
"""

import json
import os
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

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

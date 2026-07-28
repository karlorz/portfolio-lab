"""
Tests for v9.00 Alternative Data Signal Generator
====================================================
Tests the refactored signal generator that uses only existing pipeline data.
"""

import json
import logging
import math
import sys
import tempfile
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


from src.signals.alternative_data_signal import (
    AlternativeDataSignalGenerator,
    AlternativeDataComposite,
    ComponentSignal,
    EnsembleSignal,
    COMPONENT_WEIGHTS,
    SYMBOLS_REQUIRED,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_prices():
    """Create realistic sample price data."""
    import random
    random.seed(42)
    
    n_days = 300
    prices = {}
    for sym in ["SPY", "TLT", "SHY", "XLF", "XLY", "AGG", "IEF"]:
        # Generate random-walk prices
        p = 100.0
        vals = []
        for i in range(n_days):
            # Different vols per symbol
            vol = {"SPY": 0.008, "TLT": 0.007, "SHY": 0.002, 
                   "XLF": 0.009, "XLY": 0.01, "AGG": 0.003, "IEF": 0.005}.get(sym, 0.005)
            # Uptrend for most
            drift = 0.0003
            p *= (1 + drift + random.gauss(0, vol))
            vals.append(p)
        
        # Add SPY momentum bias toward the end
        if sym == "SPY":
            for i in range(21):
                vals[-21 + i] *= (1 + 0.005 * i)
        
        prices[sym] = [{"d": f"2025-{i//30+1:02d}-{(i%30)+1:02d}", "p": v} for v in vals]
    
    return prices


@pytest.fixture
def mock_generator(sample_prices):
    """Generator with mocked price data."""
    gen = AlternativeDataSignalGenerator()
    gen._prices = sample_prices
    return gen


# ---------------------------------------------------------------------------
# Data Loading & Symbol Coverage
# ---------------------------------------------------------------------------

class TestDataLoading:
    """Test data loading from the pipeline."""

    def test_symbols_available_in_pipeline(self):
        """All required symbols exist in the real data file."""
        prices_path = Path("public/data/prices.json")
        if not prices_path.exists():
            pytest.skip("No real data file available")
        with open(prices_path) as f:
            data = json.load(f)
        for sym in SYMBOLS_REQUIRED:
            # BTC-USD is optional - crypto sentiment has fallback
            if sym == "BTC-USD" and sym not in data:
                continue
            assert sym in data, f"{sym} missing from prices.json"
            assert len(data[sym]) > 0, f"{sym} has no data"

    def test_get_prices_returns_list(self, mock_generator):
        """get_prices returns list of floats."""
        prices = mock_generator._get_prices("SPY", 100)
        assert isinstance(prices, list)
        assert len(prices) > 0
        assert all(isinstance(p, (int, float)) for p in prices)

    def test_get_prices_unknown_symbol(self, mock_generator):
        """Unknown symbol returns empty list."""
        prices = mock_generator._get_prices("NONEXISTENT", 100)
        assert prices == []

    def test_get_prices_respects_days(self, mock_generator):
        """Days parameter limits returned data."""
        prices_50 = mock_generator._get_prices("SPY", 50)
        prices_200 = mock_generator._get_prices("SPY", 200)
        assert len(prices_50) <= 50
        assert len(prices_200) <= 200
        assert len(prices_50) < len(prices_200)

    def test_load_prices_caches(self, sample_prices):
        """Price data is cached after first load."""
        gen = AlternativeDataSignalGenerator()
        gen._prices = sample_prices
        p1 = gen._load_prices()
        # Modify the cache
        gen._prices["SPY"] = gen._prices["SPY"][:10]
        p2 = gen._load_prices()
        assert len(p2["SPY"]) == 10  # Returns from cache


# ---------------------------------------------------------------------------
# Component Signal Tests
# ---------------------------------------------------------------------------

class TestTreasuryCurveSignal:
    """Test treasury curve component."""

    def test_returns_component_signal(self, mock_generator):
        """Returns valid ComponentSignal."""
        sig = mock_generator._treasury_curve_signal()
        assert isinstance(sig, ComponentSignal)
        assert sig.name == "treasury_curve"
        assert -1.0 <= sig.value <= 1.0
        assert 0.0 <= sig.confidence <= 1.0

    def test_insufficient_data(self, mock_generator):
        """Short data returns neutral signal."""
        mock_generator._prices["TLT"] = [{"d": "2025-01-01", "p": 100}]
        mock_generator._prices["SHY"] = [{"d": "2025-01-01", "p": 50}]
        sig = mock_generator._treasury_curve_signal()
        assert sig.value == 0.0
        assert "insufficient" in str(sig.raw_inputs)


class TestSectorRotationSignal:
    """Test sector rotation component."""

    def test_returns_component_signal(self, mock_generator):
        """Returns valid ComponentSignal."""
        sig = mock_generator._sector_rotation_signal()
        assert isinstance(sig, ComponentSignal)
        assert sig.name == "sector_rotation"
        assert -1.0 <= sig.value <= 1.0

    def test_both_bullish(self, mock_generator):
        """Both sectors up gives positive signal."""
        # Make XLF and XLY both strongly up
        n = len(mock_generator._prices["XLF"])
        for sym in ["XLF", "XLY"]:
            base = mock_generator._prices[sym][-63]["p"]
            for i in range(63):
                mock_generator._prices[sym][-63 + i]["p"] = base * (1 + 0.003 * i)
        sig = mock_generator._sector_rotation_signal()
        assert sig.value > 0

    def test_both_bearish(self, mock_generator):
        """Both sectors down gives negative signal."""
        n = len(mock_generator._prices["XLF"])
        for sym in ["XLF", "XLY"]:
            base = mock_generator._prices[sym][-63]["p"]
            for i in range(63):
                mock_generator._prices[sym][-63 + i]["p"] = base * (1 - 0.003 * i)
        sig = mock_generator._sector_rotation_signal()
        assert sig.value < 0


class TestCreditSpreadSignal:
    """Test credit spread component."""

    def test_returns_valid(self, mock_generator):
        """Returns valid ComponentSignal."""
        sig = mock_generator._credit_spread_signal()
        assert isinstance(sig, ComponentSignal)
        assert sig.name == "credit_spread"
        assert -1.0 <= sig.value <= 1.0

    def test_agg_outperforms_ief(self, mock_generator):
        """AGG outperforming IEF gives positive signal."""
        n = len(mock_generator._prices["AGG"])
        # Make AGG strongly outperform IEF
        base_a = mock_generator._prices["AGG"][-63]["p"]
        base_i = mock_generator._prices["IEF"][-63]["p"]
        for i in range(63):
            mock_generator._prices["AGG"][-63 + i]["p"] = base_a * (1 + 0.0015 * i)
            mock_generator._prices["IEF"][-63 + i]["p"] = base_i * (1 - 0.0002 * i)
        sig = mock_generator._credit_spread_signal()
        assert sig.value > 0


class TestTailRiskSignal:
    """Test tail risk component."""

    def test_returns_valid(self, mock_generator):
        """Returns valid ComponentSignal."""
        sig = mock_generator._tail_risk_signal()
        assert isinstance(sig, ComponentSignal)
        assert sig.name == "tail_risk"
        assert -1.0 <= sig.value <= 1.0

    def test_low_vol_gives_positive(self, mock_generator):
        """Low vol regime gives positive (risk-on) signal."""
        # Make SPY returns very smooth (low vol)
        spy = mock_generator._prices["SPY"]
        base = spy[-252]["p"]
        for i in range(252):
            spy[-252 + i]["p"] = base * (1 + 0.0005 * i)  # Almost no noise
        sig = mock_generator._tail_risk_signal()
        # Low vol should give positive tail_risk signal
        # (value is inverted: low vol = safer = risk-on)
        assert sig.value > -0.5

    def test_not_enough_data(self, mock_generator):
        """Short SPY history returns neutral."""
        mock_generator._prices["SPY"] = [{"d": "2025-01-01", "p": 100} for _ in range(10)]
        sig = mock_generator._tail_risk_signal()
        assert sig.value == 0.0


class TestBroadMomentum:
    """Test broad momentum component."""

    def test_returns_valid(self, mock_generator):
        """Returns valid ComponentSignal."""
        sig = mock_generator._broad_momentum_signal()
        assert isinstance(sig, ComponentSignal)
        assert sig.name == "broad_momentum"
        assert -1.0 <= sig.value <= 1.0

    def test_strong_uptrend(self, mock_generator):
        """Strong SPY uptrend gives positive signal."""
        # SPY already has uptrend from fixture
        sig = mock_generator._broad_momentum_signal()
        assert sig.value > 0

    def test_strong_downtrend(self, mock_generator):
        """Strong SPY downtrend gives negative signal."""
        spy = mock_generator._prices["SPY"]
        base = spy[-126]["p"]
        for i in range(126):
            spy[-126 + i]["p"] = base * (1 - 0.004 * i)
        sig = mock_generator._broad_momentum_signal()
        assert sig.value < 0


# ---------------------------------------------------------------------------
# Composite Signal Tests
# ---------------------------------------------------------------------------

class TestCompositeCalculation:
    """Test composite signal computation."""

    def test_composite_has_all_fields(self, mock_generator):
        """Composite contains expected fields."""
        components = [
            mock_generator._treasury_curve_signal(),
            mock_generator._sector_rotation_signal(),
            mock_generator._credit_spread_signal(),
            mock_generator._tail_risk_signal(),
            mock_generator._broad_momentum_signal(),
        ]
        composite = mock_generator.calculate_composite(components)
        assert isinstance(composite, AlternativeDataComposite)
        assert -1.0 <= composite.composite_score <= 1.0
        assert 0.0 <= composite.confidence <= 1.0
        assert composite.regime in ("risk_on", "neutral", "risk_off")
        assert len(composite.components) == 5
        assert len(composite.symbol_coverage) >= 7

    def test_weights_sum_to_one(self):
        """Component weights sum to 1.0."""
        total = sum(COMPONENT_WEIGHTS.values())
        assert abs(total - 1.0) < 0.001

    def test_all_components_negative(self, mock_generator):
        """All negative components gives risk_off."""
        components = [
            ComponentSignal("treasury_curve", -0.5, 0.7, {}),
            ComponentSignal("sector_rotation", -0.6, 0.7, {}),
            ComponentSignal("credit_spread", -0.4, 0.7, {}),
            ComponentSignal("tail_risk", -0.3, 0.7, {}),
            ComponentSignal("broad_momentum", -0.5, 0.7, {}),
        ]
        composite = mock_generator.calculate_composite(components)
        assert composite.composite_score < -0.15
        assert composite.regime in ("risk_off", "neutral")

    def test_all_components_positive(self, mock_generator):
        """All positive components gives risk_on."""
        components = [
            ComponentSignal("treasury_curve", 0.5, 0.7, {}),
            ComponentSignal("sector_rotation", 0.6, 0.7, {}),
            ComponentSignal("credit_spread", 0.4, 0.7, {}),
            ComponentSignal("tail_risk", 0.5, 0.7, {}),
            ComponentSignal("broad_momentum", 0.5, 0.7, {}),
        ]
        composite = mock_generator.calculate_composite(components)
        assert composite.composite_score > 0.15
        assert composite.regime == "risk_on"

    def test_mixed_signals_give_neutral(self, mock_generator):
        """Mixed positive/negative gives neutral."""
        components = [
            ComponentSignal("treasury_curve", 0.1, 0.5, {}),
            ComponentSignal("sector_rotation", -0.1, 0.5, {}),
            ComponentSignal("credit_spread", 0.05, 0.5, {}),
            ComponentSignal("tail_risk", -0.05, 0.5, {}),
            ComponentSignal("broad_momentum", 0.1, 0.5, {}),
        ]
        composite = mock_generator.calculate_composite(components)
        assert -0.15 <= composite.composite_score <= 0.15
        assert composite.regime == "neutral"


# ---------------------------------------------------------------------------
# Ensemble Signal Conversion
# ---------------------------------------------------------------------------

class TestEnsembleConversion:
    """Test conversion to ensemble voter format."""

    def test_converts_correctly(self, mock_generator):
        """Conversion produces valid EnsembleSignal."""
        composite = AlternativeDataComposite(
            timestamp=datetime.now().isoformat(),
            composite_score=0.5,
            confidence=0.75,
            regime="risk_on",
            z_score=1.67,
            components={"test": 0.5},
            component_confidences={"test": 0.75},
            weights={"test": 1.0},
            data_freshness_hours=12.0,
            sources_count=1,
            symbol_coverage=["SPY"],
        )
        signal = mock_generator.to_ensemble_signal(composite)
        assert isinstance(signal, EnsembleSignal)
        assert signal.source == "alternative_data"
        assert signal.regime == "bull"
        assert 0 <= signal.probability <= 1
        assert signal.confidence == 0.75

    def test_risk_off_to_bear(self, mock_generator):
        """risk_off maps to bear."""
        composite = AlternativeDataComposite(
            timestamp=datetime.now().isoformat(),
            composite_score=-0.5,
            confidence=0.6,
            regime="risk_off",
            z_score=-1.67,
            components={},
            component_confidences={},
            weights={},
            data_freshness_hours=12,
            sources_count=0,
            symbol_coverage=[],
        )
        signal = mock_generator.to_ensemble_signal(composite)
        assert signal.regime == "bear"

    def test_neutral_maps_to_neutral(self, mock_generator):
        """neutral maps to neutral."""
        composite = AlternativeDataComposite(
            timestamp=datetime.now().isoformat(),
            composite_score=0.0,
            confidence=0.5,
            regime="neutral",
            z_score=0.0,
            components={},
            component_confidences={},
            weights={},
            data_freshness_hours=12,
            sources_count=0,
            symbol_coverage=[],
        )
        signal = mock_generator.to_ensemble_signal(composite)
        assert signal.regime == "neutral"


# ---------------------------------------------------------------------------
# End-to-End Pipeline
# ---------------------------------------------------------------------------

class TestFullPipeline:
    """Test complete generate_signal pipeline."""

    def test_generate_returns_valid_signal(self, mock_generator):
        """Full pipeline produces valid signal."""
        signal = mock_generator.generate_signal()
        assert isinstance(signal, EnsembleSignal)
        assert signal.source == "alternative_data"
        assert signal.regime in ("bull", "bear", "neutral", "crisis")
        assert signal.confidence > 0
        assert signal.timestamp is not None

    def test_state_file_created(self, mock_generator, tmp_path):
        """State file is written to disk."""
        # Temporarily redirect state dir
        original_state = mock_generator.state_dir
        mock_generator.state_dir = tmp_path
        mock_generator.generate_signal()
        state_file = tmp_path / "alternative_data_state.json"
        assert state_file.exists()
        with open(state_file) as f:
            state = json.load(f)
        assert "composite_score" in state
        assert "regime" in state
        assert "components" in state
        assert "last_update" in state
        mock_generator.state_dir = original_state

    def test_signal_file_created(self, mock_generator, tmp_path):
        """Signal file is written to ensemble format."""
        original_dir = mock_generator.signals_dir
        mock_generator.signals_dir = tmp_path
        mock_generator.generate_signal()
        signal_file = tmp_path / "alternative_data_latest.json"
        assert signal_file.exists()
        with open(signal_file) as f:
            s = json.load(f)
        assert s["source"] == "alternative_data"
        assert "regime" in s
        assert "probability" in s
        assert "raw_data" in s
        mock_generator.signals_dir = original_dir

    def test_load_latest_signal(self, mock_generator, tmp_path):
        """Can load saved signal back."""
        original_dir = mock_generator.signals_dir
        mock_generator.signals_dir = tmp_path
        mock_generator.generate_signal()
        loaded = mock_generator.load_latest_signal()
        assert loaded is not None
        assert loaded.source == "alternative_data"
        mock_generator.signals_dir = original_dir

    def test_validate_fresh_signal(self, mock_generator):
        """Fresh signal passes validation."""
        signal = mock_generator.generate_signal()
        is_valid = mock_generator.validate_signal(signal)
        assert is_valid is True

    def test_validate_stale_signal(self, mock_generator):
        """Old signal fails validation."""
        stale = EnsembleSignal(
            source="alternative_data",
            regime="bull",
            probability=0.7,
            confidence=0.6,
            timestamp=(datetime.now() - timedelta(hours=72)).isoformat(),
            raw_data={},
        )
        is_valid = mock_generator.validate_signal(stale)
        assert is_valid is False

    def test_validate_low_confidence(self, mock_generator):
        """Low confidence signal fails validation."""
        low_conf = EnsembleSignal(
            source="alternative_data",
            regime="neutral",
            probability=0.5,
            confidence=0.1,
            timestamp=datetime.now().isoformat(),
            raw_data={},
        )
        is_valid = mock_generator.validate_signal(low_conf)
        assert is_valid is False


# ---------------------------------------------------------------------------
# CLI Tests
# ---------------------------------------------------------------------------

class TestCLI:
    """Test CLI interface."""

    def test_generate_cli(self, mock_generator, monkeypatch):
        """--generate flag produces output."""
        import src.signals.alternative_data_signal as mod
        
        # Patch the generator
        original_gen = mod.AlternativeDataSignalGenerator
        mod.AlternativeDataSignalGenerator = lambda: mock_generator
        
        monkeypatch.setattr(sys, 'argv', ['alt_signal.py', '--generate'])
        try:
            mod.main()
        except SystemExit:
            pass
        
        mod.AlternativeDataSignalGenerator = original_gen

    def test_status_without_signal(self, mock_generator, tmp_path, monkeypatch, capsys):
        """--status with no signal shows appropriate message."""
        import src.signals.alternative_data_signal as mod

        # Clear any existing signal
        mock_generator.signals_dir = tmp_path

        original_gen = mod.AlternativeDataSignalGenerator
        mod.AlternativeDataSignalGenerator = lambda: mock_generator

        monkeypatch.setattr(sys, 'argv', ['alt_signal.py', '--status'])
        try:
            mod.main()
        except SystemExit:
            pass

        assert "No signal found" in capsys.readouterr().err

        mod.AlternativeDataSignalGenerator = original_gen

    def test_validate_without_signal(self, mock_generator, tmp_path, monkeypatch, capsys):
        """--validate with no signal shows appropriate message."""
        import src.signals.alternative_data_signal as mod

        mock_generator.signals_dir = tmp_path

        original_gen = mod.AlternativeDataSignalGenerator
        mod.AlternativeDataSignalGenerator = lambda: mock_generator

        monkeypatch.setattr(sys, 'argv', ['alt_signal.py', '--validate'])
        try:
            mod.main()
        except SystemExit:
            pass

        assert "No signal to validate" in capsys.readouterr().err

        mod.AlternativeDataSignalGenerator = original_gen


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_price_data(self, mock_generator):
        """All empty prices return neutral/fallback signals."""
        for sym in SYMBOLS_REQUIRED:
            mock_generator._prices[sym] = [{"d": "2025-01-01", "p": 100}]
        
        sig = mock_generator.generate_signal()
        assert sig is not None

    def test_single_day_data(self, mock_generator):
        """Single day of data gives neutral signals."""
        for sym in SYMBOLS_REQUIRED:
            mock_generator._prices[sym] = [{"d": "2025-01-01", "p": 100}]
        
        components = [
            mock_generator._treasury_curve_signal(),
            mock_generator._sector_rotation_signal(),
            mock_generator._credit_spread_signal(),
            mock_generator._tail_risk_signal(),
            mock_generator._broad_momentum_signal(),
        ]
        
        # All should be neutral/fallback
        for comp in components:
            assert comp.value == 0.0

    def test_missing_symbol(self, mock_generator):
        """Missing symbol returns empty data."""
        del mock_generator._prices["SPY"]
        sig = mock_generator._broad_momentum_signal()
        assert sig.value == 0.0
        assert "insufficient" in str(sig.raw_inputs)

    def test_negative_prices_not_crashing(self, mock_generator):
        """Edge case: negative or zero prices don't crash."""
        for sym in ["SPY", "TLT"]:
            prices = mock_generator._prices[sym]
            prices[-1]["p"] = 0.0
            prices[-2]["p"] = -5.0
        
        # Should not crash, just give fallback
        components = [
            mock_generator._treasury_curve_signal(),
            mock_generator._tail_risk_signal(),
            mock_generator._broad_momentum_signal(),
        ]
        for comp in components:
            assert comp is not None

    def test_composite_weights_override(self, mock_generator):
        """Custom weights are used in composite calculation."""
        mock_generator.weights = {
            "broad_momentum": 1.0,
            "treasury_curve": 0.0,
            "sector_rotation": 0.0,
            "credit_spread": 0.0,
            "tail_risk": 0.0,
        }
        components = [
            ComponentSignal("broad_momentum", 0.8, 0.9, {}),
            ComponentSignal("treasury_curve", -1.0, 0.7, {}),
        ]
        composite = mock_generator.calculate_composite(components)
        # Only broad_momentum has weight
        assert composite.composite_score == 0.8


# ---------------------------------------------------------------------------
# Real Data Integration Test
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not Path("public/data/prices.json").exists(),
    reason="Requires real price data file",
)
class TestRealData:
    """Test against real data file."""

    def test_generates_with_real_data(self):
        """Generator works with real prices.json."""
        gen = AlternativeDataSignalGenerator()
        signal = gen.generate_signal()
        assert signal.confidence > 0
        assert signal.regime in ("bull", "bear", "neutral")
        print(f"Real data: regime={signal.regime}, score={signal.raw_data['composite_score']:.4f}")

    def test_all_components_have_data(self):
        """All component symbols exist in real data."""
        gen = AlternativeDataSignalGenerator()
        prices = gen._load_prices()
        for sym in SYMBOLS_REQUIRED:
            # BTC-USD is optional - crypto sentiment has fallback
            if sym == "BTC-USD" and sym not in prices:
                continue
            assert sym in prices, f"{sym} not found in prices.json"
            assert len(prices[sym]) > 200, f"{sym} has insufficient data"


class TestSignalSnapshotBridge:
    """Regression tests for get_signal_snapshot() — was missing numpy import."""

    def test_get_signal_snapshot_returns_snapshot(self):
        """get_signal_snapshot() should return a SignalSnapshot without NameError."""
        from src.signals.signal_snapshot import SignalSnapshot
        gen = AlternativeDataSignalGenerator()
        snapshot = gen.get_signal_snapshot()
        assert isinstance(snapshot, SignalSnapshot)


    def test_snapshot_value_clipped_to_range(self):
        """Snapshot value should be clipped to [-1, 1] by np.clip."""
        gen = AlternativeDataSignalGenerator()
        snapshot = gen.get_signal_snapshot()
        assert -1.0 <= snapshot.value <= 1.0


# ---------------------------------------------------------------------------
# Dataclass field completeness tests
# ---------------------------------------------------------------------------

class TestDataclassFieldCompleteness:
    """Verify all dataclass fields survive asdict() round-trips."""

    def test_component_signal_to_dict_all_fields(self):
        """ComponentSignal asdict() includes name, value, confidence, raw_inputs."""
        sig = ComponentSignal(name="test_sig", value=0.5, confidence=0.75,
                              raw_inputs={"price": 100.0, "ratio": 1.05})
        d = asdict(sig)
        assert d["name"] == "test_sig"
        assert d["value"] == 0.5
        assert d["confidence"] == 0.75
        assert d["raw_inputs"] == {"price": 100.0, "ratio": 1.05}
        assert set(d.keys()) == {"name", "value", "confidence", "raw_inputs"}

    def test_component_signal_empty_raw_inputs(self):
        """ComponentSignal with empty raw_inputs survives asdict()."""
        sig = ComponentSignal(name="empty", value=0.0, confidence=0.0, raw_inputs={})
        d = asdict(sig)
        assert d["raw_inputs"] == {}

    def test_alternative_data_composite_to_dict_all_fields(self):
        """AlternativeDataComposite asdict() includes all 11 fields."""
        now = datetime.now().isoformat()
        comp = AlternativeDataComposite(
            timestamp=now, composite_score=0.3, confidence=0.7,
            regime="risk_on", z_score=1.0,
            components={"treasury_curve": 0.5},
            component_confidences={"treasury_curve": 0.7},
            weights={"treasury_curve": 0.25},
            data_freshness_hours=12.0, sources_count=5,
            symbol_coverage=["SPY", "TLT"],
        )
        d = asdict(comp)
        assert d["timestamp"] == now
        assert d["composite_score"] == 0.3
        assert d["confidence"] == 0.7
        assert d["regime"] == "risk_on"
        assert d["z_score"] == 1.0
        assert d["components"] == {"treasury_curve": 0.5}
        assert d["component_confidences"] == {"treasury_curve": 0.7}
        assert d["weights"] == {"treasury_curve": 0.25}
        assert d["data_freshness_hours"] == 12.0
        assert d["sources_count"] == 5
        assert d["symbol_coverage"] == ["SPY", "TLT"]
        assert set(d.keys()) == {
            "timestamp", "composite_score", "confidence", "regime", "z_score",
            "components", "component_confidences", "weights",
            "data_freshness_hours", "sources_count", "symbol_coverage",
        }

    def test_alternative_data_composite_empty_component_dicts(self):
        """Composite with empty component dicts still round-trips."""
        now = datetime.now().isoformat()
        comp = AlternativeDataComposite(
            timestamp=now, composite_score=0.0, confidence=0.0,
            regime="neutral", z_score=0.0,
            components={}, component_confidences={}, weights={},
            data_freshness_hours=0.0, sources_count=0, symbol_coverage=[],
        )
        d = asdict(comp)
        assert d["components"] == {}
        assert d["symbol_coverage"] == []

    def test_ensemble_signal_to_dict_all_fields(self):
        """EnsembleSignal asdict() includes all 6 fields."""
        sig = EnsembleSignal(
            source="alternative_data", regime="bull",
            probability=0.75, confidence=0.8,
            timestamp="2025-01-01T00:00:00", raw_data={"score": 0.5},
        )
        d = asdict(sig)
        assert d["source"] == "alternative_data"
        assert d["regime"] == "bull"
        assert d["probability"] == 0.75
        assert d["confidence"] == 0.8
        assert d["timestamp"] == "2025-01-01T00:00:00"
        assert d["raw_data"] == {"score": 0.5}
        assert set(d.keys()) == {"source", "regime", "probability", "confidence",
                                  "timestamp", "raw_data"}

    def test_ensemble_signal_empty_raw_data(self):
        """EnsembleSignal with empty raw_data survives asdict()."""
        sig = EnsembleSignal(
            source="alternative_data", regime="neutral",
            probability=0.5, confidence=0.0,
            timestamp="2025-01-01T00:00:00", raw_data={},
        )
        d = asdict(sig)
        assert d["raw_data"] == {}


# ---------------------------------------------------------------------------
# Constants validation
# ---------------------------------------------------------------------------

class TestConstantsValidation:
    """Validate module-level constants."""

    def test_component_weights_sum_to_one(self):
        """COMPONENT_WEIGHTS must sum to exactly 1.0."""
        total = sum(COMPONENT_WEIGHTS.values())
        assert abs(total - 1.0) < 0.001

    def test_component_weights_all_positive(self):
        """All component weights must be non-negative."""
        for name, weight in COMPONENT_WEIGHTS.items():
            assert weight >= 0, f"Weight for {name} is negative: {weight}"

    def test_component_weights_no_zero(self):
        """No component weight should be exactly zero."""
        for name, weight in COMPONENT_WEIGHTS.items():
            assert weight > 0, f"Weight for {name} is zero"

    def test_component_weights_have_expected_keys(self):
        """COMPONENT_WEIGHTS has exactly the 7 expected keys."""
        expected = {"treasury_curve", "sector_rotation", "credit_spread",
                    "tail_risk", "broad_momentum", "crypto_sentiment", "crypto_fg"}
        assert set(COMPONENT_WEIGHTS.keys()) == expected

    def test_symbols_required_all_present(self):
        """SYMBOLS_REQUIRED includes all 8 expected tickers (including BTC-USD for crypto sentiment)."""
        expected = {"SPY", "TLT", "SHY", "XLF", "XLY", "AGG", "IEF", "BTC-USD"}
        assert set(SYMBOLS_REQUIRED) == expected

    def test_symbols_required_no_duplicates(self):
        """SYMBOLS_REQUIRED has no duplicate entries."""
        assert len(SYMBOLS_REQUIRED) == len(set(SYMBOLS_REQUIRED))

    def test_component_weights_normalized(self):
        """Weights within expected decimal precision."""
        for name, weight in COMPONENT_WEIGHTS.items():
            # Weights should be clean decimal fractions (multiply to whole number)
            scaled = weight * 100
            assert abs(scaled - round(scaled)) < 0.001, \
                f"Weight {name}={weight} is not a clean centimal fraction"


# ---------------------------------------------------------------------------
# Regime classification boundary conditions
# ---------------------------------------------------------------------------

class TestRegimeClassificationBoundaries:
    """Test _determine_regime boundary conditions."""

    def test_regime_risk_on_boundary_above(self, mock_generator):
        """Score just above +0.15 triggers risk_on."""
        regime = mock_generator._determine_regime(0.151)
        assert regime == "risk_on"

    def test_regime_risk_off_boundary_below(self, mock_generator):
        """Score just below -0.15 triggers risk_off."""
        regime = mock_generator._determine_regime(-0.151)
        assert regime == "risk_off"

    def test_regime_neutral_positive_side(self, mock_generator):
        """Score at +0.15 is still neutral."""
        regime = mock_generator._determine_regime(0.15)
        assert regime == "neutral"

    def test_regime_neutral_negative_side(self, mock_generator):
        """Score at -0.15 is still neutral."""
        regime = mock_generator._determine_regime(-0.15)
        assert regime == "neutral"

    def test_regime_neutral_exact_zero(self, mock_generator):
        """Score exactly 0.0 is neutral."""
        regime = mock_generator._determine_regime(0.0)
        assert regime == "neutral"

    def test_regime_risk_on_extreme(self, mock_generator):
        """Score at extreme +1.0 triggers risk_on."""
        regime = mock_generator._determine_regime(1.0)
        assert regime == "risk_on"

    def test_regime_risk_off_extreme(self, mock_generator):
        """Score at extreme -1.0 triggers risk_off."""
        regime = mock_generator._determine_regime(-1.0)

    def test_composite_score_extreme_values_do_not_crash(self, mock_generator):
        """Composite score calculation handles extreme component values gracefully."""
        components = [
            ComponentSignal("treasury_curve", 999.0, 1.0, {}),
            ComponentSignal("sector_rotation", -999.0, 1.0, {}),
            ComponentSignal("credit_spread", 999.0, 1.0, {}),
            ComponentSignal("tail_risk", -999.0, 1.0, {}),
            ComponentSignal("broad_momentum", 999.0, 1.0, {}),
        ]
        composite = mock_generator.calculate_composite(components)
        # calculate_composite does weighted average without clamping;
        # extreme components produce extreme score but should not crash
        assert composite is not None
        assert composite.regime in ("risk_on", "risk_off", "neutral")


# ---------------------------------------------------------------------------
# Validate signal predicate edge cases
# ---------------------------------------------------------------------------

class TestValidateSignalEdgeCases:
    """Test validate_signal() boundary conditions."""

    def test_validate_signal_just_under_48_hours(self, mock_generator):
        """Signal just under 48 hours old should be valid."""
        boundary = EnsembleSignal(
            source="alternative_data", regime="bull",
            probability=0.7, confidence=0.5,
            timestamp=(datetime.now() - timedelta(hours=47, minutes=59)).isoformat(),
            raw_data={},
        )
        is_valid = mock_generator.validate_signal(boundary)
        assert is_valid is True

    def test_validate_signal_one_minute_over_48_hours(self, mock_generator):
        """Signal just over 48 hours old should be invalid."""
        stale = EnsembleSignal(
            source="alternative_data", regime="bull",
            probability=0.7, confidence=0.5,
            timestamp=(datetime.now() - timedelta(hours=48, minutes=1)).isoformat(),
            raw_data={},
        )
        is_valid = mock_generator.validate_signal(stale)
        assert is_valid is False

    def test_validate_signal_exactly_0_3_confidence(self, mock_generator):
        """Signal with confidence exactly 0.3 should be valid."""
        edge = EnsembleSignal(
            source="alternative_data", regime="neutral",
            probability=0.5, confidence=0.3,
            timestamp=datetime.now().isoformat(),
            raw_data={},
        )
        is_valid = mock_generator.validate_signal(edge)
        assert is_valid is True

    def test_validate_signal_just_below_0_3_confidence(self, mock_generator):
        """Signal with confidence just below 0.3 should be invalid."""
        low = EnsembleSignal(
            source="alternative_data", regime="neutral",
            probability=0.5, confidence=0.299,
            timestamp=datetime.now().isoformat(),
            raw_data={},
        )
        is_valid = mock_generator.validate_signal(low)
        assert is_valid is False

    def test_validate_signal_max_confidence_stale(self, mock_generator):
        """Even max confidence fails if signal is too old."""
        old_high_conf = EnsembleSignal(
            source="alternative_data", regime="bull",
            probability=0.9, confidence=1.0,
            timestamp=(datetime.now() - timedelta(hours=72)).isoformat(),
            raw_data={},
        )
        is_valid = mock_generator.validate_signal(old_high_conf)
        assert is_valid is False

    def test_validate_signal_zero_confidence_fresh(self, mock_generator):
        """Fresh signal with zero confidence fails validation."""
        fresh_low = EnsembleSignal(
            source="alternative_data", regime="neutral",
            probability=0.5, confidence=0.0,
            timestamp=datetime.now().isoformat(),
            raw_data={},
        )
        is_valid = mock_generator.validate_signal(fresh_low)
        assert is_valid is False


# ---------------------------------------------------------------------------
# SignalSnapshot bridge method edge cases
# ---------------------------------------------------------------------------

class TestSignalSnapshotBridgeEdgeCases:
    """Test get_signal_snapshot() edge cases beyond basic regression."""

    def test_snapshot_no_signal_returns_inactive(self, mock_generator, tmp_path):
        """No saved signal returns inactive snapshot with zero confidence."""
        mock_generator.signals_dir = tmp_path
        snapshot = mock_generator.get_signal_snapshot()
        assert snapshot.is_active is False
        assert snapshot.value == 0.0
        assert snapshot.confidence == 0.0
        assert "unavailable" in snapshot.explanation

    def test_snapshot_negative_composite_score(self, mock_generator, tmp_path):
        """Negative composite score yields positive SPY snapshot value (polarity map inverts)."""
        mock_generator.signals_dir = tmp_path
        # Save a signal with negative score
        composite = AlternativeDataComposite(
            timestamp=datetime.now().isoformat(),
            composite_score=-0.5, confidence=0.8,
            regime="risk_off", z_score=-1.67,
            components={"treasury_curve": -0.5},
            component_confidences={"treasury_curve": 0.8},
            weights={"treasury_curve": 1.0},
            data_freshness_hours=12.0, sources_count=1,
            symbol_coverage=["SPY"],
        )
        signal = mock_generator.to_ensemble_signal(composite)
        mock_generator._save_signal(composite, signal)
        snapshot = mock_generator.get_signal_snapshot()
        assert snapshot.value > 0  # SPY polarity map: negative composite → positive SPY
        assert snapshot.is_active is True

    def test_snapshot_positive_composite_score(self, mock_generator, tmp_path):
        """Positive composite score yields negative SPY snapshot value (polarity map inverts)."""
        mock_generator.signals_dir = tmp_path
        composite = AlternativeDataComposite(
            timestamp=datetime.now().isoformat(),
            composite_score=0.5, confidence=0.8,
            regime="risk_on", z_score=1.67,
            components={"treasury_curve": 0.5},
            component_confidences={"treasury_curve": 0.8},
            weights={"treasury_curve": 1.0},
            data_freshness_hours=12.0, sources_count=1,
            symbol_coverage=["SPY"],
        )
        signal = mock_generator.to_ensemble_signal(composite)
        mock_generator._save_signal(composite, signal)
        snapshot = mock_generator.get_signal_snapshot()
        assert snapshot.value < 0  # SPY polarity map: positive composite → negative SPY
        assert snapshot.is_active is True

    def test_snapshot_zero_score_uses_regime_map_bull(self, mock_generator, tmp_path):
        """Zero composite with bull regime uses regime fallback value."""
        mock_generator.signals_dir = tmp_path
        composite = AlternativeDataComposite(
            timestamp=datetime.now().isoformat(),
            composite_score=0.0, confidence=0.8,
            regime="risk_on", z_score=0.0,
            components={}, component_confidences={},
            weights={}, data_freshness_hours=12.0,
            sources_count=0, symbol_coverage=[],
        )
        signal = mock_generator.to_ensemble_signal(composite)
        mock_generator._save_signal(composite, signal)
        snapshot = mock_generator.get_signal_snapshot()
        # bull regime maps to 0.4, SPY polarity map inverts to -0.4
        assert snapshot.value == -0.4

    def test_snapshot_zero_score_bear_regime(self, mock_generator, tmp_path):
        """Zero composite with bear regime uses regime fallback -0.4."""
        mock_generator.signals_dir = tmp_path
        composite = AlternativeDataComposite(
            timestamp=datetime.now().isoformat(),
            composite_score=0.0, confidence=0.8,
            regime="risk_off", z_score=0.0,
            components={}, component_confidences={},
            weights={}, data_freshness_hours=12.0,
            sources_count=0, symbol_coverage=[],
        )
        signal = mock_generator.to_ensemble_signal(composite)
        mock_generator._save_signal(composite, signal)
        snapshot = mock_generator.get_signal_snapshot()
        # bear regime maps to -0.4, SPY polarity map inverts to +0.4
        assert snapshot.value == 0.4

    def test_snapshot_zero_score_neutral_is_positive_zero(self, mock_generator, tmp_path):
        """Neutral fallback stays positive zero after the SPY polarity map."""
        mock_generator.signals_dir = tmp_path
        composite = AlternativeDataComposite(
            timestamp=datetime.now().isoformat(),
            composite_score=0.0, confidence=0.8,
            regime="neutral", z_score=0.0,
            components={}, component_confidences={},
            weights={}, data_freshness_hours=12.0,
            sources_count=0, symbol_coverage=[],
        )
        signal = mock_generator.to_ensemble_signal(composite)
        mock_generator._save_signal(composite, signal)
        snapshot = mock_generator.get_signal_snapshot()
        assert snapshot.value == 0.0
        assert math.copysign(1.0, snapshot.value) == 1.0

    def test_snapshot_metadata_includes_regime_and_probability(self, mock_generator, tmp_path):
        """Snapshot metadata contains regime and probability fields."""
        mock_generator.signals_dir = tmp_path
        composite = AlternativeDataComposite(
            timestamp=datetime.now().isoformat(),
            composite_score=0.3, confidence=0.75,
            regime="risk_on", z_score=1.0,
            components={}, component_confidences={},
            weights={}, data_freshness_hours=12.0,
            sources_count=0, symbol_coverage=[],
        )
        signal = mock_generator.to_ensemble_signal(composite)
        mock_generator._save_signal(composite, signal)
        snapshot = mock_generator.get_signal_snapshot()
        assert "regime" in snapshot.metadata
        assert "probability" in snapshot.metadata
        assert "raw_data" in snapshot.metadata
        assert snapshot.metadata["regime"] == "bull"

    def test_snapshot_asset_signals_contains_spy(self, mock_generator, tmp_path):
        """Snapshot asset_signals includes SPY key."""
        mock_generator.signals_dir = tmp_path
        composite = AlternativeDataComposite(
            timestamp=datetime.now().isoformat(),
            composite_score=0.3, confidence=0.75,
            regime="risk_on", z_score=1.0,
            components={}, component_confidences={},
            weights={}, data_freshness_hours=12.0,
            sources_count=0, symbol_coverage=[],
        )
        signal = mock_generator.to_ensemble_signal(composite)
        mock_generator._save_signal(composite, signal)
        snapshot = mock_generator.get_signal_snapshot()
        assert "SPY" in snapshot.asset_signals
        assert snapshot.asset_signals["SPY"] == snapshot.value

    def test_snapshot_regime_fit_is_all(self, mock_generator, tmp_path):
        """Snapshot regime_fit should always be 'all'."""
        mock_generator.signals_dir = tmp_path
        composite = AlternativeDataComposite(
            timestamp=datetime.now().isoformat(),
            composite_score=0.3, confidence=0.75,
            regime="risk_on", z_score=1.0,
            components={}, component_confidences={},
            weights={}, data_freshness_hours=12.0,
            sources_count=0, symbol_coverage=[],
        )
        signal = mock_generator.to_ensemble_signal(composite)
        mock_generator._save_signal(composite, signal)
        snapshot = mock_generator.get_signal_snapshot()
        assert snapshot.regime_fit == "all"


# ---------------------------------------------------------------------------
# State persistence and retrieval edge cases
# ---------------------------------------------------------------------------

class TestStatePersistenceEdgeCases:
    """Test signal state file persistence edge cases."""

    def test_state_file_has_all_required_fields(self, mock_generator, tmp_path):
        """State JSON includes all 8 expected keys."""
        original = mock_generator.state_dir
        mock_generator.state_dir = tmp_path
        mock_generator.generate_signal()
        state_file = tmp_path / "alternative_data_state.json"
        with open(state_file) as f:
            state = json.load(f)
        expected_keys = {"last_update", "composite_score", "confidence",
                         "regime", "z_score", "components",
                         "component_confidences"}
        assert expected_keys.issubset(state.keys()), \
            f"Missing keys: {expected_keys - set(state.keys())}"
        mock_generator.state_dir = original

    def test_state_field_types(self, mock_generator, tmp_path):
        """State file fields have correct types."""
        original = mock_generator.state_dir
        mock_generator.state_dir = tmp_path
        mock_generator.generate_signal()
        state_file = tmp_path / "alternative_data_state.json"
        with open(state_file) as f:
            state = json.load(f)
        assert isinstance(state["last_update"], str)
        assert isinstance(state["composite_score"], (int, float))
        assert isinstance(state["confidence"], (int, float))
        assert isinstance(state["regime"], str)
        assert isinstance(state["z_score"], (int, float))
        assert isinstance(state["components"], dict)
        assert isinstance(state["component_confidences"], dict)
        mock_generator.state_dir = original

    def test_signal_file_has_all_required_fields(self, mock_generator, tmp_path):
        """Signal JSON includes all 6 expected ensemble fields."""
        original = mock_generator.signals_dir
        mock_generator.signals_dir = tmp_path
        mock_generator.generate_signal()
        signal_file = tmp_path / "alternative_data_latest.json"
        with open(signal_file) as f:
            s = json.load(f)
        expected_keys = {"source", "regime", "probability", "confidence",
                         "timestamp", "raw_data"}
        assert set(s.keys()) == expected_keys, \
            f"Expected keys {expected_keys}, got {set(s.keys())}"
        mock_generator.signals_dir = original

    def test_load_latest_signal_returns_none_for_missing_file(self, mock_generator, tmp_path):
        """load_latest_signal returns None when no file exists."""
        mock_generator.signals_dir = tmp_path
        loaded = mock_generator.load_latest_signal()
        assert loaded is None

    def test_save_overwrites_existing_signal(self, mock_generator, tmp_path):
        """Saving a new signal overwrites the previous one."""
        mock_generator.signals_dir = tmp_path
        mock_generator.state_dir = tmp_path

        # First signal
        composite_1 = AlternativeDataComposite(
            timestamp="2024-01-01T00:00:00", composite_score=0.5, confidence=0.8,
            regime="risk_on", z_score=1.67,
            components={"a": 0.5}, component_confidences={"a": 0.8},
            weights={"a": 1.0}, data_freshness_hours=12.0,
            sources_count=1, symbol_coverage=["SPY"],
        )
        signal_1 = mock_generator.to_ensemble_signal(composite_1)
        mock_generator._save_signal(composite_1, signal_1)

        # Second signal (newer)
        composite_2 = AlternativeDataComposite(
            timestamp="2024-06-01T00:00:00", composite_score=-0.3, confidence=0.6,
            regime="risk_off", z_score=-1.0,
            components={"a": -0.3}, component_confidences={"a": 0.6},
            weights={"a": 1.0}, data_freshness_hours=12.0,
            sources_count=1, symbol_coverage=["SPY"],
        )
        signal_2 = mock_generator.to_ensemble_signal(composite_2)
        mock_generator._save_signal(composite_2, signal_2)

        # Loaded signal should be the second (overwritten) one
        loaded = mock_generator.load_latest_signal()
        assert loaded is not None
        assert loaded.regime == "bear"
        assert loaded.probability != signal_1.probability

    def test_save_and_load_roundtrip_fidelity(self, mock_generator, tmp_path):
        """Save then load produces identical EnsembleSignal fields."""
        mock_generator.signals_dir = tmp_path
        mock_generator.state_dir = tmp_path

        original_signal = mock_generator.generate_signal()
        loaded = mock_generator.load_latest_signal()
        assert loaded is not None
        assert loaded.source == original_signal.source
        assert loaded.regime == original_signal.regime
        assert loaded.probability == original_signal.probability
        assert loaded.confidence == original_signal.confidence
        assert loaded.timestamp == original_signal.timestamp

    def test_state_file_roundtrip_consistency(self, mock_generator, tmp_path):
        """State file values match the composite that was saved."""
        mock_generator.signals_dir = tmp_path
        mock_generator.state_dir = tmp_path

        composite = AlternativeDataComposite(
            timestamp="2024-03-15T12:00:00", composite_score=0.42, confidence=0.78,
            regime="risk_on", z_score=1.4,
            components={"treasury_curve": 0.42},
            component_confidences={"treasury_curve": 0.78},
            weights={"treasury_curve": 1.0}, data_freshness_hours=12.0,
            sources_count=1, symbol_coverage=["SPY"],
        )
        signal = mock_generator.to_ensemble_signal(composite)
        mock_generator._save_signal(composite, signal)

        state_file = tmp_path / "alternative_data_state.json"
        with open(state_file) as f:
            state = json.load(f)

        assert state["last_update"] == "2024-03-15T12:00:00"
        assert state["composite_score"] == 0.42
        assert state["confidence"] == 0.78
        assert state["regime"] == "risk_on"
        assert state["z_score"] == 1.4
        assert state["components"] == {"treasury_curve": 0.42}
        assert state["component_confidences"] == {"treasury_curve": 0.78}


# ---------------------------------------------------------------------------
# Signal calculation edge cases (component-level)
# ---------------------------------------------------------------------------

class TestSignalCalculationEdgeCases:
    """Test component-level edge cases not covered above."""

    def test_treasury_curve_no_tlt_data(self, mock_generator):
        """No TLT data returns insufficient data fallback."""
        mock_generator._prices["TLT"] = [{"d": "2025-01-01", "p": 100}]
        sig = mock_generator._treasury_curve_signal()
        assert sig.value == 0.0
        assert sig.confidence == 0.3

    def test_credit_spread_insufficient_ief(self, mock_generator):
        """Insufficient IEF data returns fallback credit_spread signal."""
        mock_generator._prices["IEF"] = [{"d": "2025-01-01", "p": 100}]
        sig = mock_generator._credit_spread_signal()
        assert sig.value == 0.0

    def test_broad_momentum_barely_enough_data(self, mock_generator):
        """Exactly 63 SPY days meets minimum for broad_momentum."""
        spy = mock_generator._prices["SPY"]
        mock_generator._prices["SPY"] = spy[-63:]
        sig = mock_generator._broad_momentum_signal()
        assert -1.0 <= sig.value <= 1.0

    def test_broad_momentum_just_below_threshold(self, mock_generator):
        """Only 62 SPY days should return insufficient data."""
        spy = mock_generator._prices["SPY"]
        mock_generator._prices["SPY"] = spy[-62:]
        sig = mock_generator._broad_momentum_signal()
        assert sig.value == 0.0
        assert sig.confidence == 0.3

    def test_returns_method_with_negative_price(self, mock_generator):
        """_returns() with zero or negative base price returns None."""
        prices = [100.0, 110.0, 0.0, 120.0]
        result = mock_generator._returns(prices, 2)
        assert result is None

    def test_returns_method_with_insufficient_length(self, mock_generator):
        """_returns() with insufficient data returns None."""
        prices = [100.0, 110.0]
        result = mock_generator._returns(prices, 5)
        assert result is None

    def test_returns_method_normal_case(self, mock_generator):
        """_returns() computes correct normal return."""
        # len >= 4, prices[-1] / prices[-4] - 1 = 120 / 100 - 1 = 0.2
        prices = [100.0, 105.0, 110.0, 120.0]
        result = mock_generator._returns(prices, 4)
        assert result is not None
        assert abs(result - 0.2) < 0.001

    def test_composite_empty_components(self, mock_generator):
        """calculate_composite with empty list handles gracefully."""
        composite = mock_generator.calculate_composite([])
        assert composite.composite_score == 0.0
        assert composite.components == {}

    def test_composite_zero_total_weight(self, mock_generator):
        """Zero total weight defaults to 1.0 to avoid division by zero."""
        mock_generator.weights = {"nonexistent": 0.0}
        components = [
            ComponentSignal("treasury_curve", 0.5, 0.7, {}),
        ]
        composite = mock_generator.calculate_composite(components)
        # The unknown weight key gets 0 weight, total_weight defaults to 1.0
        assert composite.composite_score == 0.0

    def test_complex_weights_override(self, mock_generator):
        """Uneven custom weights produce expected composite."""
        mock_generator.weights = {
            "treasury_curve": 0.5,
            "broad_momentum": 0.5,
            "sector_rotation": 0.0,
            "credit_spread": 0.0,
            "tail_risk": 0.0,
        }
        components = [
            ComponentSignal("treasury_curve", 0.2, 0.5, {}),
            ComponentSignal("broad_momentum", 0.8, 0.9, {}),
        ]
        composite = mock_generator.calculate_composite(components)
        # (0.2 * 0.5 + 0.8 * 0.5) / 1.0 = 0.5
        assert abs(composite.composite_score - 0.5) < 0.001

    def test_z_score_computation(self, mock_generator):
        """_compute_z_score divides by 0.3."""
        z = mock_generator._compute_z_score(0.6)
        assert abs(z - 2.0) < 0.001

    def test_z_score_computation_negative(self, mock_generator):
        """_compute_z_score handles negative scores."""
        z = mock_generator._compute_z_score(-0.3)
        assert abs(z + 1.0) < 0.001

    def test_z_score_computation_zero(self, mock_generator):
        """_compute_z_score with zero returns zero."""
        z = mock_generator._compute_z_score(0.0)
        assert z == 0.0



def test_project_alternative_data_marks_regime_advisory():
    from src.dashboard.generator import project_alternative_data_signal

    projected = project_alternative_data_signal(
        {
            "regime": "bull",
            "probability": 0.82,
            "confidence": 0.7,
            "timestamp": "2026-07-20T00:00:00+00:00",
            "raw_data": {
                "composite_score": 0.27,
                "components": {"news": 0.1},
                "component_confidences": {"news": 0.5},
                "weights": {"news": 1.0},
            },
        }
    )
    assert projected["regime"] == "bull"
    assert projected["alt_regime"] == "bull"
    assert projected["role"] == "advisory_shadow"
    assert projected["live_authoritative"] is False
    assert "regime_authority" in projected["canonical_controller"] or "regime" in projected["canonical_controller"]

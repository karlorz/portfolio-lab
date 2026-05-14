"""Tests for FX Currency Carry data fetcher and signal generator."""

import json
import sqlite3
import sys
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np

from src.data.fx_fetcher import (
    CACHE_DB,
    FXMetrics,
    calculate_usd_strength_score,
    classify_carry_regime,
    classify_momentum_direction,
    classify_volatility_regime,
    init_cache,
    save_metrics,
    load_latest_metrics,
)
from src.signals.fx_carry_signal import (
    FXCarrySignal,
    FXSignalType,
    generate_signal,
    get_allocation_impact,
    get_ensemble_input,
    save_signal,
    load_latest_signal,
    USD_BULL_THRESHOLD,
    USD_BEAR_THRESHOLD,
    MAX_ALLOCATION_SHIFT,
)


class TestFXMetrics(unittest.TestCase):
    """Test FXMetrics dataclass."""
    
    def test_dataclass_creation(self):
        """Test FXMetrics can be created with all fields."""
        metrics = FXMetrics(
            timestamp="2026-05-14T16:00:00",
            uup_price=28.5,
            udn_price=21.3,
            uup_return_30d=2.5,
            udn_return_30d=-1.2,
            usd_strength_score=0.46,
            carry_regime="positive",
            momentum_direction="bullish",
            volatility_regime="low",
        )
        self.assertEqual(metrics.uup_price, 28.5)
        self.assertEqual(metrics.carry_regime, "positive")
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        metrics = FXMetrics(
            timestamp="2026-05-14T16:00:00",
            uup_price=28.5,
            udn_price=21.3,
            uup_return_30d=2.5,
            udn_return_30d=-1.2,
            usd_strength_score=0.46,
            carry_regime="positive",
            momentum_direction="bullish",
            volatility_regime="low",
        )
        d = metrics.to_dict()
        self.assertEqual(d["uup_price"], 28.5)
        self.assertEqual(d["carry_regime"], "positive")
    
    def test_to_json(self):
        """Test conversion to JSON."""
        metrics = FXMetrics(
            timestamp="2026-05-14T16:00:00",
            uup_price=28.5,
            udn_price=21.3,
            uup_return_30d=2.5,
            udn_return_30d=-1.2,
            usd_strength_score=0.46,
            carry_regime="positive",
            momentum_direction="bullish",
            volatility_regime="low",
        )
        json_str = metrics.to_json()
        self.assertIn("uup_price", json_str)
        self.assertIn("28.5", json_str)


class TestFXCalculations(unittest.TestCase):
    """Test FX calculation functions."""
    
    def test_calculate_usd_strength_score(self):
        """Test USD strength score calculation."""
        # UUP up, UDN down = strong USD
        score = calculate_usd_strength_score(3.0, -2.0)
        self.assertGreater(score, 0)
        self.assertLessEqual(score, 1.0)
        
        # UUP down, UDN up = weak USD
        score = calculate_usd_strength_score(-2.0, 3.0)
        self.assertLess(score, 0)
        self.assertGreaterEqual(score, -1.0)
        
        # Both neutral (small difference)
        score = calculate_usd_strength_score(0.5, -0.5)
        # (0.5 - (-0.5)) / 8 = 1.0 / 8 = 0.125
        self.assertAlmostEqual(score, 0.125, places=2)
    
    def test_classify_carry_regime(self):
        """Test carry regime classification."""
        # Positive carry: UUP up >2%, UDN down
        regime = classify_carry_regime(2.5, -1.5)
        self.assertEqual(regime, "positive")
        
        # Negative carry: UDN up >2%, UUP down
        regime = classify_carry_regime(-1.5, 2.5)
        self.assertEqual(regime, "negative")
        
        # Neutral: no clear signal
        regime = classify_carry_regime(1.0, -0.5)
        self.assertEqual(regime, "neutral")
    
    def test_classify_momentum_direction(self):
        """Test momentum direction classification."""
        bullish = classify_momentum_direction(2.5, -1.5)
        self.assertEqual(bullish, "bullish")
        
        bearish = classify_momentum_direction(-1.5, 2.5)
        self.assertEqual(bearish, "bearish")
        
        neutral = classify_momentum_direction(1.0, 0.5)
        self.assertEqual(neutral, "neutral")
    
    def test_classify_volatility_regime(self):
        """Test volatility regime classification."""
        low = classify_volatility_regime(5.0)
        self.assertEqual(low, "low")
        
        medium = classify_volatility_regime(10.0)
        self.assertEqual(medium, "medium")
        
        high = classify_volatility_regime(18.0)
        self.assertEqual(high, "high")


class TestFXCarrySignal(unittest.TestCase):
    """Test FX carry signal generation."""
    
    def test_usd_strength_signal(self):
        """Test USD strength signal generation."""
        metrics = FXMetrics(
            timestamp="2026-05-14T16:00:00",
            uup_price=28.5,
            udn_price=21.3,
            uup_return_30d=3.0,  # > threshold
            udn_return_30d=-1.5,  # < -1.0
            usd_strength_score=0.56,
            carry_regime="positive",
            momentum_direction="bullish",
            volatility_regime="low",
        )
        
        signal = generate_signal(metrics)
        
        self.assertEqual(signal.signal_type, FXSignalType.USD_STRENGTH.value)
        self.assertTrue(signal.is_active)
        self.assertGreater(signal.confidence, 0)
        self.assertGreater(signal.spy_shift, 0)  # Add to SPY
        self.assertLess(signal.efa_shift, 0)  # Reduce international
    
    def test_usd_weakness_signal(self):
        """Test USD weakness signal generation."""
        metrics = FXMetrics(
            timestamp="2026-05-14T16:00:00",
            uup_price=27.0,
            udn_price=22.5,
            uup_return_30d=-1.5,
            udn_return_30d=3.0,  # > threshold
            usd_strength_score=-0.56,
            carry_regime="negative",
            momentum_direction="bearish",
            volatility_regime="low",
        )
        
        signal = generate_signal(metrics)
        
        self.assertEqual(signal.signal_type, FXSignalType.USD_WEAKNESS.value)
        self.assertTrue(signal.is_active)
        self.assertGreater(signal.confidence, 0)
        self.assertLess(signal.spy_shift, 0)  # Reduce SPY
        self.assertGreater(signal.efa_shift, 0)  # Add international
    
    def test_neutral_signal(self):
        """Test neutral signal when no clear trend."""
        metrics = FXMetrics(
            timestamp="2026-05-14T16:00:00",
            uup_price=28.0,
            udn_price=21.8,
            uup_return_30d=1.0,  # Below threshold
            udn_return_30d=-0.5,
            usd_strength_score=0.19,
            carry_regime="neutral",
            momentum_direction="neutral",
            volatility_regime="low",
        )
        
        signal = generate_signal(metrics)
        
        self.assertEqual(signal.signal_type, FXSignalType.NEUTRAL.value)
        self.assertEqual(signal.confidence, 0.0)
        self.assertEqual(signal.spy_shift, 0.0)
    
    def test_high_volatility_disables_signal(self):
        """Test signal disabled in high volatility."""
        metrics = FXMetrics(
            timestamp="2026-05-14T16:00:00",
            uup_price=28.5,
            udn_price=21.3,
            uup_return_30d=3.0,
            udn_return_30d=-1.5,
            usd_strength_score=0.56,
            carry_regime="positive",
            momentum_direction="bullish",
            volatility_regime="high",  # High volatility
        )
        
        signal = generate_signal(metrics)
        
        self.assertFalse(signal.is_active)
        self.assertIn("high_volatility", signal.reason_inactive)
    
    def test_momentum_conflict_disables_signal(self):
        """Test signal disabled when both UUP/UDN positive."""
        metrics = FXMetrics(
            timestamp="2026-05-14T16:00:00",
            uup_price=28.5,
            udn_price=22.0,
            uup_return_30d=2.0,
            udn_return_30d=1.5,  # Both positive = conflict
            usd_strength_score=0.06,
            carry_regime="neutral",
            momentum_direction="neutral",
            volatility_regime="low",
        )
        
        signal = generate_signal(metrics)
        
        self.assertFalse(signal.is_active)
        self.assertIn("momentum_conflict", signal.reason_inactive)
    
    def test_allocation_shifts_within_limits(self):
        """Test allocation shifts respect maximum limits."""
        metrics = FXMetrics(
            timestamp="2026-05-14T16:00:00",
            uup_price=28.5,
            udn_price=21.3,
            uup_return_30d=10.0,  # Very strong signal
            udn_return_30d=-5.0,
            usd_strength_score=1.0,
            carry_regime="positive",
            momentum_direction="bullish",
            volatility_regime="low",
        )
        
        signal = generate_signal(metrics)
        
        self.assertLessEqual(abs(signal.spy_shift), MAX_ALLOCATION_SHIFT)
        self.assertLessEqual(abs(signal.efa_shift), MAX_ALLOCATION_SHIFT / 2)
    
    def test_signal_to_dict(self):
        """Test signal conversion to dict."""
        signal = FXCarrySignal(
            timestamp="2026-05-14T16:00:00",
            signal_type="usd_strength",
            confidence=0.75,
            spy_shift=1.5,
            efa_shift=-0.75,
            vxus_shift=-0.75,
            uup_return_30d=3.0,
            udn_return_30d=-1.5,
            usd_strength_score=0.56,
            is_active=True,
        )
        
        d = signal.to_dict()
        self.assertEqual(d["signal_type"], "usd_strength")
        self.assertEqual(d["allocation_shifts"]["spy"], 1.5)
        self.assertTrue(d["is_active"])
    
    def test_get_ensemble_input(self):
        """Test ensemble input format."""
        metrics = FXMetrics(
            timestamp="2026-05-14T16:00:00",
            uup_price=28.5,
            udn_price=21.3,
            uup_return_30d=3.0,
            udn_return_30d=-1.5,
            usd_strength_score=0.56,
            carry_regime="positive",
            momentum_direction="bullish",
            volatility_regime="low",
        )
        
        with patch("src.signals.fx_carry_signal.fetch_fx_metrics", return_value=metrics):
            ensemble = get_ensemble_input()
        
        self.assertEqual(ensemble["source"], "fx_carry")
        self.assertEqual(ensemble["signal"], "bullish")
        self.assertEqual(ensemble["weight"], 0.02)  # 2% weight
        self.assertIn("allocation_delta", ensemble)
        self.assertIn("SPY", ensemble["allocation_delta"])


class TestFXSignalPersistence(unittest.TestCase):
    """Test saving and loading signals."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_path = Path("/tmp/test_fx_signal.json")
        if self.test_path.exists():
            self.test_path.unlink()
    
    def tearDown(self):
        """Clean up test fixtures."""
        if self.test_path.exists():
            self.test_path.unlink()
    
    def test_save_and_load_signal(self):
        """Test saving and loading a signal."""
        signal = FXCarrySignal(
            timestamp="2026-05-14T16:00:00",
            signal_type="usd_strength",
            confidence=0.75,
            spy_shift=1.5,
            efa_shift=-0.75,
            vxus_shift=-0.75,
            uup_return_30d=3.0,
            udn_return_30d=-1.5,
            usd_strength_score=0.56,
            is_active=True,
        )
        
        save_signal(signal, self.test_path)
        loaded = load_latest_signal(self.test_path)
        
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.signal_type, "usd_strength")
        self.assertEqual(loaded.confidence, 0.75)
        self.assertEqual(loaded.spy_shift, 1.5)


class TestAllocationImpact(unittest.TestCase):
    """Test allocation impact reporting."""
    
    def test_allocation_impact_format(self):
        """Test allocation impact format."""
        metrics = FXMetrics(
            timestamp="2026-05-14T16:00:00",
            uup_price=28.5,
            udn_price=21.3,
            uup_return_30d=3.0,
            udn_return_30d=-1.5,
            usd_strength_score=0.56,
            carry_regime="positive",
            momentum_direction="bullish",
            volatility_regime="low",
        )
        
        with patch("src.signals.fx_carry_signal.fetch_fx_metrics", return_value=metrics):
            impact = get_allocation_impact()
        
        self.assertIn("fx_signal_active", impact)
        self.assertIn("signal_type", impact)
        self.assertIn("allocations", impact)
        self.assertIn("spy", impact["allocations"])
        self.assertIn("status", impact)


if __name__ == "__main__":
    unittest.main()

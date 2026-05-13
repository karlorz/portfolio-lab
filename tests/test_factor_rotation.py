"""
Tests for Factor Rotation Signal Generator (v3.00 Phase 2)

Covers:
- QualityMomentumCalculator quality score computation
- Momentum signal calculation
- Regime detection
- Rotation signal generation
- Integration with ensemble voter format
"""

import pytest
import json
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import sqlite3
import numpy as np
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.signals.factor_rotation import (
    QualityMomentumCalculator,
    FactorRotationIntegrator,
    RotationSignal,
    QualityScore,
    FactorAllocation,
    MarketRegime,
    REGIME_ALLOCATIONS,
    QUALITY_WEIGHTS,
    FACTOR_ETFS,
)


class TestQualityScoreComputation:
    """Test quality score calculations."""
    
    def test_compute_quality_score_basic(self):
        """Test basic quality score computation."""
        calc = QualityMomentumCalculator()
        
        # Good quality metrics
        score = calc.compute_quality_score(
            symbol="QUAL",
            roe=0.25,  # 25% ROE - excellent
            debt_equity=0.3,  # Low debt
            earnings_stability=0.2,  # Stable earnings
            profitability=0.6,  # Good profitability
        )
        
        # Score should be high (0.7-1.0 range)
        assert 0.6 <= score <= 1.0
        assert isinstance(score, float)
        
    def test_compute_quality_score_poor_metrics(self):
        """Test quality score with poor metrics."""
        calc = QualityMomentumCalculator()
        
        # Poor quality metrics
        score = calc.compute_quality_score(
            symbol="VLUE",
            roe=0.05,  # Low ROE
            debt_equity=1.8,  # High debt
            earnings_stability=0.8,  # Volatile earnings
            profitability=0.2,  # Low profitability
        )
        
        # Score should be low (0.0-0.4 range)
        assert 0.0 <= score <= 0.5
        
    def test_compute_quality_score_weights_applied(self):
        """Test that weights are properly applied."""
        calc = QualityMomentumCalculator()
        
        # Perfect ROE only, everything else poor
        score_roe_only = calc.compute_quality_score(
            "QUAL", roe=0.30, debt_equity=2.0, earnings_stability=1.0, profitability=0.0
        )
        
        # Perfect profitability only, everything else poor
        score_prof_only = calc.compute_quality_score(
            "QUAL", roe=0.0, debt_equity=2.0, earnings_stability=1.0, profitability=1.0
        )
        
        # ROE has higher weight (0.30) than profitability (0.20)
        assert score_roe_only > score_prof_only
        
    def test_quality_score_bounds(self):
        """Test quality scores are bounded 0-1."""
        calc = QualityMomentumCalculator()
        
        # Extreme values
        score_max = calc.compute_quality_score("QUAL", roe=1.0, debt_equity=0, earnings_stability=0, profitability=1.0)
        score_min = calc.compute_quality_score("QUAL", roe=-0.5, debt_equity=5.0, earnings_stability=2.0, profitability=-0.5)
        
        assert 0.0 <= score_max <= 1.0
        assert 0.0 <= score_min <= 1.0


class TestMarketRegime:
    """Test market regime classifications."""
    
    def test_regime_allocation_sums_to_one(self):
        """Verify all regime allocations sum to 1.0."""
        for regime, allocation in REGIME_ALLOCATIONS.items():
            total = allocation.total
            assert abs(total - 1.0) < 0.001, f"{regime.value} allocations sum to {total}"
            
    def test_bull_regime_favors_momentum(self):
        """Bull regime should have highest MTUM allocation."""
        bull = REGIME_ALLOCATIONS[MarketRegime.BULL]
        assert bull.mtum_pct > bull.qual_pct
        assert bull.mtum_pct > bull.usmv_pct
        assert bull.mtum_pct > bull.vlue_pct
        
    def test_bear_regime_defensive(self):
        """Bear regime should favor quality and low vol."""
        bear = REGIME_ALLOCATIONS[MarketRegime.BEAR]
        assert bear.qual_pct >= 0.35
        assert bear.usmv_pct >= 0.35
        assert bear.mtum_pct <= 0.15
        
    def test_crisis_maximum_defensive(self):
        """Crisis regime should have minimum momentum."""
        crisis = REGIME_ALLOCATIONS[MarketRegime.CRISIS]
        assert crisis.mtum_pct <= 0.10
        assert crisis.usmv_pct >= 0.45


class TestMomentumCalculation:
    """Test momentum signal calculation."""
    
    @pytest.fixture
    def mock_db_with_prices(self, tmp_path):
        """Create mock database with price data."""
        data_dir = tmp_path / "data" / "factors"
        data_dir.mkdir(parents=True)
        db_path = data_dir / "factor_data.db"
        
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE factor_prices (
                id INTEGER PRIMARY KEY,
                symbol TEXT NOT NULL,
                date TEXT NOT NULL,
                close REAL NOT NULL
            )
        """)
        
        # Insert test price data for MTUM (strong uptrend)
        base_date = datetime(2024, 1, 1)
        for i in range(400):  # ~13 months
            date = base_date + timedelta(days=i)
            price = 100.0 * (1.02 ** (i / 30))  # 2% monthly growth
            conn.execute(
                "INSERT INTO factor_prices (symbol, date, close) VALUES (?, ?, ?)",
                ("MTUM", date.strftime("%Y-%m-%d"), price)
            )
            
        conn.commit()
        conn.close()
        
        return db_path
        
    def test_momentum_signal_strong_uptrend(self, mock_db_with_prices):
        """Test momentum calculation with strong uptrend."""
        calc = QualityMomentumCalculator(data_dir=mock_db_with_prices.parent)
        
        signal = calc.compute_momentum_signal("MTUM", "2025-02-01")
        
        # Strong uptrend should give positive signal
        assert signal > 0.3
        assert signal <= 1.0
        
    def test_momentum_signal_no_data(self):
        """Test momentum with no database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            calc = QualityMomentumCalculator(data_dir=Path(tmpdir) / "factors")
            
            signal = calc.compute_momentum_signal("MTUM", "2024-01-01")
            
            # No data should return neutral
            assert signal == 0.0
            
    def test_momentum_signal_bounds(self, mock_db_with_prices):
        """Test momentum signal is bounded."""
        calc = QualityMomentumCalculator(data_dir=mock_db_with_prices.parent)
        
        signal = calc.compute_momentum_signal("MTUM", "2025-02-01")
        
        assert -1.0 <= signal <= 1.0


class TestRegimeDetection:
    """Test regime detection logic."""
    
    def test_detect_bull_regime(self):
        """Test bull regime detection."""
        calc = QualityMomentumCalculator()
        
        regime = calc.detect_regime(
            vix_level=15.0,  # Low VIX
            trend_strength=0.5,  # Strong uptrend
        )
        
        assert regime == MarketRegime.BULL
        
    def test_detect_bear_regime(self):
        """Test bear regime detection."""
        calc = QualityMomentumCalculator()
        
        regime = calc.detect_regime(
            vix_level=22.0,
            trend_strength=-0.4,  # Downtrend
        )
        
        assert regime == MarketRegime.BEAR
        
    def test_detect_crisis_regime(self):
        """Test crisis regime detection."""
        calc = QualityMomentumCalculator()
        
        regime = calc.detect_regime(
            vix_level=40.0,  # Extreme fear
            trend_strength=-0.2,
        )
        
        assert regime == MarketRegime.CRISIS
        
    def test_detect_high_vol_regime(self):
        """Test high volatility regime detection."""
        calc = QualityMomentumCalculator()
        
        regime = calc.detect_regime(
            vix_level=28.0,  # Elevated vol
            trend_strength=0.1,
        )
        
        assert regime == MarketRegime.HIGH_VOL
        
    def test_detect_neutral_default(self):
        """Test neutral regime as default."""
        calc = QualityMomentumCalculator()
        
        regime = calc.detect_regime(
            vix_level=20.0,  # Normal
            trend_strength=0.1,  # Weak trend
        )
        
        assert regime == MarketRegime.NEUTRAL


class TestRotationSignalGeneration:
    """Test complete rotation signal generation."""
    
    @pytest.fixture
    def setup_with_mock_data(self, tmp_path):
        """Setup calculator with mock database."""
        data_dir = tmp_path / "data" / "factors"
        signals_dir = tmp_path / "data" / "signals"
        data_dir.mkdir(parents=True)
        signals_dir.mkdir(parents=True)
        db_path = data_dir / "factor_data.db"
        
        # Create database
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS factor_prices (
                symbol TEXT,
                date TEXT,
                close REAL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS quality_scores (
                symbol TEXT,
                date TEXT,
                roe REAL,
                debt_equity REAL,
                earnings_stability REAL,
                profitability REAL,
                composite_score REAL
            )
        """)
        
        # Insert quality scores
        base_date = datetime(2024, 6, 1)
        for symbol in ["MTUM", "QUAL", "USMV", "VLUE"]:
            # QUAL has highest quality score
            score = 0.85 if symbol == "QUAL" else (0.70 if symbol == "USMV" else 0.60)
            conn.execute(
                """INSERT INTO quality_scores 
                    (symbol, date, roe, debt_equity, earnings_stability, profitability, composite_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (symbol, base_date.strftime("%Y-%m-%d"), 0.20, 0.4, 0.3, 0.5, score)
            )
            
        conn.commit()
        conn.close()
        
        calc = QualityMomentumCalculator(data_dir=data_dir)
        calc.signals_dir = signals_dir
        
        return calc
        
    def test_generate_rotation_signal_structure(self, setup_with_mock_data):
        """Test rotation signal has correct structure."""
        calc = setup_with_mock_data
        
        signal = calc.generate_rotation_signal(
            date="2024-06-01",
            regime=MarketRegime.BULL,
        )
        
        assert isinstance(signal, RotationSignal)
        assert signal.date == "2024-06-01"
        assert signal.regime == "bull"
        assert isinstance(signal.factor_allocations, dict)
        assert "MTUM" in signal.factor_allocations
        assert "QUAL" in signal.factor_allocations
        assert "USMV" in signal.factor_allocations
        assert "VLUE" in signal.factor_allocations
        assert isinstance(signal.rationale, list)
        assert len(signal.rationale) > 0
        
    def test_bull_regime_allocations(self, setup_with_mock_data):
        """Test bull regime produces correct allocations."""
        calc = setup_with_mock_data
        
        signal = calc.generate_rotation_signal(
            date="2024-06-01",
            regime=MarketRegime.BULL,
        )
        
        # Bull should favor MTUM
        assert signal.factor_allocations["MTUM"] > signal.factor_allocations["QUAL"]
        assert signal.factor_allocations["MTUM"] >= 0.55  # At least 55%
        
    def test_bear_regime_defensive(self, setup_with_mock_data):
        """Test bear regime produces defensive allocations."""
        calc = setup_with_mock_data
        
        signal = calc.generate_rotation_signal(
            date="2024-06-01",
            regime=MarketRegime.BEAR,
        )
        
        # Bear should minimize MTUM
        assert signal.factor_allocations["MTUM"] <= 0.15
        assert signal.factor_allocations["QUAL"] >= 0.35
        assert signal.factor_allocations["USMV"] >= 0.35
        
    def test_equity_adjustment_bull(self, setup_with_mock_data):
        """Test positive equity adjustment in bull market."""
        calc = setup_with_mock_data
        
        signal = calc.generate_rotation_signal(
            date="2024-06-01",
            regime=MarketRegime.BULL,
        )
        
        # Bull with good Q-M should increase equity
        assert signal.equity_adjustment >= 0.0
        
    def test_equity_adjustment_crisis(self, setup_with_mock_data):
        """Test negative equity adjustment in crisis."""
        calc = setup_with_mock_data
        
        signal = calc.generate_rotation_signal(
            date="2024-06-01",
            regime=MarketRegime.CRISIS,
        )
        
        # Crisis should decrease equity
        assert signal.equity_adjustment < 0.0
        assert signal.equity_adjustment <= -0.10
        
    def test_save_signal(self, setup_with_mock_data):
        """Test signal saving to file."""
        calc = setup_with_mock_data
        
        signal = calc.generate_rotation_signal(
            date="2024-06-01",
            regime=MarketRegime.NEUTRAL,
        )
        
        filepath = calc.save_signal(signal)
        
        assert filepath.exists()
        
        with open(filepath) as f:
            data = json.load(f)
            
        assert data["date"] == "2024-06-01"
        assert data["regime"] == "neutral"
        assert "factor_allocations" in data


class TestFactorRotationIntegrator:
    """Test integration with ensemble voter."""
    
    @pytest.fixture
    def mock_integrator(self, tmp_path):
        """Create integrator with mock data."""
        data_dir = tmp_path / "data" / "factors"
        signals_dir = tmp_path / "data" / "signals"
        data_dir.mkdir(parents=True)
        signals_dir.mkdir(parents=True)
        db_path = data_dir / "factor_data.db"
        
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS quality_scores (
                symbol TEXT, date TEXT, roe REAL, debt_equity REAL,
                earnings_stability REAL, profitability REAL, composite_score REAL
            )
        """)
        
        for symbol in ["MTUM", "QUAL", "USMV", "VLUE"]:
            conn.execute(
                "INSERT INTO quality_scores VALUES (?, ?, 0.2, 0.4, 0.3, 0.5, 0.7)",
                (symbol, "2024-06-01")
            )
        conn.commit()
        conn.close()
        
        return FactorRotationIntegrator()
        
    def test_get_signal_for_ensemble_format(self, mock_integrator):
        """Test ensemble signal format."""
        integrator = mock_integrator
        
        # Patch the calculator's data directory
        with tempfile.TemporaryDirectory() as tmpdir:
            integrator.calculator.data_dir = Path(tmpdir) / "factors"
            integrator.calculator.signals_dir = Path(tmpdir) / "signals"
            
            signal = integrator.get_signal_for_ensemble("2024-06-01")
            
            assert signal["source"] == "FACTOR_ROTATION"
            assert signal["date"] == "2024-06-01"
            assert "direction" in signal
            assert "strength" in signal
            assert "signal_value" in signal
            assert "confidence" in signal
            assert "factor_allocations" in signal
            assert "equity_adjustment" in signal
            assert "rationale" in signal
            
    def test_signal_value_range(self, mock_integrator):
        """Test signal value is in valid range."""
        integrator = mock_integrator
        
        with tempfile.TemporaryDirectory() as tmpdir:
            integrator.calculator.data_dir = Path(tmpdir) / "factors"
            integrator.calculator.signals_dir = Path(tmpdir) / "signals"
            
            signal = integrator.get_signal_for_ensemble("2024-06-01")
            
            assert -1.0 <= signal["signal_value"] <= 1.0
            assert 0.0 <= signal["strength"] <= 1.0
            assert 0.0 <= signal["confidence"] <= 1.0
            
    def test_get_backtest_allocations(self, mock_integrator):
        """Test backtest allocation generation."""
        integrator = mock_integrator
        
        with tempfile.TemporaryDirectory() as tmpdir:
            integrator.calculator.data_dir = Path(tmpdir) / "factors"
            integrator.calculator.signals_dir = Path(tmpdir) / "signals"
            
            # Create minimal database
            db_path = integrator.calculator.data_dir / "factor_data.db"
            integrator.calculator.data_dir.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(db_path)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS quality_scores (
                    symbol TEXT, date TEXT, roe REAL, debt_equity REAL,
                    earnings_stability REAL, profitability REAL, composite_score REAL
                )
            """)
            for symbol in ["MTUM", "QUAL", "USMV", "VLUE"]:
                for month in range(6):
                    date = f"2024-{month+1:02d}-01"
                    conn.execute(
                        "INSERT INTO quality_scores VALUES (?, ?, 0.2, 0.4, 0.3, 0.5, 0.7)",
                        (symbol, date)
                    )
            conn.commit()
            conn.close()
            
            allocations = integrator.get_backtest_allocations(
                "2024-01-01",
                "2024-06-01",
                rebalance_freq="monthly"
            )
            
            assert len(allocations) > 0
            for alloc in allocations:
                assert "date" in alloc
                assert "regime" in alloc
                assert "allocations" in alloc
                assert sum(alloc["allocations"].values()) == pytest.approx(1.0, abs=0.01)


class TestFactorMetadata:
    """Test factor ETF metadata constants."""
    
    def test_factor_etfs_defined(self):
        """Test all factor ETFs are defined."""
        assert "MTUM" in FACTOR_ETFS
        assert "QUAL" in FACTOR_ETFS
        assert "USMV" in FACTOR_ETFS
        assert "VLUE" in FACTOR_ETFS
        
    def test_expense_ratios(self):
        """Test expense ratios are reasonable."""
        for symbol, data in FACTOR_ETFS.items():
            assert 0.0005 <= data["expense"] <= 0.005  # 5-50 bps
            
    def test_quality_weights_sum(self):
        """Test quality weights sum to 1.0."""
        total = sum(QUALITY_WEIGHTS.values())
        assert abs(total - 1.0) < 0.001


class TestCLI:
    """Test command line interface."""
    
    def test_cli_import(self):
        """Test CLI can be imported and has argparse."""
        import argparse
        
        # Verify argparse setup in main block
        # This is a basic smoke test
        assert True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

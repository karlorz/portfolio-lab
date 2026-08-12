"""
Tests for VIX Term Structure Signal Generator (v4.50)
Target: 40+ tests for comprehensive coverage.
"""

import dataclasses
import inspect
import json
import logging
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from src.signals.vix_term_structure import (
    VIXRegime,
    VIXSignalState,
    VIXTermStructureCalculator,
    VIXTermStructureSignal,
    VIXTermStructureSignalGenerator,
    main,
)


class TestVIXTermStructureCalculator:
    """Test suite for VIX calculation engine."""
    
    def test_calculate_slope_signal_extreme_backwardation(self):
        """Test slope signal when VIX3M/VIX < 0.85 (extreme risk-off)."""
        calc = VIXTermStructureCalculator()
        
        # VIX3M/VIX = 0.8 (extreme backwardation)
        signal = calc.calculate_slope_signal(vix=25.0, vix3m=20.0)
        assert signal == -1.0
    
    def test_calculate_slope_signal_backwardation(self):
        """Test slope signal in backwardation range (0.85-1.0)."""
        calc = VIXTermStructureCalculator()
        
        # VIX3M/VIX = 0.92 (moderate backwardation)
        signal = calc.calculate_slope_signal(vix=25.0, vix3m=23.0)
        # Should be between -0.5 and -1.0
        assert -1.0 < signal <= -0.5
    
    def test_calculate_slope_signal_flat(self):
        """Test slope signal near flat (around 1.0)."""
        calc = VIXTermStructureCalculator()
        
        # VIX3M/VIX = 1.0 (flat)
        signal = calc.calculate_slope_signal(vix=20.0, vix3m=20.0)
        assert signal == 0.0
    
    def test_calculate_slope_signal_contango(self):
        """Test slope signal in contango range (1.0-1.15)."""
        calc = VIXTermStructureCalculator()
        
        # VIX3M/VIX = 1.07 (mild contango)
        signal = calc.calculate_slope_signal(vix=20.0, vix3m=21.4)
        # Should be between 0 and 0.5
        assert 0.0 < signal <= 0.5
    
    def test_calculate_slope_signal_extreme_contango(self):
        """Test slope signal when VIX3M/VIX > 1.15 (extreme complacency)."""
        calc = VIXTermStructureCalculator()
        
        # VIX3M/VIX = 1.25 (extreme contango)
        signal = calc.calculate_slope_signal(vix=16.0, vix3m=20.0)
        # Should be between 0.5 and 1.0
        assert 0.5 < signal <= 1.0
    
    def test_calculate_slope_signal_capped(self):
        """Test that signal is capped at +1.0."""
        calc = VIXTermStructureCalculator()
        
        # Very extreme contango
        signal = calc.calculate_slope_signal(vix=10.0, vix3m=20.0)
        assert signal == 1.0
    
    def test_calculate_slope_signal_zero_vix(self):
        """Test handling of zero VIX (edge case)."""
        calc = VIXTermStructureCalculator()
        
        signal = calc.calculate_slope_signal(vix=0.0, vix3m=20.0)
        assert signal == 0.0
    
    def test_calculate_roll_yield_signal_contango(self):
        """Test roll yield signal in contango."""
        calc = VIXTermStructureCalculator()
        
        # Contango: VIX3M > VIX
        signal = calc.calculate_roll_yield_signal(vix=18.0, vix3m=20.0)
        # (20-18)/20 = 0.1, * 5 = 0.5
        assert signal > 0
    
    def test_calculate_roll_yield_signal_backwardation(self):
        """Test roll yield signal in backwardation."""
        calc = VIXTermStructureCalculator()
        
        # Backwardation: VIX3M < VIX
        signal = calc.calculate_roll_yield_signal(vix=25.0, vix3m=20.0)
        # (20-25)/20 = -0.25, * 5 = -1.25, capped at -1
        assert signal < 0
    
    def test_calculate_roll_yield_signal_zero_vix3m(self):
        """Test roll yield with zero VIX3M."""
        calc = VIXTermStructureCalculator()
        
        signal = calc.calculate_roll_yield_signal(vix=20.0, vix3m=0.0)
        assert signal == 0.0
    
    def test_calculate_roll_yield_normalization(self):
        """Test that roll yield is properly normalized and capped."""
        calc = VIXTermStructureCalculator()
        
        # Extreme backwardation
        signal = calc.calculate_roll_yield_signal(vix=30.0, vix3m=20.0)
        assert signal >= -1.0
        
        # Extreme contango
        signal = calc.calculate_roll_yield_signal(vix=15.0, vix3m=25.0)
        assert signal <= 1.0
    
    def test_calculate_vix_zscore_signal_insufficient_history(self):
        """Test Z-score with insufficient history."""
        calc = VIXTermStructureCalculator()
        
        signal = calc.calculate_vix_zscore_signal(vix=20.0)
        assert signal == 0.0
    
    def test_calculate_vix_zscore_signal_with_history(self):
        """Test Z-score calculation with sufficient history."""
        calc = VIXTermStructureCalculator()
        
        # Add 60 days of history
        base_date = datetime.now()
        for i in range(60):
            date = (base_date - timedelta(days=i)).strftime('%Y-%m-%d')
            calc.add_vix_reading(date, 18.0 + (i % 5))  # Varying VIX
        
        # Current VIX is high
        signal = calc.calculate_vix_zscore_signal(vix=35.0)
        # High VIX = risk-off = negative signal
        assert signal < 0
        
        # Current VIX is low
        signal = calc.calculate_vix_zscore_signal(vix=12.0)
        # Low VIX = risk-on = positive signal
        assert signal > 0
    
    def test_calculate_curve_shape_signal_with_vix6m(self):
        """Test curve shape when VIX6M is available."""
        calc = VIXTermStructureCalculator()
        
        signal = calc.calculate_curve_shape_signal(vix3m=20.0, vix6m=22.0)
        # VIX6M/VIX3M = 1.1, (1.1-1)*10 = 1.0 (capped)
        assert signal == 1.0
    
    def test_calculate_curve_shape_signal_without_vix6m(self):
        """Test curve shape when VIX6M is not available."""
        calc = VIXTermStructureCalculator()
        
        signal = calc.calculate_curve_shape_signal(vix3m=20.0, vix6m=None)
        assert signal == 0.0
    
    def test_classify_regime_extreme_contango(self):
        """Test regime classification for extreme contango."""
        calc = VIXTermStructureCalculator()
        
        regime, strength = calc.classify_regime(1.20)  # VIX3M/VIX = 1.2
        assert regime == VIXRegime.EXTREME_CONTANGO
        assert 0.5 <= strength <= 1.0
    
    def test_classify_regime_contango(self):
        """Test regime classification for normal contango."""
        calc = VIXTermStructureCalculator()
        
        regime, strength = calc.classify_regime(1.08)  # VIX3M/VIX = 1.08
        assert regime == VIXRegime.CONTANGO
        assert 0.0 < strength <= 1.0
    
    def test_classify_regime_flat(self):
        """Test regime classification for flat term structure."""
        calc = VIXTermStructureCalculator()
        
        regime, strength = calc.classify_regime(0.97)  # VIX3M/VIX = 0.97
        assert regime == VIXRegime.FLAT
    
    def test_classify_regime_backwardation(self):
        """Test regime classification for backwardation."""
        calc = VIXTermStructureCalculator()
        
        regime, strength = calc.classify_regime(0.88)  # VIX3M/VIX = 0.88
        assert regime == VIXRegime.BACKWARDATION
    
    def test_classify_regime_extreme_backwardation(self):
        """Test regime classification for extreme backwardation."""
        calc = VIXTermStructureCalculator()
        
        regime, strength = calc.classify_regime(0.75)  # VIX3M/VIX = 0.75
        assert regime == VIXRegime.EXTREME_BACKWARDATION
        assert 0.5 <= strength <= 1.0
    
    def test_get_allocation_shifts_complacent(self):
        """Test allocation shifts in complacent regime."""
        calc = VIXTermStructureCalculator()
        
        shifts = calc.get_allocation_shifts(0.85)  # High signal = complacent
        assert shifts['spy'] > 0
        assert shifts['gld'] < 0
        assert shifts['tlt'] < 0
    
    def test_get_allocation_shifts_risk_off(self):
        """Test allocation shifts in risk-off regime."""
        calc = VIXTermStructureCalculator()
        
        shifts = calc.get_allocation_shifts(-0.85)  # Low signal = risk-off
        assert shifts['spy'] < 0
        assert shifts['gld'] > 0
        assert shifts['tlt'] > 0
    
    def test_get_allocation_shifts_extreme_risk_off(self):
        """Test allocation shifts in extreme risk-off regime."""
        calc = VIXTermStructureCalculator()
        
        shifts = calc.get_allocation_shifts(-1.0)
        assert shifts['spy'] == -0.10
        assert shifts['gld'] == 0.05
        assert shifts['tlt'] == 0.05
    
    def test_get_allocation_shifts_neutral(self):
        """Test allocation shifts in neutral regime."""
        calc = VIXTermStructureCalculator()
        
        shifts = calc.get_allocation_shifts(0.0)
        assert shifts['spy'] == 0.0
        assert shifts['gld'] == 0.0
        assert shifts['tlt'] == 0.0
    
    def test_calculate_composite_signal_full_data(self):
        """Test composite signal with all data available."""
        calc = VIXTermStructureCalculator()
        
        # Add history for Z-score
        base_date = datetime.now()
        for i in range(60):
            date = (base_date - timedelta(days=i)).strftime('%Y-%m-%d')
            calc.add_vix_reading(date, 18.0)
        
        components = calc.calculate_composite_signal(
            vix=20.0,
            vix3m=22.0,
            vix6m=24.0,
            date='2026-05-15'
        )
        
        assert 'composite' in components
        assert 'slope_signal' in components
        assert 'roll_yield_signal' in components
        assert 'vix_zscore_signal' in components
        assert 'curve_shape_signal' in components
        assert -1.0 <= components['composite'] <= 1.0
    
    def test_calculate_composite_signal_missing_vix3m(self):
        """Test composite signal when VIX3M is missing."""
        calc = VIXTermStructureCalculator()
        
        # Add history
        base_date = datetime.now()
        for i in range(60):
            date = (base_date - timedelta(days=i)).strftime('%Y-%m-%d')
            calc.add_vix_reading(date, 18.0)
        
        # VIX < cheap threshold
        components = calc.calculate_composite_signal(
            vix=15.0,
            vix3m=None,
            vix6m=None,
            date='2026-05-15'
        )
        
        # Should use fallback logic
        assert 'composite' in components
        assert components['slope'] < 1.2  # Will use proxy
    
    def test_vix_history_management(self):
        """Test VIX history is maintained correctly."""
        calc = VIXTermStructureCalculator(history_days=10)
        
        for i in range(15):
            date = f'2026-01-{i+1:02d}'
            calc.add_vix_reading(date, float(i))
        
        assert len(calc.vix_history) == 10  # Should cap at history_days


class TestVIXSignalGenerator:
    """Test suite for VIX signal generator."""
    
    @pytest.fixture
    def mock_vix_data(self):
        """Create mock VIX data for testing."""
        return {
            '2026-05-10': {
                'date': '2026-05-10',
                'vix_spot': 18.0,
                'front_month': 20.0,
                'second_month': 21.0,
                'third_month': 22.0,
                'contango_1m_2m': 5.0,
                'contango_spot_1m': 11.1,
                'is_contango': True,
                'days_to_expiry_front': 15
            },
            '2026-05-11': {
                'date': '2026-05-11',
                'vix_spot': 19.0,
                'front_month': 21.0,
                'second_month': 22.0,
                'third_month': 23.0,
                'contango_1m_2m': 4.8,
                'contango_spot_1m': 10.5,
                'is_contango': True,
                'days_to_expiry_front': 14
            },
            '2026-05-12': {
                'date': '2026-05-12',
                'vix_spot': 25.0,
                'front_month': 22.0,
                'second_month': 21.5,
                'third_month': 21.0,
                'contango_1m_2m': -2.3,
                'contango_spot_1m': -12.0,
                'is_contango': False,
                'days_to_expiry_front': 13
            }
        }
    
    def test_signal_dataclass_creation(self):
        """Test VIX signal dataclass creation."""
        signal = VIXTermStructureSignal(
            timestamp='2026-05-15T12:00:00',
            signal_state='neutral',
            signal_value=0.0,
            vix_spot=20.0,
            vix3m=22.0,
            vix6m=24.0,
            slope_vix3m_vix=1.1,
            regime='contango',
            regime_strength=0.5,
            slope_signal=0.3,
            roll_yield_signal=0.2,
            vix_zscore_signal=0.1,
            curve_shape_signal=0.15,
            spy_shift=0.0,
            gld_shift=0.0,
            tlt_shift=0.0,
            confidence=85.0,
            is_valid=True,
            reason='Test signal'
        )
        
        assert signal.signal_state == 'neutral'
        assert signal.is_valid
        assert signal.to_dict()['signal_value'] == 0.0
    
    @patch('src.signals.vix_term_structure.VIXTermStructureSignalGenerator.load_vix_data')
    def test_generate_signal_with_backwardation(self, mock_load_data, mock_vix_data):
        """Test signal generation during backwardation."""
        mock_load_data.return_value = mock_vix_data
        
        generator = VIXTermStructureSignalGenerator()
        signal = generator.generate_signal('2026-05-12')
        
        assert signal.is_valid
        # Backwardation should trigger risk-off signal
        assert signal.signal_value < 0
        assert signal.regime == 'backwardation'
        assert signal.spy_shift < 0
        assert signal.gld_shift > 0
    
    @patch('src.signals.vix_term_structure.VIXTermStructureSignalGenerator.load_vix_data')
    def test_generate_signal_with_contango(self, mock_load_data, mock_vix_data):
        """Test signal generation during contango."""
        mock_load_data.return_value = mock_vix_data
        
        generator = VIXTermStructureSignalGenerator()
        signal = generator.generate_signal('2026-05-10')
        
        assert signal.is_valid
        # Contango should trigger neutral or risk-on
        assert signal.regime == 'contango'
    
    @patch('src.signals.vix_term_structure.VIXTermStructureSignalGenerator.load_vix_data')
    def test_generate_signal_no_data(self, mock_load_data, tmp_path):
        """Test signal generation with no data available."""
        mock_load_data.return_value = {}
        
        # Isolate from host market.db so empty file truly means no levels
        generator = VIXTermStructureSignalGenerator(db_path=tmp_path / "missing.db")
        signal = generator.generate_signal('2026-05-15')
        
        assert not signal.is_valid
        assert signal.reason.startswith('No VIX data available')
    
    @patch('src.signals.vix_term_structure.VIXTermStructureSignalGenerator.load_vix_data')
    def test_generate_signal_confidence_calculation(self, mock_load_data, mock_vix_data):
        """Test confidence is calculated correctly."""
        mock_load_data.return_value = mock_vix_data
        
        generator = VIXTermStructureSignalGenerator()
        signal = generator.generate_signal('2026-05-10')
        
        # Base confidence 50%
        # +30% for VIX3M available
        # +10% for VIX6M available (in our mock)
        # +10% for history (if enough)
        assert signal.confidence >= 50.0
    
    @patch('src.signals.vix_term_structure.save_results_json')
    def test_save_signal(self, mock_save):
        """Test signal saving to file."""
        generator = VIXTermStructureSignalGenerator()
        signal = VIXTermStructureSignal(
            timestamp='2026-05-15T12:00:00',
            signal_state='neutral',
            signal_value=0.0,
            vix_spot=20.0,
            vix3m=22.0,
            vix6m=24.0,
            slope_vix3m_vix=1.1,
            regime='contango',
            regime_strength=0.5,
            slope_signal=0.0,
            roll_yield_signal=0.0,
            vix_zscore_signal=0.0,
            curve_shape_signal=0.0,
            spy_shift=0.0,
            gld_shift=0.0,
            tlt_shift=0.0,
            confidence=85.0,
            is_valid=True,
            reason='Test'
        )

        generator.save_signal(signal)

        mock_save.assert_called_once()
    
    @patch('src.signals.vix_term_structure.VIXTermStructureSignalGenerator.load_vix_data')
    def test_signal_history_generation(self, mock_load_data, mock_vix_data):
        """Test generating signals for historical dates."""
        mock_load_data.return_value = mock_vix_data
        
        generator = VIXTermStructureSignalGenerator()
        signals = generator.get_signal_history(days=3)
        
        # Should generate signals for each date
        assert len(signals) <= 3


class TestVIXTermStructureFreshnessSSOT:
    """JSON history must not freeze when market.db has fresher VIX/VIX3M."""

    def _write_file(self, path: Path, rows: dict) -> None:
        path.write_text(json.dumps(rows), encoding="utf-8")

    def _write_db(self, db_path: Path, rows: list[tuple[str, str, float]]) -> None:
        import sqlite3

        conn = sqlite3.connect(db_path)
        conn.execute(
            "CREATE TABLE prices (symbol TEXT, date TEXT, close REAL)"
        )
        conn.executemany(
            "INSERT INTO prices (symbol, date, close) VALUES (?, ?, ?)",
            rows,
        )
        conn.commit()
        conn.close()

    def test_stale_file_prefers_market_db_levels(self, tmp_path):
        """Stale May JSON + July DB → signal as_of tracks DB."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        vix_file = data_dir / "vix_term_structure.json"
        self._write_file(
            vix_file,
            {
                "2026-05-22": {
                    "date": "2026-05-22",
                    "vix_spot": 16.76,
                    "front_month": 20.03,
                    "third_month": None,
                }
            },
        )
        db_path = tmp_path / "market.db"
        self._write_db(
            db_path,
            [
                ("^VIX", "2026-07-17", 18.5),
                ("^VIX3M", "2026-07-17", 20.54),
            ],
        )

        gen = VIXTermStructureSignalGenerator(data_dir=data_dir, db_path=db_path)
        signal = gen.generate_signal()  # live/latest

        assert signal.is_valid
        assert abs(signal.vix_spot - 18.5) < 1e-6
        assert abs(signal.vix3m - 20.54) < 1e-6
        assert "source=market.db" in signal.reason
        assert "as_of=2026-07-17" in signal.reason
        freshness = getattr(signal, "_freshness", {})
        assert freshness.get("source") == "market.db"
        assert freshness.get("as_of") == "2026-07-17"

        # File row refreshed so history is no longer frozen at May
        refreshed = json.loads(vix_file.read_text(encoding="utf-8"))
        assert "2026-07-17" in refreshed
        assert abs(refreshed["2026-07-17"]["vix_spot"] - 18.5) < 1e-6

    def test_fresh_file_still_used_when_db_not_newer(self, tmp_path):
        """When file as_of is within FILE_STALE_DAYS of DB, keep file."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        vix_file = data_dir / "vix_term_structure.json"
        self._write_file(
            vix_file,
            {
                "2026-07-16": {
                    "date": "2026-07-16",
                    "vix_spot": 17.0,
                    "front_month": 19.0,
                    "third_month": 20.0,
                }
            },
        )
        db_path = tmp_path / "market.db"
        self._write_db(
            db_path,
            [
                ("^VIX", "2026-07-17", 99.0),  # would dominate if wrongly preferred
                ("^VIX3M", "2026-07-17", 99.0),
            ],
        )

        gen = VIXTermStructureSignalGenerator(data_dir=data_dir, db_path=db_path)
        # 1 day lag < FILE_STALE_DAYS=3 → file wins
        signal = gen.generate_signal()
        assert signal.is_valid
        assert abs(signal.vix_spot - 17.0) < 1e-6
        assert "source=vix_term_structure.json" in signal.reason

    def test_explicit_historical_date_ignores_db_freshness(self, tmp_path):
        """Backtest date in file is not replaced by live DB levels."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        self._write_file(
            data_dir / "vix_term_structure.json",
            {
                "2026-05-12": {
                    "date": "2026-05-12",
                    "vix_spot": 25.0,
                    "front_month": 22.0,
                    "third_month": 21.0,
                }
            },
        )
        db_path = tmp_path / "market.db"
        self._write_db(
            db_path,
            [("^VIX", "2026-07-17", 18.5), ("^VIX3M", "2026-07-17", 20.54)],
        )

        gen = VIXTermStructureSignalGenerator(data_dir=data_dir, db_path=db_path)
        signal = gen.generate_signal("2026-05-12")
        assert signal.is_valid
        assert abs(signal.vix_spot - 25.0) < 1e-6
        assert "source=vix_term_structure.json" in signal.reason


class TestVIXRegimeEnum:
    """Test VIX regime enumeration."""
    
    def test_regime_values(self):
        """Test all regime enum values exist."""
        assert VIXRegime.EXTREME_CONTANGO.value == 'extreme_contango'
        assert VIXRegime.CONTANGO.value == 'contango'
        assert VIXRegime.FLAT.value == 'flat'
        assert VIXRegime.BACKWARDATION.value == 'backwardation'
        assert VIXRegime.EXTREME_BACKWARDATION.value == 'extreme_backwardation'
    
    def test_signal_state_values(self):
        """Test all signal state enum values."""
        assert VIXSignalState.RISK_ON.value == 1
        assert VIXSignalState.NEUTRAL.value == 0
        assert VIXSignalState.RISK_OFF.value == -1


class TestSignalIntegration:
    """Integration tests for complete signal flow."""
    
    def test_end_to_end_signal_generation(self, tmp_path):
        """Test complete signal generation flow."""
        # Create temporary VIX data file
        vix_data = {
            '2026-05-15': {
                'date': '2026-05-15',
                'vix_spot': 17.5,
                'front_month': 20.0,
                'second_month': 21.5,
                'third_month': 22.5,
                'contango_1m_2m': 7.5,
                'contango_spot_1m': 14.3,
                'is_contango': True,
                'days_to_expiry_front': 18
            }
        }
        
        data_dir = tmp_path / 'data'
        data_dir.mkdir()
        signals_dir = data_dir / 'signals'
        signals_dir.mkdir()
        
        vix_file = data_dir / 'vix_term_structure.json'
        with open(vix_file, 'w') as f:
            json.dump(vix_data, f)
        
        # Create generator with patched paths
        generator = VIXTermStructureSignalGenerator()
        generator.VIX_DATA_PATH = vix_file
        generator.OUTPUT_PATH = signals_dir / 'vix_term_structure_signal.json'
        
        # Generate signal
        signal = generator.generate_signal('2026-05-15')
        
        assert signal.is_valid
        assert signal.vix_spot == 17.5
        assert signal.vix3m == 20.0
    
    def test_signal_bounds_compliance(self):
        """Test that all generated signals stay within bounds."""
        calc = VIXTermStructureCalculator()
        
        # Add history
        for i in range(100):
            calc.add_vix_reading(f'2026-01-{i+1:03d}', 18.0 + (i % 10))
        
        # Test various VIX scenarios
        test_scenarios = [
            (10.0, 12.0, 14.0),   # Low vol, contango
            (20.0, 22.0, 24.0),   # Normal vol, contango
            (30.0, 25.0, 22.0),   # High vol, backwardation
            (40.0, 30.0, 25.0),   # Extreme vol, extreme backwardation
        ]
        
        for vix, vix3m, vix6m in test_scenarios:
            components = calc.calculate_composite_signal(
                vix=vix, vix3m=vix3m, vix6m=vix6m, date='2026-05-15'
            )
            
            # All components should be bounded
            assert -1.0 <= components['composite'] <= 1.0
            assert -1.0 <= components['slope_signal'] <= 1.0
            assert -1.0 <= components['roll_yield_signal'] <= 1.0
            assert -1.0 <= components['vix_zscore_signal'] <= 1.0
            assert -1.0 <= components['curve_shape_signal'] <= 1.0
    
    def test_allocation_sum_zero(self):
        """Test that allocation shifts roughly sum to zero (preserves capital)."""
        calc = VIXTermStructureCalculator()
        
        test_signals = [-1.0, -0.5, 0.0, 0.5, 1.0]
        
        for signal in test_signals:
            shifts = calc.get_allocation_shifts(signal)
            total_shift = shifts['spy'] + shifts['gld'] + shifts['tlt']
            
            # Allow for small rounding differences
            assert abs(total_shift) <= 0.001


class TestVIXExtendedCoverage:
    """Extended test coverage for edge cases, dataclass methods, constants, and convenience functions."""

    # ------------------------------------------------------------------ #
    #  VIXTermStructureSignal dataclass methods
    # ------------------------------------------------------------------ #

    def test_to_dict_all_fields(self):
        """Verify to_dict() returns a dict with every dataclass field present."""
        signal = VIXTermStructureSignal(
            timestamp="2026-05-15T12:00:00",
            signal_state="risk_on",
            signal_value=0.75,
            vix_spot=15.0,
            vix3m=18.0,
            vix6m=20.0,
            slope_vix3m_vix=1.2,
            regime="contango",
            regime_strength=0.8,
            slope_signal=0.6,
            roll_yield_signal=0.4,
            vix_zscore_signal=0.3,
            curve_shape_signal=0.2,
            spy_shift=0.02,
            gld_shift=-0.01,
            tlt_shift=0.0,
            confidence=90.0,
            is_valid=True,
            reason="Test signal with all fields populated",
        )
        d = signal.to_dict()
        expected_fields = {
            "timestamp", "signal_state", "signal_value",
            "vix_spot", "vix3m", "vix6m", "slope_vix3m_vix",
            "regime", "regime_strength",
            "slope_signal", "roll_yield_signal", "vix_zscore_signal", "curve_shape_signal",
            "spy_shift", "gld_shift", "tlt_shift",
            "confidence", "is_valid", "reason",
        }
        assert set(d.keys()) == expected_fields, f"Missing fields: {expected_fields - set(d.keys())}"
        assert d["signal_value"] == 0.75
        assert d["is_valid"] is True
        assert d["vix_spot"] == 15.0
        assert d["spy_shift"] == 0.02

    def test_to_signal_snapshot_conversion(self):
        """Verify to_signal_snapshot() produces a correctly mapped SignalSnapshot."""
        signal = VIXTermStructureSignal(
            timestamp="2026-05-15T12:00:00",
            signal_state="risk_off",
            signal_value=-0.8,
            vix_spot=30.0,
            vix3m=25.0,
            vix6m=23.0,
            slope_vix3m_vix=0.833,
            regime="extreme_backwardation",
            regime_strength=0.9,
            slope_signal=-1.0,
            roll_yield_signal=-0.5,
            vix_zscore_signal=-0.6,
            curve_shape_signal=-0.3,
            spy_shift=-0.10,
            gld_shift=0.05,
            tlt_shift=0.05,
            confidence=85.0,
            is_valid=True,
            reason="Test conversion to snapshot",
        )
        snapshot = signal.to_signal_snapshot()
        assert snapshot.source == "vix_term_structure"
        assert snapshot.timestamp == "2026-05-15T12:00:00"
        assert snapshot.value == -0.8
        assert snapshot.confidence == 0.85
        assert snapshot.asset_signals == {"SPY": -0.10, "GLD": 0.05, "TLT": 0.05}
        assert snapshot.is_active is True
        assert "VIX TS" in snapshot.explanation
        assert snapshot.explanation.startswith("VIX TS:")

    def test_to_signal_snapshot_metadata(self):
        """Verify all metadata fields are correctly populated in the snapshot."""
        signal = VIXTermStructureSignal(
            timestamp="2026-05-15T12:00:00",
            signal_state="risk_on",
            signal_value=0.5,
            vix_spot=14.0,
            vix3m=16.0,
            vix6m=18.0,
            slope_vix3m_vix=1.142857,
            regime="contango",
            regime_strength=0.5,
            slope_signal=0.4,
            roll_yield_signal=0.3,
            vix_zscore_signal=0.2,
            curve_shape_signal=0.1,
            spy_shift=0.02,
            gld_shift=-0.01,
            tlt_shift=-0.01,
            confidence=80.0,
            is_valid=True,
            reason="Metadata test",
        )
        snapshot = signal.to_signal_snapshot()
        meta = snapshot.metadata
        assert meta["signal_state"] == "risk_on"
        assert meta["regime"] == "contango"
        assert meta["regime_strength"] == 0.5
        assert meta["vix_spot"] == 14.0
        assert meta["slope_signal"] == 0.4
        assert meta["roll_yield_signal"] == 0.3
        assert snapshot.regime_fit == "all"

    # ------------------------------------------------------------------ #
    #  VIXTermStructureSignal dataclass edge cases
    # ------------------------------------------------------------------ #

    def test_signal_with_none_vix_fields(self):
        """Test signal dataclass with None vix3m/vix6m fields."""
        signal = VIXTermStructureSignal(
            timestamp="2026-05-15T12:00:00",
            signal_state="neutral",
            signal_value=0.0,
            vix_spot=0.0,
            vix3m=None,
            vix6m=None,
            slope_vix3m_vix=1.0,
            regime="unknown",
            regime_strength=0.0,
            slope_signal=0.0,
            roll_yield_signal=0.0,
            vix_zscore_signal=0.0,
            curve_shape_signal=0.0,
            spy_shift=0.0,
            gld_shift=0.0,
            tlt_shift=0.0,
            confidence=0.0,
            is_valid=False,
            reason="No data",
        )
        d = signal.to_dict()
        assert d["vix3m"] is None
        assert d["vix6m"] is None
        assert d["vix_spot"] == 0.0
        assert d["is_valid"] is False

        # Snapshot should handle None gracefully
        snapshot = signal.to_signal_snapshot()
        assert snapshot.value == 0.0
        assert snapshot.is_active is False

    def test_signal_extreme_values(self):
        """Test signal dataclass with extreme boundary values."""
        signal = VIXTermStructureSignal(
            timestamp="",
            signal_state="risk_off",
            signal_value=-1.0,
            vix_spot=100.0,
            vix3m=50.0,
            vix6m=40.0,
            slope_vix3m_vix=0.5,
            regime="extreme_backwardation",
            regime_strength=1.0,
            slope_signal=-1.0,
            roll_yield_signal=-1.0,
            vix_zscore_signal=-1.0,
            curve_shape_signal=-1.0,
            spy_shift=-0.10,
            gld_shift=0.05,
            tlt_shift=0.05,
            confidence=100.0,
            is_valid=True,
            reason="",
        )
        d = signal.to_dict()
        assert d["timestamp"] == ""
        assert d["signal_value"] == -1.0
        assert d["vix_spot"] == 100.0
        assert d["confidence"] == 100.0
        assert d["reason"] == ""

    # ------------------------------------------------------------------ #
    #  Calculator edge cases – slope signal
    # ------------------------------------------------------------------ #

    def test_calculate_slope_signal_exact_boundaries(self):
        """Test slope signal at exact threshold boundaries."""
        calc = VIXTermStructureCalculator()

        # Exactly at 0.85 boundary: should be -1.0 from backwardation branch
        # (slope < 0.85 is false, so we hit the 0.85-1.0 branch)
        signal = calc.calculate_slope_signal(vix=100.0, vix3m=85.0)
        assert signal == -1.0, f"Expected -1.0 at slope 0.85, got {signal}"

        # Exactly at 1.0 boundary
        signal = calc.calculate_slope_signal(vix=50.0, vix3m=50.0)
        assert signal == 0.0, f"Expected 0.0 at slope 1.0, got {signal}"

        # Exactly at 1.15 boundary
        # slope=1.15: from the 1.0-1.15 branch, (1.15-1.0)/0.15*0.5 = 0.5
        signal = calc.calculate_slope_signal(vix=100.0, vix3m=115.0)
        assert signal == 0.5, f"Expected 0.5 at slope 1.15, got {signal}"

    def test_calculate_slope_signal_zero_vix3m(self):
        """Test slope signal with vix3m=0 and valid vix (guards against div-by-zero)."""
        calc = VIXTermStructureCalculator()
        signal = calc.calculate_slope_signal(vix=20.0, vix3m=0.0)
        assert signal == 0.0

    def test_calculate_slope_signal_negative_vix(self):
        """Test slope signal with negative VIX values."""
        calc = VIXTermStructureCalculator()
        # vix < 0 should hit the vix <= 0 guard
        signal = calc.calculate_slope_signal(vix=-5.0, vix3m=20.0)
        assert signal == 0.0

        # vix3m < 0 should hit the vix3m <= 0 guard
        signal = calc.calculate_slope_signal(vix=20.0, vix3m=-5.0)
        assert signal == 0.0

    # ------------------------------------------------------------------ #
    #  Calculator edge cases – roll yield
    # ------------------------------------------------------------------ #

    def test_calculate_roll_yield_signal_extreme_caps(self):
        """Test roll yield signal saturates at -1.0 and +1.0."""
        calc = VIXTermStructureCalculator()

        # Extreme backwardation: (20-50)/20 = -1.5, *5 = -7.5, capped at -1.0
        signal = calc.calculate_roll_yield_signal(vix=50.0, vix3m=20.0)
        assert signal == -1.0

        # Extreme contango: (20-10)/20 = 0.5, *5 = 2.5, capped at 1.0
        signal = calc.calculate_roll_yield_signal(vix=10.0, vix3m=20.0)
        assert signal == 1.0

    def test_calculate_roll_yield_signal_negative_vix3m(self):
        """Test roll yield with negative vix3m (should return 0.0)."""
        calc = VIXTermStructureCalculator()
        signal = calc.calculate_roll_yield_signal(vix=20.0, vix3m=-5.0)
        assert signal == 0.0

    def test_calculate_roll_yield_signal_equal_values(self):
        """Test roll yield when VIX equals VIX3M (flat curve)."""
        calc = VIXTermStructureCalculator()
        signal = calc.calculate_roll_yield_signal(vix=20.0, vix3m=20.0)
        # (20-20)/20 = 0, *5 = 0
        assert signal == 0.0

    # ------------------------------------------------------------------ #
    #  Calculator edge cases – VIX Z-score
    # ------------------------------------------------------------------ #

    def test_calculate_vix_zscore_signal_constant_values(self):
        """Test Z-score with uniform history (std=0) returns 0.0."""
        calc = VIXTermStructureCalculator()
        for i in range(60):
            calc.add_vix_reading(f"2026-01-{i+1:03d}", 20.0)

        # All values are 20.0, so std=0
        signal = calc.calculate_vix_zscore_signal(vix=20.0)
        assert signal == 0.0

    def test_calculate_vix_zscore_signal_just_below_boundary(self):
        """Test Z-score with 59 days (just below 60-day minimum) returns 0.0."""
        calc = VIXTermStructureCalculator()
        for i in range(59):
            calc.add_vix_reading(f"2026-01-{i+1:03d}", 18.0 + (i % 5))
        signal = calc.calculate_vix_zscore_signal(vix=25.0)
        assert signal == 0.0, "Expected 0.0 with 59 days of history (below 60-day threshold)"

    def test_calculate_vix_zscore_signal_inverted_sign(self):
        """Verify Z-score signal is inverted: high VIX -> negative signal, low VIX -> positive."""
        calc = VIXTermStructureCalculator()
        for i in range(100):
            calc.add_vix_reading(f"2026-01-{i+1:03d}", 20.0 + (i % 10))

        # Very high VIX should produce negative (risk-off) signal
        high_signal = calc.calculate_vix_zscore_signal(vix=50.0)
        assert high_signal < 0, f"Expected negative for high VIX, got {high_signal}"

        # Very low VIX should produce positive (risk-on) signal
        low_signal = calc.calculate_vix_zscore_signal(vix=10.0)
        assert low_signal > 0, f"Expected positive for low VIX, got {low_signal}"

    # ------------------------------------------------------------------ #
    #  Calculator edge cases – curve shape
    # ------------------------------------------------------------------ #

    def test_calculate_curve_shape_signal_zero_vix3m(self):
        """Test curve shape with vix3m=0 returns 0.0."""
        calc = VIXTermStructureCalculator()
        signal = calc.calculate_curve_shape_signal(vix3m=0.0, vix6m=20.0)
        assert signal == 0.0

    def test_calculate_curve_shape_signal_vix3m_negative(self):
        """Test curve shape with negative vix3m returns 0.0."""
        calc = VIXTermStructureCalculator()
        signal = calc.calculate_curve_shape_signal(vix3m=-5.0, vix6m=10.0)
        assert signal == 0.0

    def test_calculate_curve_shape_signal_caps(self):
        """Test curve shape signal saturation at -1.0 and +1.0."""
        calc = VIXTermStructureCalculator()

        # Steep curve: vix6m/vix3m = 1.2, (1.2-1)*10 = 2.0, capped at 1.0
        signal = calc.calculate_curve_shape_signal(vix3m=10.0, vix6m=12.0)
        assert signal == 1.0

        # Inverse curve: vix6m/vix3m = 0.8, (0.8-1)*10 = -2.0, capped at -1.0
        signal = calc.calculate_curve_shape_signal(vix3m=10.0, vix6m=8.0)
        assert signal == -1.0

    # ------------------------------------------------------------------ #
    #  Calculator edge cases – regime classification
    # ------------------------------------------------------------------ #

    def test_classify_regime_exact_thresholds(self):
        """Test regime classification at exact threshold boundaries."""
        calc = VIXTermStructureCalculator()

        # Exactly at EXTREME_CONTANGO threshold (1.15) -> CONTANGO branch
        regime, strength = calc.classify_regime(1.15)
        assert regime == VIXRegime.EXTREME_CONTANGO, f"Expected EXTREME_CONTANGO at 1.15, got {regime}"

        # Exactly at CONTANGO threshold (1.0) -> CONTANGO branch
        regime, strength = calc.classify_regime(1.0)
        assert regime == VIXRegime.CONTANGO, f"Expected CONTANGO at 1.0, got {regime}"

        # Exactly at FLAT_LOWER (0.95) -> FLAT branch
        regime, strength = calc.classify_regime(0.95)
        assert regime == VIXRegime.FLAT, f"Expected FLAT at 0.95, got {regime}"

        # Exactly at BACKWARDATION threshold (0.8) -> BACKWARDATION branch
        regime, strength = calc.classify_regime(0.8)
        assert regime == VIXRegime.BACKWARDATION, f"Expected BACKWARDATION at 0.8, got {regime}"

    def test_classify_regime_strength_bounds(self):
        """Verify regime strength is always in [0, 1] across all regimes."""
        calc = VIXTermStructureCalculator()
        test_slopes = [1.30, 1.15, 1.10, 1.00, 0.97, 0.95, 0.88, 0.80, 0.70, 0.50]

        for slope in test_slopes:
            regime, strength = calc.classify_regime(slope)
            assert 0.0 <= strength <= 1.0 + 1e-10, (
                f"Strength {strength:.6f} out of bounds for slope={slope}, regime={regime}"
            )

    # ------------------------------------------------------------------ #
    #  Calculator edge cases – allocation shifts
    # ------------------------------------------------------------------ #

    def test_get_allocation_shifts_boundary_values(self):
        """Test allocation shifts at exact boundary values."""
        calc = VIXTermStructureCalculator()

        # Exactly at +0.7 boundary -> complacent branch (>= 0.7)
        shifts = calc.get_allocation_shifts(0.7)
        assert shifts["spy"] == 0.05
        assert shifts["gld"] == -0.03
        assert shifts["tlt"] == -0.02

        # Exactly at +0.3 boundary -> second branch (>= 0.3)
        shifts = calc.get_allocation_shifts(0.3)
        assert shifts["spy"] == 0.02
        assert shifts["gld"] == -0.01
        assert shifts["tlt"] == -0.01

        # Exactly at -0.3 boundary -> neutral branch (>= -0.3)
        shifts = calc.get_allocation_shifts(-0.3)
        assert shifts["spy"] == 0.0

        # Exactly at -0.7 boundary -> caution branch (>= -0.7)
        shifts = calc.get_allocation_shifts(-0.7)
        assert shifts["spy"] == -0.05
        assert shifts["gld"] == 0.03
        assert shifts["tlt"] == 0.02

    # ------------------------------------------------------------------ #
    #  Constants validation
    # ------------------------------------------------------------------ #

    def test_calculator_constants(self):
        """Verify all threshold constants are present and have expected values."""
        assert VIXTermStructureCalculator.EXTREME_CONTANGO_THRESHOLD == 1.15
        assert VIXTermStructureCalculator.CONTANGO_THRESHOLD == 1.00
        assert VIXTermStructureCalculator.FLAT_UPPER == 1.00
        assert VIXTermStructureCalculator.FLAT_LOWER == 0.95
        assert VIXTermStructureCalculator.BACKWARDATION_THRESHOLD == 0.80
        assert VIXTermStructureCalculator.VIX_CHEAP == 16.0
        assert VIXTermStructureCalculator.VIX_FAIR == 20.0
        assert VIXTermStructureCalculator.VIX_EXPENSIVE == 25.0

    # ------------------------------------------------------------------ #
    #  Calculator edge cases – composite signal
    # ------------------------------------------------------------------ #

    def test_calculate_composite_signal_zero_vix(self):
        """Test composite signal with zero VIX (should not divide by zero)."""
        calc = VIXTermStructureCalculator()
        components = calc.calculate_composite_signal(
            vix=0.0, vix3m=20.0, vix6m=22.0, date="2026-05-15"
        )
        # slope = vix3m/vix would be division by zero; guarded by vix > 0 check
        # calculate_slope_signal returns 0.0 for vix <= 0
        assert "composite" in components
        assert -1.0 <= components["composite"] <= 1.0
        assert components["slope"] == 1.0  # fallback when vix <= 0

    def test_calculate_composite_signal_no_history(self):
        """Test composite signal with no Z-score history (Z-score returns 0)."""
        calc = VIXTermStructureCalculator()
        components = calc.calculate_composite_signal(
            vix=20.0, vix3m=22.0, vix6m=24.0, date="2026-05-15"
        )
        # Z-score should be 0 (no history)
        assert components["vix_zscore_signal"] == 0.0
        assert -1.0 <= components["composite"] <= 1.0
        assert components["composite"] > 0  # Contango should give positive composite

    def test_calculate_composite_signal_all_negative(self):
        """Test composite signal in extreme backwardation (all components negative)."""
        calc = VIXTermStructureCalculator()
        for i in range(100):
            calc.add_vix_reading(f"2026-01-{i+1:03d}", 20.0 + (i % 5))
        components = calc.calculate_composite_signal(
            vix=40.0, vix3m=30.0, vix6m=28.0, date="2026-05-15"
        )
        # Extreme backwardation: all component signals should be negative
        assert components["slope_signal"] < 0
        assert components["roll_yield_signal"] < 0
        assert components["vix_zscore_signal"] < 0  # high VIX, negative
        assert components["curve_shape_signal"] < 0  # inverted curve
        assert components["composite"] < 0

    # ------------------------------------------------------------------ #
    #  Generator edge cases
    # ------------------------------------------------------------------ #

    @patch("src.signals.vix_term_structure.VIXTermStructureSignalGenerator.load_vix_data")
    def test_generate_signal_date_none(self, mock_load_data):
        """Test generate_signal() with date=None uses latest resolved levels."""
        mock_load_data.return_value = {
            "2026-05-12": {
                "date": "2026-05-12",
                "vix_spot": 25.0,
                "front_month": 22.0,
                "third_month": 21.0,
            }
        }

        generator = VIXTermStructureSignalGenerator()
        # Live path also consults market.db; pin it off so load_vix_data mock wins
        with patch.object(
            generator, "fetch_levels_from_market_db", return_value=None
        ):
            signal = generator.generate_signal(date=None)

        assert signal.is_valid
        # Latest mock data is 2026-05-12 which has backwardation (vix=25, vix3m=22)
        assert signal.vix_spot == 25.0
        assert signal.signal_value < 0  # Backwardation -> negative signal

    @patch("src.signals.vix_term_structure.VIXTermStructureSignalGenerator.load_vix_data")
    def test_get_signal_history_empty(self, mock_load_data):
        """Test get_signal_history with no data returns empty list."""
        mock_load_data.return_value = {}

        generator = VIXTermStructureSignalGenerator()
        signals = generator.get_signal_history(days=30)

        assert signals == []

    @patch("src.signals.vix_term_structure.VIXTermStructureSignalGenerator.load_vix_data")
    def test_fetch_current_vix_empty_data(self, mock_load_data, tmp_path):
        """Test fetch_current_vix with no data returns None."""
        mock_load_data.return_value = {}

        generator = VIXTermStructureSignalGenerator(db_path=tmp_path / "missing.db")
        result = generator.fetch_current_vix()

        assert result is None

    @patch("src.signals.vix_term_structure.VIXTermStructureSignalGenerator.load_vix_data")
    def test_generate_signal_partial_data_vix3m_only(self, mock_load_data):
        """Test signal generation when only vix_spot and VIX3M are available (no VIX6M)."""
        mock_load_data.return_value = {
            "2026-05-15": {
                "date": "2026-05-15",
                "vix_spot": 18.0,
                "front_month": 20.0,
            }
        }

        generator = VIXTermStructureSignalGenerator()
        signal = generator.generate_signal("2026-05-15")

        assert signal.is_valid
        assert signal.vix3m == 20.0
        assert signal.vix6m is None
        # Confidence: base 50 + 30 (vix3m) = 80 (no vix6m, no history)
        assert signal.confidence == 80.0

    @patch("src.signals.vix_term_structure.VIXTermStructureSignalGenerator.load_vix_data")
    def test_generate_signal_vix_spot_only(self, mock_load_data):
        """Test signal generation when only vix_spot is available (no front_month or third_month)."""
        mock_load_data.return_value = {
            "2026-05-15": {
                "date": "2026-05-15",
                "vix_spot": 18.0,
            }
        }

        generator = VIXTermStructureSignalGenerator()
        signal = generator.generate_signal("2026-05-15")

        assert signal.is_valid
        # VIX3M is None -> uses VIX spot fallback in calculate_composite_signal
        assert signal.vix3m is None
        assert signal.vix6m is None
        # Confidence: base 50 only (no vix3m, no vix6m, no history)
        assert signal.confidence == 50.0


# ===================================================================== #
#  New test classes for expanded coverage (dataclass validation,        #
#  NaN/Inf, boundary conditions, CLI, exports, generator edge cases)    #
# ===================================================================== #


class TestDataclassFieldValidation:
    """Verify all dataclass fields via dataclasses.fields()."""

    def test_vix_term_structure_signal_field_count(self):
        """Verify VIXTermStructureSignal has exactly 20 fields."""
        fields = dataclasses.fields(VIXTermStructureSignal)
        assert len(fields) == 19, f"Expected 19 fields, got {len(fields)}"

    def test_vix_term_structure_signal_field_names(self):
        """Verify VIXTermStructureSignal field names match the source dataclass."""
        fields = dataclasses.fields(VIXTermStructureSignal)
        field_names = {f.name for f in fields}
        expected = {
            "timestamp",
            "signal_state",
            "signal_value",
            "vix_spot",
            "vix3m",
            "vix6m",
            "slope_vix3m_vix",
            "regime",
            "regime_strength",
            "slope_signal",
            "roll_yield_signal",
            "vix_zscore_signal",
            "curve_shape_signal",
            "spy_shift",
            "gld_shift",
            "tlt_shift",
            "confidence",
            "is_valid",
            "reason",
        }
        extra = field_names - expected
        missing = expected - field_names
        assert field_names == expected, f"Extra fields: {extra}, Missing fields: {missing}"

    def test_vix_term_structure_signal_field_types(self):
        """Verify VIXTermStructureSignal field type annotations."""
        fields = {f.name: f.type for f in dataclasses.fields(VIXTermStructureSignal)}
        # String fields
        assert fields["timestamp"] is str
        assert fields["signal_state"] is str
        assert fields["regime"] is str
        assert fields["reason"] is str
        # Float fields
        assert fields["signal_value"] is float
        assert fields["vix_spot"] is float
        assert fields["slope_vix3m_vix"] is float
        assert fields["regime_strength"] is float
        assert fields["slope_signal"] is float
        assert fields["roll_yield_signal"] is float
        assert fields["vix_zscore_signal"] is float
        assert fields["curve_shape_signal"] is float
        assert fields["spy_shift"] is float
        assert fields["gld_shift"] is float
        assert fields["tlt_shift"] is float
        assert fields["confidence"] is float
        # Bool field
        assert fields["is_valid"] is bool
        # Optional float fields (use == not is — Optional creates new Union objects)
        assert fields["vix3m"] == Optional[float]
        assert fields["vix6m"] == Optional[float]

    def test_vix_term_structure_signal_no_defaults(self):
        """Verify VIXTermStructureSignal has no default values (all required)."""
        for f in dataclasses.fields(VIXTermStructureSignal):
            assert f.default is dataclasses.MISSING, (
                f"Field {f.name} has default={f.default!r} (should not exist)"
            )
            assert f.default_factory is dataclasses.MISSING, (
                f"Field {f.name} has default_factory={f.default_factory!r}"
            )

    def test_vix_regime_enum_member_count(self):
        """Verify VIXRegime has exactly 5 members."""
        assert len(list(VIXRegime)) == 5

    def test_vix_signal_state_enum_member_count(self):
        """Verify VIXSignalState has exactly 3 members."""
        assert len(list(VIXSignalState)) == 3


class TestNaNInfEdgeCases:
    """Test computation methods with NaN and Inf inputs."""

    def test_slope_signal_nan_vix(self):
        """Test calculate_slope_signal with NaN VIX does not crash."""
        calc = VIXTermStructureCalculator()
        signal = calc.calculate_slope_signal(vix=math.nan, vix3m=20.0)
        # NaN arithmetic propagates; the exact result may be NaN or 0
        assert isinstance(signal, float)

    def test_slope_signal_nan_vix3m(self):
        """Test calculate_slope_signal with NaN VIX3M does not crash."""
        calc = VIXTermStructureCalculator()
        signal = calc.calculate_slope_signal(vix=20.0, vix3m=math.nan)
        assert isinstance(signal, float)

    def test_slope_signal_inf_vix(self):
        """Test calculate_slope_signal with +inf VIX (ratio = 0 -> -1.0)."""
        calc = VIXTermStructureCalculator()
        signal = calc.calculate_slope_signal(vix=math.inf, vix3m=20.0)
        # 20 / inf = 0.0, which is < 0.85 -> -1.0
        assert signal == -1.0

    def test_slope_signal_neg_inf_vix(self):
        """Test calculate_slope_signal with -inf VIX (<= 0 guard)."""
        calc = VIXTermStructureCalculator()
        signal = calc.calculate_slope_signal(vix=-math.inf, vix3m=20.0)
        # -inf <= 0 -> guard returns 0.0
        assert signal == 0.0

    def test_slope_signal_inf_vix3m(self):
        """Test calculate_slope_signal with +inf VIX3M (capped at 1.0)."""
        calc = VIXTermStructureCalculator()
        signal = calc.calculate_slope_signal(vix=20.0, vix3m=math.inf)
        assert signal == 1.0

    def test_roll_yield_nan_vix(self):
        """Test calculate_roll_yield_signal with NaN VIX does not crash."""
        calc = VIXTermStructureCalculator()
        signal = calc.calculate_roll_yield_signal(vix=math.nan, vix3m=20.0)
        assert isinstance(signal, float)

    def test_roll_yield_inf_vix3m(self):
        """Test calculate_roll_yield_signal with +inf VIX3M (capped at 1.0)."""
        calc = VIXTermStructureCalculator()
        signal = calc.calculate_roll_yield_signal(vix=20.0, vix3m=math.inf)
        # (inf - 20) / inf = 1.0, * 5 = 5.0, capped at 1.0
        assert signal == 1.0

    def test_roll_yield_neg_inf_vix3m(self):
        """Test calculate_roll_yield_signal with -inf VIX3M (<= 0 guard)."""
        calc = VIXTermStructureCalculator()
        signal = calc.calculate_roll_yield_signal(vix=20.0, vix3m=-math.inf)
        assert signal == 0.0

    def test_vix_zscore_nan_vix(self):
        """Test calculate_vix_zscore_signal with NaN VIX does not crash."""
        calc = VIXTermStructureCalculator()
        for i in range(60):
            calc.add_vix_reading(f"2026-01-{i+1:03d}", 20.0)
        signal = calc.calculate_vix_zscore_signal(vix=math.nan)
        assert isinstance(signal, float)

    def test_vix_zscore_inf_vix(self):
        """Test calculate_vix_zscore_signal with +inf VIX (extreme -> -1.0)."""
        calc = VIXTermStructureCalculator()
        for i in range(60):
            calc.add_vix_reading(f"2026-01-{i+1:03d}", 20.0 + (i % 5))
        signal = calc.calculate_vix_zscore_signal(vix=math.inf)
        # inf zscore -> inverted -> -1.0
        assert signal == -1.0

    def test_curve_shape_nan_vix3m(self):
        """Test calculate_curve_shape_signal with NaN VIX3M does not crash."""
        calc = VIXTermStructureCalculator()
        signal = calc.calculate_curve_shape_signal(vix3m=math.nan, vix6m=20.0)
        assert isinstance(signal, float)

    def test_curve_shape_nan_vix6m(self):
        """Test calculate_curve_shape_signal with NaN VIX6M does not crash."""
        calc = VIXTermStructureCalculator()
        signal = calc.calculate_curve_shape_signal(vix3m=20.0, vix6m=math.nan)
        assert isinstance(signal, float)

    def test_curve_shape_inf_vix6m(self):
        """Test calculate_curve_shape_signal with +inf VIX6M (capped at 1.0)."""
        calc = VIXTermStructureCalculator()
        signal = calc.calculate_curve_shape_signal(vix3m=20.0, vix6m=math.inf)
        # inf / 20 = inf, (inf - 1)*10 = inf, capped at 1.0
        assert signal == 1.0

    def test_classify_regime_nan(self):
        """Test classify_regime with NaN slope does not crash."""
        calc = VIXTermStructureCalculator()
        regime, strength = calc.classify_regime(math.nan)
        # NaN comparisons all False -> falls through to extreme_backwardation
        assert isinstance(regime, VIXRegime)
        assert isinstance(strength, float)

    def test_classify_regime_inf(self):
        """Test classify_regime with +inf slope."""
        calc = VIXTermStructureCalculator()
        regime, strength = calc.classify_regime(math.inf)
        assert regime == VIXRegime.EXTREME_CONTANGO
        assert 0.0 <= strength <= 1.0

    def test_get_allocation_shifts_nan(self):
        """Test get_allocation_shifts with NaN signal does not crash."""
        calc = VIXTermStructureCalculator()
        shifts = calc.get_allocation_shifts(math.nan)
        # NaN comparisons all False -> caution branch (>= -0.7 is False ... wait,
        # NaN >= -0.7 is False, so it falls to the else -> risk-off)
        assert shifts == {"spy": -0.10, "gld": 0.05, "tlt": 0.05}

    def test_get_allocation_shifts_inf(self):
        """Test get_allocation_shifts with +inf signal."""
        calc = VIXTermStructureCalculator()
        shifts = calc.get_allocation_shifts(math.inf)
        # inf >= 0.7 is True -> complacent branch
        assert shifts["spy"] == 0.05
        assert shifts["gld"] == -0.03

    def test_get_allocation_shifts_neg_inf(self):
        """Test get_allocation_shifts with -inf signal."""
        calc = VIXTermStructureCalculator()
        shifts = calc.get_allocation_shifts(-math.inf)
        # -inf >= 0.7 is False, -inf >= -0.3 is False, -inf >= -0.7 is False
        # -> else branch: risk-off
        assert shifts == {"spy": -0.10, "gld": 0.05, "tlt": 0.05}


class TestBoundaryConditions:
    """Function boundary conditions with extreme inputs."""

    def test_slope_signal_extreme_large_ratio(self):
        """Test slope signal with VIX3M/VIX ratio of 3.0 (capped at +1.0)."""
        calc = VIXTermStructureCalculator()
        signal = calc.calculate_slope_signal(vix=10.0, vix3m=30.0)
        assert signal == 1.0

    def test_slope_signal_extreme_small_ratio(self):
        """Test slope signal with VIX3M/VIX ratio of 0.1 (capped at -1.0)."""
        calc = VIXTermStructureCalculator()
        signal = calc.calculate_slope_signal(vix=100.0, vix3m=10.0)
        assert signal == -1.0

    def test_slope_signal_ratio_just_above_1(self):
        """Test slope signal with ratio just above 1.0 (1.001)."""
        calc = VIXTermStructureCalculator()
        signal = calc.calculate_slope_signal(vix=100.0, vix3m=100.1)
        assert 0.0 < signal < 0.1

    def test_slope_signal_ratio_just_below_1(self):
        """Test slope signal with ratio just below 1.0 (0.999)."""
        calc = VIXTermStructureCalculator()
        signal = calc.calculate_slope_signal(vix=100.0, vix3m=99.9)
        # slope 0.999 < 1.0 => -1.0 + (0.999 - 0.85) / 0.15 * 0.5 ≈ -0.503
        assert -0.6 < signal < -0.4

    def test_roll_yield_equal_large_values(self):
        """Test roll yield with equal very large values (should be 0)."""
        calc = VIXTermStructureCalculator()
        signal = calc.calculate_roll_yield_signal(vix=1e10, vix3m=1e10)
        assert signal == 0.0

    def test_roll_yield_vix_near_zero_positive(self):
        """Test roll yield with VIX near zero but vix3m positive."""
        calc = VIXTermStructureCalculator()
        signal = calc.calculate_roll_yield_signal(vix=1e-10, vix3m=20.0)
        # (20 - 1e-10) / 20 ≈ 1.0, * 5 = 5.0, capped at 1.0
        assert signal == 1.0

    def test_vix_zscore_single_element_history(self):
        """Test Z-score with single element history (below 60-day threshold)."""
        calc = VIXTermStructureCalculator()
        calc.add_vix_reading("2026-01-01", 20.0)
        signal = calc.calculate_vix_zscore_signal(vix=25.0)
        assert signal == 0.0

    def test_vix_zscore_exactly_60_days(self):
        """Test Z-score with exactly 60 days of history (boundary)."""
        calc = VIXTermStructureCalculator()
        for i in range(60):
            calc.add_vix_reading(f"2026-01-{i+1:03d}", 20.0 + (i % 5))
        signal = calc.calculate_vix_zscore_signal(vix=25.0)
        assert signal != 0.0  # 60 days is enough for non-zero

    def test_vix_zscore_very_large_history(self):
        """Test Z-score with 1000 days of history."""
        calc = VIXTermStructureCalculator()
        for i in range(1000):
            calc.add_vix_reading(f"2026-{i // 30 + 1:02d}-{i % 28 + 1:02d}", 18.0 + (i % 5))
        signal = calc.calculate_vix_zscore_signal(vix=25.0)
        assert -1.0 <= signal <= 1.0

    def test_vix_zscore_negative_zscore_gives_positive_signal(self):
        """Verify negative Z-score (VIX below mean) gives positive (risk-on) signal."""
        calc = VIXTermStructureCalculator()
        for i in range(100):
            calc.add_vix_reading(f"2026-01-{i+1:03d}", 30.0 + (i % 5))
        signal = calc.calculate_vix_zscore_signal(vix=20.0)
        assert signal > 0.0

    def test_classify_regime_zero_slope(self):
        """Test classify_regime with slope of 0.0 (extreme backwardation)."""
        calc = VIXTermStructureCalculator()
        regime, strength = calc.classify_regime(0.0)
        assert regime == VIXRegime.EXTREME_BACKWARDATION
        assert 0.0 <= strength <= 1.0

    def test_classify_regime_negative_slope(self):
        """Test classify_regime with negative slope."""
        calc = VIXTermStructureCalculator()
        regime, strength = calc.classify_regime(-0.5)
        assert regime == VIXRegime.EXTREME_BACKWARDATION
        assert 0.0 <= strength <= 1.0

    def test_classify_regime_extreme_high_slope(self):
        """Test classify_regime with very high slope (5.0)."""
        calc = VIXTermStructureCalculator()
        regime, strength = calc.classify_regime(5.0)
        assert regime == VIXRegime.EXTREME_CONTANGO
        assert 0.0 <= strength <= 1.0

    def test_get_allocation_shifts_above_1_saturates(self):
        """Test get_allocation_shifts with signal > 1.0 (saturates at complacent)."""
        calc = VIXTermStructureCalculator()
        shifts = calc.get_allocation_shifts(1.5)
        assert shifts["spy"] == 0.05  # Same as >= 0.7 branch

    def test_get_allocation_shifts_below_neg1_saturates(self):
        """Test get_allocation_shifts with signal < -1.0 (saturates at risk-off)."""
        calc = VIXTermStructureCalculator()
        shifts = calc.get_allocation_shifts(-1.5)
        assert shifts == {"spy": -0.10, "gld": 0.05, "tlt": 0.05}

    def test_calculator_custom_history_days(self):
        """Test VIXTermStructureCalculator with custom history_days=5."""
        calc = VIXTermStructureCalculator(history_days=5)
        assert calc.history_days == 5
        for i in range(10):
            calc.add_vix_reading(f"2026-01-{i+1:02d}", float(i))
        assert len(calc.vix_history) == 5

    def test_add_vix_reading_empty_date(self):
        """Test add_vix_reading with empty string date."""
        calc = VIXTermStructureCalculator()
        calc.add_vix_reading("", 20.0)
        assert len(calc.vix_history) == 1
        assert calc.vix_history[0] == ("", 20.0)

    def test_add_vix_reading_zero_vix(self):
        """Test add_vix_reading with zero VIX value."""
        calc = VIXTermStructureCalculator()
        calc.add_vix_reading("2026-01-01", 0.0)
        assert calc.vix_history[0] == ("2026-01-01", 0.0)

    def test_add_vix_reading_negative_vix(self):
        """Test add_vix_reading with negative VIX value."""
        calc = VIXTermStructureCalculator()
        calc.add_vix_reading("2026-01-01", -5.0)
        assert calc.vix_history[0] == ("2026-01-01", -5.0)


class TestCLIEntryPoint:
    """Test CLI entry points with capsys."""

    @patch.object(VIXTermStructureSignalGenerator, "generate_signal")
    @patch.object(VIXTermStructureSignalGenerator, "save_signal")
    def test_main_runs_and_returns_signal(self, mock_save, mock_gen):
        """Test main() returns a VIXTermStructureSignal."""
        mock_signal = MagicMock(spec=VIXTermStructureSignal)
        mock_signal.timestamp = "2026-05-15T12:00:00"
        mock_signal.signal_state = "neutral"
        mock_signal.signal_value = 0.0
        mock_signal.vix_spot = 20.0
        mock_signal.vix3m = 22.0
        mock_signal.vix6m = 24.0
        mock_signal.slope_vix3m_vix = 1.1
        mock_signal.regime = "contango"
        mock_signal.regime_strength = 0.5
        mock_signal.slope_signal = 0.3
        mock_signal.roll_yield_signal = 0.2
        mock_signal.vix_zscore_signal = 0.1
        mock_signal.curve_shape_signal = 0.15
        mock_signal.spy_shift = 0.0
        mock_signal.gld_shift = 0.0
        mock_signal.tlt_shift = 0.0
        mock_signal.confidence = 85.0
        mock_signal.is_valid = True
        mock_signal.reason = "Test signal"
        mock_gen.return_value = mock_signal

        result = main()
        assert result is mock_signal
        mock_save.assert_called_once_with(mock_signal)

    @patch.object(VIXTermStructureSignalGenerator, "generate_signal")
    @patch.object(VIXTermStructureSignalGenerator, "save_signal")
    def test_main_output_contains_key_fields(self, mock_save, mock_gen, caplog):
        """Test main() prints expected fields to stdout."""
        caplog.set_level(logging.INFO)
        mock_signal = MagicMock(spec=VIXTermStructureSignal)
        mock_signal.timestamp = "2026-05-15T12:00:00"
        mock_signal.signal_state = "risk_off"
        mock_signal.signal_value = -0.75
        mock_signal.vix_spot = 35.0
        mock_signal.vix3m = 28.0
        mock_signal.vix6m = 25.0
        mock_signal.slope_vix3m_vix = 0.8
        mock_signal.regime = "backwardation"
        mock_signal.regime_strength = 0.8
        mock_signal.slope_signal = -0.8
        mock_signal.roll_yield_signal = -0.5
        mock_signal.vix_zscore_signal = -0.6
        mock_signal.curve_shape_signal = -0.4
        mock_signal.spy_shift = -0.10
        mock_signal.gld_shift = 0.05
        mock_signal.tlt_shift = 0.05
        mock_signal.confidence = 90.0
        mock_signal.is_valid = True
        mock_signal.reason = "Backwardation detected"
        mock_gen.return_value = mock_signal

        main()

        assert "VIX TERM STRUCTURE SIGNAL GENERATOR" in caplog.text
        assert "Signal State: risk_off" in caplog.text
        assert "Signal Value: -0.750" in caplog.text
        assert "VIX Spot: 35.00" in caplog.text
        assert "Regime: backwardation" in caplog.text
        assert "Confidence: 90%" in caplog.text
        assert "Valid: True" in caplog.text
        assert "Reason: Backwardation detected" in caplog.text

    @patch.object(VIXTermStructureSignalGenerator, "generate_signal")
    @patch.object(VIXTermStructureSignalGenerator, "save_signal")
    def test_main_output_with_invalid_signal(self, mock_save, mock_gen, caplog):
        """Test main() output when signal is invalid."""
        caplog.set_level(logging.INFO)
        mock_signal = MagicMock(spec=VIXTermStructureSignal)
        mock_signal.timestamp = "2026-05-15T12:00:00"
        mock_signal.signal_state = "neutral"
        mock_signal.signal_value = 0.0
        mock_signal.vix_spot = 0.0
        mock_signal.vix3m = None
        mock_signal.vix6m = None
        mock_signal.slope_vix3m_vix = 1.0
        mock_signal.regime = "unknown"
        mock_signal.regime_strength = 0.0
        mock_signal.slope_signal = 0.0
        mock_signal.roll_yield_signal = 0.0
        mock_signal.vix_zscore_signal = 0.0
        mock_signal.curve_shape_signal = 0.0
        mock_signal.spy_shift = 0.0
        mock_signal.gld_shift = 0.0
        mock_signal.tlt_shift = 0.0
        mock_signal.confidence = 0.0
        mock_signal.is_valid = False
        mock_signal.reason = "No VIX data available"
        mock_gen.return_value = mock_signal

        main()
        assert "Valid: False" in caplog.text
        assert "No VIX data available" in caplog.text

    def test_cli_guard_present(self):
        """Test that __name__ == '__main__' guard exists in source."""
        import src.signals.vix_term_structure as module
        source = inspect.getsource(module)
        assert "if __name__ == '__main__':" in source


class TestExportCompleteness:
    """Verify __all__ coverage in vix_term_structure module."""

    def test_all_exports_exist(self):
        """Verify every name in __all__ actually exists in the module."""
        import src.signals.vix_term_structure as module
        for name in module.__all__:
            assert hasattr(module, name), (
                f"'{name}' declared in __all__ but not defined in module"
            )

    def test_all_exports_are_types(self):
        """Verify all __all__ members are classes (no plain functions exported)."""
        import src.signals.vix_term_structure as module
        for name in module.__all__:
            obj = getattr(module, name)
            assert isinstance(obj, type), f"'{name}' is {type(obj).__name__}, not a class/type"

    def test_all_count(self):
        """Verify __all__ contains exactly 5 exports."""
        import src.signals.vix_term_structure as module
        assert len(module.__all__) == 5


class TestGeneratorEdgeCases:
    """Additional generator edge case tests."""

    @patch.object(VIXTermStructureSignalGenerator, "load_vix_data")
    def test_generate_signal_nonexistent_date_uses_latest(self, mock_load_data, tmp_path):
        """Test generate_signal with a date not in data fetches current."""
        mock_load_data.return_value = {
            "2026-05-10": {"vix_spot": 18.0, "front_month": 20.0, "third_month": 22.0},
            "2026-05-11": {"vix_spot": 19.0, "front_month": 21.0, "third_month": 23.0},
        }
        generator = VIXTermStructureSignalGenerator(db_path=tmp_path / "missing.db")
        # Date '2026-05-15' not in data -> falls back to latest file row
        signal = generator.generate_signal("2026-05-15")
        assert signal.is_valid
        assert signal.vix_spot == 19.0  # latest is 2026-05-11

    @patch.object(VIXTermStructureSignalGenerator, "load_vix_data")
    def test_generate_signal_missing_vix_spot_key(self, mock_load_data, tmp_path):
        """Test generate_signal when vix_spot key is missing (defaults to 0)."""
        mock_load_data.return_value = {
            "2026-05-15": {
                "date": "2026-05-15",
                "front_month": 20.0,
                "third_month": 22.0,
            }
        }
        generator = VIXTermStructureSignalGenerator(db_path=tmp_path / "missing.db")
        signal = generator.generate_signal("2026-05-15")
        assert signal.is_valid
        assert signal.vix_spot == 0.0  # default from .get("vix_spot", 0)
        assert signal.vix3m == 20.0

    @patch.object(VIXTermStructureSignalGenerator, "load_vix_data")
    def test_generate_signal_string_vix_spot_raises_type_error(self, mock_load_data):
        """Test generate_signal with string vix_spot raises TypeError (not guarded)."""
        mock_load_data.return_value = {
            "2026-05-15": {
                "date": "2026-05-15",
                "vix_spot": "high",
                "front_month": 20.0,
                "third_month": 22.0,
            }
        }
        generator = VIXTermStructureSignalGenerator()
        with pytest.raises(TypeError):
            generator.generate_signal("2026-05-15")

    @patch.object(VIXTermStructureSignalGenerator, "load_vix_data")
    def test_fetch_current_vix_returns_latest(self, mock_load_data, tmp_path):
        """Test fetch_current_vix returns latest entry from data when DB absent."""
        mock_load_data.return_value = {
            "2026-05-10": {"vix_spot": 18.0},
            "2026-05-11": {"vix_spot": 19.0},
            "2026-05-12": {"vix_spot": 20.0},
        }
        generator = VIXTermStructureSignalGenerator(db_path=tmp_path / "missing.db")
        result = generator.fetch_current_vix()
        assert result is not None
        assert result.get("vix_spot") == 20.0

    @patch.object(VIXTermStructureSignalGenerator, "load_vix_data")
    def test_save_signal_exception_handling(self, mock_load_data):
        """Test save_signal handles file write exception gracefully."""
        generator = VIXTermStructureSignalGenerator()
        signal = VIXTermStructureSignal(
            timestamp="2026-05-15T12:00:00",
            signal_state="neutral",
            signal_value=0.0,
            vix_spot=0.0,
            vix3m=None,
            vix6m=None,
            slope_vix3m_vix=1.0,
            regime="unknown",
            regime_strength=0.0,
            slope_signal=0.0,
            roll_yield_signal=0.0,
            vix_zscore_signal=0.0,
            curve_shape_signal=0.0,
            spy_shift=0.0,
            gld_shift=0.0,
            tlt_shift=0.0,
            confidence=0.0,
            is_valid=False,
            reason="test save exception",
        )
        with patch("builtins.open", side_effect=OSError("Permission denied")):
            generator.save_signal(signal)  # Should not raise

    def test_load_vix_data_file_not_found(self):
        """Test load_vix_data when file does not exist returns empty dict."""
        generator = VIXTermStructureSignalGenerator()
        generator.VIX_DATA_PATH = Path("/nonexistent/path/should/not/exist.json")
        result = generator.load_vix_data()
        assert result == {}

    def test_load_vix_data_corrupt_json(self, tmp_path):
        """Test load_vix_data with corrupt JSON file."""
        data_file = tmp_path / "vix_term_structure.json"
        data_file.write_text("{invalid json}")
        generator = VIXTermStructureSignalGenerator()
        generator.VIX_DATA_PATH = data_file
        result = generator.load_vix_data()
        assert result == {}

    def test_generate_signal_empty_data_dict(self, tmp_path):
        """Test generate_signal with empty data dict returns invalid signal."""
        generator = VIXTermStructureSignalGenerator(db_path=tmp_path / "missing.db")
        with patch.object(generator, "load_vix_data", return_value={}):
            signal = generator.generate_signal("2026-05-15")
            assert not signal.is_valid
            assert signal.reason.startswith("No VIX data available")


class TestCompositeSignalFallback:
    """Test composite signal fallback logic for each VIX level threshold.

    When vix3m is None the code assigns a proxy vix3m based on VIX level,
    then recalculates slope_signal via calculate_slope_signal().
    - VIX < VIX_CHEAP:  proxy = vix * 1.1  (slope 1.10) -> slope_signal = 0.333
    - VIX >= VIX_CHEAP: proxy = vix * 0.9  (slope 0.90) -> slope_signal = -0.833
    """

    def _calc_with_history(self):
        calc = VIXTermStructureCalculator()
        for i in range(60):
            calc.add_vix_reading(f"2026-01-{i+1:03d}", 18.0)
        return calc

    def test_vix3m_none_vix_cheap(self):
        """Test fallback when VIX3M is None and VIX is below cheap threshold."""
        calc = self._calc_with_history()
        components = calc.calculate_composite_signal(
            vix=14.0, vix3m=None, vix6m=None, date="2026-05-15"
        )
        # VIX(14.0) < VIX_CHEAP(16.0) -> proxy=14*1.1=15.4, slope=1.1 -> signal=0.333
        assert components["slope_signal"] == pytest.approx(1 / 3, abs=0.001)

    def test_vix3m_none_vix_between_cheap_and_fair(self):
        """Test fallback when VIX3M is None and VIX is between cheap and fair."""
        calc = self._calc_with_history()
        components = calc.calculate_composite_signal(
            vix=18.0, vix3m=None, vix6m=None, date="2026-05-15"
        )
        # VIX_CHEAP(16.0) <= 18.0 < VIX_FAIR(20.0) -> slope_signal_assigned=0.0
        # proxy=18*0.9=16.2, slope=0.9 -> recalculated signal=-0.833
        assert components["slope_signal"] == pytest.approx(-0.83333, abs=0.001)

    def test_vix3m_none_vix_between_fair_and_expensive(self):
        """Test fallback when VIX3M is None and VIX is between fair and expensive."""
        calc = self._calc_with_history()
        components = calc.calculate_composite_signal(
            vix=22.0, vix3m=None, vix6m=None, date="2026-05-15"
        )
        # VIX_FAIR(20.0) <= 22.0 < VIX_EXPENSIVE(25.0) -> recalculated signal=-0.833
        assert components["slope_signal"] == pytest.approx(-0.83333, abs=0.001)

    def test_vix3m_none_vix_expensive(self):
        """Test fallback when VIX3M is None and VIX is above expensive threshold."""
        calc = self._calc_with_history()
        components = calc.calculate_composite_signal(
            vix=30.0, vix3m=None, vix6m=None, date="2026-05-15"
        )
        # VIX(30.0) >= VIX_EXPENSIVE(25.0) -> recalculated signal=-0.833
        assert components["slope_signal"] == pytest.approx(-0.83333, abs=0.001)

    def test_vix3m_none_at_exact_cheap_threshold(self):
        """Test fallback when VIX equals cheap threshold exactly."""
        calc = self._calc_with_history()
        components = calc.calculate_composite_signal(
            vix=16.0, vix3m=None, vix6m=None, date="2026-05-15"
        )
        # 16.0 < 16.0 is False -> 16.0 < 20.0 is True -> assigned 0.0 -> recalculated -0.833
        assert components["slope_signal"] == pytest.approx(-0.83333, abs=0.001)

    def test_vix3m_none_at_exact_fair_threshold(self):
        """Test fallback when VIX equals fair threshold exactly."""
        calc = self._calc_with_history()
        components = calc.calculate_composite_signal(
            vix=20.0, vix3m=None, vix6m=None, date="2026-05-15"
        )
        # 20.0 < 16.0: False. 20.0 < 20.0: False. 20.0 < 25.0: True -> assigned -0.3
        # proxy=20*0.9=18, slope=0.9 -> recalculated -0.833
        assert components["slope_signal"] == pytest.approx(-0.83333, abs=0.001)

    def test_vix3m_none_at_exact_expensive_threshold(self):
        """Test fallback when VIX equals expensive threshold exactly."""
        calc = self._calc_with_history()
        components = calc.calculate_composite_signal(
            vix=25.0, vix3m=None, vix6m=None, date="2026-05-15"
        )
        # All comparisons False -> else branch: assigned -0.8 -> recalculated -0.833
        assert components["slope_signal"] == pytest.approx(-0.83333, abs=0.001)


class TestSignalDataclassConvenience:
    """Test convenience methods on VIXTermStructureSignal dataclass."""

    def test_to_dict_with_minimal_signal(self):
        """Test to_dict() with minimal (empty) signal."""
        signal = VIXTermStructureSignal(
            timestamp="",
            signal_state="",
            signal_value=0.0,
            vix_spot=0.0,
            vix3m=None,
            vix6m=None,
            slope_vix3m_vix=0.0,
            regime="",
            regime_strength=0.0,
            slope_signal=0.0,
            roll_yield_signal=0.0,
            vix_zscore_signal=0.0,
            curve_shape_signal=0.0,
            spy_shift=0.0,
            gld_shift=0.0,
            tlt_shift=0.0,
            confidence=0.0,
            is_valid=False,
            reason="",
        )
        d = signal.to_dict()
        assert d["timestamp"] == ""
        assert d["vix3m"] is None
        assert d["vix6m"] is None
        assert d["is_valid"] is False
        assert d["confidence"] == 0.0

    def test_to_signal_snapshot_with_invalid_signal(self):
        """Test to_signal_snapshot() with an invalid (is_valid=False) signal."""
        signal = VIXTermStructureSignal(
            timestamp="2026-05-15T12:00:00",
            signal_state="neutral",
            signal_value=0.0,
            vix_spot=0.0,
            vix3m=None,
            vix6m=None,
            slope_vix3m_vix=1.0,
            regime="unknown",
            regime_strength=0.0,
            slope_signal=0.0,
            roll_yield_signal=0.0,
            vix_zscore_signal=0.0,
            curve_shape_signal=0.0,
            spy_shift=0.0,
            gld_shift=0.0,
            tlt_shift=0.0,
            confidence=0.0,
            is_valid=False,
            reason="No data",
        )
        snapshot = signal.to_signal_snapshot()
        assert snapshot.is_active is False
        assert snapshot.value == 0.0
        assert snapshot.confidence == 0.0

    def test_to_dict_roundtrip(self):
        """Verify to_dict() output can recreate a signal with the same fields."""
        original = VIXTermStructureSignal(
            timestamp="2026-05-15T12:00:00",
            signal_state="risk_on",
            signal_value=0.75,
            vix_spot=15.0,
            vix3m=18.0,
            vix6m=20.0,
            slope_vix3m_vix=1.2,
            regime="extreme_contango",
            regime_strength=0.8,
            slope_signal=0.6,
            roll_yield_signal=0.4,
            vix_zscore_signal=0.3,
            curve_shape_signal=0.2,
            spy_shift=0.05,
            gld_shift=-0.03,
            tlt_shift=-0.02,
            confidence=90.0,
            is_valid=True,
            reason="Roundtrip test",
        )
        d = original.to_dict()
        restored = VIXTermStructureSignal(**d)
        assert restored.timestamp == original.timestamp
        assert restored.signal_value == original.signal_value
        assert restored.vix_spot == original.vix_spot
        assert restored.vix3m == original.vix3m
        assert restored.vix6m == original.vix6m
        assert restored.confidence == original.confidence
        assert restored.is_valid == original.is_valid
        assert restored.reason == original.reason


class TestRegimeClassificationEdgeCases:
    """Edge cases for regime classification strength calculation."""

    def test_regime_strength_at_precise_boundaries(self):
        """Verify regime strength at precise regime boundary slopes."""
        calc = VIXTermStructureCalculator()

        # EXTREME_CONTANGO at 1.30: strength = (1.30-1.15)/0.15 + 0.5 = 1.5 -> capped to 1.0
        regime, strength = calc.classify_regime(1.30)
        assert regime == VIXRegime.EXTREME_CONTANGO
        assert strength == 1.0

        # CONTANGO at 1.07: strength = (1.07-1.0)/0.15 = 0.4666...
        regime, strength = calc.classify_regime(1.07)
        assert regime == VIXRegime.CONTANGO
        assert 0.4 <= strength <= 0.5

        # FLAT at 0.96: strength = (1.0-0.96)/0.05 = 0.8
        regime, strength = calc.classify_regime(0.96)
        assert regime == VIXRegime.FLAT
        assert strength == pytest.approx(0.8, abs=1e-9)

        # BACKWARDATION at 0.85: strength = (0.95-0.85)/0.15 = 0.666...
        regime, strength = calc.classify_regime(0.85)
        assert regime == VIXRegime.BACKWARDATION
        assert 0.66 <= strength <= 0.67

        # EXTREME_BACKWARDATION at 0.70: strength = (0.80-0.70)/0.10 + 0.5 = 1.5 -> capped at 1.0
        regime, strength = calc.classify_regime(0.70)
        assert regime == VIXRegime.EXTREME_BACKWARDATION
        assert strength == 1.0

    def test_regime_strength_flat_upper_boundary(self):
        """Test regime strength when slope is exactly 1.0 (upper contango boundary -> EXTREME_CONTANGO)."""
        calc = VIXTermStructureCalculator()
        # Actually 1.0 is CONTANGO threshold via >= 1.0 check
        regime, strength = calc.classify_regime(1.0)
        assert regime == VIXRegime.CONTANGO


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

"""
Tests for VIX Term Structure Signal Generator (v4.50)
Target: 40+ tests for comprehensive coverage.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import json
from datetime import datetime, timedelta

from src.signals.vix_term_structure import (
    VIXRegime,
    VIXSignalState,
    VIXTermStructureCalculator,
    VIXTermStructureSignalGenerator,
    VIXTermStructureSignal,
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
    def test_generate_signal_no_data(self, mock_load_data):
        """Test signal generation with no data available."""
        mock_load_data.return_value = {}
        
        generator = VIXTermStructureSignalGenerator()
        signal = generator.generate_signal('2026-05-15')
        
        assert not signal.is_valid
        assert signal.reason == 'No VIX data available'
    
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
    
    @patch('builtins.open')
    @patch('json.dump')
    def test_save_signal(self, mock_json_dump, mock_open):
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
        
        mock_json_dump.assert_called_once()
    
    @patch('src.signals.vix_term_structure.VIXTermStructureSignalGenerator.load_vix_data')
    def test_signal_history_generation(self, mock_load_data, mock_vix_data):
        """Test generating signals for historical dates."""
        mock_load_data.return_value = mock_vix_data
        
        generator = VIXTermStructureSignalGenerator()
        signals = generator.get_signal_history(days=3)
        
        # Should generate signals for each date
        assert len(signals) <= 3


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
        assert snapshot.confidence == 85.0
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
        """Test generate_signal() with date=None uses datetime.now()."""
        mock_load_data.return_value = {
            "2026-05-12": {
                "date": "2026-05-12",
                "vix_spot": 25.0,
                "front_month": 22.0,
                "third_month": 21.0,
            }
        }

        generator = VIXTermStructureSignalGenerator()
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
    def test_fetch_current_vix_empty_data(self, mock_load_data):
        """Test fetch_current_vix with no data returns None."""
        mock_load_data.return_value = {}

        generator = VIXTermStructureSignalGenerator()
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


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

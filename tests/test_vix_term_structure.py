"""
Tests for VIX Term Structure Signal Generator (v4.50)
Target: 40+ tests for comprehensive coverage.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import json
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

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


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

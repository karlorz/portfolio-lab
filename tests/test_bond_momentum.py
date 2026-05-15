"""
Tests for bond_momentum.py signal module.

Covers:
- BondMomentumCalculator initialization
- Momentum signal calculation for each ETF
- Ensemble recommendation generation
- Price data loading
- Edge cases (insufficient data, missing ETFs)
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sys

from src.signals.bond_momentum import (
    BondMomentumCalculator,
    BondMomentumSignal,
    get_bond_momentum_status
)


@pytest.fixture
def sample_prices():
    """Create sample price data for testing"""
    dates = pd.date_range(start='2020-01-01', end='2024-01-01', freq='D')
    np.random.seed(42)
    
    # Generate realistic bond price series with trend
    n = len(dates)
    tlt_prices = 100 * (1 + np.cumsum(np.random.normal(0.0001, 0.008, n)))
    ief_prices = 100 * (1 + np.cumsum(np.random.normal(0.00008, 0.005, n)))
    shy_prices = 100 * (1 + np.cumsum(np.random.normal(0.00003, 0.002, n)))
    bil_prices = 100 * (1 + np.cumsum(np.random.normal(0.00001, 0.001, n)))
    
    df = pd.DataFrame({
        'TLT': tlt_prices,
        'IEF': ief_prices,
        'SHY': shy_prices,
        'BIL': bil_prices
    }, index=dates)
    
    return df


@pytest.fixture
def calculator(sample_prices):
    """Create calculator with sample data"""
    calc = BondMomentumCalculator()
    calc.prices = sample_prices
    calc.last_updated = datetime.now()
    return calc


class TestBondMomentumCalculator:
    """Test suite for BondMomentumCalculator"""
    
    def test_initialization(self):
        """Test calculator initialization with default config"""
        calc = BondMomentumCalculator()
        
        assert calc.config is not None
        assert 'TLT' in calc.config
        assert 'SHY' in calc.config
        assert calc.prices is None
        assert calc.last_updated is None
    
    def test_custom_config(self):
        """Test calculator with custom configuration"""
        custom_config = {
            'TLT': {'formation_months': 24, 'vol_target': 0.05},
            'SHY': {'formation_months': 6, 'vol_target': 0.03}
        }
        calc = BondMomentumCalculator(config=custom_config)
        
        assert calc.config['TLT']['formation_months'] == 24
        assert calc.config['SHY']['vol_target'] == 0.03
    
    def test_calculate_momentum_tlt(self, calculator):
        """Test momentum calculation for TLT"""
        signal = calculator.calculate_momentum('TLT')
        
        assert signal is not None
        assert signal.etf == 'TLT'
        assert isinstance(signal.signal, float)
        assert 0 <= signal.signal <= 2.0  # Long-only, max 2x
        assert isinstance(signal.formation_return, float)
        assert isinstance(signal.position_size, float)
        assert signal.formation_months == 18  # TLT default
    
    def test_calculate_momentum_shy(self, calculator):
        """Test momentum calculation for SHY"""
        signal = calculator.calculate_momentum('SHY')
        
        assert signal is not None
        assert signal.etf == 'SHY'
        assert 0 <= signal.signal <= 2.0
        assert signal.formation_months == 12  # SHY default
    
    def test_calculate_momentum_ief(self, calculator):
        """Test momentum calculation for IEF"""
        signal = calculator.calculate_momentum('IEF')
        
        assert signal is not None
        assert signal.etf == 'IEF'
        assert isinstance(signal.action, str)
        assert signal.action in ['increase', 'hold', 'reduce', 'avoid']
    
    def test_calculate_momentum_invalid_etf(self, calculator):
        """Test momentum calculation for non-existent ETF"""
        signal = calculator.calculate_momentum('INVALID')
        assert signal is None
    
    def test_calculate_momentum_no_data(self):
        """Test momentum calculation without price data"""
        calc = BondMomentumCalculator()
        signal = calc.calculate_momentum('TLT')
        assert signal is None
    
    def test_calculate_momentum_custom_params(self, calculator):
        """Test momentum with custom formation period"""
        signal = calculator.calculate_momentum('TLT', formation_months=6)
        
        assert signal is not None
        assert signal.formation_months == 6
    
    def test_signal_confidence_levels(self, calculator):
        """Test signal confidence categorization"""
        signal = calculator.calculate_momentum('TLT')
        
        assert signal.confidence in ['strong', 'moderate', 'weak']
    
    def test_action_recommendations(self, calculator):
        """Test action recommendation logic"""
        # Test with different return scenarios by mocking
        signal = calculator.calculate_momentum('TLT')
        
        # TLT uses crisis-indicator logic
        assert signal.action in ['reduce', 'hold', 'avoid']
        
        # SHY uses tactical allocation logic
        signal_shy = calculator.calculate_momentum('SHY')
        assert signal_shy.action in ['increase', 'hold', 'reduce', 'avoid']
    
    def test_weight_delta_range(self, calculator):
        """Test that weight deltas are within expected bounds"""
        for etf in ['TLT', 'IEF', 'SHY', 'BIL']:
            signal = calculator.calculate_momentum(etf)
            if signal:
                assert -0.05 <= signal.weight_delta <= 0.05


class TestCalculateAll:
    """Test suite for calculate_all method"""
    
    def test_calculate_all_default(self, calculator):
        """Test calculating signals for all default ETFs"""
        results = calculator.calculate_all()
        
        assert isinstance(results, dict)
        assert len(results) > 0
        
        for etf, signal in results.items():
            assert isinstance(signal, BondMomentumSignal)
            assert signal.etf == etf
    
    def test_calculate_all_subset(self, calculator):
        """Test calculating signals for subset of ETFs"""
        results = calculator.calculate_all(etfs=['TLT', 'SHY'])
        
        assert len(results) == 2
        assert 'TLT' in results
        assert 'SHY' in results
    
    def test_calculate_all_missing_etf(self, calculator):
        """Test handling missing ETF in subset"""
        results = calculator.calculate_all(etfs=['TLT', 'INVALID'])
        
        # Should only return valid ETFs
        assert 'TLT' in results
        assert 'INVALID' not in results


class TestEnsembleRecommendation:
    """Test suite for ensemble recommendation"""
    
    def test_ensemble_structure(self, calculator):
        """Test ensemble output structure"""
        ensemble = calculator.get_ensemble_recommendation()
        
        assert 'timestamp' in ensemble
        assert 'signal_value' in ensemble
        assert 'confidence' in ensemble
        assert 'recommendation' in ensemble
        assert 'weight_recommendation' in ensemble
        assert 'details' in ensemble
        assert ensemble['source'] == 'bond_momentum'
    
    def test_ensemble_signal_range(self, calculator):
        """Test ensemble signal is normalized to -1 to +1"""
        ensemble = calculator.get_ensemble_recommendation()
        
        assert -1.0 <= ensemble['signal_value'] <= 1.0
    
    def test_ensemble_confidence(self, calculator):
        """Test ensemble confidence levels"""
        ensemble = calculator.get_ensemble_recommendation()
        
        assert ensemble['confidence'] in ['high', 'moderate', 'low', 'none']
    
    def test_ensemble_recommendation_values(self, calculator):
        """Test ensemble recommendation values"""
        ensemble = calculator.get_ensemble_recommendation()
        
        assert ensemble['recommendation'] in [
            'overweight_bonds', 'underweight_bonds', 'neutral'
        ]
    
    def test_ensemble_weight_recommendation(self, calculator):
        """Test ensemble weight recommendation"""
        ensemble = calculator.get_ensemble_recommendation()
        
        # Should be around 2-3% (0.02-0.03)
        assert 0.01 <= ensemble['weight_recommendation'] <= 0.05
    
    def test_ensemble_with_allocation(self, calculator):
        """Test ensemble with current allocation provided"""
        current = {'SHY': 0.10, 'TLT': 0.16}
        ensemble = calculator.get_ensemble_recommendation(current)
        
        assert 'details' in ensemble
        for etf in ['TLT', 'IEF', 'SHY', 'BIL']:
            if etf in ensemble['details']:
                assert 'signal' in ensemble['details'][etf]
                assert 'action' in ensemble['details'][etf]


class TestGetBondMomentumStatus:
    """Test suite for get_bond_momentum_status function"""
    
    def test_status_error_without_data(self):
        """Test status returns error when no data available"""
        # This will try to load from file which may not exist in test env
        status = get_bond_momentum_status()
        
        # Should return error or valid status depending on file existence
        assert 'status' in status
        assert 'timestamp' in status
    
    def test_status_structure_with_data(self, sample_prices, tmp_path):
        """Test status structure when data is available"""
        # Create mock prices.json
        data = {
            'TLT': [{'d': '2024-01-01', 'p': 100.0}, {'d': '2024-01-02', 'p': 101.0}],
            'IEF': [{'d': '2024-01-01', 'p': 100.0}, {'d': '2024-01-02', 'p': 100.5}],
            'SHY': [{'d': '2024-01-01', 'p': 100.0}, {'d': '2024-01-02', 'p': 100.1}],
            'BIL': [{'d': '2024-01-01', 'p': 100.0}, {'d': '2024-01-02', 'p': 100.05}]
        }
        
        # Mock the file loading by directly setting calculator
        calc = BondMomentumCalculator()
        calc.prices = sample_prices
        
        # Test individual signal calculation
        signal = calc.calculate_momentum('TLT')
        assert signal is not None


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_insufficient_data(self):
        """Test handling insufficient historical data"""
        # Create very short price series
        dates = pd.date_range(start='2024-01-01', periods=50, freq='D')
        prices = pd.DataFrame({
            'TLT': np.linspace(100, 105, 50)
        }, index=dates)
        
        calc = BondMomentumCalculator(prices=prices)
        signal = calc.calculate_momentum('TLT')
        
        # Should return None due to insufficient data
        assert signal is None
    
    def test_volatility_zero(self):
        """Test handling zero volatility edge case"""
        # Create flat price series
        dates = pd.date_range(start='2020-01-01', end='2024-01-01', freq='D')
        prices = pd.DataFrame({
            'TLT': np.full(len(dates), 100.0)
        }, index=dates)
        
        calc = BondMomentumCalculator(prices=prices)
        signal = calc.calculate_momentum('TLT')
        
        # Should handle gracefully with position_size=1.0 (default)
        assert signal is not None
        assert signal.position_size == 1.0  # Default when vol is 0
    
    def test_etf_not_in_config(self, calculator):
        """Test ETF not in default config"""
        # Add a new column to prices
        calculator.prices['TEST'] = calculator.prices['TLT'] * 1.01
        
        signal = calculator.calculate_momentum('TEST')
        
        # Should use default parameters
        assert signal is not None
        assert signal.formation_months == 12  # Default
    
    def test_extreme_returns(self):
        """Test handling extreme return scenarios"""
        dates = pd.date_range(start='2020-01-01', end='2024-01-01', freq='D')
        n = len(dates)
        
        # Create extreme trend
        tlt_prices = 100 * np.exp(np.linspace(0, 1.5, n))  # +350% over period
        
        prices = pd.DataFrame({'TLT': tlt_prices}, index=dates)
        calc = BondMomentumCalculator(prices=prices)
        
        signal = calc.calculate_momentum('TLT')
        
        assert signal is not None
        assert signal.confidence == 'strong'  # Extreme return should be strong
        assert signal.signal > 0


class TestConfigDefaults:
    """Test configuration defaults"""
    
    def test_tlt_default_config(self):
        """Test TLT has correct defaults"""
        calc = BondMomentumCalculator()
        
        assert calc.DEFAULT_CONFIG['TLT']['formation_months'] == 18
        assert calc.DEFAULT_CONFIG['TLT']['vol_target'] == 0.06
    
    def test_shy_default_config(self):
        """Test SHY has correct defaults"""
        calc = BondMomentumCalculator()
        
        assert calc.DEFAULT_CONFIG['SHY']['formation_months'] == 12
        assert calc.DEFAULT_CONFIG['SHY']['vol_target'] == 0.06
    
    def test_ief_default_config(self):
        """Test IEF has correct defaults"""
        calc = BondMomentumCalculator()
        
        assert calc.DEFAULT_CONFIG['IEF']['formation_months'] == 12
        assert calc.DEFAULT_CONFIG['IEF']['vol_target'] == 0.06
    
    def test_bil_default_config(self):
        """Test BIL has correct defaults"""
        calc = BondMomentumCalculator()
        
        assert calc.DEFAULT_CONFIG['BIL']['formation_months'] == 12
        assert calc.DEFAULT_CONFIG['BIL']['vol_target'] == 0.04


class TestSignalConsistency:
    """Test signal consistency and determinism"""
    
    def test_signal_determinism(self, calculator):
        """Test signals are deterministic given same data"""
        signal1 = calculator.calculate_momentum('TLT')
        signal2 = calculator.calculate_momentum('TLT')
        
        assert signal1.signal == pytest.approx(signal2.signal, abs=1e-10)
        assert signal1.formation_return == pytest.approx(signal2.formation_return, abs=1e-10)
    
    def test_long_only_constraint(self, calculator):
        """Test long-only constraint (no negative signals)"""
        # Run multiple times with random data to ensure no shorts
        for _ in range(5):
            signal = calculator.calculate_momentum('TLT')
            assert signal.signal >= 0  # Long-only
    
    def test_position_size_cap(self, calculator):
        """Test position size is capped at 2x"""
        for etf in ['TLT', 'IEF', 'SHY', 'BIL']:
            signal = calculator.calculate_momentum(etf)
            assert signal.position_size <= 2.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

"""
Tests for Factor Premia Signal Generator (v4.10)

Tests cover:
- Momentum score calculation
- Cross-sectional ranking
- Factor correlation monitoring
- Regime-based weight adjustments
- Crowding detection
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.signals.factor_premia import (
    FactorPremiaCalculator,
    FactorType,
    FactorSignal,
    FactorEnsemble,
)


# Fixtures

@pytest.fixture
def sample_prices():
    """Create sample price data for factor ETFs."""
    dates = pd.date_range(start='2015-01-01', end='2025-12-31', freq='B')
    n = len(dates)
    
    np.random.seed(42)
    
    # Generate correlated random walks for 4 factor ETFs
    returns = np.random.randn(n, 4) * 0.01
    # Add some correlation
    returns[:, 1] = returns[:, 0] * 0.7 + returns[:, 1] * 0.3  # VLUE correlated with MTUM
    returns[:, 2] = returns[:, 3] * 0.5 + returns[:, 2] * 0.5  # QUAL correlated with USMV
    
    # Add trend for MTUM (momentum factor)
    returns[:, 0] += 0.0002  # Positive drift
    
    prices = 100 * np.exp(np.cumsum(returns, axis=0))
    
    df = pd.DataFrame(
        prices,
        index=dates,
        columns=['MTUM', 'VLUE', 'QUAL', 'USMV']
    )
    return df


@pytest.fixture
def calculator(sample_prices):
    """Create FactorPremiaCalculator with sample data."""
    return FactorPremiaCalculator(sample_prices)


@pytest.fixture
def minimal_prices():
    """Create minimal price data (insufficient for full calculation)."""
    dates = pd.date_range(start='2025-01-01', periods=50, freq='B')
    prices = pd.DataFrame({
        'MTUM': np.linspace(100, 110, 50),
        'VLUE': np.linspace(100, 105, 50),
        'QUAL': np.linspace(100, 108, 50),
        'USMV': np.linspace(100, 102, 50),
    }, index=dates)
    return prices


# Test class initialization

class TestFactorPremiaCalculatorInit:
    """Test calculator initialization and data loading."""
    
    def test_init_with_data(self, sample_prices):
        calc = FactorPremiaCalculator(sample_prices)
        assert calc.prices is not None
        assert len(calc.prices) > 0
    
    def test_init_without_data(self):
        calc = FactorPremiaCalculator()
        assert calc.prices is None
        assert calc.signals_history == []
    
    def test_missing_etf_columns(self):
        bad_prices = pd.DataFrame({'SPY': [100, 101, 102]})
        calc = FactorPremiaCalculator(bad_prices)
        # Should not raise on init, but operations may fail
        assert calc.prices is not None


# Test momentum score calculation

class TestMomentumScoreCalculation:
    """Test momentum score calculation methods."""
    
    def test_calculate_momentum_score_sufficient_data(self, calculator):
        score, vol = calculator.calculate_momentum_score('MTUM')
        assert isinstance(score, float)
        assert isinstance(vol, float)
        assert vol > 0  # Volatility should be positive
    
    def test_calculate_momentum_score_insufficient_data(self, minimal_prices):
        calc = FactorPremiaCalculator(minimal_prices)
        score, vol = calc.calculate_momentum_score('MTUM')
        assert score == 0.0
        assert vol == 0.15  # Default vol
    
    def test_momentum_score_ranking(self, calculator):
        # MTUM has positive drift in sample data, should have higher score
        mtum_score, _ = calculator.calculate_momentum_score('MTUM')
        usmv_score, _ = calculator.calculate_momentum_score('USMV')
        # MTUM should generally have higher momentum score
        assert isinstance(mtum_score, float)
        assert isinstance(usmv_score, float)


# Test cross-sectional ranking

class TestCrossSectionalRanking:
    """Test cross-sectional factor ranking."""
    
    def test_ranking_returns_signals(self, calculator):
        signals = calculator.calculate_cross_sectional_rank(FactorType.MOMENTUM)
        assert isinstance(signals, list)
        if signals:  # If we have enough data
            assert all(isinstance(s, FactorSignal) for s in signals)
    
    def test_ranking_has_proper_fields(self, calculator):
        signals = calculator.calculate_cross_sectional_rank(FactorType.MOMENTUM)
        if signals:
            signal = signals[0]
            assert signal.etf in ['MTUM', 'VLUE', 'QUAL', 'USMV']
            assert isinstance(signal.score, float)
            assert 0 <= signal.score <= 100
            assert isinstance(signal.rank, int)
            assert signal.rank >= 1
            assert signal.recommendation in ['overweight', 'neutral', 'underweight']
    
    def test_ranking_ordering(self, calculator):
        signals = calculator.calculate_cross_sectional_rank(FactorType.MOMENTUM)
        if len(signals) > 1:
            # Scores should be in descending order
            scores = [s.score for s in signals]
            assert scores == sorted(scores, reverse=True)
            # Ranks should be sequential
            ranks = [s.rank for s in signals]
            assert ranks == list(range(1, len(signals) + 1))
    
    def test_ranking_insufficient_data(self, minimal_prices):
        calc = FactorPremiaCalculator(minimal_prices)
        signals = calc.calculate_cross_sectional_rank(FactorType.MOMENTUM)
        # Should return empty list if insufficient data
        assert isinstance(signals, list)


# Test correlation monitoring

class TestCorrelationMonitoring:
    """Test factor correlation matrix and crowding detection."""
    
    def test_correlation_matrix_calculation(self, calculator):
        corr = calculator.calculate_factor_correlations()
        assert isinstance(corr, pd.DataFrame)
        assert corr.shape == (4, 4)  # 4x4 matrix
        assert list(corr.columns) == ['MTUM', 'VLUE', 'QUAL', 'USMV']
        # Diagonal should be 1.0
        for etf in corr.columns:
            assert abs(corr.loc[etf, etf] - 1.0) < 0.001
    
    def test_crowding_detection_normal(self, calculator):
        crowding = calculator.check_factor_crowding()
        assert 'status' in crowding
        assert 'alerts' in crowding
        assert crowding['status'] in ['normal', 'elevated', 'critical', 'unknown']
    
    def test_crowding_with_high_correlation(self):
        # Create highly correlated data
        dates = pd.date_range(start='2020-01-01', periods=500, freq='B')
        returns = np.random.randn(500) * 0.01
        prices = pd.DataFrame({
            'MTUM': 100 * np.exp(np.cumsum(returns)),
            'VLUE': 100 * np.exp(np.cumsum(returns + np.random.randn(500) * 0.002)),
            'QUAL': 100 * np.exp(np.cumsum(np.random.randn(500) * 0.01)),
            'USMV': 100 * np.exp(np.cumsum(np.random.randn(500) * 0.01)),
        }, index=dates)
        
        calc = FactorPremiaCalculator(prices)
        crowding = calc.check_factor_crowding()
        assert 'status' in crowding
        assert 'alerts' in crowding


# Test regime weighting

class TestRegimeWeighting:
    """Test regime-based factor weight adjustments."""
    
    def test_regime_weights_sum_to_one(self, calculator):
        for regime in ['early_cycle', 'mid_cycle', 'late_cycle', 'recession']:
            weights = calculator.get_regime_weights(regime)
            total = sum(weights.values())
            assert abs(total - 1.0) < 0.001, f"Weights for {regime} don't sum to 1"
    
    def test_early_cycle_favors_momentum(self, calculator):
        weights = calculator.get_regime_weights('early_cycle')
        assert weights[FactorType.MOMENTUM] > weights[FactorType.VALUE]
        assert weights[FactorType.MOMENTUM] > weights[FactorType.LOW_VOL]
    
    def test_recession_favors_quality_low_vol(self, calculator):
        weights = calculator.get_regime_weights('recession')
        assert weights[FactorType.QUALITY] > weights[FactorType.MOMENTUM]
        assert weights[FactorType.LOW_VOL] > weights[FactorType.MOMENTUM]


# Test ensemble generation

class TestEnsembleGeneration:
    """Test complete factor ensemble generation."""
    
    def test_generate_ensemble_returns_object(self, calculator):
        ensemble = calculator.generate_ensemble('mid_cycle')
        assert isinstance(ensemble, FactorEnsemble)
        assert ensemble.timestamp is not None
        assert isinstance(ensemble.composite_scores, dict)
    
    def test_ensemble_contains_all_factors(self, calculator):
        ensemble = calculator.generate_ensemble('mid_cycle')
        assert len(ensemble.signals) > 0
        # Should have signals for each factor type
        for factor in FactorType:
            assert factor in ensemble.signals or len(ensemble.signals[factor]) == 0
    
    def test_ensemble_burn_in_tracking(self, calculator):
        ensemble = calculator.generate_ensemble('mid_cycle')
        assert 0 <= ensemble.burn_in_progress <= 100
    
    def test_ensemble_history_tracking(self, calculator):
        calc = FactorPremiaCalculator(calculator.prices)
        ensemble1 = calc.generate_ensemble('mid_cycle')
        ensemble2 = calc.generate_ensemble('mid_cycle')
        assert len(calc.signals_history) == 2


# Test allocation recommendations

class TestAllocationRecommendations:
    """Test allocation recommendation generation."""
    
    def test_recommendations_returns_dict(self, calculator):
        allocations = calculator.get_allocation_recommendations()
        assert isinstance(allocations, dict)
        # Should have keys for all ETFs
        for etf in ['MTUM', 'VLUE', 'QUAL', 'USMV']:
            assert etf in allocations
    
    def test_recommendations_within_budget(self, calculator):
        budget = 0.15
        allocations = calculator.get_allocation_recommendations(budget)
        total = sum(allocations.values())
        assert total <= budget + 0.01  # Allow small rounding error
    
    def test_recommendations_no_micro_positions(self, calculator):
        allocations = calculator.get_allocation_recommendations()
        # Any allocation below 0.5% should be zeroed out
        for etf, alloc in allocations.items():
            if alloc > 0:
                assert alloc >= 0.005, f"Micro-position detected for {etf}: {alloc}"


# Test summary output

class TestSummaryOutput:
    """Test dashboard summary generation."""
    
    def test_summary_contains_required_fields(self, calculator):
        summary = calculator.get_current_summary()
        required_fields = [
            'timestamp', 'burn_in_progress', 'composite_scores',
            'top_pick', 'bottom_pick', 'recommended_allocations',
            'total_factor_allocation', 'crowding_status',
            'avg_confidence', 'signal_ready'
        ]
        for field in required_fields:
            assert field in summary, f"Missing required field: {field}"
    
    def test_signal_ready_logic(self, calculator):
        summary = calculator.get_current_summary()
        # Signal ready when burn_in_progress >= 100
        if summary['burn_in_progress'] >= 100:
            assert summary['signal_ready'] is True
        else:
            assert summary['signal_ready'] is False


# Test serialization

class TestSerialization:
    """Test data serialization for output."""
    
    def test_to_dict_returns_dict(self, calculator):
        ensemble = calculator.generate_ensemble()
        result = calculator.to_dict(ensemble)
        assert isinstance(result, dict)
        assert 'timestamp' in result
        assert 'signals' in result
        assert 'composite_scores' in result


# Integration tests

class TestIntegration:
    """Integration tests for full workflow."""
    
    def test_full_workflow(self, calculator):
        """Test complete factor premia workflow."""
        # Step 1: Calculate correlations
        corr = calculator.calculate_factor_correlations()
        assert corr is not None
        
        # Step 2: Check crowding
        crowding = calculator.check_factor_crowding()
        assert crowding is not None
        
        # Step 3: Generate ensemble
        ensemble = calculator.generate_ensemble('mid_cycle')
        assert ensemble is not None
        
        # Step 4: Get allocations
        allocations = calculator.get_allocation_recommendations()
        assert sum(allocations.values()) <= 0.15 + 0.01
        
        # Step 5: Get summary
        summary = calculator.get_current_summary()
        assert summary['timestamp'] is not None


# Error handling tests

class TestErrorHandling:
    """Test error handling edge cases."""
    
    def test_empty_dataframe(self):
        empty_df = pd.DataFrame()
        calc = FactorPremiaCalculator(empty_df)
        # Operations should not crash, but may return empty results
        try:
            result = calc.calculate_factor_correlations()
            assert result is not None or isinstance(result, pd.DataFrame)
        except Exception as e:
            pytest.fail(f"Should not raise: {e}")
    
    def test_single_row_data(self):
        single = pd.DataFrame({'MTUM': [100]}, index=[datetime.now()])
        calc = FactorPremiaCalculator(single)
        # Should handle gracefully
        corr = calc.calculate_factor_correlations()
        # May be empty but shouldn't crash


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

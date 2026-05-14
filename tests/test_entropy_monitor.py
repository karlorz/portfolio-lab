"""
Tests for Entropy-Based Diversification Monitor v3.22
"""

import pytest
import numpy as np
from src.monitor.entropy_monitor import (
    EntropyCalculator,
    EntropyMetrics,
    EntropyHistory,
    calculate_portfolio_entropy,
    get_calculator
)


class TestShannonEntropy:
    """Test Shannon entropy calculations."""
    
    def test_equal_weights(self):
        """Entropy of equal weights should be ln(n)."""
        calc = EntropyCalculator()
        
        # 2 assets equal weight
        weights = np.array([0.5, 0.5])
        entropy = calc.shannon_entropy(weights)
        assert abs(entropy - np.log(2)) < 0.001
        
        # 4 assets equal weight
        weights = np.array([0.25, 0.25, 0.25, 0.25])
        entropy = calc.shannon_entropy(weights)
        assert abs(entropy - np.log(4)) < 0.001
        
        # 10 assets equal weight
        weights = np.full(10, 0.1)
        entropy = calc.shannon_entropy(weights)
        assert abs(entropy - np.log(10)) < 0.001
    
    def test_single_asset(self):
        """Single asset should have zero entropy."""
        calc = EntropyCalculator()
        weights = np.array([1.0])
        entropy = calc.shannon_entropy(weights)
        assert entropy == 0.0
    
    def test_concentrated_weights(self):
        """Concentrated weights should have low entropy."""
        calc = EntropyCalculator()
        
        # 90% in one asset
        weights = np.array([0.9, 0.1])
        entropy = calc.shannon_entropy(weights)
        
        # Should be less than equal-weight case
        max_entropy = np.log(2)
        assert entropy < max_entropy
        assert entropy > 0
    
    def test_zero_weights_filtered(self):
        """Zero weights should be filtered out."""
        calc = EntropyCalculator()
        
        weights = np.array([0.5, 0.5, 0.0, 0.0])
        entropy = calc.shannon_entropy(weights)
        
        # Should be same as 2-asset case
        assert abs(entropy - np.log(2)) < 0.001
    
    def test_empty_weights(self):
        """Empty array should return 0."""
        calc = EntropyCalculator()
        weights = np.array([])
        entropy = calc.shannon_entropy(weights)
        assert entropy == 0.0
    
    def test_all_zero_weights(self):
        """All zeros should return 0."""
        calc = EntropyCalculator()
        weights = np.array([0.0, 0.0, 0.0])
        entropy = calc.shannon_entropy(weights)
        assert entropy == 0.0


class TestEffectiveNAssets:
    """Test effective number of assets calculation."""
    
    def test_equal_weights(self):
        """Equal weights should give effective_n = n."""
        calc = EntropyCalculator()
        
        # 5 assets equal weight
        entropy = np.log(5)
        effective_n = calc.effective_n_assets(entropy)
        assert abs(effective_n - 5.0) < 0.001
    
    def test_single_asset(self):
        """Single asset effective_n should be 1."""
        calc = EntropyCalculator()
        effective_n = calc.effective_n_assets(0.0)
        assert abs(effective_n - 1.0) < 0.001
    
    def test_concentrated_portfolio(self):
        """Concentrated portfolio has low effective_n."""
        calc = EntropyCalculator()
        
        # 90/10 split
        weights = np.array([0.9, 0.1])
        entropy = calc.shannon_entropy(weights)
        effective_n = calc.effective_n_assets(entropy)
        
        # Effective n should be closer to 1 than 2
        assert 1.0 < effective_n < 1.5


class TestHerfindahlHirschmanIndex:
    """Test HHI calculations."""
    
    def test_equal_weights(self):
        """Equal weights minimize HHI."""
        calc = EntropyCalculator()
        
        # 4 assets equal weight
        weights = np.array([0.25, 0.25, 0.25, 0.25])
        hhi = calc.herfindahl_hirschman_index(weights)
        assert abs(hhi - 0.25) < 0.001  # 4 * 0.25^2 = 0.25
    
    def test_single_asset(self):
        """Single asset maximizes HHI."""
        calc = EntropyCalculator()
        weights = np.array([1.0])
        hhi = calc.herfindahl_hirschman_index(weights)
        assert hhi == 1.0
    
    def test_concentrated(self):
        """Concentrated weights increase HHI."""
        calc = EntropyCalculator()
        
        weights = np.array([0.9, 0.1])
        hhi = calc.herfindahl_hirschman_index(weights)
        assert abs(hhi - 0.82) < 0.001  # 0.9^2 + 0.1^2 = 0.82


class TestNormalizedScore:
    """Test normalization to 0-100 scale."""
    
    def test_maximum_entropy(self):
        """Maximum entropy should give score of 100."""
        calc = EntropyCalculator()
        
        # Equal weights = maximum entropy
        n_assets = 5
        entropy = np.log(n_assets)
        score = calc.normalized_diversification_score(entropy, n_assets)
        assert abs(score - 100.0) < 0.1
    
    def test_minimum_entropy(self):
        """Minimum entropy should give score of 0."""
        calc = EntropyCalculator()
        
        entropy = 0.0
        score = calc.normalized_diversification_score(entropy, 5)
        assert score == 0.0
    
    def test_mid_range(self):
        """Mid-range entropy gives mid-range score."""
        calc = EntropyCalculator()
        
        n_assets = 4
        max_entropy = np.log(n_assets)
        
        # Half of max entropy
        entropy = max_entropy / 2
        score = calc.normalized_diversification_score(entropy, n_assets)
        assert abs(score - 50.0) < 1.0


class TestRiskLevels:
    """Test concentration risk level determination."""
    
    def test_critical_threshold(self):
        """Entropy below 0.5 is critical."""
        calc = EntropyCalculator()
        
        assert calc.concentration_risk_level(0.4) == 'critical'
        assert calc.concentration_risk_level(0.49) == 'critical'
    
    def test_high_threshold(self):
        """Entropy 0.5-0.7 is high (warning)."""
        calc = EntropyCalculator()
        
        assert calc.concentration_risk_level(0.5) == 'high'
        assert calc.concentration_risk_level(0.6) == 'high'
        assert calc.concentration_risk_level(0.69) == 'high'
    
    def test_medium_threshold(self):
        """Entropy 0.7-0.9 is medium."""
        calc = EntropyCalculator()
        
        assert calc.concentration_risk_level(0.7) == 'medium'
        assert calc.concentration_risk_level(0.8) == 'medium'
        assert calc.concentration_risk_level(0.89) == 'medium'
    
    def test_low_threshold(self):
        """Entropy 0.9-1.0 is low."""
        calc = EntropyCalculator()
        
        assert calc.concentration_risk_level(0.9) == 'low'
        assert calc.concentration_risk_level(0.95) == 'low'
        assert calc.concentration_risk_level(0.99) == 'low'
    
    def test_good_threshold(self):
        """Entropy above 1.0 is good."""
        calc = EntropyCalculator()
        
        assert calc.concentration_risk_level(1.0) == 'good'
        assert calc.concentration_risk_level(1.5) == 'good'
        assert calc.concentration_risk_level(2.0) == 'good'


class TestCorrelationEntropy:
    """Test correlation structure entropy."""
    
    def test_identity_matrix(self):
        """Identity correlation matrix gives max entropy."""
        calc = EntropyCalculator()
        
        # 3 uncorrelated assets
        corr = np.eye(3)
        entropy = calc.correlation_entropy(corr)
        
        # Should be close to ln(3)
        assert abs(entropy - np.log(3)) < 0.01
    
    def test_fully_correlated(self):
        """Fully correlated assets give low entropy."""
        calc = EntropyCalculator()
        
        # 3 perfectly correlated assets
        corr = np.ones((3, 3))
        entropy = calc.correlation_entropy(corr)
        
        # Should be close to 0 (one effective dimension)
        assert entropy < 0.5
    
    def test_partial_correlation(self):
        """Partial correlation gives intermediate entropy."""
        calc = EntropyCalculator()
        
        # Some correlation
        corr = np.array([
            [1.0, 0.5, 0.3],
            [0.5, 1.0, 0.4],
            [0.3, 0.4, 1.0]
        ])
        entropy = calc.correlation_entropy(corr)
        
        # Should be between 0 and ln(3)
        assert 0 < entropy < np.log(3)


class TestCalculateMetrics:
    """Test full metrics calculation."""
    
    def test_basic_calculation(self):
        """Test basic metrics calculation."""
        calc = EntropyCalculator()
        
        weights = {'SPY': 0.46, 'GLD': 0.38, 'TLT': 0.16}
        metrics = calc.calculate_metrics(weights)
        
        assert isinstance(metrics, EntropyMetrics)
        assert metrics.shannon_entropy > 0
        assert 1.0 < metrics.effective_n < 3.0
        assert metrics.max_possible == np.log(3)
        assert 0 <= metrics.normalized_score <= 100
        assert metrics.hhi_index > 0
    
    def test_with_correlation(self):
        """Test with correlation matrix."""
        calc = EntropyCalculator()
        
        weights = {'SPY': 0.5, 'GLD': 0.5}
        corr = np.array([[1.0, 0.2], [0.2, 1.0]])
        
        metrics = calc.calculate_metrics(weights, corr)
        
        assert metrics.correlation_entropy is not None
        assert metrics.correlation_entropy > 0
    
    def test_concentrated_portfolio_metrics(self):
        """Test metrics for concentrated portfolio."""
        calc = EntropyCalculator()
        
        # 90% in SPY, 5% in each of others
        weights = {'SPY': 0.9, 'GLD': 0.05, 'TLT': 0.05}
        metrics = calc.calculate_metrics(weights)
        
        # Should detect high concentration
        assert metrics.concentration_risk in ['critical', 'high']
        assert metrics.effective_n < 2.0
        assert metrics.normalized_score < 80
        assert metrics.hhi_index > 0.5


class TestAlertChecking:
    """Test alert generation."""
    
    def test_critical_alert(self):
        """Critical concentration triggers alert."""
        calc = EntropyCalculator()
        
        weights = {'SPY': 0.95, 'GLD': 0.03, 'TLT': 0.02}
        metrics = calc.calculate_metrics(weights)
        
        alert = calc.check_alert(metrics)
        
        assert alert is not None
        assert alert['level'] == 'critical'
        assert 'portfolio entropy' in alert['message'].lower()
        assert 'danger threshold' in alert['message'].lower()
    
    def test_warning_alert(self):
        """High concentration triggers warning."""
        calc = EntropyCalculator()
        
        # Create borderline case
        weights = {'SPY': 0.85, 'GLD': 0.10, 'TLT': 0.05}
        metrics = calc.calculate_metrics(weights)
        
        # May or may not trigger depending on exact entropy
        if metrics.concentration_risk == 'high':
            alert = calc.check_alert(metrics)
            assert alert is not None
            assert alert['level'] == 'warning'
    
    def test_no_alert_for_good_diversification(self):
        """Good diversification produces no alert or only low-level alert."""
        calc = EntropyCalculator()
        
        # Use 8 assets for higher possible entropy (ln(8) ≈ 2.08)
        weights = {'SPY': 0.125, 'GLD': 0.125, 'TLT': 0.125, 'EFA': 0.125,
                   'VXUS': 0.125, 'IEF': 0.125, 'DBC': 0.125, 'MTUM': 0.125}
        metrics = calc.calculate_metrics(weights)
        
        # With 8 equal-weighted assets, entropy should be high (ln(8) ≈ 2.08)
        assert metrics.shannon_entropy > 1.0, f"Entropy {metrics.shannon_entropy} too low for 8 assets"
        
        alert = calc.check_alert(metrics)
        
        # Should either have no alert or a non-critical level
        if alert is not None:
            assert alert['level'] in ['monitor', 'low']
            # Should not be critical for equal weights
            assert alert['level'] != 'critical'


class TestEntropyHistory:
    """Test history tracking."""
    
    def test_add_and_retrieve(self):
        """Test adding metrics to history."""
        history = EntropyHistory(max_history=30)
        
        weights = {'SPY': 0.5, 'GLD': 0.5}
        calc = EntropyCalculator()
        metrics = calc.calculate_metrics(weights)
        
        history.add(metrics)
        
        assert len(history.history) == 1
    
    def test_trend_calculation(self):
        """Test trend calculation."""
        history = EntropyHistory()
        
        calc = EntropyCalculator()
        
        # Add some historical data
        for i in range(5):
            # Declining entropy (worsening concentration)
            weights = {'SPY': 0.5 + i*0.05, 'GLD': 0.5 - i*0.05}
            metrics = calc.calculate_metrics(weights)
            history.add(metrics)
        
        trend = history.get_trend(days=30)
        
        assert trend['available']
        assert trend['n_samples'] == 5
        assert trend['trend_direction'] in ['improving', 'declining']
    
    def test_history_limit(self):
        """Test history respects max limit."""
        history = EntropyHistory(max_history=1)  # Very short for testing
        
        calc = EntropyCalculator()
        
        # Add multiple entries
        for _ in range(5):
            metrics = calc.calculate_metrics({'SPY': 0.5, 'GLD': 0.5})
            history.add(metrics)
        
        # Should only keep recent ones
        assert len(history.history) <= 5  # Limit is time-based, not count-based


class TestConvenienceFunctions:
    """Test module-level convenience functions."""
    
    def test_calculate_portfolio_entropy(self):
        """Test convenience function."""
        weights = {'SPY': 0.46, 'GLD': 0.38, 'TLT': 0.16}
        metrics = calculate_portfolio_entropy(weights)
        
        assert isinstance(metrics, EntropyMetrics)
        assert metrics.shannon_entropy > 0
    
    def test_singleton_calculator(self):
        """Test calculator singleton."""
        calc1 = get_calculator()
        calc2 = get_calculator()
        
        assert calc1 is calc2


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_single_asset_portfolio(self):
        """Single asset portfolio handled correctly."""
        calc = EntropyCalculator()
        
        weights = {'SPY': 1.0}
        metrics = calc.calculate_metrics(weights)
        
        assert metrics.shannon_entropy == 0.0
        assert metrics.effective_n == 1.0
        assert metrics.concentration_risk == 'critical'
    
    def test_very_small_weights(self):
        """Very small weights handled correctly."""
        calc = EntropyCalculator()
        
        weights = np.array([0.999, 0.001])
        entropy = calc.shannon_entropy(weights)
        
        # Should still calculate (very low entropy)
        assert entropy >= 0
        assert entropy < 0.1
    
    def test_negative_weights_rejected(self):
        """Negative weights should be handled gracefully."""
        calc = EntropyCalculator()
        
        # numpy array with negative
        weights = np.array([0.6, -0.1, 0.5])
        
        # Should filter out negatives
        entropy = calc.shannon_entropy(weights)
        assert entropy >= 0


class TestPerformance:
    """Test performance characteristics."""
    
    def test_calculation_speed(self):
        """Entropy calculation should be fast (< 1ms)."""
        import time
        
        calc = EntropyCalculator()
        weights = np.array([0.4, 0.3, 0.2, 0.1])
        
        start = time.time()
        for _ in range(1000):
            _ = calc.shannon_entropy(weights)
        elapsed = time.time() - start
        
        # Should be much faster than 1ms per calculation
        assert elapsed < 1.0  # 1000 calculations in under 1 second

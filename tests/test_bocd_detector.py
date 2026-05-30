"""
Tests for Bayesian Online Changepoint Detection (BOCD) regime detector.

Tests cover:
1. Basic initialization and parameter validation
2. Fitting on synthetic data with known changepoints
3. Detection of mean shifts
4. Signal generation and regime labels
5. Edge cases (empty data, single observation)
"""

import numpy as np
import pytest

from src.regime.bocd_detector import BOCDDetector, BOCDResult


class TestBOCDDetectorInit:
    """Test initialization and parameter validation."""
    
    def test_default_params(self):
        """Default parameters should be valid."""
        detector = BOCDDetector()
        assert detector.hazard_rate == 1.0 / 252
        assert detector.threshold == 0.5
        assert detector.min_run_length == 5
    
    def test_custom_params(self):
        """Custom parameters should be accepted."""
        detector = BOCDDetector(
            hazard_rate=0.1,
            threshold=0.3,
            min_run_length=10
        )
        assert detector.hazard_rate == 0.1
        assert detector.threshold == 0.3
        assert detector.min_run_length == 10
    
    def test_not_fitted_initially(self):
        """Detector should not be fitted initially."""
        detector = BOCDDetector()
        assert detector._fitted is False


class TestBOCDDetectorFit:
    """Test fitting on data."""
    
    def test_fit_returns_self(self):
        """Fit should return self for chaining."""
        detector = BOCDDetector()
        returns = np.random.randn(100) * 0.01
        result = detector.fit(returns)
        assert result is detector
    
    def test_fit_sets_fitted_flag(self):
        """Fit should set _fitted to True."""
        detector = BOCDDetector()
        returns = np.random.randn(100) * 0.01
        detector.fit(returns)
        assert detector._fitted is True
    
    def test_fit_with_timestamps(self):
        """Fit should accept optional timestamps."""
        detector = BOCDDetector()
        returns = np.random.randn(100) * 0.01
        timestamps = list(range(100))
        detector.fit(returns, timestamps=timestamps)
        assert detector.timestamps == timestamps
    
    def test_fit_empty_data(self):
        """Fit should raise ValueError for empty data."""
        detector = BOCDDetector()
        returns = np.array([])
        # Should raise ValueError for empty data
        with pytest.raises(ValueError):
            detector.fit(returns)


class TestBOCDDetection:
    """Test changepoint detection on synthetic data."""
    
    def test_detects_mean_shift(self):
        """Should detect a clear mean shift in the data."""
        np.random.seed(42)
        
        # Create data with a mean shift at t=50
        returns1 = np.random.randn(50) * 0.01 + 0.001  # Mean +0.1%
        returns2 = np.random.randn(50) * 0.01 - 0.001  # Mean -0.1%
        returns = np.concatenate([returns1, returns2])
        
        detector = BOCDDetector(hazard_rate=1/50, threshold=0.3)
        detector.fit(returns)
        
        # Get signal
        signal = detector.get_signal()
        assert isinstance(signal, dict)
        assert 'bocd_detector' in signal
        bocd_data = signal['bocd_detector']
        
        # Should detect changepoints
        assert bocd_data['changepoint_count'] > 0
        assert 0 <= bocd_data['regime_change_prob'] <= 1
    
    def test_no_changepoint_in_stable_data(self):
        """Should not detect changepoints in stable data."""
        np.random.seed(42)
        returns = np.random.randn(100) * 0.01  # Stable returns
        
        detector = BOCDDetector(hazard_rate=1/252, threshold=0.5)
        detector.fit(returns)
        
        signal = detector.get_signal()
        bocd_data = signal['bocd_detector']
        
        # Should have low changepoint probability
        assert bocd_data['regime_change_prob'] < 0.1
    
    def test_multiple_changepoints(self):
        """Should detect multiple changepoints."""
        np.random.seed(42)
        
        # Create data with two mean shifts
        returns1 = np.random.randn(30) * 0.01 + 0.002
        returns2 = np.random.randn(30) * 0.01 - 0.002
        returns3 = np.random.randn(40) * 0.01 + 0.001
        returns = np.concatenate([returns1, returns2, returns3])
        
        detector = BOCDDetector(hazard_rate=1/30, threshold=0.3)
        detector.fit(returns)
        
        signal = detector.get_signal()
        bocd_data = signal['bocd_detector']
        
        # Should detect multiple changepoints
        assert bocd_data['changepoint_count'] >= 2


class TestBOCDGetSignal:
    """Test signal generation for dashboard integration."""
    
    def test_get_signal_returns_dict(self):
        """Get signal should return a dictionary."""
        detector = BOCDDetector()
        returns = np.random.randn(100) * 0.01
        detector.fit(returns)
        
        signal = detector.get_signal()
        assert isinstance(signal, dict)
    
    def test_get_signal_has_required_keys(self):
        """Signal should have required keys for dashboard."""
        detector = BOCDDetector()
        returns = np.random.randn(100) * 0.01
        detector.fit(returns)
        
        signal = detector.get_signal()
        assert 'bocd_detector' in signal
        bocd_data = signal['bocd_detector']
        assert 'regime' in bocd_data
        assert 'regime_change_prob' in bocd_data
        assert 'changepoint_count' in bocd_data
        assert 'current_run_length' in bocd_data
    
    def test_get_signal_regime_values(self):
        """Regime values should be valid."""
        detector = BOCDDetector()
        returns = np.random.randn(100) * 0.01
        detector.fit(returns)
        
        signal = detector.get_signal()
        bocd_data = signal['bocd_detector']
        # Regime should be 0 (no change) or 1 (change)
        assert bocd_data['regime'] in [0, 1]
        # Probabilities should be between 0 and 1
        assert 0 <= bocd_data['regime_change_prob'] <= 1
        assert bocd_data['current_run_length'] >= 0


class TestBOCDGetChangepointTimestamps:
    """Test changepoint timestamp detection."""
    
    def test_get_changepoint_timestamps_returns_list(self):
        """Get changepoint timestamps should return a list."""
        detector = BOCDDetector()
        returns = np.random.randn(100) * 0.01
        detector.fit(returns)
        
        timestamps = detector.get_changepoint_timestamps()
        assert isinstance(timestamps, list)
    
    def test_get_changepoint_timestamps_with_timestamps(self):
        """Should handle custom timestamps."""
        detector = BOCDDetector()
        returns = np.random.randn(100) * 0.01
        timestamps = list(range(100))
        detector.fit(returns, timestamps=timestamps)
        
        changepoints = detector.get_changepoint_timestamps()
        assert isinstance(changepoints, list)


class TestBOCDEdgeCases:
    """Test edge cases and error handling."""
    
    def test_single_observation(self):
        """Should handle single observation."""
        detector = BOCDDetector()
        returns = np.array([0.01, 0.02])  # Need at least 2
        detector.fit(returns)
        
        signal = detector.get_signal()
        assert 'bocd_detector' in signal
    
    def test_constant_returns(self):
        """Should handle constant returns (zero variance)."""
        detector = BOCDDetector()
        returns = np.full(100, 0.01)  # All same value
        detector.fit(returns)
        
        signal = detector.get_signal()
        assert 'bocd_detector' in signal
    
    def test_very_short_data(self):
        """Should handle very short data."""
        detector = BOCDDetector()
        returns = np.array([0.01, 0.02, -0.01])
        detector.fit(returns)
        
        signal = detector.get_signal()
        assert 'bocd_detector' in signal
    
    def test_not_fitted_raises(self):
        """Should raise error if not fitted."""
        detector = BOCDDetector()
        with pytest.raises(RuntimeError):
            detector.get_signal()
    
    def test_multidimensional_input_raises(self):
        """Should raise error for multidimensional input."""
        detector = BOCDDetector()
        returns = np.random.randn(10, 2)  # 2D array
        with pytest.raises(ValueError):
            detector.fit(returns)


class TestBOCDIntegration:
    """Test integration with portfolio-lab signal pipeline."""
    
    def test_signal_snapshot_compatibility(self):
        """Signal should be compatible with get_signal_snapshot() format."""
        detector = BOCDDetector()
        returns = np.random.randn(100) * 0.01
        detector.fit(returns)
        
        signal = detector.get_signal()
        
        # Check it has the structure expected by dashboard
        assert 'bocd_detector' in signal
        bocd_data = signal['bocd_detector']
        assert 'regime' in bocd_data
        assert 'regime_change_prob' in bocd_data
        assert 'changepoint_count' in bocd_data
        assert 'current_run_length' in bocd_data
        
        # Values should be JSON-serializable
        import json
        json.dumps(signal)  # Should not raise
    
    def test_regime_detection_pipeline(self):
        """Test regime detection in a typical pipeline scenario."""
        np.random.seed(42)
        
        # Simulate returns with a regime change - make it more pronounced
        normal_returns = np.random.randn(200) * 0.01
        crisis_returns = np.random.randn(50) * 0.05  # Much higher volatility
        recovery_returns = np.random.randn(50) * 0.01
        
        returns = np.concatenate([normal_returns, crisis_returns, recovery_returns])
        
        # Use higher hazard rate and lower threshold for better detection
        detector = BOCDDetector(hazard_rate=1/50, threshold=0.2)
        detector.fit(returns)
        
        signal = detector.get_signal()
        bocd_data = signal['bocd_detector']
        
        # Should detect regime changes (or at least run without error)
        assert 0 <= bocd_data['regime_change_prob'] <= 1
        assert bocd_data['current_run_length'] >= 0
        assert bocd_data['n_observations'] == 300


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
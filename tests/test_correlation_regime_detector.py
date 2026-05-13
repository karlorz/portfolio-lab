#!/usr/bin/env python3
"""
Tests for correlation regime detector — data classes, regime classification,
Ledoit-Wolf shrinkage, hierarchical clustering, risk parity weights,
adaptive allocation, and persistence.
"""
import sys
import os
import json
import sqlite3
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.strategy.correlation_regime_detector import (
    RegimeType, CorrelationRegime, RegimeClassification,
    AdaptiveWeights, CorrelationRegimeDetector,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_detector(tmp_path, assets=None):
    """Create a CorrelationRegimeDetector with a test database."""
    if assets is None:
        assets = ['SPY', 'GLD', 'TLT']
    db_path = str(tmp_path / "test_regimes.db")
    with patch.object(Path, 'mkdir', lambda *a, **k: None):
        detector = CorrelationRegimeDetector.__new__(CorrelationRegimeDetector)
    detector.assets = assets
    detector.lookback_window = 252
    detector.min_observations = 63
    detector.db_path = Path(db_path)
    detector._regime_matrices = {r: None for r in RegimeType}
    detector._current_regime = None
    detector._regime_start_date = None
    # Initialize DB
    detector._init_db()
    return detector


def _make_returns_df(n_days=100, seed=42, assets=None):
    """Create synthetic daily returns DataFrame."""
    np.random.seed(seed)
    if assets is None:
        assets = ['SPY', 'GLD', 'TLT']
    dates = pd.date_range(end=datetime.now(), periods=n_days, freq='B')
    data = {}
    for ticker in assets:
        data[ticker] = np.random.normal(0.0003, 0.012, n_days)
    return pd.DataFrame(data, index=dates)


def _make_corr_matrix(n=3):
    """Create a simple positive-definite correlation matrix."""
    np.random.seed(42)
    A = np.random.randn(n, n) * 0.3
    cov = A @ A.T + np.eye(n)
    d = np.sqrt(np.diag(cov))
    corr = cov / np.outer(d, d)
    return corr


# ---------------------------------------------------------------------------
# Enum tests
# ---------------------------------------------------------------------------

class TestRegimeType:
    """Test RegimeType enum."""

    def test_values(self):
        assert RegimeType.NORMAL.value == "normal"
        assert RegimeType.HIGH_VOL.value == "high_vol"
        assert RegimeType.CRISIS.value == "crisis"
        assert RegimeType.RECOVERY.value == "recovery"

    def test_all_members(self):
        assert len(RegimeType) == 4


# ---------------------------------------------------------------------------
# Data class tests
# ---------------------------------------------------------------------------

class TestCorrelationRegime:
    """Test CorrelationRegime dataclass."""

    def test_creation(self):
        corr = np.eye(3)
        cov = np.eye(3) * 0.04
        vols = np.array([0.2, 0.2, 0.2])
        regime = CorrelationRegime(
            regime=RegimeType.NORMAL, assets=['SPY', 'GLD', 'TLT'],
            correlation_matrix=corr, covariance_matrix=cov,
            volatilities=vols, start_date='2025-01-01', end_date='2026-01-01',
            sample_size=252, stability_score=0.85,
        )
        assert regime.regime == RegimeType.NORMAL
        assert regime.sample_size == 252
        assert regime.stability_score == 0.85


class TestRegimeClassification:
    """Test RegimeClassification dataclass."""

    def test_creation(self):
        rc = RegimeClassification(
            timestamp=datetime.now().isoformat(),
            current_regime=RegimeType.NORMAL,
            regime_probability={RegimeType.NORMAL: 0.6, RegimeType.HIGH_VOL: 0.4},
            confidence=0.6,
            features={'vix': 18.0, 'realized_vol_30d': 0.16},
            days_in_current_regime=5,
            transition_probability=0.2,
        )
        assert rc.current_regime == RegimeType.NORMAL
        assert rc.confidence == 0.6
        assert len(rc.features) == 2


class TestAdaptiveWeights:
    """Test AdaptiveWeights dataclass."""

    def test_creation(self):
        aw = AdaptiveWeights(
            timestamp=datetime.now().isoformat(),
            regime=RegimeType.NORMAL,
            assets=['SPY', 'GLD', 'TLT'],
            weights=np.array([0.46, 0.38, 0.16]),
            risk_contributions=np.array([0.33, 0.33, 0.33]),
            effective_correlation='regime_normal',
            diversification_ratio=1.5,
        )
        assert len(aw.assets) == 3
        assert aw.diversification_ratio == 1.5


# ---------------------------------------------------------------------------
# Detector init tests
# ---------------------------------------------------------------------------

class TestDetectorInit:
    """Test CorrelationRegimeDetector initialization."""

    def test_default_assets(self, tmp_path):
        detector = _make_detector(tmp_path, assets=None)
        # Default assets include SPY, GLD, TLT, etc.
        assert 'SPY' in detector.assets or len(detector.assets) > 5

    def test_custom_assets(self, tmp_path):
        detector = _make_detector(tmp_path, assets=['SPY', 'TLT'])
        assert detector.assets == ['SPY', 'TLT']

    def test_db_created(self, tmp_path):
        detector = _make_detector(tmp_path)
        assert detector.db_path.exists()

    def test_regime_matrices_initialized(self, tmp_path):
        detector = _make_detector(tmp_path)
        for r in RegimeType:
            assert r in detector._regime_matrices

    def test_thresholds_defined(self, tmp_path):
        detector = _make_detector(tmp_path)
        assert 'vix_normal' in detector.REGIME_THRESHOLDS
        assert 'vix_crisis' in detector.REGIME_THRESHOLDS


# ---------------------------------------------------------------------------
# Regime classification tests
# ---------------------------------------------------------------------------

class TestClassifyRegime:
    """Test classify_regime method."""

    def test_normal_regime(self, tmp_path):
        """Low VIX, stable correlations → NORMAL."""
        detector = _make_detector(tmp_path)
        rc = detector.classify_regime(
            vix=15.0, realized_vol_30d=0.12, spy_trend_50d=0.03,
            correlation_stability=0.8, historical_vix_percentile=30.0,
        )
        assert rc.current_regime == RegimeType.NORMAL
        assert rc.confidence > 0

    def test_crisis_regime(self, tmp_path):
        """Very high VIX, high vol, high percentile → CRISIS."""
        detector = _make_detector(tmp_path)
        rc = detector.classify_regime(
            vix=45.0, realized_vol_30d=0.35, spy_trend_50d=-0.10,
            correlation_stability=0.1, historical_vix_percentile=97.0,
        )
        assert rc.current_regime == RegimeType.CRISIS

    def test_high_vol_regime(self, tmp_path):
        """Elevated VIX (30-40), high percentile → HIGH_VOL."""
        detector = _make_detector(tmp_path)
        rc = detector.classify_regime(
            vix=35.0, realized_vol_30d=0.22, spy_trend_50d=-0.02,
            correlation_stability=0.2, historical_vix_percentile=80.0,
        )
        assert rc.current_regime == RegimeType.HIGH_VOL

    def test_recovery_regime(self, tmp_path):
        """High VIX percentile + positive trend → RECOVERY."""
        detector = _make_detector(tmp_path)
        rc = detector.classify_regime(
            vix=28.0, realized_vol_30d=0.18, spy_trend_50d=0.06,
            correlation_stability=0.6, historical_vix_percentile=75.0,
        )
        assert rc.current_regime == RegimeType.RECOVERY

    def test_probabilities_sum_to_one(self, tmp_path):
        detector = _make_detector(tmp_path)
        rc = detector.classify_regime(
            vix=18.0, realized_vol_30d=0.15, spy_trend_50d=0.02,
            correlation_stability=0.7, historical_vix_percentile=50.0,
        )
        total = sum(rc.regime_probability.values())
        assert abs(total - 1.0) < 0.01

    def test_features_populated(self, tmp_path):
        detector = _make_detector(tmp_path)
        rc = detector.classify_regime(
            vix=20.0, realized_vol_30d=0.16, spy_trend_50d=0.03,
            correlation_stability=0.7, historical_vix_percentile=50.0,
        )
        assert 'vix' in rc.features
        assert 'realized_vol_30d' in rc.features
        assert 'spy_trend_50d' in rc.features
        assert 'correlation_stability' in rc.features
        assert 'vix_percentile' in rc.features

    def test_transition_probability_bounded(self, tmp_path):
        detector = _make_detector(tmp_path)
        rc = detector.classify_regime(
            vix=18.0, realized_vol_30d=0.15, spy_trend_50d=0.02,
            correlation_stability=0.7, historical_vix_percentile=50.0,
        )
        assert 0.0 <= rc.transition_probability <= 1.0


# ---------------------------------------------------------------------------
# Transition probability tests
# ---------------------------------------------------------------------------

class TestTransitionProbability:
    """Test _calculate_transition_probability."""

    def test_high_when_probs_close(self, tmp_path):
        """When top-2 probs are close, transition probability is higher."""
        detector = _make_detector(tmp_path)
        probs = {RegimeType.NORMAL: 0.35, RegimeType.HIGH_VOL: 0.30,
                 RegimeType.CRISIS: 0.20, RegimeType.RECOVERY: 0.15}
        tp = detector._calculate_transition_probability(
            RegimeType.NORMAL, probs, stability=0.5
        )
        assert tp > 0.2  # Close probabilities

    def test_low_when_stable(self, tmp_path):
        """High stability → lower transition probability."""
        detector = _make_detector(tmp_path)
        probs = {RegimeType.NORMAL: 0.7, RegimeType.HIGH_VOL: 0.1,
                 RegimeType.CRISIS: 0.1, RegimeType.RECOVERY: 0.1}
        tp = detector._calculate_transition_probability(
            RegimeType.NORMAL, probs, stability=0.9
        )
        assert tp < 0.5

    def test_bounded_01(self, tmp_path):
        """Transition probability is always in [0, 1]."""
        detector = _make_detector(tmp_path)
        for stability in [0.0, 0.5, 1.0]:
            probs = {RegimeType.NORMAL: 0.25, RegimeType.HIGH_VOL: 0.25,
                     RegimeType.CRISIS: 0.25, RegimeType.RECOVERY: 0.25}
            tp = detector._calculate_transition_probability(
                RegimeType.NORMAL, probs, stability=stability
            )
            assert 0.0 <= tp <= 1.0


# ---------------------------------------------------------------------------
# Correlation matrix estimation tests
# ---------------------------------------------------------------------------

class TestEstimateCorrelationMatrix:
    """Test estimate_correlation_matrix."""

    def test_returns_correlation_regime(self, tmp_path):
        detector = _make_detector(tmp_path)
        returns = _make_returns_df(n_days=100)
        regime = detector.estimate_correlation_matrix(returns, RegimeType.NORMAL)
        assert isinstance(regime, CorrelationRegime)
        assert regime.regime == RegimeType.NORMAL
        assert regime.sample_size == 100

    def test_correlation_diagonal_is_one(self, tmp_path):
        detector = _make_detector(tmp_path)
        returns = _make_returns_df(n_days=100)
        regime = detector.estimate_correlation_matrix(returns, RegimeType.NORMAL)
        diag = np.diag(regime.correlation_matrix)
        np.testing.assert_allclose(diag, 1.0, atol=0.01)

    def test_correlation_bounded(self, tmp_path):
        detector = _make_detector(tmp_path)
        returns = _make_returns_df(n_days=100)
        regime = detector.estimate_correlation_matrix(returns, RegimeType.NORMAL)
        assert np.all(regime.correlation_matrix >= -1.01)
        assert np.all(regime.correlation_matrix <= 1.01)

    def test_stability_score_positive(self, tmp_path):
        detector = _make_detector(tmp_path)
        returns = _make_returns_df(n_days=100)
        regime = detector.estimate_correlation_matrix(returns, RegimeType.NORMAL)
        assert regime.stability_score > 0

    def test_insufficient_data_raises(self, tmp_path):
        detector = _make_detector(tmp_path)
        returns = _make_returns_df(n_days=10)  # < min_observations
        with pytest.raises(ValueError, match="Insufficient"):
            detector.estimate_correlation_matrix(returns, RegimeType.NORMAL)

    def test_no_shrinkage(self, tmp_path):
        detector = _make_detector(tmp_path)
        returns = _make_returns_df(n_days=100)
        regime = detector.estimate_correlation_matrix(
            returns, RegimeType.NORMAL, use_shrinkage=False
        )
        assert isinstance(regime, CorrelationRegime)

    def test_cached_in_regime_matrices(self, tmp_path):
        detector = _make_detector(tmp_path)
        returns = _make_returns_df(n_days=100)
        detector.estimate_correlation_matrix(returns, RegimeType.NORMAL)
        assert detector._regime_matrices[RegimeType.NORMAL] is not None


# ---------------------------------------------------------------------------
# Ledoit-Wolf shrinkage tests
# ---------------------------------------------------------------------------

class TestLedoitWolfShrinkage:
    """Test _ledoit_wolf_shrinkage."""

    def test_returns_covariance_shape(self, tmp_path):
        detector = _make_detector(tmp_path)
        returns = _make_returns_df(n_days=100, assets=['SPY', 'GLD', 'TLT'])
        sample_cov = returns.cov().values
        shrunk = detector._ledoit_wolf_shrinkage(returns, sample_cov)
        assert shrunk.shape == (3, 3)

    def test_symmetric(self, tmp_path):
        detector = _make_detector(tmp_path)
        returns = _make_returns_df(n_days=100)
        sample_cov = returns.cov().values
        shrunk = detector._ledoit_wolf_shrinkage(returns, sample_cov)
        np.testing.assert_allclose(shrunk, shrunk.T, atol=1e-10)

    def test_positive_diagonal(self, tmp_path):
        detector = _make_detector(tmp_path)
        returns = _make_returns_df(n_days=100)
        sample_cov = returns.cov().values
        shrunk = detector._ledoit_wolf_shrinkage(returns, sample_cov)
        assert np.all(np.diag(shrunk) > 0)


# ---------------------------------------------------------------------------
# Hierarchical clustering tests
# ---------------------------------------------------------------------------

class TestHierarchicalClustering:
    """Test hierarchical_clustering."""

    def test_returns_list_of_lists(self, tmp_path):
        detector = _make_detector(tmp_path, assets=['SPY', 'GLD', 'TLT'])
        corr = _make_corr_matrix(3)
        clusters = detector.hierarchical_clustering(corr, ['SPY', 'GLD', 'TLT'])
        assert isinstance(clusters, list)
        for c in clusters:
            assert isinstance(c, list)

    def test_all_assets_assigned(self, tmp_path):
        detector = _make_detector(tmp_path, assets=['SPY', 'GLD', 'TLT'])
        corr = _make_corr_matrix(3)
        clusters = detector.hierarchical_clustering(corr, ['SPY', 'GLD', 'TLT'])
        all_assets = [a for c in clusters for a in c]
        assert set(all_assets) == {'SPY', 'GLD', 'TLT'}

    def test_high_correlation_same_cluster(self, tmp_path):
        """Highly correlated assets should end up in the same cluster."""
        assets = ['A', 'B', 'C']
        corr = np.array([
            [1.0, 0.9, 0.1],
            [0.9, 1.0, 0.1],
            [0.1, 0.1, 1.0],
        ])
        detector = _make_detector(tmp_path, assets=assets)
        clusters = detector.hierarchical_clustering(corr, assets)
        # A and B should be in the same cluster
        for c in clusters:
            if 'A' in c:
                assert 'B' in c
                break


# ---------------------------------------------------------------------------
# Risk parity weights tests
# ---------------------------------------------------------------------------

class TestRiskParityWeights:
    """Test calculate_risk_parity_weights."""

    def test_weights_sum_to_one(self, tmp_path):
        detector = _make_detector(tmp_path)
        cov = np.array([
            [0.04, 0.005, 0.002],
            [0.005, 0.03, 0.001],
            [0.002, 0.001, 0.02],
        ])
        weights, rc = detector.calculate_risk_parity_weights(cov, ['A', 'B', 'C'])
        assert abs(weights.sum() - 1.0) < 0.01

    def test_weights_non_negative(self, tmp_path):
        detector = _make_detector(tmp_path)
        cov = np.array([
            [0.04, 0.005, 0.002],
            [0.005, 0.03, 0.001],
            [0.002, 0.001, 0.02],
        ])
        weights, rc = detector.calculate_risk_parity_weights(cov, ['A', 'B', 'C'])
        assert np.all(weights >= 0)

    def test_risk_contributions_similar(self, tmp_path):
        """Risk contributions should be roughly equal for risk parity."""
        detector = _make_detector(tmp_path)
        cov = np.array([
            [0.04, 0.005, 0.002],
            [0.005, 0.03, 0.001],
            [0.002, 0.001, 0.02],
        ])
        weights, rc = detector.calculate_risk_parity_weights(cov, ['A', 'B', 'C'])
        # Risk contributions should be roughly equal (within 10%)
        mean_rc = rc.mean()
        assert np.all(np.abs(rc - mean_rc) < 0.15)

    def test_custom_target_risk(self, tmp_path):
        detector = _make_detector(tmp_path)
        cov = np.array([
            [0.04, 0.005],
            [0.005, 0.03],
        ])
        target = np.array([0.6, 0.4])
        weights, rc = detector.calculate_risk_parity_weights(cov, ['A', 'B'], target)
        assert abs(weights.sum() - 1.0) < 0.01

    def test_returns_two_arrays(self, tmp_path):
        detector = _make_detector(tmp_path)
        cov = np.array([[0.04, 0.005], [0.005, 0.03]])
        result = detector.calculate_risk_parity_weights(cov, ['A', 'B'])
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Adaptive allocation tests
# ---------------------------------------------------------------------------

class TestGetAdaptiveAllocation:
    """Test get_adaptive_allocation."""

    def test_returns_adaptive_weights(self, tmp_path):
        detector = _make_detector(tmp_path, assets=['SPY', 'GLD', 'TLT'])
        returns = _make_returns_df(n_days=100, assets=['SPY', 'GLD', 'TLT'])
        detector.estimate_correlation_matrix(returns, RegimeType.NORMAL)
        aw = detector.get_adaptive_allocation(
            RegimeType.NORMAL, ['SPY', 'GLD', 'TLT']
        )
        assert isinstance(aw, AdaptiveWeights)
        assert aw.regime == RegimeType.NORMAL

    def test_weights_sum_to_one(self, tmp_path):
        detector = _make_detector(tmp_path, assets=['SPY', 'GLD', 'TLT'])
        returns = _make_returns_df(n_days=100, assets=['SPY', 'GLD', 'TLT'])
        detector.estimate_correlation_matrix(returns, RegimeType.NORMAL)
        aw = detector.get_adaptive_allocation(
            RegimeType.NORMAL, ['SPY', 'GLD', 'TLT']
        )
        assert abs(aw.weights.sum() - 1.0) < 0.01

    def test_subset_of_assets(self, tmp_path):
        """Can request a subset of available assets."""
        detector = _make_detector(tmp_path, assets=['SPY', 'GLD', 'TLT'])
        returns = _make_returns_df(n_days=100, assets=['SPY', 'GLD', 'TLT'])
        detector.estimate_correlation_matrix(returns, RegimeType.NORMAL)
        aw = detector.get_adaptive_allocation(
            RegimeType.NORMAL, ['SPY', 'TLT']
        )
        assert aw.assets == ['SPY', 'TLT']
        assert len(aw.weights) == 2

    def test_no_matrix_raises(self, tmp_path):
        """Raises ValueError if no matrix available and no returns provided."""
        detector = _make_detector(tmp_path, assets=['SPY', 'GLD', 'TLT'])
        with pytest.raises(ValueError, match="No correlation matrix"):
            detector.get_adaptive_allocation(RegimeType.NORMAL, ['SPY'])

    def test_no_matching_assets_raises(self, tmp_path):
        """Raises ValueError if requested assets not in regime matrix."""
        detector = _make_detector(tmp_path, assets=['SPY', 'GLD', 'TLT'])
        returns = _make_returns_df(n_days=100, assets=['SPY', 'GLD', 'TLT'])
        detector.estimate_correlation_matrix(returns, RegimeType.NORMAL)
        with pytest.raises(ValueError, match="No matching"):
            detector.get_adaptive_allocation(RegimeType.NORMAL, ['NONEXISTENT'])

    def test_estimates_from_returns_if_needed(self, tmp_path):
        """Estimates correlation matrix from returns_data if not cached."""
        detector = _make_detector(tmp_path, assets=['SPY', 'GLD', 'TLT'])
        returns = _make_returns_df(n_days=100, assets=['SPY', 'GLD', 'TLT'])
        aw = detector.get_adaptive_allocation(
            RegimeType.NORMAL, ['SPY', 'GLD', 'TLT'], returns_data=returns
        )
        assert isinstance(aw, AdaptiveWeights)

    def test_diversification_ratio_positive(self, tmp_path):
        detector = _make_detector(tmp_path, assets=['SPY', 'GLD', 'TLT'])
        returns = _make_returns_df(n_days=100, assets=['SPY', 'GLD', 'TLT'])
        detector.estimate_correlation_matrix(returns, RegimeType.NORMAL)
        aw = detector.get_adaptive_allocation(
            RegimeType.NORMAL, ['SPY', 'GLD', 'TLT']
        )
        assert aw.diversification_ratio >= 1.0


# ---------------------------------------------------------------------------
# Persistence tests
# ---------------------------------------------------------------------------

class TestSaveRegimeMatrix:
    """Test save_regime_matrix persistence."""

    def test_saves_to_db(self, tmp_path):
        detector = _make_detector(tmp_path, assets=['SPY', 'GLD', 'TLT'])
        returns = _make_returns_df(n_days=100, assets=['SPY', 'GLD', 'TLT'])
        regime = detector.estimate_correlation_matrix(returns, RegimeType.NORMAL)
        detector.save_regime_matrix(regime)

        conn = sqlite3.connect(str(detector.db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM correlation_matrices")
        count = cursor.fetchone()[0]
        conn.close()
        assert count == 1

    def test_upsert_on_duplicate(self, tmp_path):
        """INSERT OR REPLACE on same regime+start_date."""
        detector = _make_detector(tmp_path, assets=['SPY', 'GLD', 'TLT'])
        returns = _make_returns_df(n_days=100, assets=['SPY', 'GLD', 'TLT'])
        regime = detector.estimate_correlation_matrix(returns, RegimeType.NORMAL)
        detector.save_regime_matrix(regime)
        detector.save_regime_matrix(regime)  # Upsert

        conn = sqlite3.connect(str(detector.db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM correlation_matrices")
        count = cursor.fetchone()[0]
        conn.close()
        assert count == 1


class TestSaveRegimeClassification:
    """Test save_regime_classification persistence."""

    def test_saves_to_db(self, tmp_path):
        detector = _make_detector(tmp_path)
        rc = detector.classify_regime(
            vix=18.0, realized_vol_30d=0.16, spy_trend_50d=0.03,
            correlation_stability=0.7, historical_vix_percentile=50.0,
        )
        detector.save_regime_classification(rc)

        conn = sqlite3.connect(str(detector.db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM regime_classifications")
        count = cursor.fetchone()[0]
        conn.close()
        assert count == 1


class TestSaveAdaptiveAllocation:
    """Test save_adaptive_allocation persistence."""

    def test_saves_to_db(self, tmp_path):
        detector = _make_detector(tmp_path, assets=['SPY', 'GLD', 'TLT'])
        returns = _make_returns_df(n_days=100, assets=['SPY', 'GLD', 'TLT'])
        detector.estimate_correlation_matrix(returns, RegimeType.NORMAL)
        aw = detector.get_adaptive_allocation(
            RegimeType.NORMAL, ['SPY', 'GLD', 'TLT']
        )
        detector.save_adaptive_allocation(aw)

        conn = sqlite3.connect(str(detector.db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM adaptive_allocations")
        count = cursor.fetchone()[0]
        conn.close()
        assert count == 1


# ---------------------------------------------------------------------------
# Regime history tests
# ---------------------------------------------------------------------------

class TestGetRegimeHistory:
    """Test get_regime_history."""

    def test_empty_history(self, tmp_path):
        detector = _make_detector(tmp_path)
        history = detector.get_regime_history(days=90)
        assert isinstance(history, list)
        assert len(history) == 0

    def test_returns_saved_classifications(self, tmp_path):
        detector = _make_detector(tmp_path)
        rc = detector.classify_regime(
            vix=18.0, realized_vol_30d=0.16, spy_trend_50d=0.03,
            correlation_stability=0.7, historical_vix_percentile=50.0,
        )
        detector.save_regime_classification(rc)
        history = detector.get_regime_history(days=90)
        assert len(history) == 1
        assert 'timestamp' in history[0]
        assert 'regime' in history[0]
        assert 'confidence' in history[0]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

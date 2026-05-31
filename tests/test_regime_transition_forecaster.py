"""Tests for regime transition forecaster.

TDD red phase — these tests define the behavior before implementation.
"""

import pytest
import numpy as np
from src.regime.regime_transition_forecaster import (
    RegimeTransitionForecaster,
    RegimeForecast,
    REGIMES,
    DEFAULT_PERSISTENCE,
)


class TestRegimeTransitionForecaster:
    """Test suite for RegimeTransitionForecaster."""

    def test_regimes_constant(self):
        """REGIMES should be the 5 portfolio-lab regimes."""
        assert set(REGIMES) == {"NORMAL", "CRISIS", "LOW_VOL", "HIGH_VOL", "RECOVERY"}

    def test_default_persistence(self):
        """DEFAULT_PERSISTENCE should have entries for all 5 regimes."""
        assert set(DEFAULT_PERSISTENCE.keys()) == set(REGIMES)
        assert DEFAULT_PERSISTENCE["NORMAL"] == pytest.approx(7.6, abs=0.1)
        assert DEFAULT_PERSISTENCE["CRISIS"] == pytest.approx(9.9, abs=0.1)
        assert DEFAULT_PERSISTENCE["RECOVERY"] == pytest.approx(1.4, abs=0.1)

    def test_fit_basic(self):
        """fit() should compute a valid transition matrix from a sequence."""
        forecaster = RegimeTransitionForecaster()
        # Simple sequence: NORMAL→NORMAL→CRISIS→CRISIS→NORMAL
        labels = ["NORMAL", "NORMAL", "CRISIS", "CRISIS", "NORMAL"]
        forecaster.fit(labels)
        assert forecaster.is_fitted
        matrix = forecaster.transition_matrix
        # Should be 5×5
        assert matrix.shape == (5, 5)
        # Rows should sum to 1.0 (probability distribution)
        for i in range(5):
            assert matrix[i].sum() == pytest.approx(1.0, abs=1e-6)

    def test_fit_transition_probabilities(self):
        """Transition matrix should reflect the input sequence."""
        forecaster = RegimeTransitionForecaster()
        # NORMAL always stays NORMAL, CRISIS always goes to NORMAL
        labels = ["NORMAL", "NORMAL", "NORMAL", "CRISIS", "NORMAL", "NORMAL"]
        forecaster.fit(labels)
        matrix = forecaster.transition_matrix
        normal_idx = REGIMES.index("NORMAL")
        crisis_idx = REGIMES.index("CRISIS")
        # P(NORMAL→NORMAL) should be high
        assert matrix[normal_idx, normal_idx] > 0.7
        # P(CRISIS→NORMAL) should be ~1.0 (only observed transition, small smoothing)
        assert matrix[crisis_idx, normal_idx] > 0.95

    def test_fit_with_persistence_smoothing(self):
        """Unobserved transitions should get small smoothing, not zero."""
        forecaster = RegimeTransitionForecaster()
        # Only NORMAL and CRISIS in sequence
        labels = ["NORMAL"] * 10 + ["CRISIS"] * 5 + ["NORMAL"] * 5
        forecaster.fit(labels)
        matrix = forecaster.transition_matrix
        # All entries should be > 0 (smoothing applied)
        assert np.all(matrix > 0)
        # Rows should still sum to 1
        for i in range(5):
            assert matrix[i].sum() == pytest.approx(1.0, abs=1e-6)

    def test_forecast_basic(self):
        """forecast() should return probability distribution over regimes."""
        forecaster = RegimeTransitionForecaster()
        labels = ["NORMAL"] * 20 + ["CRISIS"] * 5 + ["NORMAL"] * 10
        forecaster.fit(labels)
        forecast = forecaster.forecast("NORMAL", horizon_days=5)
        assert isinstance(forecast, RegimeForecast)
        assert set(forecast.probabilities.keys()) == set(REGIMES)
        # Probabilities should sum to ~1
        total = sum(forecast.probabilities.values())
        assert total == pytest.approx(1.0, abs=0.01)
        # Most likely regime should be NORMAL (highest persistence)
        assert forecast.most_likely == "NORMAL"

    def test_forecast_crisis_persistence(self):
        """Forecast from CRISIS should show elevated CRISIS probability."""
        forecaster = RegimeTransitionForecaster()
        labels = (
            ["NORMAL"] * 30
            + ["CRISIS"] * 15
            + ["NORMAL"] * 20
            + ["HIGH_VOL"] * 10
            + ["NORMAL"] * 25
        )
        forecaster.fit(labels)
        # Short horizon: CRISIS should still have high probability
        forecast_short = forecaster.forecast("CRISIS", horizon_days=1)
        assert forecast_short.probabilities["CRISIS"] > 0.5
        # Longer horizon: probabilities should shift toward NORMAL
        forecast_long = forecaster.forecast("CRISIS", horizon_days=20)
        assert forecast_long.probabilities["NORMAL"] > forecast_short.probabilities["NORMAL"]

    def test_forecast_recovery_short_persistence(self):
        """RECOVERY has shortest persistence (1.4d) — should decay fastest."""
        forecaster = RegimeTransitionForecaster()
        labels = ["NORMAL"] * 30 + ["RECOVERY"] * 3 + ["NORMAL"] * 20
        forecaster.fit(labels)
        forecast = forecaster.forecast("RECOVERY", horizon_days=5)
        # After 5 days, RECOVERY probability should be low
        assert forecast.probabilities["RECOVERY"] < 0.3

    def test_forecast_horizon_1_day(self):
        """1-day forecast should be close to the transition matrix row."""
        forecaster = RegimeTransitionForecaster()
        labels = (
            ["NORMAL"] * 20
            + ["CRISIS"] * 10
            + ["HIGH_VOL"] * 10
            + ["LOW_VOL"] * 10
            + ["RECOVERY"] * 5
            + ["NORMAL"] * 20
        )
        forecaster.fit(labels)
        forecast = forecaster.forecast("NORMAL", horizon_days=1)
        # Should be close to transition matrix row (blended with persistence)
        normal_idx = REGIMES.index("NORMAL")
        for i, regime in enumerate(REGIMES):
            assert forecast.probabilities[regime] == pytest.approx(
                forecaster.transition_matrix[normal_idx, i], abs=0.05,
            )

    def test_forecast_not_fitted_raises(self):
        """forecast() before fit() should raise ValueError."""
        forecaster = RegimeTransitionForecaster()
        with pytest.raises(ValueError, match="(?i)not fitted"):
            forecaster.forecast("NORMAL", horizon_days=5)

    def test_persistence_decay_rate(self):
        """Persistence parameters should be convertible to daily exit probability."""
        forecaster = RegimeTransitionForecaster()
        labels = ["NORMAL"] * 50
        forecaster.fit(labels)
        exit_probs = forecaster.persistence_exit_probs
        assert set(exit_probs.keys()) == set(REGIMES)
        # Shorter persistence → higher exit probability
        assert exit_probs["RECOVERY"] > exit_probs["CRISIS"]
        # All exit probs should be in (0, 1)
        for p in exit_probs.values():
            assert 0 < p < 1

    def test_empty_sequence_raises(self):
        """fit() with empty list should raise ValueError."""
        forecaster = RegimeTransitionForecaster()
        with pytest.raises(ValueError, match="empty"):
            forecaster.fit([])

    def test_single_label_raises(self):
        """fit() with single label should raise ValueError (no transitions)."""
        forecaster = RegimeTransitionForecaster()
        with pytest.raises(ValueError, match="at least 2"):
            forecaster.fit(["NORMAL"])

    def test_unknown_regime_raises(self):
        """fit() with unknown regime label should raise ValueError."""
        forecaster = RegimeTransitionForecaster()
        with pytest.raises(ValueError, match="Unknown regime"):
            forecaster.fit(["NORMAL", "FAKE_REGIME", "CRISIS"])

    def test_regime_forecast_dataclass(self):
        """RegimeForecast should have the expected fields."""
        forecast = RegimeForecast(
            current_regime="NORMAL",
            horizon_days=5,
            probabilities={"NORMAL": 0.7, "CRISIS": 0.1, "LOW_VOL": 0.1, "HIGH_VOL": 0.05, "RECOVERY": 0.05},
            most_likely="NORMAL",
            transition_matrix=np.ones((5, 5)) / 5,
            persistence_params=DEFAULT_PERSISTENCE,
        )
        assert forecast.current_regime == "NORMAL"
        assert forecast.horizon_days == 5
        assert forecast.most_likely == "NORMAL"

    def test_regime_forecast_from_numpy_labels(self):
        """Should accept numpy integer labels with a regime map."""
        forecaster = RegimeTransitionForecaster()
        # Simulate numpy labels from TwoStageKMeansRegime
        labels = np.array([0, 0, 1, 1, 0, 2, 2, 0, 3, 0, 4, 0])
        regime_map = {0: "NORMAL", 1: "CRISIS", 2: "LOW_VOL", 3: "HIGH_VOL", 4: "RECOVERY"}
        forecaster.fit(labels, regime_map=regime_map)
        assert forecaster.is_fitted
        forecast = forecaster.forecast("NORMAL", horizon_days=3)
        assert isinstance(forecast, RegimeForecast)

    def test_compute_persistence_from_data(self):
        """compute_persistence() should estimate regime durations from data."""
        forecaster = RegimeTransitionForecaster()
        # NORMAL runs of 8, 10, 6 (avg 8), CRISIS runs of 10, 8 (avg 9)
        labels = (
            ["NORMAL"] * 8
            + ["CRISIS"] * 10
            + ["NORMAL"] * 10
            + ["CRISIS"] * 8
            + ["NORMAL"] * 6
        )
        forecaster.fit(labels)
        # Auto-computed persistence should be close to empirical
        assert forecaster.persistence_params["NORMAL"] == pytest.approx(8.0, abs=2.0)
        assert forecaster.persistence_params["CRISIS"] == pytest.approx(9.0, abs=2.0)

    def test_matrix_stochastic(self):
        """Transition matrix rows should be valid probability distributions."""
        forecaster = RegimeTransitionForecaster()
        labels = (
            ["NORMAL"] * 40
            + ["CRISIS"] * 10
            + ["HIGH_VOL"] * 15
            + ["LOW_VOL"] * 20
            + ["RECOVERY"] * 5
            + ["NORMAL"] * 10
        )
        forecaster.fit(labels)
        matrix = forecaster.transition_matrix
        # All non-negative
        assert np.all(matrix >= 0)
        # Rows sum to 1
        np.testing.assert_allclose(matrix.sum(axis=1), 1.0, atol=1e-6)

    def test_forecast_convergence_to_stationary(self):
        """Long-horizon forecast should converge toward stationary distribution."""
        forecaster = RegimeTransitionForecaster()
        labels = (
            ["NORMAL"] * 50
            + ["CRISIS"] * 10
            + ["NORMAL"] * 30
        )
        forecaster.fit(labels)
        # Very long horizon
        forecast = forecaster.forecast("CRISIS", horizon_days=100)
        # Should converge — probabilities should be stable
        forecast_200 = forecaster.forecast("CRISIS", horizon_days=200)
        for regime in REGIMES:
            assert abs(forecast.probabilities[regime] - forecast_200.probabilities[regime]) < 0.01

    def test_get_signal_format(self):
        """get_signal() should return a dict with forecast fields."""
        forecaster = RegimeTransitionForecaster()
        labels = ["NORMAL"] * 20 + ["CRISIS"] * 5 + ["NORMAL"] * 10
        forecaster.fit(labels)
        signal = forecaster.get_signal(current_regime="NORMAL")
        assert "regime" in signal
        assert "confidence" in signal
        assert "forecast_probs" in signal
        assert "horizon_days" in signal
        assert isinstance(signal["forecast_probs"], dict)
        assert signal["regime"] == "NORMAL"

    def test_lowercase_regime_normalization(self):
        """Should accept lowercase regime names from classify_vix_regime."""
        forecaster = RegimeTransitionForecaster()
        labels = ["NORMAL"] * 20 + ["CRISIS"] * 5 + ["NORMAL"] * 10
        forecaster.fit(labels)
        # Lowercase input should be normalized
        forecast = forecaster.forecast("normal", horizon_days=3)
        assert isinstance(forecast, RegimeForecast)
        assert forecast.current_regime == "NORMAL"

    def test_lowercase_fit_labels(self):
        """fit() should accept lowercase labels and normalize them."""
        forecaster = RegimeTransitionForecaster()
        labels = ["normal", "normal", "crisis", "crisis", "normal"]
        forecaster.fit(labels)
        assert forecaster.is_fitted
        forecast = forecaster.forecast("normal", horizon_days=1)
        assert isinstance(forecast, RegimeForecast)

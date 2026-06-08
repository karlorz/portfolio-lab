#!/usr/bin/env python3
"""
Tests for src/agents/predictive_model.py — PredictionResult, CandidateTrajectory,
PredictiveModel (VAR(1)), and TrajectoryOptimizer (CEM).

No ML dependencies — numpy only. Safe for standard test suite.
"""

import numpy as np
import pytest

from src.agents.predictive_model import (
    PredictionResult,
    CandidateTrajectory,
    PredictiveModel,
    TrajectoryOptimizer,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_prices():
    """Generate 260 days of 4-asset price data with slight drift."""
    np.random.seed(42)
    n_days = 260
    n_assets = 4
    returns = np.random.normal(0.0003, 0.01, (n_days, n_assets))
    prices = np.cumprod(1 + returns, axis=0) * 100
    return prices


@pytest.fixture
def fitted_model(simple_prices):
    """Return a PredictiveModel fitted on simple_prices."""
    model = PredictiveModel(n_assets=4)
    model.fit(simple_prices)
    return model


@pytest.fixture
def optimizer():
    """Return a default TrajectoryOptimizer."""
    return TrajectoryOptimizer(n_assets=4)


# ---------------------------------------------------------------------------
# PredictionResult dataclass
# ---------------------------------------------------------------------------

class TestPredictionResult:
    def test_construction(self):
        r = PredictionResult(
            expected_returns=np.array([0.01, 0.02]),
            covariance=np.eye(2) * 0.01,
            trajectories=np.zeros((5, 2)),
            step_confidence=np.ones(5),
            valid=True,
        )
        assert r.valid is True
        assert r.expected_returns.shape == (2,)
        assert r.covariance.shape == (2, 2)
        assert r.trajectories.shape == (5, 2)
        assert r.step_confidence.shape == (5,)

    def test_default_metadata(self):
        r = PredictionResult(
            expected_returns=np.zeros(3),
            covariance=np.eye(3),
            trajectories=np.zeros((5, 3)),
            step_confidence=np.ones(5),
            valid=False,
        )
        assert r.metadata == {}

    def test_custom_metadata(self):
        r = PredictionResult(
            expected_returns=np.zeros(3),
            covariance=np.eye(3),
            trajectories=np.zeros((5, 3)),
            step_confidence=np.ones(5),
            valid=True,
            metadata={"horizon": 5, "fitted": True},
        )
        assert r.metadata["horizon"] == 5
        assert r.metadata["fitted"] is True


# ---------------------------------------------------------------------------
# CandidateTrajectory dataclass
# ---------------------------------------------------------------------------

class TestCandidateTrajectory:
    def test_construction(self):
        ct = CandidateTrajectory(
            allocations=np.ones((3, 2)) * 0.5,
            expected_return=0.05,
            expected_risk=0.10,
            score=0.5,
            feasible=True,
        )
        assert ct.expected_return == 0.05
        assert ct.feasible is True
        assert ct.allocations.shape == (3, 2)

    def test_default_step_scores(self):
        ct = CandidateTrajectory(
            allocations=np.ones((3, 2)) * 0.5,
            expected_return=0.0,
            expected_risk=0.0,
            score=0.0,
        )
        assert len(ct.step_scores) == 0

    def test_custom_step_scores(self):
        scores = np.array([0.3, 0.5, 0.7])
        ct = CandidateTrajectory(
            allocations=np.ones((3, 2)) * 0.5,
            expected_return=0.05,
            expected_risk=0.10,
            score=0.5,
            step_scores=scores,
        )
        np.testing.assert_array_equal(ct.step_scores, scores)

    def test_feasible_default(self):
        ct = CandidateTrajectory(
            allocations=np.ones((1, 2)),
            expected_return=0.0,
            expected_risk=0.0,
            score=0.0,
        )
        assert ct.feasible is True

    def test_infeasible(self):
        ct = CandidateTrajectory(
            allocations=np.ones((1, 2)),
            expected_return=0.0,
            expected_risk=0.0,
            score=0.0,
            feasible=False,
        )
        assert ct.feasible is False


# ---------------------------------------------------------------------------
# PredictiveModel
# ---------------------------------------------------------------------------

class TestPredictiveModelInit:
    def test_defaults(self):
        model = PredictiveModel()
        assert model.n_assets == 4
        assert model.window == 252
        assert model.horizon == 5
        assert model.use_bootstrap is True
        assert model.n_bootstrap == 100
        assert model.is_fitted is False

    def test_custom_params(self):
        model = PredictiveModel(n_assets=3, window=120, horizon=10, use_bootstrap=False, n_bootstrap=50)
        assert model.n_assets == 3
        assert model.window == 120
        assert model.horizon == 10
        assert model.use_bootstrap is False
        assert model.n_bootstrap == 50

    def test_initial_state(self):
        model = PredictiveModel()
        assert model._A is None
        assert model._c is None
        assert model._residuals is None
        assert model._price_history == []


class TestPredictiveModelUpdatePrices:
    def test_update_single(self):
        model = PredictiveModel(n_assets=4)
        model.update_prices(np.array([100.0, 50.0, 30.0, 20.0]))
        assert len(model._price_history) == 1
        np.testing.assert_array_equal(model._price_history[0], [100, 50, 30, 20])

    def test_update_multiple(self):
        model = PredictiveModel(n_assets=2)
        model.update_prices(np.array([100.0, 50.0]))
        model.update_prices(np.array([101.0, 51.0]))
        assert len(model._price_history) == 2

    def test_update_wrong_ndim(self):
        model = PredictiveModel(n_assets=4)
        with pytest.raises(ValueError, match="Expected"):
            model.update_prices(np.array([[100, 50, 30, 20]]))

    def test_update_wrong_n_assets(self):
        model = PredictiveModel(n_assets=4)
        with pytest.raises(ValueError, match="Expected"):
            model.update_prices(np.array([100.0, 50.0]))

    def test_update_trims_history(self):
        model = PredictiveModel(n_assets=2, window=5, horizon=2)
        for i in range(20):
            model.update_prices(np.array([float(i), float(i)]))
        assert len(model._price_history) <= 5 + 2 + 1  # window + horizon + 1

    def test_update_copies_array(self):
        model = PredictiveModel(n_assets=2)
        prices = np.array([100.0, 50.0])
        model.update_prices(prices)
        prices[0] = 999.0
        assert model._price_history[0][0] == 100.0


class TestPredictiveModelFit:
    def test_fit_success(self, simple_prices):
        model = PredictiveModel(n_assets=4)
        result = model.fit(simple_prices)
        assert result is True
        assert model.is_fitted is True
        assert model._A is not None
        assert model._c is not None
        assert model._residuals is not None

    def test_fit_insufficient_data(self):
        model = PredictiveModel(n_assets=4, window=252)
        short_prices = np.random.randn(100, 4) * 10 + 100
        result = model.fit(short_prices)
        assert result is False
        assert model.is_fitted is False

    def test_fit_exactly_min_data(self):
        np.random.seed(7)
        model = PredictiveModel(n_assets=4, window=252)
        prices = np.cumprod(1 + np.random.normal(0, 0.01, (254, 4)), axis=0) * 100
        result = model.fit(prices)
        assert result is True

    def test_fit_stores_price_history(self, simple_prices):
        model = PredictiveModel(n_assets=4)
        model.fit(simple_prices)
        assert len(model._price_history) == simple_prices.shape[0]

    def test_fit_coefficient_shapes(self, simple_prices):
        model = PredictiveModel(n_assets=4)
        model.fit(simple_prices)
        assert model._A.shape == (4, 4)
        assert model._c.shape == (4,)

    def test_fit_residuals_shape(self, simple_prices):
        model = PredictiveModel(n_assets=4)
        model.fit(simple_prices)
        # residuals: (window - 1, n_assets)
        assert model._residuals.shape[1] == 4

    def test_fit_different_n_assets(self):
        np.random.seed(11)
        model = PredictiveModel(n_assets=2, window=50)
        prices = np.cumprod(1 + np.random.normal(0, 0.01, (100, 2)), axis=0) * 100
        result = model.fit(prices)
        assert result is True
        assert model._A.shape == (2, 2)
        assert model._c.shape == (2,)

    def test_fit_constant_prices(self):
        """Constant prices → zero returns → should still fit."""
        model = PredictiveModel(n_assets=2, window=10)
        prices = np.ones((50, 2)) * 100.0
        result = model.fit(prices)
        # With constant returns, OLS may produce zero coefficients but should not error
        assert result is True
        assert model.is_fitted is True


class TestPredictiveModelPredict:
    def test_predict_before_fit(self):
        model = PredictiveModel(n_assets=4)
        result = model.predict()
        assert result.valid is False
        assert result.metadata["fitted"] is False

    def test_predict_after_fit(self, fitted_model):
        result = fitted_model.predict()
        assert result.valid is True
        assert result.expected_returns.shape == (4,)
        assert result.covariance.shape == (4, 4)
        assert result.trajectories.shape == (5, 4)  # default horizon
        assert result.step_confidence.shape == (5,)

    def test_predict_custom_horizon(self, fitted_model):
        result = fitted_model.predict(horizon=10)
        assert result.trajectories.shape[0] == 10
        assert result.step_confidence.shape[0] == 10

    def test_predict_horizon_zero_uses_default(self, fitted_model):
        """horizon=0 is falsy, so 'horizon or self.horizon' uses default."""
        result = fitted_model.predict(horizon=0)
        assert result.trajectories.shape[0] == 5  # falls back to self.horizon

    def test_predict_negative_horizon_treated_as_one(self, fitted_model):
        result = fitted_model.predict(horizon=-5)
        assert result.trajectories.shape[0] == 1

    def test_predict_no_bootstrap(self, simple_prices):
        model = PredictiveModel(n_assets=4, use_bootstrap=False, n_bootstrap=0)
        model.fit(simple_prices)
        result = model.predict()
        assert result.valid is True
        assert result.metadata.get("n_bootstrap", 0) == 0

    def test_predict_honors_use_bootstrap_false_even_with_bootstrap_count(self, simple_prices, monkeypatch):
        model = PredictiveModel(n_assets=4, use_bootstrap=False, n_bootstrap=50)
        model.fit(simple_prices)

        def fail_if_sampled(*args, **kwargs):
            raise AssertionError("bootstrap residual sampling should be skipped")

        monkeypatch.setattr(np.random, "choice", fail_if_sampled)

        result = model.predict()

        assert result.valid is True
        assert result.metadata["n_bootstrap"] == 0

    def test_predict_seeded_bootstrap_replays_step_confidence(self, simple_prices):
        model1 = PredictiveModel(n_assets=4, random_state=123)
        model2 = PredictiveModel(n_assets=4, random_state=123)
        model1.fit(simple_prices)
        model2.fit(simple_prices)

        result1 = model1.predict(horizon=4, n_bootstrap=25)
        result2 = model2.predict(horizon=4, n_bootstrap=25)

        assert result1.metadata["n_bootstrap"] == 25
        assert result2.metadata["n_bootstrap"] == 25
        np.testing.assert_array_equal(result1.step_confidence, result2.step_confidence)

    def test_predict_accepts_generator_for_bootstrap_replay(self, simple_prices):
        model1 = PredictiveModel(n_assets=4, random_state=np.random.default_rng(456))
        model2 = PredictiveModel(n_assets=4, random_state=np.random.default_rng(456))
        model1.fit(simple_prices)
        model2.fit(simple_prices)

        result1 = model1.predict(horizon=4, n_bootstrap=25)
        result2 = model2.predict(horizon=4, n_bootstrap=25)

        np.testing.assert_array_equal(result1.step_confidence, result2.step_confidence)

    def test_predict_custom_n_bootstrap(self, fitted_model):
        result = fitted_model.predict(n_bootstrap=10)
        assert result.valid is True

    def test_predict_no_cov(self, fitted_model):
        result = fitted_model.predict(return_cov=False)
        # Should still return covariance (fallback identity)
        assert result.covariance is not None

    def test_predict_metadata(self, fitted_model):
        result = fitted_model.predict(horizon=7)
        assert result.metadata["horizon"] == 7
        assert result.metadata["fitted"] is True

    def test_predict_confidence_bounded(self, fitted_model):
        result = fitted_model.predict()
        assert np.all(result.step_confidence >= 0.0)
        assert np.all(result.step_confidence <= 1.0)

    def test_predict_covariance_symmetric(self, fitted_model):
        result = fitted_model.predict()
        cov = result.covariance
        np.testing.assert_array_almost_equal(cov, cov.T, decimal=10)

    def test_predict_covariance_positive_semidefinite(self, fitted_model):
        result = fitted_model.predict()
        eigenvalues = np.linalg.eigvalsh(result.covariance)
        assert np.all(eigenvalues >= -1e-8)

    def test_predict_with_few_residuals(self):
        """Test fallback path when residuals have 4-10 samples (not enough for bootstrap)."""
        np.random.seed(99)
        model = PredictiveModel(n_assets=2, window=10, n_bootstrap=0)
        prices = np.cumprod(1 + np.random.normal(0, 0.01, (50, 2)), axis=0) * 100
        model.fit(prices)
        result = model.predict()
        assert result.valid is True

    def test_predict_no_price_history_after_fit(self, simple_prices):
        """If price history is cleared after fit, predict returns empty."""
        model = PredictiveModel(n_assets=4)
        model.fit(simple_prices)
        model.clear_history()
        result = model.predict()
        assert result.valid is False


class TestPredictiveModelEmptyResult:
    def test_empty_result_shape(self):
        model = PredictiveModel(n_assets=3, horizon=7)
        result = model._empty_result()
        assert result.valid is False
        assert result.expected_returns.shape == (3,)
        assert result.covariance.shape == (3, 3)
        assert result.trajectories.shape == (7, 3)
        assert result.step_confidence.shape == (7,)
        np.testing.assert_array_equal(result.expected_returns, 0)

    def test_empty_result_metadata(self):
        model = PredictiveModel()
        result = model._empty_result()
        assert result.metadata["fitted"] is False
        assert result.metadata["reason"] == "insufficient_data"


class TestPredictiveModelGetLoadState:
    def test_get_state_unfitted(self):
        model = PredictiveModel(n_assets=4)
        state = model.get_state()
        assert state["n_assets"] == 4
        assert state["window"] == 252
        assert state["horizon"] == 5
        assert state["A"] is None
        assert state["c"] is None
        assert state["is_fitted"] is False

    def test_get_state_fitted(self, fitted_model):
        state = fitted_model.get_state()
        assert state["is_fitted"] is True
        assert state["A"] is not None
        assert state["c"] is not None
        assert isinstance(state["A"], list)
        assert isinstance(state["c"], list)

    def test_round_trip(self, fitted_model):
        state = fitted_model.get_state()
        model2 = PredictiveModel()
        model2.load_state(state)
        assert model2.n_assets == fitted_model.n_assets
        assert model2.window == fitted_model.window
        assert model2.horizon == fitted_model.horizon
        assert model2.is_fitted is True
        np.testing.assert_array_almost_equal(model2._A, fitted_model._A)
        np.testing.assert_array_almost_equal(model2._c, fitted_model._c)

    def test_load_state_partial(self):
        model = PredictiveModel(n_assets=4)
        model.load_state({"n_assets": 3, "is_fitted": True})
        assert model.n_assets == 3
        assert model.is_fitted is True

    def test_load_state_with_arrays_no_history(self, fitted_model):
        """load_state restores A/c but not price_history or residuals, so predict falls back."""
        state = fitted_model.get_state()
        model2 = PredictiveModel()
        model2.load_state(state)
        # Model is fitted but lacks price history, so predict returns empty
        result = model2.predict()
        assert result.valid is False

    def test_get_state_history_length(self, fitted_model, simple_prices):
        state = fitted_model.get_state()
        assert state["history_length"] == simple_prices.shape[0]


class TestPredictiveModelClearHistory:
    def test_clear_after_fit(self, fitted_model):
        assert len(fitted_model._price_history) > 0
        fitted_model.clear_history()
        assert len(fitted_model._price_history) == 0
        # Model remains fitted
        assert fitted_model.is_fitted is True

    def test_clear_preserves_coefficients(self, fitted_model):
        A_before = fitted_model._A.copy()
        c_before = fitted_model._c.copy()
        fitted_model.clear_history()
        np.testing.assert_array_equal(fitted_model._A, A_before)
        np.testing.assert_array_equal(fitted_model._c, c_before)


# ---------------------------------------------------------------------------
# TrajectoryOptimizer
# ---------------------------------------------------------------------------

class TestTrajectoryOptimizerInit:
    def test_defaults(self):
        opt = TrajectoryOptimizer()
        assert opt.n_assets == 4
        assert opt.n_candidates == 50
        assert opt.n_elite == 10
        assert opt.n_iterations == 3
        np.testing.assert_array_almost_equal(opt.default_weights, [0.46, 0.38, 0.16, 0.0])

    def test_custom_n_assets(self):
        opt = TrajectoryOptimizer(n_assets=3)
        assert opt.n_assets == 3
        # Equal weight default for non-4 assets
        np.testing.assert_array_almost_equal(opt.default_weights, [1/3, 1/3, 1/3])

    def test_custom_params(self):
        opt = TrajectoryOptimizer(n_assets=2, n_candidates=20, n_elite=5, n_iterations=2)
        assert opt.n_candidates == 20
        assert opt.n_elite == 5
        assert opt.n_iterations == 2

    def test_constraints_4_assets(self):
        opt = TrajectoryOptimizer(n_assets=4)
        assert len(opt.min_weights) == 4
        assert len(opt.max_weights) == 4
        assert opt.min_weights[0] == 0.30  # SPY floor
        assert opt.max_weights[0] == 0.60

    def test_constraints_non4_assets(self):
        opt = TrajectoryOptimizer(n_assets=3)
        assert opt.min_weights[0] == 0.30
        assert opt.min_weights[1] == 0.05
        assert opt.max_weights[1] == 0.70


class TestTrajectoryOptimizerSetConstraints:
    def test_set_min_weights(self, optimizer):
        new_min = np.array([0.40, 0.30, 0.10, 0.0])
        optimizer.set_constraints(min_weights=new_min)
        np.testing.assert_array_equal(optimizer.min_weights, new_min)

    def test_set_max_weights(self, optimizer):
        new_max = np.array([0.50, 0.40, 0.20, 0.02])
        optimizer.set_constraints(max_weights=new_max)
        np.testing.assert_array_equal(optimizer.max_weights, new_max)

    def test_set_both(self, optimizer):
        new_min = np.array([0.35, 0.25, 0.08, 0.0])
        new_max = np.array([0.55, 0.45, 0.22, 0.03])
        optimizer.set_constraints(min_weights=new_min, max_weights=new_max)
        np.testing.assert_array_equal(optimizer.min_weights, new_min)
        np.testing.assert_array_equal(optimizer.max_weights, new_max)


class TestTrajectoryOptimizerGenerate:
    def test_generate_with_valid_prediction(self, optimizer, fitted_model):
        np.random.seed(42)
        prediction = fitted_model.predict(horizon=3)
        current_weights = np.array([0.46, 0.38, 0.16, 0.0])
        candidates = optimizer.generate_trajectories(prediction, current_weights, horizon=3)
        assert len(candidates) > 0
        # Sorted by score descending
        for i in range(len(candidates) - 1):
            assert candidates[i].score >= candidates[i + 1].score

    def test_generate_with_invalid_prediction(self, optimizer):
        invalid_pred = PredictionResult(
            expected_returns=np.zeros(4),
            covariance=np.eye(4) * 0.01,
            trajectories=np.zeros((5, 4)),
            step_confidence=np.ones(5),
            valid=False,
        )
        current_weights = np.array([0.46, 0.38, 0.16, 0.0])
        candidates = optimizer.generate_trajectories(invalid_pred, current_weights, horizon=3)
        assert len(candidates) == 1
        assert candidates[0].score == 0.0

    def test_candidate_allocations_shape(self, optimizer, fitted_model):
        np.random.seed(7)
        prediction = fitted_model.predict(horizon=4)
        current_weights = np.array([0.46, 0.38, 0.16, 0.0])
        candidates = optimizer.generate_trajectories(prediction, current_weights, horizon=4)
        for c in candidates:
            assert c.allocations.shape == (4, 4)  # horizon x n_assets

    def test_candidate_step_scores(self, optimizer, fitted_model):
        np.random.seed(13)
        prediction = fitted_model.predict(horizon=3)
        current_weights = np.array([0.46, 0.38, 0.16, 0.0])
        candidates = optimizer.generate_trajectories(prediction, current_weights, horizon=3)
        for c in candidates:
            assert len(c.step_scores) == 3

    def test_seeded_generate_trajectories_replays_top_candidate(self, fitted_model):
        prediction = fitted_model.predict(horizon=3, n_bootstrap=0)
        current_weights = np.array([0.46, 0.38, 0.16, 0.0])
        opt1 = TrajectoryOptimizer(n_assets=4, n_candidates=20, n_elite=5, n_iterations=2, random_state=789)
        opt2 = TrajectoryOptimizer(n_assets=4, n_candidates=20, n_elite=5, n_iterations=2, random_state=789)

        candidates1 = opt1.generate_trajectories(prediction, current_weights, horizon=3)
        candidates2 = opt2.generate_trajectories(prediction, current_weights, horizon=3)

        np.testing.assert_array_equal(candidates1[0].allocations, candidates2[0].allocations)
        assert candidates1[0].score == pytest.approx(candidates2[0].score)

    def test_generate_trajectories_accepts_generator_for_replay(self, fitted_model):
        prediction = fitted_model.predict(horizon=3, n_bootstrap=0)
        current_weights = np.array([0.46, 0.38, 0.16, 0.0])
        opt1 = TrajectoryOptimizer(
            n_assets=4,
            n_candidates=20,
            n_elite=5,
            n_iterations=2,
            random_state=np.random.default_rng(101),
        )
        opt2 = TrajectoryOptimizer(
            n_assets=4,
            n_candidates=20,
            n_elite=5,
            n_iterations=2,
            random_state=np.random.default_rng(101),
        )

        candidates1 = opt1.generate_trajectories(prediction, current_weights, horizon=3)
        candidates2 = opt2.generate_trajectories(prediction, current_weights, horizon=3)

        np.testing.assert_array_equal(candidates1[0].allocations, candidates2[0].allocations)
        assert candidates1[0].score == pytest.approx(candidates2[0].score)


class TestTrajectoryOptimizerSelect:
    def test_select_from_candidates(self, optimizer, fitted_model):
        np.random.seed(42)
        prediction = fitted_model.predict(horizon=3)
        current_weights = np.array([0.46, 0.38, 0.16, 0.0])
        candidates = optimizer.generate_trajectories(prediction, current_weights, horizon=3)
        optimal = optimizer.select_optimal(candidates)
        assert isinstance(optimal, CandidateTrajectory)

    def test_select_empty_list(self, optimizer):
        result = optimizer.select_optimal([])
        assert isinstance(result, CandidateTrajectory)
        assert result.expected_return == 0.0

    def test_select_prefers_feasible(self, optimizer):
        feasible = CandidateTrajectory(
            allocations=np.ones((3, 4)) * 0.25,
            expected_return=0.02,
            expected_risk=0.05,
            score=0.4,
            feasible=True,
        )
        infeasible = CandidateTrajectory(
            allocations=np.ones((3, 4)) * 0.25,
            expected_return=0.10,
            expected_risk=0.02,
            score=5.0,
            feasible=False,
        )
        result = optimizer.select_optimal([infeasible, feasible])
        # Should prefer feasible even if infeasible has higher score
        assert result.feasible is True

    def test_select_risk_aversion(self, optimizer):
        low_risk = CandidateTrajectory(
            allocations=np.ones((3, 4)) * 0.25,
            expected_return=0.03,
            expected_risk=0.01,
            score=3.0,
            feasible=True,
        )
        high_risk = CandidateTrajectory(
            allocations=np.ones((3, 4)) * 0.25,
            expected_return=0.08,
            expected_risk=0.10,
            score=0.8,
            feasible=True,
        )
        # Low risk aversion → prefer high return
        result_low = optimizer.select_optimal([low_risk, high_risk], risk_aversion=0.1)
        # High risk aversion → prefer low risk
        result_high = optimizer.select_optimal([low_risk, high_risk], risk_aversion=10.0)
        assert result_low.expected_return >= result_high.expected_return


class TestTrajectoryOptimizerScoring:
    def test_score_allocation(self, optimizer):
        weights = np.array([0.46, 0.38, 0.16, 0.0])
        expected_returns = np.array([0.01, 0.005, 0.002, 0.0])
        cov = np.eye(4) * 0.01
        score = optimizer._score_allocation(weights, expected_returns, cov)
        assert isinstance(score, float)

    def test_score_zero_risk(self, optimizer):
        """Zero covariance → risk = sqrt(1e-10) → very large score."""
        weights = np.array([0.5, 0.5, 0.0, 0.0])
        expected_returns = np.array([0.01, 0.01, 0.0, 0.0])
        cov = np.zeros((4, 4))  # zero covariance
        score = optimizer._score_allocation(weights, expected_returns, cov)
        # With epsilon in sqrt, risk is tiny but nonzero → very high score
        assert score > 100  # return/risk with near-zero risk is huge

    def test_expected_return(self, optimizer):
        weights = np.array([0.5, 0.3, 0.2, 0.0])
        expected_returns = np.array([0.10, 0.05, 0.02, 0.0])
        ret = optimizer._expected_return(weights, expected_returns)
        assert abs(ret - (0.5 * 0.10 + 0.3 * 0.05 + 0.2 * 0.02)) < 1e-10

    def test_expected_risk(self, optimizer):
        weights = np.array([0.5, 0.5, 0.0, 0.0])
        cov = np.diag([0.04, 0.01, 0.01, 0.01])
        risk = optimizer._expected_risk(weights, cov)
        assert risk > 0
        # sqrt(0.5^2 * 0.04 + 0.5^2 * 0.01) ≈ sqrt(0.01 + 0.0025) ≈ 0.1118
        assert abs(risk - 0.1118) < 0.01


class TestTrajectoryOptimizerDefaultTrajectory:
    def test_default_trajectory(self, optimizer):
        current = np.array([0.46, 0.38, 0.16, 0.0])
        result = optimizer._default_trajectory(current, horizon=3)
        assert len(result) == 1
        assert result[0].score == 0.0
        assert result[0].feasible is True
        assert result[0].allocations.shape == (3, 4)

    def test_default_trajectory_single(self, optimizer):
        result = optimizer._default_trajectory_single()
        assert isinstance(result, CandidateTrajectory)
        assert result.allocations.shape == (5, 4)  # default horizon=5
        assert result.score == 0.0


# ---------------------------------------------------------------------------
# Integration / edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_fit_predict_single_asset(self):
        np.random.seed(33)
        model = PredictiveModel(n_assets=1, window=30)
        prices = np.cumprod(1 + np.random.normal(0, 0.01, (100, 1)), axis=0) * 100
        assert model.fit(prices) is True
        result = model.predict()
        assert result.valid is True
        assert result.expected_returns.shape == (1,)

    def test_fit_large_n_assets(self):
        np.random.seed(44)
        n = 8
        model = PredictiveModel(n_assets=n, window=50)
        prices = np.cumprod(1 + np.random.normal(0, 0.01, (100, n)), axis=0) * 100
        assert model.fit(prices) is True
        result = model.predict()
        assert result.valid is True
        assert result.expected_returns.shape == (n,)
        assert result.covariance.shape == (n, n)

    def test_update_then_predict(self):
        """Fit, then update prices, then predict should use updated history."""
        np.random.seed(55)
        model = PredictiveModel(n_assets=2, window=30)
        prices = np.cumprod(1 + np.random.normal(0, 0.01, (80, 2)), axis=0) * 100
        model.fit(prices)
        # Add more observations
        for i in range(5):
            model.update_prices(prices[-1] * (1 + np.random.normal(0, 0.01, 2)))
        result = model.predict()
        assert result.valid is True

    def test_multiple_fits(self):
        """Fitting multiple times should overwrite previous state."""
        np.random.seed(66)
        model = PredictiveModel(n_assets=2, window=30)
        prices1 = np.cumprod(1 + np.random.normal(0.001, 0.01, (80, 2)), axis=0) * 100
        prices2 = np.cumprod(1 + np.random.normal(-0.001, 0.02, (80, 2)), axis=0) * 100
        model.fit(prices1)
        c1 = model._c.copy()
        model.fit(prices2)
        c2 = model._c.copy()
        # Different data should produce different intercepts (almost surely)
        assert not np.allclose(c1, c2, atol=1e-6)

    def test_optimizer_with_2_assets(self):
        np.random.seed(77)
        opt = TrajectoryOptimizer(n_assets=2, n_candidates=10, n_iterations=1)
        pred = PredictionResult(
            expected_returns=np.array([0.01, 0.005]),
            covariance=np.eye(2) * 0.01,
            trajectories=np.zeros((3, 2)),
            step_confidence=np.ones(3),
            valid=True,
        )
        current = np.array([0.6, 0.4])
        candidates = opt.generate_trajectories(pred, current, horizon=3)
        assert len(candidates) > 0

    def test_prediction_result_with_high_vol(self):
        """High volatility should still produce valid predictions."""
        np.random.seed(88)
        model = PredictiveModel(n_assets=2, window=30)
        prices = np.cumprod(1 + np.random.normal(0, 0.05, (80, 2)), axis=0) * 100
        model.fit(prices)
        result = model.predict()
        assert result.valid is True

    def test_state_serialization_preserves_predict(self, fitted_model):
        """After save/load state, predictions should match."""
        np.random.seed(99)
        state = fitted_model.get_state()
        model2 = PredictiveModel()
        model2.load_state(state)
        # Load price history too for prediction
        for p in fitted_model._price_history:
            model2._price_history.append(p.copy())
        r1 = fitted_model.predict(horizon=3, n_bootstrap=0)
        r2 = model2.predict(horizon=3, n_bootstrap=0)
        np.testing.assert_array_almost_equal(r1.expected_returns, r2.expected_returns)

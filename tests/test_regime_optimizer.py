"""
Tests for v6.01: Regime-Constrained Portfolio Optimizer.

Covers:
- Covariance matrix building and blending
- All 3 optimization modes (min_vol, max_sharpe, risk_parity)
- Hard constraint enforcement
- Regime probability extraction
- Edge cases: single regime, fallback, infeasible regimes
- Integration with regime classifier state format
"""

import sys
import os
import json
import numpy as np
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from src.strategy.regime_optimizer import (
    RegimeConstrainedOptimizer,
    RegimeCovarianceBuilder,
    RegimeCovariance,
    OptimizerResult,
    BASE_ALLOCATION,
    HARD_BOUNDS,
    ASSETS,
    CORE_ASSETS,
    REGIME_COVARIANCES,
    REGIME_EXPECTED_RETURNS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create a temporary data directory."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


@pytest.fixture
def normal_regime_state(temp_data_dir):
    """Create a regime state file indicating NORMAL conditions."""
    state = {
        "current_regime": "normal",
        "previous_regime": None,
        "regime_start_date": "2026-04-01",
        "last_reading": {
            "regime": "normal",
            "confidence": 0.7,
            "timestampped": datetime.now().isoformat(),
        },
        "last_updated": datetime.now().isoformat(),
    }
    state_path = temp_data_dir / "regime_classifier_state.json"
    state_path.write_text(json.dumps(state))
    return temp_data_dir


@pytest.fixture
def crisis_regime_state(temp_data_dir):
    """Create a regime state file indicating CRISIS conditions."""
    state = {
        "current_regime": "crisis",
        "previous_regime": "high_vol",
        "regime_start_date": "2026-05-01",
        "last_reading": {
            "regime": "crisis",
            "confidence": 0.9,
            "timestampped": datetime.now().isoformat(),
        },
        "last_updated": datetime.now().isoformat(),
    }
    state_path = temp_data_dir / "regime_classifier_state.json"
    state_path.write_text(json.dumps(state))
    return temp_data_dir


@pytest.fixture
def low_vol_regime_state(temp_data_dir):
    """Create a regime state file indicating LOW_VOL conditions."""
    state = {
        "current_regime": "low_vol",
        "previous_regime": "normal",
        "regime_start_date": "2026-05-10",
        "last_reading": {
            "regime": "low_vol",
            "confidence": 0.75,
            "timestampped": datetime.now().isoformat(),
        },
        "last_updated": datetime.now().isoformat(),
    }
    state_path = temp_data_dir / "regime_classifier_state.json"
    state_path.write_text(json.dumps(state))
    return temp_data_dir


# ---------------------------------------------------------------------------
# Test: RegimeCovarianceBuilder
# ---------------------------------------------------------------------------


class TestRegimeCovarianceBuilder:
    """Test covariance matrix building utilities."""

    def test_regime_probabilities_normal(self):
        """Normal regime should have 70% on normal, rest distributed."""
        reading = {"regime": "normal", "confidence": 0.7, "previous_regime": None}
        probs = RegimeCovarianceBuilder.regime_probabilities(reading)
        assert abs(sum(probs.values()) - 1.0) < 0.01
        assert probs["normal"] >= 0.60 and probs["normal"] <= 0.80
        for r in REGIME_COVARIANCES:
            assert r in probs

    def test_regime_probabilities_crisis_confidence(self):
        """Crisis should have 90% confidence."""
        reading = {"regime": "crisis", "confidence": 0.9, "previous_regime": "high_vol"}
        probs = RegimeCovarianceBuilder.regime_probabilities(reading)
        assert abs(sum(probs.values()) - 1.0) < 0.01
        assert probs["crisis"] >= 0.80

    def test_regime_probabilities_with_previous(self):
        """Previous regime should get extra probability weight."""
        reading = {"regime": "recovery", "confidence": 0.7, "previous_regime": "crisis"}
        probs = RegimeCovarianceBuilder.regime_probabilities(reading)
        assert probs["crisis"] > 0.03  # Previous regime gets more than others

    def test_build_cov_matrix_creates_all_entries(self):
        """Blended covariance should have entries for all assets."""
        probs = {"normal": 1.0}
        for r in REGIME_COVARIANCES:
            if r != "normal":
                probs[r] = 0.0
        blended = RegimeCovarianceBuilder.build_cov_matrix(probs)
        for a in ASSETS:
            for b in ASSETS:
                assert a in blended
                assert b in blended[a]

    def test_cov_matrix_pure_normal(self):
        """100% normal regime should match normal covariance exactly."""
        probs = {"normal": 1.0, "low_vol": 0.0, "high_vol": 0.0,
                 "crisis": 0.0, "recovery": 0.0, "unknown": 0.0}
        blended = RegimeCovarianceBuilder.build_cov_matrix(probs, CORE_ASSETS)
        for a in CORE_ASSETS:
            for b in CORE_ASSETS:
                expected = REGIME_COVARIANCES["normal"][a][b]
                assert abs(blended[a][b] - expected) < 1e-10

    def test_cov_matrix_50_50_blend(self):
        """50/50 blend of normal and crisis should be halfway between."""
        probs = {"normal": 0.5, "crisis": 0.5, "low_vol": 0.0,
                 "high_vol": 0.0, "recovery": 0.0, "unknown": 0.0}
        blended = RegimeCovarianceBuilder.build_cov_matrix(probs, CORE_ASSETS)
        for a in CORE_ASSETS:
            for b in CORE_ASSETS:
                midway = (REGIME_COVARIANCES["normal"][a][b] +
                          REGIME_COVARIANCES["crisis"][a][b]) / 2
                assert abs(blended[a][b] - midway) < 1e-10

    def test_cov_to_numpy_shape(self):
        """Numpy conversion should produce correct shape matrix."""
        probs = {"normal": 1.0}
        for r in REGIME_COVARIANCES:
            if r != "normal":
                probs[r] = 0.0
        blended = RegimeCovarianceBuilder.build_cov_matrix(probs)
        cov = RegimeCovarianceBuilder.cov_to_numpy(blended)
        assert cov.shape == (len(ASSETS), len(ASSETS))

    def test_cov_to_numpy_symmetry(self):
        """Numpy cov matrix must be symmetric."""
        probs = {"crisis": 0.5, "normal": 0.3, "high_vol": 0.2,
                 "low_vol": 0.0, "recovery": 0.0, "unknown": 0.0}
        blended = RegimeCovarianceBuilder.build_cov_matrix(probs)
        cov = RegimeCovarianceBuilder.cov_to_numpy(blended)
        assert np.allclose(cov, cov.T)

    def test_cov_to_numpy_psd(self):
        """Covariance matrix should be positive semi-definite."""
        probs = {"normal": 0.6, "crisis": 0.4, "low_vol": 0.0,
                 "high_vol": 0.0, "recovery": 0.0, "unknown": 0.0}
        blended = RegimeCovarianceBuilder.build_cov_matrix(probs)
        cov = RegimeCovarianceBuilder.cov_to_numpy(blended)
        eigenvalues = np.linalg.eigvalsh(cov)
        assert eigenvalues.min() >= -1e-8

    def test_all_regimes_have_covariances(self):
        """All regimes should have defined covariance entries for all assets."""
        for regime in REGIME_COVARIANCES:
            cov = REGIME_COVARIANCES[regime]
            for a in ASSETS:
                assert a in cov, f"Regime {regime} missing asset {a}"
                for b in ASSETS:
                    assert b in cov[a], f"Regime {regime} missing {a}->{b}"

    def test_all_regimes_have_expected_returns(self):
        """All regimes should have expected returns for all core assets."""
        for regime in REGIME_EXPECTED_RETURNS:
            rets = REGIME_EXPECTED_RETURNS[regime]
            for a in ASSETS:
                assert a in rets, f"Regime {regime} missing return for {a}"


# ---------------------------------------------------------------------------
# Test: Optimizer Initialization and State
# ---------------------------------------------------------------------------


class TestRegimeOptimizerInit:
    """Test optimizer initialization."""

    def test_init_default(self, tmp_path):
        """Default initialization should work."""
        optimizer = RegimeConstrainedOptimizer(data_dir=tmp_path / "data")
        assert optimizer.last_result is None
        assert optimizer.current_regime == "normal"
        assert optimizer.regime_confidence == 0.7
        assert optimizer.risk_free_rate == 0.04

    def test_init_custom_rf(self, tmp_path):
        """Custom risk-free rate should be stored."""
        optimizer = RegimeConstrainedOptimizer(data_dir=tmp_path / "data", risk_free_rate=0.03)
        assert optimizer.risk_free_rate == 0.03

    def test_init_loads_previous_state(self, temp_data_dir):
        """Loading previous state should restore regime info."""
        # Create a previous state
        state_path = temp_data_dir / "regime_optimizer_state.json"
        state = {
            "current_regime": "high_vol",
            "regime_confidence": 0.8,
            "last_updated": datetime.now().isoformat(),
            "method": "min_vol",
        }
        state_path.write_text(json.dumps(state))

        optimizer = RegimeConstrainedOptimizer(data_dir=temp_data_dir)
        assert optimizer.current_regime == "high_vol"
        assert optimizer.regime_confidence == 0.8

    def test_regime_loading(self, normal_regime_state):
        """Regime state should load from classifier state file."""
        optimizer = RegimeConstrainedOptimizer(data_dir=normal_regime_state)
        state = optimizer._load_regime_state()
        assert state["regime"] == "normal"
        assert state["confidence"] == 0.7

    def test_regime_loading_crisis(self, crisis_regime_state):
        """Crisis regime should load correctly."""
        optimizer = RegimeConstrainedOptimizer(data_dir=crisis_regime_state)
        state = optimizer._load_regime_state()
        assert state["regime"] == "crisis"
        assert state["confidence"] == 0.9
        assert state["previous_regime"] == "high_vol"


# ---------------------------------------------------------------------------
# Test: Covariance Building Integration
# ---------------------------------------------------------------------------


class TestRegimeCovarianceIntegration:
    """Test covariance building integrated with optimizer."""

    def test_build_regime_cov_normal(self, normal_regime_state):
        """Should build proper covariance for normal regime."""
        optimizer = RegimeConstrainedOptimizer(data_dir=normal_regime_state)
        regime_cov = optimizer.build_regime_covariance()
        assert regime_cov.regime == "normal"
        assert regime_cov.confidence >= 0.6
        assert len(regime_cov.regime_probs) == len(REGIME_COVARIANCES)

    def test_build_regime_cov_crisis(self, crisis_regime_state):
        """Should build proper covariance for crisis regime."""
        optimizer = RegimeConstrainedOptimizer(data_dir=crisis_regime_state)
        regime_cov = optimizer.build_regime_covariance()
        assert regime_cov.regime == "crisis"
        assert regime_cov.confidence >= 0.8
        assert regime_cov.blended

    def test_covariance_differentiates_regimes(self, normal_regime_state, crisis_regime_state):
        """Crisis covariance should show higher SPY-GLD covariance than normal."""
        # Compare raw regime covariances (before blending with probability floor)
        crisis_raw = REGIME_COVARIANCES["crisis"]["SPY"]["GLD"]
        normal_raw = REGIME_COVARIANCES["normal"]["SPY"]["GLD"]
        assert crisis_raw > normal_raw * 3, \
            f"Crisis raw SPY-GLD cov {crisis_raw:.4f} should be > normal {normal_raw:.4f}"

        # Verify raw regime SPY vols are differentiated
        crisis_spy_vol = np.sqrt(REGIME_COVARIANCES["crisis"]["SPY"]["SPY"])
        normal_spy_vol = np.sqrt(REGIME_COVARIANCES["normal"]["SPY"]["SPY"])
        assert crisis_spy_vol > normal_spy_vol, \
            f"Crisis SPY vol {crisis_spy_vol:.2%} should be > normal {normal_spy_vol:.2%}"


# ---------------------------------------------------------------------------
# Test: Optimization Methods
# ---------------------------------------------------------------------------


class TestOptimizationMethods:
    """Test all 3 optimization modes."""

    def test_min_vol_runs(self, normal_regime_state):
        """Min vol optimization should run and produce weights."""
        optimizer = RegimeConstrainedOptimizer(data_dir=normal_regime_state)
        result = optimizer.optimize(method="min_vol")
        assert result.method == "min_vol"
        assert len(result.weights) == len(ASSETS)
        assert abs(sum(result.weights.values()) - 1.0) < 0.01

    def test_min_vol_weights_within_bounds(self, normal_regime_state):
        """Min vol weights should respect hard bounds."""
        optimizer = RegimeConstrainedOptimizer(data_dir=normal_regime_state)
        result = optimizer.optimize(method="min_vol")
        for asset, weight in result.weights.items():
            lo, hi = HARD_BOUNDS.get(asset, (0.0, 1.0))
            assert weight >= lo - 0.01, f"{asset} weight {weight:.4f} < lower bound {lo}"
            assert weight <= hi + 0.01, f"{asset} weight {weight:.4f} > upper bound {hi}"

    def test_min_vol_crisis_shifts_to_safe(self, crisis_regime_state, normal_regime_state):
        """Crisis regime should shift weight away from SPY vs normal."""
        opt_crisis = RegimeConstrainedOptimizer(data_dir=crisis_regime_state)
        opt_normal = RegimeConstrainedOptimizer(data_dir=normal_regime_state)
        result_crisis = opt_crisis.optimize(method="min_vol")
        result_normal = opt_normal.optimize(method="min_vol")
        # SPY weight should be lower in crisis
        assert result_crisis.weights["SPY"] <= result_normal.weights["SPY"]

    def test_min_vol_low_vol_shifts_to_equity(self, low_vol_regime_state, normal_regime_state):
        """Low vol regime should have higher SPY weight than normal."""
        opt_low = RegimeConstrainedOptimizer(data_dir=low_vol_regime_state)
        opt_normal = RegimeConstrainedOptimizer(data_dir=normal_regime_state)
        result_low = opt_low.optimize(method="min_vol")
        result_normal = opt_normal.optimize(method="min_vol")
        # SPY weight should be higher or equal in low vol
        assert result_low.weights["SPY"] >= result_normal.weights["SPY"] - 0.02

    def test_max_sharpe_runs(self, normal_regime_state):
        """Max Sharpe optimization should run and produce weights."""
        optimizer = RegimeConstrainedOptimizer(data_dir=normal_regime_state)
        result = optimizer.optimize(method="max_sharpe")
        assert result.method == "max_sharpe"
        assert abs(sum(result.weights.values()) - 1.0) < 0.01

    def test_max_sharpe_higher_return_than_min_vol(self, normal_regime_state):
        """Max Sharpe should produce higher expected return than min vol."""
        optimizer = RegimeConstrainedOptimizer(data_dir=normal_regime_state)
        min_vol_result = optimizer.optimize(method="min_vol")
        sharpe_result = optimizer.optimize(method="max_sharpe")
        # Max Sharpe should target higher expected returns
        assert sharpe_result.expected_return >= min_vol_result.expected_return - 0.01

    def test_risk_parity_runs(self, normal_regime_state):
        """Risk parity optimization should run and produce weights."""
        optimizer = RegimeConstrainedOptimizer(data_dir=normal_regime_state)
        result = optimizer.optimize(method="risk_parity")
        assert result.method == "risk_parity"
        assert abs(sum(result.weights.values()) - 1.0) < 0.01
        # May use entropic RP or fall back to min vol
        assert result.solver_status is not None

    def test_risk_parity_constraints(self, normal_regime_state):
        """Risk parity should respect hard bounds."""
        optimizer = RegimeConstrainedOptimizer(data_dir=normal_regime_state)
        result = optimizer.optimize(method="risk_parity")
        for asset, weight in result.weights.items():
            lo, hi = HARD_BOUNDS.get(asset, (0.0, 1.0))
            assert weight >= lo - 0.01, f"{asset} weight {weight:.4f} < lower bound {lo}"
            assert weight <= hi + 0.01, f"{asset} weight {weight:.4f} > upper bound {hi}"

    def test_solver_status_optimal(self, normal_regime_state):
        """Solver should converge to optimal."""
        optimizer = RegimeConstrainedOptimizer(data_dir=normal_regime_state)
        result = optimizer.optimize(method="min_vol")
        assert result.solver_status in ("optimal", "optimal_inaccurate")

    def test_solver_fast(self, normal_regime_state):
        """Solver should complete in under 500ms."""
        optimizer = RegimeConstrainedOptimizer(data_dir=normal_regime_state)
        result = optimizer.optimize(method="min_vol")
        assert result.solver_time_ms < 500

    def test_unknown_method_falls_back(self, normal_regime_state):
        """Unknown method should fall back to min_vol."""
        optimizer = RegimeConstrainedOptimizer(data_dir=normal_regime_state)
        result = optimizer.optimize(method="invalid_method")
        assert result.method == "min_vol"

    def test_constraints_satisfied_flag(self, normal_regime_state):
        """Constraints satisfied flag should be True for successful solve."""
        optimizer = RegimeConstrainedOptimizer(data_dir=normal_regime_state)
        result = optimizer.optimize(method="min_vol")
        assert result.constraints_satisfied


# ---------------------------------------------------------------------------
# Test: Edge Cases and Robustness
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Test edge cases for the optimizer."""

    def test_no_regime_state_file(self, tmp_path):
        """Missing regime state should default to normal."""
        optimizer = RegimeConstrainedOptimizer(data_dir=tmp_path / "data")
        result = optimizer.optimize(method="min_vol")
        assert result.regime == "normal"
        assert abs(sum(result.weights.values()) - 1.0) < 0.01

    def test_empty_regime_state(self, temp_data_dir):
        """Empty regime state file should default safely."""
        state_path = temp_data_dir / "regime_classifier_state.json"
        state_path.write_text("{}")
        optimizer = RegimeConstrainedOptimizer(data_dir=temp_data_dir)
        result = optimizer.optimize(method="min_vol")
        assert result.regime in ("normal", "unknown")
        assert abs(sum(result.weights.values()) - 1.0) < 0.01

    def test_corrupted_regime_state(self, temp_data_dir):
        """Corrupted regime state should fall back gracefully."""
        state_path = temp_data_dir / "regime_classifier_state.json"
        state_path.write_text("{{{corrupted}}")
        optimizer = RegimeConstrainedOptimizer(data_dir=temp_data_dir)
        result = optimizer.optimize(method="min_vol")
        assert result is not None
        assert abs(sum(result.weights.values()) - 1.0) < 0.01

    def test_regime_prob_sum_to_one(self, normal_regime_state):
        """Regime probabilities should always sum to 1."""
        optimizer = RegimeConstrainedOptimizer(data_dir=normal_regime_state)
        regime_cov = optimizer.build_regime_covariance()
        total = sum(regime_cov.regime_probs.values())
        assert abs(total - 1.0) < 0.01

    def test_result_dataclass_fields(self, normal_regime_state):
        """OptimizerResult should have all expected fields."""
        optimizer = RegimeConstrainedOptimizer(data_dir=normal_regime_state)
        result = optimizer.optimize(method="min_vol")
        assert result.timestamp is not None
        assert result.expected_return is not None
        assert result.expected_volatility > 0
        assert result.solver_status is not None

    def test_state_persistence(self, normal_regime_state):
        """State should persist after optimization."""
        optimizer = RegimeConstrainedOptimizer(data_dir=normal_regime_state)
        result = optimizer.optimize(method="min_vol")
        state_path = normal_regime_state / "regime_optimizer_state.json"
        assert state_path.exists()
        state = json.loads(state_path.read_text())
        assert "weights" in state
        assert "method" in state
        assert state["method"] == "min_vol"
        assert state["current_regime"] == "normal"

    def test_core_assets_weight_distribution(self, normal_regime_state):
        """Core assets (SPY/GLD/TLT) should account for at least 75% of weight."""
        optimizer = RegimeConstrainedOptimizer(data_dir=normal_regime_state)
        result = optimizer.optimize(method="min_vol")
        core_weight = sum(result.weights.get(a, 0) for a in CORE_ASSETS)
        assert core_weight >= 0.75, f"Core weight {core_weight:.2%} < 75% threshold"


# ---------------------------------------------------------------------------
# Test: CLI Integration
# ---------------------------------------------------------------------------


class TestCLI:
    """Test CLI integration (simplified, avoiding sys.exit)."""

    def test_optimize_cli_parses_method(self, monkeypatch, normal_regime_state):
        """CLI should parse --mode argument correctly."""
        import src.strategy.regime_optimizer as ro

        # Monkeypatch data_dir before CLI runs
        original_init = ro.RegimeConstrainedOptimizer.__init__

        def patched_init(self, data_dir=None, risk_free_rate=0.04, estimator="ewma", cost_aversion=0.0, gp_lookback=504):
            original_init(self, data_dir=normal_regime_state, risk_free_rate=risk_free_rate, estimator=estimator, cost_aversion=cost_aversion, gp_lookback=gp_lookback)

        monkeypatch.setattr(ro.RegimeConstrainedOptimizer, "__init__", patched_init)
        monkeypatch.setattr(sys, "argv", ["regime_optimizer.py", "optimize", "--mode", "max_sharpe"])

        # Should not raise
        ro.main()

    def test_status_with_state(self, monkeypatch, normal_regime_state):
        """Status should display state file contents."""
        import src.strategy.regime_optimizer as ro

        # Create a state file first
        optimizer = ro.RegimeConstrainedOptimizer(data_dir=normal_regime_state)
        optimizer.optimize(method="min_vol")

        monkeypatch.setattr(sys, "argv", ["regime_optimizer.py", "status"])
        # Should not raise
        ro.main()

    def test_cov_command(self, monkeypatch, normal_regime_state):
        """Cov command should print covariance details."""
        import src.strategy.regime_optimizer as ro

        original_init = ro.RegimeConstrainedOptimizer.__init__

        def patched_init(self, data_dir=None, risk_free_rate=0.04, estimator="ewma", cost_aversion=0.0, gp_lookback=504):
            original_init(self, data_dir=normal_regime_state, risk_free_rate=risk_free_rate, estimator=estimator, cost_aversion=cost_aversion, gp_lookback=gp_lookback)

        monkeypatch.setattr(ro.RegimeConstrainedOptimizer, "__init__", patched_init)
        monkeypatch.setattr(sys, "argv", ["regime_optimizer.py", "cov"])

        ro.main()

    def test_all_commands(self, monkeypatch, normal_regime_state):
        """All command should run all 3 methods."""
        import src.strategy.regime_optimizer as ro

        original_init = ro.RegimeConstrainedOptimizer.__init__

        def patched_init(self, data_dir=None, risk_free_rate=0.04, estimator="ewma", cost_aversion=0.0, gp_lookback=504):
            original_init(self, data_dir=normal_regime_state, risk_free_rate=risk_free_rate, estimator=estimator, cost_aversion=cost_aversion, gp_lookback=gp_lookback)

        monkeypatch.setattr(ro.RegimeConstrainedOptimizer, "__init__", patched_init)
        monkeypatch.setattr(sys, "argv", ["regime_optimizer.py", "all"])

        ro.main()

    # ── v6.07: Cost-Aware Optimization Tests ─────────────────────────────────

    def test_cost_aware_solve_basic(self, normal_regime_state):
        """Cost-aware optimization should produce valid weights."""
        import src.strategy.regime_optimizer as ro

        opt = ro.RegimeConstrainedOptimizer(
            data_dir=normal_regime_state, cost_aversion=0.01
        )
        result = opt.optimize(method="cost_aware")

        assert result is not None
        assert result.method == "cost_aware"
        assert abs(sum(result.weights.values()) - 1.0) < 0.01
        for asset, w in result.weights.items():
            lo, hi = ro.HARD_BOUNDS.get(asset, (0.0, 1.0))
            assert lo - 0.001 <= w <= hi + 0.001, f"{asset}: {w:.4f} not in [{lo:.2f}, {hi:.2f}]"
        assert result.solver_status in ("optimal", "optimal_inaccurate")

    def test_cost_aware_returns_reasonable(self, normal_regime_state):
        """Cost-aware optimizer should return reasonable expected metrics."""
        import src.strategy.regime_optimizer as ro

        opt = ro.RegimeConstrainedOptimizer(
            data_dir=normal_regime_state, cost_aversion=0.01
        )
        result = opt.optimize(method="cost_aware")
        assert -0.20 <= result.expected_return <= 0.30
        assert 0.02 <= result.expected_volatility <= 0.50

    def test_cost_aware_converges_quickly(self, normal_regime_state):
        """Cost-aware optimization should solve quickly."""
        import src.strategy.regime_optimizer as ro

        opt = ro.RegimeConstrainedOptimizer(
            data_dir=normal_regime_state, cost_aversion=0.01
        )
        result = opt.optimize(method="cost_aware")
        assert result.solver_time_ms < 5000

    def test_cost_aversion_parameter_changes_weights(self, normal_regime_state):
        """Higher cost aversion should keep weights closer to base."""
        import src.strategy.regime_optimizer as ro

        opt_low = ro.RegimeConstrainedOptimizer(
            data_dir=normal_regime_state, cost_aversion=0.001
        )
        opt_high = ro.RegimeConstrainedOptimizer(
            data_dir=normal_regime_state, cost_aversion=0.1
        )
        result_low = opt_low.optimize(method="cost_aware")
        result_high = opt_high.optimize(method="cost_aware")

        base = ro.BASE_ALLOCATION
        dev_low = sum(abs(result_low.weights.get(a, 0) - base.get(a, 0)) for a in base)
        dev_high = sum(abs(result_high.weights.get(a, 0) - base.get(a, 0)) for a in base)
        assert dev_high <= dev_low + 0.01, (
            f"High cost ({dev_high:.4f}) should not deviate more than "
            f"low cost ({dev_low:.4f})"
        )

    def test_cost_aware_vs_min_vol_different(self, normal_regime_state):
        """Cost-aware and min_vol should produce different results."""
        import src.strategy.regime_optimizer as ro

        opt = ro.RegimeConstrainedOptimizer(
            data_dir=normal_regime_state, cost_aversion=0.05
        )
        result_ca = opt.optimize(method="cost_aware")
        result_mv = opt.optimize(method="min_vol")

        ca_w = [result_ca.weights.get(a, 0) for a in ro.ASSETS[:3]]
        mv_w = [result_mv.weights.get(a, 0) for a in ro.ASSETS[:3]]
        diff = sum(abs(c - m) for c, m in zip(ca_w, mv_w))
        assert diff > 0.001, f"cost_aware vs min_vol should differ (diff={diff:.4f})"

    def test_cost_aware_with_real_tca(self):
        """Cost-aware should work with real TCA calibration factors."""
        import src.strategy.regime_optimizer as ro

        opt = ro.RegimeConstrainedOptimizer(cost_aversion=0.01)
        result = opt.optimize(method="cost_aware")
        assert result is not None
        if result.solver_status != "infeasible":
            assert abs(sum(result.weights.values()) - 1.0) < 0.01

    def test_cost_model_smoke(self):
        """AlmgrenChrissCostModel should load and produce estimates."""
        from src.strategy.almgren_chriss_cost import AlmgrenChrissCostModel

        model = AlmgrenChrissCostModel()
        params = model.get_cost_params(["SPY", "GLD", "TLT"])
        assert params.spread["SPY"] > 0
        assert params.impact["SPY"] > 0

        est = model.estimate_turnover_cost(
            {"SPY": 0.46, "GLD": 0.38},
            {"SPY": 0.50, "GLD": 0.34},
        )
        assert est.total_cost_bps > 0
        assert est.active_turnover_pct > 0

    def test_cost_aware_cli_in_all(self, monkeypatch, normal_regime_state):
        """CLI 'all' command should include cost_aware mode without error."""
        import sys
        import src.strategy.regime_optimizer as ro

        original_init = ro.RegimeConstrainedOptimizer.__init__

        def patched_init(self, data_dir=None, risk_free_rate=0.04, estimator="ewma", cost_aversion=0.01, gp_lookback=504):
            original_init(self, data_dir=normal_regime_state,
                          risk_free_rate=risk_free_rate,
                          estimator=estimator,
                          cost_aversion=cost_aversion,
                          gp_lookback=gp_lookback)

        monkeypatch.setattr(ro.RegimeConstrainedOptimizer, "__init__", patched_init)
        monkeypatch.setattr(sys, "argv", ["regime_optimizer.py", "all"])
        ro.main()

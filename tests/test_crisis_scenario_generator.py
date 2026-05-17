#!/usr/bin/env python3
"""Tests for v8.08 Automated Crisis Scenario Generator."""

import json
import os
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.monitor.crisis_scenario_generator import (
    ASSETS,
    BASE_ALLOCATION,
    BASE_CORRELATION,
    CRISIS_TEMPLATES,
    NORMAL_DAILY_VOL,
    CrisisAssessment,
    CrisisTemplate,
    ScenarioOutcome,
    generate_scenarios,
    get_shocked_correlation,
    list_crisis_templates,
    load_latest_assessment,
    run_full_assessment,
    save_assessment,
)


# ─────────────────────────────────────────────
#  Fixtures
# ─────────────────────────────────────────────


@pytest.fixture
def sample_template():
    """Get 2008 financial crisis template."""
    return CRISIS_TEMPLATES["2008_financial"]


@pytest.fixture
def temp_state_dir():
    """Temporary state directory for JSON persistence tests."""
    with tempfile.TemporaryDirectory() as tmp:
        yield tmp


# ─────────────────────────────────────────────
#  CrisisTemplate Tests
# ─────────────────────────────────────────────


class TestCrisisTemplate:
    def test_all_templates_defined(self):
        assert len(CRISIS_TEMPLATES) >= 9, "Should have at least 9 crisis templates"

    def test_template_has_required_fields(self, sample_template):
        assert sample_template.name
        assert sample_template.description
        assert sample_template.date_range
        assert len(sample_template.asset_shocks) == len(ASSETS)
        assert 0 < sample_template.likelihood_weight <= 1
        assert sample_template.severity in ("moderate", "severe", "extreme")
        assert sample_template.regime_type in (
            "correlation_crash", "flight_to_safety", "inflation_shock",
            "liquidity_crisis", "normal"
        )

    def test_asset_shocks_all_assets_present(self, sample_template):
        for asset in ASSETS:
            assert asset in sample_template.asset_shocks
            shock = sample_template.asset_shocks[asset]
            assert "mean_return" in shock
            assert "vol_mult" in shock
            assert shock["vol_mult"] >= 0.5

    def test_likelihood_weights_sum(self):
        total = sum(t.likelihood_weight for t in CRISIS_TEMPLATES.values())
        assert abs(total - 1.0) < 0.01, f"Likelihood weights should sum to ~1.0, got {total}"

    def test_normal_market_has_baseline_vol(self):
        normal = CRISIS_TEMPLATES["normal_market"]
        for asset in ASSETS:
            shock = normal.asset_shocks[asset]
            assert shock["vol_mult"] == 1.0, f"{asset} should have 1.0x vol in normal"

    def test_to_dict_roundtrip(self, sample_template):
        d = sample_template.to_dict()
        assert d["name"] == sample_template.name
        assert d["severity"] == sample_template.severity
        assert "mortgage" in d["description"].lower()


# ─────────────────────────────────────────────
#  Correlation Matrix Tests
# ─────────────────────────────────────────────


class TestGetShockedCorrelation:
    def test_returns_square_matrix(self, sample_template):
        corr = get_shocked_correlation(sample_template)
        assert corr.shape == (7, 7)

    def test_positive_definite(self, sample_template):
        corr = get_shocked_correlation(sample_template)
        # All eigenvalues should be positive (with tolerance)
        eigvals = np.linalg.eigvalsh(corr)
        assert np.all(eigvals > -1e-6), "Correlation matrix must be positive semi-definite"

    def test_diagonal_ones(self, sample_template):
        corr = get_shocked_correlation(sample_template)
        assert np.allclose(np.diag(corr), 1.0)

    def test_symmetric(self, sample_template):
        corr = get_shocked_correlation(sample_template)
        assert np.allclose(corr, corr.T)

    def test_crisis_increases_correlations(self):
        normal = get_shocked_correlation(CRISIS_TEMPLATES["normal_market"])
        crisis = get_shocked_correlation(CRISIS_TEMPLATES["2008_financial"])
        # Average absolute off-diagonal correlation should be higher in crisis
        normal_off_diag = np.abs(normal - np.eye(7)).mean()
        crisis_off_diag = np.abs(crisis - np.eye(7)).mean()
        assert crisis_off_diag >= normal_off_diag * 0.8

    def test_liquidity_crisis_max_correlation(self):
        corr = get_shocked_correlation(CRISIS_TEMPLATES["low_probability_tail"])
        off_diag = np.abs(corr - np.eye(7)).mean()
        assert off_diag > 0.25, "Liquidity crisis should have high correlations"

    def test_recovery_has_lower_correlation(self):
        recovery = get_shocked_correlation(CRISIS_TEMPLATES["2020_recovery"])
        crisis = get_shocked_correlation(CRISIS_TEMPLATES["2008_financial"])
        recovery_off = np.abs(recovery - np.eye(7)).mean()
        crisis_off = np.abs(crisis - np.eye(7)).mean()
        assert recovery_off <= crisis_off * 1.1

    def test_inverse_neg_corrs_less_negative(self):
        """In crisis, negative correlations should move toward zero."""
        normal_corr = get_shocked_correlation(CRISIS_TEMPLATES["normal_market"])
        crisis_corr = get_shocked_correlation(CRISIS_TEMPLATES["low_probability_tail"])
        # SPY-GLD is negative in normal; in crisis it should be less negative
        spy_idx = ASSETS.index("SPY")
        gld_idx = ASSETS.index("GLD")
        normal_spy_gld = normal_corr[spy_idx, gld_idx]
        crisis_spy_gld = crisis_corr[spy_idx, gld_idx]
        if normal_spy_gld < 0:
            assert crisis_spy_gld >= normal_spy_gld, "Negative corrs should become less negative in crisis"

    def test_values_clamped(self, sample_template):
        """All correlation values should be within valid range."""
        corr = get_shocked_correlation(sample_template)
        assert np.all(corr >= -1.0) and np.all(corr <= 1.0)


# ─────────────────────────────────────────────
#  Scenario Generation Tests
# ─────────────────────────────────────────────


class TestGenerateScenarios:
    def test_basic_generation(self, sample_template):
        outcomes, assessment = generate_scenarios(
            template=sample_template, n_scenarios=100, horizon_days=30, seed=42
        )
        assert len(outcomes) == 100
        assert assessment.n_scenarios == 100

    def test_all_outcomes_have_required_fields(self, sample_template):
        outcomes, _ = generate_scenarios(
            template=sample_template, n_scenarios=50, horizon_days=30, seed=42
        )
        for o in outcomes:
            assert o.scenario_name
            assert o.scenario_type
            assert o.severity
            assert o.portfolio_loss_pct is not None
            assert o.equity_drawdown_pct is not None
            assert o.bond_drawdown_pct is not None
            assert o.gold_return_pct is not None
            assert o.crypto_return_pct is not None
            assert o.cvar_impact is not None
            assert o.entropy_impact is not None
            assert o.worst_single_day is not None
            assert o.recovery_days_est > 0

    def test_reproducible_seed(self, sample_template):
        out1, _ = generate_scenarios(
            template=sample_template, n_scenarios=50, horizon_days=30, seed=12345
        )
        out2, _ = generate_scenarios(
            template=sample_template, n_scenarios=50, horizon_days=30, seed=12345
        )
        losses1 = [o.portfolio_loss_pct for o in out1]
        losses2 = [o.portfolio_loss_pct for o in out2]
        assert losses1 == losses2, "Same seed should produce identical results"

    def test_different_seeds(self, sample_template):
        out1, _ = generate_scenarios(
            template=sample_template, n_scenarios=50, horizon_days=30, seed=1
        )
        out2, _ = generate_scenarios(
            template=sample_template, n_scenarios=50, horizon_days=30, seed=999
        )
        losses1 = [o.portfolio_loss_pct for o in out1]
        losses2 = [o.portfolio_loss_pct for o in out2]
        assert losses1 != losses2, "Different seeds should produce different results"

    def test_crisis_larger_than_normal(self):
        crisis_out, _ = generate_scenarios(
            template=CRISIS_TEMPLATES["2008_financial"], n_scenarios=200, horizon_days=30, seed=42
        )
        normal_out, _ = generate_scenarios(
            template=CRISIS_TEMPLATES["normal_market"], n_scenarios=200, horizon_days=30, seed=42
        )
        crisis_median = np.median([o.portfolio_loss_pct for o in crisis_out])
        normal_median = np.median([o.portfolio_loss_pct for o in normal_out])
        assert crisis_median < normal_median, "Crisis scenarios should have worse outcomes"

    def test_portfolio_value_in_assessment(self, sample_template):
        _, assessment = generate_scenarios(
            template=sample_template, n_scenarios=50, horizon_days=30,
            portfolio_value=500000.0, seed=42
        )
        assert assessment.portfolio_value == 500000.0

    def test_custom_weights(self, sample_template):
        weights = {"SPY": 0.6, "GLD": 0.3, "TLT": 0.1}
        _, assessment = generate_scenarios(
            template=sample_template, n_scenarios=50, horizon_days=30,
            weights=weights, seed=42
        )
        assert assessment.portfolio_value == 100000.0

    def test_worst_case_found(self, sample_template):
        outcomes, assessment = generate_scenarios(
            template=sample_template, n_scenarios=100, horizon_days=30, seed=42
        )
        assert assessment.worst_case is not None
        worst_loss = min(o.portfolio_loss_pct for o in outcomes)
        assert assessment.worst_case.portfolio_loss_pct == worst_loss

    def test_expected_shortfall_calculation(self, sample_template):
        outcomes, assessment = generate_scenarios(
            template=sample_template, n_scenarios=200, horizon_days=30, seed=42
        )
        losses = sorted([o.portfolio_loss_pct for o in outcomes])
        es_actual = np.mean(losses[:10])  # worst 5% of 200 = 10
        assert abs(assessment.expected_shortfall - float(es_actual)) < 0.01

    def test_volatilities_reasonable(self, sample_template):
        """Crisis volatilities should be higher than normal."""
        outcomes, _ = generate_scenarios(
            template=sample_template, n_scenarios=200, horizon_days=30, seed=42
        )
        # SPY returns should have higher vol in crisis
        spy_returns = [o.equity_drawdown_pct for o in outcomes]
        spy_vol = np.std(spy_returns)
        assert spy_vol > 1.0, "SPY 30-day returns should be volatile in crisis"

    def test_recovery_days_reasonable(self, sample_template):
        outcomes, _ = generate_scenarios(
            template=sample_template, n_scenarios=100, horizon_days=30, seed=42
        )
        for o in outcomes:
            assert 1 <= o.recovery_days_est <= 5000


# ─────────────────────────────────────────────
#  Full Assessment Tests
# ─────────────────────────────────────────────


class TestRunFullAssessment:
    def test_runs_all_crises(self):
        assessment = run_full_assessment(
            n_scenarios_per_crisis=50, portfolio_value=100000.0, seed=42
        )
        expected = 50 * len(CRISIS_TEMPLATES)
        assert assessment.n_scenarios == expected
        assert assessment.median_loss_pct is not None

    def test_assessment_has_all_fields(self):
        assessment = run_full_assessment(n_scenarios_per_crisis=50, seed=42)
        assert assessment.timestamp
        assert assessment.n_scenarios > 0
        assert assessment.worst_case is not None
        assert assessment.expected_shortfall is not None
        assert assessment.p95_loss_pct is not None
        assert assessment.recommendation

    def test_recommendation_changes_with_severity(self):
        """Better portfolios should have better recommendations."""
        # Very aggressive portfolio
        agg_assessment = run_full_assessment(
            n_scenarios_per_crisis=50, weights={"SPY": 0.9, "GLD": 0.05, "TLT": 0.05}, seed=42
        )
        # Conservative portfolio
        cons_assessment = run_full_assessment(
            n_scenarios_per_crisis=50, weights={"SPY": 0.30, "GLD": 0.20, "TLT": 0.50}, seed=42
        )
        agg_loss = agg_assessment.median_loss_pct
        cons_loss = cons_assessment.median_loss_pct
        assert agg_loss <= cons_loss, "Aggressive portfolio should have worse losses than conservative"

    def test_flight_to_safety_detected(self):
        assessment = run_full_assessment(
            n_scenarios_per_crisis=30,
            weights={"SPY": 0.30, "GLD": 0.20, "TLT": 0.50},
            seed=42
        )
        assert assessment.has_flight_to_safety_buffer is True

    def test_full_assessment_reproducible(self):
        a1 = run_full_assessment(n_scenarios_per_crisis=30, seed=12345)
        a2 = run_full_assessment(n_scenarios_per_crisis=30, seed=12345)
        assert a1.median_loss_pct == a2.median_loss_pct
        assert a1.expected_shortfall == a2.expected_shortfall


# ─────────────────────────────────────────────
#  Persistence Tests
# ─────────────────────────────────────────────


class TestPersistence:
    def test_save_assessment(self, temp_state_dir, sample_template):
        with patch("src.monitor.crisis_scenario_generator.STATE_DIR", Path(temp_state_dir)):
            _, assessment = generate_scenarios(
                template=sample_template, n_scenarios=50, seed=42
            )
            path = save_assessment(assessment)
            assert Path(path).exists()
            with open(path) as f:
                data = json.load(f)
            assert data["n_scenarios"] == 50
            assert data["median_loss_pct"] is not None

    def test_save_and_load_latest(self, temp_state_dir, sample_template):
        with patch("src.monitor.crisis_scenario_generator.STATE_DIR", Path(temp_state_dir)):
            _, assessment = generate_scenarios(
                template=sample_template, n_scenarios=50, seed=42
            )
            save_assessment(assessment)
            loaded = load_latest_assessment()
            assert loaded is not None
            assert loaded["n_scenarios"] == 50
            assert loaded["median_loss_pct"] == assessment.median_loss_pct

    def test_load_no_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch("src.monitor.crisis_scenario_generator.STATE_DIR", Path(tmp)):
                loaded = load_latest_assessment()
                assert loaded is None

    def test_scenario_outcome_to_dict(self):
        outcome = ScenarioOutcome(
            scenario_name="test", scenario_type="correlation_crash", severity="severe",
            portfolio_loss_pct=-5.0, equity_drawdown_pct=-8.0, bond_drawdown_pct=2.0,
            gold_return_pct=1.0, crypto_return_pct=-15.0, cvar_impact=2.5,
            entropy_impact=1.5, recovery_days_est=30, worst_single_day=-3.0
        )
        d = outcome.to_dict()
        assert d["scenario_name"] == "test"
        assert d["portfolio_loss_pct"] == -5.0

    def test_crisis_assessment_to_dict(self, sample_template):
        _, assessment = generate_scenarios(
            template=sample_template, n_scenarios=50, seed=42
        )
        d = assessment.to_dict()
        assert d["n_scenarios"] == 50
        assert d["worst_case"] is not None
        assert "recommendation" in d


# ─────────────────────────────────────────────
#  Utility Tests
# ─────────────────────────────────────────────


class TestUtils:
    def test_list_templates(self):
        names = list_crisis_templates()
        assert len(names) == len(CRISIS_TEMPLATES)
        assert "2008_financial" in names
        assert "normal_market" in names

    def test_all_assets_in_normal_vol(self):
        for asset in ASSETS:
            assert asset in NORMAL_DAILY_VOL
            assert NORMAL_DAILY_VOL[asset] > 0

    def test_base_correlation_complete(self):
        """Every asset pair should have a correlation defined."""
        for i, a in enumerate(ASSETS):
            for b in ASSETS[i+1:]:
                key = (a, b)
                rev_key = (b, a)
                assert key in BASE_CORRELATION or rev_key in BASE_CORRELATION, \
                    f"No correlation defined for {a}-{b}"

    def test_base_allocation_keys(self):
        assert all(k in BASE_ALLOCATION for k in ["SPY", "GLD", "TLT"])
        total = sum(BASE_ALLOCATION.values())
        assert abs(total - 1.0) < 0.01


# ─────────────────────────────────────────────
#  Edge Case Tests
# ─────────────────────────────────────────────


class TestEdgeCases:
    def test_single_scenario(self, sample_template):
        outcomes, assessment = generate_scenarios(
            template=sample_template, n_scenarios=1, horizon_days=5, seed=42
        )
        assert len(outcomes) == 1
        assert assessment.n_scenarios == 1
        assert assessment.worst_case is not None

    def test_zero_horizon(self, sample_template):
        outcomes, assessment = generate_scenarios(
            template=sample_template, n_scenarios=10, horizon_days=1, seed=42
        )
        assert len(outcomes) == 10

    def test_large_n(self, sample_template):
        outcomes, assessment = generate_scenarios(
            template=sample_template, n_scenarios=5000, horizon_days=5, seed=42
        )
        assert len(outcomes) == 5000
        assert assessment.median_loss_pct is not None

    def test_all_crisis_types_different(self):
        """Each crisis should produce meaningfully different outcomes."""
        results = {}
        for name in ["2008_financial", "2020_covid", "2022_rate_hawk", "normal_market"]:
            _, assessment = generate_scenarios(
                template=CRISIS_TEMPLATES[name], n_scenarios=200, horizon_days=30, seed=42
            )
            results[name] = assessment.median_loss_pct
        # All should differ
        assert len(set(round(v, 4) for v in results.values())) >= 3, \
            "At least 3 of 4 crisis types should produce different median losses"

    def test_full_assessment_edge(self):
        """Full assessment with minimal scenarios."""
        assessment = run_full_assessment(n_scenarios_per_crisis=10, seed=42)
        assert assessment.n_scenarios == 10 * len(CRISIS_TEMPLATES)
        assert assessment.worst_case is not None


if __name__ == "__main__":
    pytest.main(["-v", __file__])

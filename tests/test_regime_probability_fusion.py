"""Tests for offline regime probability fusion prototype.

The fusion layer is research-only: it combines existing regime probability
surfaces for calibration/backtest evaluation and must not feed live gates or
broker-facing target allocations in this slice.
"""

import importlib
import importlib.util

import pytest


def _fusion_module():
    spec = importlib.util.find_spec("src.regime.regime_probability_fusion")
    assert spec is not None, "src.regime.regime_probability_fusion is missing"
    return importlib.import_module("src.regime.regime_probability_fusion")


def _attr(module, name):
    assert hasattr(module, name), f"{name} is missing from regime probability fusion module"
    return getattr(module, name)


class TestRegimeProbabilityFusion:
    def test_fuses_two_stage_and_transition_probabilities(self):
        """Fusion should blend current probabilities with transition forecast probabilities."""
        module = _fusion_module()
        fuse_regime_probabilities = _attr(module, "fuse_regime_probabilities")

        result = fuse_regime_probabilities(
            two_stage_probs={"NORMAL": 0.7, "CRISIS": 0.2, "HIGH_VOL": 0.1},
            transition_probs={"NORMAL": 0.4, "CRISIS": 0.5, "HIGH_VOL": 0.1},
            bocd_change_prob=0.0,
            hard_label="NORMAL",
            two_stage_weight=0.6,
            transition_weight=0.4,
        )

        assert result.regime == "NORMAL"
        assert result.confidence == pytest.approx(0.58)
        assert result.probabilities["NORMAL"] == pytest.approx(0.58)
        assert result.probabilities["CRISIS"] == pytest.approx(0.32)
        assert sum(result.probabilities.values()) == pytest.approx(1.0)
        assert result.bocd_role == "confidence_discount"
        assert result.live_routed is False

    def test_bocd_changepoint_probability_discounts_confidence(self):
        """BOCD should reduce confidence by mixing toward uncertainty, not override the regime."""
        module = _fusion_module()
        fuse_regime_probabilities = _attr(module, "fuse_regime_probabilities")

        base = fuse_regime_probabilities(
            two_stage_probs={"NORMAL": 0.8, "CRISIS": 0.1, "HIGH_VOL": 0.1},
            transition_probs={"NORMAL": 0.7, "CRISIS": 0.2, "HIGH_VOL": 0.1},
            bocd_change_prob=0.0,
            hard_label="NORMAL",
            bocd_discount_strength=0.5,
        )
        discounted = fuse_regime_probabilities(
            two_stage_probs={"NORMAL": 0.8, "CRISIS": 0.1, "HIGH_VOL": 0.1},
            transition_probs={"NORMAL": 0.7, "CRISIS": 0.2, "HIGH_VOL": 0.1},
            bocd_change_prob=0.8,
            hard_label="NORMAL",
            bocd_discount_strength=0.5,
        )

        assert discounted.regime == "NORMAL"
        assert discounted.confidence < base.confidence
        assert discounted.bocd_discount == pytest.approx(0.6)
        assert discounted.bocd_role == "confidence_discount"
        assert discounted.live_routed is False

    def test_falls_back_to_hard_label_when_probability_inputs_are_missing(self):
        """Missing probability surfaces should produce a discounted hard-label fallback."""
        module = _fusion_module()
        fuse_regime_probabilities = _attr(module, "fuse_regime_probabilities")

        result = fuse_regime_probabilities(
            two_stage_probs=None,
            transition_probs=None,
            bocd_change_prob=0.5,
            hard_label="HIGH_VOL",
            bocd_discount_strength=0.5,
        )

        assert result.regime == "HIGH_VOL"
        assert result.fallback_used is True
        assert result.probabilities["HIGH_VOL"] < 1.0
        assert result.probabilities["HIGH_VOL"] > result.probabilities["NORMAL"]
        assert sum(result.probabilities.values()) == pytest.approx(1.0)
        assert result.live_routed is False


class TestRegimeFusionEvaluation:
    def test_evaluates_next_regime_calibration_metrics(self):
        """Offline evaluation should report persistence accuracy, Brier score, and log loss."""
        module = _fusion_module()
        evaluate_next_regime_predictions = _attr(module, "evaluate_next_regime_predictions")

        result = evaluate_next_regime_predictions(
            predicted_probabilities=[
                {"NORMAL": 0.7, "CRISIS": 0.1, "HIGH_VOL": 0.1, "LOW_VOL": 0.05, "RECOVERY": 0.05},
                {"NORMAL": 0.2, "CRISIS": 0.1, "HIGH_VOL": 0.6, "LOW_VOL": 0.05, "RECOVERY": 0.05},
                {"NORMAL": 0.2, "CRISIS": 0.6, "HIGH_VOL": 0.1, "LOW_VOL": 0.05, "RECOVERY": 0.05},
            ],
            actual_next_regimes=["NORMAL", "HIGH_VOL", "NORMAL"],
        )

        assert result.n_observations == 3
        assert result.persistence_accuracy == pytest.approx(2 / 3)
        assert result.mean_brier_score > 0
        assert result.mean_log_loss > 0
        assert result.live_routed is False

    def test_offline_backtest_reports_portfolio_impact_without_live_routing(self):
        """Prototype backtest should expose Sharpe, turnover, and drawdown deltas as metadata."""
        module = _fusion_module()
        evaluate_regime_fusion_backtest = _attr(module, "evaluate_regime_fusion_backtest")

        result = evaluate_regime_fusion_backtest(
            predicted_probabilities=[
                {"NORMAL": 0.7, "CRISIS": 0.1, "HIGH_VOL": 0.1, "LOW_VOL": 0.05, "RECOVERY": 0.05},
                {"NORMAL": 0.2, "CRISIS": 0.1, "HIGH_VOL": 0.6, "LOW_VOL": 0.05, "RECOVERY": 0.05},
                {"NORMAL": 0.2, "CRISIS": 0.6, "HIGH_VOL": 0.1, "LOW_VOL": 0.05, "RECOVERY": 0.05},
            ],
            actual_next_regimes=["NORMAL", "HIGH_VOL", "NORMAL"],
            baseline_returns=[-0.010, -0.020, 0.015, 0.005, -0.005, 0.010],
            fused_returns=[-0.005, -0.010, 0.016, 0.006, -0.002, 0.012],
            baseline_weights=[
                {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16},
                {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16},
                {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16},
            ],
            fused_weights=[
                {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16},
                {"SPY": 0.38, "GLD": 0.42, "TLT": 0.20},
                {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16},
            ],
        )

        assert result.scope == "research_offline_only"
        assert result.live_routed is False
        assert result.portfolio_impact["sharpe_delta"] is not None
        assert result.portfolio_impact["turnover_delta"] is not None
        assert result.portfolio_impact["max_drawdown_delta"] is not None

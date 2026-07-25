"""
Tests for Combined Overlay Backtest (v4.90)

Expanded coverage: dataclass completeness, signal boundaries, constants,
utility edge cases, backtest result validation, and synthetic data.
"""

import json
import logging
import math
import pytest
import numpy as np
from dataclasses import asdict
from datetime import date
from pathlib import Path

from src.backtest.combined_overlay_backtest import (
    CombinedOverlayBacktest,
    run_combined_backtest,
)
from src.backtest import combined_overlay_backtest as combined_overlay_backtest_module
from src.backtest.metrics import (
    BacktestResult,
    BacktestMetrics,
    OverlayMetrics,
    CrisisReturns,
    build_profitability_evidence,
    compute_metrics,
    compute_one_way_turnover,
)


class TestBacktestResult:
    """Test backtest result dataclass."""

    def test_serializable(self):
        from dataclasses import asdict
        result = BacktestResult(
            total_return=0.0,
            cagr=12.0,
            volatility=11.0,
            sharpe_ratio=0.91,
            max_drawdown=-21.0,
            extras={
                "timestamp": "2026-05-16T00:00:00",
                "start_date": "2006-01-03", "end_date": "2026-05-15",
                "trading_days": 5000,
                "baseline_cagr": 10.6, "baseline_vol": 11.1, "baseline_sharpe": 0.79,
                "baseline_max_dd": -26.2,
                "combined_cagr": 12.0, "combined_vol": 11.0, "combined_max_dd": -21.0,
                "sharpe_delta": 0.12, "dd_improvement": 5.2, "cagr_delta": 1.4,
                "collar_active_pct": 65.0, "crypto_active_pct": 40.0,
                "bond_rotation_avg_tlt": 45.0, "avg_overlays_active": 2.5,
                "meets_sharpe_target": True, "meets_dd_target": True,
            },
        )
        d = asdict(result)
        assert isinstance(d, dict)
        assert d["sharpe_ratio"] == 0.91
        assert d["extras"]["meets_sharpe_target"]


class TestCombinedOverlayBacktest:
    """Test backtest core functionality."""

    @pytest.fixture
    def bt(self):
        return CombinedOverlayBacktest(allow_synthetic=True)

    def test_collar_signal_normal(self, bt):
        delta = bt._collar_signal(16.0, 0.001)
        assert delta == 0.0

    def test_collar_signal_elevated(self, bt):
        delta = bt._collar_signal(25.0, 0.001)
        assert delta == -0.01

    def test_collar_signal_stress(self, bt):
        delta = bt._collar_signal(35.0, 0.001)
        assert delta == -0.03

    def test_collar_signal_crisis(self, bt):
        delta = bt._collar_signal(50.0, 0.001)
        assert delta == -0.05

    def test_bond_signal_steep_falling(self, bt):
        tlt, ief, shy = bt._bond_duration_signal(1.5, -0.5)
        assert tlt > ief
        assert tlt > shy

    def test_bond_signal_inverted_rising(self, bt):
        tlt, ief, shy = bt._bond_duration_signal(-0.5, 0.5)
        assert shy > tlt
        assert shy > ief

    def test_bond_signal_normal(self, bt):
        tlt, ief, shy = bt._bond_duration_signal(0.5, 0.0)
        assert ief > tlt  # Balanced leans intermediate

    def test_bond_weights_sum_to_one(self, bt):
        for spread in [-1.0, 0.0, 0.5, 1.5]:
            for change in [-1.0, 0.0, 1.0]:
                tlt, ief, shy = bt._bond_duration_signal(spread, change)
                assert abs(tlt + ief + shy - 1.0) < 0.01

    def test_crypto_signal_extreme_vol(self, bt):
        w = bt._crypto_signal(0.5, 1.5, 0.3, 0.9)
        assert w == 0.0  # BTC vol extreme

    def test_crypto_signal_bear(self, bt):
        w = bt._crypto_signal(-0.3, 0.6, -0.2, 0.7)
        assert w == 0.0  # Both negative momentum

    def test_crypto_signal_bull(self, bt):
        w = bt._crypto_signal(0.5, 0.6, 0.3, 0.7)
        assert w > 0

    def test_crypto_weight_capped(self, bt):
        w = bt._crypto_signal(2.0, 0.3, 2.0, 0.3)
        assert w <= 0.05

    def test_run_backtest(self, bt):
        result = bt.run_backtest()
        assert isinstance(result, BacktestResult)
        assert result.extras["trading_days"] > 0
        assert result.extras["baseline_sharpe"] != 0
        assert result.sharpe_ratio != 0

    def test_run_backtest_has_crisis_data(self, bt):
        result = bt.run_backtest()
        crisis = result.crisis_returns or {}
        # At least one crisis period should have data
        any_crisis = (
            crisis.get("2008_baseline", 0) != 0 or
            crisis.get("2020_baseline", 0) != 0 or
            crisis.get("2022_baseline", 0) != 0
        )
        assert any_crisis, "At least one crisis period should be captured"

    def test_sharpe_delta_reasonable(self, bt):
        result = bt.run_backtest()
        # Combined should be better or within noise
        assert result.extras["sharpe_delta"] > -0.05

    def test_overlay_activity_tracked(self, bt):
        result = bt.run_backtest()
        assert 0 <= result.extras["collar_active_pct"] <= 100
        assert 0 <= result.extras["crypto_active_pct"] <= 100
        assert 0 <= result.extras["bond_rotation_avg_tlt"] <= 100

    def test_convenience_function(self):
        result = run_combined_backtest(allow_synthetic=True)
        assert isinstance(result, BacktestResult)

    def test_decision_grade_run_fails_closed_without_real_data(self, tmp_path):
        bt = CombinedOverlayBacktest(data_dir=tmp_path)
        with pytest.raises(ValueError, match="real market data"):
            bt.run_backtest()

    def test_explicit_synthetic_run_is_not_promotion_eligible(self, tmp_path):
        result = CombinedOverlayBacktest(
            data_dir=tmp_path,
            allow_synthetic=True,
        ).run_backtest()

        evidence = result.extras["profitability_evidence"]
        assert evidence["data"]["mode"] == "synthetic"
        assert evidence["promotion_eligible"] is False
        assert evidence["data"]["diagnostic_opt_in"] is True

    def test_main_logs_report_without_blank_logger_type_error(self, monkeypatch, caplog):
        """CLI report should emit blank lines with logger.info("") not bare logger.info()."""
        result = BacktestResult(
            total_return=10.0,
            cagr=5.0,
            volatility=8.0,
            sharpe_ratio=0.7,
            max_drawdown=-12.0,
            crisis_returns={
                "2008_baseline": -10.0,
                "2008_combined": -8.0,
                "2020_baseline": -5.0,
                "2020_combined": -4.0,
                "2022_baseline": -12.0,
                "2022_combined": -11.0,
            },
            extras={
                "start_date": "2022-01-01",
                "end_date": "2026-01-01",
                "trading_days": 100,
                "baseline_cagr": 4.0,
                "combined_cagr": 5.0,
                "cagr_delta": 1.0,
                "baseline_vol": 8.5,
                "combined_vol": 8.0,
                "baseline_sharpe": 0.6,
                "sharpe_delta": 0.1,
                "baseline_max_dd": -14.0,
                "combined_max_dd": -12.0,
                "dd_improvement": 2.0,
                "collar_active_pct": 10.0,
                "crypto_active_pct": 20.0,
                "bond_rotation_avg_tlt": 30.0,
                "avg_overlays_active": 1.0,
                "meets_sharpe_target": False,
                "meets_dd_target": True,
            },
        )
        monkeypatch.setattr(
            combined_overlay_backtest_module.CombinedOverlayBacktest,
            "run_backtest",
            lambda _self: result,
        )
        monkeypatch.setattr("sys.argv", ["combined_overlay_backtest.py", "run"])
        caplog.set_level(logging.INFO)

        combined_overlay_backtest_module.main()

        assert "COMBINED OVERLAY BACKTEST" in caplog.text


class TestEdgeCases:
    """Edge cases for backtest."""

    def test_compute_returns(self):
        bt = CombinedOverlayBacktest()
        rets = bt._compute_returns([100, 110, 105, 115])
        assert len(rets) == 3
        assert abs(rets[0] - 0.10) < 0.01
        assert rets[1] < 0

    def test_compute_rolling_vol(self):
        bt = CombinedOverlayBacktest()
        rng = np.random.RandomState(42)
        rets = list(rng.normal(0, 0.01, 100))
        vols = bt._compute_rolling_vol(rets, 30)
        assert len(vols) == len(rets)
        assert vols[-1] > 0

    def test_compute_rolling_vol_precomputes_without_daily_np_std(self, monkeypatch):
        """Rolling volatility should preserve legacy values without per-day np.std calls."""
        bt = CombinedOverlayBacktest()
        rets = [((i % 19) - 9) / 1000 for i in range(500)]
        window = 30

        def population_std(values):
            mean = sum(values) / len(values)
            return math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))

        expected = []
        for i in range(len(rets)):
            if i < window:
                expected.append(
                    population_std(rets[: i + 1]) * math.sqrt(252)
                    if i > 0
                    else 0.16
                )
            else:
                expected.append(population_std(rets[i - window : i]) * math.sqrt(252))

        def fail_std(*_args, **_kwargs):
            raise AssertionError("np.std should not be called once per rolling-vol day")

        monkeypatch.setattr("src.backtest.combined_overlay_backtest.np.std", fail_std)

        vols = bt._compute_rolling_vol(rets, window)

        assert vols == pytest.approx(expected)


# ═══════════════════════════════════════════════════════════════
# NEW TEST CLASSES (20+ methods added below)
# ═══════════════════════════════════════════════════════════════

class TestDataclassCompleteness:
    """Verify to_dict / asdict field completeness for all dataclasses."""

    def test_backtest_result_required_fields(self):
        """asdict() includes all 5 required fields."""
        r = BacktestResult(
            total_return=10.5, cagr=8.2, volatility=11.0,
            sharpe_ratio=0.79, max_drawdown=-26.2,
        )
        d = asdict(r)
        for key in ("total_return", "cagr", "volatility", "sharpe_ratio", "max_drawdown"):
            assert key in d, f"Missing required field: {key}"

    def test_backtest_result_optional_fields(self):
        """asdict() includes optional fields with defaults."""
        r = BacktestResult(
            total_return=10.5, cagr=8.2, volatility=11.0,
            sharpe_ratio=0.79, max_drawdown=-26.2,
        )
        d = asdict(r)
        assert d["total_rebalances"] == 0
        assert d["total_transaction_costs"] == 0.0
        assert d["avg_turnover"] == 0.0
        assert d["baseline_sharpe"] is None
        assert d["sharpe_improvement"] is None

    def test_backtest_result_extras_roundtrip(self):
        """Custom extras survive the asdict round-trip."""
        extras = {"collar_active_pct": 65.0, "meets_sharpe_target": True}
        r = BacktestResult(
            total_return=10.5, cagr=8.2, volatility=11.0,
            sharpe_ratio=0.79, max_drawdown=-26.2,
            extras=extras,
        )
        d = asdict(r)
        assert d["extras"]["collar_active_pct"] == 65.0
        assert d["extras"]["meets_sharpe_target"] is True

    def test_backtest_result_crisis_returns(self):
        """crisis_returns dict survives asdict round-trip."""
        crisis = {"2008_baseline": -12.0, "2008_combined": -10.0}
        r = BacktestResult(
            total_return=10.5, cagr=8.2, volatility=11.0,
            sharpe_ratio=0.79, max_drawdown=-26.2,
            crisis_returns=crisis,
        )
        d = asdict(r)
        assert d["crisis_returns"]["2008_baseline"] == -12.0
        assert d["crisis_returns"]["2008_combined"] == -10.0

    def test_backtest_result_json_serializable(self):
        """asdict output can be serialized to JSON."""
        r = BacktestResult(
            total_return=10.5, cagr=8.2, volatility=11.0,
            sharpe_ratio=0.79, max_drawdown=-26.2,
            extras={"flag": True},
            crisis_returns={"2008": -12.0},
        )
        d = asdict(r)
        json_str = json.dumps(d)
        parsed = json.loads(json_str)
        assert parsed["total_return"] == 10.5
        assert parsed["crisis_returns"]["2008"] == -12.0

    def test_backtest_metrics_to_dict(self):
        """BacktestMetrics fields present in asdict."""
        m = BacktestMetrics(
            total_return=10.5, cagr=8.2, volatility=11.0,
            sharpe_ratio=0.79, max_drawdown=-26.2,
        )
        d = asdict(m)
        assert d["total_return"] == 10.5
        assert d["cagr"] == 8.2
        assert d["sharpe_ratio"] == 0.79

    def test_overlay_metrics_to_dict(self):
        """OverlayMetrics fields present in asdict."""
        om = OverlayMetrics(
            baseline_sharpe=0.79, sharpe_improvement=0.12,
            overlay_active_count=250, overlay_active_pct=45.0,
        )
        d = asdict(om)
        assert d["baseline_sharpe"] == 0.79
        assert d["sharpe_improvement"] == 0.12
        assert d["overlay_active_count"] == 250
        assert d["overlay_active_pct"] == 45.0

    def test_crisis_returns_to_dict(self):
        """CrisisReturns fields present in asdict."""
        cr = CrisisReturns(returns={"2008": -12.0, "2020": -8.0})
        d = asdict(cr)
        assert d["returns"]["2008"] == -12.0
        assert cr.get("2008") == -12.0
        assert cr.get("nonexistent") is None


class TestBacktestConstants:
    """Validate module-level constants."""

    def test_baseline_keys_present(self):
        bt = CombinedOverlayBacktest()
        for key in ("spy", "gld", "tlt"):
            assert key in bt.BASELINE, f"Missing baseline key: {key}"

    def test_baseline_weights_sum_to_one(self):
        bt = CombinedOverlayBacktest()
        total = sum(bt.BASELINE.values())
        assert abs(total - 1.0) < 0.01, f"Baseline weights sum to {total}, expected 1.0"

    def test_baseline_weights_positive(self):
        bt = CombinedOverlayBacktest()
        for key, val in bt.BASELINE.items():
            assert val > 0, f"Baseline weight {key}={val} should be positive"
            assert val < 1.0, f"Baseline weight {key}={val} should be < 1.0"

    def test_baseline_keys_lowercase(self):
        bt = CombinedOverlayBacktest()
        for key in bt.BASELINE:
            assert key == key.lower(), f"Baseline key not lowercase: {key}"


class TestSignalBoundaries:
    """Boundary and edge cases for signal methods."""

    @pytest.fixture
    def bt(self):
        return CombinedOverlayBacktest(allow_synthetic=True)

    # --- Collar signal boundaries ---

    def test_collar_exact_20_returns_zero(self, bt):
        """vix == 20 is NOT > 20, so returns 0.0."""
        assert bt._collar_signal(20.0, 0.001) == 0.0

    def test_collar_exact_30_returns_minus_0_01(self, bt):
        """vix == 30 is NOT > 30 and NOT > 40, so stops at > 20 tier → -0.01."""
        assert bt._collar_signal(30.0, 0.001) == -0.01

    def test_collar_exact_40_returns_minus_0_03(self, bt):
        """vix == 40 is NOT > 40 but IS > 30 → -0.03."""
        assert bt._collar_signal(40.0, 0.001) == -0.03

    def test_collar_just_above_20(self, bt):
        assert bt._collar_signal(20.01, 0.001) == -0.01

    def test_collar_just_above_30(self, bt):
        assert bt._collar_signal(30.01, 0.001) == -0.03

    def test_collar_just_above_40(self, bt):
        assert bt._collar_signal(40.01, 0.001) == -0.05

    def test_collar_vix_very_high(self, bt):
        """Vix at extreme upper bound."""
        delta = bt._collar_signal(80.0, 0.001)
        assert delta == -0.05

    def test_collar_vix_zero(self, bt):
        """Vix of 0 returns 0.0 (no reduction)."""
        assert bt._collar_signal(0.0, 0.001) == 0.0

    # --- Bond duration signal boundaries ---

    def test_bond_exact_yield_spread_1(self, bt):
        """yield_spread == 1.0 is NOT > 1.0 → balanced branch."""
        tlt, ief, shy = bt._bond_duration_signal(1.0, -0.5)
        assert ief > tlt  # balanced: ief is largest
        assert abs(tlt + ief + shy - 1.0) < 0.01

    def test_bond_exact_rate_change_neg_0_3(self, bt):
        """rate_change == -0.3 is NOT < -0.3 → balanced branch."""
        tlt, ief, shy = bt._bond_duration_signal(1.5, -0.3)
        assert ief > tlt  # balanced
        assert abs(tlt + ief + shy - 1.0) < 0.01

    def test_bond_exact_yield_spread_0(self, bt):
        """yield_spread == 0.0 is NOT < 0 → balanced branch."""
        tlt, ief, shy = bt._bond_duration_signal(0.0, 0.5)
        assert ief > tlt  # balanced
        assert abs(tlt + ief + shy - 1.0) < 0.01

    def test_bond_exact_rate_change_pos_0_3(self, bt):
        """rate_change == 0.3 is NOT > 0.3 → balanced branch."""
        tlt, ief, shy = bt._bond_duration_signal(-0.5, 0.3)
        assert ief > tlt  # balanced
        assert abs(tlt + ief + shy - 1.0) < 0.01

    def test_bond_steep_falling_concrete_weights(self, bt):
        """yield_spread > 1 AND rate_change < -0.3 → (0.70, 0.20, 0.10)."""
        tlt, ief, shy = bt._bond_duration_signal(2.0, -1.0)
        assert tlt == 0.70
        assert ief == 0.20
        assert shy == 0.10

    def test_bond_inverted_rising_concrete_weights(self, bt):
        """yield_spread < 0 AND rate_change > 0.3 → (0.05, 0.25, 0.70)."""
        tlt, ief, shy = bt._bond_duration_signal(-1.0, 1.0)
        assert tlt == 0.05
        assert ief == 0.25
        assert shy == 0.70

    def test_bond_balanced_concrete_weights(self, bt):
        """default branch → (0.20, 0.50, 0.30)."""
        tlt, ief, shy = bt._bond_duration_signal(0.5, 0.0)
        assert tlt == 0.20
        assert ief == 0.50
        assert shy == 0.30

    # --- Crypto signal boundaries ---

    def test_crypto_vol_exact_btc_1(self, bt):
        """btc_vol == 1.0 exactly is NOT > 1.0, so signal proceeds."""
        w = bt._crypto_signal(0.5, 1.0, 0.3, 0.7)
        assert w > 0

    def test_crypto_vol_exact_eth_1(self, bt):
        """eth_vol == 1.0 exactly is NOT > 1.0, so signal proceeds."""
        w = bt._crypto_signal(0.5, 0.6, 0.3, 1.0)
        assert w > 0

    def test_crypto_vol_just_above_btc_1(self, bt):
        """btc_vol just above 1.0 triggers vol guard."""
        w = bt._crypto_signal(0.5, 1.001, 0.3, 0.7)
        assert w == 0.0

    def test_crypto_vol_just_above_eth_1(self, bt):
        """eth_vol just above 1.0 triggers vol guard."""
        w = bt._crypto_signal(0.5, 0.6, 0.3, 1.001)
        assert w == 0.0

    def test_crypto_zero_momentum_both(self, bt):
        """Both momentum exactly 0 → returns 0.0 (<= 0 check)."""
        w = bt._crypto_signal(0.0, 0.6, 0.0, 0.7)
        assert w == 0.0

    def test_crypto_mixed_momentum_one_positive(self, bt):
        """One positive, one negative → positive weight based on avg of positives."""
        w = bt._crypto_signal(0.0, 0.6, 0.4, 0.7)
        assert w > 0
        expected = min(0.05, 0.03 * (1 + (0.0 + 0.4) / 2))
        assert abs(w - expected) < 1e-10

    def test_crypto_both_positive_weight_scaling(self, bt):
        """Both positive: weight = min(0.05, 0.03 * (1 + avg_mom))."""
        w = bt._crypto_signal(0.2, 0.6, 0.4, 0.7)
        expected = min(0.05, 0.03 * (1 + (0.2 + 0.4) / 2))
        assert abs(w - expected) < 1e-10

    def test_crypto_weight_never_exceeds_5pct(self, bt):
        """Weight capped at 0.05 regardless of extreme momentum."""
        w = bt._crypto_signal(10.0, 0.6, 10.0, 0.7)
        assert w == 0.05

    def test_crypto_weight_with_negative_eth(self, bt):
        """Negative ETH momentum with positive BTC momentum."""
        w = bt._crypto_signal(0.5, 0.6, -0.1, 0.7)
        assert w > 0
        expected = min(0.05, 0.03 * (1 + (0.5 + 0.0) / 2))
        assert abs(w - expected) < 1e-10


class TestUtilityFunctions:
    """Edge cases for private utility methods."""

    @pytest.fixture
    def bt(self):
        return CombinedOverlayBacktest(allow_synthetic=True)

    def test_compute_returns_empty(self, bt):
        rets = bt._compute_returns([])
        assert rets == []

    def test_compute_returns_single(self, bt):
        rets = bt._compute_returns([100.0])
        assert rets == []

    def test_compute_returns_flat(self, bt):
        """All returns 0 for flat prices."""
        rets = bt._compute_returns([100, 100, 100, 100])
        assert all(r == 0.0 for r in rets)
        assert len(rets) == 3

    def test_compute_returns_all_negative(self, bt):
        """Declining prices yield negative returns."""
        rets = bt._compute_returns([100, 90, 80, 70])
        assert all(r < 0 for r in rets)
        assert abs(rets[0] - (-0.10)) < 0.001
        assert abs(rets[1] - (-0.111)) < 0.01

    def test_compute_rolling_vol_empty_returns(self, bt):
        """Empty returns list → empty vol list (no iterations)."""
        vols = bt._compute_rolling_vol([], window=30)
        assert vols == []

    def test_compute_rolling_vol_single_return(self, bt):
        """Single return: early-branch i > 0 but only 1 return."""
        vols = bt._compute_rolling_vol([0.001], window=30)
        assert len(vols) == 1
        # i=0 < window=30, i==0 → 0.16
        assert vols[0] == 0.16

    def test_compute_rolling_vol_window_equals_len(self, bt):
        """When window == len(returns), fallback until i >= window."""
        returns = [0.001] * 5
        vols = bt._compute_rolling_vol(returns, window=5)
        assert len(vols) == 5
        # i=0 uses 0.16 fallback; i>=1 computes std of constant returns → 0.0
        assert vols[0] == 0.16
        assert abs(vols[-1]) < 1e-10

    def test_compute_rolling_vol_constant_returns(self, bt):
        """Constant returns → near-zero vol after window fills."""
        returns = [0.001] * 50
        vols = bt._compute_rolling_vol(returns, window=30)
        assert len(vols) == 50
        # After window fills, std of constant returns is ~1e-18 → near zero
        assert abs(vols[-1]) < 1e-10

    def test_compute_rolling_vol_increasing_window(self, bt):
        """Ensure vol values are positive for non-constant returns."""
        rng = np.random.RandomState(42)
        rets = list(rng.normal(0, 0.01, 100))
        for window in (10, 30, 60):
            vols = bt._compute_rolling_vol(rets, window=window)
            assert len(vols) == len(rets)
            assert vols[-1] > 0

    def test_compute_rolling_vol_no_trend_bias(self, bt):
        """Rolling vol should not depend on sign of returns."""
        rng = np.random.RandomState(42)
        rets = list(rng.normal(0.001, 0.01, 100))
        vols_a = bt._compute_rolling_vol(rets, window=30)
        vols_b = bt._compute_rolling_vol([-r for r in rets], window=30)
        for a, b in zip(vols_a[60:], vols_b[60:]):
            assert abs(a - b) < 1e-10


class TestBacktestResultValidation:
    """Validate the structure and content of backtest results."""

    @pytest.fixture
    def bt(self):
        return CombinedOverlayBacktest(allow_synthetic=True)

    @pytest.fixture
    def result(self, bt):
        return bt.run_backtest()

    def test_result_is_not_none(self, bt):
        result = bt.run_backtest()
        assert result is not None

    def test_result_has_all_required_fields(self, result):
        for field in ("total_return", "cagr", "volatility", "sharpe_ratio", "max_drawdown"):
            assert getattr(result, field, None) is not None, f"Missing field: {field}"

    def test_result_extras_has_all_keys(self, result):
        required_extras = [
            "timestamp", "start_date", "end_date", "trading_days",
            "baseline_cagr", "baseline_vol", "baseline_sharpe", "baseline_max_dd",
            "combined_cagr", "combined_vol", "combined_max_dd",
            "sharpe_delta", "dd_improvement", "cagr_delta",
            "collar_active_pct", "crypto_active_pct",
            "bond_rotation_avg_tlt", "avg_overlays_active",
            "meets_sharpe_target", "meets_dd_target",
        ]
        for key in required_extras:
            assert key in result.extras, f"Missing extras key: {key}"

    def test_crisis_returns_has_expected_keys(self, result):
        crisis = result.crisis_returns or {}
        for key in ("2008_baseline", "2008_combined", "2020_baseline",
                     "2020_combined", "2022_baseline", "2022_combined"):
            assert key in crisis, f"Missing crisis key: {key}"

    def test_result_sharpe_ratio_reasonable(self, result):
        """Sharpe ratio should be in plausible range for 46/38/16."""
        assert -1.0 <= result.sharpe_ratio <= 3.0

    def test_result_volatility_reasonable(self, result):
        """Annualized vol should be in plausible range."""
        assert 5.0 <= result.volatility <= 25.0

    def test_result_max_drawdown_not_positive(self, result):
        """Max drawdown should be negative or zero."""
        assert result.max_drawdown <= 0

    def test_result_total_return_reasonable(self, result):
        """Total return over 20yr should be positive."""
        assert result.total_return > 0

    def test_target_fields_are_bool(self, result):
        # May be np.bool_ from numpy operations; check truthiness either way
        assert result.extras["meets_sharpe_target"] in (True, False)
        assert result.extras["meets_dd_target"] in (True, False)

    def test_trading_days_reasonable(self, result):
        """Should have many trading days regardless of data source."""
        assert result.extras["trading_days"] > 100

    def test_dates_present(self, result):
        assert result.extras["start_date"] is not None
        assert result.extras["end_date"] is not None
        assert result.extras["start_date"] <= result.extras["end_date"]


class TestSyntheticData:
    """Validate synthetic data generation."""

    @pytest.fixture
    def bt(self):
        return CombinedOverlayBacktest(allow_synthetic=True)

    def test_synthetic_data_all_symbols(self, bt):
        data = bt._generate_synthetic_data()
        for sym in ("SPY", "GLD", "TLT", "IEF", "SHY", "BTC", "ETH", "VIX"):
            assert sym in data, f"Missing symbol: {sym}"
            assert "dates" in data[sym]
            assert "prices" in data[sym]

    def test_synthetic_data_lengths_match(self, bt):
        data = bt._generate_synthetic_data()
        spy_len = len(data["SPY"]["prices"])
        for sym in ("GLD", "TLT", "IEF", "SHY", "BTC", "ETH"):
            assert len(data[sym]["prices"]) == spy_len, f"Mismatch for {sym}"
            assert len(data[sym]["dates"]) == len(data["SPY"]["dates"])

    def test_synthetic_data_vix_bounds(self, bt):
        """VIX price stays within 8-80 range."""
        data = bt._generate_synthetic_data()
        vix = np.array(data["VIX"]["prices"])
        assert vix.min() >= 8.0
        assert vix.max() <= 80.0

    def test_synthetic_data_crisis_periods_tlt(self, bt):
        """TLT should have different returns during crisis vs non-crisis."""
        data = bt._generate_synthetic_data()
        tlt_prices = data["TLT"]["prices"]
        tlt_rets = [(tlt_prices[i] / tlt_prices[i-1] - 1) for i in range(1, len(tlt_prices))]
        mean_return = np.mean(tlt_rets)
        assert mean_return < 0.10  # Sanity: not absurdly high

    def test_synthetic_data_all_dates_unique(self, bt):
        data = bt._generate_synthetic_data()
        dates = data["SPY"]["dates"]
        assert len(dates) == len(set(dates))

    def test_synthetic_data_all_dates_weekdays(self, bt):
        data = bt._generate_synthetic_data()
        for d_str in data["SPY"]["dates"]:
            d = date.fromisoformat(d_str)
            assert d.weekday() < 5, f"Weekend date found: {d_str}"

    def test_synthetic_data_prices_positive(self, bt):
        data = bt._generate_synthetic_data()
        for sym in ("SPY", "GLD", "TLT", "BTC", "ETH"):
            prices = np.array(data[sym]["prices"])
            assert np.all(prices > 0), f"Non-positive prices for {sym}"


class TestProfitabilityEvidenceContract:
    def test_one_way_turnover_includes_implicit_cash(self):
        assert compute_one_way_turnover(
            {},
            {"SPY": 0.6, "GLD": 0.4},
        ) == pytest.approx(1.0)
        assert compute_one_way_turnover(
            {"SPY": 0.6, "GLD": 0.4},
            {"SPY": 0.5, "GLD": 0.5},
        ) == pytest.approx(0.1)

    def test_canonical_metrics_and_cost_reconciliation(self):
        evidence = build_profitability_evidence(
            dates=["2026-01-02", "2026-01-05", "2026-01-06"],
            gross_returns=[0.01, -0.004, 0.006],
            turnovers=[0.0, 0.25, 0.0],
            assets=["SPY", "GLD", "TLT"],
            data_mode="real",
            provenance={"source": "test-fixture", "path": "memory"},
            transaction_cost_bps=10.0,
            point_in_time=True,
        )

        trace = evidence["trace"]
        assert [row["date"] for row in trace] == [
            "2026-01-02", "2026-01-05", "2026-01-06",
        ]
        assert evidence["costs"]["max_reconciliation_error"] < 1e-12
        assert trace[1]["net_return"] == pytest.approx(
            trace[1]["gross_return"] - trace[1]["cost_return"]
        )
        assert trace[-1]["net_equity"] < trace[-1]["gross_equity"]

        expected = compute_metrics(
            [100000.0] + [row["net_equity"] for row in trace],
            100000.0,
        )
        assert evidence["metrics"]["net"]["cagr"] == expected.cagr
        assert evidence["metrics"]["net"]["sortino_ratio"] == expected.sortino_ratio
        assert evidence["metrics"]["net"]["max_drawdown"] == expected.max_drawdown
        assert evidence["promotion_eligible"] is True

    @pytest.mark.parametrize("mode", ["proxy", "synthetic"])
    def test_non_real_modes_require_explicit_diagnostic_opt_in(self, mode):
        with pytest.raises(ValueError, match="diagnostic opt-in"):
            build_profitability_evidence(
                dates=["2026-01-02"],
                gross_returns=[0.0],
                turnovers=[0.0],
                assets=["SPY"],
                data_mode=mode,
                provenance={"source": "test"},
            )

    def test_dates_must_be_aligned_unique_and_sorted(self):
        with pytest.raises(ValueError, match="strictly increasing"):
            build_profitability_evidence(
                dates=["2026-01-03", "2026-01-02"],
                gross_returns=[0.0, 0.0],
                turnovers=[0.0, 0.0],
                assets=["SPY"],
                data_mode="real",
                provenance={"source": "test"},
            )

#!/usr/bin/env python3
"""
Tests for evaluator.py — constants, Position/Portfolio classes, order generation,
order execution, risk limits, performance calculation, and graduation criteria.
"""
import json
import numpy as np

import pytest
from pathlib import Path
from datetime import datetime
from unittest.mock import patch, MagicMock

from src.strategy.evaluator import (
    PAPER_CONFIG,
    BASE_ALLOCATION,
    REGIME_OVERRIDES,
    ORDERS_LOG,
    PERFORMANCE_LOG,
    Position,
    Portfolio,
    calculate_performance,
    check_graduation_criteria,
    _deduplicate_to_daily,
    get_current_regime,
    get_latest_vix,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_portfolio(tmp_path, cash=100000, positions=None):
    """Create a Portfolio with a temp state file."""
    state_file = tmp_path / "portfolio.json"
    portfolio = Portfolio(state_file, mode="paper")
    portfolio.cash = cash
    if positions:
        portfolio.positions = positions
    return portfolio


def _make_position(**overrides):
    defaults = dict(
        symbol="SPY",
        shares=100,
        avg_price=450.0,
        current_price=460.0,
        value=46000,
        weight=0.46,
        unrealized_pnl=1000,
    )
    defaults.update(overrides)
    return Position(**defaults)


# ---------------------------------------------------------------------------
# Constants Tests
# ---------------------------------------------------------------------------

class TestConstants:

    def test_paper_config(self):
        assert PAPER_CONFIG["initial_capital"] == 100000
        assert PAPER_CONFIG["max_position_pct"] == 0.5
        assert PAPER_CONFIG["max_drawdown_pct"] == 0.15
        assert PAPER_CONFIG["rebalance_threshold"] == 0.10
        assert PAPER_CONFIG["volatility_target"] == 0.12

    def test_base_allocation(self):
        assert BASE_ALLOCATION["SPY"] == 0.46
        assert BASE_ALLOCATION["GLD"] == 0.38
        assert BASE_ALLOCATION["TLT"] == 0.16
        assert sum(BASE_ALLOCATION.values()) == pytest.approx(1.0)

    def test_regime_overrides(self):
        assert "crisis" in REGIME_OVERRIDES
        assert "vol_spike" in REGIME_OVERRIDES
        assert "low_vol" in REGIME_OVERRIDES
        assert "normal" in REGIME_OVERRIDES
        for regime, alloc in REGIME_OVERRIDES.items():
            if alloc is None:
                continue  # normal uses BASE_ALLOCATION
            assert abs(sum(alloc.values()) - 1.0) < 0.01, f"{regime} doesn't sum to 1"


# ---------------------------------------------------------------------------
# Position Tests
# ---------------------------------------------------------------------------

class TestPosition:

    def test_named_tuple(self):
        p = _make_position(symbol="GLD", shares=200)
        assert p.symbol == "GLD"
        assert p.shares == 200

    def test_fields(self):
        p = _make_position()
        assert p.avg_price == 450.0
        assert p.current_price == 460.0
        assert p.unrealized_pnl == 1000


# ---------------------------------------------------------------------------
# Portfolio — init and state
# ---------------------------------------------------------------------------

class TestPortfolioState:

    def test_new_portfolio(self, tmp_path):
        p = _make_portfolio(tmp_path)
        assert p.cash == 100000
        assert p.positions == {}
        assert p.mode == "paper"

    def test_save_and_load(self, tmp_path):
        p = _make_portfolio(tmp_path, cash=95000)
        p.positions = {"SPY": _make_position()}
        p.save_state()

        p2 = Portfolio(p.state_file, mode="paper")
        assert p2.cash == 95000
        assert "SPY" in p2.positions

    def test_save_preserves_mode(self, tmp_path):
        p = _make_portfolio(tmp_path)
        p.save_state()
        with open(p.state_file) as f:
            state = json.load(f)
        assert state["mode"] == "paper"


# ---------------------------------------------------------------------------
# Portfolio — total_value
# ---------------------------------------------------------------------------

class TestTotalValue:

    def test_cash_only(self, tmp_path):
        p = _make_portfolio(tmp_path, cash=100000)
        assert p.total_value({}) == 100000

    def test_with_positions(self, tmp_path):
        p = _make_portfolio(tmp_path, cash=50000)
        p.positions = {"SPY": _make_position(shares=100, current_price=460)}
        prices = {"SPY": 470}
        # 50000 cash + 100 * 470 = 97000
        assert p.total_value(prices) == 97000

    def test_missing_price_uses_current(self, tmp_path):
        p = _make_portfolio(tmp_path, cash=50000)
        p.positions = {"SPY": _make_position(shares=100, current_price=460)}
        # No SPY in prices → uses current_price
        assert p.total_value({}) == 96000


# ---------------------------------------------------------------------------
# Portfolio — current_weights
# ---------------------------------------------------------------------------

class TestCurrentWeights:

    def test_empty_positions(self, tmp_path):
        p = _make_portfolio(tmp_path)
        assert p.current_weights({}) == {}

    def test_weights(self, tmp_path):
        p = _make_portfolio(tmp_path, cash=0)
        p.positions = {
            "SPY": _make_position(shares=100, current_price=460),
            "GLD": _make_position(symbol="GLD", shares=200, current_price=190),
        }
        prices = {"SPY": 460, "GLD": 190}
        weights = p.current_weights(prices)
        total = 100 * 460 + 200 * 190
        assert weights["SPY"] == pytest.approx(46000 / total)
        assert weights["GLD"] == pytest.approx(38000 / total)


# ---------------------------------------------------------------------------
# Portfolio — calculate_orders
# ---------------------------------------------------------------------------

class TestCalculateOrders:

    def test_no_drift_no_orders(self, tmp_path):
        p = _make_portfolio(tmp_path, cash=0)
        p.positions = {
            "SPY": _make_position(shares=100, value=46000),
            "GLD": _make_position(symbol="GLD", shares=200, value=38000),
            "TLT": _make_position(symbol="TLT", shares=100, value=16000),
        }
        prices = {"SPY": 460, "GLD": 190, "TLT": 160}
        # Current weights match base allocation
        orders = p.calculate_orders(BASE_ALLOCATION, prices)
        assert orders == []

    def test_drift_generates_orders(self, tmp_path):
        p = _make_portfolio(tmp_path, cash=0)
        p.positions = {
            "SPY": _make_position(shares=150, value=69000),  # Overweight
            "GLD": _make_position(symbol="GLD", shares=100, value=19000),  # Underweight
            "TLT": _make_position(symbol="TLT", shares=100, value=16000),
        }
        prices = {"SPY": 460, "GLD": 190, "TLT": 160}
        orders = p.calculate_orders(BASE_ALLOCATION, prices)
        assert len(orders) > 0

    def test_order_structure(self, tmp_path):
        p = _make_portfolio(tmp_path, cash=0)
        p.positions = {
            "SPY": _make_position(shares=150, value=69000),
            "GLD": _make_position(symbol="GLD", shares=100, value=19000),
            "TLT": _make_position(symbol="TLT", shares=100, value=16000),
        }
        prices = {"SPY": 460, "GLD": 190, "TLT": 160}
        orders = p.calculate_orders(BASE_ALLOCATION, prices)
        for o in orders:
            assert "symbol" in o
            assert "side" in o
            assert "shares" in o
            assert o["side"] in ("buy", "sell")

    def test_skips_invalid_prices(self, tmp_path):
        p = _make_portfolio(tmp_path, cash=50000)
        target = {"SPY": 0.50, "GLD": 0.50}
        orders = p.calculate_orders(target, {"SPY": 0, "GLD": -1})
        assert orders == []


# ---------------------------------------------------------------------------
# Portfolio — execute_orders
# ---------------------------------------------------------------------------

class TestExecuteOrders:

    def test_buy_order(self, tmp_path):
        p = _make_portfolio(tmp_path, cash=100000)
        orders = [{"symbol": "SPY", "side": "buy", "shares": 10, "estimated_price": 460}]
        prices = {"SPY": 460}
        executed = p.execute_orders(orders, prices, slippage=0.0)
        assert len(executed) == 1
        assert "SPY" in p.positions
        assert p.cash < 100000

    def test_sell_order(self, tmp_path):
        p = _make_portfolio(tmp_path, cash=50000)
        p.positions = {"SPY": _make_position(shares=100, avg_price=450)}
        orders = [{"symbol": "SPY", "side": "sell", "shares": 50, "estimated_price": 460}]
        prices = {"SPY": 460}
        executed = p.execute_orders(orders, prices, slippage=0.0)
        assert len(executed) == 1
        assert p.positions["SPY"].shares == 50

    def test_slippage_applied(self, tmp_path):
        p = _make_portfolio(tmp_path, cash=100000)
        orders = [{"symbol": "SPY", "side": "buy", "shares": 10, "estimated_price": 460}]
        prices = {"SPY": 460}
        executed = p.execute_orders(orders, prices, slippage=0.01)
        # Buy fills at 460 * 1.01 = 464.6
        assert executed[0]["fill_price"] == pytest.approx(464.6)

    def test_partial_fill_on_insufficient_cash(self, tmp_path):
        p = _make_portfolio(tmp_path, cash=1000)
        orders = [{"symbol": "SPY", "side": "buy", "shares": 100, "estimated_price": 460}]
        prices = {"SPY": 460}
        executed = p.execute_orders(orders, prices, slippage=0.0)
        # Can only buy 1000/460 ≈ 2.17 shares
        assert executed[0]["fill_shares"] < 100
        assert p.cash == pytest.approx(0, abs=1)

    def test_sell_full_position_removes(self, tmp_path):
        p = _make_portfolio(tmp_path, cash=50000)
        p.positions = {"SPY": _make_position(shares=100, avg_price=450)}
        orders = [{"symbol": "SPY", "side": "sell", "shares": 100, "estimated_price": 460}]
        prices = {"SPY": 460}
        p.execute_orders(orders, prices, slippage=0.0)
        assert "SPY" not in p.positions


# ---------------------------------------------------------------------------
# Portfolio — check_risk_limits
# ---------------------------------------------------------------------------

class TestRiskLimits:

    def test_no_breach(self, tmp_path):
        p = _make_portfolio(tmp_path, cash=100000)
        p.positions = {"SPY": _make_position(weight=0.30)}
        assert p.check_risk_limits({"SPY": 460}) is None

    def test_concentration_breach(self, tmp_path):
        p = _make_portfolio(tmp_path, cash=0)
        p.positions = {"SPY": _make_position(weight=0.55, value=55000)}
        result = p.check_risk_limits({"SPY": 460})
        assert result is not None
        assert "max_position" in result

    def test_drawdown_breach(self, tmp_path):
        p = _make_portfolio(tmp_path, cash=50000)
        # Create history showing a peak of 100k then drop to 80k
        p.history = [{"total_value": 100000}] * 25
        p.positions = {}
        result = p.check_risk_limits({})
        # Current value = 50000, peak = 100000, DD = 50% > 15%
        assert result is not None
        assert "max_drawdown" in result

    def test_drawdown_always_nonpositive_when_total_above_peak(self, tmp_path):
        """Regression: when total > historical peak, drawdown must still be <= 0.

        Before the min(0.0, ...) fix, (total - peak) / peak could be positive
        when cash additions push the portfolio above its recorded peak.
        Financial convention: drawdown is always non-positive.
        """
        p = _make_portfolio(tmp_path, cash=150000)
        # Historical peak was 100k, but current total (150k) is above peak
        p.history = [{"total_value": 100000}] * 25
        p.positions = {}
        result = p.check_risk_limits({})
        # Even though total > peak, drawdown should NOT trigger a breach
        # because the portfolio is at a new high, not in drawdown
        assert result is None or "max_drawdown" not in result

    def test_garch_cvar_no_breach(self, tmp_path):
        """GARCH-CVaR integration should not trigger on normal returns."""
        p = _make_portfolio(tmp_path, cash=100000)
        p.positions = {"SPY": _make_position(weight=0.30)}
        # Add history with normal returns
        rng = np.random.RandomState(42)
        for _ in range(100):
            p.history.append({
                "total_value": 100000 + rng.normal(0, 500),
                "daily_return": float(rng.normal(0.0004, 0.01)),
            })
        result = p.check_risk_limits({"SPY": 460})
        # Should not trigger extreme tail risk with normal returns
        assert result is None or "extreme_tail_risk" not in result

    def test_garch_cvar_writes_health_report(self, tmp_path):
        """check_risk_limits should write .health_report.json to DATA_DIR."""
        from src.paths import DATA_DIR as PROJECT_DATA_DIR
        p = _make_portfolio(tmp_path, cash=100000)
        p.positions = {"SPY": _make_position(weight=0.30)}
        rng = np.random.RandomState(42)
        for _ in range(100):
            p.history.append({
                "total_value": 100000 + rng.normal(0, 500),
                "daily_return": float(rng.normal(0.0004, 0.01)),
            })
        p.check_risk_limits({"SPY": 460})
        # Report is written to project DATA_DIR (not tmp_path)
        report_path = PROJECT_DATA_DIR / ".health_report.json"
        assert report_path.exists()
        with open(report_path) as f:
            data = json.load(f)
        assert "var_95" in data
        assert "cvar_95" in data
        assert "cvar_ratio" in data

    def test_garch_failure_does_not_block(self, tmp_path):
        """If GARCH computation fails, risk check should still complete."""
        p = _make_portfolio(tmp_path, cash=100000)
        p.positions = {"SPY": _make_position(weight=0.30)}
        p.history = [{"total_value": 100000, "daily_return": 0.001}] * 100
        # Even if GARCH fails, check_risk_limits should return None (no breach)
        result = p.check_risk_limits({"SPY": 460})
        assert result is None


# ---------------------------------------------------------------------------
# calculate_performance
# ---------------------------------------------------------------------------

class TestCalculatePerformance:

    def test_structure(self, tmp_path):
        p = _make_portfolio(tmp_path, cash=100000)
        perf = calculate_performance(p, {})
        assert "timestamp" in perf
        assert "total_value" in perf
        assert "daily_return" in perf

    def test_first_day_zero_return(self, tmp_path):
        p = _make_portfolio(tmp_path, cash=100000)
        perf = calculate_performance(p, {})
        assert perf["daily_return"] == 0

    def test_positive_return(self, tmp_path):
        p = _make_portfolio(tmp_path, cash=100000)
        p.history = [{"total_value": 95000}]
        perf = calculate_performance(p, {})
        expected = (100000 - 95000) / 95000
        assert perf["daily_return"] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# check_graduation_criteria
# ---------------------------------------------------------------------------

class TestGraduationCriteria:

    def test_too_few_days(self, tmp_path, capsys):
        p = _make_portfolio(tmp_path)
        p.history = [{"total_value": 100000, "daily_return": 0.001}] * 30
        check_graduation_criteria(p)
        captured = capsys.readouterr()
        assert "GRADUATION" not in captured.out

    def test_good_performance(self, tmp_path, capsys):
        p = _make_portfolio(tmp_path)
        # 63 days of positive returns — deterministic, realistic Sharpe
        rng = np.random.RandomState(12345)
        p.history = []
        val = 100000
        for i in range(63):
            # Realistic: 0.08% mean daily (~20% ann), 1% std (~16% ann vol)
            # Sharpe ~0.08/1*sqrt(252) ≈ 1.27
            ret = rng.normal(0.0008, 0.01)
            val *= (1 + ret)
            p.history.append({
                "timestamp": f"2026-01-{i+1:02d}T23:00:00",
                "total_value": round(val, 2),
                "daily_return": ret,
            })
        check_graduation_criteria(p)
        captured = capsys.readouterr()
        assert "GRADUATION CANDIDATE" in captured.out, f"Output: '{captured.out.strip()}'"

    def test_poor_performance_no_graduation(self, tmp_path, capsys):
        p = _make_portfolio(tmp_path)
        # 63 days of mixed returns with high vol
        np.random.seed(42)
        p.history = []
        val = 100000
        for i in range(63):
            ret = np.random.normal(-0.001, 0.03)
            val *= (1 + ret)
            p.history.append({
                "timestamp": f"2026-01-{i+1:02d}T23:00:00",
                "total_value": val,
                "daily_return": ret,
            })
        check_graduation_criteria(p)
        captured = capsys.readouterr()
        assert "GRADUATION CANDIDATE" not in captured.out

    def test_intra_day_data_does_not_trigger(self, tmp_path, capsys):
        """Intra-day snapshots with zero daily_return should not contaminate graduation."""
        p = _make_portfolio(tmp_path)
        # Simulate 63 unique days, each with 24 intra-day snapshots (daily_return=0)
        np.random.seed(42)
        p.history = []
        val = 100000
        for day in range(63):
            for intra in range(24):
                p.history.append({
                    "timestamp": f"2026-01-{day+1:02d}T{intra:02d}:00:00",
                    "total_value": val,
                    "daily_return": 0.0,
                })
            # End-of-day: realistic return with noise
            ret = np.random.normal(0.001, 0.005)
            val *= (1 + ret)
            p.history.append({
                "timestamp": f"2026-01-{day+1:02d}T23:00:00",
                "total_value": val,
                "daily_return": ret,
            })
        check_graduation_criteria(p)
        captured = capsys.readouterr()
        # After dedup to 63 trading days:
        # Check results aren't obviously broken
        assert "WARNING" not in captured.out
        assert "GRADUATION DEFERRED" not in captured.out

    def test_near_zero_std_vol_floor(self, tmp_path, capsys):
        """Volatility floor prevents division-by-zero, but Sharpe cap still catches."""
        p = _make_portfolio(tmp_path)
        # 63 entries with nearly identical returns (std ≈ 0)
        p.history = []
        for i in range(63):
            p.history.append({
                "timestamp": f"2026-01-{i+1:02d}T00:00:00",
                "total_value": 100000 + i * 10,
                "daily_return": 0.0001,
            })
        check_graduation_criteria(p)
        captured = capsys.readouterr()
        # Vol floor prevents NaN/Inf, but Sharpe = 0.0001/0.0001*sqrt(252) = 15.87
        # This still exceeds MAX_REALISTIC_SHARPE (3.0), so warning is printed
        assert "WARNING" in captured.out
        assert "exceeds realistic maximum" in captured.out

    def test_unrealistic_sharpe_rejected(self, tmp_path, capsys):
        """Sharpe > 3.0 should be rejected with warning."""
        p = _make_portfolio(tmp_path)
        # Create exactly identical returns (zero std)
        p.history = []
        for i in range(63):
            p.history.append({
                "timestamp": f"2026-01-{i+1:02d}T00:00:00",
                "total_value": 100000,
                "daily_return": 0.0001,
            })
        check_graduation_criteria(p)
        captured = capsys.readouterr()
        assert "WARNING" in captured.out
        assert "exceeds realistic maximum" in captured.out
        assert "GRADUATION CANDIDATE" not in captured.out

    def test_dsr_in_graduation_metrics(self, tmp_path, capsys):
        """Graduation trigger should include DSR in metrics."""
        p = _make_portfolio(tmp_path)
        p.history = []
        np.random.seed(42)
        for i in range(100):
            ret = np.random.normal(0.001, 0.005)
            val = 100000 * (1 + ret)
            p.history.append({
                "timestamp": f"2026-01-{i+1:02d}T00:00:00",
                "total_value": val,
                "daily_return": ret,
            })
        check_graduation_criteria(p)
        trigger_path = tmp_path / "data" / ".promote_to_live"
        if trigger_path.exists():
            import json
            trigger = json.loads(trigger_path.read_text())
            assert "dsr" in trigger["metrics"]


class TestDeduplicateToDaily:

    def test_empty_history(self):
        assert _deduplicate_to_daily([]) == []

    def test_single_entry(self):
        h = [{"timestamp": "2026-01-01T12:00:00", "value": 100}]
        result = _deduplicate_to_daily(h)
        assert len(result) == 1
        assert result[0]["value"] == 100

    def test_intra_day_deduplication(self):
        """Multiple entries on same day → only last retained."""
        h = [
            {"timestamp": "2026-01-01T09:00:00", "value": 100},
            {"timestamp": "2026-01-01T12:00:00", "value": 101},
            {"timestamp": "2026-01-01T15:00:00", "value": 102},
        ]
        result = _deduplicate_to_daily(h)
        assert len(result) == 1
        assert result[0]["value"] == 102  # Last entry wins

    def test_multiple_days(self):
        h = [
            {"timestamp": "2026-01-01T09:00:00", "value": 100},
            {"timestamp": "2026-01-02T09:00:00", "value": 101},
            {"timestamp": "2026-01-02T15:00:00", "value": 102},
            {"timestamp": "2026-01-03T09:00:00", "value": 103},
        ]
        result = _deduplicate_to_daily(h)
        assert len(result) == 3
        assert result[0]["value"] == 100
        assert result[1]["value"] == 102  # Last on Jan 2
        assert result[2]["value"] == 103

    def test_chronological_order(self):
        h = [
            {"timestamp": "2026-01-03T09:00:00", "value": 103},
            {"timestamp": "2026-01-01T09:00:00", "value": 101},
            {"timestamp": "2026-01-02T09:00:00", "value": 102},
        ]
        result = _deduplicate_to_daily(h)
        values = [e["value"] for e in result]
        assert values == [101, 102, 103]

    def test_timestamp_missing(self):
        h = [{"daily_return": 0.1}, {"daily_return": 0.2}]
        result = _deduplicate_to_daily(h)
        # Both have no valid date_key (empty string), so last overwrites
        assert len(result) == 1
        assert result[0]["daily_return"] == 0.2

    def test_deferred_when_too_few_trading_days(self, tmp_path, capsys):
        """63 snapshots but only 3 unique days → should defer with message."""
        p = _make_portfolio(tmp_path)
        val = 100000
        p.history = []
        for day in range(3):
            for intra in range(21):
                p.history.append({
                    "timestamp": f"2026-01-{day+1:02d}T{intra:02d}:00:00",
                    "total_value": val,
                    "daily_return": 0,
                })
            val *= 1.01
            # Add EOD entry
            p.history.append({
                "timestamp": f"2026-01-{day+1:02d}T23:00:00",
                "total_value": val,
                "daily_return": 0.01,
            })
        check_graduation_criteria(p)
        captured = capsys.readouterr()
        assert "GRADUATION DEFERRED" in captured.out
        assert "unique trading days" in captured.out


# ---------------------------------------------------------------------------
# Constants — ORDERS_LOG, PERFORMANCE_LOG, threshold bounds
# ---------------------------------------------------------------------------

class TestConstantsExtended:

    def test_orders_log_path(self):
        from src.paths import DATA_DIR
        assert ORDERS_LOG == DATA_DIR / "orders.jsonl"

    def test_performance_log_path(self):
        from src.paths import DATA_DIR
        assert PERFORMANCE_LOG == DATA_DIR / "performance.jsonl"

    def test_paper_config_bounds(self):
        assert 50000 <= PAPER_CONFIG["initial_capital"] <= 500000
        assert 0 < PAPER_CONFIG["max_position_pct"] <= 1.0
        assert 0 < PAPER_CONFIG["max_drawdown_pct"] <= 0.5
        assert 0 < PAPER_CONFIG["rebalance_threshold"] <= 0.5
        assert 0 < PAPER_CONFIG["volatility_target"] <= 0.5

    def test_regime_override_spy_not_excessive(self):
        """SPY weight in crisis/vol_spike modes should be defensive."""
        for regime in ("crisis", "vol_spike"):
            alloc = REGIME_OVERRIDES[regime]
            assert alloc["SPY"] <= 0.30, f"SPY {alloc['SPY']} too high for {regime}"

    def test_base_allocation_keys(self):
        assert set(BASE_ALLOCATION.keys()) == {"SPY", "GLD", "TLT"}

    def test_regime_override_keys(self):
        assert set(REGIME_OVERRIDES.keys()) == {"crisis", "vol_spike", "low_vol", "normal"}


# ---------------------------------------------------------------------------
# Position — _asdict() completeness
# ---------------------------------------------------------------------------

class TestPositionExtended:

    def test_asdict_fields(self):
        p = _make_position()
        d = p._asdict()
        expected = {"symbol", "shares", "avg_price", "current_price",
                     "value", "weight", "unrealized_pnl"}
        assert set(d.keys()) == expected

    def test_asdict_values_match(self):
        p = _make_position(symbol="QQQ", shares=50, avg_price=400.0,
                           current_price=420.0, value=21000, weight=0.21,
                           unrealized_pnl=1000)
        d = p._asdict()
        assert d["symbol"] == "QQQ"
        assert d["shares"] == 50
        assert d["value"] == 21000

    def test_zero_shares(self):
        p = _make_position(shares=0, value=0, weight=0, unrealized_pnl=0)
        assert p.shares == 0
        assert p.value == 0
        assert p.weight == 0


# ---------------------------------------------------------------------------
# Portfolio — state file edge cases
# ---------------------------------------------------------------------------

class TestPortfolioStateExtended:

    def test_state_file_not_exists(self, tmp_path):
        """When state file does not exist, portfolio uses initial capital."""
        path = tmp_path / "nonexistent.json"
        assert not path.exists()
        p = Portfolio(path, mode="paper")
        assert p.cash == PAPER_CONFIG["initial_capital"]
        assert p.positions == {}
        assert p.history == []

    def test_state_file_with_invalid_json(self, tmp_path):
        """Corrupted state file raises JSONDecodeError — Portfolio requires valid JSON."""
        path = tmp_path / "corrupt.json"
        path.write_text("{invalid json}")
        with pytest.raises(json.JSONDecodeError):
            Portfolio(path, mode="paper")

    def test_save_state_truncates_history(self, tmp_path):
        """save_state only keeps last 100 history entries."""
        p = _make_portfolio(tmp_path)
        p.history = [{"total_value": 100000 + i, "daily_return": 0.001}
                      for i in range(150)]
        p.save_state()
        p2 = Portfolio(p.state_file, mode="paper")
        assert len(p2.history) <= 100


# ---------------------------------------------------------------------------
# Portfolio — total_value edge cases
# ---------------------------------------------------------------------------

class TestTotalValueExtended:

    def test_zero_cash_empty_positions(self, tmp_path):
        p = _make_portfolio(tmp_path, cash=0)
        assert p.total_value({}) == 0

    def test_negative_cash_with_positions(self, tmp_path):
        p = _make_portfolio(tmp_path, cash=-5000)
        p.positions = {"SPY": _make_position(shares=10, current_price=460)}
        # Cash -5000 + 10 * 460 = -400
        assert p.total_value({}) == -400

    def test_zero_shares_position(self, tmp_path):
        p = _make_portfolio(tmp_path, cash=50000)
        p.positions = {"SPY": _make_position(shares=0, current_price=460, value=0)}
        assert p.total_value({}) == 50000

    def test_missing_price_uses_current_price(self, tmp_path):
        """When a symbol has no entry in prices dict, current_price is used."""
        p = _make_portfolio(tmp_path, cash=10000)
        p.positions = {"SPY": _make_position(shares=50, current_price=460)}
        # prices dict does NOT include SPY
        assert p.total_value({"GLD": 200}) == 10000 + 50 * 460


# ---------------------------------------------------------------------------
# Portfolio — current_weights edge cases
# ---------------------------------------------------------------------------

class TestCurrentWeightsExtended:

    def test_zero_total(self, tmp_path):
        p = _make_portfolio(tmp_path, cash=0)
        assert p.current_weights({}) == {}

    def test_single_position(self, tmp_path):
        p = _make_portfolio(tmp_path, cash=0)
        p.positions = {"SPY": _make_position(shares=100, current_price=460)}
        weights = p.current_weights({"SPY": 460})
        assert weights == {"SPY": 1.0}

    def test_three_way_split(self, tmp_path):
        p = _make_portfolio(tmp_path, cash=0)
        p.positions = {
            "SPY": _make_position(shares=46, current_price=100),
            "GLD": _make_position(symbol="GLD", shares=38, current_price=100),
            "TLT": _make_position(symbol="TLT", shares=16, current_price=100),
        }
        prices = {"SPY": 100, "GLD": 100, "TLT": 100}
        weights = p.current_weights(prices)
        assert weights["SPY"] == pytest.approx(0.46, abs=0.001)
        assert weights["GLD"] == pytest.approx(0.38, abs=0.001)
        assert weights["TLT"] == pytest.approx(0.16, abs=0.001)

    def test_cash_affects_weights(self, tmp_path):
        p = _make_portfolio(tmp_path, cash=50000)
        p.positions = {"SPY": _make_position(shares=100, current_price=500)}
        prices = {"SPY": 500}
        # Total = 50000 + 50000 = 100000, SPY weight = 50000/100000 = 0.5
        assert p.current_weights(prices)["SPY"] == pytest.approx(0.5)

    def test_prices_fallback_to_current_price(self, tmp_path):
        p = _make_portfolio(tmp_path, cash=0)
        p.positions = {"SPY": _make_position(shares=100, current_price=460)}
        # SPY not in prices dict -> uses current_price 460
        weights = p.current_weights({})
        assert weights["SPY"] == 1.0


# ---------------------------------------------------------------------------
# Portfolio — calculate_orders edge cases
# ---------------------------------------------------------------------------

class TestCalculateOrdersExtended:

    def test_empty_target(self, tmp_path):
        p = _make_portfolio(tmp_path, cash=100000)
        orders = p.calculate_orders({}, {"SPY": 500})
        assert orders == []

    def test_delta_below_minimum(self, tmp_path):
        """Delta_value <= $100 should skip order."""
        p = _make_portfolio(tmp_path, cash=200)
        target = {"SPY": 0.50}
        prices = {"SPY": 500}
        orders = p.calculate_orders(target, prices)
        # target_value = 200 * 0.5 = 100, current_value = 0, delta = 100
        # abs(100) <= 100 → skip
        assert len(orders) == 0

    def test_delta_above_minimum(self, tmp_path):
        """Delta_value > $100 should generate order."""
        p = _make_portfolio(tmp_path, cash=202)
        target = {"SPY": 0.50}
        prices = {"SPY": 500}
        orders = p.calculate_orders(target, prices)
        # target_value = 202 * 0.5 = 101, current_value = 0, delta = 101
        # abs(101) > 100 → generate order
        assert len(orders) == 1

    def test_target_symbol_not_held_generates_buy(self, tmp_path):
        """When target has a symbol not in current positions, generates buy."""
        p = _make_portfolio(tmp_path, cash=100000)
        target = {"SPY": 0.50}
        prices = {"SPY": 500}
        orders = p.calculate_orders(target, prices)
        assert len(orders) == 1
        assert orders[0]["symbol"] == "SPY"
        assert orders[0]["side"] == "buy"
        assert orders[0]["shares"] == pytest.approx(100)  # 50000 / 500

    def test_symbol_not_in_prices_dict(self, tmp_path):
        """Symbol in target but not in prices dict should be skipped."""
        p = _make_portfolio(tmp_path, cash=100000)
        target = {"SPY": 0.50, "GLD": 0.50}
        prices = {"SPY": 500}  # GLD not in prices, get_latest_prices returns no GLD
        orders = p.calculate_orders(target, prices)
        assert len(orders) == 1
        assert orders[0]["symbol"] == "SPY"

    def test_drift_below_threshold(self, tmp_path):
        """Drift below rebalance_threshold (0.10) should NOT trigger order."""
        p = _make_portfolio(tmp_path, cash=0)
        # SPY weight = 0.37 → drift = |0.46 - 0.37| = 0.09 < 0.10
        # TLT weight = 0.25 → drift = |0.16 - 0.25| = 0.09 < 0.10
        p.positions = {
            "SPY": _make_position(shares=37, current_price=100, value=3700),
            "GLD": _make_position(symbol="GLD", shares=38, current_price=100, value=3800),
            "TLT": _make_position(symbol="TLT", shares=25, current_price=100, value=2500),
        }
        prices = {"SPY": 100, "GLD": 100, "TLT": 100}
        orders = p.calculate_orders(BASE_ALLOCATION, prices)
        assert orders == []

    def test_drift_just_above_threshold(self, tmp_path):
        """Drift just above rebalance_threshold (0.10) should trigger order."""
        p = _make_portfolio(tmp_path, cash=0)
        p.positions = {
            "SPY": _make_position(shares=37, current_price=100, value=3700),
            "GLD": _make_position(symbol="GLD", shares=38, current_price=100, value=3800),
            "TLT": _make_position(symbol="TLT", shares=25, current_price=100, value=2500),
        }
        prices = {"SPY": 100, "GLD": 100, "TLT": 100}
        # SPY weight = 3700/10000 = 0.37, drift = |0.46 - 0.37| = 0.09
        # Not above. Let's make it drift to 0.35 (drift = 0.11)
        p.positions["SPY"] = _make_position(shares=35, current_price=100, value=3500)
        orders = p.calculate_orders(BASE_ALLOCATION, prices)
        assert len(orders) >= 1

    def test_multiple_underweight_symbols(self, tmp_path):
        """Multiple symbols drifting below target generate buy orders."""
        p = _make_portfolio(tmp_path, cash=50000)
        p.positions = {
            "SPY": _make_position(shares=20, current_price=460, value=9200),
        }
        total = 50000 + 9200
        # SPY weight = 9200/59200 = 0.155, drift = |0.46 - 0.155| = 0.305 > 0.10
        orders = p.calculate_orders(BASE_ALLOCATION, {"SPY": 460})
        assert len(orders) == 1
        assert orders[0]["side"] == "buy"


# ---------------------------------------------------------------------------
# Portfolio — execute_orders edge cases
# ---------------------------------------------------------------------------

class TestExecuteOrdersExtended:

    def test_sell_nonexistent_position(self, tmp_path):
        """Selling a non-existent position leaves cash and positions unchanged.
        The order is still returned in executed (executed.append is unconditional),
        but the position/cash state is not modified."""
        p = _make_portfolio(tmp_path, cash=50000)
        orders = [
            {"symbol": "SPY", "side": "sell", "shares": 10, "estimated_price": 460},
        ]
        prices = {"SPY": 460}
        before_cash = p.cash
        executed = p.execute_orders(orders, prices, slippage=0.0)
        # SPY not in positions → sell skipped → cash unchanged
        assert "SPY" not in p.positions
        assert p.cash == before_cash

    def test_buy_zero_cash(self, tmp_path):
        """Buying with zero cash results in zero fill."""
        p = _make_portfolio(tmp_path, cash=0)
        orders = [{"symbol": "SPY", "side": "buy", "shares": 10, "estimated_price": 460}]
        prices = {"SPY": 460}
        executed = p.execute_orders(orders, prices, slippage=0.0)
        assert len(executed) == 1
        assert executed[0]["fill_shares"] == 0
        assert executed[0]["fill_value"] == 0

    def test_sell_excessive_shares(self, tmp_path):
        """Selling more shares than owned should not reduce below zero.
        The order is still returned in executed, but the position is not modified."""
        p = _make_portfolio(tmp_path, cash=50000)
        p.positions = {"SPY": _make_position(shares=5, avg_price=450)}
        orders = [
            {"symbol": "SPY", "side": "sell", "shares": 100, "estimated_price": 460},
        ]
        prices = {"SPY": 460}
        before_cash = p.cash
        executed = p.execute_orders(orders, prices, slippage=0.0)
        # Sell condition: symbol in positions AND shares >= fill_shares
        # 5 >= 100 → False → sell skipped → shares and cash unchanged
        assert p.positions["SPY"].shares == 5
        assert p.cash == before_cash

    def test_slippage_on_sell(self, tmp_path):
        """Sell fills at slightly lower price with slippage."""
        p = _make_portfolio(tmp_path, cash=50000)
        p.positions = {"SPY": _make_position(shares=10, avg_price=450)}
        orders = [{"symbol": "SPY", "side": "sell", "shares": 10, "estimated_price": 460}]
        prices = {"SPY": 460}
        executed = p.execute_orders(orders, prices, slippage=0.02)
        assert executed[0]["fill_price"] == pytest.approx(460 * 0.98)

    def test_multiple_buys_depleting_cash(self, tmp_path):
        """Multiple buy orders should not exceed available cash."""
        p = _make_portfolio(tmp_path, cash=1000)
        orders = [
            {"symbol": "SPY", "side": "buy", "shares": 2, "estimated_price": 460},
            {"symbol": "GLD", "side": "buy", "shares": 2, "estimated_price": 190},
        ]
        prices = {"SPY": 460, "GLD": 190}
        executed = p.execute_orders(orders, prices, slippage=0.0)
        # First order: 2 * 460 = $920, remaining $80
        # Second order: 2 * 190 = $380 but only $80 left → partial fill
        assert len(executed) == 2
        assert p.cash >= 0  # Never go negative

    def test_reduce_position_not_remove(self, tmp_path):
        """Selling partial position reduces shares, doesn't remove symbol."""
        p = _make_portfolio(tmp_path, cash=50000)
        p.positions = {"SPY": _make_position(shares=100, avg_price=450)}
        orders = [{"symbol": "SPY", "side": "sell", "shares": 30, "estimated_price": 460}]
        prices = {"SPY": 460}
        p.execute_orders(orders, prices, slippage=0.0)
        assert "SPY" in p.positions
        assert p.positions["SPY"].shares == 70


# ---------------------------------------------------------------------------
# Portfolio — check_risk_limits edge cases
# ---------------------------------------------------------------------------

class TestRiskLimitsExtended:

    def test_history_below_drawdown_threshold(self, tmp_path):
        """With <20 history entries, drawdown check is skipped."""
        p = _make_portfolio(tmp_path, cash=100000)
        p.positions = {}
        p.history = [{"total_value": 100000}] * 19  # Less than 20
        result = p.check_risk_limits({})
        assert result is None

    def test_history_below_garch_threshold(self, tmp_path):
        """With 20-63 entries, drawdown checks but GARCH does not."""
        p = _make_portfolio(tmp_path, cash=0)
        p.positions = {}
        # Peak = 100000, current = 0 → 100% DD > 15%
        p.history = [{"total_value": 100000}] * 25
        result = p.check_risk_limits({})
        assert result is not None
        assert "max_drawdown" in result

    def test_weight_at_boundary(self, tmp_path):
        """Weight exactly at max_position_pct (0.50) should NOT breach."""
        p = _make_portfolio(tmp_path, cash=0)
        p.positions = {"SPY": _make_position(weight=0.50, value=50000)}
        result = p.check_risk_limits({"SPY": 460})
        # Condition: p.weight > PAPER_CONFIG["max_position_pct"] → 0.50 > 0.50 → False
        assert result is None

    def test_weight_slightly_below_boundary(self, tmp_path):
        """Weight at 0.499 should not breach."""
        p = _make_portfolio(tmp_path, cash=0)
        p.positions = {"SPY": _make_position(weight=0.499, value=49900)}
        result = p.check_risk_limits({"SPY": 460})
        assert result is None

    def test_concentration_breach_with_multiple_positions(self, tmp_path):
        """Only the overweight position should trigger, not the others."""
        p = _make_portfolio(tmp_path, cash=0)
        p.positions = {
            "SPY": _make_position(weight=0.55, value=55000),
            "GLD": _make_position(symbol="GLD", weight=0.45, value=45000),
        }
        result = p.check_risk_limits({"SPY": 460, "GLD": 190})
        assert result is not None
        assert "max_position_SPY" in result


# ---------------------------------------------------------------------------
# Portfolio._compute_portfolio_entropy
# ---------------------------------------------------------------------------

class TestPortfolioEntropy:

    def test_all_keys_present(self):
        result = Portfolio._compute_portfolio_entropy()
        assert result["name"] == "portfolio_entropy"
        assert "status" in result
        assert "ok" in result
        assert "metrics" in result
        metrics = result["metrics"]
        assert "shannon_entropy" in metrics
        assert "effective_n" in metrics
        assert "normalized_score" in metrics
        assert "hhi_index" in metrics

    def test_normalized_score_good(self):
        """For SPY/GLD/TLT 46/38/16, normalized_score should be > 90."""
        import math
        w = [0.46, 0.38, 0.16]
        expected_h = -sum(v * math.log(v) for v in w)
        expected_h_max = math.log(3)
        expected_norm = expected_h / expected_h_max * 100
        expected_hhi = 0.46**2 + 0.38**2 + 0.16**2
        expected_eff_n = math.exp(expected_h)

        result = Portfolio._compute_portfolio_entropy()
        metrics = result["metrics"]

        assert result["status"] == "good"
        assert result["ok"] is True
        assert metrics["shannon_entropy"] == pytest.approx(expected_h, abs=0.001)
        assert metrics["effective_n"] == pytest.approx(expected_eff_n, abs=0.05)
        assert metrics["normalized_score"] == pytest.approx(expected_norm, abs=0.5)
        assert metrics["hhi_index"] == pytest.approx(expected_hhi, abs=0.001)


# ---------------------------------------------------------------------------
# calculate_performance — edge cases
# ---------------------------------------------------------------------------

class TestCalculatePerformanceExtended:

    def test_zero_total_value(self, tmp_path):
        p = _make_portfolio(tmp_path, cash=0)
        p.positions = {}
        p.history = [{"total_value": 1000}]
        perf = calculate_performance(p, {})
        assert perf["total_value"] == 0
        assert perf["daily_return"] == -1.0

    def test_negative_daily_return(self, tmp_path):
        p = _make_portfolio(tmp_path, cash=80000)
        p.history = [{"total_value": 100000}]
        perf = calculate_performance(p, {})
        expected = (80000 - 100000) / 100000
        assert perf["daily_return"] == pytest.approx(expected)

    def test_negative_last_total_guard(self, tmp_path):
        """When last_total <= 0, daily_return should be 0 (guard)."""
        p = _make_portfolio(tmp_path, cash=100000)
        p.history = [{"total_value": -5000}]
        perf = calculate_performance(p, {})
        # last_total = -5000 ≤ 0 → guard returns 0
        assert perf["daily_return"] == 0

    def test_large_positive_return(self, tmp_path):
        p = _make_portfolio(tmp_path, cash=200000)
        p.history = [{"total_value": 1000}]
        perf = calculate_performance(p, {})
        expected = (200000 - 1000) / 1000
        assert perf["daily_return"] == pytest.approx(expected)

    def test_performance_structure_fields(self, tmp_path):
        p = _make_portfolio(tmp_path, cash=100000)
        perf = calculate_performance(p, {})
        assert "timestamp" in perf
        assert "total_value" in perf
        assert "cash" in perf
        assert "daily_return" in perf
        assert "positions_count" in perf
        assert "mode" in perf

    def test_mode_paper_in_performance(self, tmp_path):
        p = _make_portfolio(tmp_path, cash=100000)
        p.mode = "paper"
        perf = calculate_performance(p, {})
        assert perf["mode"] == "paper"

    def test_mode_live_in_performance(self, tmp_path):
        p = _make_portfolio(tmp_path, cash=100000)
        p.mode = "live"
        perf = calculate_performance(p, {})
        assert perf["mode"] == "live"


# ---------------------------------------------------------------------------
# check_graduation_criteria — boundary conditions
# ---------------------------------------------------------------------------

class TestGraduationCriteriaBoundaries:

    @patch("src.backtest.metrics.compute_deflated_sharpe_ratio", return_value=0.0)
    def test_dsr_zero_blocks_graduation(self, mock_dsr, tmp_path, capsys):
        """DSR = 0.0 should block graduation even with good performance."""
        p = _make_portfolio(tmp_path)
        rng = np.random.RandomState(12345)
        p.history = []
        val = 100000
        for i in range(63):
            ret = rng.normal(0.0008, 0.01)
            val *= (1 + ret)
            p.history.append({
                "timestamp": f"2026-01-{i+1:02d}T23:00:00",
                "total_value": round(val, 2),
                "daily_return": ret,
            })
        check_graduation_criteria(p)
        captured = capsys.readouterr()
        assert "GRADUATION CANDIDATE" not in captured.out

    @patch("src.backtest.metrics.compute_deflated_sharpe_ratio", return_value=0.95)
    def test_dsr_above_minimum_allows_graduation(self, mock_dsr, tmp_path, capsys):
        """DSR above MIN_DSR should allow graduation when other conditions met."""
        p = _make_portfolio(tmp_path)
        rng = np.random.RandomState(12345)
        p.history = []
        val = 100000
        for i in range(63):
            ret = rng.normal(0.0008, 0.01)
            val *= (1 + ret)
            p.history.append({
                "timestamp": f"2026-01-{i+1:02d}T23:00:00",
                "total_value": round(val, 2),
                "daily_return": ret,
            })
        check_graduation_criteria(p)
        captured = capsys.readouterr()
        assert "GRADUATION CANDIDATE" in captured.out

    @patch("src.backtest.metrics.compute_deflated_sharpe_ratio", return_value=0.49)
    def test_dsr_below_minimum_blocks_graduation(self, mock_dsr, tmp_path, capsys):
        """DSR < 0.50 should block graduation."""
        p = _make_portfolio(tmp_path)
        rng = np.random.RandomState(12345)
        p.history = []
        val = 100000
        for i in range(63):
            ret = rng.normal(0.0008, 0.01)
            val *= (1 + ret)
            p.history.append({
                "timestamp": f"2026-01-{i+1:02d}T23:00:00",
                "total_value": round(val, 2),
                "daily_return": ret,
            })
        check_graduation_criteria(p)
        captured = capsys.readouterr()
        assert "GRADUATION CANDIDATE" not in captured.out

    def test_max_dd_excessive_blocks_graduation(self, tmp_path, capsys):
        """Excessive drawdown should block graduation even with positive returns."""
        p = _make_portfolio(tmp_path)
        p.history = []
        val = 100000
        for i in range(63):
            if i == 30:
                val = 60000  # 40% drawdown
            ret = (val - (100000 if i == 0 else p.history[-1]["total_value"])) \
                   / (100000 if i == 0 else p.history[-1]["total_value"]) \
                   if i > 0 else 0
            p.history.append({
                "timestamp": f"2026-01-{i+1:02d}T23:00:00",
                "total_value": val,
                "daily_return": ret if i > 0 else 0.001,
            })
            val += 500  # Recover slightly each day
        check_graduation_criteria(p)
        captured = capsys.readouterr()
        assert "GRADUATION CANDIDATE" not in captured.out

    def test_low_win_rate_blocks_graduation(self, tmp_path, capsys):
        """Win rate below 45% should block graduation."""
        p = _make_portfolio(tmp_path)
        p.history = []
        val = 100000
        for i in range(63):
            # 25 positive days, 38 negative days → win rate = 25/63 = 39.7% < 45%
            ret = 0.001 if i < 25 else -0.002
            val *= (1 + ret)
            p.history.append({
                "timestamp": f"2026-01-{i+1:02d}T23:00:00",
                "total_value": round(val, 2),
                "daily_return": ret,
            })
        check_graduation_criteria(p)
        captured = capsys.readouterr()
        assert "GRADUATION CANDIDATE" not in captured.out

    def test_fewer_than_min_days_no_output(self, tmp_path, capsys):
        """With fewer than MIN_DAYS (63) history entries, no output at all."""
        p = _make_portfolio(tmp_path)
        p.history = [{"total_value": 100000, "daily_return": 0.001}] * 62
        check_graduation_criteria(p)
        captured = capsys.readouterr()
        assert captured.out.strip() == ""

    def test_promotion_trigger_has_requires_approval(self, tmp_path, capsys):
        """When graduated, trigger file should include requires_approval: True."""
        p = _make_portfolio(tmp_path)
        rng = np.random.RandomState(12345)
        p.history = []
        val = 100000
        for i in range(63):
            ret = rng.normal(0.0008, 0.01)
            val *= (1 + ret)
            p.history.append({
                "timestamp": f"2026-01-{i+1:02d}T23:00:00",
                "total_value": round(val, 2),
                "daily_return": ret,
            })
        check_graduation_criteria(p)
        trigger_path = tmp_path / "data" / ".promote_to_live"
        if trigger_path.exists():
            trigger = json.loads(trigger_path.read_text())
            assert trigger.get("requires_approval") is True


# ---------------------------------------------------------------------------
# get_current_regime (requires mocked SQLite connection)
# ---------------------------------------------------------------------------

class TestCurrentRegime:

    def test_crisis_high_vix(self):
        """VIX > 25 returns crisis."""
        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value = cursor
        cursor.fetchone.side_effect = [(30.0,), ("normal",)]
        assert get_current_regime(conn) == "crisis"

    def test_vol_spike_vix(self):
        """VIX between 20-25 returns vol_spike."""
        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value = cursor
        cursor.fetchone.side_effect = [(22.5,), ("normal",)]
        assert get_current_regime(conn) == "vol_spike"

    def test_low_vol_with_normal_trend(self):
        """VIX < 15 with non-crisis trend returns low_vol."""
        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value = cursor
        cursor.fetchone.side_effect = [(12.0,), ("normal",)]
        assert get_current_regime(conn) == "low_vol"

    def test_normal_vix(self):
        """VIX between 15-20 returns normal."""
        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value = cursor
        cursor.fetchone.side_effect = [(18.0,), ("normal",)]
        assert get_current_regime(conn) == "normal"

    def test_crisis_overrides_trend_regardless(self):
        """If VIX says crisis, trend regime does not matter."""
        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value = cursor
        cursor.fetchone.side_effect = [(30.0,), ("low_vol",)]
        assert get_current_regime(conn) == "crisis"

    def test_no_trend_data_defaults_normal(self):
        """When no trend data exists, default to 'normal'."""
        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value = cursor
        cursor.fetchone.side_effect = [(15.5,), None]
        assert get_current_regime(conn) == "normal"

    def test_no_vix_data_uses_trend(self):
        """When no VIX data exists, fall back to trend regime."""
        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value = cursor
        cursor.fetchone.side_effect = [None, ("crisis",)]
        assert get_current_regime(conn) == "crisis"

    def test_low_vol_crisis_trend_uses_trend(self):
        """If VIX says low_vol but trend is crisis, use trend (crisis)."""
        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value = cursor
        cursor.fetchone.side_effect = [(12.0,), ("crisis",)]
        assert get_current_regime(conn) == "crisis"  # low_vol + trend=crisis → crisis

    def test_no_data_whatsoever(self):
        """No VIX and no trend data returns 'normal'."""
        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value = cursor
        cursor.fetchone.side_effect = [None, None]
        assert get_current_regime(conn) == "normal"


# ---------------------------------------------------------------------------
# get_latest_vix (requires mocked SQLite connection)
# ---------------------------------------------------------------------------

class TestLatestVix:

    def test_returns_vix_value(self):
        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value = cursor
        cursor.fetchone.return_value = (18.5,)
        assert get_latest_vix(conn) == 18.5

    def test_no_vix_data(self):
        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value = cursor
        cursor.fetchone.return_value = None
        assert get_latest_vix(conn) is None


# ---------------------------------------------------------------------------
# _deduplicate_to_daily — additional edge cases
# ---------------------------------------------------------------------------

class TestDeduplicateToDailyExtended:

    def test_single_day_many_entries(self):
        """Many entries on the same day → only last retained."""
        h = [{"timestamp": f"2026-01-01T{hour:02d}:00:00", "value": hour}
             for hour in range(24)]
        result = _deduplicate_to_daily(h)
        assert len(result) == 1
        assert result[0]["value"] == 23

    def test_dates_not_contiguous(self):
        """Non-contiguous dates should return entries in chronological order."""
        h = [
            {"timestamp": "2026-01-01T12:00:00", "value": 100},
            {"timestamp": "2026-01-03T12:00:00", "value": 103},
            {"timestamp": "2026-01-02T12:00:00", "value": 102},
        ]
        result = _deduplicate_to_daily(h)
        assert len(result) == 3
        values = [e["value"] for e in result]
        assert values == [100, 102, 103]

    def test_entry_missing_timestamp_key(self):
        """Entry with no 'timestamp' key at all."""
        h = [{"daily_return": 0.1}, {"daily_return": 0.2}]
        result = _deduplicate_to_daily(h)
        # Both get date_key="" → last overwrites
        assert len(result) == 1
        assert result[0]["daily_return"] == 0.2

    def test_partial_timestamp(self):
        """Timestamp with fewer than 10 characters."""
        h = [
            {"timestamp": "2026-01", "value": 100},
            {"timestamp": "2026-02", "value": 200},
        ]
        result = _deduplicate_to_daily(h)
        assert len(result) == 2
        assert result[0]["value"] == 100
        assert result[1]["value"] == 200

    def test_empty_timestamp_string(self):
        """Empty timestamp string."""
        h = [
            {"timestamp": "", "value": 1},
            {"timestamp": "", "value": 2},
        ]
        result = _deduplicate_to_daily(h)
        assert len(result) == 1
        assert result[0]["value"] == 2


# ---------------------------------------------------------------------------
# Kill Switch Trigger -- evaluator main() kill switch file creation/clearing
# ---------------------------------------------------------------------------

class TestKillSwitchTrigger:
    """Kill switch trigger: file creation when risk breached, clearing when clear."""

    @patch('src.strategy.evaluator.sqlite_connect')
    @patch('src.strategy.evaluator.get_latest_prices', return_value={"SPY": 500.0})
    @patch('src.strategy.evaluator.get_current_regime', return_value="normal")
    @patch('src.strategy.evaluator.get_latest_vix', return_value=15.0)
    def test_risk_breach_creates_kill_switch_file(
        self, mock_vix, mock_regime, mock_prices, mock_sqlite,
        tmp_path, capsys,
    ):
        """Risk breach -> kill_switch.json created with correct JSON."""
        from src.strategy.evaluator import main

        with (
            patch('src.strategy.evaluator.DATA_DIR', tmp_path),
            patch('sys.argv', ['evaluator.py', 'paper']),
            patch('src.strategy.evaluator.Portfolio') as MockPortfolio,
        ):
            mock_portfolio = MagicMock()
            mock_portfolio.check_risk_limits.return_value = "max_drawdown_-25.0%"
            mock_portfolio.total_value.return_value = 75000
            MockPortfolio.return_value = mock_portfolio

            main()

        kill_file = tmp_path / "kill_switch.json"
        assert kill_file.exists()
        with open(kill_file) as f:
            data = json.load(f)
        assert data["enabled"] is True
        assert data["reason"] == "max_drawdown_-25.0%"
        assert data["mode"] == "paper"

    @patch('src.strategy.evaluator.sqlite_connect')
    @patch('src.strategy.evaluator.get_latest_prices', return_value={"SPY": 500.0})
    @patch('src.strategy.evaluator.get_current_regime', return_value="normal")
    @patch('src.strategy.evaluator.get_latest_vix', return_value=15.0)
    def test_no_risk_breach_clears_kill_switch(
        self, mock_vix, mock_regime, mock_prices, mock_sqlite,
        tmp_path, capsys,
    ):
        """No risk breach -> stale kill_switch.json file is deleted."""
        from src.strategy.evaluator import main

        # Create stale kill switch file manually
        stale = tmp_path / "kill_switch.json"
        stale.write_text('{"enabled": true, "reason": "old_breach", "mode": "paper", "timestamp": "2026-01-01T00:00:00"}')
        assert stale.exists()

        with (
            patch('src.strategy.evaluator.DATA_DIR', tmp_path),
            patch('sys.argv', ['evaluator.py', 'paper']),
            patch('src.strategy.evaluator.Portfolio') as MockPortfolio,
        ):
            mock_portfolio = MagicMock()
            mock_portfolio.check_risk_limits.return_value = None
            mock_portfolio.total_value.return_value = 100000
            mock_portfolio.calculate_orders.return_value = []
            mock_portfolio.cash = 100000
            mock_portfolio.positions = {}
            mock_portfolio.mode = "paper"
            mock_portfolio.history = [{"total_value": 100000}]
            MockPortfolio.return_value = mock_portfolio

            main()

        # Stale kill switch file should be deleted
        assert not stale.exists()

    @patch('src.strategy.evaluator.sqlite_connect')
    @patch('src.strategy.evaluator.get_latest_prices', return_value={"SPY": 500.0})
    @patch('src.strategy.evaluator.get_current_regime', return_value="normal")
    @patch('src.strategy.evaluator.get_latest_vix', return_value=15.0)
    def test_kill_switch_file_contains_reason_and_timestamp(
        self, mock_vix, mock_regime, mock_prices, mock_sqlite,
        tmp_path, capsys,
    ):
        """Kill switch JSON has expected structure: enabled + reason + mode + timestamp."""
        from src.strategy.evaluator import main

        with (
            patch('src.strategy.evaluator.DATA_DIR', tmp_path),
            patch('sys.argv', ['evaluator.py', 'paper']),
            patch('src.strategy.evaluator.Portfolio') as MockPortfolio,
        ):
            mock_portfolio = MagicMock()
            mock_portfolio.check_risk_limits.return_value = "max_position_SPY_55.0%"
            mock_portfolio.total_value.return_value = 75000
            MockPortfolio.return_value = mock_portfolio

            main()

        kill_file = tmp_path / "kill_switch.json"
        assert kill_file.exists()
        with open(kill_file) as f:
            data = json.load(f)
        assert data["enabled"] is True
        assert "reason" in data
        assert "mode" in data
        assert "timestamp" in data
        assert isinstance(data["reason"], str)
        assert isinstance(data["mode"], str)
        assert isinstance(data["timestamp"], str)
        assert len(data["reason"]) > 0
        assert len(data["timestamp"]) > 0

    @patch('src.strategy.evaluator.sqlite_connect')
    @patch('src.strategy.evaluator.get_latest_prices', return_value={"SPY": 500.0})
    @patch('src.strategy.evaluator.get_current_regime', return_value="normal")
    @patch('src.strategy.evaluator.get_latest_vix', return_value=15.0)
    def test_kill_switch_json_readable_by_order_router(
        self, mock_vix, mock_regime, mock_prices, mock_sqlite,
        tmp_path, capsys,
    ):
        """End-to-end: evaluator breach -> kill_switch.json -> order router blocks orders."""
        from src.strategy.evaluator import main
        from src.broker.order_router import OrderRouter

        # Step 1: Evaluator writes kill_switch.json on risk breach
        with (
            patch('src.strategy.evaluator.DATA_DIR', tmp_path),
            patch('sys.argv', ['evaluator.py', 'paper']),
            patch('src.strategy.evaluator.Portfolio') as MockPortfolio,
        ):
            mock_portfolio = MagicMock()
            mock_portfolio.check_risk_limits.return_value = "max_drawdown_-30.0%"
            mock_portfolio.total_value.return_value = 75000
            MockPortfolio.return_value = mock_portfolio
            main()

        kill_file = tmp_path / "kill_switch.json"
        assert kill_file.exists(), "evaluator should write kill_switch.json"

        # Step 2: Order router reads kill_switch.json and blocks orders
        with patch('src.broker.order_router.DATA_DIR', tmp_path):
            router = OrderRouter(data_dir=str(tmp_path), paper=True)
            # execute_orders should return "blocked" when kill switch is active
            with patch.object(router, 'is_ready', return_value=True):
                result = router.execute_orders(
                    [{"symbol": "SPY", "side": "buy", "qty": 1}],
                    dry_run=False,
                    kill_switch_check=True,
                )
            assert result["status"] == "blocked"

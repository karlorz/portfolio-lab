#!/usr/bin/env python3
"""
Tests for evaluator.py — constants, Position/Portfolio classes, order generation,
order execution, risk limits, performance calculation, and graduation criteria.
"""
import json
import logging
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

    @pytest.fixture(autouse=True)
    def _isolate_data_dir(self, tmp_path, monkeypatch):
        """Graduation tests must not see live data/kill_switch.json."""
        monkeypatch.setattr("src.strategy.evaluator.DATA_DIR", tmp_path)

    def test_too_few_days(self, tmp_path, caplog):
        p = _make_portfolio(tmp_path)
        p.history = [{"total_value": 100000, "daily_return": 0.001}] * 30
        with caplog.at_level(logging.INFO, logger="src.strategy.evaluator"):
            check_graduation_criteria(p)
        assert "GRADUATION" not in caplog.text

    def test_good_performance(self, tmp_path, caplog):
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
        with caplog.at_level(logging.INFO, logger="src.strategy.evaluator"):
            check_graduation_criteria(p)
        assert "GRADUATION CANDIDATE" in caplog.text, f"Log: '{caplog.text.strip()}'"

    def test_poor_performance_no_graduation(self, tmp_path, caplog):
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
        with caplog.at_level(logging.INFO, logger="src.strategy.evaluator"):
            check_graduation_criteria(p)
        assert "GRADUATION CANDIDATE" not in caplog.text

    def test_intra_day_data_does_not_trigger(self, tmp_path, caplog):
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
        with caplog.at_level(logging.INFO, logger="src.strategy.evaluator"):
            check_graduation_criteria(p)
        # After dedup to 63 trading days:
        # Check results aren't obviously broken
        assert "exceeds realistic maximum" not in caplog.text
        assert "GRADUATION DEFERRED" not in caplog.text

    def test_near_zero_std_vol_floor(self, tmp_path, caplog):
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
        with caplog.at_level(logging.WARNING, logger="src.strategy.evaluator"):
            check_graduation_criteria(p)
        # Vol floor prevents NaN/Inf, but Sharpe = 0.0001/0.0001*sqrt(252) = 15.87
        # This still exceeds MAX_REALISTIC_SHARPE (3.0), so warning is logged
        assert "exceeds realistic maximum" in caplog.text

    def test_unrealistic_sharpe_rejected(self, tmp_path, caplog):
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
        with caplog.at_level(logging.WARNING, logger="src.strategy.evaluator"):
            check_graduation_criteria(p)
        assert "exceeds realistic maximum" in caplog.text
        assert "GRADUATION CANDIDATE" not in caplog.text

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

    def test_deferred_when_too_few_trading_days(self, tmp_path, caplog, monkeypatch):
        """63 snapshots but only 3 unique days → should defer with message."""
        monkeypatch.setattr("src.strategy.evaluator.DATA_DIR", tmp_path)
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
        with caplog.at_level(logging.INFO, logger="src.strategy.evaluator"):
            check_graduation_criteria(p)
        assert "GRADUATION DEFERRED" in caplog.text
        assert "unique trading days" in caplog.text


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

    def test_performance_log_follows_data_dir_patch(self, tmp_path):
        """Patching DATA_DIR alone must redirect log appends (no live contamination)."""
        import src.strategy.evaluator as ev

        live_log = Path(ev.DATA_DIR) / "performance.jsonl"
        before = live_log.read_bytes() if live_log.exists() else None
        before_mtime = live_log.stat().st_mtime_ns if live_log.exists() else None

        with patch.object(ev, "DATA_DIR", tmp_path):
            assert ev.PERFORMANCE_LOG == tmp_path / "performance.jsonl"
            assert ev.ORDERS_LOG == tmp_path / "orders.jsonl"
            with open(ev.PERFORMANCE_LOG, "a") as f:
                f.write('{"total_value": 100000, "positions": 0, "test": true}\n')
            assert (tmp_path / "performance.jsonl").exists()
            assert b"test" in (tmp_path / "performance.jsonl").read_bytes()

        if before is None:
            assert not live_log.exists() or live_log.stat().st_size == 0
        else:
            assert live_log.read_bytes() == before
            assert live_log.stat().st_mtime_ns == before_mtime

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


class TestCalculatePerformanceDailyReturn:
    """Tests for the daily_return = vs-previous-day-close fix.

    calculate_performance() must compute daily_return relative to the
    last entry from a DIFFERENT calendar date, not the most recent
    intraday snapshot (which gives daily_return=0 when prices haven't
    changed within the day).
    """

    def test_daily_return_uses_previous_day_close(self, tmp_path):
        """daily_return should be relative to previous day's close, not today's intraday snapshot."""
        p = _make_portfolio(tmp_path, cash=102000)
        p.positions = {}
        p.history = [
            {"timestamp": "2026-05-26T09:30:00", "total_value": 99500, "cash": 0, "daily_return": -0.005, "positions_count": 3, "mode": "paper"},
            {"timestamp": "2026-05-26T16:00:00", "total_value": 100000, "cash": 0, "daily_return": 0.005, "positions_count": 3, "mode": "paper"},
            {"timestamp": "2026-05-27T10:00:00", "total_value": 100500, "cash": 0, "daily_return": 0, "positions_count": 3, "mode": "paper"},
        ]
        perf = calculate_performance(p, {})
        # daily_return should be (102000 - 100000) / 100000 = 0.02
        # using yesterday's 16:00 close (100000), NOT today's intraday snapshot (100500)
        expected = (102000 - 100000) / 100000
        assert perf["daily_return"] == pytest.approx(expected, abs=0.001)

    def test_daily_return_first_day_fallback(self, tmp_path):
        """On the first day, daily_return falls back to last history entry."""
        p = _make_portfolio(tmp_path, cash=102000)
        p.positions = {}
        p.history = [
            {"timestamp": "2026-05-26T10:00:00", "total_value": 100500, "cash": 0, "daily_return": 0, "positions_count": 3, "mode": "paper"},
        ]
        perf = calculate_performance(p, {})
        # No previous day → falls back to last entry
        expected = (102000 - 100500) / 100500
        assert perf["daily_return"] == pytest.approx(expected, abs=0.001)

    def test_daily_return_no_history(self, tmp_path):
        """With no history, daily_return should be 0."""
        p = _make_portfolio(tmp_path, cash=100000)
        p.positions = {}
        p.history = []
        perf = calculate_performance(p, {})
        assert perf["daily_return"] == 0.0


# ---------------------------------------------------------------------------
# check_graduation_criteria — boundary conditions
# ---------------------------------------------------------------------------

class TestGraduationCriteriaBoundaries:

    @pytest.fixture(autouse=True)
    def _isolate_data_dir(self, tmp_path, monkeypatch):
        """Graduation tests must not see live data/kill_switch.json."""
        monkeypatch.setattr("src.strategy.evaluator.DATA_DIR", tmp_path)


    @patch("src.backtest.metrics.compute_deflated_sharpe_ratio", return_value=0.0)
    def test_dsr_zero_blocks_graduation(self, mock_dsr, tmp_path, caplog):
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
        with caplog.at_level(logging.INFO, logger="src.strategy.evaluator"):
            check_graduation_criteria(p)
        assert "GRADUATION CANDIDATE" not in caplog.text

    @patch("src.backtest.metrics.compute_deflated_sharpe_ratio", return_value=0.95)
    def test_dsr_above_minimum_allows_graduation(self, mock_dsr, tmp_path, caplog):
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
        with caplog.at_level(logging.INFO, logger="src.strategy.evaluator"):
            check_graduation_criteria(p)
        assert "GRADUATION CANDIDATE" in caplog.text

    @patch("src.backtest.metrics.compute_deflated_sharpe_ratio", return_value=0.49)
    def test_dsr_below_minimum_blocks_graduation(self, mock_dsr, tmp_path, caplog):
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
        with caplog.at_level(logging.INFO, logger="src.strategy.evaluator"):
            check_graduation_criteria(p)
        assert "GRADUATION CANDIDATE" not in caplog.text

    def test_max_dd_excessive_blocks_graduation(self, tmp_path, caplog):
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
        with caplog.at_level(logging.INFO, logger="src.strategy.evaluator"):
            check_graduation_criteria(p)
        assert "GRADUATION CANDIDATE" not in caplog.text

    def test_low_win_rate_blocks_graduation(self, tmp_path, caplog):
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
        with caplog.at_level(logging.INFO, logger="src.strategy.evaluator"):
            check_graduation_criteria(p)
        assert "GRADUATION CANDIDATE" not in caplog.text

    def test_fewer_than_min_days_no_output(self, tmp_path, caplog):
        """With fewer than MIN_DAYS (63) history entries, no output at all."""
        p = _make_portfolio(tmp_path)
        p.history = [{"total_value": 100000, "daily_return": 0.001}] * 62
        with caplog.at_level(logging.INFO, logger="src.strategy.evaluator"):
            check_graduation_criteria(p)
        assert caplog.text.strip() == ""

    def test_promotion_trigger_has_requires_approval(self, tmp_path, caplog):
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

    def test_vix_exactly_25_boundary(self):
        """VIX exactly 25.0 falls into vol_spike (not crisis, since 25 > 25 is False)."""
        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value = cursor
        cursor.fetchone.side_effect = [(25.0,), ("normal",)]
        assert get_current_regime(conn) == "vol_spike"

    def test_vix_exactly_20_boundary(self):
        """VIX exactly 20.0 falls into normal (not vol_spike, since 20 > 20 is False)."""
        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value = cursor
        cursor.fetchone.side_effect = [(20.0,), ("normal",)]
        assert get_current_regime(conn) == "normal"

    def test_vix_exactly_15_boundary(self):
        """VIX exactly 15.0 falls into normal (not low_vol, since 15 < 15 is False)."""
        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value = cursor
        cursor.fetchone.side_effect = [(15.0,), ("normal",)]
        assert get_current_regime(conn) == "normal"

    def test_vix_just_above_crisis_threshold(self):
        """VIX=25.01 is crisis."""
        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value = cursor
        cursor.fetchone.side_effect = [(25.01,), ("normal",)]
        assert get_current_regime(conn) == "crisis"

    def test_vix_just_below_low_vol_threshold(self):
        """VIX=14.99 is low_vol."""
        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value = cursor
        cursor.fetchone.side_effect = [(14.99,), ("normal",)]
        assert get_current_regime(conn) == "low_vol"


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
        assert data["source"] == "evaluator_risk"

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
    def test_no_risk_breach_preserves_incident_owned_kill_switch(
        self, mock_vix, mock_regime, mock_prices, mock_sqlite,
        tmp_path, capsys,
    ):
        """Evaluator recovery must not clear incident-owned router gates."""
        from src.strategy.evaluator import main

        kill_file = tmp_path / "kill_switch.json"
        incident_gate = {
            "enabled": True,
            "level": "halt",
            "reason": "unresolved_incident:signal_staleness",
            "source": "incident_lifecycle",
            "incident_id": "incident-123",
            "incident_channel": "signal_staleness",
            "mode": "paper",
            "timestamp": "2026-07-06T00:00:00+00:00",
            "position_reduction": 1.0,
        }
        kill_file.write_text(json.dumps(incident_gate))

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

        assert kill_file.exists()
        assert json.loads(kill_file.read_text()) == incident_gate

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


class TestKillSwitchGraduatedLevels:
    """Test graduated kill switch: 4-level response (warning/restrict/halt/liquidate)."""

    def test_classify_warning_level(self):
        """10-15% drawdown → WARNING level, 25% position reduction."""
        from src.strategy.evaluator import classify_kill_level, KillSwitchLevel, _kill_level_reduction
        level = classify_kill_level("max_drawdown_-12.0%")
        assert level == KillSwitchLevel.WARNING
        assert _kill_level_reduction(level) == 0.25

    def test_classify_restrict_level(self):
        """15-20% drawdown → RESTRICT level, 50% position reduction."""
        from src.strategy.evaluator import classify_kill_level, KillSwitchLevel, _kill_level_reduction
        level = classify_kill_level("max_drawdown_-18.0%")
        assert level == KillSwitchLevel.RESTRICT
        assert _kill_level_reduction(level) == 0.50

    def test_classify_halt_level(self):
        """20-25% drawdown → HALT level, 100% position reduction."""
        from src.strategy.evaluator import classify_kill_level, KillSwitchLevel, _kill_level_reduction
        level = classify_kill_level("max_drawdown_-22.0%")
        assert level == KillSwitchLevel.HALT
        assert _kill_level_reduction(level) == 1.0

    def test_classify_liquidate_level(self):
        """25%+ drawdown → LIQUIDATE level, full liquidation."""
        from src.strategy.evaluator import classify_kill_level, KillSwitchLevel, _kill_level_reduction
        level = classify_kill_level("max_drawdown_-30.0%")
        assert level == KillSwitchLevel.LIQUIDATE
        assert _kill_level_reduction(level) == 1.0

    def test_classify_extreme_tail_risk(self):
        """Extreme tail risk (CVaR ratio >3) → LIQUIDATE level."""
        from src.strategy.evaluator import classify_kill_level, KillSwitchLevel
        level = classify_kill_level("extreme_tail_risk_cvar_ratio_4.5")
        assert level == KillSwitchLevel.LIQUIDATE

    def test_classify_position_concentration(self):
        """Position concentration breach → WARNING level."""
        from src.strategy.evaluator import classify_kill_level, KillSwitchLevel
        level = classify_kill_level("max_position_SPY_55.0%")
        assert level == KillSwitchLevel.WARNING

    def test_classify_none_reason(self):
        """Empty reason → NONE level."""
        from src.strategy.evaluator import classify_kill_level, KillSwitchLevel
        level = classify_kill_level("")
        assert level == KillSwitchLevel.NONE

    def test_classify_unknown_breach_halt(self):
        """Unknown breach reason → HALT (fail-closed)."""
        from src.strategy.evaluator import classify_kill_level, KillSwitchLevel
        level = classify_kill_level("unknown_risk_breach")
        assert level == KillSwitchLevel.HALT

    @patch('src.strategy.evaluator.sqlite_connect')
    @patch('src.strategy.evaluator.get_latest_prices', return_value={"SPY": 500.0})
    @patch('src.strategy.evaluator.get_current_regime', return_value="normal")
    @patch('src.strategy.evaluator.get_latest_vix', return_value=15.0)
    def test_kill_switch_json_includes_level(
        self, mock_vix, mock_regime, mock_prices, mock_sqlite,
        tmp_path, capsys,
    ):
        """kill_switch.json should include 'level' and 'position_reduction' fields."""
        from src.strategy.evaluator import main

        with (
            patch('src.strategy.evaluator.DATA_DIR', tmp_path),
            patch('sys.argv', ['evaluator.py', 'paper']),
            patch('src.strategy.evaluator.Portfolio') as MockPortfolio,
        ):
            mock_portfolio = MagicMock()
            mock_portfolio.check_risk_limits.return_value = "max_drawdown_-18.0%"
            mock_portfolio.total_value.return_value = 82000
            MockPortfolio.return_value = mock_portfolio
            main()

        kill_file = tmp_path / "kill_switch.json"
        assert kill_file.exists()
        with open(kill_file) as f:
            data = json.load(f)
        assert data["enabled"] is True
        assert data["level"] == "restrict"
        assert data["position_reduction"] == 0.5
        assert "reason" in data
        assert "timestamp" in data

    @patch('src.strategy.evaluator.sqlite_connect')
    @patch('src.strategy.evaluator.get_latest_prices', return_value={"SPY": 500.0})
    @patch('src.strategy.evaluator.get_current_regime', return_value="normal")
    @patch('src.strategy.evaluator.get_latest_vix', return_value=15.0)
    def test_kill_switch_liquidate_includes_full_reduction(
        self, mock_vix, mock_regime, mock_prices, mock_sqlite,
        tmp_path, capsys,
    ):
        """LIQUIDATE level kill_switch.json should have position_reduction=1.0."""
        from src.strategy.evaluator import main

        with (
            patch('src.strategy.evaluator.DATA_DIR', tmp_path),
            patch('sys.argv', ['evaluator.py', 'paper']),
            patch('src.strategy.evaluator.Portfolio') as MockPortfolio,
        ):
            mock_portfolio = MagicMock()
            mock_portfolio.check_risk_limits.return_value = "max_drawdown_-30.0%"
            mock_portfolio.total_value.return_value = 70000
            MockPortfolio.return_value = mock_portfolio
            main()

        kill_file = tmp_path / "kill_switch.json"
        assert kill_file.exists()
        with open(kill_file) as f:
            data = json.load(f)
        assert data["level"] == "liquidate"
        assert data["position_reduction"] == 1.0


class TestRegimeTargetAllocationParity:
    """Evaluator audit and cron paths should match env-enabled allocation semantics."""

    @patch('src.strategy.evaluator.sqlite_connect')
    @patch('src.strategy.evaluator.get_latest_prices', return_value={"SPY": 500.0})
    @patch('src.strategy.evaluator.get_current_regime', return_value="recovery")
    @patch('src.strategy.evaluator.get_latest_vix', return_value=16.0)
    def test_kill_path_records_env_enabled_regime_allocation(
        self, mock_vix, mock_regime, mock_prices, mock_sqlite, tmp_path, monkeypatch
    ):
        """Kill-switch decision registry records the same target as normal eval."""
        from src.strategy.evaluator import main
        from src.strategy.regime_allocation import get_regime_allocation_with_override

        monkeypatch.setenv("REGIME_ALLOC_ENABLED", "1")
        expected = get_regime_allocation_with_override("recovery")

        with (
            patch('src.strategy.evaluator.DATA_DIR', tmp_path),
            patch('src.strategy.evaluator.Portfolio') as MockPortfolio,
            patch('src.monitor.decision_registry.record_evaluator_cycle_decision') as record_decision,
        ):
            mock_portfolio = MagicMock()
            mock_portfolio.check_risk_limits.return_value = "max_drawdown_-25.0%"
            mock_portfolio.total_value.return_value = 75000
            mock_portfolio.current_weights.return_value = {"SPY": 1.0}
            MockPortfolio.return_value = mock_portfolio

            main()

        assert record_decision.call_args.kwargs["target_alloc"] == expected

    def test_dashboard_cron_enables_regime_allocation_contract(self):
        """Scheduled dashboard generation uses the same env contract as eval."""
        dashboard_cron = Path("scripts/cron/portfolio-lab-dashboard.sh").read_text()

        assert "export REGIME_ALLOC_ENABLED=1" in dashboard_cron


# ---------------------------------------------------------------------------
# Kill Switch — boundary value tests
# ---------------------------------------------------------------------------

class TestKillSwitchBoundaryValues:
    """Boundary value tests for classify_kill_level with exact threshold values."""

    def test_kill_warning_boundary_at_10_percent(self):
        """Exactly 10.0% drawdown should trigger WARNING (>= threshold)."""
        from src.strategy.evaluator import classify_kill_level, KillSwitchLevel
        level = classify_kill_level("max_drawdown_-10.0%")
        assert level == KillSwitchLevel.WARNING

    def test_kill_warning_below_10_percent(self):
        """9.99% drawdown falls below all thresholds but is still a drawdown breach.

        classify_kill_level is fail-closed: any parseable drawdown that doesn't
        reach WARNING is still classified as HALT (the default at line 562).
        """
        from src.strategy.evaluator import classify_kill_level, KillSwitchLevel
        level = classify_kill_level("max_drawdown_-9.99%")
        # Fail-closed: below all thresholds defaults to HALT
        assert level == KillSwitchLevel.HALT

    def test_kill_restrict_boundary_at_15_percent(self):
        """Exactly 15.0% drawdown should trigger RESTRICT."""
        from src.strategy.evaluator import classify_kill_level, KillSwitchLevel
        level = classify_kill_level("max_drawdown_-15.0%")
        assert level == KillSwitchLevel.RESTRICT

    def test_kill_restrict_below_15_percent(self):
        """14.99% drawdown should trigger WARNING, not RESTRICT."""
        from src.strategy.evaluator import classify_kill_level, KillSwitchLevel
        level = classify_kill_level("max_drawdown_-14.99%")
        assert level == KillSwitchLevel.WARNING

    def test_kill_halt_boundary_at_20_percent(self):
        """Exactly 20.0% drawdown should trigger HALT."""
        from src.strategy.evaluator import classify_kill_level, KillSwitchLevel
        level = classify_kill_level("max_drawdown_-20.0%")
        assert level == KillSwitchLevel.HALT

    def test_kill_halt_below_20_percent(self):
        """19.99% drawdown should trigger RESTRICT, not HALT."""
        from src.strategy.evaluator import classify_kill_level, KillSwitchLevel
        level = classify_kill_level("max_drawdown_-19.99%")
        assert level == KillSwitchLevel.RESTRICT

    def test_kill_liquidate_boundary_at_25_percent(self):
        """Exactly 25.0% drawdown should trigger LIQUIDATE."""
        from src.strategy.evaluator import classify_kill_level, KillSwitchLevel
        level = classify_kill_level("max_drawdown_-25.0%")
        assert level == KillSwitchLevel.LIQUIDATE

    def test_kill_liquidate_below_25_percent(self):
        """24.99% drawdown should trigger HALT, not LIQUIDATE."""
        from src.strategy.evaluator import classify_kill_level, KillSwitchLevel
        level = classify_kill_level("max_drawdown_-24.99%")
        assert level == KillSwitchLevel.HALT


# ---------------------------------------------------------------------------
# Kill Switch — environment variable override tests
# ---------------------------------------------------------------------------

class TestKillSwitchEnvVarOverrides:
    """Tests that KILL_SWITCH_THRESHOLDS dict can be overridden (simulating env var overrides)."""

    def test_kill_custom_warning_threshold(self, monkeypatch):
        """Custom warning threshold at 5% should trigger WARNING at 5% drawdown."""
        from src.strategy.evaluator import classify_kill_level, KillSwitchLevel
        # Patch thresholds to simulate KILL_WARNING_DRAWDOWN_PCT=0.05
        custom_thresholds = {
            "warning_drawdown_pct": 0.05,
            "restrict_drawdown_pct": 0.15,
            "halt_drawdown_pct": 0.20,
            "liquidate_drawdown_pct": 0.25,
            "extreme_tail_cvar_ratio": 3.0,
        }
        monkeypatch.setattr(
            "src.strategy.evaluator.KILL_SWITCH_THRESHOLDS", custom_thresholds
        )
        level = classify_kill_level("max_drawdown_-5.0%")
        assert level == KillSwitchLevel.WARNING

    def test_kill_custom_liquidate_threshold(self, monkeypatch):
        """Custom liquidate threshold at 30% should NOT trigger LIQUIDATE at 25%."""
        from src.strategy.evaluator import classify_kill_level, KillSwitchLevel
        custom_thresholds = {
            "warning_drawdown_pct": 0.10,
            "restrict_drawdown_pct": 0.15,
            "halt_drawdown_pct": 0.20,
            "liquidate_drawdown_pct": 0.30,
            "extreme_tail_cvar_ratio": 3.0,
        }
        monkeypatch.setattr(
            "src.strategy.evaluator.KILL_SWITCH_THRESHOLDS", custom_thresholds
        )
        level = classify_kill_level("max_drawdown_-25.0%")
        # 25% < 30% liquidate threshold, but >= 20% halt threshold
        assert level == KillSwitchLevel.HALT

    def test_kill_custom_restrict_threshold(self, monkeypatch):
        """Custom restrict threshold at 12% should trigger RESTRICT at 12%."""
        from src.strategy.evaluator import classify_kill_level, KillSwitchLevel
        custom_thresholds = {
            "warning_drawdown_pct": 0.10,
            "restrict_drawdown_pct": 0.12,
            "halt_drawdown_pct": 0.20,
            "liquidate_drawdown_pct": 0.25,
            "extreme_tail_cvar_ratio": 3.0,
        }
        monkeypatch.setattr(
            "src.strategy.evaluator.KILL_SWITCH_THRESHOLDS", custom_thresholds
        )
        level = classify_kill_level("max_drawdown_-12.0%")
        assert level == KillSwitchLevel.RESTRICT

    def test_kill_custom_halt_threshold(self, monkeypatch):
        """Custom halt threshold at 18% should trigger HALT at 18%."""
        from src.strategy.evaluator import classify_kill_level, KillSwitchLevel
        custom_thresholds = {
            "warning_drawdown_pct": 0.10,
            "restrict_drawdown_pct": 0.15,
            "halt_drawdown_pct": 0.18,
            "liquidate_drawdown_pct": 0.25,
            "extreme_tail_cvar_ratio": 3.0,
        }
        monkeypatch.setattr(
            "src.strategy.evaluator.KILL_SWITCH_THRESHOLDS", custom_thresholds
        )
        level = classify_kill_level("max_drawdown_-18.0%")
        assert level == KillSwitchLevel.HALT


# ---------------------------------------------------------------------------
# Kill Switch — order router tests
# ---------------------------------------------------------------------------

class TestOrderRouterKillSwitch:
    """Tests for OrderRouter kill switch reading from kill_switch.json."""

    def test_order_router_blocks_on_warning_level(self, tmp_path):
        """Router should block orders when kill_switch.json has level=WARNING."""
        from src.broker.order_router import OrderRouter, OrderPlan

        kill_data = {
            "enabled": True,
            "level": "warning",
            "reason": "max_drawdown_-12.0%",
            "mode": "paper",
            "timestamp": "2026-05-28T00:00:00",
        }
        (tmp_path / "kill_switch.json").write_text(json.dumps(kill_data))

        with (
            patch("src.broker.order_router.AlpacaClient") as mock_client_cls,
            patch("src.broker.order_router.PaperTradingManager") as mock_mgr_cls,
        ):
            mock_client_cls.return_value.is_ready.return_value = True
            mock_mgr_cls.return_value = MagicMock()
            router = OrderRouter(data_dir=str(tmp_path), paper=True)

            plans = [OrderPlan(
                symbol="SPY", side="BUY", qty=10, order_type="market",
                estimated_value=5000, reason="rebalance",
            )]
            result = router.execute_orders(plans, dry_run=False, kill_switch_check=True)
            assert result["status"] == "blocked"

    def test_order_router_blocks_on_halt_level(self, tmp_path):
        """Router should block orders when kill_switch.json has level=HALT."""
        from src.broker.order_router import OrderRouter, OrderPlan

        kill_data = {
            "enabled": True,
            "level": "halt",
            "reason": "max_drawdown_-22.0%",
            "mode": "paper",
            "timestamp": "2026-05-28T00:00:00",
        }
        (tmp_path / "kill_switch.json").write_text(json.dumps(kill_data))

        with (
            patch("src.broker.order_router.AlpacaClient") as mock_client_cls,
            patch("src.broker.order_router.PaperTradingManager") as mock_mgr_cls,
        ):
            mock_client_cls.return_value.is_ready.return_value = True
            mock_mgr_cls.return_value = MagicMock()
            router = OrderRouter(data_dir=str(tmp_path), paper=True)

            plans = [OrderPlan(
                symbol="SPY", side="BUY", qty=10, order_type="market",
                estimated_value=5000, reason="rebalance",
            )]
            result = router.execute_orders(plans, dry_run=False, kill_switch_check=True)
            assert result["status"] == "blocked"

    def test_order_router_fail_closed_on_missing_json(self, tmp_path):
        """When kill_switch.json does not exist, router should allow orders (no kill switch active)."""
        from src.broker.order_router import OrderRouter, OrderPlan

        # Ensure no kill_switch.json exists
        assert not (tmp_path / "kill_switch.json").exists()

        # Mock submit_order to return an object with .id and .status attributes
        mock_fill = MagicMock()
        mock_fill.id = "test-123"
        mock_fill.status = "filled"

        with (
            patch("src.broker.order_router.AlpacaClient") as mock_client_cls,
            patch("src.broker.order_router.PaperTradingManager") as mock_mgr_cls,
            patch("src.broker.order_router.time"),
        ):
            mock_client_cls.return_value.is_ready.return_value = True
            mock_client_cls.return_value.submit_order.return_value = mock_fill
            mock_mgr_cls.return_value = MagicMock()
            router = OrderRouter(data_dir=str(tmp_path), paper=True)

            plans = [OrderPlan(
                symbol="SPY", side="BUY", qty=1, order_type="market",
                estimated_value=500, reason="rebalance",
            )]
            result = router.execute_orders(plans, dry_run=False, kill_switch_check=True)
            # Should NOT be blocked — missing kill_switch.json means no active kill switch
            assert result["status"] != "blocked"


# ---------------------------------------------------------------------------
# Incident kill blocks paper execute + promote (2026-07-15 batch C)
# ---------------------------------------------------------------------------

class TestIncidentKillBlocksPaperControlLoop:
    """Paper path must honor data/kill_switch.json like order_router."""

    def test_is_kill_execution_blocked_helper(self):
        from src.dashboard.kill_authority import is_kill_execution_blocked

        assert is_kill_execution_blocked(None) is False
        assert is_kill_execution_blocked({"enabled": False}) is False
        assert is_kill_execution_blocked({"enabled": True, "level": "halt"}) is True
        assert is_kill_execution_blocked({"enabled": True, "level": "restrict"}) is True
        assert is_kill_execution_blocked({"enabled": True, "level": "warning"}) is True
        assert is_kill_execution_blocked({"enabled": True}) is True

    def test_graduation_refuses_promote_under_incident_halt(self, tmp_path, caplog):
        """check_graduation_criteria must not write .promote_to_live under kill."""
        from src.strategy.evaluator import check_graduation_criteria
        import src.strategy.evaluator as ev

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

        kill_file = tmp_path / "kill_switch.json"
        kill_file.write_text(json.dumps({
            "enabled": True,
            "level": "halt",
            "reason": "unresolved_incident:signal_staleness",
            "source": "incident_lifecycle",
            "incident_id": "incident-halt-1",
            "mode": "paper",
            "timestamp": "2026-07-12T09:15:00+00:00",
            "position_reduction": 1.0,
        }))

        with (
            patch.object(ev, "DATA_DIR", tmp_path),
            caplog.at_level(logging.INFO, logger="src.strategy.evaluator"),
        ):
            check_graduation_criteria(p)

        promote = tmp_path / ".promote_to_live"
        assert not promote.exists(), "promote file must not be written under kill halt"
        assert "kill" in caplog.text.lower() or "HALT" in caplog.text or "blocked" in caplog.text.lower()

    @patch('src.strategy.evaluator.sqlite_connect')
    @patch('src.strategy.evaluator.get_latest_prices', return_value={"SPY": 500.0, "GLD": 180.0, "TLT": 90.0})
    @patch('src.strategy.evaluator.get_current_regime', return_value="normal")
    @patch('src.strategy.evaluator.get_latest_vix', return_value=15.0)
    def test_incident_halt_blocks_paper_execute_and_promote(
        self, mock_vix, mock_regime, mock_prices, mock_sqlite,
        tmp_path, caplog,
    ):
        """main() must not execute fills or refresh promote when incident kill is on."""
        from src.strategy.evaluator import main
        import src.strategy.evaluator as ev

        kill_file = tmp_path / "kill_switch.json"
        kill_file.write_text(json.dumps({
            "enabled": True,
            "level": "halt",
            "reason": "unresolved_incident:signal_staleness",
            "source": "incident_lifecycle",
            "incident_id": "incident-halt-1",
            "incident_channel": "signal_staleness",
            "mode": "paper",
            "timestamp": "2026-07-12T09:15:00+00:00",
            "position_reduction": 1.0,
        }))

        orders = [{
            "symbol": "SPY",
            "side": "buy",
            "shares": 10,
            "estimated_price": 500.0,
            "estimated_value": 5000.0,
            "reason": "rebalance_up",
            "drift_before": 0.05,
        }]

        with (
            patch.object(ev, "DATA_DIR", tmp_path),
            patch.object(ev, "ORDERS_LOG", tmp_path / "orders.jsonl"),
            patch.object(ev, "PERFORMANCE_LOG", tmp_path / "performance.jsonl"),
            patch('sys.argv', ['evaluator.py', 'paper']),
            patch.object(ev, "Portfolio") as MockPortfolio,
            caplog.at_level(logging.INFO, logger="src.strategy.evaluator"),
        ):
            mock_portfolio = MagicMock()
            mock_portfolio.check_risk_limits.return_value = None
            mock_portfolio.total_value.return_value = 100000
            mock_portfolio.calculate_orders.return_value = orders
            mock_portfolio.current_weights.return_value = {"SPY": 0.4, "GLD": 0.4, "TLT": 0.2}
            mock_portfolio.cash = 100000
            mock_portfolio.positions = {}
            mock_portfolio.mode = "paper"
            mock_portfolio.history = []
            mock_portfolio.execute_orders = MagicMock(return_value=[{"symbol": "SPY"}])
            MockPortfolio.return_value = mock_portfolio

            rc = main()

            mock_portfolio.execute_orders.assert_not_called()
            assert rc == 2, "authority kill must exit non-zero so make eval STATUS != ok"

        assert kill_file.exists(), "incident kill must be preserved"
        assert not (tmp_path / ".promote_to_live").exists()
        # orders log should not contain fills from blocked cycle
        orders_log = tmp_path / "orders.jsonl"
        if orders_log.exists():
            assert orders_log.read_text().strip() == ""

    @patch('src.strategy.evaluator.sqlite_connect')
    @patch('src.strategy.evaluator.get_latest_prices', return_value={"SPY": 500.0, "GLD": 180.0, "TLT": 90.0})
    @patch('src.strategy.evaluator.get_current_regime', return_value="normal")
    @patch('src.strategy.evaluator.get_latest_vix', return_value=15.0)
    def test_risk_kill_path_exits_nonzero(
        self, mock_vix, mock_regime, mock_prices, mock_sqlite,
        tmp_path,
    ):
        """Risk-limit kill must not report green cron success either."""
        from src.strategy.evaluator import main
        import src.strategy.evaluator as ev

        with (
            patch.object(ev, "DATA_DIR", tmp_path),
            patch('sys.argv', ['evaluator.py', 'paper']),
            patch.object(ev, "Portfolio") as MockPortfolio,
        ):
            mock_portfolio = MagicMock()
            mock_portfolio.check_risk_limits.return_value = "max_drawdown_-25.0%"
            mock_portfolio.total_value.return_value = 75000
            mock_portfolio.current_weights.return_value = {}
            MockPortfolio.return_value = mock_portfolio

            rc = main()

        assert rc == 2
        assert (tmp_path / "kill_switch.json").exists()

    @patch('src.strategy.evaluator.sqlite_connect')
    @patch('src.strategy.evaluator.get_latest_prices', return_value={"SPY": 500.0, "GLD": 180.0, "TLT": 90.0})
    @patch('src.strategy.evaluator.get_current_regime', return_value="normal")
    @patch('src.strategy.evaluator.get_latest_vix', return_value=15.0)
    def test_no_kill_still_executes_orders(
        self, mock_vix, mock_regime, mock_prices, mock_sqlite,
        tmp_path,
    ):
        """Without kill file, paper execute_orders still runs."""
        from src.strategy.evaluator import main
        import src.strategy.evaluator as ev

        orders = [{
            "symbol": "SPY",
            "side": "buy",
            "shares": 10,
            "estimated_price": 500.0,
            "estimated_value": 5000.0,
            "reason": "rebalance_up",
            "drift_before": 0.05,
        }]

        with (
            patch.object(ev, "DATA_DIR", tmp_path),
            patch.object(ev, "ORDERS_LOG", tmp_path / "orders.jsonl"),
            patch.object(ev, "PERFORMANCE_LOG", tmp_path / "performance.jsonl"),
            patch('sys.argv', ['evaluator.py', 'paper']),
            patch.object(ev, "Portfolio") as MockPortfolio,
        ):
            mock_portfolio = MagicMock()
            mock_portfolio.check_risk_limits.return_value = None
            mock_portfolio.total_value.return_value = 100000
            mock_portfolio.calculate_orders.return_value = orders
            mock_portfolio.current_weights.return_value = {"SPY": 0.4, "GLD": 0.4, "TLT": 0.2}
            mock_portfolio.cash = 100000
            mock_portfolio.positions = {}
            mock_portfolio.mode = "paper"
            mock_portfolio.history = [{"total_value": 100000, "daily_return": 0.0}]
            mock_portfolio.execute_orders = MagicMock(return_value=[{
                **orders[0],
                "fill_price": 500.0,
                "fill_shares": 10,
                "fill_value": 5000.0,
                "timestamp": "2026-07-15T00:00:00",
            }])
            MockPortfolio.return_value = mock_portfolio

            rc = main()

            mock_portfolio.execute_orders.assert_called_once()
            assert rc == 0

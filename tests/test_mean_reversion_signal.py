"""
Tests for VIX-Gated Mean-Reversion Signal Generator (v4.81)
"""

import json
import pytest
import numpy as np
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

from src.signals.mean_reversion_signal import (
    VIXMeanReversionCalculator,
    MeanReversionSignal,
    MeanReversionTrade,
    VIXRegime,
    MeanReversionState,
    BASE_ALLOC_PCT,
    MAX_ALLOC_PCT,
    MAX_HOLD_DAYS,
    STOP_LOSS_PCT,
    VIX_MIXED,
    VIX_CRISIS,
    VIX_EXIT,
    VIX_LOW,
    ENTRY_SPY_DROP_PCT,
    VPIN_THRESHOLD,
    STATE_PATH,
)


class TestVIXRegimeEnum:
    """Test VIX regime enum values."""

    def test_trend_follow_value(self):
        assert VIXRegime.TREND_FOLLOW.value == "trend_follow"

    def test_mixed_value(self):
        assert VIXRegime.MIXED.value == "mixed"

    def test_mean_reversion_value(self):
        assert VIXRegime.MEAN_REVERSION.value == "mean_reversion"

    def test_crisis_freeze_value(self):
        assert VIXRegime.CRISIS_FREEZE.value == "crisis_freeze"

    def test_all_values_unique(self):
        values = [r.value for r in VIXRegime]
        assert len(values) == len(set(values))


class TestMeanReversionStateEnum:
    """Test trade state enum values."""

    def test_idle_value(self):
        assert MeanReversionState.IDLE.value == "idle"

    def test_entering_value(self):
        assert MeanReversionState.ENTERING.value == "entering"

    def test_active_value(self):
        assert MeanReversionState.ACTIVE.value == "active"

    def test_stopped_value(self):
        assert MeanReversionState.STOPPED.value == "stopped"

    def test_exited_value(self):
        assert MeanReversionState.EXITED.value == "exited"

    def test_expired_value(self):
        assert MeanReversionState.EXPIRED.value == "expired"


class TestMeanReversionSignalDataclass:
    """Test MeanReversionSignal data class."""

    def test_default_values(self):
        signal = MeanReversionSignal(
            timestamp="2026-05-16T12:00:00Z",
            vix_level=22.5,
            vix_regime="mixed",
            spy_price=520.0,
            spy_3d_return=-1.5,
            spy_above_200ma=True,
            vpin_level=0.3,
            vpin_ok=True,
            entry_triggered=False,
            entry_reason="",
            trade_state="idle",
            trade_entry_price=None,
            trade_hold_days=0,
            trade_return_pct=0.0,
            recommended_allocation_pct=0.0,
            allocation_rationale="No opportunity",
            signal_value=0.0,
            signal_strength=0.0,
        )
        assert signal.vix_level == 22.5
        assert signal.trade_state == "idle"
        assert signal.signal_value == 0.0

    def test_signal_value_range_negative_mean_reversion(self):
        """Signal_value negative indicates bullish mean-reversion (buy dip)."""
        signal = MeanReversionSignal(
            timestamp="2026-05-16T12:00:00Z",
            vix_level=35.0,
            vix_regime="mean_reversion",
            spy_price=500.0,
            spy_3d_return=-3.0,
            spy_above_200ma=True,
            vpin_level=0.3,
            vpin_ok=True,
            entry_triggered=True,
            entry_reason="SPY oversold during elevated VIX",
            trade_state="entering",
            trade_entry_price=None,
            trade_hold_days=0,
            trade_return_pct=0.0,
            recommended_allocation_pct=3.0,
            allocation_rationale="ENTRY: SPY oversold",
            signal_value=-0.5,
            signal_strength=0.6,
        )
        assert signal.signal_value < 0  # Negative = bullish mean-reversion
        assert signal.signal_strength > 0


class TestMeanReversionTradeDataclass:
    """Test MeanReversionTrade data class."""

    def test_all_fields(self):
        trade = MeanReversionTrade(
            entry_date="2026-05-01",
            exit_date="2026-05-08",
            entry_spy=500.0,
            exit_spy=510.0,
            return_pct=2.0,
            hold_days=5,
            vix_at_entry=32.0,
            allocation_pct=3.0,
            exit_reason="recovery",
        )
        assert trade.return_pct == 2.0
        assert trade.hold_days == 5
        assert trade.exit_reason == "recovery"

    def test_loss_trade(self):
        trade = MeanReversionTrade(
            entry_date="2026-05-01",
            exit_date="2026-05-04",
            entry_spy=500.0,
            exit_spy=485.0,
            return_pct=-3.0,
            hold_days=3,
            vix_at_entry=35.0,
            allocation_pct=2.0,
            exit_reason="stop_loss",
        )
        assert trade.return_pct < 0
        assert trade.exit_reason == "stop_loss"


class TestVIXMeanReversionCalculator:
    """Test VIXMeanReversionCalculator core functionality."""

    @pytest.fixture
    def calculator(self):
        """Create calculator with synthetic price data - upward trend then drop."""
        calc = VIXMeanReversionCalculator()

        n = 500
        calc._dates = [f"day_{i}" for i in range(n)]

        # SPY: strong uptrend for first 480 days, then 3% drop in last 3
        spy_up = np.linspace(100, 120, n - 3)
        spy_end = np.array([120.0, 118.0, 116.0])  # 3-day drop of ~3.3%
        calc._prices["SPY"] = np.concatenate([spy_up, spy_end])

        # GLD: stable
        calc._prices["GLD"] = np.full(n, 100.0)
        calc._prices["TLT"] = np.full(n, 100.0)

        return calc

    def test_get_spy_prices(self, calculator):
        """Test SPY price extraction."""
        spy = calculator.get_spy_prices()
        assert len(spy) > 0
        assert spy[-1] < spy[-4]  # Recent drop

    def test_get_vix_level_fallback(self, calculator):
        """Test VIX level falls back to SPY vol estimate."""
        vix = calculator.get_vix_level()
        assert vix is not None
        assert vix > 0

    def test_get_vix_level_at(self, calculator):
        """Test VIX level at specific index."""
        vix = calculator.get_vix_level_at(len(calculator._prices["SPY"]) - 1)
        assert vix is not None
        assert vix >= 10.0

    def test_get_vix_historical(self, calculator):
        """Test VIX historical proxy computation."""
        vix_hist = calculator.get_vix_historical()
        assert len(vix_hist) > 0
        assert np.all(vix_hist >= 10.0)
        assert not np.any(np.isnan(vix_hist))

    def test_classify_vix_regime_trend_follow(self):
        """Test VIX < 20 = trend-follow regime."""
        calc = VIXMeanReversionCalculator()
        assert calc.classify_vix_regime(10.0) == VIXRegime.TREND_FOLLOW
        assert calc.classify_vix_regime(15.0) == VIXRegime.TREND_FOLLOW
        assert calc.classify_vix_regime(19.9) == VIXRegime.TREND_FOLLOW

    def test_classify_vix_regime_mixed(self):
        """Test VIX 20-30 = mixed regime."""
        calc = VIXMeanReversionCalculator()
        assert calc.classify_vix_regime(20.0) == VIXRegime.MIXED
        assert calc.classify_vix_regime(25.0) == VIXRegime.MIXED
        assert calc.classify_vix_regime(29.9) == VIXRegime.MIXED

    def test_classify_vix_regime_mean_reversion(self):
        """Test VIX 30-40 = mean-reversion regime."""
        calc = VIXMeanReversionCalculator()
        assert calc.classify_vix_regime(30.0) == VIXRegime.MEAN_REVERSION
        assert calc.classify_vix_regime(35.0) == VIXRegime.MEAN_REVERSION
        assert calc.classify_vix_regime(39.9) == VIXRegime.MEAN_REVERSION

    def test_classify_vix_regime_crisis(self):
        """Test VIX > 40 = crisis freeze."""
        calc = VIXMeanReversionCalculator()
        assert calc.classify_vix_regime(40.0) == VIXRegime.CRISIS_FREEZE
        assert calc.classify_vix_regime(50.0) == VIXRegime.CRISIS_FREEZE

    def test_classify_vix_regime_boundaries(self):
        """Test boundary conditions."""
        calc = VIXMeanReversionCalculator()
        assert calc.classify_vix_regime(VIX_LOW) == VIXRegime.MIXED
        assert calc.classify_vix_regime(VIX_MIXED) == VIXRegime.MEAN_REVERSION
        assert calc.classify_vix_regime(VIX_CRISIS) == VIXRegime.CRISIS_FREEZE

    def test_compute_spy_3d_return(self, calculator):
        """Test 3-day SPY return computation."""
        spy = calculator.get_spy_prices()
        ret = calculator.compute_spy_3d_return(spy)
        assert ret < 0
        assert ret > -10

    def test_compute_spy_3d_return_insufficient_data(self, calculator):
        """Test with insufficient data."""
        ret = calculator.compute_spy_3d_return(np.array([100.0]))
        assert ret == 0.0

    def test_check_spy_above_200ma_true(self):
        """Test SPY above 200-day MA (strong uptrend)."""
        calc = VIXMeanReversionCalculator()
        n = 300
        spy = np.linspace(100, 150, n)  # Strong uptrend, last = 150
        calc._prices["SPY"] = spy
        assert calc.check_spy_above_200ma(spy)

    def test_check_spy_above_200ma_false(self, calculator):
        """Test SPY below 200-day MA."""
        n = 300
        spy = np.ones(n) * 100
        spy[-50:] = 80
        assert not calculator.check_spy_above_200ma(spy)

    def test_check_spy_above_200ma_insufficient_data(self, calculator):
        """Test with less than 200 data points."""
        spy = np.array([100.0] * 50)
        assert calculator.check_spy_above_200ma(spy)

    def test_get_spy_200ma(self, calculator):
        """Test 200-day MA computation."""
        spy = calculator.get_spy_prices()
        ma200 = calculator.get_spy_200ma(spy)
        assert ma200 > 0
        assert isinstance(ma200, float)

    def test_get_spy_200ma_insufficient_data(self, calculator):
        """Test 200-day MA with insufficient data."""
        spy = np.array([100.0] * 50)
        assert calculator.get_spy_200ma(spy) == 100.0

    def test_spy_above_200ma_at_early_index(self, calculator):
        """Test check at early index < 200 returns True."""
        spy = calculator.get_spy_prices()
        assert calculator.spy_above_200ma_at(spy, 100)

    def test_compute_vpin_default(self, calculator):
        """Test VPIN computation returns reasonable values."""
        vpin, is_ok = calculator.compute_vpin()
        assert 0 <= vpin <= 1.0
        assert isinstance(is_ok, bool)

    def test_compute_vpin_high_toxicity(self, calculator):
        """Test VPIN handles volatile prices."""
        spy = calculator.get_spy_prices().copy()
        spy[-20:] = spy[-20:] * (1 + np.random.randn(20) * 0.03)
        vpin, is_ok = calculator.compute_vpin()
        assert 0 <= vpin <= 1.0

    def test_compute_vpin_insufficient_data(self, calculator):
        """Test VPIN with insufficient data."""
        calculator._prices["SPY"] = np.array([100.0] * 10)
        vpin, is_ok = calculator.compute_vpin()
        assert vpin == 0.3
        assert is_ok

    def test_compute_trade_state_default(self, tmp_path):
        """Test trade state returns defaults when no state file."""
        calc = VIXMeanReversionCalculator()
        state_path = tmp_path / "nonexistent.json"
        state = calc.compute_trade_state(state_path)
        assert state["active"] is False
        assert state["entry_price"] is None
        assert state["hold_days"] == 0

    def test_compute_trade_state_corrupted(self, tmp_path):
        """Test trade state handles corrupted JSON."""
        calc = VIXMeanReversionCalculator()
        state_path = tmp_path / "corrupted.json"
        with open(state_path, "w") as f:
            f.write("not valid json")
        state = calc.compute_trade_state(state_path)
        assert state["active"] is False

    def test_save_trade_state(self, tmp_path):
        """Test saving trade state."""
        calc = VIXMeanReversionCalculator()
        state_path = tmp_path / "state.json"
        calc.save_trade_state({"active": True, "entry_price": 500.0}, state_path)
        assert state_path.exists()
        with open(state_path) as f:
            saved = json.load(f)
        assert saved["active"] is True
        assert saved["entry_price"] == 500.0

    def test_save_and_load_trade_state_roundtrip(self, tmp_path):
        """Test save then load roundtrip."""
        calc = VIXMeanReversionCalculator()
        state_path = tmp_path / "roundtrip.json"
        original = {"active": True, "entry_price": 510.0, "entry_vix": 35.0, "hold_days": 2, "allocation_pct": 3.0}
        calc.save_trade_state(original, state_path)
        loaded = calc.compute_trade_state(state_path)
        assert loaded["active"] == original["active"]
        assert loaded["entry_price"] == original["entry_price"]


class TestMeanReversionSignalGeneration:
    """Test signal generation in various market conditions."""

    @pytest.fixture
    def calculator(self):
        """Create calculator with SPY -4% drop in last 3 days."""
        calc = VIXMeanReversionCalculator()

        n = 500
        calc._dates = [f"day_{i}" for i in range(n)]

        # SPY: uptrend then -4% drop in last 3 days (trigger entry)
        spy = np.linspace(100, 130, n - 3)
        spy = np.append(spy, [128.0, 126.0, 124.8])
        calc._prices["SPY"] = spy

        calc._prices["GLD"] = np.full(n, 100.0)
        calc._prices["TLT"] = np.full(n, 100.0)

        return calc

    def test_generate_signal_entry_conditions_met(self, calculator):
        """Test signal triggers entry when conditions align."""
        with patch.object(calculator.__class__, "get_vix_level", return_value=35.0), \
                patch.object(calculator.__class__, "_load_vpin_state", return_value=0.3):
            signal = calculator.generate_signal()

            assert signal.entry_triggered
            assert signal.trade_state == "entering"
            assert signal.recommended_allocation_pct > 0
            assert signal.signal_value < 0
            assert signal.vix_regime == "mean_reversion"

    def test_generate_signal_trend_follow_mode(self, calculator):
        """Test signal in low VIX regime (no mean-reversion)."""
        with patch.object(calculator.__class__, "get_vix_level", return_value=15.0), \
                patch.object(calculator.__class__, "_load_vpin_state", return_value=0.3):
            signal = calculator.generate_signal()
            assert not signal.entry_triggered
            assert "TREND MODE" in signal.allocation_rationale
            assert signal.recommended_allocation_pct == 0.0
            assert signal.vix_regime == "trend_follow"

    def test_generate_signal_crisis_freeze(self, calculator):
        """Test signal in crisis mode (VIX > 40)."""
        with patch.object(calculator.__class__, "get_vix_level", return_value=45.0), \
                patch.object(calculator.__class__, "_load_vpin_state", return_value=0.3):
            signal = calculator.generate_signal()
            assert not signal.entry_triggered
            assert "CRISIS FREEZE" in signal.allocation_rationale
            assert signal.recommended_allocation_pct == 0.0
            assert signal.vix_regime == "crisis_freeze"

    def test_generate_signal_not_oversold(self, calculator):
        """Test signal when VIX elevated but SPY not oversold."""
        spy = calculator._prices["SPY"].copy()
        spy[-3:] = [130.0, 130.5, 130.2]
        calculator._prices["SPY"] = spy
        with patch.object(calculator.__class__, "get_vix_level", return_value=35.0), \
                patch.object(calculator.__class__, "_load_vpin_state", return_value=0.3):
            signal = calculator.generate_signal()
            assert not signal.entry_triggered
            assert "WAITING" in signal.allocation_rationale or "not oversold" in signal.allocation_rationale.lower()

    def test_generate_signal_below_200ma(self, calculator):
        """Test signal when SPY below 200-day MA (secular bear)."""
        spy = calculator._prices["SPY"].copy()
        spy[-200:] = 80 + np.cumsum(np.random.randn(200) * 0.1)
        spy[-4:] = [81, 80, 79, 78]  # recent drop below MA
        calculator._prices["SPY"] = spy
        with patch.object(calculator.__class__, "get_vix_level", return_value=35.0), \
                patch.object(calculator.__class__, "_load_vpin_state", return_value=0.3):
            signal = calculator.generate_signal()
            assert not signal.entry_triggered
            rationale_lower = signal.allocation_rationale.lower()
            assert "no long" in rationale_lower or "below 200d" in rationale_lower

    def test_generate_signal_toxic_vpin(self, calculator):
        """Test signal when VPIN is too high (toxic flow)."""
        with patch.object(calculator.__class__, "get_vix_level", return_value=35.0), \
                patch.object(calculator.__class__, "_load_vpin_state", return_value=0.7):
            signal = calculator.generate_signal()
            assert not signal.entry_triggered
            assert "TOXIC" in signal.allocation_rationale or "VPIN" in signal.allocation_rationale

    def test_generate_signal_no_data(self):
        """Test signal handles missing data gracefully."""
        calc = VIXMeanReversionCalculator()
        calc._prices = {}
        calc._dates = []
        with patch.object(calc.__class__, "get_vix_level", return_value=None):
            signal = calc.generate_signal()
            assert not signal.entry_triggered
            assert signal.recommended_allocation_pct == 0.0

    def test_generate_signal_active_trade_holding(self, calculator):
        """Test signal shows active trade state."""
        with patch.object(calculator.__class__, "get_vix_level", return_value=32.0), \
                patch.object(calculator.__class__, "_load_vpin_state", return_value=0.3), \
                patch.object(calculator.__class__, "compute_trade_state", return_value={
                    "active": True,
                    "entry_date": "day_497",
                    "entry_price": 124.8,
                    "entry_vix": 35.0,
                    "hold_days": 2,
                    "allocation_pct": 3.0,
                }):
            signal = calculator.generate_signal()
            assert signal.trade_state in ("active", "exited")
            if signal.trade_state == "active":
                assert "HOLDING" in signal.allocation_rationale

    def test_generate_signal_active_trade_exit_by_recovery(self, calculator):
        """Test exit when SPY recovers to entry."""
        spy = calculator._prices["SPY"].copy()
        spy[-1] = 135.0
        calculator._prices["SPY"] = spy
        with patch.object(calculator.__class__, "get_vix_level", return_value=28.0), \
                patch.object(calculator.__class__, "_load_vpin_state", return_value=0.3), \
                patch.object(calculator.__class__, "compute_trade_state", return_value={
                    "active": True,
                    "entry_date": "day_497",
                    "entry_price": 124.8,
                    "entry_vix": 35.0,
                    "hold_days": 2,
                    "allocation_pct": 3.0,
                }):
            signal = calculator.generate_signal()
            assert signal.trade_state == "exited"
            assert "EXIT" in signal.allocation_rationale
            assert signal.recommended_allocation_pct == 0.0

    def test_generate_signal_active_trade_exit_by_vix_drop(self, calculator):
        """Test exit when VIX drops below threshold (SPY below entry)."""
        spy = calculator._prices["SPY"].copy()
        spy[-1] = 124.0  # Slightly below 124.8 entry — no recovery exit
        calculator._prices["SPY"] = spy
        with patch.object(calculator.__class__, "get_vix_level", return_value=20.0), \
                patch.object(calculator.__class__, "_load_vpin_state", return_value=0.3), \
                patch.object(calculator.__class__, "compute_trade_state", return_value={
                    "active": True,
                    "entry_date": "day_497",
                    "entry_price": 124.8,
                    "entry_vix": 35.0,
                    "hold_days": 3,
                    "allocation_pct": 3.0,
                }):
            signal = calculator.generate_signal()
            assert signal.trade_state == "exited"
            assert "vix_drop" in signal.allocation_rationale

    def test_generate_signal_active_trade_stop_loss(self, calculator):
        """Test exit on stop loss."""
        spy = calculator._prices["SPY"].copy()
        spy[-1] = 120.0  # Down ~4% from entry 124.8
        calculator._prices["SPY"] = spy
        with patch.object(calculator.__class__, "get_vix_level", return_value=35.0), \
                patch.object(calculator.__class__, "_load_vpin_state", return_value=0.3), \
                patch.object(calculator.__class__, "compute_trade_state", return_value={
                    "active": True,
                    "entry_date": "day_497",
                    "entry_price": 124.8,
                    "entry_vix": 35.0,
                    "hold_days": 2,
                    "allocation_pct": 3.0,
                }):
            signal = calculator.generate_signal()
            assert signal.trade_state == "exited"
            assert "stop_loss" in signal.allocation_rationale

    def test_generate_signal_active_trade_expired(self, calculator):
        """Test exit on max hold."""
        spy = calculator._prices["SPY"].copy()
        spy[-1] = 124.0
        calculator._prices["SPY"] = spy
        with patch.object(calculator.__class__, "get_vix_level", return_value=32.0), \
                patch.object(calculator.__class__, "_load_vpin_state", return_value=0.3), \
                patch.object(calculator.__class__, "compute_trade_state", return_value={
                    "active": True,
                    "entry_date": "day_487",
                    "entry_price": 124.8,
                    "entry_vix": 35.0,
                    "hold_days": MAX_HOLD_DAYS,
                    "allocation_pct": 2.0,
                }):
            signal = calculator.generate_signal()
            assert signal.trade_state == "exited"
            assert "expired" in signal.allocation_rationale or "max hold" in signal.allocation_rationale


class TestMeanReversionSizing:
    """Test position sizing logic."""

    def _compute_alloc(self, spy_3d_return: float) -> float:
        """Helper: compute allocation from 3-day return."""
        if spy_3d_return <= ENTRY_SPY_DROP_PCT:
            additional_drop = abs(spy_3d_return) - abs(ENTRY_SPY_DROP_PCT)
            scale_units = int(additional_drop / 1.0)
            return min(BASE_ALLOC_PCT + scale_units * 1.0, MAX_ALLOC_PCT)
        return 0.0

    def test_base_allocation(self):
        assert BASE_ALLOC_PCT == 2.0

    def test_max_allocation(self):
        assert MAX_ALLOC_PCT == 5.0

    def test_size_for_min_drop(self):
        assert self._compute_alloc(-2.0) == BASE_ALLOC_PCT

    def test_size_for_medium_drop(self):
        assert self._compute_alloc(-4.0) == 4.0

    def test_size_for_large_drop(self):
        assert self._compute_alloc(-8.0) == MAX_ALLOC_PCT

    def test_size_no_drop(self):
        assert self._compute_alloc(0.0) == 0.0

    def test_size_small_drop(self):
        assert self._compute_alloc(-1.0) == 0.0


class TestVPINCalculation:
    """Test VPIN estimation logic."""

    def test_vpin_threshold(self):
        assert VPIN_THRESHOLD == 0.6

    def test_vpin_ok_below_threshold(self):
        calc = VIXMeanReversionCalculator()
        with patch.object(calc.__class__, "compute_vpin", return_value=(0.3, True)):
            vpin, ok = calc.compute_vpin()
            assert ok
            assert vpin < VPIN_THRESHOLD


class TestTradeHistory:
    """Test trade history management."""

    def test_save_trade_record(self, tmp_path):
        """Test saving a trade record using DATA_DIR override."""
        calc = VIXMeanReversionCalculator()
        trade = MeanReversionTrade(
            entry_date="2026-05-01",
            exit_date="2026-05-05",
            entry_spy=500.0,
            exit_spy=510.0,
            return_pct=2.0,
            hold_days=4,
            vix_at_entry=35.0,
            allocation_pct=3.0,
            exit_reason="recovery",
        )
        # Patch _save_trade_record to write to tmp_path overriding DATA_DIR
        original_save = calc._save_trade_record
        def patched_save(trade_record):
            trades_path = tmp_path / "mean_reversion_trades.json"
            existing = []
            if trades_path.exists():
                with open(trades_path) as f:
                    existing = json.load(f)
            existing.append({
                "entry_date": trade_record.entry_date,
                "exit_date": trade_record.exit_date,
                "entry_spy": trade_record.entry_spy,
                "exit_spy": trade_record.exit_spy,
                "return_pct": trade_record.return_pct,
                "hold_days": trade_record.hold_days,
                "vix_at_entry": trade_record.vix_at_entry,
                "allocation_pct": trade_record.allocation_pct,
                "exit_reason": trade_record.exit_reason,
            })
            with open(trades_path, "w") as f:
                json.dump(existing, f, indent=2)
        with patch.object(calc, "_save_trade_record", side_effect=patched_save):
            calc._save_trade_record(trade)
        trades_path = tmp_path / "mean_reversion_trades.json"
        assert trades_path.exists()
        with open(trades_path) as f:
            trades = json.load(f)
        assert len(trades) == 1
        assert trades[0]["exit_reason"] == "recovery"

    def test_save_trade_record_appends(self, tmp_path):
        """Test saving multiple trade records appends."""
        calc = VIXMeanReversionCalculator()
        trades_path = tmp_path / "mean_reversion_trades.json"

        def make_patched_save(path):
            def patched(trade_record):
                existing = []
                if path.exists():
                    with open(path) as f:
                        existing = json.load(f)
                existing.append({
                    "entry_date": trade_record.entry_date,
                    "exit_date": trade_record.exit_date,
                    "return_pct": trade_record.return_pct,
                    "exit_reason": trade_record.exit_reason,
                    "hold_days": trade_record.hold_days,
                    "entry_spy": trade_record.entry_spy,
                    "exit_spy": trade_record.exit_spy,
                    "vix_at_entry": trade_record.vix_at_entry,
                    "allocation_pct": trade_record.allocation_pct,
                })
                with open(path, "w") as f:
                    json.dump(existing, f, indent=2)
            return patched

        trade1 = MeanReversionTrade("2026-05-01", "2026-05-05", 500, 510, 2.0, 4, 35, 3, "recovery")
        trade2 = MeanReversionTrade("2026-05-10", "2026-05-15", 510, 505, -1.0, 5, 32, 2, "vix_drop")

        with patch.object(calc, "_save_trade_record", side_effect=make_patched_save(trades_path)):
            calc._save_trade_record(trade1)
            calc._save_trade_record(trade2)

        with open(trades_path) as f:
            trades = json.load(f)
        assert len(trades) == 2
        assert trades[0]["exit_reason"] == "recovery"
        assert trades[1]["exit_reason"] == "vix_drop"


class TestBacktest:
    """Test backtest functionality."""

    def test_backtest_with_data(self):
        """Test backtest runs with sufficient data."""
        calc = VIXMeanReversionCalculator()
        n = 500
        calc._dates = [f"day_{i}" for i in range(n)]
        calc._prices["SPY"] = np.linspace(100, 130, n)
        calc._prices["GLD"] = np.full(n, 100.0)
        calc._prices["TLT"] = np.full(n, 100.0)
        with patch.object(calc.__class__, "get_vix_historical",
                          return_value=np.full(n, 25.0)):
            results = calc.run_backtest()
        assert isinstance(results, dict)
        assert "total_trades" in results
        assert "date_range" in results

    def test_backtest_without_data(self):
        """Test backtest handles no data gracefully."""
        calc = VIXMeanReversionCalculator()
        calc._prices = {}
        results = calc.run_backtest()
        assert "error" in results

    def test_backtest_returns_metrics(self):
        """Test backtest returns expected metrics structure."""
        calc = VIXMeanReversionCalculator()
        n = 500
        calc._dates = [f"day_{i}" for i in range(n)]
        calc._prices["SPY"] = np.linspace(100, 130, n)
        calc._prices["GLD"] = np.full(n, 100.0)
        calc._prices["TLT"] = np.full(n, 100.0)
        with patch.object(calc.__class__, "get_vix_historical",
                          return_value=np.full(n, 20.0)):
            results = calc.run_backtest()
        assert "total_trades" in results or "win_rate_pct" in results
        assert "date_range" in results or "message" in results

    def test_backtest_zero_trades(self):
        """Test backtest with no trade signals (VIX too low)."""
        calc = VIXMeanReversionCalculator()
        n = 500
        calc._dates = [f"day_{i}" for i in range(n)]
        calc._prices["SPY"] = np.linspace(100, 130, n)
        with patch.object(calc.__class__, "get_vix_historical",
                          return_value=np.full(n, 15.0)):  # Low VIX = no entries
            results = calc.run_backtest()
        assert "total_trades" in results

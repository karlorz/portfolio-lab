"""Tests for LiveTransitionManager — paper→live ramp protocol."""

from contextlib import ExitStack, contextmanager
from unittest.mock import patch

import pytest

from src.broker.alpaca import (
    LiveTransitionManager,
    RampPhase,
    RAMP_ALLOCATION_PCT,
    check_alpaca_status,
)
from src.strategy.graduation_checklist import CheckResult


AUTO_GRADUATION_CRITERIA = (
    "min_trading_days",
    "min_sharpe",
    "max_drawdown",
    "min_win_rate",
    "health_checks",
    "min_tca_orders",
    "circuit_breaker_confidence",
    "min_dsr",
    "regime_coverage",
    "signal_diversity",
    "sharpe_ci_lower",
)


def _graduation_results(*, auto_pass: bool = True, manual_pass: bool = True) -> dict[str, CheckResult]:
    results = {
        name: CheckResult(name, auto_pass, 1.0 if auto_pass else 0.0, 1.0, name)
        for name in AUTO_GRADUATION_CRITERIA
    }
    results["manual_approval"] = CheckResult(
        "manual_approval",
        manual_pass,
        1.0 if manual_pass else 0.0,
        1.0,
        "manual approval",
    )
    return results


@contextmanager
def _allow_paper_live_gate(auto_pass: bool = True, manual_pass: bool = True):
    """Patch paper-live gate dependencies to deterministic passing/failing fixtures."""
    with ExitStack() as stack:
        stack.enter_context(patch(
            "src.strategy.graduation_checklist.GraduationChecklist.check",
            return_value=_graduation_results(auto_pass=auto_pass, manual_pass=manual_pass),
        ))
        stack.enter_context(patch(
            "src.data.fred_data.get_fred_md_cache_health",
            return_value={"status": "ok", "source_mode": "live", "api_key_configured": True},
        ))
        stack.enter_context(patch(
            "src.data.fred_readiness.assess_fred_readiness",
            return_value={"ready": True, "blocking": False, "status": "ok", "mode": "live"},
        ))
        yield


class TestRampPhase:
    """Tests for RampPhase enum and constants."""

    def test_six_phases_defined(self):
        assert len(RampPhase) == 6

    def test_phase_order(self):
        phases = [RampPhase.PAPER, RampPhase.PHASE_1, RampPhase.PHASE_2,
                  RampPhase.PHASE_3, RampPhase.PHASE_4, RampPhase.LIVE]
        for p in phases:
            assert isinstance(p, RampPhase)

    def test_allocation_pct_monotonically_increasing(self):
        phases = [RampPhase.PAPER, RampPhase.PHASE_1, RampPhase.PHASE_2,
                  RampPhase.PHASE_3, RampPhase.PHASE_4, RampPhase.LIVE]
        pcts = [RAMP_ALLOCATION_PCT[p.value] for p in phases]
        assert pcts == sorted(pcts)
        assert pcts[0] == 0.0
        assert pcts[-1] == 1.0

    def test_allocation_values(self):
        assert RAMP_ALLOCATION_PCT[RampPhase.PAPER.value] == 0.0
        assert RAMP_ALLOCATION_PCT[RampPhase.PHASE_1.value] == 0.01
        assert RAMP_ALLOCATION_PCT[RampPhase.PHASE_2.value] == 0.05
        assert RAMP_ALLOCATION_PCT[RampPhase.PHASE_3.value] == 0.25
        assert RAMP_ALLOCATION_PCT[RampPhase.PHASE_4.value] == 0.50
        assert RAMP_ALLOCATION_PCT[RampPhase.LIVE.value] == 1.00


class TestLiveTransitionManager:
    """Tests for LiveTransitionManager."""

    def test_default_state_is_paper(self, tmp_path):
        mgr = LiveTransitionManager(state_path=str(tmp_path / "ramp.json"))
        assert mgr.phase == RampPhase.PAPER
        assert mgr.allocation_pct == 0.0
        assert mgr.days_at_phase == 0

    def test_state_persists_to_disk(self, tmp_path):
        state_path = tmp_path / "ramp.json"
        mgr = LiveTransitionManager(state_path=str(state_path))
        mgr._state["phase"] = RampPhase.PHASE_2.value
        mgr._state["days_at_phase"] = 10
        mgr._save_state()

        mgr2 = LiveTransitionManager(state_path=str(state_path))
        assert mgr2.phase == RampPhase.PHASE_2
        assert mgr2.days_at_phase == 10

    def test_corrupt_state_file_defaults_to_paper(self, tmp_path):
        state_path = tmp_path / "ramp.json"
        state_path.write_text("not json{{{")
        mgr = LiveTransitionManager(state_path=str(state_path))
        assert mgr.phase == RampPhase.PAPER

    def test_can_advance_paper_needs_graduation(self, tmp_path):
        """PAPER phase needs 63 days + alpaca connected before advancing."""
        mgr = LiveTransitionManager(state_path=str(tmp_path / "ramp.json"))
        # Not enough days
        assert not mgr.can_advance()

    def test_can_advance_with_sufficient_days_and_live_start_gates(self, tmp_path):
        """PAPER phase can advance with 63+ days, Alpaca, checklist, manual approval, and FRED readiness."""
        mgr = LiveTransitionManager(state_path=str(tmp_path / "ramp.json"))
        mgr._state["days_at_phase"] = 63
        mgr._save_state()
        with _allow_paper_live_gate(), patch(
            "src.broker.alpaca.check_alpaca_status",
            return_value={"connected": True, "paper": False},
        ):
            assert mgr.can_advance()

    def test_cannot_advance_paper_when_graduation_checklist_fails(self, tmp_path):
        """PAPER phase cannot start live ramp when any automatic graduation criterion fails."""
        mgr = LiveTransitionManager(state_path=str(tmp_path / "ramp.json"))
        mgr._state["days_at_phase"] = 63
        mgr._save_state()

        with _allow_paper_live_gate(auto_pass=False, manual_pass=True), patch(
            "src.broker.alpaca.check_alpaca_status",
            return_value={"connected": True, "paper": False},
        ):
            assert not mgr.can_advance()

    def test_cannot_advance_paper_without_manual_approval(self, tmp_path):
        """PAPER phase cannot start live ramp until manual approval is explicit."""
        mgr = LiveTransitionManager(state_path=str(tmp_path / "ramp.json"))
        mgr._state["days_at_phase"] = 63
        mgr._save_state()

        with _allow_paper_live_gate(auto_pass=True, manual_pass=False), patch(
            "src.broker.alpaca.check_alpaca_status",
            return_value={"connected": True, "paper": False},
        ):
            assert not mgr.can_advance()

    def test_cannot_advance_paper_when_live_fred_readiness_fails(self, tmp_path):
        """PAPER phase cannot start live ramp when FRED readiness fails under live policy."""
        mgr = LiveTransitionManager(state_path=str(tmp_path / "ramp.json"))
        mgr._state["days_at_phase"] = 63
        mgr._save_state()

        with patch(
            "src.strategy.graduation_checklist.GraduationChecklist.check",
            return_value=_graduation_results(auto_pass=True, manual_pass=True),
        ), patch(
            "src.data.fred_data.get_fred_md_cache_health",
            return_value={"status": "unavailable", "source_mode": "synthetic", "api_key_configured": False},
        ), patch(
            "src.data.fred_readiness.assess_fred_readiness",
            return_value={
                "ready": False,
                "blocking": True,
                "status": "critical",
                "mode": "live",
                "reason": "missing_fred_api_key",
            },
        ) as assess_fred, patch(
            "src.broker.alpaca.check_alpaca_status",
            return_value={"connected": True, "paper": False},
        ):
            assert not mgr.can_advance()
            assert assess_fred.call_args.kwargs["mode"] == "live"

    @pytest.mark.parametrize("flag", ["trading_blocked", "account_blocked", "trade_suspended_by_user", "transfers_blocked"])
    def test_cannot_advance_paper_when_live_account_restricted(self, tmp_path, flag):
        """PAPER phase cannot start live ramp with broker account restrictions."""
        mgr = LiveTransitionManager(state_path=str(tmp_path / "ramp.json"))
        mgr._state["days_at_phase"] = 63
        mgr._save_state()
        alpaca_status = {"connected": True, "paper": False, flag: True}

        with _allow_paper_live_gate(), patch(
            "src.broker.alpaca.check_alpaca_status",
            return_value=alpaca_status,
        ):
            assert not mgr.can_advance()

    def test_cannot_advance_paper_while_alpaca_status_is_paper_mode(self, tmp_path):
        """PAPER phase cannot start live ramp until Alpaca status is checking the live account."""
        mgr = LiveTransitionManager(state_path=str(tmp_path / "ramp.json"))
        mgr._state["days_at_phase"] = 63
        mgr._save_state()

        with _allow_paper_live_gate(), patch(
            "src.broker.alpaca.check_alpaca_status",
            return_value={"connected": True, "paper": True},
        ):
            assert not mgr.can_advance()

    def test_cannot_advance_live_phase_while_alpaca_status_is_paper_mode(self, tmp_path):
        """Live ramp phases cannot advance while the broker client is still in paper mode."""
        mgr = LiveTransitionManager(state_path=str(tmp_path / "ramp.json"))
        mgr._state["phase"] = RampPhase.PHASE_1.value
        mgr._state["days_at_phase"] = 21
        mgr._save_state()

        with patch(
            "src.broker.alpaca.check_alpaca_status",
            return_value={"connected": True, "paper": True},
        ):
            assert not mgr.can_advance()

    def test_can_advance_blocked_without_alpaca(self, tmp_path):
        """Cannot advance without alpaca connectivity."""
        mgr = LiveTransitionManager(state_path=str(tmp_path / "ramp.json"))
        mgr._state["days_at_phase"] = 100
        mgr._save_state()
        with patch("src.broker.alpaca.check_alpaca_status", return_value={"connected": False}):
            assert not mgr.can_advance()

    def test_cannot_advance_from_live(self, tmp_path):
        """LIVE phase is terminal — no further advancement."""
        mgr = LiveTransitionManager(state_path=str(tmp_path / "ramp.json"))
        mgr._state["phase"] = RampPhase.LIVE.value
        mgr._save_state()
        assert not mgr.can_advance()

    def test_advance_succeeds_when_prerequisites_met(self, tmp_path):
        mgr = LiveTransitionManager(state_path=str(tmp_path / "ramp.json"))
        mgr._state["days_at_phase"] = 63
        mgr._save_state()
        with _allow_paper_live_gate(), patch(
            "src.broker.alpaca.check_alpaca_status",
            return_value={"connected": True, "paper": False},
        ):
            result = mgr.advance()
        assert result["success"] is True
        assert result["new_phase"] == RampPhase.PHASE_1.value
        assert mgr.phase == RampPhase.PHASE_1

    def test_advance_fails_when_prerequisites_not_met(self, tmp_path):
        mgr = LiveTransitionManager(state_path=str(tmp_path / "ramp.json"))
        # 0 days → cannot advance
        with patch("src.broker.alpaca.check_alpaca_status", return_value={"connected": True}):
            result = mgr.advance()
        assert result["success"] is False

    def test_advance_resets_phase_tracking(self, tmp_path):
        mgr = LiveTransitionManager(state_path=str(tmp_path / "ramp.json"))
        mgr._state["days_at_phase"] = 63
        mgr._state["max_drawdown_pct"] = 0.05
        mgr._save_state()
        with _allow_paper_live_gate(), patch(
            "src.broker.alpaca.check_alpaca_status",
            return_value={"connected": True, "paper": False},
        ):
            mgr.advance()
        assert mgr.days_at_phase == 0
        assert mgr.max_drawdown_pct == 0.0

    def test_advance_logs_entry(self, tmp_path):
        mgr = LiveTransitionManager(state_path=str(tmp_path / "ramp.json"))
        mgr._state["days_at_phase"] = 63
        mgr._save_state()
        with _allow_paper_live_gate(), patch(
            "src.broker.alpaca.check_alpaca_status",
            return_value={"connected": True, "paper": False},
        ):
            mgr.advance()
        log = mgr._state.get("advancement_log", [])
        assert len(log) == 1
        assert log[0]["from"] == RampPhase.PAPER.value
        assert log[0]["to"] == RampPhase.PHASE_1.value

    def test_rollback_from_phase_1(self, tmp_path):
        mgr = LiveTransitionManager(state_path=str(tmp_path / "ramp.json"))
        mgr._state["phase"] = RampPhase.PHASE_1.value
        mgr._save_state()
        result = mgr.rollback("risk breach")
        assert result["success"] is True
        assert mgr.phase == RampPhase.PAPER

    def test_rollback_from_paper_fails(self, tmp_path):
        mgr = LiveTransitionManager(state_path=str(tmp_path / "ramp.json"))
        result = mgr.rollback("no reason")
        assert result["success"] is False

    def test_rollback_logs_entry(self, tmp_path):
        mgr = LiveTransitionManager(state_path=str(tmp_path / "ramp.json"))
        mgr._state["phase"] = RampPhase.PHASE_3.value
        mgr._save_state()
        mgr.rollback("drawdown exceeded")
        log = mgr._state.get("advancement_log", [])
        assert len(log) == 1
        assert log[0]["from"] == RampPhase.PHASE_3.value
        assert log[0]["to"] == RampPhase.PHASE_2.value
        assert "drawdown exceeded" in log[0]["reason"]

    def test_rollback_resets_tracking(self, tmp_path):
        mgr = LiveTransitionManager(state_path=str(tmp_path / "ramp.json"))
        mgr._state["phase"] = RampPhase.PHASE_2.value
        mgr._state["days_at_phase"] = 15
        mgr._state["max_drawdown_pct"] = 0.20
        mgr._save_state()
        mgr.rollback("test")
        assert mgr.days_at_phase == 0
        assert mgr.max_drawdown_pct == 0.0

    def test_update_daily_tracks_equity(self, tmp_path):
        mgr = LiveTransitionManager(state_path=str(tmp_path / "ramp.json"))
        mgr._state["phase"] = RampPhase.PHASE_1.value
        mgr.update_daily(100000)
        assert mgr.days_at_phase == 1
        assert mgr._state["peak_equity"] == 100000

    def test_update_daily_tracks_drawdown(self, tmp_path):
        mgr = LiveTransitionManager(state_path=str(tmp_path / "ramp.json"))
        mgr._state["phase"] = RampPhase.PHASE_1.value
        mgr.update_daily(100000)  # peak
        mgr.update_daily(95000)   # -5%
        assert mgr.max_drawdown_pct == pytest.approx(0.05, abs=0.001)

    def test_update_daily_auto_rollback_on_drawdown_breach(self, tmp_path):
        mgr = LiveTransitionManager(state_path=str(tmp_path / "ramp.json"))
        mgr._state["phase"] = RampPhase.PHASE_1.value
        # PHASE_1 max drawdown = 10%
        mgr.update_daily(100000)   # peak
        mgr.update_daily(88000)    # -12%, exceeds 10% limit
        assert mgr.phase == RampPhase.PAPER  # auto-rollback

    def test_get_status(self, tmp_path):
        mgr = LiveTransitionManager(state_path=str(tmp_path / "ramp.json"))
        with patch("src.broker.alpaca.check_alpaca_status", return_value={"connected": True, "paper": True}):
            status = mgr.get_status()
        assert status["phase"] == RampPhase.PAPER.value
        assert status["allocation_pct"] == 0.0
        assert "can_advance" in status
        assert "alpaca_status" in status

    def test_full_ramp_cycle(self, tmp_path):
        """Simulate a full PAPER → LIVE ramp cycle."""
        mgr = LiveTransitionManager(state_path=str(tmp_path / "ramp.json"))
        alpaca_ok = {"connected": True, "paper": False}

        # PAPER → PHASE_1 (need 63 days)
        mgr._state["days_at_phase"] = 63
        mgr._save_state()
        with _allow_paper_live_gate(), patch("src.broker.alpaca.check_alpaca_status", return_value=alpaca_ok):
            result = mgr.advance()
        assert result["success"] is True
        assert mgr.phase == RampPhase.PHASE_1

        # PHASE_1 → PHASE_2 (need 21 days)
        mgr._state["days_at_phase"] = 21
        mgr._save_state()
        with patch("src.broker.alpaca.check_alpaca_status", return_value=alpaca_ok):
            result = mgr.advance()
        assert result["success"] is True
        assert mgr.phase == RampPhase.PHASE_2

        # PHASE_2 → PHASE_3
        mgr._state["days_at_phase"] = 21
        mgr._save_state()
        with patch("src.broker.alpaca.check_alpaca_status", return_value=alpaca_ok):
            result = mgr.advance()
        assert result["success"] is True
        assert mgr.phase == RampPhase.PHASE_3

        # PHASE_3 → PHASE_4 (need 42 days)
        mgr._state["days_at_phase"] = 42
        mgr._save_state()
        with patch("src.broker.alpaca.check_alpaca_status", return_value=alpaca_ok):
            result = mgr.advance()
        assert result["success"] is True
        assert mgr.phase == RampPhase.PHASE_4

        # PHASE_4 → LIVE (need 42 days)
        mgr._state["days_at_phase"] = 42
        mgr._save_state()
        with patch("src.broker.alpaca.check_alpaca_status", return_value=alpaca_ok):
            result = mgr.advance()
        assert result["success"] is True
        assert mgr.phase == RampPhase.LIVE
        assert mgr.allocation_pct == 1.0

    def test_trading_blocked_prevents_advance(self, tmp_path):
        """A blocked trading account cannot advance beyond PAPER."""
        mgr = LiveTransitionManager(state_path=str(tmp_path / "ramp.json"))
        mgr._state["phase"] = RampPhase.PHASE_1.value
        mgr._state["days_at_phase"] = 25
        mgr._save_state()
        blocked = {"connected": True, "trading_blocked": True, "paper": False}
        with patch("src.broker.alpaca.check_alpaca_status", return_value=blocked):
            assert not mgr.can_advance()


class TestCheckAlpacaStatus:
    """Tests for check_alpaca_status() with paper/live detection."""

    def test_paper_mode_default(self):
        """Default should be paper mode."""
        with patch.dict("os.environ", {}, clear=False):
            # Remove ALPACA_PAPER if set
            import os
            os.environ.pop("ALPACA_PAPER", None)
            status = check_alpaca_status()
            assert status["paper"] is True

    def test_live_mode_via_env(self):
        """ALPACA_PAPER=false enables live mode detection."""
        with patch.dict("os.environ", {"ALPACA_PAPER": "false"}):
            status = check_alpaca_status()
            assert status["paper"] is False

    def test_paper_mode_explicit(self):
        """ALPACA_PAPER=true keeps paper mode."""
        with patch.dict("os.environ", {"ALPACA_PAPER": "true"}):
            status = check_alpaca_status()
            assert status["paper"] is True

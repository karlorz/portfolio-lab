"""Tests for src.monitor.alerting — webhook alerting + staleness checks."""

import os
import json
import time
from unittest.mock import patch, MagicMock

import pytest

from src.monitor.alerting import (
    AlertLevel,
    AlertChannel,
    send_alert,
    check_staleness_and_alert,
    check_drift_and_alert,
    check_ic_decay_and_alert,
    _last_alert_time,
    ALERT_MIN_INTERVAL_SECONDS,
)


@pytest.fixture(autouse=True)
def _isolate_incident_manager(tmp_path):
    """Keep send_alert incident side effects out of repo data/ during tests."""
    from src.monitor.incident_manager import IncidentManager

    manager = IncidentManager(
        log_path=tmp_path / "incidents.jsonl",
        summary_path=tmp_path / "incidents.json",
    )
    with patch("src.monitor.alerting.get_incident_manager", return_value=manager):
        yield manager


class TestAlertLevel:
    def test_levels_exist(self):
        assert AlertLevel.PASS == "pass"
        assert AlertLevel.WARN == "warn"
        assert AlertLevel.HALT == "halt"


class TestAlertChannel:
    def test_channels_exist(self):
        assert AlertChannel.SIGNAL_STALENESS == "signal_staleness"
        assert AlertChannel.EVALUATOR_ERROR == "evaluator_error"
        assert AlertChannel.PORTFOLIO_DRIFT == "portfolio_drift"
        assert AlertChannel.CRON_FAILURE == "cron_failure"
        assert AlertChannel.IC_DECAY == "ic_decay"


class TestSendAlert:
    def test_disabled_when_no_webhook(self):
        """When ALERT_WEBHOOK_URL is empty, alerting is silently disabled."""
        with patch("src.monitor.alerting.ALERT_WEBHOOK_URL", ""):
            result = send_alert(
                AlertChannel.SIGNAL_STALENESS,
                AlertLevel.WARN,
                "test message",
            )
        assert result is True  # Silently succeeds (no-op)

    def test_disabled_webhook_still_records_incident(self, tmp_path):
        """Dashboard-only alerting should still persist incident lifecycle state."""
        from src.monitor.incident_manager import IncidentManager

        manager = IncidentManager(
            log_path=tmp_path / "incidents.jsonl",
            summary_path=tmp_path / "incidents.json",
        )

        with patch("src.monitor.alerting.ALERT_WEBHOOK_URL", ""), \
             patch("src.monitor.alerting.get_incident_manager", return_value=manager):
            result = send_alert(
                AlertChannel.PORTFOLIO_DRIFT,
                AlertLevel.WARN,
                "Portfolio drift exceeding 5%",
                details={"drift_pct": 7.5},
            )

        assert result is True
        lines = (tmp_path / "incidents.jsonl").read_text().splitlines()
        assert len(lines) == 1
        event = json.loads(lines[0])
        assert event["event"] == "opened"
        assert event["channel"] == "portfolio_drift"
        assert event["severity"] == "p2"

    @patch("src.monitor.alerting.urllib.request.urlopen")
    @patch("src.monitor.alerting.ALERT_WEBHOOK_URL", "https://hooks.example.com/test")
    def test_dedup_suppressed_alert_skips_incident_update(self, mock_urlopen, tmp_path):
        """Lifecycle must honor the same min-interval as webhook delivery."""
        from src.monitor.incident_manager import IncidentManager

        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp
        _last_alert_time.clear()
        manager = IncidentManager(
            log_path=tmp_path / "incidents.jsonl",
            summary_path=tmp_path / "incidents.json",
            kill_switch_path=tmp_path / "kill_switch.json",
            escalation_cycles=1,
        )

        with patch("src.monitor.alerting.get_incident_manager", return_value=manager):
            assert send_alert(AlertChannel.IC_DECAY, AlertLevel.WARN, "IC warning") is True
            assert send_alert(AlertChannel.IC_DECAY, AlertLevel.WARN, "IC warning updated") is True

        assert mock_urlopen.call_count == 1
        events = [json.loads(line) for line in (tmp_path / "incidents.jsonl").read_text().splitlines()]
        assert [event["event"] for event in events] == ["opened"]
        assert events[0]["alert_count"] == 1

    def test_repeated_warn_without_webhook_does_not_ratchet_alert_count(self, tmp_path):
        """Dashboard-only mode still applies lifecycle dedup."""
        from src.monitor.incident_manager import IncidentManager

        manager = IncidentManager(
            log_path=tmp_path / "incidents.jsonl",
            summary_path=tmp_path / "incidents.json",
            kill_switch_path=tmp_path / "kill_switch.json",
            escalation_cycles=1,
        )
        _last_alert_time.clear()

        with patch("src.monitor.alerting.ALERT_WEBHOOK_URL", ""), \
             patch("src.monitor.alerting.get_incident_manager", return_value=manager):
            for i in range(5):
                assert send_alert(
                    AlertChannel.SIGNAL_STALENESS,
                    AlertLevel.WARN,
                    f"optional unavailable refresh {i}",
                ) is True

        events = [json.loads(line) for line in (tmp_path / "incidents.jsonl").read_text().splitlines()]
        assert [event["event"] for event in events] == ["opened"]
        assert events[0]["alert_count"] == 1
        open_incidents = manager.open_incidents()
        assert len(open_incidents) == 1
        assert open_incidents[0].alert_count == 1

    def test_pass_always_resolves_even_within_dedup_window(self, tmp_path):
        from src.monitor.incident_manager import IncidentManager

        manager = IncidentManager(
            log_path=tmp_path / "incidents.jsonl",
            summary_path=tmp_path / "incidents.json",
        )
        _last_alert_time.clear()

        with patch("src.monitor.alerting.ALERT_WEBHOOK_URL", ""), \
             patch("src.monitor.alerting.get_incident_manager", return_value=manager):
            assert send_alert(AlertChannel.PORTFOLIO_DRIFT, AlertLevel.WARN, "drift") is True
            assert send_alert(AlertChannel.PORTFOLIO_DRIFT, AlertLevel.PASS, "ok") is True

        events = [json.loads(line) for line in (tmp_path / "incidents.jsonl").read_text().splitlines()]
        assert [event["event"] for event in events] == ["opened", "resolved"]
        assert manager.open_incidents() == []

    def test_warn_after_pass_reopens_despite_prior_warn_interval(self, tmp_path):
        """PASS clears channel dedup so a fresh WARN can open again."""
        from src.monitor.incident_manager import IncidentManager

        manager = IncidentManager(
            log_path=tmp_path / "incidents.jsonl",
            summary_path=tmp_path / "incidents.json",
        )
        _last_alert_time.clear()

        with patch("src.monitor.alerting.ALERT_WEBHOOK_URL", ""), \
             patch("src.monitor.alerting.get_incident_manager", return_value=manager):
            assert send_alert(AlertChannel.SIGNAL_STALENESS, AlertLevel.WARN, "a") is True
            assert send_alert(AlertChannel.SIGNAL_STALENESS, AlertLevel.PASS, "clear") is True
            assert send_alert(AlertChannel.SIGNAL_STALENESS, AlertLevel.WARN, "b") is True

        events = [json.loads(line) for line in (tmp_path / "incidents.jsonl").read_text().splitlines()]
        assert [event["event"] for event in events] == ["opened", "resolved", "opened"]

    def test_level_transition_warn_to_halt_records_immediately(self, tmp_path):
        from src.monitor.incident_manager import IncidentManager

        manager = IncidentManager(
            log_path=tmp_path / "incidents.jsonl",
            summary_path=tmp_path / "incidents.json",
        )
        _last_alert_time.clear()

        with patch("src.monitor.alerting.ALERT_WEBHOOK_URL", ""), \
             patch("src.monitor.alerting.get_incident_manager", return_value=manager):
            assert send_alert(AlertChannel.SIGNAL_STALENESS, AlertLevel.WARN, "optional") is True
            assert send_alert(AlertChannel.SIGNAL_STALENESS, AlertLevel.HALT, "required") is True

        events = [json.loads(line) for line in (tmp_path / "incidents.jsonl").read_text().splitlines()]
        assert [event["event"] for event in events] == ["opened", "updated"]
        assert events[1]["severity"] == "p0"
        assert events[1]["alert_count"] == 2

    @patch("src.monitor.alerting.urllib.request.urlopen")
    @patch("src.monitor.alerting.ALERT_WEBHOOK_URL", "https://hooks.example.com/test")
    def test_sends_webhook_post(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        # Clear dedup tracker
        _last_alert_time.clear()

        result = send_alert(
            AlertChannel.SIGNAL_STALENESS,
            AlertLevel.WARN,
            "2 signals stale",
            details={"stale": ["alt_data", "garch_cvar"]},
        )
        assert result is True
        mock_urlopen.assert_called_once()

        # Verify payload structure
        call_args = mock_urlopen.call_args
        request = call_args[0][0]
        assert request.method == "POST"
        payload = json.loads(request.data)
        assert payload["channel"] == "signal_staleness"
        assert payload["level"] == "warn"
        assert payload["message"] == "2 signals stale"
        assert payload["source"] == "portfolio-lab"
        assert "timestamp" in payload
        assert payload["details"]["stale"] == ["alt_data", "garch_cvar"]

    @patch("src.monitor.alerting.urllib.request.urlopen")
    @patch("src.monitor.alerting.ALERT_WEBHOOK_URL", "https://hooks.example.com/test")
    def test_dedup_suppresses_repeated_alerts(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp
        _last_alert_time.clear()

        # First alert goes through
        result1 = send_alert(AlertChannel.CRON_FAILURE, AlertLevel.HALT, "cron failed")
        assert result1 is True
        assert mock_urlopen.call_count == 1

        # Second identical alert is suppressed (within dedup interval)
        result2 = send_alert(AlertChannel.CRON_FAILURE, AlertLevel.HALT, "cron failed again")
        assert result2 is True  # Returns True (suppressed, not failed)
        assert mock_urlopen.call_count == 1  # Still only 1 call

    @patch("src.monitor.alerting.urllib.request.urlopen")
    @patch("src.monitor.alerting.ALERT_WEBHOOK_URL", "https://hooks.example.com/test")
    def test_webhook_failure_returns_false(self, mock_urlopen):
        mock_urlopen.side_effect = Exception("connection refused")
        _last_alert_time.clear()

        result = send_alert(
            AlertChannel.EVALUATOR_ERROR,
            AlertLevel.WARN,
            "evaluator crashed",
        )
        assert result is False


class TestCheckStalenessAndAlert:
    @patch("src.monitor.alerting.send_alert")
    def test_all_fresh_sends_pass(self, mock_send):
        staleness = {
            "stale_signals": [],
            "healthy_count": 5,
            "total_count": 5,
        }
        check_staleness_and_alert(staleness)
        mock_send.assert_called_once()
        call_args = mock_send.call_args
        assert call_args[0][0] == AlertChannel.SIGNAL_STALENESS
        assert call_args[0][1] == AlertLevel.PASS

    @patch("src.monitor.alerting.send_alert")
    def test_some_stale_sends_warn(self, mock_send):
        staleness = {
            "stale_signals": ["ensemble_voting", "garch_cvar"],
            "healthy_count": 3,
            "total_count": 5,
        }
        check_staleness_and_alert(staleness)
        mock_send.assert_called_once()
        call_args = mock_send.call_args
        assert call_args[0][0] == AlertChannel.SIGNAL_STALENESS
        assert call_args[0][1] == AlertLevel.WARN

    @patch("src.monitor.alerting.send_alert")
    def test_all_stale_sends_halt(self, mock_send):
        staleness = {
            "stale_signals": ["ensemble_voting", "alternative_data", "garch_cvar", "behavioral_sentiment", "smart_rebalance"],
            "healthy_count": 0,
            "total_count": 5,
        }
        check_staleness_and_alert(staleness)
        mock_send.assert_called_once()
        call_args = mock_send.call_args
        assert call_args[0][0] == AlertChannel.SIGNAL_STALENESS
        assert call_args[0][1] == AlertLevel.HALT

    @patch("src.monitor.alerting.send_alert")
    def test_empty_data_no_alert(self, mock_send):
        staleness = {"stale_signals": [], "healthy_count": 0, "total_count": 0}
        check_staleness_and_alert(staleness)
        mock_send.assert_not_called()

    @patch("src.monitor.alerting.send_alert")
    def test_unavailable_without_stale_sends_warn_not_pass(self, mock_send):
        """Empty stale + non-empty unavailable must not claim all-fresh PASS."""
        unavailable = [
            "behavioral_sentiment",
            "calendar_seasonality",
            "crypto_allocation",
            "factor_rotation",
            "stacking_ensemble",
            "kurtosis_regime",
            "collar",
            "bond_momentum",
            "risk_decomposition",
            "two_stage_regime",
            "regime_transition",
            "fred_macro",
        ]
        staleness = {
            "stale_signals": [],
            "unavailable_signals": unavailable,
            "healthy_count": 11,
            "total_count": 23,
        }
        check_staleness_and_alert(staleness)
        # Primary channel is SIGNAL_STALENESS WARN; optional second call is
        # SIGNAL_RECOVERY when live kill_switch is already sustained halt.
        assert mock_send.call_count >= 1
        call_args = mock_send.call_args_list[0]
        assert call_args[0][0] == AlertChannel.SIGNAL_STALENESS
        assert call_args[0][1] == AlertLevel.WARN
        message = call_args[0][2]
        assert "All 23 signals fresh" not in message
        assert "unavailable" in message.lower()
        details = call_args.kwargs.get("details") or (
            call_args[1].get("details") if len(call_args) > 1 else None
        )
        if details is None and call_args[0][3:]:
            details = call_args[0][3]
        # kwargs path used by send_alert(..., details=)
        if details is None:
            details = call_args.kwargs.get("details")
        assert details is not None
        assert details.get("unavailable_count") == 12
        assert details.get("unavailable_signals") == unavailable
        assert "unavailable_ownership" in details
        assert details.get("recovery", {}).get("actionable_unavailable_count", 0) >= 1

    @patch("src.monitor.alerting.send_alert")
    def test_stale_and_unavailable_mentions_both(self, mock_send):
        """Stale WARN should surface unavailable count in the message."""
        staleness = {
            "stale_signals": ["ensemble_voting"],
            "unavailable_signals": ["fred_macro", "bond_momentum"],
            "healthy_count": 3,
            "total_count": 6,
        }
        check_staleness_and_alert(staleness)
        mock_send.assert_called_once()
        assert mock_send.call_args[0][1] == AlertLevel.WARN
        message = mock_send.call_args[0][2]
        assert "stale" in message.lower()
        assert "unavailable" in message.lower()
        details = mock_send.call_args.kwargs.get("details")
        assert details["unavailable_count"] == 2


class TestCheckDriftAndAlert:
    @patch("src.monitor.alerting.send_alert")
    def test_low_drift_pass(self, mock_send):
        check_drift_and_alert(3.2)
        mock_send.assert_called_once()
        assert mock_send.call_args[0][1] == AlertLevel.PASS

    @patch("src.monitor.alerting.send_alert")
    def test_moderate_drift_warn(self, mock_send):
        check_drift_and_alert(7.5)
        mock_send.assert_called_once()
        assert mock_send.call_args[0][1] == AlertLevel.WARN

    @patch("src.monitor.alerting.send_alert")
    def test_high_drift_halt(self, mock_send):
        check_drift_and_alert(12.0)
        mock_send.assert_called_once()
        assert mock_send.call_args[0][1] == AlertLevel.HALT

    @patch("src.monitor.alerting.send_alert")
    def test_negative_drift_warn(self, mock_send):
        check_drift_and_alert(-6.0)
        mock_send.assert_called_once()
        assert mock_send.call_args[0][1] == AlertLevel.WARN


class TestClassifySignalStalenessIntentionalGaps:
    def test_intentional_lab_gaps_only_pass(self):
        from src.monitor.alerting import classify_signal_staleness, AlertLevel

        result = classify_signal_staleness(
            {
                "stale_signals": [],
                "unavailable_signals": [
                    "behavioral_sentiment",
                    "stacking_ensemble",
                    "fred_macro",
                ],
                "healthy_count": 20,
                "total_count": 23,
                "unavailable_ownership": [
                    {
                        "signal": "behavioral_sentiment",
                        "intentional_when_ml_off": True,
                        "intentional_lab_gap": True,
                    },
                    {
                        "signal": "stacking_ensemble",
                        "intentional_when_ml_off": True,
                        "intentional_lab_gap": True,
                    },
                    {
                        "signal": "fred_macro",
                        "intentional_when_fred_unconfigured": True,
                        "intentional_lab_gap": True,
                    },
                ],
            }
        )
        assert result is not None
        level, message, details = result
        assert level == AlertLevel.PASS
        assert details.get("policy") == "intentional_lab_gaps_only_pass"
        assert "lab gaps" in message.lower() or "fresh" in message.lower()

    def test_actionable_unavailable_still_warns(self):
        from src.monitor.alerting import classify_signal_staleness, AlertLevel

        result = classify_signal_staleness(
            {
                "stale_signals": [],
                "unavailable_signals": ["risk_decomposition", "behavioral_sentiment"],
                "healthy_count": 20,
                "total_count": 23,
                "unavailable_ownership": [
                    {
                        "signal": "risk_decomposition",
                        "intentional_lab_gap": False,
                    },
                    {
                        "signal": "behavioral_sentiment",
                        "intentional_when_ml_off": True,
                        "intentional_lab_gap": True,
                    },
                ],
            }
        )
        assert result is not None
        level, message, _details = result
        assert level == AlertLevel.WARN
        assert "risk_decomposition" in message
        assert "behavioral_sentiment" not in message


class TestCheckIcDecayAndAlert:
    @patch("src.monitor.alerting.send_alert")
    def test_all_healthy_sends_pass(self, mock_send):
        ic_decay = {
            "alternative_data": {"status": "healthy", "ic_rolling": 0.15, "ic_trend": "stable"},
            "cross_asset_rv": {"status": "healthy", "ic_rolling": 0.12, "ic_trend": "stable"},
        }
        check_ic_decay_and_alert(ic_decay)
        mock_send.assert_called_once()
        assert mock_send.call_args[0][0] == AlertChannel.IC_DECAY
        assert mock_send.call_args[0][1] == AlertLevel.PASS

    @patch("src.monitor.alerting.send_alert")
    def test_warning_signal_sends_warn(self, mock_send):
        ic_decay = {
            "alternative_data": {"status": "warning", "ic_rolling": 0.07, "ic_trend": "decaying"},
            "cross_asset_rv": {"status": "healthy", "ic_rolling": 0.12, "ic_trend": "stable"},
        }
        check_ic_decay_and_alert(ic_decay)
        mock_send.assert_called_once()
        assert mock_send.call_args[0][0] == AlertChannel.IC_DECAY
        assert mock_send.call_args[0][1] == AlertLevel.WARN
        assert "alternative_data" in mock_send.call_args[0][2]

    @patch("src.monitor.alerting.send_alert")
    def test_critical_signal_sends_halt(self, mock_send):
        ic_decay = {
            "alternative_data": {"status": "critical", "ic_rolling": 0.02, "ic_trend": "decaying"},
            "cross_asset_rv": {"status": "warning", "ic_rolling": 0.07, "ic_trend": "decaying"},
        }
        check_ic_decay_and_alert(ic_decay)
        mock_send.assert_called_once()
        assert mock_send.call_args[0][0] == AlertChannel.IC_DECAY
        assert mock_send.call_args[0][1] == AlertLevel.HALT
        assert "alternative_data" in mock_send.call_args[0][2]

    @patch("src.monitor.alerting.send_alert")
    def test_empty_data_no_alert(self, mock_send):
        check_ic_decay_and_alert({})
        mock_send.assert_not_called()

    @patch("src.monitor.alerting.send_alert")
    def test_error_key_no_alert(self, mock_send):
        check_ic_decay_and_alert({"error": "monitor unavailable"})
        mock_send.assert_not_called()

    @patch("src.monitor.alerting.send_alert")
    def test_all_insufficient_data_sends_pass_not_halt(self, mock_send):
        """Warm-up IC history must PASS (clear false HALT) not escalate kill."""
        check_ic_decay_and_alert({
            "ensemble_duration": {
                "status": "insufficient_data",
                "ic_rolling": -0.48,
                "observations": 6,
            },
            "behavioral_sentiment": {
                "status": "insufficient_data",
                "ic_rolling": -0.71,
                "observations": 6,
            },
        })
        mock_send.assert_called_once()
        assert mock_send.call_args[0][0] == AlertChannel.IC_DECAY
        assert mock_send.call_args[0][1] == AlertLevel.PASS
        assert "warming up" in mock_send.call_args[0][2].lower()

    @patch("src.monitor.alerting.send_alert")
    def test_mixed_warning_and_critical_sends_halt(self, mock_send):
        """When both warning and critical signals exist, alert is HALT."""
        ic_decay = {
            "multi_speed_mom": {"status": "critical", "ic_rolling": 0.01, "ic_trend": "decaying"},
            "alternative_data": {"status": "warning", "ic_rolling": 0.08, "ic_trend": "decaying"},
            "cross_asset_rv": {"status": "healthy", "ic_rolling": 0.15, "ic_trend": "stable"},
        }
        check_ic_decay_and_alert(ic_decay)
        mock_send.assert_called_once()
        assert mock_send.call_args[0][1] == AlertLevel.HALT
        # Details should include both critical and warning signals
        details = mock_send.call_args[1].get("details") or mock_send.call_args[0][3] if len(mock_send.call_args[0]) > 3 else None
        # The function passes details as keyword arg

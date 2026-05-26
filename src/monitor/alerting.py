"""
Alerting module for production pipeline events.

Sends configurable webhook notifications for:
- Signal staleness (when signals exceed TTL)
- Evaluator errors (strategy evaluator failures)
- Portfolio drift (>5% from target allocation)
- Cron job failures (pipeline step failures)

Uses state-transition alerting: fires on PASS→WARN→HALT transitions,
not threshold breaches, to avoid alert fatigue.

Webhook URL configured via ALERT_WEBHOOK_URL env var.
When unset, alerting is silently disabled (dashboard-only mode).
"""

import json
import logging
import os
import urllib.request
import urllib.error
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

ALERT_WEBHOOK_URL = os.environ.get("ALERT_WEBHOOK_URL", "").strip()
ALERT_MIN_INTERVAL_SECONDS = int(os.environ.get("ALERT_MIN_INTERVAL_SECONDS", "300"))


class AlertLevel(str, Enum):
    """Alert severity levels following PASS→WARN→HALT state machine."""
    PASS = "pass"
    WARN = "warn"
    HALT = "halt"


class AlertChannel(str, Enum):
    """Alert categories for different pipeline events."""
    SIGNAL_STALENESS = "signal_staleness"
    EVALUATOR_ERROR = "evaluator_error"
    PORTFOLIO_DRIFT = "portfolio_drift"
    CRON_FAILURE = "cron_failure"
    IC_DECAY = "ic_decay"


# Track last alert time per (channel, level) to enforce dedup interval
_last_alert_time: Dict[str, datetime] = {}


def _should_suppress(key: str) -> bool:
    """Check if an alert for this key was sent recently."""
    last = _last_alert_time.get(key)
    if last is None:
        return False
    elapsed = (datetime.now(timezone.utc) - last).total_seconds()
    return elapsed < ALERT_MIN_INTERVAL_SECONDS


def _record_alert(key: str) -> None:
    """Record that an alert was just sent for this key."""
    _last_alert_time[key] = datetime.now(timezone.utc)


def send_alert(
    channel: AlertChannel,
    level: AlertLevel,
    message: str,
    details: Optional[Dict] = None,
) -> bool:
    """Send an alert via the configured webhook.

    Args:
        channel: Alert category (staleness, evaluator error, etc.)
        level: Severity level (PASS, WARN, HALT)
        message: Human-readable alert message
        details: Optional dict with structured alert details

    Returns:
        True if alert was sent (or alerting is disabled), False on send failure.
    """
    if not ALERT_WEBHOOK_URL:
        logger.debug("Alerting disabled — no ALERT_WEBHOOK_URL configured")
        return True

    # Dedup: suppress repeated alerts within min interval
    dedup_key = f"{channel.value}:{level.value}"
    if _should_suppress(dedup_key):
        logger.debug("Alert suppressed (dedup): %s", dedup_key)
        return True

    payload = {
        "channel": channel.value,
        "level": level.value,
        "message": message,
        "details": details or {},
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": "portfolio-lab",
    }

    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            ALERT_WEBHOOK_URL,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            if resp.status >= 400:
                logger.warning("Alert webhook returned HTTP %d", resp.status)
                return False
        _record_alert(dedup_key)
        logger.info("Alert sent: [%s] %s — %s", level.value, channel.value, message)
        return True
    except urllib.error.URLError as e:
        logger.warning("Alert webhook failed: %s", e)
        return False
    except Exception as e:
        logger.warning("Alert send error: %s", e)
        return False


def check_staleness_and_alert(staleness_data: Dict) -> None:
    """Check signal staleness data and fire alerts on state transitions.

    Args:
        staleness_data: Output from DashboardGenerator._check_signal_staleness()
    """
    stale_signals = staleness_data.get("stale_signals", [])
    healthy_count = staleness_data.get("healthy_count", 0)
    total_count = staleness_data.get("total_count", 0)

    if total_count == 0:
        return

    if len(stale_signals) == 0:
        # All signals healthy — PASS state
        send_alert(
            AlertChannel.SIGNAL_STALENESS,
            AlertLevel.PASS,
            f"All {total_count} signals fresh",
        )
    elif healthy_count > 0:
        # Some signals stale — WARN state
        send_alert(
            AlertChannel.SIGNAL_STALENESS,
            AlertLevel.WARN,
            f"{len(stale_signals)}/{total_count} signals stale: {', '.join(stale_signals)}",
            details={"stale_signals": stale_signals, "healthy_count": healthy_count},
        )
    else:
        # All signals stale — HALT state
        send_alert(
            AlertChannel.SIGNAL_STALENESS,
            AlertLevel.HALT,
            f"ALL {total_count} signals stale — pipeline may be down",
            details={"stale_signals": stale_signals},
        )


def check_drift_and_alert(drift_pct: float, symbol: str = "") -> None:
    """Check portfolio drift and fire alerts.

    Args:
        drift_pct: Current drift percentage from target allocation
        symbol: Optional symbol for single-asset drift
    """
    label = f" ({symbol})" if symbol else ""
    if abs(drift_pct) <= 5.0:
        send_alert(
            AlertChannel.PORTFOLIO_DRIFT,
            AlertLevel.PASS,
            f"Portfolio drift within tolerance{label}: {drift_pct:.1f}%",
        )
    elif abs(drift_pct) <= 10.0:
        send_alert(
            AlertChannel.PORTFOLIO_DRIFT,
            AlertLevel.WARN,
            f"Portfolio drift exceeding 5%{label}: {drift_pct:.1f}%",
            details={"drift_pct": drift_pct, "symbol": symbol},
        )
    else:
        send_alert(
            AlertChannel.PORTFOLIO_DRIFT,
            AlertLevel.HALT,
            f"CRITICAL portfolio drift exceeding 10%{label}: {drift_pct:.1f}%",
            details={"drift_pct": drift_pct, "symbol": symbol},
        )


def check_ic_decay_and_alert(ic_decay_data: Dict) -> None:
    """Check IC decay report and fire alerts for degrading signals.

    Args:
        ic_decay_data: Output from compute_ic_decay_report(), mapping
            signal_name -> {"status": "healthy"|"warning"|"critical"|...,
                           "ic_rolling": float, "ic_trend": str, ...}
    """
    if not ic_decay_data or "error" in ic_decay_data:
        return

    warning_signals = []
    critical_signals = []
    healthy_count = 0

    for signal_name, data in ic_decay_data.items():
        status = data.get("status", "unknown")
        if status == "critical":
            critical_signals.append(signal_name)
        elif status == "warning":
            warning_signals.append(signal_name)
        elif status == "healthy":
            healthy_count += 1

    total = len(ic_decay_data)
    if total == 0:
        return

    if not warning_signals and not critical_signals:
        send_alert(
            AlertChannel.IC_DECAY,
            AlertLevel.PASS,
            f"All {total} signals have healthy IC",
        )
    elif critical_signals:
        send_alert(
            AlertChannel.IC_DECAY,
            AlertLevel.HALT,
            f"{len(critical_signals)} signal(s) with CRITICAL IC decay: {', '.join(critical_signals)}",
            details={
                "critical_signals": critical_signals,
                "warning_signals": warning_signals,
                "healthy_count": healthy_count,
            },
        )
    elif warning_signals:
        send_alert(
            AlertChannel.IC_DECAY,
            AlertLevel.WARN,
            f"{len(warning_signals)} signal(s) with IC decay warning: {', '.join(warning_signals)}",
            details={
                "warning_signals": warning_signals,
                "healthy_count": healthy_count,
            },
        )

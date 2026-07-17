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

from src.monitor.incident_manager import IncidentManager

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
_incident_manager: Optional[IncidentManager] = None


def get_incident_manager() -> IncidentManager:
    """Return the process-local incident manager."""
    global _incident_manager
    if _incident_manager is None:
        _incident_manager = IncidentManager()
    return _incident_manager


def _record_incident_transition(
    channel: AlertChannel,
    level: AlertLevel,
    message: str,
    details: Optional[Dict],
) -> None:
    """Persist alert lifecycle state without blocking webhook delivery."""
    try:
        get_incident_manager().record_alert(
            channel=channel,
            level=level,
            message=message,
            details=details or {},
        )
    except (OSError, ValueError, TypeError) as e:
        logger.warning("Incident lifecycle record failed: %s", e)


def _should_suppress(key: str) -> bool:
    """Check if an alert for this key was sent recently."""
    last = _last_alert_time.get(key)
    if last is None:
        return False
    elapsed = (datetime.now(timezone.utc) - last).total_seconds()
    return elapsed < ALERT_MIN_INTERVAL_SECONDS


def _record_alert(key: str) -> None:
    """Record that an alert was just processed for this key (lifecycle + webhook)."""
    _last_alert_time[key] = datetime.now(timezone.utc)


def _clear_channel_dedup(channel: AlertChannel) -> None:
    """Drop all dedup keys for a channel so a fresh open can fire after PASS."""
    prefix = f"{channel.value}:"
    for key in [k for k in _last_alert_time if k.startswith(prefix)]:
        del _last_alert_time[key]


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
    # PASS always resolves lifecycle and clears channel dedup so a later
    # WARN/HALT can open a new incident immediately.
    if level == AlertLevel.PASS:
        _record_incident_transition(channel, level, message, details)
        _clear_channel_dedup(channel)
        if not ALERT_WEBHOOK_URL:
            logger.debug("Alerting disabled — no ALERT_WEBHOOK_URL configured")
            return True
        # PASS notifications are not deduped; still best-effort webhook.
        return _post_webhook(channel, level, message, details)

    # Dedup applies to both lifecycle and webhook for non-PASS levels so
    # dashboard regen cannot ratchet alert_count while notifications are
    # suppressed (default ALERT_MIN_INTERVAL_SECONDS).
    dedup_key = f"{channel.value}:{level.value}"
    if _should_suppress(dedup_key):
        logger.debug("Alert suppressed (dedup): %s", dedup_key)
        return True

    _record_incident_transition(channel, level, message, details)
    _record_alert(dedup_key)

    if not ALERT_WEBHOOK_URL:
        logger.debug("Alerting disabled — no ALERT_WEBHOOK_URL configured")
        return True

    return _post_webhook(channel, level, message, details)


def _post_webhook(
    channel: AlertChannel,
    level: AlertLevel,
    message: str,
    details: Optional[Dict],
) -> bool:
    """POST webhook payload; returns False on transport/HTTP failure."""
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
        logger.info("Alert sent: [%s] %s — %s", level.value, channel.value, message)
        return True
    except urllib.error.URLError as e:
        logger.warning("Alert webhook failed: %s", e)
        return False
    except Exception as e:
        logger.warning("Alert send error: %s", e)
        return False


def classify_signal_staleness(
    staleness_data: Dict,
) -> Optional[tuple[AlertLevel, str, Dict]]:
    """Classify signal staleness into PASS/WARN/HALT (pure; no I/O).

    Returns ``(level, message, details)`` or ``None`` when there is nothing
    to report (``total_count == 0``).

    Policy:
        - Stale signals drive WARN/HALT as before.
        - Non-empty ``unavailable_signals`` (list) must not produce an
          all-fresh PASS even when ``stale_signals`` is empty.
        - When both stale and unavailable, prioritise stale in the message
          but keep unavailable in details.
    """
    stale_raw = staleness_data.get("stale_signals") or []
    stale_signals = [str(x) for x in stale_raw] if isinstance(stale_raw, list) else []
    unavailable_raw = staleness_data.get("unavailable_signals") or []
    unavailable_signals = (
        [str(x) for x in unavailable_raw] if isinstance(unavailable_raw, list) else []
    )
    unavailable_count = len(unavailable_signals)
    healthy_count = int(staleness_data.get("healthy_count") or 0)
    total_count = int(staleness_data.get("total_count") or 0)

    if total_count == 0:
        return None

    details: Dict = {
        "stale_signals": stale_signals,
        "healthy_count": healthy_count,
        "total_count": total_count,
        "unavailable_count": unavailable_count,
    }
    if unavailable_signals:
        details["unavailable_signals"] = unavailable_signals

    if not stale_signals and unavailable_count == 0:
        return (
            AlertLevel.PASS,
            f"All {total_count} signals fresh",
            details,
        )

    if not stale_signals and unavailable_count > 0:
        names = ", ".join(unavailable_signals[:8])
        suffix = f": {names}" if names else ""
        if len(unavailable_signals) > 8:
            suffix += f" (+{len(unavailable_signals) - 8} more)"
        details["policy"] = "unavailable_signals_nonempty_blocks_all_fresh_pass"
        return (
            AlertLevel.WARN,
            (
                f"{unavailable_count}/{total_count} signals unavailable "
                f"(partial availability; not all-fresh){suffix}"
            ),
            details,
        )

    if healthy_count > 0:
        extra = ""
        if unavailable_count:
            extra = f"; {unavailable_count} unavailable"
        return (
            AlertLevel.WARN,
            f"{len(stale_signals)}/{total_count} signals stale: "
            f"{', '.join(stale_signals)}{extra}",
            details,
        )

    return (
        AlertLevel.HALT,
        f"ALL {total_count} signals stale — pipeline may be down",
        details,
    )


def check_staleness_and_alert(staleness_data: Dict) -> None:
    """Check signal staleness data and fire alerts on state transitions.

    Args:
        staleness_data: Output from DashboardGenerator._check_signal_staleness()
    """
    classified = classify_signal_staleness(staleness_data)
    if classified is None:
        return
    level, message, details = classified
    send_alert(
        AlertChannel.SIGNAL_STALENESS,
        level,
        message,
        details=details if level != AlertLevel.PASS else None,
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
        ic_decay_data: Output from compute_ic_decay_report(). Accepts either
            the wrapper shape with a ``signals`` mapping or the legacy direct
            signal mapping.
    """
    if not ic_decay_data or "error" in ic_decay_data:
        return

    signal_rows = ic_decay_data.get("signals") if isinstance(ic_decay_data, dict) else None
    if isinstance(signal_rows, dict):
        ic_decay_data = signal_rows

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

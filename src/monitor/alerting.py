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
from typing import Dict, Optional

from src.monitor.incident_manager import IncidentManager

logger = logging.getLogger(__name__)

ALERT_WEBHOOK_URL = os.environ.get("ALERT_WEBHOOK_URL", "").strip()
# Secret-safe file override (Item 35 / I2): path to a file whose (trimmed)
# contents are the webhook URL. The URL itself is never logged or exposed.
ALERT_WEBHOOK_URL_FILE = os.environ.get("ALERT_WEBHOOK_URL_FILE", "").strip()
ALERT_MIN_INTERVAL_SECONDS = int(os.environ.get("ALERT_MIN_INTERVAL_SECONDS", "300"))

# Producer-aware stale-only exception. This is deliberately separate from
# optional-unavailable criticality: the generated section remains required,
# but projection lag alone cannot sole-page kill.
_ADVISORY_STALE_ONLY_SIGNALS = frozenset({"alternative_data"})


class AlertLevel(str, Enum):
    """Alert severity levels following PASS→WARN→HALT state machine."""
    PASS = "pass"
    WARN = "warn"
    HALT = "halt"


class AlertChannel(str, Enum):
    """Alert categories for different pipeline events."""
    SIGNAL_STALENESS = "signal_staleness"
    SIGNAL_RECOVERY = "signal_recovery"
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


def _resolve_webhook_url() -> str:
    """Resolve the webhook URL: ``ALERT_WEBHOOK_URL`` env > module constant > ``ALERT_WEBHOOK_URL_FILE`` file > disabled.

    Call-time resolution (env/file read at call) so tests can set/clear
    cleanly; the module constant is kept as a fallback for the existing
    ``patch("src.monitor.alerting.ALERT_WEBHOOK_URL", ...)`` seam.
    """
    env_url = os.environ.get("ALERT_WEBHOOK_URL", "").strip()
    if env_url:
        return env_url
    if ALERT_WEBHOOK_URL:
        return ALERT_WEBHOOK_URL
    file_path = os.environ.get("ALERT_WEBHOOK_URL_FILE", "").strip()
    if file_path:
        try:
            with open(file_path, encoding="utf-8") as f:
                return f.read().strip()
        except OSError:
            logger.warning("ALERT_WEBHOOK_URL_FILE unreadable: %s", file_path)
    return ""


def webhook_config_state() -> tuple[bool, str]:
    """Return ``(configured, source)`` for runtime disclosure (Item 35 / I2).

    Source ∈ env / file / none. The URL itself is never returned, logged, or
    exposed — only the boolean + source are disclosed.
    """
    if os.environ.get("ALERT_WEBHOOK_URL", "").strip() or ALERT_WEBHOOK_URL:
        return True, "env"
    if os.environ.get("ALERT_WEBHOOK_URL_FILE", "").strip():
        return True, "file"
    return False, "none"


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
        if not _resolve_webhook_url():
            logger.debug("Alerting disabled — no webhook URL configured")
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

    if not _resolve_webhook_url():
        logger.debug("Alerting disabled — no webhook URL configured")
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
            _resolve_webhook_url(),
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


def _is_advisory_stale_signal(name: str, signal_roles: Optional[Dict] = None) -> bool:
    """True when a stale signal is advisory-only and must not sole-page kill."""
    key = str(name)
    if signal_roles and isinstance(signal_roles, dict):
        role = str(signal_roles.get(key) or "").lower()
        if role in {
            "advisory_shadow",
            "advisory_non_routed",
            "advisory_sleeve",
            "advisory",
        }:
            return True
        if role in {"execution_routed", "required", "authoritative"}:
            return False
    from src.monitor.signal_ownership import blocks_all_fresh

    return key in _ADVISORY_STALE_ONLY_SIGNALS or not blocks_all_fresh(key)


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
        - Batch IM DP: sole stale advisory_shadow (e.g. alternative_data) is
          PASS — advisory cannot sole-escalate kill. Required stale still WARN.
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
    signal_roles = staleness_data.get("signal_roles")
    if not isinstance(signal_roles, dict):
        signal_roles = {}

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
    projection_lag_raw = staleness_data.get("projection_lag_signals") or []
    projection_lag_signals = (
        [str(x) for x in projection_lag_raw] if isinstance(projection_lag_raw, list) else []
    )
    if projection_lag_signals:
        details["projection_lag_signals"] = projection_lag_signals
        details["policy_note"] = (
            "projection_lag: producer fresher than embedded signals; "
            "not treated as producer-stale for kill escalation"
        )

    # Split actionable vs advisory stale before level decision (Batch IM DP).
    advisory_stale = [
        s for s in stale_signals if _is_advisory_stale_signal(s, signal_roles)
    ]
    actionable_stale = [
        s for s in stale_signals if not _is_advisory_stale_signal(s, signal_roles)
    ]
    if advisory_stale:
        details["advisory_stale"] = advisory_stale
    if actionable_stale != stale_signals:
        details["actionable_stale"] = actionable_stale

    if not actionable_stale and unavailable_count == 0:
        if advisory_stale:
            details["policy"] = "advisory_shadow_stale_only_pass"
            return (
                AlertLevel.PASS,
                (
                    f"All required signals fresh "
                    f"({len(advisory_stale)} advisory-only stale skipped: "
                    f"{', '.join(advisory_stale[:4])})"
                ),
                details,
            )
        if projection_lag_signals:
            return (
                AlertLevel.PASS,
                (
                    f"All {total_count} signals producer-fresh "
                    f"({len(projection_lag_signals)} projection lag)"
                ),
                details,
            )
        return (
            AlertLevel.PASS,
            f"All {total_count} signals fresh",
            details,
        )

    if not actionable_stale and unavailable_count > 0:
        # Intentional lab gaps (ML-off research, FRED key absent) must not block
        # all-fresh PASS / sticky warning kill. Prefer ownership annotation when
        # present; otherwise treat full unavailable list as actionable.
        ownership = staleness_data.get("unavailable_ownership")
        from src.monitor.signal_ownership import blocks_all_fresh

        actionable_unavailable = [
            signal for signal in unavailable_signals if blocks_all_fresh(signal)
        ]
        intentional_count = 0
        optional_unavailable: list[str] = [
            signal
            for signal in unavailable_signals
            if not blocks_all_fresh(signal)
        ]
        if isinstance(ownership, list) and ownership:
            actionable_unavailable = [
                str(r.get("signal"))
                for r in ownership
                if isinstance(r, dict)
                and r.get("blocks_all_fresh", True)
                and not (
                    r.get("intentional_lab_gap")
                    or r.get("intentional_when_ml_off")
                    or r.get("intentional_when_fred_unconfigured")
                )
            ]
            intentional_count = sum(
                1
                for row in ownership
                if isinstance(row, dict) and row.get("intentional_lab_gap")
            )
            details["actionable_unavailable"] = actionable_unavailable
            details["intentional_lab_gap_count"] = intentional_count
            optional_unavailable = [
                str(r.get("signal"))
                for r in ownership
                if isinstance(r, dict) and not r.get("blocks_all_fresh", True)
            ]
        details["optional_advisory_unavailable"] = optional_unavailable

        if not actionable_unavailable:
            details["policy"] = (
                "advisory_or_intentional_only_pass"
                if advisory_stale or optional_unavailable
                else "intentional_lab_gaps_only_pass"
            )
            return (
                AlertLevel.PASS,
                (
                    f"All required signals fresh "
                    f"({intentional_count} intentional lab gaps skipped"
                    + (
                        f"; {len(optional_unavailable)} optional advisory unavailable"
                        if optional_unavailable
                        else ""
                    )
                    + (
                        f"; {len(advisory_stale)} advisory-only stale"
                        if advisory_stale
                        else ""
                    )
                    + ")"
                    if intentional_count or advisory_stale or optional_unavailable
                    else f"All {total_count} signals fresh"
                ),
                details,
            )

        names = ", ".join(actionable_unavailable[:8])
        suffix = f": {names}" if names else ""
        if len(actionable_unavailable) > 8:
            suffix += f" (+{len(actionable_unavailable) - 8} more)"
        details["policy"] = "unavailable_signals_nonempty_blocks_all_fresh_pass"
        return (
            AlertLevel.WARN,
            (
                f"{len(actionable_unavailable)}/{total_count} signals unavailable "
                f"(partial availability; not all-fresh){suffix}"
            ),
            details,
        )

    # Actionable stale remains → WARN/HALT using actionable list only.
    # HALT when no healthy signals remain (pipeline down for required set),
    # even if some of the stale list is advisory-only noise.
    if healthy_count == 0 and actionable_stale:
        return (
            AlertLevel.HALT,
            f"ALL {total_count} signals stale — pipeline may be down",
            details,
        )

    extra = ""
    if unavailable_count:
        extra = f"; {unavailable_count} unavailable"
    if advisory_stale:
        extra += f"; {len(advisory_stale)} advisory-only stale skipped"
    return (
        AlertLevel.WARN,
        f"{len(actionable_stale)}/{total_count} signals stale: "
        f"{', '.join(actionable_stale)}{extra}",
        details,
    )


def check_staleness_and_alert(staleness_data: Dict) -> None:
    """Check signal staleness data and fire alerts on state transitions.

    Args:
        staleness_data: Output from DashboardGenerator._check_signal_staleness()
    """
    # Annotate ownership for operator recovery (does not change level policy).
    try:
        from src.monitor.signal_ownership import annotate_unavailable_signals, recovery_summary

        ml_on = os.environ.get("PORTFOLIO_LAB_ENABLE_ML", "0") == "1"
        ownership = annotate_unavailable_signals(
            staleness_data.get("unavailable_signals") or [],
            ml_enabled=ml_on,
        )
        if ownership:
            staleness_data = dict(staleness_data)
            staleness_data["unavailable_ownership"] = ownership
            staleness_data["recovery"] = recovery_summary(ownership)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Signal ownership annotation skipped: %s", exc)

    classified = classify_signal_staleness(staleness_data)
    if classified is None:
        return
    level, message, details = classified
    if details is not None and isinstance(staleness_data, dict):
        if staleness_data.get("unavailable_ownership"):
            details = dict(details)
            details["unavailable_ownership"] = staleness_data["unavailable_ownership"]
            details["recovery"] = staleness_data.get("recovery")
    send_alert(
        AlertChannel.SIGNAL_STALENESS,
        level,
        message,
        details=details if level != AlertLevel.PASS else None,
    )
    # Distinct recovery channel when unavailability is sustained under kill halt.
    try:
        check_sustained_unavailability_and_alert(staleness_data)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Sustained unavailability recovery check skipped: %s", exc)


def check_sustained_unavailability_and_alert(
    staleness_data: Dict,
    *,
    data_dir: Optional[str] = None,
    min_unavailable: int | None = None,
    min_hours: float | None = None,
) -> bool:
    """Fire SIGNAL_RECOVERY WARN when unavailability persists under kill halt.

    Does **not** clear kill_switch. Returns True when a recovery alert was sent.
    """
    from pathlib import Path

    from src.paths import DATA_DIR as _DEFAULT_DATA

    root = Path(data_dir) if data_dir is not None else Path(_DEFAULT_DATA)
    threshold = min_unavailable
    if threshold is None:
        try:
            threshold = int(os.environ.get("SIGNAL_RECOVERY_MIN_UNAVAILABLE", "5"))
        except ValueError:
            threshold = 5
    hours_needed = min_hours
    if hours_needed is None:
        try:
            hours_needed = float(os.environ.get("SIGNAL_RECOVERY_MIN_HOURS", "2"))
        except ValueError:
            hours_needed = 2.0

    unavailable = staleness_data.get("unavailable_signals") or []
    if not isinstance(unavailable, list):
        unavailable = []
    ownership = staleness_data.get("unavailable_ownership")
    if not ownership:
        try:
            from src.monitor.signal_ownership import annotate_unavailable_signals, recovery_summary

            ml_on = os.environ.get("PORTFOLIO_LAB_ENABLE_ML", "0") == "1"
            ownership = annotate_unavailable_signals(unavailable, ml_enabled=ml_on)
            recovery = recovery_summary(ownership)
        except Exception:
            ownership = []
            recovery = {}
    else:
        recovery = staleness_data.get("recovery") or {}

    actionable = [
        r
        for r in (ownership or [])
        if isinstance(r, dict)
        and not (
            r.get("intentional_lab_gap")
            or r.get("intentional_when_ml_off")
            or r.get("intentional_when_fred_unconfigured")
        )
    ]
    if len(actionable) < threshold:
        return False

    # Require active kill authority (sustained under halt / restrict)
    kill_path = root / "kill_switch.json"
    kill_level = None
    kill_enabled = False
    kill_age_hours = None
    if kill_path.exists():
        try:
            kill = json.loads(kill_path.read_text(encoding="utf-8"))
            if isinstance(kill, dict):
                kill_enabled = bool(kill.get("enabled"))
                kill_level = kill.get("level")
                ts = kill.get("timestamp")
                if isinstance(ts, str) and ts:
                    try:
                        kt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                        if kt.tzinfo is None:
                            kt = kt.replace(tzinfo=timezone.utc)
                        kill_age_hours = (
                            datetime.now(timezone.utc) - kt.astimezone(timezone.utc)
                        ).total_seconds() / 3600.0
                    except ValueError:
                        kill_age_hours = None
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            kill_enabled = False

    if not kill_enabled:
        return False
    # Only under execution-blocking kill levels. WARNING-level kill often comes
    # from this same SIGNAL_RECOVERY channel (WARN→p2→kill warning), which would
    # re-fire forever under its own advisory authority.
    level_norm = str(kill_level or "").strip().lower()
    if level_norm not in {"halt", "restrict"}:
        return False
    if kill_age_hours is not None and kill_age_hours < hours_needed:
        return False
    # If no kill timestamp, still fire when threshold met (fail-visible)

    jobs = recovery.get("jobs_to_rerun") if isinstance(recovery, dict) else None
    targets = recovery.get("make_targets") if isinstance(recovery, dict) else None
    message = (
        f"Sustained overlay unavailability under kill "
        f"(level={kill_level}): {len(actionable)} actionable signals unavailable. "
        f"Re-run producers then make ops-regen; do not auto-clear kill."
    )
    details = {
        "unavailable_ownership": ownership,
        "recovery": recovery,
        "actionable_unavailable_count": len(actionable),
        "kill_level": kill_level,
        "kill_age_hours": round(kill_age_hours, 2) if kill_age_hours is not None else None,
        "jobs_to_rerun": jobs,
        "make_targets": targets,
        "policy": "recovery_advisory_only_no_kill_clear",
    }
    send_alert(
        AlertChannel.SIGNAL_RECOVERY,
        AlertLevel.WARN,
        message,
        details=details,
    )
    return True


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
    eligible_critical = []
    eligible_warning = []
    ineligible_critical = []
    healthy_count = 0
    insufficient_count = 0

    for signal_name, data in ic_decay_data.items():
        if not isinstance(data, dict):
            continue
        status = data.get("status", "unknown")
        # Task 2A: only contract-aligned rows may drive halt-authoritative IC
        # control alerts; descriptive status is preserved for operators.
        control_eligible = bool(data.get("control_eligible"))
        if status == "critical":
            critical_signals.append(signal_name)
            if control_eligible:
                eligible_critical.append(signal_name)
            else:
                ineligible_critical.append(signal_name)
        elif status == "warning":
            warning_signals.append(signal_name)
            if control_eligible:
                eligible_warning.append(signal_name)
        elif status == "healthy":
            healthy_count += 1
        elif status == "insufficient_data":
            insufficient_count += 1

    total = len(ic_decay_data)
    if total == 0:
        return

    if eligible_critical:
        send_alert(
            AlertChannel.IC_DECAY,
            AlertLevel.HALT,
            f"{len(eligible_critical)} signal(s) with CRITICAL IC decay: {', '.join(eligible_critical)}",
            details={
                "critical_signals": critical_signals,
                "control_eligible_critical_signals": eligible_critical,
                "warning_signals": warning_signals,
                "healthy_count": healthy_count,
            },
        )
    elif eligible_warning:
        send_alert(
            AlertChannel.IC_DECAY,
            AlertLevel.WARN,
            f"{len(eligible_warning)} signal(s) with IC decay warning: {', '.join(eligible_warning)}",
            details={
                "warning_signals": warning_signals,
                "control_eligible_warning_signals": eligible_warning,
                "healthy_count": healthy_count,
            },
        )
    elif critical_signals or warning_signals:
        # Descriptive-only critical/warning rows (misaligned/legacy/ambiguous)
        # are disclosed but cannot escalate. PASS here is safe for incidents:
        # ic_decay incidents are manual-review-required and never auto-resolve.
        # G3 (2026-08-11): message counts derive from the same descriptive
        # lists that populate `details` so operators can reconcile the message
        # with details.critical_signals / warning_signals (observed mismatch:
        # message "5 critical, 0 warning" vs details lists 4+1). In this branch
        # every critical/warning row is ineligible (eligible branches above),
        # so totals equal the ineligible subsets — the message wording is
        # unchanged, only the count source is unified with `details`.
        send_alert(
            AlertChannel.IC_DECAY,
            AlertLevel.PASS,
            (
                f"IC control evidence ineligible: {len(critical_signals)} "
                f"critical, {len(warning_signals)} warning "
                "signal(s) lack complete contract alignment"
            ),
            details={
                "critical_signals": critical_signals,
                "warning_signals": warning_signals,
                "healthy_count": healthy_count,
                "ineligible_critical_signals": ineligible_critical,
                "ineligible_warning_signals": [
                    s for s in warning_signals if s not in eligible_warning
                ],
                "policy": "control_ineligible_no_escalation",
            },
        )
    else:
        # PASS clears prior false HALT from thin-history critical misclassification.
        # Warm-up (all insufficient_data) is not an operational failure.
        if healthy_count == 0 and insufficient_count > 0:
            send_alert(
                AlertChannel.IC_DECAY,
                AlertLevel.PASS,
                (
                    f"IC monitor warming up: {insufficient_count} signal(s) below "
                    f"min observations for status (no kill escalation)"
                ),
                details={
                    "insufficient_count": insufficient_count,
                    "policy": "thin_history_no_kill",
                },
            )
        elif healthy_count > 0:
            send_alert(
                AlertChannel.IC_DECAY,
                AlertLevel.PASS,
                f"All {healthy_count} resolved signals have healthy IC"
                + (f" ({insufficient_count} warming up)" if insufficient_count else ""),
            )
        else:
            return

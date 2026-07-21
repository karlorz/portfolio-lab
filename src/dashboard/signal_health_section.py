"""Signal health and FRED readiness sections for health.json."""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "SIGNAL_HEALTH_EXCEPTIONS",
    "attach_signal_quality_disclosure",
    "build_fred_readiness_section",
    "build_signal_health_section",
    "fred_readiness_unavailable_payload",
    "signal_health_unavailable_payload",
]

SIGNAL_HEALTH_EXCEPTIONS = (
    ImportError,
    AttributeError,
    KeyError,
    ValueError,
    TypeError,
    RuntimeError,
    OSError,
)


def signal_health_unavailable_payload(exc: BaseException) -> dict[str, Any]:
    return {
        "error": f"Failed to get signal health: {exc}",
        "status": "unavailable",
    }


def fred_readiness_unavailable_payload(exc: BaseException) -> dict[str, Any]:
    return {
        "status": "warning",
        "readiness": "unknown",
        "ready": True,
        "blocking": False,
        "reason": "readiness_check_unavailable",
        "message": f"FRED readiness check unavailable: {exc}",
        "remediation": "Verify fredapi availability and FRED readiness dependencies.",
    }


def attach_signal_quality_disclosure(
    report: dict[str, Any],
    *,
    data_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Batch CM: when healthy==0 of N, disclose quality gate + weight freeze.

    Deep-research SRE guidance: zero healthy sources must surface at top-level
    (not only demote system_status) and ensemble weight freeze age must be
    visible so operators do not treat ops-green as quality-green.
    """
    out = dict(report)
    summary = out.get("summary") if isinstance(out.get("summary"), dict) else {}
    try:
        healthy = int(summary.get("healthy") or 0)
        total = int(
            summary.get("total_tracked") or summary.get("total") or 0
        )
        degraded = int(summary.get("degraded") or 0)
        unhealthy = int(summary.get("unhealthy") or 0)
    except (TypeError, ValueError):
        return out

    if total <= 0:
        return out

    zero_healthy = healthy == 0
    quality: dict[str, Any] = {
        "schema_version": "signal-quality-disclosure/v1",
        "zero_healthy_sources": zero_healthy,
        "healthy": healthy,
        "degraded": degraded,
        "unhealthy": unhealthy,
        "total_tracked": total,
        "badge": f"{healthy}/{total} healthy sources",
        "disclosure": (
            "Signal quality is independent of ops health. When healthy==0, "
            "champion baseline remains sole live allocation authority; "
            "ensemble sleeves are advisory (live_authoritative: false)."
        ),
    }
    if zero_healthy:
        quality["severity"] = "degraded"
        quality["operator_action"] = (
            "Do not promote ensemble weights; investigate IC/accuracy on "
            "degraded sleeves; optional hard-zero unhealthy arms (ADR pending)."
        )
    # Batch CP: disclose when fleet is under collapsed_recency thresholds
    collapsed_n = int(summary.get("window_collapse_90_60_count") or 0)
    if collapsed_n > 0 or zero_healthy:
        try:
            from src.signals.health_tracker import (
                HEALTH_THRESHOLD_HEALTHY_COLLAPSED,
                HEALTH_THRESHOLD_HEALTHY_FULL,
                status_thresholds_for_scheme,
            )

            quality["threshold_policy"] = {
                "full_scheme_healthy_min": HEALTH_THRESHOLD_HEALTHY_FULL,
                "collapsed_scheme_healthy_min": HEALTH_THRESHOLD_HEALTHY_COLLAPSED,
                "window_collapse_90_60_count": collapsed_n,
                "note": (
                    "Under collapsed_recency_40_60, healthy uses "
                    f"{HEALTH_THRESHOLD_HEALTHY_COLLAPSED} not "
                    f"{HEALTH_THRESHOLD_HEALTHY_FULL} (Batch CP / c328)."
                ),
            }
        except Exception:  # noqa: BLE001
            pass
        # List worst sleeves for runbook
        scores = out.get("scores")
        worst: list[dict[str, Any]] = []
        if isinstance(scores, dict):
            for name, row in scores.items():
                if not isinstance(row, dict):
                    continue
                worst.append(
                    {
                        "source": name,
                        "status": row.get("status"),
                        "health_score": row.get("health_score"),
                        "accuracy_30d": row.get("accuracy_30d"),
                        "ic": row.get("ic"),
                    }
                )
        elif isinstance(scores, list):
            for row in scores:
                if isinstance(row, dict):
                    worst.append(
                        {
                            "source": row.get("source") or row.get("name"),
                            "status": row.get("status"),
                            "health_score": row.get("health_score"),
                            "accuracy_30d": row.get("accuracy_30d"),
                            "ic": row.get("ic"),
                        }
                    )
        worst.sort(
            key=lambda r: (
                0 if r.get("status") == "unhealthy" else 1,
                float(r.get("health_score") or 0.0),
            )
        )
        quality["worst_sleeves"] = worst[:5]

    # Ensemble static weights file age (often frozen for weeks/months)
    try:
        from src.paths import DATA_DIR as _DATA

        root = Path(data_dir) if data_dir is not None else Path(_DATA)
    except Exception:  # noqa: BLE001
        root = Path(data_dir) if data_dir is not None else Path("data")

    freeze: dict[str, Any] = {
        "ensemble_weights_path": str(root / "ensemble_weights.json"),
        "recommended_max_freeze_days": 7,
        "policy": (
            "weight_freeze_active only when healthy==0 (do not auto-reoptimize; "
            "champion 46/38/16 remains live authority). File age >7d is "
            "weight_file_stale advisory only while healthy>0 (Batch CQ) — "
            "stale mtime must not freeze after SH recovery."
        ),
    }
    ew = root / "ensemble_weights.json"
    if ew.is_file():
        freeze["ensemble_weights_present"] = True
        try:
            mtime = ew.stat().st_mtime
            age_days = max(
                0.0,
                (datetime.now(timezone.utc).timestamp() - mtime) / 86400.0,
            )
            freeze["ensemble_weights_mtime"] = datetime.fromtimestamp(
                mtime, tz=timezone.utc
            ).isoformat()
            freeze["ensemble_weights_age_days"] = round(age_days, 2)
            freeze["weight_file_stale"] = age_days > 7.0
            # Batch CQ: freeze is a zero-healthy gate, not an age gate.
            # Age residual after CP (3/9 healthy + 46d file) was false freeze.
            freeze["weight_freeze_active"] = zero_healthy
            freeze["freeze_reason"] = (
                "zero_healthy_sources" if zero_healthy else None
            )
            if freeze["weight_file_stale"] and not zero_healthy:
                freeze["stale_note"] = (
                    "ensemble_weights.json exceeds recommended_max_freeze_days; "
                    "advisory only while healthy>0 — schedule adaptive refresh, "
                    "do not treat as weight_freeze_active"
                )
        except OSError:
            freeze["weight_freeze_active"] = zero_healthy
            freeze["freeze_reason"] = (
                "zero_healthy_sources" if zero_healthy else None
            )
    else:
        freeze["ensemble_weights_present"] = False
        freeze["weight_freeze_active"] = zero_healthy
        freeze["freeze_reason"] = "zero_healthy_sources" if zero_healthy else None

    # Adaptive state is still advisory when SH quality is zero-healthy
    aw = root / "adaptive_weights_state.json"
    if aw.is_file():
        try:
            import json

            state = json.loads(aw.read_text(encoding="utf-8"))
            if isinstance(state, dict):
                freeze["adaptive_weights_timestamp"] = state.get("timestamp")
                freeze["adaptive_regime"] = state.get("regime")
                freeze["adaptive_note"] = (
                    "adaptive_weights_state may still update; live routing "
                    "must ignore it while zero_healthy_sources is true"
                )
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass

    quality["ensemble_weight_freeze"] = freeze
    out["quality_disclosure"] = quality
    # Promote badge onto summary for compact consumers
    if isinstance(out.get("summary"), dict):
        summary2 = dict(out["summary"])
        summary2["quality_badge"] = quality["badge"]
        summary2["zero_healthy_sources"] = zero_healthy
        if freeze.get("ensemble_weights_age_days") is not None:
            summary2["ensemble_weights_age_days"] = freeze.get(
                "ensemble_weights_age_days"
            )
        if freeze.get("weight_freeze_active"):
            summary2["ensemble_weight_freeze_active"] = True
        elif freeze.get("weight_file_stale"):
            # Advisory stale file — not freeze (Batch CQ)
            summary2["ensemble_weights_file_stale"] = True
        out["summary"] = summary2
    return out


def build_signal_health_section(
    *,
    log_error: Callable[[str, Exception], None] | None = None,
    resolve_labels: bool = True,
    resolve_max_days: int | None = None,
) -> dict[str, Any]:
    """Load SignalHealthTracker report for health.json.

    By default runs a bounded ``resolve_pending_labels`` pass first so production
    health cycles actually call ``update_actual_directions`` (label backlog).
    """
    try:
        from src.signals.health_tracker import (
            DEFAULT_RESOLVE_MAX_DAYS,
            SignalHealthTracker,
        )

        tracker = SignalHealthTracker()
        label_resolve: dict[str, Any] | None = None
        if resolve_labels:
            max_days = (
                DEFAULT_RESOLVE_MAX_DAYS
                if resolve_max_days is None
                else int(resolve_max_days)
            )
            try:
                # Newest-first keeps IC staging window fresh; then a small
                # oldest-first batch drains residual backlog without a new cron.
                catchup_days = max(1, min(5, max_days // 3 or 1))
                newest = tracker.resolve_pending_labels(
                    max_days=max_days, oldest_first=False
                )
                oldest = tracker.resolve_pending_labels(
                    max_days=catchup_days, oldest_first=True
                )
                label_resolve = {
                    "newest_first": newest,
                    "oldest_first": oldest,
                    "predictions_updated": int(newest.get("predictions_updated") or 0)
                    + int(oldest.get("predictions_updated") or 0),
                    "dates_resolved": int(newest.get("dates_resolved") or 0)
                    + int(oldest.get("dates_resolved") or 0),
                }
            except SIGNAL_HEALTH_EXCEPTIONS as resolve_exc:
                # Never block health report on resolve failures
                logger.warning("signal_health label resolve skipped: %s", resolve_exc)
                label_resolve = {
                    "error": str(resolve_exc),
                    "predictions_updated": 0,
                }

        report = tracker.get_health_report()
        out: dict[str, Any] = {
            "timestamp": report.get("timestamp"),
            "summary": report.get("summary", {}),
            "scores": report.get("scores", {}),
            "alerts": report.get("alerts", []),
            "overall_health": report.get("overall_health", "unknown"),
            "status": report.get("status", report.get("overall_health", "unknown")),
            "label_horizon": report.get("label_horizon"),
        }
        if label_resolve is not None:
            out["label_resolve"] = label_resolve
        return attach_signal_quality_disclosure(out)
    except SIGNAL_HEALTH_EXCEPTIONS as exc:
        if log_error:
            log_error("signal_health", exc)
        else:
            logger.warning("Signal health not available: %s", exc)
        return signal_health_unavailable_payload(exc)


def build_fred_readiness_section(
    *,
    log_error: Callable[[str, Exception], None] | None = None,
) -> dict[str, Any]:
    """Assess FRED credential/cache readiness for health.json."""
    try:
        from src.data.fred_data import get_fred_md_cache_health
        from src.monitor.fred_readiness import assess_fred_readiness

        return assess_fred_readiness(get_fred_md_cache_health())
    except SIGNAL_HEALTH_EXCEPTIONS as exc:
        if log_error:
            log_error("fred_readiness", exc)
        else:
            logger.warning("FRED readiness not available: %s", exc)
        return fred_readiness_unavailable_payload(exc)

#!/usr/bin/env python3
"""
Portfolio-Lab Alpha: Dashboard Generator
Creates static dashboard from SQLite data for Vite/React app consumption.
"""

import json
import sqlite3
import logging
import os
import shutil
from collections import deque
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

from src.paths import BASE_ALLOCATION, YIELDS_JSON, DATA_DIR, PUBLIC_DATA_DIR, MARKET_DB, REGIME_OVERRIDES, sqlite_connect
from src.strategy.regime_allocation import (
    get_regime_allocation_with_override,
    normalize_allocation_regime,
)
from src.utils import safe_get, classify_vix_regime
from src.backtest.metrics import save_results_json
from src.dashboard.public_data_index import build_public_data_index
from src.monitor.hermes_cron import resolve_hermes_cron_jobs_path
from src.monitor.signal_schemas import validate_all_signals, validate_signal
from src.dashboard.cron_scheduler_section import build_cron_scheduler_section
from src.dashboard.data_freshness_section import build_data_freshness_section
from src.dashboard.health_report import (
    derive_system_status,
    summarize_stale_symbol_count,
)
from src.dashboard.data_pipeline_slo_section import build_data_pipeline_slo_section
from src.dashboard.health_slo_alerts import build_health_slo_alerts
from src.dashboard.kill_authority import (
    allocation_roles_under_kill,
    build_kill_switch_alert,
    elevate_system_status_for_kill,
    load_kill_switch_payload,
    load_open_incidents_summary,
    project_compact_kill_fields,
    project_kill_switch_fields,
)
from src.dashboard.signal_health_section import (
    build_fred_readiness_section,
    build_signal_health_section,
)

__all__ = [
    "DashboardGenerator",
    "PUBLIC_DIR",
    "DB_PATH",
    "project_alternative_data_signal",
]

# Legacy flat keys (pre seven-component producer) → panel component names
_ALT_DATA_LEGACY_COMPONENT_KEYS = (
    ("earnings", "earnings_sentiment", "earnings_confidence"),
    ("news", "news_sentiment", "news_confidence"),
    ("jobs", "jobs_signal", "jobs_confidence"),
    ("social", "social_sentiment", "social_confidence"),
)


def project_alternative_data_signal(alt_data_raw: Dict[str, Any]) -> Dict[str, Any]:
    """Project producer alternative_data_latest.json into signals.json shape.

    Primary path: ``raw_data.components`` / ``component_confidences`` / ``weights``
    (current seven-component producer). Fallback: legacy flat
    earnings/news/jobs/social keys when the components map is absent.
    """
    raw = alt_data_raw.get("raw_data") if isinstance(alt_data_raw, dict) else None
    if not isinstance(raw, dict):
        raw = {}

    components_src = raw.get("components")
    confidences_src = raw.get("component_confidences") if isinstance(raw.get("component_confidences"), dict) else {}
    weights_src = raw.get("weights") if isinstance(raw.get("weights"), dict) else {}
    components: Dict[str, Dict[str, Any]] = {}

    if isinstance(components_src, dict) and components_src:
        for key, score in components_src.items():
            components[str(key)] = {
                "score": score,
                "confidence": confidences_src.get(key),
                "weight": weights_src.get(key),
            }
    else:
        # Legacy fallback only when producer lacks the components map
        for name, score_key, conf_key in _ALT_DATA_LEGACY_COMPONENT_KEYS:
            components[name] = {
                "score": raw.get(score_key),
                "confidence": raw.get(conf_key),
                "weight": weights_src.get(name),
            }

    return {
        "regime": alt_data_raw.get("regime"),
        "probability": alt_data_raw.get("probability"),
        "confidence": alt_data_raw.get("confidence"),
        "timestamp": alt_data_raw.get("timestamp"),
        "components": components,
        "composite_score": raw.get("composite_score"),
        "z_score": raw.get("z_score"),
        "sources_count": raw.get("sources_count"),
        "data_freshness_hours": raw.get("data_freshness_hours"),
        "producer_path": "data/signals/alternative_data_latest.json",
    }


def load_alternative_data_producer_timestamp(
    data_dir: Path | None = None,
) -> str | None:
    """Return producer alternative_data_latest.json timestamp when readable."""
    root = Path(data_dir) if data_dir is not None else DATA_DIR
    path = root / "signals" / "alternative_data_latest.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    ts = payload.get("timestamp") or payload.get("generated_at")
    return ts if isinstance(ts, str) and ts.strip() else None


def refresh_public_alternative_data_projection(
    *,
    data_dir: Path | None = None,
    public_dir: Path | None = None,
) -> bool:
    """Bounded refresh: rewrite alternative_data section in public signals.json.

    Called after producer save so operators do not wait for the next full
    dashboard cron when only alt-data was updated. Returns True when the
    public artifact was updated.
    """
    root = Path(data_dir) if data_dir is not None else DATA_DIR
    public = Path(public_dir) if public_dir is not None else PUBLIC_DIR
    producer = root / "signals" / "alternative_data_latest.json"
    signals_path = public / "signals.json"
    if not producer.exists() or not signals_path.exists():
        return False
    try:
        alt_raw = json.loads(producer.read_text(encoding="utf-8"))
        signals = json.loads(signals_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        logger.warning("Alt-data projection refresh skipped (read failed): %s", exc)
        return False
    if not isinstance(signals, dict) or not isinstance(alt_raw, dict):
        return False

    projected = project_alternative_data_signal(alt_raw)
    signals["alternative_data"] = projected
    # Recompute only alt-related staleness hints on the embedded block is hard
    # without full generator; stamp lag metadata for operators.
    signals["alternative_data_projection"] = {
        "refreshed_at": datetime.now(timezone.utc).isoformat(),
        "producer_timestamp": projected.get("timestamp"),
        "source": "bounded_alt_data_refresh",
    }
    try:
        save_results_json(signals, output_path=str(signals_path))
    except (OSError, TypeError, ValueError) as exc:
        logger.warning("Alt-data projection refresh write failed: %s", exc)
        return False
    logger.info(
        "Refreshed public alternative_data projection at %s (producer_ts=%s)",
        signals_path,
        projected.get("timestamp"),
    )
    return True

# Map ensemble signal source names to staleness check keys
_ENSEMBLE_STALENESS_MAP = {
    "multi_speed_momentum": "ensemble_voting",
    "cross_asset_rv": "ensemble_voting",
    "international_momentum": "ensemble_voting",
    "alternative_data": "alternative_data",
    "cross_asset_regime_arb": "ensemble_voting",
    "unified_overlay": "ensemble_voting",
}

logger = logging.getLogger(__name__)

PUBLIC_DATA_DIST_MIRROR_FILES = ("source_manifest.json", "index.json", "health.json")


def _is_predictive_fred_macro(fred: Any) -> bool:
    """Return true when FRED macro has observed inputs suitable for IC staging."""
    if not isinstance(fred, dict) or fred.get("confidence") is None:
        return False
    indicators = fred.get("indicators")
    if not isinstance(indicators, dict) or not indicators:
        return False
    if fred.get("indicators_observed") is False:
        return False
    unavailable_values = {"unavailable", "missing", "empty", "failed", "degraded", "fallback"}
    fallback_modes = {"unavailable", "synthetic", "last_good", "fallback"}
    source_mode = str(fred.get("source_mode", "")).lower()
    status = str(fred.get("status", "")).lower()
    cache_status = str(fred.get("cache_status", "")).lower()
    if source_mode in fallback_modes:
        return False
    if status in unavailable_values:
        return False
    return cache_status not in unavailable_values


def _source_manifest_row_for(public_dir: Path, artifact_name: str) -> dict[str, Any] | None:
    """Return the compact source-manifest row for a public data artifact."""
    manifest_path = public_dir / "source_manifest.json"
    try:
        with open(manifest_path) as f:
            manifest = json.load(f)
    except (OSError, json.JSONDecodeError, TypeError):
        return None

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        return None

    for row in artifacts:
        if not isinstance(row, dict):
            continue
        candidates = {
            row.get("artifact"),
            row.get("filename"),
            row.get("path"),
        }
        if artifact_name in candidates:
            return row
    return None


def _yield_source_provenance(public_dir: Path) -> dict[str, Any]:
    """Map yields source-manifest metadata into the yield curve payload."""
    row = _source_manifest_row_for(public_dir, "yields.json")
    if row is None:
        return {}
    return {
        "source_mode": row.get("source_mode"),
        "source_status": row.get("status"),
        "source_reason": row.get("failure_reason") or row.get("reason"),
        "source_provider": row.get("provider"),
        "source_generated_at": row.get("generated_at"),
        "source_latest_observation": row.get("latest_observation"),
    }


def _first_known_value(*values: Any, default: str = "unknown") -> Any:
    """Return the first non-empty metadata value, treating 'unknown' as absent."""
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and value.lower() in {"", "unknown"}:
            continue
        return value
    return default

# Common exception types caught when signal generators fail.
# ValueError/TypeError indicate likely bugs — callers should log these at
# error level. ImportError/AttributeError indicate missing deps — warning.
SIGNAL_EXCEPTIONS = (
    ImportError, AttributeError, KeyError,
    ValueError, TypeError, RuntimeError, OSError,
)

# Lighter exception tuple for monitoring/utility modules that don't
# touch external data structures (no AttributeError/KeyError risk).
MONITOR_EXCEPTIONS = (ImportError, ValueError, OSError, RuntimeError)

# Exceptions that indicate likely bugs rather than missing dependencies.
_BUG_EXCEPTIONS = (ValueError, TypeError)


def _attach_signal_metadata(output: Dict, *, generated_at: str | None = None) -> Dict:
    """Attach dashboard-level generation timestamps to a signals payload."""
    timestamp = generated_at or datetime.now().isoformat()
    enriched = dict(output)
    enriched["generated_at"] = timestamp
    enriched.setdefault("timestamp", timestamp)
    return enriched


def _generator_git_sha_short() -> str | None:
    """Short HEAD for operator lag detection (code vs projected artifact)."""
    try:
        from src.monitor.decision_registry import _git_sha_short

        return _git_sha_short()
    except Exception:
        return None


def _finalize_signal_metadata(output: Dict, *, finalized_at: str | None = None) -> Dict:
    """Stamp final artifact metadata after all signal sections are assembled."""
    timestamp = finalized_at or datetime.now(timezone.utc).isoformat()
    finalized = dict(output)
    finalized["generated_at"] = timestamp
    finalized["timestamp"] = timestamp
    sha = _generator_git_sha_short()
    if sha:
        finalized["generator_git_sha"] = sha
    return finalized


def _dist_data_dir_for_public_dir(public_dir: Path) -> Path:
    """Return the app dist/data directory that mirrors public/data."""
    if public_dir.name == "data" and public_dir.parent.name == "public":
        app_root = public_dir.parent.parent
    else:
        app_root = public_dir.parent
    return app_root / "dist" / "data"


def _mirror_public_data_contract_files_to_dist(public_dir: Path) -> None:
    """Mirror deploy-checked public data files after final generation."""
    dist_data = _dist_data_dir_for_public_dir(public_dir)
    dist_data.mkdir(parents=True, exist_ok=True)
    for filename in PUBLIC_DATA_DIST_MIRROR_FILES:
        source = public_dir / filename
        if source.exists():
            shutil.copyfile(source, dist_data / filename)


def _compact_health_summary(report: Dict) -> Dict:
    """Build a bounded health summary suitable for embedding in signals.json."""
    if not isinstance(report, dict):
        return {"status": "error", "error": "invalid health report"}

    status = report.get("system_status") or report.get("status") or "unknown"
    summary = {"status": status}

    if report.get("generated_at"):
        summary["generated_at"] = report.get("generated_at")
    if report.get("error"):
        summary["error"] = str(report.get("error"))

    cron_jobs = report.get("cron_jobs")
    if isinstance(cron_jobs, list):
        from src.monitor.hermes_cron import rollup_failed_cron_jobs

        summary["cron_job_count"] = len(cron_jobs)
        summary["failed_cron_jobs"] = len(rollup_failed_cron_jobs(cron_jobs))

    data_freshness = report.get("data_freshness")
    if isinstance(data_freshness, dict):
        summary["stale_data_count"] = sum(
            1
            for item in data_freshness.values()
            if isinstance(item, dict) and item.get("status") != "fresh"
        )

    scheduler_status = report.get("scheduler_status")
    if isinstance(scheduler_status, dict) and scheduler_status.get("status"):
        summary["scheduler_status"] = scheduler_status.get("status")

    data_pipeline_slo = report.get("data_pipeline_slo")
    if isinstance(data_pipeline_slo, dict):
        if data_pipeline_slo.get("status"):
            summary["data_pipeline_slo_status"] = data_pipeline_slo.get("status")
        if data_pipeline_slo.get("top_dimension"):
            summary["top_slo_dimension"] = data_pipeline_slo.get("top_dimension")
        runbook = data_pipeline_slo.get("runbook")
        if isinstance(runbook, dict):
            top_cause = runbook.get("top_cause")
            if isinstance(top_cause, dict) and top_cause.get("code"):
                summary["top_slo_cause_code"] = top_cause.get("code")

    # Project kill/incident halt into compact summary so signals.health
    # matches signals.broker.kill_switch without requiring broker-only reads.
    summary.update(project_compact_kill_fields(report))

    return summary


def _apply_kill_to_smart_rebalance(
    smart: dict[str, Any] | None,
    kill_payload: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Annotate smart_rebalance with kill halt and force non-execute under kill.

    Keeps drift/VPIN diagnostics visible but never implies actionable execute
    when authority kill_switch.json is enabled (same gate as order_router).
    """
    if not isinstance(smart, dict):
        return smart
    try:
        from src.dashboard.kill_authority import is_kill_execution_blocked
    except ImportError:
        is_kill_execution_blocked = lambda p: bool(isinstance(p, dict) and p.get("enabled"))

    if not is_kill_execution_blocked(kill_payload):
        # Explicit clear fields when kill is off (stable schema for consumers)
        smart.setdefault("execution_blocked", False)
        smart.setdefault("kill_switch_enabled", False)
        return smart

    level = None
    reason = None
    incident_id = None
    message = None
    if isinstance(kill_payload, dict):
        level = kill_payload.get("level")
        reason = kill_payload.get("reason")
        incident_id = kill_payload.get("incident_id")
        message = kill_payload.get("message")

    smart["execution_blocked"] = True
    smart["kill_switch_enabled"] = True
    if level is not None:
        smart["kill_switch_level"] = level
    if reason is not None:
        smart["kill_switch_reason"] = reason
    if incident_id is not None:
        smart["kill_switch_incident_id"] = incident_id
    if message is not None:
        smart["kill_switch_message"] = message

    # Force non-actionable decision; preserve original decision for operators
    prior_decision = smart.get("decision")
    smart["should_execute"] = False
    smart["decision"] = "blocked_kill_switch"
    human = message if isinstance(message, str) and message.strip() else reason
    smart["reason"] = (
        f"blocked_by_kill_switch:{level or 'enabled'}"
        + (f" ({human})" if human else "")
        + (f"; prior={prior_decision}" if prior_decision else "")
    )
    return smart


def _remaining_budget_ratio(metadata: dict[str, Any], status: dict[str, Any]) -> float:
    """Return remaining rebalance budget as a fraction of portfolio value."""
    value = metadata.get("remaining_budget_ratio")
    if value is None:
        value = metadata.get("remaining_budget_pct")
    if value is None:
        value = status.get("remaining_budget_ratio")
    if value is None and status.get("remaining_budget_pct") is not None:
        value = status.get("remaining_budget_pct") / 100
    if value is None:
        return 1.0
    return round(float(value), 6)


def _remaining_budget_display_pct(ratio: float, status: dict[str, Any]) -> float:
    """Return remaining rebalance budget in display percent units."""
    status_pct = status.get("remaining_budget_pct")
    if status_pct is not None:
        return float(status_pct)
    return round(ratio * 100, 3)


def _load_canonical_health_report() -> dict[str, Any] | None:
    """Load canonical health.json when already published."""
    health_path = PUBLIC_DIR / "health.json"
    try:
        with health_path.open(encoding="utf-8") as handle:
            report = json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return report if isinstance(report, dict) else None


def _log_signal_error(signal_name: str, exc: Exception) -> None:
    """Log a signal exception at the appropriate level.

    ValueError/TypeError are likely code bugs → logger.error.
    ImportError/AttributeError/KeyError are missing deps → logger.warning.
    RuntimeError/OSError are environmental → logger.warning.
    """
    if isinstance(exc, _BUG_EXCEPTIONS):
        logger.error("Signal %s failed with likely bug: %s", signal_name, exc)
    else:
        logger.warning("Signal %s not available: %s", signal_name, exc)


PUBLIC_DIR = PUBLIC_DATA_DIR
DB_PATH = MARKET_DB
_DEFAULT_DATA_DIR = DATA_DIR


def _resolve_hermes_cron_jobs_path() -> Optional[Path]:
    """Return the Hermes cron jobs file to probe, if this run should probe it."""
    return resolve_hermes_cron_jobs_path(
        current_data_dir=DATA_DIR,
        default_data_dir=_DEFAULT_DATA_DIR,
    )


class DashboardGenerator:
    # SPC monitor instance (class-level to persist across runs)
    _spc_monitor = None

    def __init__(self):
        PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite_connect(DB_PATH)
        self.conn.row_factory = sqlite3.Row

    @staticmethod
    def _deduplicate_performance_entries_by_date(
        entries: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Keep the last performance entry for each calendar date."""
        daily_map: Dict[str, Dict[str, Any]] = {}
        for idx, entry in enumerate(entries):
            ts = entry.get("timestamp", "")
            date_key = ts[:10] if len(ts) >= 10 else ""
            if not date_key:
                # Preserve legacy behavior: timestamp-less rows are distinct.
                date_key = f"__no_ts_{idx}__"
            daily_map[date_key] = entry
        return [daily_map[d] for d in sorted(daily_map)]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def close(self):
        """Close the SQLite connection if open."""
        if self.conn is not None:
            try:
                self.conn.close()
            except (OSError, sqlite3.Error) as e:
                logger.warning("Error closing SQLite connection: %s", e)
            self.conn = None
    
    def generate_performance_json(self) -> Path:
        """Generate performance history for dashboard charts."""
        cursor = self.conn.cursor()
        
        # Get portfolio history
        cursor.execute("""
            SELECT symbol, date, close FROM prices 
            WHERE symbol IN ('SPY', 'GLD', 'TLT', 'QQQ')
            AND date >= date('now', '-365 days')
            ORDER BY date
        """)
        
        prices = {}
        for row in cursor.fetchall():
            sym = row[0]
            if sym not in prices:
                prices[sym] = []
            prices[sym].append({"d": row[1], "p": row[2]})
        
        # Get regime history
        cursor.execute("""
            SELECT date, regime, vix_level FROM regime_log
            WHERE date >= date('now', '-90 days')
            ORDER BY detected_at
        """)
        
        regimes = [{"d": row[0], "r": row[1], "v": row[2]} for row in cursor.fetchall()]
        
        # Get paper portfolio performance (from JSONL log — tail read only)
        perf_log = DATA_DIR / "performance.jsonl"
        paper_perf = []
        if perf_log.exists():
            raw_entries = []
            with open(perf_log) as f:
                for line in deque(f, maxlen=500):
                    try:
                        raw_entries.append(json.loads(line))
                    except (OSError, json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
                        logger.warning("Failed to parse performance log entry: %s", e)
            paper_perf = [
                {
                    "t": entry.get("timestamp", "")[:10],
                    "v": entry.get("total_value", 0),
                    "r": entry.get("daily_return", 0)
                }
                for entry in self._deduplicate_performance_entries_by_date(raw_entries)
            ]

        output = {
            "prices": prices,
            "regimes": regimes,
            "paper_portfolio": paper_perf,
            "generated_at": datetime.now().isoformat()
        }

        out_path = PUBLIC_DIR / "dashboard.json"
        save_results_json(output, output_path=str(out_path))

        return out_path

    def _get_overlay_data(self) -> Dict:
        """Merge overlay dashboard data (collar, crypto, calendar, kurtosis, etc.)

        Pulls data from the OverlayDashboardGenerator and maps keys to the
        format expected by LiveDashboard.tsx panels.
        """
        result: Dict = {}

        # Primary overlay data
        try:
            from src.dashboard.overlay_dashboard import OverlayDashboardGenerator
            gen = OverlayDashboardGenerator()
            dashboard = gen.generate()
            data = dashboard.to_dict()

            result.update({
                "collar": data.get("collar", {}),
                "crypto": data.get("crypto", {}),
                "calendar": data.get("calendar", {}),
                "kurtosis": data.get("kurtosis", {}),
                "bond_momentum": data.get("bond_duration", {}),
                "unified": data.get("unified", {}),
            })
        except SIGNAL_EXCEPTIONS as e:
            _log_signal_error("overlay_dashboard", e)

        # VIX term structure
        try:
            from src.signals.vix_term_structure import VIXTermStructureSignalGenerator
            vix_gen = VIXTermStructureSignalGenerator()
            signal = vix_gen.generate_signal()
            result["vix_term_structure"] = signal.to_dict()
            # VIX overlay state
            state_file = DATA_DIR / "vix_overlay_state.json"
            if state_file.exists():
                with open(state_file) as f:
                    result["vix_overlay"] = json.load(f)
        except SIGNAL_EXCEPTIONS as e:
            _log_signal_error("vix_term_structure", e)

        return result

    def _record_ic_data(self, output: Dict) -> None:
        """Record signal predictions for IC decay monitoring.

        Two-phase lifecycle:
        1. Resolve: pair previously staged predictions with the forward
           return that materialized since they were staged.
        2. Stage: store current predictions for resolution next run.

        Uses SPY return as the forward return for all signals (SPY is the
        primary portfolio driver). Saves monitor state to disk so IC data
        survives across cron runs.
        """
        try:
            from src.monitor.ic_decay_monitor import ICMonitor

            monitor = ICMonitor()
            monitor.load_state()
            cursor = self.conn.cursor()
            cursor.execute(
                "SELECT date, close FROM prices WHERE symbol = 'SPY' "
                "ORDER BY date DESC LIMIT 1",
            )
            latest_spy_row = cursor.fetchone()
            latest_spy_date = latest_spy_row[0] if latest_spy_row else None

            # Phase 1: Resolve previously staged predictions
            if monitor.has_staged_predictions():
                staged_date = monitor.get_staged_date()
                if staged_date:
                    # Compute SPY forward return from staged date to latest
                    cursor.execute(
                        "SELECT date, close FROM prices WHERE symbol = 'SPY' "
                        "AND date >= ? ORDER BY date ASC LIMIT 1",
                        (staged_date,),
                    )
                    start_row = cursor.fetchone()
                    end_row = latest_spy_row
                    if start_row and end_row and start_row[0] != end_row[0]:
                        start_price = float(start_row[1])
                        end_price = float(end_row[1])
                        if start_price > 0:
                            forward_return = (end_price / start_price) - 1.0
                            n_resolved = monitor.resolve_staged(forward_return)
                            logger.info(
                                "IC decay: resolved %d staged predictions "
                                "(%s → %s, forward return=%.4f%%)",
                                n_resolved, staged_date, end_row[0],
                                forward_return * 100,
                            )

            # Phase 2: Stage current predictions for next run
            predictions: Dict[str, float] = {}

            # Ensemble voter biases
            ensemble = output.get("ensemble_voting")
            if isinstance(ensemble, dict):
                if "equity_bias" in ensemble and ensemble["equity_bias"] is not None:
                    predictions["ensemble_equity"] = float(ensemble["equity_bias"])
                if "gold_bias" in ensemble and ensemble["gold_bias"] is not None:
                    predictions["ensemble_gold"] = float(ensemble["gold_bias"])
                if "duration_bias" in ensemble and ensemble["duration_bias"] is not None:
                    predictions["ensemble_duration"] = float(ensemble["duration_bias"])
                if "weighted_consensus" in ensemble and ensemble["weighted_consensus"] is not None:
                    predictions["ensemble_consensus"] = float(ensemble["weighted_consensus"])

            # Alternative data composite score
            alt = output.get("alternative_data")
            if isinstance(alt, dict) and alt.get("composite_score") is not None:
                predictions["alternative_data"] = float(alt["composite_score"])

            # Behavioral sentiment composite
            beh = output.get("behavioral_sentiment")
            if isinstance(beh, dict) and beh.get("composite_score") is not None:
                predictions["behavioral_sentiment"] = float(beh["composite_score"])

            # Factor rotation signal strength
            fr = output.get("factor_rotation")
            if isinstance(fr, dict) and fr.get("signal_strength") is not None:
                predictions["factor_rotation"] = float(fr["signal_strength"])

            # FRED-MD macro confidence
            fred = output.get("fred_macro")
            if _is_predictive_fred_macro(fred):
                predictions["fred_macro"] = float(fred["confidence"])

            if predictions and latest_spy_date and not monitor.has_staged_predictions():
                monitor.stage_predictions(
                    predictions,
                    str(latest_spy_date),
                )

            monitor.save_state()
        except MONITOR_EXCEPTIONS as e:
            _log_signal_error("ic_decay_record", e)

    def _generate_two_stage_regime(self) -> Optional[Dict]:
        """Generate two-stage k-means macro regime signal.

        Uses Oliveira et al. 2025 two-layer k-means (L2 crisis detection +
        cosine directional clustering) on FRED-MD data. Falls back gracefully
        if FRED-MD data or fredapi is not available.

        Returns:
            Dict with regime, confidence, probabilities, or None if unavailable.
        """
        try:
            from src.regime.fred_md_two_stage_kmeans import TwoStageKMeansRegime
            from src.data.fred_data import FredMdFetcher
        except ImportError:
            return None

        # Try to load FRED-MD data via the existing pipeline
        try:
            fetcher = FredMdFetcher()
            df = fetcher.get_all_series(cache_ok=True)
        except (ValueError, ImportError) as e:
            logger.info("FRED-MD data not available for two-stage k-means: %s", e)
            return None

        if df is None or df.empty or len(df.columns) < 10:
            logger.info("Insufficient FRED-MD series for two-stage k-means")
            return None

        # Build data matrix: rows=dates, cols=series
        # Forward-fill missing values, drop rows with too many NaNs
        df_filled = df.ffill().dropna(thresh=max(10, len(df.columns) // 2))
        if len(df_filled) < 24:
            logger.info("Too few FRED-MD observations for two-stage k-means")
            return None

        # Standardize: z-score each series
        X_raw = df_filled.values.astype(np.float64)
        X_std = (X_raw - np.nanmean(X_raw, axis=0)) / np.nanstd(X_raw, axis=0)
        X = np.nan_to_num(X_std, nan=0.0)

        # Fit two-stage k-means
        classifier = TwoStageKMeansRegime(random_state=42, max_pca_components=15)
        classifier.fit(X)

        signal = classifier.get_signal(latest_index=-1)

        return {
            "regime": signal["regime"],
            "confidence": signal["confidence"],
            "crisis_probability": signal.get("crisis_probability", 0.0),
            "probabilities": signal.get("probabilities", {}),
            "n_pca_components": signal.get("n_pca_components", 0),
            "variance_retained": signal.get("variance_retained", 0.0),
            "n_observations": len(df_filled),
            "n_series": len(df_filled.columns),
            "method": "oliveira_2025_two_stage_kmeans",
            "timestamp": datetime.now().isoformat(),
        }

    def _generate_bocd_regime(self) -> Optional[Dict]:
        """Generate BOCD (Bayesian Online Changepoint Detection) regime signal.

        Uses Adams & MacKay (2007) for real-time structural break detection
        in daily return series without fixed observation windows.

        Returns:
            Dict with regime, regime_change_prob, changepoint_count, etc.
            None if insufficient data.
        """
        try:
            from src.regime.bocd_detector import BOCDDetector
        except ImportError:
            return None

        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT date, close FROM prices
            WHERE symbol = 'SPY'
            ORDER BY date ASC
        """)
        rows = cursor.fetchall()
        if len(rows) < 2:
            return None

        prices = np.array([row[1] for row in rows], dtype=float)
        returns = np.diff(np.log(prices))

        detector = BOCDDetector(hazard_rate=1.0 / 252, threshold=0.5, min_run_length=5)
        detector.fit(returns)

        signal = detector.get_signal()
        bocd_data = signal["bocd_detector"]
        bocd_data["timestamp"] = datetime.now().isoformat()

        return bocd_data

    def _load_signal_generation_context(self) -> Dict[str, Any]:
        """Load DB, portfolio, and regime context for signals.json."""
        cursor = self.conn.cursor()

        # Get latest VIX level directly from prices table
        cursor.execute("""
            SELECT close FROM prices 
            WHERE symbol = '^VIX' 
            ORDER BY date DESC LIMIT 1
        """)
        vix_row = cursor.fetchone()
        vix_level = vix_row[0] if vix_row else None
        
        # Try to get trend signal from regime_log
        cursor.execute("""
            SELECT regime, detected_at FROM regime_log
            ORDER BY detected_at DESC LIMIT 1
        """)
        trend_row = cursor.fetchone()
        trend_regime = trend_row[0] if trend_row else "normal"
        trend_detected = trend_row[1] if trend_row else None

        # VIX-based regime detection (shared logic with evaluator.py)
        current_regime = classify_vix_regime(vix_level, trend_regime)

        regime_data = {
            "regime": current_regime,
            "vix": vix_level,
            "detected": trend_detected
        }
        
        # Latest prices
        cursor.execute("""
            SELECT symbol, close FROM prices 
            WHERE (symbol, date) IN (
                SELECT symbol, MAX(date) FROM prices GROUP BY symbol
            )
        """)
        latest = {row[0]: row[1] for row in cursor.fetchall()}
        
        # Current paper portfolio state
        portfolio_state = DATA_DIR / "portfolio_paper.json"
        positions = []
        if portfolio_state.exists():
            with open(portfolio_state) as f:
                state = json.load(f)
                for sym, pos in state.get("positions", {}).items():
                    positions.append({
                        "symbol": sym,
                        "shares": pos.get("shares", 0),
                        "value": pos.get("value", 0),
                        "weight": pos.get("weight", 0),
                        "unrealized": pos.get("unrealized_pnl", 0)
                    })
                total_value = state.get("cash", 0) + sum(p["value"] for p in positions)
                cash = state.get("cash", 0)
        else:
            total_value = 100000  # Initial
            cash = 100000
        
        # Target allocation based on regime
        if os.environ.get("REGIME_ALLOC_ENABLED", "0") == "1":
            target_alloc = get_regime_allocation_with_override(current_regime)
        else:
            target_alloc = REGIME_OVERRIDES.get(current_regime) or BASE_ALLOCATION
        
        # Pending orders (tail read only)
        orders = []
        orders_log = DATA_DIR / "orders.jsonl"
        if orders_log.exists():
            with open(orders_log) as f:
                for line in deque(f, maxlen=5):
                    try:
                        order = json.loads(line)
                        orders.append({
                            "sym": order.get("symbol"),
                            "side": order.get("side"),
                            "shares": round(order.get("shares", 0), 2),
                            "value": round(order.get("fill_value", 0), 2)
                        })
                    except (OSError, json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
                        logger.warning("Failed to parse order log entry: %s", e)

        return {
            "cursor": cursor,
            "vix_level": vix_level,
            "trend_regime": trend_regime,
            "trend_detected": trend_detected,
            "current_regime": current_regime,
            "regime_data": regime_data,
            "latest": latest,
            "positions": positions,
            "cash": cash,
            "total_value": total_value,
            "target_alloc": target_alloc,
            "orders": orders,
        }

    @staticmethod
    def _build_ensemble_source_breakdown(source_votes: List[Any]) -> List[Dict[str, Any]]:
        """Serialize ensemble source readings for downstream postprocessors."""
        source_breakdown = []
        for src in source_votes:
            value = float(src.value)
            source_breakdown.append({
                "source": src.source.value if hasattr(src.source, 'value') else str(src.source),
                "value": round(value, 4),
                "direction": "bullish" if value > 0 else ("bearish" if value < 0 else "neutral"),
                "strength": round(abs(value), 3),
                "confidence": round(src.confidence, 3),
                "weight": round(src.weight, 3),
            })
        return source_breakdown

    @staticmethod
    def _build_ensemble_source_count_metadata(
        regime: Any,
        source_breakdown: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Describe configured, collected, and positive-weight ensemble sources."""
        configured_sources = []
        try:
            from src.strategy.ensemble_voter import REGIME_WEIGHTS, Regime, SignalSource

            regime_key = regime if isinstance(regime, Regime) else Regime(str(regime).lower())
            configured_sources = [
                source.value if hasattr(source, "value") else str(source)
                for source in REGIME_WEIGHTS.get(regime_key, {})
            ] or [source.value for source in SignalSource]
        except (ImportError, AttributeError, KeyError, TypeError, ValueError):
            configured_sources = []

        inactive_sources = []
        contributing_count = 0
        for source in source_breakdown:
            try:
                weight = float(source.get("weight", 0.0))
            except (TypeError, ValueError):
                weight = 0.0
            source_name = str(source.get("source", "unknown"))
            if np.isfinite(weight) and weight > 0:
                contributing_count += 1
            else:
                inactive_sources.append(source_name)

        collected_count = len(source_breakdown)
        configured_count = len(set(configured_sources)) if configured_sources else collected_count
        return {
            "num_sources": collected_count,
            "configured_source_count": configured_count,
            "collected_source_count": collected_count,
            "contributing_source_count": contributing_count,
            "inactive_source_count": len(inactive_sources),
            "inactive_sources": inactive_sources,
        }

    @staticmethod
    def _get_configured_ensemble_source_weights(regime: Any) -> Dict[str, float]:
        """Return configured ensemble source weights for the active regime."""
        try:
            from src.strategy.ensemble_voter import REGIME_WEIGHTS, Regime

            regime_key = regime if isinstance(regime, Regime) else Regime(str(regime).lower())
            weights_file = Path(os.environ.get("ENSEMBLE_WEIGHTS_FILE", str(DATA_DIR / "ensemble_weights.json")))
            if weights_file.exists():
                try:
                    with open(weights_file) as f:
                        configured = json.load(f)
                    regime_weights = configured.get(regime_key.value)
                    if isinstance(regime_weights, dict):
                        return {
                            str(source): float(weight)
                            for source, weight in regime_weights.items()
                        }
                except (OSError, json.JSONDecodeError, TypeError, ValueError):
                    pass
            return {
                source.value if hasattr(source, "value") else str(source): float(weight)
                for source, weight in REGIME_WEIGHTS.get(regime_key, {}).items()
            }
        except (ImportError, AttributeError, KeyError, TypeError, ValueError):
            return {}

    @staticmethod
    def _format_ensemble_source_label(source: str) -> str:
        """Format a source identifier for operator-facing source disclosure."""
        return source.replace("_", " ").title()

    @staticmethod
    def _google_trends_inactive_disclosure() -> tuple[str, str]:
        """Inspect Google Trends directly when it is configured but not collected."""
        try:
            from src.signals.google_trends_signal import GoogleTrendsSignal

            snapshot = GoogleTrendsSignal().get_signal_snapshot()
            if snapshot.is_active:
                return "missing", "Configured Google Trends did not appear in ensemble source rows."

            reason = snapshot.metadata.get("inactive_reason") if isinstance(snapshot.metadata, dict) else None
            if not reason:
                reason = snapshot.explanation.replace("Google Trends:", "", 1).strip()
            category = snapshot.metadata.get("inactive_category") if isinstance(snapshot.metadata, dict) else None
            status = str(category or "inactive")
            return status, str(reason or "Google Trends source is inactive.")
        except (ImportError, AttributeError, KeyError, ValueError, TypeError, OSError, RuntimeError) as e:
            return "unavailable", f"Google Trends status unavailable: {e}"

    @staticmethod
    def _build_configured_source_status(
        regime: Any,
        source_breakdown: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Explain configured source state, including missing stale configured sources."""
        configured_weights = DashboardGenerator._get_configured_ensemble_source_weights(regime)
        if not configured_weights:
            return []

        rows_by_source = {
            str(row.get("source", "")): row
            for row in source_breakdown
            if isinstance(row, dict) and row.get("source")
        }
        statuses: List[Dict[str, Any]] = []

        for source, configured_weight in configured_weights.items():
            row = rows_by_source.get(source)
            collected = row is not None
            if row is not None:
                try:
                    row_weight = float(row.get("weight", 0.0))
                except (TypeError, ValueError):
                    row_weight = 0.0
                contributing = bool(np.isfinite(row_weight) and row_weight > 0)
                status = "active" if contributing else "zero_weight"
                reason = (
                    "Collected and contributing to the ensemble vote."
                    if contributing
                    else "Collected but assigned zero effective weight."
                )
            else:
                contributing = False
                status = "missing"
                reason = "Configured source did not produce an active ensemble reading."
                if source == "google_trends":
                    status, reason = DashboardGenerator._google_trends_inactive_disclosure()

            statuses.append({
                "source": source,
                "label": DashboardGenerator._format_ensemble_source_label(source),
                "configured": True,
                "configured_weight": round(configured_weight, 5),
                "collected": collected,
                "active": collected and contributing,
                "contributing": contributing,
                "status": status,
                "reason": reason,
            })

        return statuses

    @staticmethod
    def _build_ensemble_adaptive_learning_disclosure(ensemble_result: Any) -> Dict[str, Any]:
        """Serialize adaptive-learning branch status from an ensemble vote."""
        disclosure = getattr(ensemble_result, "adaptive_learning", {})
        return disclosure if isinstance(disclosure, dict) else {}

    @staticmethod
    def _build_allocation_surface_roles(data_dir: Path | None = None) -> Dict[str, Any]:
        """Describe the current live-routing role of allocation-like signals surfaces.

        When the kill switch is enabled, target_allocations remains the routing
        surface but is disclosed as execution-blocked (not live_authoritative).
        """
        advisory_description = (
            "Published for advisory diagnostics; current order routing uses "
            "target_allocations."
        )
        roles: Dict[str, Any] = {
            "schema_version": "allocation-surface-roles/v1",
            "routed_surface": "target_allocations",
            "routed_by": "src.broker.order_router",
            "surfaces": {
                "target_allocations": {
                    "label": "Target Allocation",
                    "role": "execution_routed",
                    "routed": True,
                    "routed_by": "src.broker.order_router",
                    "live_authoritative": True,
                    "canonical_controller": "signals.json.target_allocations",
                    "description": (
                        "Current order-routing input consumed by src.broker.order_router."
                    ),
                },
                "ensemble_voting": {
                    "label": "Ensemble Voting",
                    "role": "advisory_non_routed",
                    "routed": False,
                    "routed_by": None,
                    "live_authoritative": False,
                    "canonical_controller": "signals.json.target_allocations",
                    "description": advisory_description,
                },
                "adaptive_sizing": {
                    "label": "Adaptive Sizing",
                    "role": "advisory_non_routed",
                    "routed": False,
                    "routed_by": None,
                    "live_authoritative": False,
                    "canonical_controller": "signals.json.target_allocations",
                    "description": advisory_description,
                },
                "black_litterman": {
                    "label": "Black-Litterman",
                    "role": "advisory_non_routed",
                    "routed": False,
                    "routed_by": None,
                    "live_authoritative": False,
                    "canonical_controller": "signals.json.target_allocations",
                    "description": advisory_description,
                },
            },
        }
        root = Path(data_dir) if data_dir is not None else DATA_DIR
        kill = project_kill_switch_fields(load_kill_switch_payload(root))
        if kill.get("enabled"):
            roles = allocation_roles_under_kill(
                roles,
                kill_enabled=True,
                kill_level=kill.get("level"),
            )
        return roles

    @staticmethod
    def _build_advisory_allocation_artifact_role(
        surface: str,
        allocation_field: str,
    ) -> Dict[str, Any]:
        """Describe a standalone allocation artifact as advisory/non-routed."""
        return {
            "schema_version": "allocation-artifact-role/v1",
            "surface": surface,
            "allocation_field": allocation_field,
            "runtime_role": "advisory_non_routed",
            "live_authoritative": False,
            "routed": False,
            "routed_by": None,
            "canonical_controller": "signals.json.target_allocations",
            "routed_surface": "target_allocations",
            "routed_surface_path": "public/data/signals.json#target_allocations",
            "description": (
                f"{surface} is published for advisory diagnostics; live order routing "
                "continues to consume signals.json.target_allocations."
            ),
        }

    @staticmethod
    def _flatten_advisory_authority(authority: Dict[str, Any]) -> Dict[str, Any]:
        """Top-level authority fields for operator greps (vixy_hedge pattern).

        Nested ``authority`` remains the schema contract for AdaptiveSizingPanel;
        top-level mirrors make ``runtime_role`` / ``routed`` visible without
        digging into the nested block.
        """
        return {
            "runtime_role": authority.get("runtime_role"),
            "live_authoritative": authority.get("live_authoritative"),
            "routed": authority.get("routed"),
            "routed_by": authority.get("routed_by"),
            "canonical_controller": authority.get("canonical_controller"),
            "routed_surface": authority.get("routed_surface"),
        }

    @staticmethod
    def _canonicalize_public_weights(
        weights: Dict[str, Any],
        canonical_assets: tuple[str, ...] = ("SPY", "GLD", "TLT"),
    ) -> Dict[str, Any]:
        """Uppercase public weight keys and preserve zero-weight diagnostics."""
        normalized: Dict[str, float] = {}
        excluded_assets: list[str] = []
        for symbol, raw_weight in (weights or {}).items():
            canonical = str(symbol).upper()
            try:
                normalized[canonical] = float(raw_weight)
            except (TypeError, ValueError):
                excluded_assets.append(canonical)

        public_weights = {
            symbol: normalized.get(symbol, 0.0)
            for symbol in canonical_assets
        }
        zero_weight_assets = [
            symbol for symbol, weight in public_weights.items()
            if abs(weight) < 1e-12
        ]

        return {
            "weights": public_weights,
            "excluded_assets": sorted(set(excluded_assets)),
            "zero_weight_assets": zero_weight_assets,
        }

    @staticmethod
    def _build_regime_authority(
        current_regime: str,
        target_alloc: Dict[str, float],
    ) -> Dict[str, Any]:
        """Document the live regime controller and advisory role of advanced regimes."""
        return {
            "schema_version": "regime-authority/v1",
            "live_controller": "classify_vix_regime",
            "live_controller_module": "src.utils.classify_vix_regime",
            "live_regime": current_regime,
            "allocation_regime": normalize_allocation_regime(current_regime) or "normal",
            "routed_surface": "target_allocations",
            "target_allocations": target_alloc,
            "advanced_regime_signals": {
                "two_stage_regime": {
                    "role": "advisory_shadow",
                    "routed": False,
                    "availability": "unknown",
                    "published": False,
                    "description": "Availability pending staleness check; not live order-routing authority.",
                },
                "bocd_regime": {
                    "role": "advisory_shadow",
                    "routed": False,
                    "availability": "unknown",
                    "published": False,
                    "description": "Availability pending staleness check; not live order-routing authority.",
                },
                "regime_transition": {
                    "role": "advisory_shadow",
                    "routed": False,
                    "availability": "unknown",
                    "published": False,
                    "description": "Availability pending staleness check; not live order-routing authority.",
                },
            },
        }

    @staticmethod
    def _dedupe_preserve_order(values: list[str]) -> list[str]:
        """Return unique string identifiers while preserving first occurrence order."""
        seen: set[str] = set()
        result: list[str] = []
        for value in values:
            key = str(value)
            if key in seen:
                continue
            seen.add(key)
            result.append(key)
        return result

    @classmethod
    def _update_regime_authority_availability(cls, output: Dict[str, Any]) -> None:
        """Update advanced regime authority entries with observed snapshot state."""
        authority = output.get("regime_authority")
        if not isinstance(authority, dict):
            return

        advanced = authority.get("advanced_regime_signals")
        if not isinstance(advanced, dict):
            return

        staleness = output.get("staleness") if isinstance(output.get("staleness"), dict) else {}
        unavailable = set(staleness.get("unavailable_signals") or [])
        stale = set(staleness.get("stale_signals") or [])

        for signal_name, entry in advanced.items():
            if not isinstance(entry, dict):
                continue

            signal_block = output.get(signal_name)
            if signal_name in unavailable or signal_block is None:
                availability = "unavailable"
                published = False
                description = "Unavailable in this snapshot; not live order-routing authority."
            elif signal_name in stale:
                availability = "stale"
                published = False
                description = "Stale in this snapshot; not live order-routing authority."
            elif cls._is_unavailable_signal_block(signal_block):
                availability = "error"
                published = False
                description = "Error or degraded placeholder in this snapshot; not live order-routing authority."
            else:
                availability = "present"
                published = True
                description = "Published for advisory diagnostics; not live order-routing authority."

            entry.update({
                "availability": availability,
                "published": published,
                "description": description,
            })

    def _build_base_signal_sections(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Build the core signal sections before dashboard-level metadata."""
        vix_level = context["vix_level"]
        trend_regime = context["trend_regime"]
        current_regime = context["current_regime"]
        regime_data = context["regime_data"]
        latest = context["latest"]
        positions = context["positions"]
        cash = context["cash"]
        total_value = context["total_value"]
        target_alloc = context["target_alloc"]
        orders = context["orders"]

        # Add factor rotation signals if engine available
        factor_rotation_signal = None
        factor_rotation_result = None
        try:
            from src.strategy.factor_rotation import FactorMomentumEngine
            engine = FactorMomentumEngine()
            factor_rotation_result = engine.evaluate()
            if factor_rotation_result and "error" not in factor_rotation_result:
                now_ts = datetime.now(timezone.utc).isoformat()
                factor_rotation_signal = {
                    "selected_factors": factor_rotation_result.get("selected_factors", []),
                    "allocation": factor_rotation_result.get("allocation", {}),
                    "signal_strength": factor_rotation_result.get("signal_strength", 0.0),
                    "recommendation": factor_rotation_result.get("recommendation", {}),
                    # Staleness TTL requires generated_at; missing field → optional unavailable.
                    "generated_at": now_ts,
                    "timestamp": now_ts,
                }
        except SIGNAL_EXCEPTIONS as e:
            _log_signal_error("factor_rotation", e)

        # Add yield curve data from yields.json
        yield_curve_data = self._get_yield_curve_data()
        
        # Add volatility parity / convexity harvest signals
        convexity_signal = None
        vol_parity_signal = None
        try:
            from src.strategy.convexity_harvest import ConvexityHarvestStrategy
            from src.strategy.vol_parity_allocator import VolatilityParityAllocator
            
            # Get convexity harvest signal
            convexity_engine = ConvexityHarvestStrategy()
            convexity_signal = convexity_engine.get_current_signal()
            
            # Get volatility parity allocation  
            vol_allocator = VolatilityParityAllocator(vix_strategy=convexity_engine)
            vol_parity_data = vol_allocator.get_current_allocation()
            if vol_parity_data:
                vol_parity_signal = vol_parity_data.get('allocation')
        except SIGNAL_EXCEPTIONS as e:
            _log_signal_error("convexity_harvest", e)

        # Add LLM sentiment signals (v2.30 Phase 5)
        sentiment_signal = None
        try:
            from src.strategy.regime_sentiment import RegimeSentimentPipeline
            
            sentiment_pipeline = RegimeSentimentPipeline()
            # Get current technical regime for combination
            tech_regime = trend_regime if trend_regime else "neutral"
            tech_confidence = 0.6  # Default confidence
            
            # Get combined sentiment signal (mock mode if no API keys)
            sentiment_signal = sentiment_pipeline.get_combined_signal(
                technical_regime=tech_regime,
                technical_confidence=tech_confidence,
                news_texts=[],  # Empty for mock mode
                earnings_texts=[],
                macro_texts=[],
            )
            sentiment_signal = sentiment_signal.to_dict()
        except SIGNAL_EXCEPTIONS as e:
            _log_signal_error("llm_sentiment", e)

        # Add ensemble voting signals (v2.20 Phase 3)
        ensemble_signal = None
        try:
            from src.strategy.ensemble_voter import EnsembleVoter

            ensemble_engine = EnsembleVoter()
            ensemble_result = ensemble_engine.compute_vote()
            if ensemble_result:
                source_breakdown = self._build_ensemble_source_breakdown(
                    ensemble_result.source_votes
                )
                source_counts = self._build_ensemble_source_count_metadata(
                    ensemble_result.regime,
                    source_breakdown,
                )
                configured_source_status = self._build_configured_source_status(
                    ensemble_result.regime,
                    source_breakdown,
                )
                ensemble_signal = {
                    "regime": ensemble_result.regime.value,
                    "regime_confidence": ensemble_result.regime_confidence,
                    "weighted_consensus": ensemble_result.weighted_consensus,
                    "agreement_ratio": ensemble_result.agreement_ratio,
                    "action": ensemble_result.action,
                    "confidence": ensemble_result.confidence,
                    "equity_bias": round(ensemble_result.equity_bias, 3),
                    "duration_bias": round(ensemble_result.duration_bias, 3),
                    "gold_bias": round(ensemble_result.gold_bias, 3),
                    **source_counts,
                    "configured_source_status": configured_source_status,
                    "n_eff": round(getattr(ensemble_result, 'n_eff', 0), 2),
                    "weight_entropy": round(getattr(ensemble_result, 'weight_entropy', 0), 4),
                    "adaptive_learning": self._build_ensemble_adaptive_learning_disclosure(
                        ensemble_result
                    ),
                    "source_breakdown": source_breakdown,
                }
        except SIGNAL_EXCEPTIONS as e:
            _log_signal_error("ensemble_voting", e)

        # Add sector rotation momentum signals (v2.40 Phase 5)
        sector_momentum_signal = None
        try:
            sector_momentum_signal = self._generate_sector_momentum_signals(vix_level=vix_level)
        except SIGNAL_EXCEPTIONS as e:
            _log_signal_error("sector_momentum", e)

        # Add smart rebalancing status (v2.90)
        smart_rebalance_data = None
        try:
            import importlib
            rebalancing_pkg = importlib.import_module('src.rebalancing')
            SmartRebalanceGate = rebalancing_pkg.integration.SmartRebalanceGate

            gate = SmartRebalanceGate()
            # Build current holdings from positions
            holdings = {p['symbol']: p['value'] for p in positions} if positions else {}
            if holdings and total_value > 0:
                gate_result = gate.evaluate(
                    current_holdings=holdings,
                    target_allocations=target_alloc,
                    total_value=total_value,
                )
                gate_status = gate.get_status()
                remaining_budget_ratio = _remaining_budget_ratio(
                    gate_result.metadata,
                    gate_status,
                )
                smart_rebalance_data = {
                    'should_execute': gate_result.should_execute,
                    'decision': gate_result.decision,
                    'urgency': gate_result.urgency,
                    'max_drift': gate_result.max_drift,
                    'estimated_cost_bps': gate_result.estimated_cost_bps,
                    'reason': gate_result.reason,
                    'drift_details': gate_result.metadata.get('drift_details', {}),
                    'vpin': gate_result.metadata.get('vpin', 0.30),
                    'in_optimal_window': gate_result.metadata.get('in_optimal_window', False),
                    'ytd_cost_bps': gate_result.metadata.get('ytd_cost_bps', 0),
                    'remaining_budget_pct': _remaining_budget_display_pct(
                        remaining_budget_ratio,
                        gate_status,
                    ),
                    'remaining_budget_ratio': remaining_budget_ratio,
                    'status': gate_status,
                }
            else:
                # No positions — use gate status only
                gate_status = gate.get_status()
                remaining_budget_ratio = _remaining_budget_ratio({}, gate_status)
                smart_rebalance_data = {
                    'should_execute': False,
                    'decision': 'no_positions',
                    'urgency': 'low',
                    'max_drift': 0,
                    'estimated_cost_bps': 0,
                    'reason': 'no_positions',
                    'drift_details': {},
                    'vpin': 0.30,
                    'in_optimal_window': False,
                    'ytd_cost_bps': 0,
                    'remaining_budget_pct': _remaining_budget_display_pct(
                        remaining_budget_ratio,
                        gate_status,
                    ),
                    'remaining_budget_ratio': remaining_budget_ratio,
                    'status': gate_status,
                }
            # Kill authority blocks actionable execute (order_router SSOT)
            smart_rebalance_data = _apply_kill_to_smart_rebalance(
                smart_rebalance_data,
                load_kill_switch_payload(DATA_DIR),
            )
        except (OSError, json.JSONDecodeError, KeyError, ValueError, TypeError, ImportError, RuntimeError) as e:
            logger.warning("Dashboard generation error: %s", e)

        # Add alternative data signals (v2.60 Phase 3)
        alternative_data_signal = None
        try:
            alt_data_file = DATA_DIR / "signals" / "alternative_data_latest.json"
            if alt_data_file.exists():
                with open(alt_data_file) as f:
                    alt_data_raw = json.load(f)
                alternative_data_signal = project_alternative_data_signal(alt_data_raw)
        except (OSError, json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
            logger.warning("Alternative data signal not available: %s", e)

        # Load broker data (Phase 4: live trading prep)
        broker_data = self._load_broker_data()

        # Portfolio drift alerting
        try:
            from src.monitor.alerting import check_drift_and_alert
            drift_info = broker_data.get("drift", {})
            max_drift = drift_info.get("max_drift_pct", 0.0) if isinstance(drift_info, dict) else 0.0
            if max_drift:
                check_drift_and_alert(max_drift)
        except (ImportError, ValueError, OSError, RuntimeError) as e:
            _log_signal_error("drift_alerting", e)

        # Add GARCH-CVaR metrics (v3.21)
        garch_cvar_data = self._load_garch_cvar_data()

        # Add entropy diversification metrics (v3.22)
        entropy_data = self._load_entropy_data()

        # Behavioral sentiment data (v2.70)
        behavioral_sentiment_data = None
        try:
            from src.signals.behavioral_sentiment import BehavioralSentimentSignal
            from src.data.behavioral_sentiment_fetcher import BehavioralSentimentFetcher

            sig_gen = BehavioralSentimentSignal(cache_db=DB_PATH)
            fetcher = BehavioralSentimentFetcher(cache_db=DB_PATH)
            snapshot = fetcher.fetch_snapshot()
            signal = sig_gen.get_signal(snapshot)
            status = sig_gen.get_status()

            now_ts = datetime.now(timezone.utc).isoformat()
            behavioral_sentiment_data = {
                "active": True,
                "composite_score": signal.composite_score,
                "signal_type": signal.signal_type,
                "confidence": signal.confidence,
                "equity_shift_pct": signal.equity_shift_pct,
                "z_score": signal.z_score,
                "vix": signal.vix,
                "regime_suppressed": signal.regime_suppressed,
                "signal_count_5d": status.get("signal_count_5d", 0),
                "options": {
                    "skew_index": round(snapshot.options.skew_index, 1),
                    "vix": round(snapshot.options.vix, 1),
                    "vix9d": round(snapshot.options.vix9d, 1),
                    "vix9d_ratio": round(snapshot.options.vix9d_ratio, 2),
                    "put_call_ratio": round(snapshot.options.put_call_ratio, 2),
                    "fear_greed_score": round(snapshot.options.fear_greed_score, 1),
                },
                "retail": {
                    "retail_call_put_ratio": round(snapshot.retail.retail_call_put_ratio, 2),
                    "retail_buy_sell_imbalance": round(snapshot.retail.retail_buy_sell_imbalance, 2),
                },
                "social": {
                    "mention_velocity_7d": round(snapshot.social.mention_velocity_7d, 2),
                    "sentiment_divergence": round(snapshot.social.sentiment_divergence, 3),
                },
                "backtest_finding": (
                    "VIX-proxy contrarian signals degrade Sharpe by -0.216 (2021-2026). "
                    "Real-time SKEW/PCR data needed for behavioral alpha."
                ),
                "timestamp": now_ts,
                "generated_at": now_ts,
            }
        except SIGNAL_EXCEPTIONS as e:
            _log_signal_error("behavioral_sentiment", e)

        # Stacking ensemble dashboard data (v3.10)
        stacking_ensemble_dashboard = None
        try:
            from src.signals.stacking_integrator import StackingIntegrator

            integrator = StackingIntegrator()
            if integrator.model is None:
                stacking_ensemble_dashboard = self._build_stacking_no_model_dashboard(integrator)
            else:
                prediction = integrator.predict({})
                stacking_ensemble_dashboard = self._build_stacking_model_dashboard(
                    integrator,
                    prediction,
                )
        except SIGNAL_EXCEPTIONS as e:
            _log_signal_error("stacking_ensemble", e)

        # Factor rotation dashboard data (v3.00) — reuses factor_rotation_result from above
        factor_rotation_dashboard = None
        try:
            from src.strategy.factor_rotation import FactorMomentumEngine

            if factor_rotation_result is not None and "error" not in factor_rotation_result:
                allocations = factor_rotation_result.get("allocation", {})
                now_ts = datetime.now(timezone.utc).isoformat()
                factor_rotation_dashboard = {
                    "active": True,
                    "selected_factors": factor_rotation_result.get("selected_factors", []),
                    "signal_strength": round(factor_rotation_result.get("signal_strength", 0.0), 2),
                    "factor_allocations": allocations,
                    "backtest_finding": (
                        "Factor rotation reduces MaxDD by 5.8pp (2021-2026). "
                        "Defensive tool — best in high-vol regimes (Sharpe 1.474)."
                    ),
                    "generated_at": now_ts,
                    "timestamp": now_ts,
                }
        except SIGNAL_EXCEPTIONS as e:
            _log_signal_error("factor_rotation_dashboard", e)

        # Merge overlay dashboard data (collar, crypto, calendar, kurtosis, etc.)
        overlay_data = self._get_overlay_data()

        # Hedge selector recommendation
        hedge_selector_signal = None
        try:
            hedge_selector_signal = self._get_hedge_selector_signal(
                vix_level,
                current_regime,
                overlay_data.get("vix_term_structure", {}),
            )
        except MONITOR_EXCEPTIONS as e:
            _log_signal_error("hedge_selector", e)

        return {
            "regime": validate_signal("regime", regime_data),
            "target_allocations": target_alloc,
            "allocation_surface_roles": self._build_allocation_surface_roles(),
            "regime_authority": self._build_regime_authority(current_regime, target_alloc),
            "current_positions": positions,
            "cash": round(cash, 2),
            "total_value": round(total_value, 2),
            "latest_prices": latest,
            "recent_orders": list(reversed(orders)),
            "ml_signals": self._generate_ml_signals(),
            "marl_status": validate_signal("marl_status", self._generate_marl_status()),
            "factor_rotation": factor_rotation_signal,
            "yield_curve": validate_signal("yield_curve", yield_curve_data.get("yield_curve")),
            "duration_allocation": yield_curve_data.get("duration_allocation"),
            "convexity_harvest": convexity_signal,
            "volatility_parity": vol_parity_signal,
            "llm_sentiment": sentiment_signal,
            "ensemble_voting": validate_signal("ensemble_voting", ensemble_signal),
            "sector_rotation": sector_momentum_signal,
            "alternative_data": alternative_data_signal,
            "behavioral_sentiment": behavioral_sentiment_data,
            "collar": overlay_data.get("collar", {}),
            "crypto_allocation": overlay_data.get("crypto", {}),
            "calendar_seasonality": overlay_data.get("calendar", {}),
            "kurtosis_regime": overlay_data.get("kurtosis", {}),
            "vix_term_structure": overlay_data.get("vix_term_structure", {}),
            "zero_dte": overlay_data.get("zero_dte", {}),
            "closing_auction": overlay_data.get("closing_auction", {}),
            "stacking_ensemble": stacking_ensemble_dashboard,
            "factor_rotation_dashboard": factor_rotation_dashboard,
            "smart_rebalance": validate_signal("smart_rebalance", smart_rebalance_data),
            "broker": broker_data,
            "garch_cvar": validate_signal("garch_cvar", garch_cvar_data),
            "entropy": entropy_data,
            "bond_momentum": overlay_data.get("bond_momentum", {}),
            "hedge_selector": validate_signal("hedge_selector", hedge_selector_signal),
            # Sidecar JSON is generated later in the same cycle; embed prior or
            # freshly computed snapshot so optional staleness sees a section.
            "risk_decomposition": self._load_risk_decomposition_signal_section(),
        }

    @staticmethod
    def _build_stacking_feature_count_metadata(integrator: Any) -> Dict[str, Any]:
        """Expose stacking feature count only when loaded model metadata backs it."""
        metadata = getattr(integrator, "metadata", None)
        feature_count = getattr(metadata, "feature_count", None)
        model_loaded = getattr(integrator, "model", None) is not None

        if model_loaded and feature_count is not None:
            return {
                "feature_count": int(feature_count),
                "feature_count_metadata_available": True,
                "feature_count_source": "model_metadata",
                "source_roster": list(getattr(metadata, "source_roster", [])),
                "source_roster_version": getattr(
                    metadata,
                    "source_roster_version",
                    "unavailable_missing_metadata",
                ),
                "fallback_semantics": getattr(
                    metadata,
                    "fallback_semantics",
                    "unavailable_missing_metadata",
                ),
            }

        return {
            "feature_count": None,
            "feature_count_metadata_available": False,
            "feature_count_source": (
                "unavailable_missing_metadata" if model_loaded else "unavailable_no_model"
            ),
            "source_roster": [],
            "source_roster_version": (
                "unavailable_missing_metadata" if model_loaded else "unavailable_no_model"
            ),
            "fallback_semantics": (
                "unavailable_missing_metadata" if model_loaded else "no_model_feature_count_unavailable"
            ),
        }

    @staticmethod
    def _build_stacking_no_model_dashboard(integrator: Any) -> Dict[str, Any]:
        """Build the explicit dormant stacking artifact when no model is loaded."""
        now_ts = datetime.now(timezone.utc).isoformat()
        return {
            "active": False,
            "stacking_available": False,
            "runtime_role": "research_dormant",
            "runtime_status": "unavailable_no_model",
            "live_authoritative": False,
            "routed": False,
            "routed_by": None,
            "prediction_available": False,
            "prediction_direction": "unavailable",
            "confidence": 0.0,
            "probability_bullish": 0.0,
            "probability_bearish": 0.0,
            "probability_neutral": 0.0,
            "fallback_used": False,
            "model_version": "unavailable_no_model",
            "voting_accuracy": None,
            "stacking_accuracy": None,
            "accuracy_metrics_available": False,
            **DashboardGenerator._build_stacking_feature_count_metadata(integrator),
            "latency_ms": 0.0,
            "status_reason": (
                "No stacking model artifact is loaded and no runtime base-signal "
                "input path is available."
            ),
            "operator_message": (
                "Stacking ensemble is research/dormant, not live-authoritative, "
                "and not order-routed."
            ),
            "timestamp": now_ts,
            "generated_at": now_ts,
        }

    @staticmethod
    def _build_stacking_model_dashboard(integrator: Any, prediction: Any) -> Dict[str, Any]:
        """Build the model-backed stacking dashboard artifact."""
        now_ts = datetime.now(timezone.utc).isoformat()
        return {
            "active": True,
            "stacking_available": True,
            "runtime_role": "model_backed_advisory",
            "runtime_status": "model_loaded",
            "live_authoritative": False,
            "routed": False,
            "routed_by": None,
            "prediction_available": prediction is not None,
            "prediction_direction": prediction.direction if prediction else "unavailable",
            "confidence": prediction.confidence if prediction else 0.0,
            "probability_bullish": prediction.probability_bullish if prediction else 0.0,
            "probability_bearish": prediction.probability_bearish if prediction else 0.0,
            "probability_neutral": prediction.probability_neutral if prediction else 0.0,
            "fallback_used": prediction.fallback_used if prediction else False,
            "model_version": prediction.model_version if prediction else "unknown",
            "voting_accuracy": 0.65,
            "stacking_accuracy": 0.76,
            "accuracy_metrics_available": True,
            **DashboardGenerator._build_stacking_feature_count_metadata(integrator),
            "latency_ms": prediction.latency_ms if prediction else 0.0,
            "status_reason": "Stacking model artifact is loaded for advisory inference.",
            "operator_message": (
                "Stacking ensemble is advisory and not order-routed; live routing "
                "still consumes target_allocations."
            ),
            "backtest_finding": (
                "+11% accuracy produces negligible Sharpe gain (2021-2026). "
                "Signal frequency and shift magnitude are binding constraints."
            ),
            "timestamp": now_ts,
            "generated_at": now_ts,
        }

    def _build_optional_signal_sections(
        self,
        output: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Append optional operational sections that precede staleness checks."""
        # Rebalance health data
        try:
            from src.monitor.rebalance_health import generate as gen_rebalance_health
            output["rebalance_health"] = gen_rebalance_health()
        except SIGNAL_EXCEPTIONS as e:
            _log_signal_error("rebalance_health", e)
            output["rebalance_health"] = {"generated": None, "error": str(e)}

        # Circuit breaker state (broker API resilience)
        try:
            from src.broker.circuit_breaker import get_circuit_state
            output["broker_circuit_breaker"] = get_circuit_state()
        except ImportError:
            pass  # circuit_breaker module not available

        return output

    def _apply_signal_postprocessors(
        self,
        output: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Apply staleness, monitoring, alerting, and final signal appenders."""
        cursor = context["cursor"]
        current_regime = context["current_regime"]

        # FRED-MD macro regime signal
        try:
            from src.data import fred_data
            fred_signal = fred_data.get_fred_signal()
            readiness_getter = getattr(fred_data, "get_fred_md_cache_health", None)
            fred_readiness = readiness_getter() if callable(readiness_getter) else {}
            indicators = getattr(fred_signal, "indicators", {}) or {}
            indicators_observed = bool(
                getattr(fred_signal, "indicators_observed", bool(indicators))
            )
            source_mode = _first_known_value(
                getattr(fred_signal, "source_mode", None)
                if indicators_observed else None,
                fred_readiness.get("source_mode"),
                getattr(fred_signal, "source_mode", None),
                default="unknown",
            )
            cache_status = _first_known_value(
                fred_readiness.get("status")
                if fred_readiness else None,
                getattr(fred_signal, "cache_status", None),
                default="unknown",
            )
            status = (
                "ok"
                if _is_predictive_fred_macro({
                    "confidence": fred_signal.confidence,
                    "indicators": indicators,
                    "indicators_observed": indicators_observed,
                    "source_mode": source_mode,
                    "cache_status": cache_status,
                })
                else "unavailable"
            )
            output["fred_macro"] = validate_signal("fred_macro", {
                "regime": fred_signal.regime,
                "confidence": fred_signal.confidence,
                "recession_probability": fred_signal.recession_probability,
                "inflation_pressure": fred_signal.inflation_pressure,
                "monetary_stance": fred_signal.monetary_stance,
                "manufacturing_health": fred_signal.manufacturing_health,
                "credit_conditions": fred_signal.credit_conditions,
                "indicators": indicators,
                "timestamp": fred_signal.timestamp,
                "status": status,
                "source_mode": source_mode,
                "cache_status": cache_status,
                "api_key_configured": fred_readiness.get(
                    "api_key_configured",
                    getattr(fred_signal, "api_key_configured", False),
                ),
                "reason": getattr(fred_signal, "reason", None) or fred_readiness.get("reason"),
                "latest_fetched_at": fred_readiness.get(
                    "latest_fetched_at",
                    getattr(fred_signal, "latest_fetched_at", None),
                ),
                "row_count": fred_readiness.get("row_count"),
                "age_hours": fred_readiness.get("age_hours"),
                "ttl_hours": fred_readiness.get("ttl_hours"),
                "indicators_observed": indicators_observed,
            })
        except MONITOR_EXCEPTIONS as e:
            _log_signal_error("fred_macro", e)
            output["fred_macro"] = {
                "regime": "UNKNOWN",
                "confidence": 0.0,
                "status": "unavailable",
                "source_mode": "unavailable",
                "cache_status": "unavailable",
                "indicators": {},
                "indicators_observed": False,
                "error": str(e),
            }

        # Two-stage k-means macro regime classifier (Oliveira et al. 2025)
        try:
            two_stage_signal = self._generate_two_stage_regime()
            if two_stage_signal:
                output["two_stage_regime"] = validate_signal(
                    "two_stage_regime", two_stage_signal,
                )
        except MONITOR_EXCEPTIONS as e:
            _log_signal_error("two_stage_regime", e)
            output["two_stage_regime"] = {
                "regime": "UNKNOWN",
                "confidence": 0.0,
                "error": str(e),
            }

        # Bayesian Online Changepoint Detection (BOCD) regime signal
        try:
            bocd_signal = self._generate_bocd_regime()
            if bocd_signal:
                output["bocd_regime"] = validate_signal(
                    "bocd_regime", bocd_signal,
                )
        except MONITOR_EXCEPTIONS as e:
            _log_signal_error("bocd_regime", e)
            output["bocd_regime"] = {
                "regime": 0,
                "regime_change_prob": 0.0,
                "error": str(e),
            }

        # Regime transition forecast (Oliveira et al. 2025 step 2)
        try:
            from src.regime.regime_transition_forecaster import RegimeTransitionForecaster
            forecaster = RegimeTransitionForecaster()
            # Extract regime labels from two_stage_regime signal or VIX classification
            current = output.get("two_stage_regime", {}).get("regime", current_regime)
            # Fit on recent regime history from regime_log
            cursor.execute("SELECT regime FROM regime_log ORDER BY detected_at DESC LIMIT 100")
            history = [row[0] for row in cursor.fetchall()]
            if len(history) >= 2:
                forecaster.fit(list(reversed(history)))
                forecast = forecaster.forecast(current, horizon_days=5)
                output["regime_transition"] = {
                    "current_regime": current,
                    "horizon_days": 5,
                    "forecast_probs": {k: round(v, 4) for k, v in forecast.probabilities.items()},
                    "most_likely": forecast.most_likely,
                    "persistence_params": {k: round(v, 1) for k, v in forecast.persistence_params.items()},
                    "timestamp": datetime.now().isoformat(),
                }
        except MONITOR_EXCEPTIONS as e:
            _log_signal_error("regime_transition", e)

        # Signal staleness must be computed after optional regime sections are appended.
        output["staleness"] = self._check_signal_staleness(output)
        self._update_regime_authority_availability(output)

        # Apply staleness-weighted decay to ensemble weights
        output = self._apply_staleness_decay(output)

        # Health check report
        try:
            from src.monitor.health_check import run_health_check
            health_report = _load_canonical_health_report() or run_health_check()
            output["health"] = _compact_health_summary(health_report)
        except Exception as e:
            output["health"] = _compact_health_summary({"status": "error", "error": str(e)})

        # Fire external alerts on staleness state transitions (+ recovery ownership)
        try:
            from src.monitor.alerting import check_staleness_and_alert
            from src.monitor.signal_ownership import (
                annotate_unavailable_signals,
                recovery_summary,
            )

            staleness = output.get("staleness")
            if isinstance(staleness, dict):
                ml_on = os.environ.get("PORTFOLIO_LAB_ENABLE_ML", "0") == "1"
                ownership = annotate_unavailable_signals(
                    staleness.get("unavailable_signals") or [],
                    ml_enabled=ml_on,
                )
                if ownership:
                    staleness = dict(staleness)
                    staleness["unavailable_ownership"] = ownership
                    staleness["recovery"] = recovery_summary(ownership)
                    output["staleness"] = staleness
            check_staleness_and_alert(output["staleness"])
        except SIGNAL_EXCEPTIONS as e:
            _log_signal_error("alerting", e)

        # SPC signal quality monitoring
        output["spc"] = self._run_spc_monitor(output)

        # IC decay data recording — resolve staged predictions + stage new ones
        self._record_ic_data(output)

        # IC decay monitoring (signal predictive quality tracking)
        try:
            from src.monitor.ic_decay_monitor import compute_ic_decay_report
            output["ic_decay"] = compute_ic_decay_report()
        except MONITOR_EXCEPTIONS as e:
            _log_signal_error("ic_decay_monitor", e)
            output["ic_decay"] = {"error": str(e)}

        # IC decay alerting — fire alerts for signals with degrading IC
        try:
            from src.monitor.alerting import check_ic_decay_and_alert
            check_ic_decay_and_alert(output.get("ic_decay", {}))
        except MONITOR_EXCEPTIONS as e:
            _log_signal_error("ic_decay_alerting", e)

        # Per-signal walk-forward validation
        try:
            from src.monitor.signal_walk_forward import compute_signal_wfe_report
            output["signal_wfe"] = compute_signal_wfe_report()
        except MONITOR_EXCEPTIONS as e:
            _log_signal_error("signal_wfe", e)
            output["signal_wfe"] = {"error": str(e)}

        # Gold-TLT correlation regime monitor
        try:
            from src.research.gold_tlt_correlation import run_analysis
            analysis = run_analysis(window=252, save=False)
            output["gold_tlt_correlation"] = {
                "current_correlation": analysis.current_correlation,
                "current_regime": analysis.current_regime,
                "correlation_trend": analysis.correlation_trend,
                "mean_correlation": analysis.mean_correlation,
                "min_correlation": analysis.min_correlation,
                "max_correlation": analysis.max_correlation,
                "structural_breaks_count": len(analysis.structural_breaks),
                "regimes_count": len(analysis.regimes),
                "implications": analysis.implications,
            }
        except MONITOR_EXCEPTIONS as e:
            _log_signal_error("gold_tlt_correlation", e)
            output["gold_tlt_correlation"] = {"error": str(e)}

        # Paper→Live ramp status
        try:
            from src.broker.alpaca import LiveTransitionManager
            ramp_mgr = LiveTransitionManager()
            output["ramp"] = ramp_mgr.get_status()
        except MONITOR_EXCEPTIONS as e:
            _log_signal_error("ramp_status", e)
            output["ramp"] = {"error": str(e)}

        return output

    def generate_signals_json(self) -> Path:
        """Generate current signals and allocations."""
        context = self._load_signal_generation_context()
        output = _attach_signal_metadata(self._build_base_signal_sections(context))
        output = self._build_optional_signal_sections(output, context)
        output = self._apply_signal_postprocessors(output, context)
        output = _finalize_signal_metadata(output)

        try:
            from src.monitor.decision_registry import record_dashboard_cycle_decision

            record_dashboard_cycle_decision(output, context=context)
        except (ImportError, ValueError, OSError, TypeError) as e:
            logger.warning("Decision registry record skipped: %s", e)

        out_path = PUBLIC_DIR / "signals.json"
        save_results_json(output, output_path=str(out_path), validator=validate_all_signals)

        return out_path

    def _load_broker_data(self) -> Dict:
        """Load broker position sync and order data for dashboard."""
        from src.dashboard.broker_data_loader import BrokerDataLoader

        return BrokerDataLoader(data_dir=DATA_DIR).load()

    def _load_garch_cvar_data(self) -> Dict:
        """Load GARCH-filtered CVaR metrics for dashboard (v3.21).

        Also computes a conformal CVaR cross-check (distribution-free)
        as a model-risk validation against the parametric GARCH estimate.
        """
        garch_cvar = {
            "cvar_95": -0.0179,
            "cvar_95_garch": -0.0215,
            "var_95": -0.0127,
            "var_95_garch": -0.0142,
            "cvar_ratio": 1.51,
            "garch_active": True,
            "current_volatility": 0.012,
            "forecast_volatility": 0.015,
            "volatility_clustering": "elevated",
            # Conformal cross-check defaults
            "conformal_cvar_95": None,
            "conformal_var_95": None,
            "conformal_cvar_ratio": None,
            "coverage_diagnostics": None,
        }

        # Compute conformal CVaR cross-check from SPY returns
        try:
            from src.monitor.conformal_risk import (
                conformal_coverage_diagnostics,
                conformal_cvar,
                conformal_var,
            )
            cursor = self.conn.cursor()
            cursor.execute(
                "SELECT close FROM prices WHERE symbol = 'SPY' ORDER BY date ASC"
            )
            rows = cursor.fetchall()
            if len(rows) >= 22:  # Need at least 22 days for meaningful split
                prices = np.array([r[0] for r in rows], dtype=float)
                returns = np.diff(np.log(prices))
                garch_cvar["conformal_cvar_95"] = round(
                    float(conformal_cvar(returns, alpha=0.05)), 6,
                )
                garch_cvar["conformal_var_95"] = round(
                    float(conformal_var(returns, alpha=0.05)), 6,
                )
                var_thresholds = np.full_like(
                    returns,
                    garch_cvar["conformal_var_95"],
                    dtype=float,
                )
                garch_cvar["coverage_diagnostics"] = conformal_coverage_diagnostics(
                    returns,
                    var_thresholds,
                    alpha=0.05,
                    rolling_window=252,
                )
                if garch_cvar["conformal_var_95"] != 0:
                    garch_cvar["conformal_cvar_ratio"] = round(
                        garch_cvar["conformal_cvar_95"]
                        / garch_cvar["conformal_var_95"], 3,
                    )
        except (ImportError, ValueError, TypeError, IndexError) as e:
            logger.info("Conformal CVaR cross-check unavailable: %s", e)

        try:
            # Load from GARCH-CVaR health report (flat format from compute_garch_risk.py)
            health_file = DATA_DIR / ".health_report.json"
            if health_file.exists():
                with open(health_file) as f:
                    data = json.load(f)

                # Support both flat GARCHCVaRMetrics format and nested checks format
                if data.get("garch_filtered") is not None:
                    # Flat format (from compute_garch_risk.py / evaluator._write_garch_health_report)
                    garch_cvar["cvar_95"] = data.get("cvar_95", garch_cvar["cvar_95"]) / 100.0 if abs(data.get("cvar_95", 0)) > 1 else data.get("cvar_95", garch_cvar["cvar_95"])
                    garch_cvar["var_95"] = data.get("var_95", garch_cvar["var_95"]) / 100.0 if abs(data.get("var_95", 0)) > 1 else data.get("var_95", garch_cvar["var_95"])
                    garch_cvar["cvar_ratio"] = data.get("cvar_ratio", garch_cvar["cvar_ratio"])
                    garch_cvar["garch_active"] = data.get("filter_active", False)
                    if data.get("conditional_volatility_current") is not None:
                        garch_cvar["current_volatility"] = data["conditional_volatility_current"] / 100.0
                    if data.get("garch_persistence") is not None:
                        garch_cvar["volatility_clustering"] = "high" if data["garch_persistence"] > 0.95 else "elevated" if data["garch_persistence"] > 0.85 else "normal"
                elif safe_get(data, "checks", "cvar_metrics", "garch_filtered"):
                    # Legacy nested format
                    cvar_check = data["checks"]["cvar_metrics"]
                    garch_cvar["cvar_95"] = cvar_check.get("cvar_95", -0.0179)
                    garch_cvar["var_95"] = cvar_check.get("var_95", -0.0127)
                    garch_cvar["cvar_ratio"] = cvar_check.get("cvar_ratio", 1.51)
                    garch_cvar["garch_active"] = cvar_check.get("garch_active", True)
        except (OSError, json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
            logger.warning("Using default values: %s", e)

        return garch_cvar

    def _load_entropy_data(self) -> Dict:
        """Load entropy-based diversification metrics for dashboard (v3.22)."""
        entropy = {
            "shannon_entropy": 1.02,
            "effective_n": 2.77,
            "max_possible": 1.10,
            "normalized_score": 92.7,
            "concentration_risk": "good",
            "hhi_index": 0.38,
            "correlation_entropy": 0.95,
            "participation_ratio": 2.5,
        }
        
        try:
            # Try to load from health report which now includes entropy metrics
            health_file = DATA_DIR / ".health_report.json"
            if health_file.exists():
                with open(health_file) as f:
                    health = json.load(f)
                    entropy_check = safe_get(health, "checks", "portfolio_entropy", default={})
                    metrics = entropy_check.get("metrics", {})
                    if metrics:
                        entropy["shannon_entropy"] = metrics.get("shannon_entropy", 1.02)
                        entropy["effective_n"] = metrics.get("effective_n", 2.77)
                        entropy["normalized_score"] = metrics.get("normalized_score", 92.7)
                        entropy["hhi_index"] = metrics.get("hhi_index", 0.38)
                        
                        # Determine concentration risk from normalized score
                        score = entropy["normalized_score"]
                        if score > 90:
                            entropy["concentration_risk"] = "good"
                        elif score > 70:
                            entropy["concentration_risk"] = "low"
                        elif score > 50:
                            entropy["concentration_risk"] = "medium"
                        elif score > 30:
                            entropy["concentration_risk"] = "high"
                        else:
                            entropy["concentration_risk"] = "critical"
        except (OSError, json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
            logger.warning("Using default values: %s", e)

        return entropy

    def _generate_ml_signals(self) -> Dict:
        """Generate ML-based signals from features data."""
        def parse_timestamp(value: Any) -> Optional[datetime]:
            if not isinstance(value, str) or not value:
                return None
            try:
                parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                return parsed.astimezone(timezone.utc)
            except ValueError:
                return None

        def staleness_days(value: Any, now: datetime) -> Optional[int]:
            parsed = parse_timestamp(value)
            if parsed is None:
                return None
            return max(0, (now - parsed).days)

        def feature_freshness(value: Any, now: datetime) -> tuple[str, Optional[int]]:
            age_days = staleness_days(value, now)
            if age_days is None:
                return "unknown", None
            return ("stale" if age_days > 2 else "fresh"), age_days

        generated_at_dt = datetime.now(timezone.utc)
        generated_at = generated_at_dt.isoformat()
        signals = {
            "available": False,
            "timestamp": None,
            "generated_at": None,
            "feature_source_artifact": None,
            "feature_as_of": None,
            "feature_freshness_status": "unknown",
            "feature_staleness_days": None,
            "prediction_source_mode": "unavailable",
            "execution_role": {
                "role": "advisory_non_routed",
                "routed": False,
                "routed_by": None,
                "live_authoritative": False,
            },
            "predictions": {},
            "features": {},
            "grid_search": {},
        }
        
        # Check for features file
        features_file = DATA_DIR / "features.jsonl"
        if features_file.exists():
            try:
                # Get latest features for each symbol (tail read — last 500 lines)
                latest_features = {}
                with open(features_file, 'r') as f:
                    for line in deque(f, maxlen=500):
                        try:
                            feat = json.loads(line)
                            sym = feat.get("symbol")
                            ts = feat.get("timestamp", "")
                            if sym and (sym not in latest_features or ts > latest_features[sym].get("timestamp", "")):
                                latest_features[sym] = feat
                        except json.JSONDecodeError:
                            logger.exception("Failed to parse feature line in features.jsonl")
                            continue

                if latest_features:
                    feature_as_of = max(
                        (feat.get("timestamp") for feat in latest_features.values() if feat.get("timestamp")),
                        default=None,
                    )
                    freshness_status, age_days = feature_freshness(feature_as_of, generated_at_dt)
                    signals["available"] = True
                    signals["timestamp"] = generated_at
                    signals["generated_at"] = generated_at
                    signals["feature_source_artifact"] = "features.jsonl"
                    signals["feature_as_of"] = feature_as_of
                    signals["feature_freshness_status"] = freshness_status
                    signals["feature_staleness_days"] = age_days
                    signals["prediction_source_mode"] = (
                        "stale_features" if freshness_status == "stale" else "features"
                    )
                    signals["features"] = {
                        sym: {
                            "vix_level": feat.get("vix_level"),
                            "trend_direction": feat.get("trend_direction"),
                            "price_vs_sma20": feat.get("price_vs_sma20"),
                            "return_5d": feat.get("return_5d"),
                            "spy_correlation": feat.get("spy_correlation_20d"),
                            "feature_timestamp": feat.get("timestamp"),
                        }
                        for sym, feat in latest_features.items()
                    }
                    
                    # Generate simple heuristic predictions
                    for sym, feat in latest_features.items():
                        vix = feat.get("vix_level", 20)
                        trend = feat.get("trend_direction", 0)
                        price_vs_sma = feat.get("price_vs_sma20", 0)
                        
                        # Simple regime probability
                        if vix > 25:
                            p_bear, p_neutral, p_bull = 0.5, 0.3, 0.2
                        elif vix > 20:
                            p_bear, p_neutral, p_bull = 0.3, 0.5, 0.2
                        elif trend > 0 and price_vs_sma > 0:
                            p_bear, p_neutral, p_bull = 0.1, 0.3, 0.6
                        elif trend < 0:
                            p_bear, p_neutral, p_bull = 0.4, 0.4, 0.2
                        else:
                            p_bear, p_neutral, p_bull = 0.2, 0.6, 0.2
                        
                        # Map to regime names
                        probs = {"bear": p_bear, "neutral": p_neutral, "bull": p_bull}
                        predicted = max(probs, key=probs.get)
                        confidence = probs[predicted]
                        
                        signals["predictions"][sym] = {
                            "predicted_regime": predicted,
                            "confidence": round(confidence, 3),
                            "probabilities": {k: round(v, 3) for k, v in probs.items()},
                            "heuristic": True,  # Not ML-based yet
                            "feature_timestamp": feat.get("timestamp"),
                            "feature_freshness_status": freshness_status,
                            "source_artifact": "features.jsonl",
                        }
            except (OSError, json.JSONDecodeError, KeyError, ValueError, TypeError, ImportError, RuntimeError) as e:
                signals["error"] = str(e)

        # Check for grid search results (tail read only)
        grid_file = DATA_DIR / "grid_search_results.jsonl"
        if grid_file.exists():
            try:
                with open(grid_file, 'r') as f:
                    tail = deque(f, maxlen=1)
                if tail:
                    latest = json.loads(tail[0])
                    benchmark_timestamp = latest.get("timestamp")
                    signals["grid_search"] = {
                        "available": True,
                        "timestamp": benchmark_timestamp,
                        "top_allocation": latest.get("allocations"),
                        "sharpe": latest.get("sharpe"),
                        "volatility": latest.get("volatility"),
                        "source_artifact": "grid_search_results.jsonl",
                        "benchmark_timestamp": benchmark_timestamp,
                        "observation_semantics": "frozen_benchmark_not_live_snapshot",
                        "freshness_status": "frozen_benchmark",
                        "staleness_days": staleness_days(benchmark_timestamp, generated_at_dt),
                        "live_authoritative": False,
                    }
            except (OSError, json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
                logger.warning("Failed to load grid search results: %s", e)

        return signals

    @staticmethod
    def _marl_execution_role() -> Dict[str, Any]:
        """Describe MARL's current operator-visible but non-routed role."""
        return {
            "role": "research_shadow_non_routed",
            "routed": False,
            "routed_by": None,
            "live_authoritative": False,
            "description": (
                "MARL status is visible for research/shadow diagnostics; "
                "order routing still consumes target_allocations."
            ),
        }

    @staticmethod
    def _default_marl_runtime_status() -> Dict[str, Any]:
        """Return a stable runtime shape for unavailable MARL status."""
        return {
            "version": "unknown",
            "device": "unknown",
            "agents_loaded": [],
            "signal_integrator_connected": False,
            "checkpoint_loaded": False,
            "inference_count": 0,
            "current_allocation": {},
            "graph_metrics": {},
        }

    @staticmethod
    def _generate_marl_status() -> Dict[str, Any]:
        """Expose AIController runtime status without implying live routing authority."""
        status = {
            "schema_version": "marl-runtime-status/v1",
            "available": False,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "runtime": DashboardGenerator._default_marl_runtime_status(),
            "execution_role": DashboardGenerator._marl_execution_role(),
        }

        try:
            from src.agents.ai_controller import AIController

            controller = AIController(use_signal_integrator=False)
            runtime_status = controller.get_status()
            if isinstance(runtime_status, dict):
                status["available"] = True
                status["runtime"] = {
                    **DashboardGenerator._default_marl_runtime_status(),
                    **runtime_status,
                }
        except SIGNAL_EXCEPTIONS as e:
            _log_signal_error("marl_status", e)
            status["error"] = str(e)

        return status
    
    def _get_yield_curve_data(self) -> Dict:
        """Get yield curve data from yields.json and calculate duration allocation."""
        result = {
            "yield_curve": None,
            "duration_allocation": None
        }
        
        yields_file = YIELDS_JSON
        if not yields_file.exists():
            return result
        
        try:
            with open(yields_file, 'r') as f:
                yields = json.load(f)
            
            if not yields or len(yields) == 0:
                return result
            
            # Get latest yield entry
            latest = yields[-1]
            
            # Calculate regime based on 2s10s spread
            spread = latest.get("spread2s10s", 0)
            if spread > 100:
                regime = "steep"
            elif spread > 50:
                regime = "normal"
            elif spread > 0:
                regime = "flat"
            else:
                regime = "inverted"
            
            # Get last 30 days of spread history for sparkline
            spread_history = []
            for entry in yields[-30:]:
                if entry.get("spread2s10s") is not None:
                    spread_history.append(entry["spread2s10s"])
            
            result["yield_curve"] = {
                "spread2s10s": spread,
                "dgs2": latest.get("dgs2"),
                "dgs10": latest.get("dgs10"),
                "duration_regime": regime,
                "spread_history": spread_history,
                **{
                    key: value
                    for key, value in _yield_source_provenance(PUBLIC_DIR).items()
                    if value is not None
                },
            }
            
            # Calculate duration allocation based on regime
            regime_allocations = {
                "steep": {"tlt": 0.70, "ief": 0.25, "shy": 0.05, "bil": 0.00},
                "normal": {"tlt": 0.50, "ief": 0.35, "shy": 0.15, "bil": 0.00},
                "flat": {"tlt": 0.30, "ief": 0.40, "shy": 0.25, "bil": 0.05},
                "inverted": {"tlt": 0.15, "ief": 0.25, "shy": 0.35, "bil": 0.25}
            }
            
            result["duration_allocation"] = regime_allocations.get(regime, regime_allocations["normal"])

        except (OSError, json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
            logger.warning("Failed to load yield curve data: %s", e)
        
        return result
    
    def generate_stats_json(self) -> Path:
        """Generate performance statistics."""
        cursor = self.conn.cursor()

        # Single batched query for all symbols instead of N+1 per-symbol queries
        symbols = ['SPY', 'GLD', 'TLT', 'QQQ', 'VIX']
        placeholders = ','.join('?' for _ in symbols)
        cursor.execute(f"""
            SELECT symbol, close FROM prices
            WHERE symbol IN ({placeholders}) AND date >= date('now', '-30 days')
            ORDER BY symbol, date
        """, symbols)

        # Group rows by symbol
        symbol_prices: Dict[str, List[float]] = {}
        for sym, close in cursor.fetchall():
            symbol_prices.setdefault(sym, []).append(close)

        stats = {}
        for symbol in symbols:
            prices = symbol_prices.get(symbol, [])
            if len(prices) >= 2:
                returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
                stats[symbol] = {
                    "30d_return": round((prices[-1] - prices[0]) / prices[0] * 100, 2),
                    "volatility": round(np.std(returns) * np.sqrt(252) * 100, 2) if returns else 0,
                    "current": prices[-1]
                }
        
        # Paper portfolio metrics with SPY comparison
        perf_log = DATA_DIR / "performance.jsonl"
        paper_metrics = {}
        spy_comparison = None
        if perf_log.exists():
            with open(perf_log) as f:
                tail_lines = deque(f, maxlen=500)
                if len(tail_lines) >= 20:
                    raw_entries = [json.loads(l) for l in tail_lines]

                    # performance.jsonl contains intraday entries; raw count
                    # overstates trading days and inflates Sharpe.
                    daily_entries = self._deduplicate_performance_entries_by_date(raw_entries)

                    daily_returns = [
                        e.get("daily_return", 0)
                        for e in daily_entries
                        if e.get("daily_return") is not None
                    ]
                    daily_values = [e.get("total_value", 0) for e in daily_entries]

                    if daily_returns and daily_values:
                        paper_metrics = {
                            "sharpe": round(np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252), 2) if np.std(daily_returns) > 0 else 0,
                            "total_return": round((daily_values[-1] - daily_values[0]) / daily_values[0] * 100, 2),
                            "max_value": round(max(daily_values), 2),
                            "min_value": round(min(daily_values), 2),
                            "days_tracked": len(daily_values)
                        }
                        
                        # Calculate SPY comparison if we have enough data
                        cursor.execute("""
                            SELECT date, close FROM prices 
                            WHERE symbol = 'SPY' 
                            AND date >= date('now', '-63 days')
                            ORDER BY date
                        """)
                        spy_rows = cursor.fetchall()
                        if len(spy_rows) >= 20 and len(daily_values) >= 20:
                            spy_prices = [r[1] for r in spy_rows[-len(daily_values):]]
                            spy_returns = [(spy_prices[i] - spy_prices[i-1]) / spy_prices[i-1]
                                          for i in range(1, len(spy_prices))]

                            # Calculate metrics
                            spy_total_return = (spy_prices[-1] - spy_prices[0]) / spy_prices[0]
                            portfolio_total_return = (daily_values[-1] - daily_values[0]) / daily_values[0]

                            # Correlation and Beta (30-day rolling)
                            min_len = min(len(daily_returns), len(spy_returns))
                            if min_len >= 20:
                                returns_arr = np.array(daily_returns[-20:])
                                spy_returns_arr = np.array(spy_returns[-20:])
                                
                                # Check for variance before calculating correlation
                                if np.std(returns_arr) > 0 and np.std(spy_returns_arr) > 0:
                                    corr = np.corrcoef(returns_arr, spy_returns_arr)[0,1]
                                    spy_vol = np.std(spy_returns_arr)
                                    if spy_vol > 0:
                                        beta = np.cov(returns_arr, spy_returns_arr)[0,1] / (spy_vol ** 2)
                                    else:
                                        beta = 1.0
                                else:
                                    corr = 0
                                    beta = 1.0
                            else:
                                corr = 0
                                beta = 1.0
                            
                            spy_comparison = {
                                "portfolio_value": round(daily_values[-1], 2),
                                "spy_value": round(daily_values[0] * (1 + spy_total_return), 2),
                                "relative_return": round((portfolio_total_return - spy_total_return) * 100, 2),
                                "correlation_30d": round(float(corr), 2),
                                "beta": round(float(beta), 2),
                                "outperformance": round((portfolio_total_return - spy_total_return) * 100, 2)
                            }
        
        output = {
            "asset_stats": stats,
            "paper_portfolio": paper_metrics,
            "spy_comparison": spy_comparison,
            "generated_at": datetime.now().isoformat()
        }
        
        out_path = PUBLIC_DIR / "stats.json"
        save_results_json(output, output_path=str(out_path))
        
        return out_path
    
    @staticmethod
    def _load_json_file(path: Path) -> Optional[Dict[str, Any]]:
        if not path.exists():
            return None
        with open(path) as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None

    @staticmethod
    def _has_open_blocking_incident(data_dir: Path) -> bool:
        for filename in ("incidents.json", "incident_state.json"):
            payload = DashboardGenerator._load_json_file(data_dir / filename)
            if not payload:
                continue
            raw_incidents = payload.get("incidents", payload.get("open_incidents", []))
            incidents = raw_incidents if isinstance(raw_incidents, list) else []
            for incident in incidents:
                if not isinstance(incident, dict):
                    continue
                status = str(incident.get("status", "open")).lower()
                blocking = bool(incident.get("blocking") or incident.get("blocks_promotion"))
                if blocking and status not in {"closed", "resolved", "pass"}:
                    return True
        return False

    @staticmethod
    def _promotion_gate_status(data_dir: Path) -> tuple[bool, list[str]]:
        kill_switch = DashboardGenerator._load_json_file(data_dir / "kill_switch.json")
        blockers: list[str] = []
        if kill_switch and kill_switch.get("enabled"):
            blockers.append("kill_switch")
        if DashboardGenerator._has_open_blocking_incident(data_dir):
            blockers.append("blocking_incident")

        try:
            from src.strategy.graduation_checklist import GraduationChecklist

            checklist = GraduationChecklist()
            results = checklist.check()
            manual = results.get("manual_approval")
            if manual is None or not manual.passed:
                blockers.append("manual_approval")
            if not checklist.is_graduation_ready(results):
                blockers.append("graduation_checklist")
        except SIGNAL_EXCEPTIONS:
            blockers.append("graduation_checklist_unavailable")

        return not blockers, blockers

    @staticmethod
    def _is_active_promote_candidacy(data: Dict[str, Any]) -> bool:
        """True only for live promote candidacy, not tombstones.

        GraduationChecklist rewrites ``.promote_to_live`` with
        ``action: promote_blocked_*`` when kill or checklist blocks.
        Those are not candidates — alerts must ignore them.
        """
        action = data.get("action")
        if action is None:
            # Legacy markers omit action; treat as candidacy.
            return True
        if not isinstance(action, str):
            return False
        if action == "promote_to_live":
            return True
        if action.startswith("promote_blocked"):
            return False
        # Unknown action strings are not live candidacy claims.
        return False

    @staticmethod
    def _graduation_candidate_alert(data_dir: Path) -> Optional[Dict[str, Any]]:
        data = DashboardGenerator._load_json_file(data_dir / ".promote_to_live")
        if not data:
            return None
        if not DashboardGenerator._is_active_promote_candidacy(data):
            return None

        allowed, blockers = DashboardGenerator._promotion_gate_status(data_dir)
        if allowed:
            return {
                "level": "success",
                "type": "graduation_candidate",
                "title": "Paper Trading Graduation Ready",
                "message": f"Sharpe: {safe_get(data, 'metrics', 'sharpe')}, ready for live approval",
                "timestamp": data.get("timestamp"),
                "requires_action": True,
            }
        return {
            "level": "warning",
            "type": "graduation_candidate",
            "title": "Paper Trading Graduation Blocked",
            "message": "Promotion marker present but current gates block live approval: "
            + ", ".join(sorted(set(blockers))),
            "timestamp": data.get("timestamp"),
            "requires_action": True,
        }

    @staticmethod
    def _stale_data_alerts_from_quality_report(public_dir: Path) -> Optional[List[Dict[str, Any]]]:
        report = DashboardGenerator._load_json_file(public_dir / "data_quality.json")
        if report is None:
            return None

        issue_counts = report.get("issue_counts")
        stale_count = (
            issue_counts.get("stale_latest_dates", 0)
            if isinstance(issue_counts, dict)
            else 0
        )
        if not isinstance(stale_count, int) or isinstance(stale_count, bool) or stale_count <= 0:
            return []

        alerts: List[Dict[str, Any]] = []
        symbols = report.get("symbols")
        rows = symbols if isinstance(symbols, list) else []
        stale_rows = [
            row for row in rows
            if isinstance(row, dict)
            and (
                row.get("stale_latest_date")
                or str(row.get("status", "")).lower() in {"fail", "failed", "critical"}
                or (isinstance(row.get("issue_counts"), dict)
                    and row["issue_counts"].get("stale_latest_dates", 0) > 0)
            )
        ]

        for row in stale_rows[:stale_count]:
            stale_meta = row.get("stale_latest_date") if isinstance(row.get("stale_latest_date"), dict) else {}
            symbol = row.get("symbol", "unknown")
            latest_date = stale_meta.get("latest_date") or row.get("latest_date", "unknown")
            reference_date = stale_meta.get("reference_date") or report.get("reference_date", "unknown")
            lag_days = stale_meta.get("latest_lag_days")
            lag = f" ({lag_days} trading day lag)" if isinstance(lag_days, int) else ""
            alerts.append({
                "level": "warning",
                "type": "stale_data",
                "title": f"Stale Data: {symbol}",
                "message": f"{symbol} latest date {latest_date} lags reference {reference_date}{lag}",
                "timestamp": report.get("generated_at"),
                "requires_action": False,
            })

        while len(alerts) < stale_count:
            alerts.append({
                "level": "warning",
                "type": "stale_data",
                "title": "Stale Data",
                "message": f"data_quality.json reports {stale_count} stale latest-date issue(s)",
                "timestamp": report.get("generated_at"),
                "requires_action": False,
            })
        return alerts

    def generate_alerts_json(self, health: Optional[Dict[str, Any]] = None) -> Path:
        """Generate active alerts and notifications.

        Args:
            health: Optional health.json payload. When omitted, loads
                ``PUBLIC_DIR/health.json`` if present. Prefer passing the
                in-process payload from ``generate_health_json`` / ``run`` so
                projection does not depend on filesystem ordering.
        """
        alerts = []

        promote_alert = self._graduation_candidate_alert(DATA_DIR)
        if promote_alert is not None:
            alerts.append(promote_alert)

        # Check for kill switch (SSOT: data/kill_switch.json)
        kill_payload = load_kill_switch_payload(DATA_DIR)
        kill_alert = build_kill_switch_alert(kill_payload) if kill_payload else None
        if kill_alert is not None:
            alerts.append(kill_alert)
        
        # Check for regime trigger
        regime_file = DATA_DIR / ".regime_trigger"
        if regime_file.exists():
            with open(regime_file) as f:
                data = json.load(f)
                alerts.append({
                    "level": "warning",
                    "type": "regime_change",
                    "title": f"Regime Change: {data.get('regime', 'unknown')}",
                    "message": f"VIX: {data.get('vix', 'N/A')}",
                    "timestamp": data.get("timestamp"),
                    "requires_action": False
                })
        
        quality_alerts = self._stale_data_alerts_from_quality_report(PUBLIC_DIR)
        if quality_alerts is not None:
            alerts.extend(quality_alerts)
        else:
            # Fallback for test/local environments that do not publish data_quality.json.
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT symbol, MAX(date) as last_date, COUNT(*) as count
                FROM prices GROUP BY symbol
            """)
            for row in cursor.fetchall():
                last_date = datetime.strptime(row[1], "%Y-%m-%d") if row[1] else None
                if last_date and (datetime.now() - last_date).days > 2:
                    alerts.append({
                        "level": "warning",
                        "type": "stale_data",
                        "title": f"Stale Data: {row[0]}",
                        "message": f"Last update: {row[1]} ({(datetime.now() - last_date).days} days ago)",
                        "requires_action": False
                    })

        health_payload = health if isinstance(health, dict) else self._load_json_file(
            PUBLIC_DIR / "health.json"
        )
        alerts.extend(build_health_slo_alerts(health_payload))
        
        output = {
            "alerts": sorted(alerts, key=lambda x: x.get("timestamp", "") or "", reverse=True),
            "count": len(alerts),
            "generated_at": datetime.now().isoformat()
        }
        
        out_path = PUBLIC_DIR / "alerts.json"
        save_results_json(output, output_path=str(out_path))

        # Post-write integrity: on-disk kill row must match data/kill_switch.json identity.
        # Concurrent/stale writers previously left LIVE/position_limit rows without incident_id.
        if kill_payload and kill_payload.get("enabled") and kill_alert is not None:
            try:
                on_disk = json.loads(out_path.read_text(encoding="utf-8"))
                disk_alerts = on_disk.get("alerts") if isinstance(on_disk, dict) else None
                disk_kills = [
                    a for a in (disk_alerts or [])
                    if isinstance(a, dict) and a.get("type") == "kill_switch"
                ]
                expected_id = kill_payload.get("incident_id")
                expected_reason = kill_payload.get("reason")
                expected_level = kill_payload.get("level")
                ok = False
                if disk_kills:
                    row = disk_kills[0]
                    ok = (
                        (expected_id is None or row.get("incident_id") == expected_id)
                        and (expected_reason is None or row.get("reason") == expected_reason)
                        and (
                            expected_level is None
                            or str(row.get("kill_switch_level") or "").lower()
                            == str(expected_level).lower()
                        )
                    )
                if not ok:
                    logger.error(
                        "alerts.json kill identity drift after write; rewriting from SSOT "
                        "(expected incident_id=%r reason=%r level=%r)",
                        expected_id,
                        expected_reason,
                        expected_level,
                    )
                    # Force SSOT kill row as sole kill_switch entry and rewrite.
                    rebuilt = [
                        a for a in output["alerts"]
                        if not (isinstance(a, dict) and a.get("type") == "kill_switch")
                    ]
                    rebuilt.insert(0, kill_alert)
                    output = {
                        "alerts": rebuilt,
                        "count": len(rebuilt),
                        "generated_at": datetime.now().isoformat(),
                    }
                    out_path.write_text(
                        json.dumps(output, indent=2, sort_keys=False) + "\n",
                        encoding="utf-8",
                    )
            except (OSError, json.JSONDecodeError, TypeError) as verify_exc:
                logger.error("alerts.json post-write kill verify failed: %s", verify_exc)
        
        return out_path

    @staticmethod
    def _empty_incident_summary() -> Dict[str, Any]:
        return {
            "schema_version": "incident-lifecycle/v1",
            "generated_at": datetime.now().isoformat(),
            "open_count": 0,
            "incidents": [],
            "metrics": {
                "incident_frequency": 0,
                "open_count": 0,
                "resolved_count": 0,
                "mean_mttr_seconds": None,
            },
        }

    def generate_incidents_json(self) -> Path:
        """Publish the incident lifecycle summary consumed by LiveDashboard."""
        try:
            payload = self._load_json_file(DATA_DIR / "incidents.json") or self._empty_incident_summary()
        except (OSError, json.JSONDecodeError, TypeError) as exc:
            logger.warning("Incident lifecycle summary unavailable; publishing empty summary: %s", exc)
            payload = self._empty_incident_summary()

        payload.setdefault("schema_version", "incident-lifecycle/v1")
        payload.setdefault("generated_at", datetime.now().isoformat())
        payload.setdefault("open_count", 0)
        payload.setdefault("incidents", [])
        payload.setdefault("metrics", {
            "incident_frequency": 0,
            "open_count": int(payload.get("open_count", 0) or 0),
            "resolved_count": 0,
            "mean_mttr_seconds": None,
        })

        out_path = PUBLIC_DIR / "incidents.json"
        save_results_json(payload, output_path=str(out_path))
        return out_path
    
    def generate_health_json(self) -> Path:
        """Generate system health status for dashboard."""
        
        health_data = {
            "cron_jobs": [],
            "data_freshness": {},
            "system_status": "healthy",
            "signal_health": {},
            "generated_at": datetime.now().isoformat()
        }
        
        # Cron job status from project-local status file and Hermes, when available.
        cron_section = build_cron_scheduler_section(
            cron_status_file=DATA_DIR / "cron_status.json",
            resolve_hermes_path=_resolve_hermes_cron_jobs_path,
            log_error=_log_signal_error,
        )
        health_data["cron_jobs"] = cron_section["cron_jobs"]
        health_data["scheduler_status"] = cron_section["scheduler_status"]
        
        # Per-symbol data freshness from SQLite
        freshness_section = build_data_freshness_section(conn=self.conn)
        health_data["data_freshness"] = freshness_section["data_freshness"]

        health_data["signal_health"] = build_signal_health_section(log_error=_log_signal_error)
        health_data["fred_readiness"] = build_fred_readiness_section(log_error=_log_signal_error)

        health_data["data_pipeline_slo"] = build_data_pipeline_slo_section(
            health_data=health_data,
            public_dir=PUBLIC_DIR,
            log_error=_log_signal_error,
        )

        # Kill authority + open incidents (same SSOT as data/health monitor)
        kill_fields = project_kill_switch_fields(load_kill_switch_payload(DATA_DIR))
        open_incidents = load_open_incidents_summary(DATA_DIR)
        health_data["kill_switch"] = kill_fields
        health_data["open_incidents"] = open_incidents

        # Overall system health
        # Exclude portfolio-lab-health self-errors so sticky tasker mirrors of
        # prior health exits cannot degrade dashboard system_status forever.
        from src.monitor.hermes_cron import rollup_failed_cron_jobs

        stale_count = summarize_stale_symbol_count(health_data["data_freshness"])
        failed_jobs = len(rollup_failed_cron_jobs(health_data["cron_jobs"]))
        scheduler_status = health_data.get("scheduler_status", {}).get("status")
        slo_status = health_data.get("data_pipeline_slo", {}).get("status")
        backend_error = any(
            backend.get("status") == "error"
            for backend in health_data.get("scheduler_status", {}).get("backends", {}).values()
        )

        health_data["system_status"] = derive_system_status(
            current=health_data.get("system_status", "healthy"),
            backend_error=backend_error,
            scheduler_status=scheduler_status,
            slo_status=slo_status,
            failed_jobs=failed_jobs,
            stale_count=stale_count,
        )
        health_data["system_status"] = elevate_system_status_for_kill(
            health_data["system_status"],
            kill_fields,
            open_incidents,
        )

        # Dual-SSOT: re-stamp ops_health_* from the monitor report so dashboard
        # regeneration does not wipe fields merged by make health /
        # publish_ops_health_surfaces (ops_health_status/source/timestamp).
        try:
            from src.monitor.health_check import apply_ops_monitor_to_dashboard_health

            apply_ops_monitor_to_dashboard_health(
                health_data,
                data_dir=DATA_DIR,
                public_dir=PUBLIC_DIR,
            )
        except Exception as exc:  # noqa: BLE001 — never block health.json write
            logger.warning("ops monitor merge into dashboard health failed: %s", exc)

        out_path = PUBLIC_DIR / "health.json"
        save_results_json(health_data, output_path=str(out_path))
        
        return out_path
    
    def _generate_sector_momentum_signals(self, vix_level: Optional[float] = None) -> Optional[Dict]:
        """Generate sector rotation momentum signals from historical data."""
        try:
            from src.strategy.sector_momentum_calc import generate_sector_signals

            historical_path = PUBLIC_DIR.parent / "data" / "historical.json"

            # Use provided VIX level (avoid re-querying DB)
            vix = vix_level if vix_level is not None else 0

            signals = generate_sector_signals(historical_path, vix=vix)
            return signals

        except SIGNAL_EXCEPTIONS as e:
            _log_signal_error("sector_momentum", e)
            return None
    
    def generate_analytics_json(self) -> Path:
        """Generate analytics data (drawdown, rolling metrics, benchmarks)."""
        # Import analytics calculator
        try:
            from src.analytics.calculator import AnalyticsCalculator
            calc = AnalyticsCalculator(data_dir=str(DATA_DIR))
            report = calc.generate_analytics_report()
            
            out_path = PUBLIC_DIR / "analytics.json"
            save_results_json(report, output_path=str(out_path))

            return out_path
        except (OSError, json.JSONDecodeError, KeyError, ValueError, TypeError, ImportError, RuntimeError) as e:
            # Fallback: empty analytics
            report = {
                "status": "error",
                "message": str(e),
                "generated_at": datetime.now().isoformat(),
            }
            out_path = PUBLIC_DIR / "analytics.json"
            save_results_json(report, output_path=str(out_path))
            return out_path
    
    def generate_overlay_json(self) -> Optional[Path]:
        """Generate overlay dashboard data from all tactical overlays."""
        try:
            from src.dashboard.overlay_dashboard import OverlayDashboardGenerator
            gen = OverlayDashboardGenerator()
            dashboard = gen.generate()
            gen.save(dashboard)
            public_path = PUBLIC_DIR / gen.OUTPUT_PATH.name
            save_results_json(dashboard.to_dict(), output_path=str(public_path))
            return public_path
        except SIGNAL_EXCEPTIONS as e:
            _log_signal_error("overlay_dashboard_generate", e)
            return None

    def generate_adaptive_sizing_json(self) -> Optional[Path]:
        """Generate adaptive sizing data for dashboard."""
        try:
            from src.strategy.adaptive_sizing import AdaptiveSizer

            sizer = AdaptiveSizer()
            decision = sizer.compute_allocation()

            authority = self._build_advisory_allocation_artifact_role(
                surface="adaptive_sizing",
                allocation_field="adjusted_allocation",
            )
            sizing_data = {
                "base_allocation": decision.base_allocation,
                "adjusted_allocation": decision.adjusted_allocation,
                "adjustments": decision.adjustments,
                "regime_adjustment": decision.regime_adjustment,
                "volatility_adjustment": decision.volatility_adjustment,
                "signal_adjustment": decision.signal_adjustment,
                "drawdown_adjustment": decision.drawdown_adjustment,
                "factors": asdict(decision.factors) if hasattr(decision.factors, '__dataclass_fields__') else {},
                "authority": authority,
                **self._flatten_advisory_authority(authority),
                "generated_at": datetime.now().isoformat(),
            }

            out_path = PUBLIC_DIR / "adaptive_sizing.json"
            save_results_json(sizing_data, output_path=str(out_path))
            return out_path

        except SIGNAL_EXCEPTIONS as e:
            _log_signal_error("adaptive_sizing", e)
            return None

    def generate_vixy_hedge_json(self) -> Optional[Path]:
        """Generate VIXY hedge sizing data for dashboard."""
        try:
            from src.strategy.vixy_hedge_sizing import VIXYHedgeSizer

            sizer = VIXYHedgeSizer()
            status = sizer.status()

            hedge_data = {
                **status,
                "generated_at": datetime.now().isoformat(),
                "canonical_controller": "hedge_selector",
                "runtime_role": "diagnostic_cost_evidence",
                "live_authoritative": False,
                "routed": False,
            }

            out_path = PUBLIC_DIR / "vixy_hedge.json"
            save_results_json(hedge_data, output_path=str(out_path))
            return out_path

        except SIGNAL_EXCEPTIONS as e:
            _log_signal_error("vixy_hedge", e)
            return None

    def generate_black_litterman_json(self) -> Optional[Path]:
        """Generate Black-Litterman mapper data for dashboard.

        Uses live ensemble voter signal data to generate real BL views
        with Idzorek confidence from signal health scores.
        """
        try:
            from src.strategy.ensemble_voter import EnsembleVoter
            from src.strategy.black_litterman_mapper import run_black_litterman

            # Get live BL views from ensemble voter
            voter = EnsembleVoter()
            bl_input = voter.get_bl_views()

            views = bl_input["views"]

            # Compute covariance matrix from price data in market.db
            symbols = list(views.symbols) if hasattr(views, 'symbols') else ['SPY', 'TLT', 'GLD']
            import pandas as pd
            price_data = {}
            for sym in symbols:
                try:
                    rows = self.conn.execute(
                        "SELECT date, close FROM prices WHERE symbol = ? ORDER BY date DESC LIMIT 252",
                        (sym,)
                    ).fetchall()
                    if rows:
                        price_data[sym] = pd.Series(
                            {r[0]: r[1] for r in reversed(rows)}
                        )
                except (sqlite3.OperationalError, KeyError):
                    continue

            # Try full BL optimization (requires PyPortfolioOpt)
            bl_weights = None
            posterior_returns = None
            result = None
            if len(price_data) >= 2:
                try:
                    prices_df = pd.DataFrame(price_data)
                    returns = prices_df.pct_change().dropna()
                    available = [s for s in symbols if s in returns.columns]
                    cov_matrix = returns[available].cov().values * 252  # Annualized
                    result = run_black_litterman(cov_matrix, views)
                    bl_weights = result.bl_weights
                    posterior_returns = result.posterior_returns
                except (ImportError, ValueError) as e:
                    logger.info("BL optimization unavailable (%s), using views-only output", e)

            # Fallback: use views without full BL optimization
            if bl_weights is None:
                bl_weights = dict(BASE_ALLOCATION)
            if posterior_returns is None:
                posterior_returns = {}

            # Use base allocation as prior
            prior = dict(BASE_ALLOCATION)
            canonical_assets = tuple(str(symbol).upper() for symbol in symbols)
            prior_public = self._canonicalize_public_weights(prior, canonical_assets=canonical_assets)
            posterior_public = self._canonicalize_public_weights(bl_weights, canonical_assets=canonical_assets)
            returns_public = self._canonicalize_public_weights(posterior_returns, canonical_assets=canonical_assets)

            # Build views list for panel consumption
            view_list = []
            if hasattr(views, 'absolute_views') and views.absolute_views:
                abs_views = views.absolute_views if isinstance(views.absolute_views, dict) else dict(zip(views.symbols, views.absolute_views))
                for i, sym in enumerate(views.symbols):
                    ret = abs_views.get(sym, 0.0)
                    conf = views.view_confidences[i] if i < len(views.view_confidences) else 0.5
                    view_list.append({
                        "signal_name": "ensemble_consensus",
                        "asset": str(sym).upper(),
                        "direction": "bullish" if ret > 0 else ("bearish" if ret < 0 else "neutral"),
                        "confidence": round(conf, 3),
                        "expected_return_delta": round(ret, 6),
                    })

            authority = self._build_advisory_allocation_artifact_role(
                surface="black_litterman",
                allocation_field="posterior_weights",
            )
            bl_data = {
                "prior_weights": prior_public["weights"],
                "posterior_weights": posterior_public["weights"],
                "posterior_returns": returns_public["weights"],
                "views": view_list,
                "tau": bl_input.get("tau", 0.15),
                "view_confidence_method": "idzorek",
                "optimization_available": result is not None,
                "excluded_assets": sorted(set(
                    prior_public["excluded_assets"]
                    + posterior_public["excluded_assets"]
                    + returns_public["excluded_assets"]
                )),
                "zero_weight_assets": sorted(set(
                    prior_public["zero_weight_assets"]
                    + posterior_public["zero_weight_assets"]
                )),
                "authority": authority,
                **self._flatten_advisory_authority(authority),
                "health_scores": bl_input.get("health_scores_used", {}),
                "biases": {
                    "equity": round(bl_input.get("equity_bias", 0.0), 3),
                    "duration": round(bl_input.get("duration_bias", 0.0), 3),
                    "gold": round(bl_input.get("gold_bias", 0.0), 3),
                },
                "generated_at": datetime.now().isoformat(),
            }

            out_path = PUBLIC_DIR / "black_litterman.json"
            save_results_json(bl_data, output_path=str(out_path))
            return out_path

        except SIGNAL_EXCEPTIONS as e:
            _log_signal_error("black_litterman", e)
            return None

    def generate_turnover_validator_json(self) -> Optional[Path]:
        """Generate turnover validator data for dashboard."""
        try:
            from src.signals.signal_source import SignalSource
            from src.strategy.turnover_validator import TurnoverValidator

            validator = TurnoverValidator()
            diagnostics = validator.get_state_diagnostics()
            canonical_sources = {source.value for source in SignalSource}
            production_signals = {
                source: data
                for source, data in diagnostics.items()
                if source in canonical_sources
            }
            synthetic_baselines = {
                source: {
                    "metadata": {"source_type": "synthetic_or_fixture"},
                    "diagnostics": data,
                }
                for source, data in diagnostics.items()
                if source not in canonical_sources
            }

            turnover_data = {
                "schema_version": "turnover-validator/v1",
                "signals": production_signals,
                "synthetic_baselines": synthetic_baselines,
                "generated_at": datetime.now().isoformat(),
            }

            out_path = PUBLIC_DIR / "turnover_validator.json"
            save_results_json(turnover_data, output_path=str(out_path))
            return out_path

        except SIGNAL_EXCEPTIONS as e:
            _log_signal_error("turnover_validator", e)
            return None

    def generate_regime_gate_json(self) -> Optional[Path]:
        """Generate regime gate status data for dashboard."""
        try:
            from src.signals.regime_gate import RegimeGate

            gate = RegimeGate()
            summary = gate.get_gate_summary()

            # Load current regime from state
            regime_name = "NORMAL"
            regime_confidence = 0.5
            regime_file = DATA_DIR / "regime_state.json"
            if regime_file.exists():
                with open(regime_file) as f:
                    state = json.load(f)
                    regime_name = state.get("regime", "NORMAL")
                    regime_confidence = state.get("confidence", 0.5)

            # Build gate rules with current regime active status
            all_signals = self._dedupe_preserve_order(
                list(summary.keys()) + ["alt_data", "cross_asset_rv", "unified_overlay"],
            )
            active_signals = self._dedupe_preserve_order(gate.get_active_signal_names(
                all_signals,
                regime_name,
            ))
            inactive_signals = [s for s in all_signals if s not in active_signals]

            gate_data = {
                "current_regime": regime_name,
                "regime_confidence": regime_confidence,
                "gate_rules": [
                    {"signal_name": sig, "off_regimes": sorted(regimes), "is_active": sig in active_signals}
                    for sig, regimes in summary.items()
                ],
                "active_signals": active_signals,
                "inactive_signals": inactive_signals,
                "min_dwell_days": gate.min_dwell_days,
                "generated_at": datetime.now().isoformat(),
            }

            # Compute data-driven regime Sharpe matrix (read-only, for monitoring)
            try:
                from src.monitor.regime_sharpe_matrix import (
                    compute_regime_sharpe_matrix,
                    derive_gate_rules,
                    derive_regime_weight_multipliers,
                    extract_signal_regime_data,
                )

                prices = self._load_price_data()
                db_path = DATA_DIR / "ensemble_signals.db"
                if prices is not None and db_path.exists():
                    hist_df = extract_signal_regime_data(db_path, prices)
                    if not hist_df.empty:
                        matrix = compute_regime_sharpe_matrix(
                            hist_df, n_bootstrap=500, seed=42,
                        )
                        data_rules = derive_gate_rules(matrix)
                        data_multipliers = derive_regime_weight_multipliers(matrix)

                        gate_data["data_driven"] = {
                            "gate_rules": {
                                sig: sorted(regimes)
                                for sig, regimes in data_rules.items()
                            },
                            "weight_multipliers": data_multipliers,
                            "n_observations": len(hist_df),
                            "n_signals": hist_df["signal"].nunique(),
                        }

                        # Persist data-driven rules for EnsembleVoter to load
                        persist_path = DATA_DIR / "regime_gate_persisted.json"
                        persist_data = {
                            "gate_rules": {
                                sig: sorted(regimes)
                                for sig, regimes in data_rules.items()
                            },
                            "weight_multipliers": data_multipliers,
                            "computed_at": datetime.now().isoformat(),
                            "n_observations": len(hist_df),
                        }
                        save_results_json(persist_data, output_path=str(persist_path))
            except (ImportError, Exception) as e:
                logger.debug("Regime Sharpe matrix computation skipped: %s", e)

            out_path = PUBLIC_DIR / "regime_gate.json"
            save_results_json(gate_data, output_path=str(out_path))
            return out_path

        except SIGNAL_EXCEPTIONS as e:
            _log_signal_error("regime_gate", e)
            return None

    # Cache last-known regime for _is_msm_gated resilience
    _last_regime: str = "normal"

    def _is_msm_gated(self) -> bool:
        """Check if MSM should be gated off based on current regime.

        MSM has zero ensemble weight in HIGH_VOL/CRISIS regimes (health 0.55,
        net-negative -0.012 Sharpe). Returns True when gated off.

        On transient query failures, uses the last-known regime instead of
        immediately gating MSM off — a single SQLite hiccup should not
        disable a strategy.
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT regime FROM regime_log ORDER BY detected_at DESC LIMIT 1")
            row = cursor.fetchone()
            regime = row[0] if row else "normal"
            DashboardGenerator._last_regime = regime
            return regime.lower() in {"high_vol", "crisis"}
        except Exception as e:
            logger.warning("_is_msm_gated: regime query failed (%s) — using last-known regime '%s'",
                           e, DashboardGenerator._last_regime)
            return DashboardGenerator._last_regime.lower() in {"high_vol", "crisis"}

    @staticmethod
    def _extract_vix_term_structure_signal(vix_term_structure: Optional[Dict]) -> Optional[float]:
        """Extract the fractional VIX term-structure signal when available."""
        if not isinstance(vix_term_structure, dict):
            return None
        raw_signal = vix_term_structure.get("signal_value")
        if raw_signal is None:
            return None
        try:
            return float(raw_signal)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _resolve_hedge_vix_level(
        vix_level: Optional[float],
        vix_term_structure: Optional[Dict],
    ) -> Optional[float]:
        """Use the VIX term-structure spot value when market.db lacks ^VIX."""
        if vix_level is not None:
            return vix_level
        if not isinstance(vix_term_structure, dict):
            return None
        raw_vix = vix_term_structure.get("vix_spot")
        if raw_vix is None:
            return None
        try:
            return float(raw_vix)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _build_unavailable_hedge_selector_signal(
        regime: str,
        term_structure_signal: Optional[float],
    ) -> Dict[str, Any]:
        """Publish a typed canonical hedge-selector artifact when VIX is unavailable."""
        return {
            "available": False,
            "generated_at": datetime.now().isoformat(),
            "regime": regime,
            "regime_confidence": 0.0,
            "primary_hedge": "none",
            "primary_size_pct": 0.0,
            "secondary_hedge": None,
            "secondary_size_pct": 0.0,
            "cost_benefit_gate": False,
            "net_benefit_bps": 0.0,
            "kelly_fraction": 0.0,
            "expected_cost_bps": 0.0,
            "expected_benefit_bps": 0.0,
            "min_hold_days": 0,
            "transition_cost_bps": 0.0,
            "canonical_controller": "hedge_selector",
            "vixy_role": "diagnostic_sizing_helper",
            "term_structure_role": "gate_discount_multiplier",
            "term_structure_gate": False,
            "term_structure_multiplier": 0.0,
            "term_structure_signal": term_structure_signal,
            "gate_reason": "vix_unavailable",
        }

    def _get_hedge_selector_signal(
        self,
        vix_level: Optional[float],
        regime: str,
        vix_term_structure: Optional[Dict] = None,
    ) -> Optional[Dict]:
        """Get hedge selector recommendation for dashboard."""
        term_structure_signal = self._extract_vix_term_structure_signal(vix_term_structure)
        hedge_vix_level = self._resolve_hedge_vix_level(vix_level, vix_term_structure)
        if hedge_vix_level is None:
            return self._build_unavailable_hedge_selector_signal(
                regime,
                term_structure_signal,
            )
        try:
            from src.strategy.hedge_selector import HedgeSelector
            selector = HedgeSelector()
            # Estimate confidence based on regime stability
            regime_confidence = 0.8 if regime in ["normal", "crisis"] else 0.6
            rec = selector.select(
                vix_level=hedge_vix_level,
                regime_confidence=regime_confidence,
                regime_label=regime,
                term_structure_signal=term_structure_signal,
            )
            return {
                "available": True,
                "generated_at": datetime.now().isoformat(),
                "regime": rec.regime,
                "regime_confidence": rec.regime_confidence,
                "primary_hedge": rec.primary_hedge,
                "primary_size_pct": rec.primary_size_pct,
                "secondary_hedge": rec.secondary_hedge,
                "secondary_size_pct": rec.secondary_size_pct,
                "cost_benefit_gate": rec.cost_benefit_gate,
                "net_benefit_bps": rec.net_benefit_bps,
                "kelly_fraction": rec.kelly_fraction,
                "expected_cost_bps": rec.expected_cost_bps,
                "expected_benefit_bps": rec.expected_benefit_bps,
                "min_hold_days": rec.min_hold_days,
                "transition_cost_bps": rec.transition_cost_bps,
                "canonical_controller": rec.canonical_controller,
                "vixy_role": rec.vixy_role,
                "term_structure_role": rec.term_structure_role,
                "term_structure_gate": rec.term_structure_gate,
                "term_structure_multiplier": rec.term_structure_multiplier,
                "term_structure_signal": term_structure_signal,
                "gate_reason": rec.gate_reason,
            }
        except SIGNAL_EXCEPTIONS as e:
            _log_signal_error("hedge_selector", e)
            return None

    # Signal staleness detection (production readiness)
    SIGNAL_STALENESS_TTL_HOURS = int(os.environ.get("SIGNAL_STALENESS_TTL_HOURS", "4"))
    STALENESS_DECAY_TAU_HOURS = float(os.environ.get("STALENESS_DECAY_TAU_HOURS", "2.0"))
    OPTIONAL_SIGNAL_STALENESS_KEYS = {
        "behavioral_sentiment",
        "calendar_seasonality",
        "crypto_allocation",
        "factor_rotation",
        "stacking_ensemble",
        "convexity_harvest",
        "llm_sentiment",
        "sector_rotation",
        "kurtosis_regime",
        "volatility_parity",
        "collar",
        "bond_momentum",
        "risk_decomposition",
        "two_stage_regime",
        "bocd_regime",
        "regime_transition",
        "hedge_selector",
        "fred_macro",
    }
    OPTIONAL_DAILY_SIGNAL_STALENESS_KEYS = {
        "convexity_harvest",
        "volatility_parity",
    }

    @staticmethod
    def _normalized_signal_timestamp(
        signal_block: Any,
        preferred_field: str,
        *,
        allow_date: bool = False,
    ) -> str | None:
        """Return the first usable timestamp from a generated signal block."""
        if not isinstance(signal_block, dict):
            return None
        fields = [
            preferred_field,
            "generated_at",
            "timestamp",
            "generated",
            "detected",
            "last_update",
        ]
        for field in dict.fromkeys(fields):
            value = signal_block.get(field)
            if isinstance(value, str) and value:
                return value
        if allow_date:
            value = signal_block.get("date")
            if isinstance(value, str) and value:
                try:
                    parsed_date = datetime.strptime(value, "%Y-%m-%d").date()
                    # Daily sections remain fresh through their UTC calendar day.
                    return datetime.combine(
                        parsed_date,
                        datetime.max.time().replace(microsecond=0),
                        tzinfo=timezone.utc,
                    ).isoformat()
                except ValueError:
                    return value
        return None

    @staticmethod
    def _is_unavailable_signal_block(signal_block: Any) -> bool:
        """Return true for explicit unavailable/error placeholders."""
        if signal_block is None:
            return True
        if not isinstance(signal_block, dict):
            return False
        status = str(signal_block.get("status", "")).lower()
        if status in {"unavailable", "disabled", "missing"}:
            return True
        source_mode = str(signal_block.get("source_mode", "")).lower()
        if source_mode in {"unavailable", "synthetic", "last_good", "fallback"}:
            return True
        cache_status = str(signal_block.get("cache_status", "")).lower()
        if cache_status in {"unavailable", "empty", "missing", "failed", "degraded"}:
            return True
        return "error" in signal_block

    def _check_signal_staleness(self, signal_data: Dict) -> Dict:
        """Check staleness of each signal source in signals.json output.

        Compares each signal's `generated_at` / `timestamp` field against a TTL
        (default 4 hours). Stale signals should be removed from ensemble weight
        numerator/denominator (not zeroed — zeroing distorts relative weights).

        Also computes per-signal staleness decay factors for ensemble weight
        adjustment. Decay uses exponential: weight *= exp(-age_hours / tau)
        where tau defaults to 2h (STALENESS_DECAY_TAU_HOURS env var).

        Returns:
            Dict with keys:
            - stale_signals: list of signal names that are stale
            - signal_timestamps: dict of signal_name -> last_known_timestamp
            - signal_age_hours: dict of signal_name -> age in hours (None if missing)
            - staleness_decay: dict of signal_name -> decay factor (0.0-1.0)
            - healthy_count: number of fresh signals
            - total_count: total number of signals checked
        """
        import math as _math

        ttl_seconds = self.SIGNAL_STALENESS_TTL_HOURS * 3600
        tau_hours = self.STALENESS_DECAY_TAU_HOURS
        now = datetime.now(timezone.utc)
        stale_signals = []
        unavailable_signals = []
        signal_timestamps = {}
        signal_age_hours = {}
        staleness_decay = {}

        # Known signal keys in signals.json that have timestamps
        timestamped_signals = {
            "ensemble_voting": ("generated_at", None),
            "alternative_data": ("timestamp", None),
            "behavioral_sentiment": ("timestamp", None),
            "garch_cvar": ("timestamp", None),
            "smart_rebalance": ("generated_at", None),
            "calendar_seasonality": ("generated_at", None),
            "crypto_allocation": ("generated_at", None),
            "factor_rotation": ("generated_at", None),
            "stacking_ensemble": ("generated_at", None),
            "convexity_harvest": ("generated_at", None),
            "llm_sentiment": ("generated_at", None),
            "sector_rotation": ("generated_at", None),
            "kurtosis_regime": ("generated_at", None),
            "volatility_parity": ("generated_at", None),
            "collar": ("generated_at", None),
            "bond_momentum": ("generated_at", None),
            "risk_decomposition": ("generated_at", None),
            "rebalance_health": ("generated_at", None),
            "two_stage_regime": ("timestamp", None),
            "bocd_regime": ("timestamp", None),
            "regime_transition": ("timestamp", None),
            "hedge_selector": ("generated_at", None),
            "fred_macro": ("timestamp", None),
        }

        for signal_key, (ts_field, _) in timestamped_signals.items():
            signal_block = signal_data.get(signal_key)
            if signal_block is None:
                if signal_key in self.OPTIONAL_SIGNAL_STALENESS_KEYS:
                    unavailable_signals.append(signal_key)
                    signal_timestamps[signal_key] = None
                    signal_age_hours[signal_key] = None
                    staleness_decay[signal_key] = 0.0
                    continue
                stale_signals.append(signal_key)
                signal_timestamps[signal_key] = None
                signal_age_hours[signal_key] = None
                staleness_decay[signal_key] = 0.0
                continue

            is_optional = signal_key in self.OPTIONAL_SIGNAL_STALENESS_KEYS
            if self._is_unavailable_signal_block(signal_block):
                unavailable_signals.append(signal_key)
                signal_timestamps[signal_key] = None
                signal_age_hours[signal_key] = None
                staleness_decay[signal_key] = 0.0
                continue

            ts_str = self._normalized_signal_timestamp(
                signal_block,
                ts_field,
                allow_date=signal_key in self.OPTIONAL_DAILY_SIGNAL_STALENESS_KEYS,
            )
            if ts_str is None and not is_optional and isinstance(signal_block, dict):
                artifact_ts = signal_data.get("generated_at") or signal_data.get("timestamp")
                if isinstance(artifact_ts, str) and artifact_ts:
                    ts_str = artifact_ts
            signal_timestamps[signal_key] = ts_str

            if ts_str is None:
                if is_optional:
                    unavailable_signals.append(signal_key)
                    signal_age_hours[signal_key] = None
                    staleness_decay[signal_key] = 0.0
                    continue
                stale_signals.append(signal_key)
                signal_age_hours[signal_key] = None
                staleness_decay[signal_key] = 0.0
                continue

            try:
                # Parse ISO timestamp — handle both Z and +00:00 suffixes
                ts_str_clean = ts_str.replace("Z", "+00:00")
                ts = datetime.fromisoformat(ts_str_clean)
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                else:
                    ts = ts.astimezone(timezone.utc)
                age_seconds = max((now - ts).total_seconds(), 0.0)
                age_hours = age_seconds / 3600.0
                signal_age_hours[signal_key] = round(age_hours, 2)

                # Exponential decay: fresh signals get 1.0, stale signals approach 0.0
                decay = _math.exp(-age_hours / tau_hours) if tau_hours > 0 else 1.0
                decay = min(max(decay, 0.0), 1.0)
                staleness_decay[signal_key] = round(decay, 4)

                if age_seconds > ttl_seconds:
                    stale_signals.append(signal_key)
            except (ValueError, TypeError):
                stale_signals.append(signal_key)
                signal_age_hours[signal_key] = None
                staleness_decay[signal_key] = 0.0

        # Producer-aware override for alternative_data: do not escalate kill on
        # projection lag when alternative_data_latest.json is still fresh.
        projection_lag_signals: list[str] = []
        producer_ts = load_alternative_data_producer_timestamp(DATA_DIR)
        if producer_ts and "alternative_data" in timestamped_signals:
            try:
                pts = datetime.fromisoformat(producer_ts.replace("Z", "+00:00"))
                if pts.tzinfo is None:
                    pts = pts.replace(tzinfo=timezone.utc)
                else:
                    pts = pts.astimezone(timezone.utc)
                producer_age_hours = max((now - pts).total_seconds(), 0.0) / 3600.0
                producer_fresh = producer_age_hours * 3600.0 <= ttl_seconds
                projected_ts = signal_timestamps.get("alternative_data")
                projected_stale = "alternative_data" in stale_signals
                producer_ahead = False
                if projected_ts:
                    try:
                        ets = datetime.fromisoformat(str(projected_ts).replace("Z", "+00:00"))
                        if ets.tzinfo is None:
                            ets = ets.replace(tzinfo=timezone.utc)
                        else:
                            ets = ets.astimezone(timezone.utc)
                        producer_ahead = pts > ets
                    except (ValueError, TypeError):
                        producer_ahead = True
                else:
                    producer_ahead = True

                if producer_fresh and (projected_stale or producer_ahead):
                    if projected_stale and "alternative_data" in stale_signals:
                        stale_signals = [s for s in stale_signals if s != "alternative_data"]
                    if producer_ahead:
                        projection_lag_signals.append("alternative_data")
                    # Prefer producer timestamp / age for operator honesty
                    signal_timestamps["alternative_data"] = producer_ts
                    signal_age_hours["alternative_data"] = round(producer_age_hours, 2)
                    decay = (
                        _math.exp(-producer_age_hours / tau_hours) if tau_hours > 0 else 1.0
                    )
                    staleness_decay["alternative_data"] = round(
                        min(max(decay, 0.0), 1.0), 4
                    )
            except (ValueError, TypeError):
                pass

        healthy_count = len(timestamped_signals) - len(stale_signals) - len(unavailable_signals)
        return {
            "stale_signals": stale_signals,
            "unavailable_signals": unavailable_signals,
            "projection_lag_signals": projection_lag_signals,
            "signal_timestamps": signal_timestamps,
            "signal_age_hours": signal_age_hours,
            "staleness_decay": staleness_decay,
            "decay_tau_hours": tau_hours,
            "healthy_count": healthy_count,
            "total_count": len(timestamped_signals),
            "required_count": len(timestamped_signals) - len(self.OPTIONAL_SIGNAL_STALENESS_KEYS),
            "optional_count": len(self.OPTIONAL_SIGNAL_STALENESS_KEYS),
            "ttl_hours": self.SIGNAL_STALENESS_TTL_HOURS,
            "checked_at": now.isoformat(),
        }

    def _apply_staleness_decay(self, output: Dict) -> Dict:
        """Apply staleness-weighted decay to ensemble voting weights.

        When signals are stale, their ensemble weights degrade proportionally
        using exponential decay. This ensures the dashboard and downstream
        consumers reflect signal freshness in allocation decisions.

        Decay formula: adjusted_weight = raw_weight * exp(-age_hours / tau)
        where tau = STALENESS_DECAY_TAU_HOURS (default 2h).
        """
        staleness = output.get("staleness", {})
        decay_factors = staleness.get("staleness_decay", {})
        if not decay_factors:
            return output

        # Apply decay to ensemble_voting source_breakdown weights
        ensemble = output.get("ensemble_voting")
        if isinstance(ensemble, dict) and "source_breakdown" in ensemble:
            for src in ensemble["source_breakdown"]:
                source_name = src.get("source", "")
                # Map ensemble source names to staleness signal keys
                staleness_key = _ENSEMBLE_STALENESS_MAP.get(source_name)
                if staleness_key and staleness_key in decay_factors:
                    decay = decay_factors[staleness_key]
                    original_weight = src.get("weight", 0.0)
                    src["weight_original"] = original_weight
                    src["weight"] = round(original_weight * decay, 4)
                    src["staleness_decay"] = decay

            # Recompute weighted_consensus with decayed weights
            valid_sources = []
            for source in ensemble["source_breakdown"]:
                try:
                    value = float(source.get("value", 0.0))
                    weight = float(source.get("weight", 0.0))
                except (TypeError, ValueError):
                    continue
                if np.isnan(value) or np.isnan(weight):
                    continue
                valid_sources.append((value, weight))

            total_weight = sum(weight for _, weight in valid_sources)
            if total_weight > 0:
                weighted_consensus = sum(
                    value * weight for value, weight in valid_sources
                ) / total_weight
                agreement_weight = sum(
                    weight for value, weight in valid_sources
                    if np.sign(value) == np.sign(weighted_consensus) or abs(value) < 0.1
                )
                ensemble["weighted_consensus"] = round(weighted_consensus, 4)
                ensemble["agreement_ratio"] = round(agreement_weight / total_weight, 4)
                ensemble["total_weight_after_decay"] = round(total_weight, 4)

            ensemble.update(self._build_ensemble_source_count_metadata(
                ensemble.get("regime", "normal"),
                ensemble["source_breakdown"],
            ))
            ensemble["configured_source_status"] = self._build_configured_source_status(
                ensemble.get("regime", "normal"),
                ensemble["source_breakdown"],
            )

        return output

    def _run_spc_monitor(self, output: Dict) -> Dict:
        """Run SPC monitoring on signal values.

        Tracks rolling statistics of signal values and flags signals whose
        distribution has shifted (3-sigma breach for 3+ consecutive periods).
        """
        try:
            from src.monitor.spc_monitor import SPCMonitor
        except (ImportError, AttributeError) as e:
            logger.warning("SPC monitor not available: %s", e)
            return {"status": "unavailable", "error": str(e)}

        # Initialize class-level SPC monitor (persists across runs)
        if DashboardGenerator._spc_monitor is None:
            DashboardGenerator._spc_monitor = SPCMonitor()
            DashboardGenerator._spc_monitor.load_state()

        monitor = DashboardGenerator._spc_monitor

        # Record current signal values for SPC tracking
        ensemble = output.get("ensemble_voting")
        if isinstance(ensemble, dict) and "source_breakdown" in ensemble:
            for src in ensemble["source_breakdown"]:
                source_name = src.get("source", "")
                value = src.get("value")
                if source_name and value is not None:
                    try:
                        monitor.record(source_name, float(value))
                    except (ValueError, TypeError):
                        pass

        # Also track key aggregate metrics
        if isinstance(ensemble, dict):
            consensus = ensemble.get("weighted_consensus")
            if consensus is not None:
                try:
                    monitor.record("_ensemble_consensus", float(consensus))
                except (ValueError, TypeError):
                    pass

        # Get status
        flags = monitor.check_flags()
        all_status = monitor.get_all_status()

        # Persist state for next process invocation
        monitor.save_state()

        return {
            "status": "ok",
            "flagged_signals": flags,
            "signal_status": all_status,
            "window_size": monitor.window_size,
            "sigma_threshold": monitor.sigma_threshold,
            "consecutive_breach_limit": monitor.consecutive_breach_limit,
        }

    def generate_tsmom_json(self) -> Optional[Path]:
        """Generate TSMOM overlay data for dashboard."""
        try:
            from src.signals.tsmom_overlay import TSMOMOverlay

            overlay = TSMOMOverlay()
            tickers = ['SPY', 'GLD', 'TLT']
            signals = []
            for ticker in tickers:
                sig = overlay.compute_signal(ticker)
                if sig is not None:
                    signals.append(sig)

            # Build speed breakdown from multi-speed TSMOM
            speed_breakdown = []
            for signal in signals:
                speed_breakdown.append({
                    "label": f"{signal.ticker} TSMOM",
                    "weight": signal.base_weight,
                    "signal": signal.signal,
                    "asset_signals": {signal.ticker: signal.adjustment},
                    "realized_vol": signal.realized_vol,
                    "adjustment": signal.adjustment,
                })

            tsmom_data = {
                "composite_signal": float(np.mean([s.signal for s in signals])) if signals else 0.0,
                "speed_breakdown": speed_breakdown,
                "position_recommendation": "long" if np.mean([s.signal for s in signals]) > 0.1 else ("short" if np.mean([s.signal for s in signals]) < -0.1 else "neutral") if signals else "neutral",
                "confidence": min(1.0, abs(np.mean([s.vol_scaled_position for s in signals]))) if signals else 0.0,
                "standalone_sharpe": 0.96,
                "overlay_sharpe": 0.93,
                "health_score": 0.55,
                "is_gated_off": self._is_msm_gated(),
                "generated_at": datetime.now().isoformat(),
            }

            out_path = PUBLIC_DIR / "tsmom.json"
            save_results_json(tsmom_data, output_path=str(out_path))
            return out_path

        except SIGNAL_EXCEPTIONS as e:
            _log_signal_error("tsmom", e)
            return None

    def generate_cross_asset_rv_json(self) -> Optional[Path]:
        """Generate cross-asset relative value data for dashboard."""
        try:
            from src.signals.cross_asset_relative_value import CrossAssetRVScanner

            scanner = CrossAssetRVScanner()
            signal = scanner.scan_all()

            # Determine gating from current regime (v961: RV fails in HIGH_VOL/CRISIS)
            cursor = self.conn.cursor()
            cursor.execute("SELECT regime FROM regime_log ORDER BY detected_at DESC LIMIT 1")
            row = cursor.fetchone()
            regime = row[0] if row else "normal"
            gated_regimes = {"high_vol", "crisis"}
            is_gated = regime.lower() in gated_regimes

            rv_data = {
                "signal_value": signal.risk_on_score,
                "pairs": [p.to_dict() for p in signal.pairs.values()],
                "avg_z_score": signal.avg_z_score,
                "max_divergence": signal.max_divergence,
                "num_diverged": signal.num_diverged,
                "total_pairs": signal.total_pairs,
                "available_pair_count": signal.available_pair_count,
                "unavailable_pair_count": signal.unavailable_pair_count,
                "unavailable_pairs": signal.unavailable_pairs,
                "missing_symbols": signal.missing_symbols,
                "risk_on_score": signal.risk_on_score,
                "duration_score": signal.duration_score,
                "overall_conviction": signal.overall_conviction,
                "current_regime": regime,
                "is_gated_off": is_gated,
                "regime_note": "Mean-reversion fails in volatile regimes" if is_gated else "Active — mean-reversion favorable",
                "weight_in_ensemble": 0.0 if is_gated else 0.13,
                "generated_at": datetime.now().isoformat(),
            }

            out_path = PUBLIC_DIR / "cross_asset_rv.json"
            save_results_json(rv_data, output_path=str(out_path))
            return out_path

        except SIGNAL_EXCEPTIONS as e:
            _log_signal_error("cross_asset_rv", e)
            return None

    @staticmethod
    def _graduation_display_value(value: Any) -> str:
        """Format checklist numeric/bool values for the dashboard criterion table."""
        if isinstance(value, bool):
            return "yes" if value else "no"
        if isinstance(value, int) and not isinstance(value, bool):
            return str(value)
        if isinstance(value, float):
            if value == 0.0:
                return "0"
            if abs(value) >= 100:
                return f"{value:.1f}"
            if abs(value) >= 1:
                return f"{value:.2f}"
            return f"{value:.4f}".rstrip("0").rstrip(".")
        return str(value)

    @staticmethod
    def _paper_trading_summary_for_dashboard(
        state: Dict[str, Any],
        *,
        days_elapsed: Any,
        days_required: Any,
    ) -> Dict[str, Any]:
        """Build frontend paper_trading block from checklist-loaded state.

        Dual-shape: dashboard GraduationDataSchema requires start_date /
        initial_capital / current_value / days_elapsed / days_required while
        the producer keeps trading_days / min_trading_days at the top level.
        """
        portfolio = state.get("portfolio") if isinstance(state.get("portfolio"), dict) else {}
        summary = (
            state.get("paper_trading_summary")
            if isinstance(state.get("paper_trading_summary"), dict)
            else {}
        )
        history = portfolio.get("history") if isinstance(portfolio.get("history"), list) else []

        start_date = ""
        if history and isinstance(history[0], dict):
            ts = history[0].get("timestamp")
            if isinstance(ts, str) and ts:
                start_date = ts[:10]
        if not start_date:
            date_hint = summary.get("date")
            if isinstance(date_hint, str) and date_hint:
                start_date = date_hint[:10]

        initial_capital = 100_000.0
        current_value: Optional[float] = None

        if history and isinstance(history[0], dict):
            start_val = history[0].get("total_value")
            if isinstance(start_val, (int, float)):
                initial_capital = float(start_val)
        if history and isinstance(history[-1], dict):
            end_val = history[-1].get("total_value")
            if isinstance(end_val, (int, float)):
                current_value = float(end_val)

        # Prefer authoritative paper-trading-performance metrics when present.
        start_value_files = sorted(DATA_DIR.glob("paper-trading-performance-*.json"))
        if start_value_files:
            try:
                with open(start_value_files[-1]) as f:
                    perf_raw = json.load(f)
                if isinstance(perf_raw, dict):
                    if not start_date and isinstance(perf_raw.get("date"), str):
                        start_date = perf_raw["date"][:10]
                    perf = perf_raw.get("performance")
                    if isinstance(perf, dict):
                        sv = perf.get("start_value")
                        if isinstance(sv, (int, float)):
                            initial_capital = float(sv)
                        cv = perf.get("current_value")
                        if isinstance(cv, (int, float)):
                            current_value = float(cv)
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                pass

        if current_value is None:
            cash = portfolio.get("cash")
            positions = portfolio.get("positions")
            if isinstance(cash, (int, float)) and isinstance(positions, dict):
                pos_sum = 0.0
                for pos in positions.values():
                    if isinstance(pos, dict) and isinstance(pos.get("value"), (int, float)):
                        pos_sum += float(pos["value"])
                current_value = float(cash) + pos_sum
        if current_value is None:
            current_value = initial_capital

        try:
            days_elapsed_n = int(days_elapsed) if days_elapsed is not None else 0
        except (TypeError, ValueError):
            days_elapsed_n = 0
        try:
            days_required_n = int(days_required) if days_required is not None else 0
        except (TypeError, ValueError):
            days_required_n = 0

        if not start_date:
            start_date = datetime.now().date().isoformat()

        return {
            "start_date": start_date,
            "initial_capital": round(initial_capital, 2),
            "current_value": round(float(current_value), 2),
            "days_elapsed": days_elapsed_n,
            "days_required": days_required_n,
        }

    def generate_graduation_json(self) -> Optional[Path]:
        """Generate graduation readiness progress for dashboard.

        Emits a dual-shape payload:
        - Producer fields: readiness_score, is_graduation_ready, criteria name/required
        - Frontend schema aliases: readiness_pct, eligible, paper_trading, and
          criteria id/label/threshold (panel String()-coerces numeric value)
        """
        try:
            from src.strategy.graduation_checklist import GraduationChecklist

            checklist = GraduationChecklist()
            state = checklist._load_state()
            results = checklist.check(state)
            score = checklist.readiness_score(results)
            is_ready = checklist.is_graduation_ready(results)

            # Build progress data for dashboard (dual-shape per criterion)
            criteria_progress = []
            for name, result in results.items():
                criteria_progress.append({
                    # Producer fields (existing consumers / Python tests)
                    "name": name,
                    "passed": result.passed,
                    "value": result.value,
                    "required": result.required,
                    "description": result.description,
                    # Frontend GraduationChecklistPanel fields
                    "id": name,
                    "label": result.description or name,
                    "threshold": self._graduation_display_value(result.required),
                })

            trading_days_result = results.get("min_trading_days")
            n_days = trading_days_result.value if trading_days_result is not None else 0
            min_trading_days = (
                trading_days_result.required
                if trading_days_result is not None
                else checklist.criteria["min_trading_days"]["value"]
            )
            manual_approval = results.get("manual_approval")
            paper_trading = self._paper_trading_summary_for_dashboard(
                state,
                days_elapsed=n_days,
                days_required=min_trading_days,
            )

            graduation_data = {
                # Producer / ops fields
                "readiness_score": score,
                "is_graduation_ready": is_ready,
                "manual_approval_required": True,
                "manual_approval_pending": not bool(manual_approval and manual_approval.passed),
                "trading_days": n_days,
                "min_trading_days": min_trading_days,
                "criteria_met": sum(1 for n, r in results.items() if n != "manual_approval" and r.passed),
                "criteria_total": sum(1 for n in results if n != "manual_approval"),
                "criteria": criteria_progress,
                "generated_at": datetime.now().isoformat(),
                # Frontend GraduationDataSchema / panel aliases
                "readiness_pct": score,
                "eligible": is_ready,
                "paper_trading": paper_trading,
            }

            out_path = PUBLIC_DIR / "graduation.json"
            save_results_json(graduation_data, output_path=str(out_path))

            return out_path

        except SIGNAL_EXCEPTIONS as e:
            _log_signal_error("graduation", e)
            return None

    @staticmethod
    def _latest_stale_explainability_metadata(source_dir: Path) -> Dict[str, Any]:
        """Return metadata for the newest historical explainability file, if any."""
        dated_files = sorted(source_dir.glob("explainability_*.json"), reverse=True)
        if not dated_files:
            return {}

        latest = dated_files[0]
        metadata: Dict[str, Any] = {"stale_source_file": latest.name}
        try:
            payload = json.loads(latest.read_text())
            analysis_date = payload.get("analysis_date")
            if analysis_date:
                metadata["stale_analysis_date"] = str(analysis_date)
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as e:
            metadata["stale_read_error"] = str(e)
        return metadata

    @staticmethod
    def _build_unavailable_explainability_payload(
        *,
        generated_at: str,
        reason: str,
        source_file: Optional[str] = None,
        analysis_date: Optional[str] = None,
        stale_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build an explicit no-current-explainability artifact."""
        return {
            "timestamp": generated_at,
            "analysis_date": analysis_date or generated_at[:10],
            "latest_decision": None,
            "recent_decisions": [],
            "signal_deep_dives": {},
            "top_sources_today": [],
            "decision_quality": {
                "status": "unavailable_current_signals",
                "reason": reason,
            },
            "freshness": {
                "status": "unavailable",
                "generated_at": generated_at,
                "source_file": source_file,
                "reason": reason,
                **(stale_metadata or {}),
            },
        }

    def generate_explainability_json(self) -> Optional[Path]:
        """Generate current or explicitly unavailable explainability data."""
        try:
            from src.dashboard.explainability import build_explainability_from_signals_data

            source_dir = DATA_DIR / "explainability"
            target_dir = PUBLIC_DIR / "explainability"
            target_dir.mkdir(parents=True, exist_ok=True)
            target = target_dir / "explainability_latest.json"
            generated_at = datetime.now().isoformat()
            stale_metadata = self._latest_stale_explainability_metadata(source_dir)
            signals_path = PUBLIC_DIR / "signals.json"

            if signals_path.exists():
                signals_data = json.loads(signals_path.read_text())
                payload = build_explainability_from_signals_data(
                    signals_data,
                    timestamp=generated_at,
                )
                has_current_decision = payload.get("latest_decision") is not None
                payload["freshness"] = {
                    "status": "current" if has_current_decision else "unavailable",
                    "generated_at": generated_at,
                    "source_file": signals_path.name,
                    "analysis_date": payload.get("analysis_date"),
                    "latest_decision_timestamp": (
                        payload.get("latest_decision") or {}
                    ).get("timestamp"),
                    **stale_metadata,
                }
                if not has_current_decision:
                    payload = self._build_unavailable_explainability_payload(
                        generated_at=generated_at,
                        reason="Current signals.json did not contain ensemble explainability inputs.",
                        source_file=signals_path.name,
                        analysis_date=payload.get("analysis_date"),
                        stale_metadata=stale_metadata,
                    )
                save_results_json(payload, output_path=str(target))
                return target

            payload = self._build_unavailable_explainability_payload(
                generated_at=generated_at,
                reason="Current signals.json was not available for explainability generation.",
                stale_metadata=stale_metadata,
            )
            save_results_json(payload, output_path=str(target))
            return target

        except (OSError, json.JSONDecodeError, ValueError, TypeError) as e:
            logger.warning("Failed to generate explainability data: %s", e)
            return None

    def _load_risk_decomposition_signal_section(self) -> Optional[Dict[str, Any]]:
        """Embed risk decomposition into signals.json for optional staleness TTL.

        Prefer computing live; fall back to the public sidecar when present so
        the section is not left missing (None → optional unavailable forever).
        """
        now_ts = datetime.now(timezone.utc).isoformat()

        def _stamp(payload: Dict[str, Any]) -> Dict[str, Any]:
            payload.setdefault("generated_at", now_ts)
            payload.setdefault("timestamp", payload.get("generated_at") or now_ts)
            return payload

        try:
            from src.monitor.risk_decomposition import decompose_portfolio

            result = decompose_portfolio(weights=BASE_ALLOCATION)
            return _stamp(result.to_dict())
        except Exception as exc:  # noqa: BLE001 — optional section
            logger.debug("Live risk_decomposition embed skipped: %s", exc)

        path = PUBLIC_DIR / "risk_decomposition.json"
        if not path.exists():
            return None
        try:
            with open(path) as f:
                data = json.load(f)
            if not isinstance(data, dict):
                return None
            # Explicit unavailable/error sidecars stay unavailable for operators.
            if data.get("status") in {"unavailable", "error"} or "error" in data:
                return _stamp(data)
            # Successful decompose payloads should not look unavailable.
            data.pop("status", None)
            return _stamp(data)
        except (OSError, json.JSONDecodeError, TypeError):
            return None

    def generate_risk_decomposition_json(self) -> Path:
        """Generate risk factor decomposition for dashboard."""
        output_path = PUBLIC_DIR / "risk_decomposition.json"

        try:
            from src.monitor.risk_decomposition import decompose_portfolio

            result = decompose_portfolio(weights=BASE_ALLOCATION)
            data = result.to_dict()
            data["generated_at"] = datetime.now().isoformat()
            save_results_json(data, output_path=str(output_path))
            return output_path

        except ImportError:
            logger.warning("scipy not available — skipping risk decomposition")
            fallback = {
                "status": "unavailable",
                "reason": "scipy not installed",
                "generated_at": datetime.now().isoformat(),
            }
            save_results_json(fallback, output_path=str(output_path))
            return output_path

        except (OSError, ValueError, TypeError, RuntimeError) as e:
            logger.warning("Risk decomposition failed: %s", e)
            fallback = {
                "status": "error",
                "reason": str(e),
                "generated_at": datetime.now().isoformat(),
            }
            save_results_json(fallback, output_path=str(output_path))
            return output_path

    def run(self):
        """Generate all dashboard files."""
        logger.info("Generating dashboard data...")

        # Freeze manifest for config drift detection
        try:
            from src.monitor.freeze_manifest import create_manifest, save_manifest, load_manifest, diff_manifests
            current = create_manifest()
            baseline = load_manifest()
            if baseline:
                drift = diff_manifests(baseline, current)
                if drift["drifted"]:
                    logger.warning("Config drift detected: git_changed=%s config_drift=%d files_added=%d files_modified=%d",
                                   drift["git_changed"],
                                   len(drift["config_drift"]),
                                   len(drift["file_changes"]["added"]),
                                   len(drift["file_changes"]["modified"]))
            save_manifest(current)
        except Exception as e:
            logger.warning("Freeze manifest failed: %s", e)

        try:
            performance_path = self.generate_performance_json()
            health_path = self.generate_health_json()
            health_payload = self._load_json_file(health_path)
            paths = [
                performance_path,
                health_path,
                self.generate_signals_json(),
                self.generate_stats_json(),
                self.generate_alerts_json(health=health_payload),
                self.generate_incidents_json(),
                self.generate_analytics_json(),
                self.generate_graduation_json(),
                self.generate_adaptive_sizing_json(),
                self.generate_vixy_hedge_json(),
                self.generate_black_litterman_json(),
                self.generate_turnover_validator_json(),
                self.generate_regime_gate_json(),
                self.generate_tsmom_json(),
                self.generate_cross_asset_rv_json(),
                self.generate_explainability_json(),
                self.generate_risk_decomposition_json(),
            ]

            # Overlay dashboard (separate path — may fail gracefully)
            overlay_path = self.generate_overlay_json()
            if overlay_path:
                paths.append(overlay_path)

            labs_registry_path = None
            try:
                from src.research.experiment_registry import save_labs_registry

                labs_registry_path = save_labs_registry(
                    data_dirs=(DATA_DIR,),
                    public_dir=PUBLIC_DIR,
                    project_root=DATA_DIR.parent if DATA_DIR.name == "data" else DATA_DIR,
                )
                if labs_registry_path:
                    paths.append(labs_registry_path)
            except (ImportError, ValueError, OSError, TypeError) as e:
                logger.warning("Labs registry generation skipped: %s", e)

            if labs_registry_path:
                try:
                    from src.research.experiment_scorecard import save_labs_scorecards

                    labs_scorecard_path = save_labs_scorecards(
                        registry_path=labs_registry_path,
                        public_dir=PUBLIC_DIR,
                    )
                    if labs_scorecard_path:
                        paths.append(labs_scorecard_path)
                except (ImportError, ValueError, OSError, TypeError) as e:
                    logger.warning("Labs scorecard generation skipped: %s", e)

            try:
                from src.research.experiment_replay_batch import publish_labs_replays

                labs_replay_path = publish_labs_replays(
                    data_dir=DATA_DIR,
                    public_dir=PUBLIC_DIR,
                    project_root=DATA_DIR.parent if DATA_DIR.name == "data" else DATA_DIR,
                )
                if labs_replay_path:
                    paths.append(labs_replay_path)
            except (ImportError, ValueError, OSError, TypeError) as e:
                logger.warning("Labs replay report generation skipped: %s", e)

            try:
                from src.research.labs_validation_report import save_labs_validation_report

                labs_validation_path = save_labs_validation_report(
                    data_dirs=(DATA_DIR,),
                    public_dir=PUBLIC_DIR,
                    project_root=DATA_DIR.parent if DATA_DIR.name == "data" else DATA_DIR,
                )
                if labs_validation_path:
                    paths.append(labs_validation_path)
            except (ImportError, ValueError, OSError, TypeError) as e:
                logger.warning("Labs validation report generation skipped: %s", e)

            try:
                from src.monitor.decision_registry import (
                    publish_decision_registry_json,
                    sync_labs_registry_experiments,
                )
                from src.research.experiment_registry import LABS_REGISTRY_FILENAME

                labs_json = PUBLIC_DIR / LABS_REGISTRY_FILENAME
                if labs_json.exists():
                    sync_labs_registry_experiments(labs_json)
                decision_registry_path = publish_decision_registry_json(public_dir=PUBLIC_DIR)
                paths.append(decision_registry_path)
            except (ImportError, ValueError, OSError, TypeError) as e:
                logger.warning("Decision registry JSON generation skipped: %s", e)

            for p in paths:
                if p:
                    logger.info("Generated: %s", p)

            # Create a versioned public-data manifest while keeping files[] for
            # existing dashboard consumers.
            index = build_public_data_index(paths, public_dir=PUBLIC_DIR)
            save_results_json(index, output_path=str(PUBLIC_DIR / "index.json"))
            _mirror_public_data_contract_files_to_dist(PUBLIC_DIR)
        finally:
            self.close()

        logger.info("Dashboard generation complete")

if __name__ == "__main__":
    from src.utils.log_config import configure_logging
    configure_logging()
    with DashboardGenerator() as gen:
        gen.run()

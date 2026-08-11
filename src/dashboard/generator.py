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
import sys
from collections import deque
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

from src.paths import BASE_ALLOCATION, YIELDS_JSON, DATA_DIR, PUBLIC_DATA_DIR, MARKET_DB, REGIME_OVERRIDES, sqlite_connect
from src.strategy.regime_allocation import (
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
from src.dashboard.signal_section_builder import SignalSectionBuilder
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
    "project_smart_rebalance_budget_onto_health",
    "project_paper_return_ssot_onto_health",
    "project_voting_mass_quality_onto_health",
    "project_reentry_eligibility_onto_health",
    "project_pending_artifact_cron_onto_health",
    "project_execution_timeline_onto_health",
    "project_repo_public_mirror_lag_onto_health",
]

# Re-export health-projection cluster (moved to health_projections.py by Item 8)
from src.dashboard.health_projections import (
    _parse_rebalance_clock,
    project_smart_rebalance_budget_onto_health,
    project_execution_timeline_onto_health,
    project_repo_public_mirror_lag_onto_health,
    project_pending_artifact_cron_onto_health,
    project_reentry_eligibility_onto_health,
    project_voting_mass_quality_onto_health,
    project_paper_return_ssot_onto_health,
    _apply_kill_to_smart_rebalance,
    _remaining_budget_ratio,
    _remaining_budget_display_pct,
    _load_canonical_health_report,
)

# Re-export provenance/dual-write cluster (moved to provenance.py by Item 9)
from src.dashboard.provenance import (
    _apply_partial_patch_git_sha_honesty,
    _enrich_duration_allocation_provenance,
    _source_manifest_row_for,
    _yield_source_provenance,
    _first_known_value,
    _attach_signal_metadata,
    _generator_git_sha_short,
    _stamp_generator_git_sha,
    _canonical_file_content_hash,
    _attach_dual_write_provenance,
    finalize_dual_write_provenance_after_sync,
    _finalize_signal_metadata,
    _dist_data_dir_for_public_dir,
    _mirror_public_data_contract_files_to_dist,
    PUBLIC_DATA_DIST_MIRROR_FILES,
    SIGNAL_EXCEPTIONS,
    MONITOR_EXCEPTIONS,
    _BUG_EXCEPTIONS,
)

# Re-export alt-data cluster (moved to alternative_data.py by Item 10)
from src.dashboard.alternative_data import (
    _ALT_DATA_LEGACY_COMPONENT_KEYS,
    project_alternative_data_signal,
    load_alternative_data_producer_timestamp,
    refresh_public_alternative_data_projection,
    _is_predictive_fred_macro,
    _ENSEMBLE_STALENESS_MAP,
)

from src.dashboard.sections_ensemble import _EnsembleSectionsMixin
from src.dashboard.sections_hedge import _HedgeSectionsMixin
from src.dashboard.sections_regime import _RegimeAuthorityMixin

# Legacy flat keys (pre seven-component producer) → panel component names



logger = logging.getLogger(__name__)

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
        # Batch EE: dual-signal pending vs artifact-fresh reconcile on compact
        summary = project_pending_artifact_cron_onto_health(summary, cron_jobs)

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

    # Keep the IC quality plane available to compact consumers without copying
    # the raw signal-history report into signals.health.
    ic_summary = report.get("ic_decay_summary")
    if isinstance(ic_summary, dict):
        summary["ic_decay_summary"] = dict(ic_summary)

    # Batch BN: surface signal_health rollup so compact view discloses 0/N healthy
    signal_health = report.get("signal_health")
    if isinstance(signal_health, dict):
        sh_summary = signal_health.get("summary")
        if isinstance(sh_summary, dict):
            summary["signal_health_healthy"] = sh_summary.get("healthy")
            summary["signal_health_degraded"] = sh_summary.get("degraded")
            summary["signal_health_unhealthy"] = sh_summary.get("unhealthy")
            summary["signal_health_total_tracked"] = sh_summary.get(
                "total_tracked"
            ) or sh_summary.get("total")
            # Batch CM: quality badge + freeze flag for compact consumers
            if sh_summary.get("quality_badge"):
                summary["signal_health_quality_badge"] = sh_summary.get("quality_badge")
            if sh_summary.get("zero_healthy_sources"):
                summary["signal_health_zero_healthy"] = True
            else:
                # Batch CR: clear sticky zero-healthy when SH recovered
                summary["signal_health_zero_healthy"] = False
            # Batch CQ/CR: always project freeze True/False (never leave sticky True)
            freeze_active = bool(sh_summary.get("ensemble_weight_freeze_active"))
            summary["ensemble_weight_freeze_active"] = freeze_active
            if sh_summary.get("ensemble_weights_age_days") is not None:
                summary["ensemble_weights_age_days"] = sh_summary.get(
                    "ensemble_weights_age_days"
                )
            if sh_summary.get("ensemble_weights_file_stale"):
                summary["ensemble_weights_file_stale"] = True
            elif not freeze_active:
                summary["ensemble_weights_file_stale"] = bool(
                    sh_summary.get("ensemble_weights_file_stale")
                )
        overall = signal_health.get("overall_health") or signal_health.get("status")
        if overall:
            summary["signal_health_status"] = overall
        qd = signal_health.get("quality_disclosure")
        if isinstance(qd, dict) and qd.get("badge"):
            summary["signal_quality_badge"] = qd.get("badge")
            # Prefer quality_disclosure freeze fields when summary omitted them
            freeze_block = qd.get("ensemble_weight_freeze")
            if isinstance(freeze_block, dict):
                if "weight_freeze_active" in freeze_block:
                    summary["ensemble_weight_freeze_active"] = bool(
                        freeze_block.get("weight_freeze_active")
                    )
                if freeze_block.get("ensemble_weights_age_days") is not None:
                    summary["ensemble_weights_age_days"] = freeze_block.get(
                        "ensemble_weights_age_days"
                    )
                if freeze_block.get("weight_file_stale") is not None:
                    summary["ensemble_weights_file_stale"] = bool(
                        freeze_block.get("weight_file_stale")
                    )

    return summary




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


class DashboardGenerator(_EnsembleSectionsMixin, _HedgeSectionsMixin, _RegimeAuthorityMixin):
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

    def _get_signal_section_builder(self) -> SignalSectionBuilder:
        """Return the collaborator that owns signals.json section assembly."""
        return SignalSectionBuilder(self, generator_module=sys.modules[__name__])
    
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
        
        # Canonical daily regime history is JSONL; SQLite may be empty/stale.
        from src.regime.regime_history import load_daily_regime_history

        regimes = load_daily_regime_history(DATA_DIR / "regime_log.json").records[-90:]
        
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

        output = _stamp_generator_git_sha({
            "prices": prices,
            "regimes": regimes,
            "paper_portfolio": paper_perf,
            "generated_at": datetime.now(timezone.utc).isoformat()
        })

        out_path = PUBLIC_DIR / "dashboard.json"
        save_results_json(output, output_path=str(out_path))

        return out_path

    @staticmethod
    def _unavailable_zero_dte_payload() -> Dict[str, Any]:
        """Schema-compatible zero_dte panel when no producer is wired.

        LiveDashboard expects positions/config fields; never publish silent {}.
        Not live order-routing authority.
        """
        now_ts = datetime.now().isoformat()
        return {
            "positions": [],
            "config": None,
            "weekly_trades_used": 0,
            "total_premium_collected_mtd": 0.0,
            "status": "unavailable",
            "runtime_status": "unavailable_no_producer",
            "active": False,
            "live_authoritative": False,
            "reason": "zero_dte producer not wired into overlay merge",
            "generated_at": now_ts,
            "timestamp": now_ts,
        }

    @staticmethod
    def _unavailable_closing_auction_payload() -> Dict[str, Any]:
        """Schema-compatible closing_auction panel when no producer is wired."""
        now_ts = datetime.now().isoformat()
        return {
            "signals": [],
            "last_update": None,
            "market_open": False,
            "status": "unavailable",
            "runtime_status": "unavailable_no_producer",
            "active": False,
            "live_authoritative": False,
            "reason": "closing_auction producer not wired into overlay merge",
            "generated_at": now_ts,
            "timestamp": now_ts,
        }

    @staticmethod
    def _is_populated_overlay_section(value: Any) -> bool:
        """True when overlay merge produced a non-empty, non-placeholder section."""
        if not isinstance(value, dict) or not value:
            return False
        status = str(value.get("status") or value.get("runtime_status") or "").lower()
        if status in {"unavailable", "unavailable_no_producer"}:
            return False
        # Real producers set active/positions/signals or domain fields
        if value.get("positions") or value.get("signals"):
            return True
        if value.get("active") is True:
            return True
        # Any domain payload beyond honesty metadata counts as populated
        meta_keys = {
            "status",
            "runtime_status",
            "active",
            "live_authoritative",
            "reason",
            "generated_at",
            "timestamp",
            "last_update",
            "market_open",
            "config",
            "weekly_trades_used",
            "total_premium_collected_mtd",
            "positions",
            "signals",
        }
        return any(k not in meta_keys for k in value.keys())

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
            # Pass through producers if overlay ever ships them
            if isinstance(data.get("zero_dte"), dict):
                result["zero_dte"] = data["zero_dte"]
            if isinstance(data.get("closing_auction"), dict):
                result["closing_auction"] = data["closing_auction"]
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

        # Dead schema surfaces: never publish silent {} without honesty fields
        if not self._is_populated_overlay_section(result.get("zero_dte")):
            result["zero_dte"] = self._unavailable_zero_dte_payload()
        if not self._is_populated_overlay_section(result.get("closing_auction")):
            result["closing_auction"] = self._unavailable_closing_auction_payload()

        return result

    def _record_ic_data(self, output: Dict) -> None:
        """Record signal predictions for IC decay monitoring.

        Two-phase lifecycle (Task 2B — per-signal):
        1. Resolve: each staged prediction is paired with the forward return of
           its own declared target asset (SPY/GLD/TLT) once its intended
           horizon has elapsed; other entries stay staged.
        2. Stage: store canonical current predictions (equity/gold/duration
           biases, alternative-data SPY-facing value, behavioral normalized
           equity shift) for resolution next runs.

        Consensus, factor rotation, and FRED are NOT staged into correlation
        control history until their basket/outcome/metric contracts are
        implemented; their exclusion is disclosed in the IC summary.
        Saves monitor state to disk so IC data survives across cron runs.
        """
        try:
            from src.monitor.ic_decay_monitor import ICMonitor

            monitor = ICMonitor()
            monitor.load_state()
            cursor = self.conn.cursor()

            def _latest_bar(symbol: str):
                cursor.execute(
                    "SELECT date, close FROM prices WHERE symbol = ? "
                    "ORDER BY date DESC LIMIT 1",
                    (symbol,),
                )
                return cursor.fetchone()

            latest_spy_row = _latest_bar("SPY")

            # Phase 1: Resolve previously staged predictions per target asset.
            staged_entries = getattr(monitor, "_staged", None) or {}
            for target_asset in ("SPY", "GLD", "TLT"):
                latest = _latest_bar(target_asset)
                if not latest:
                    continue
                latest_date, latest_close = latest
                staged_dates = sorted(
                    {
                        entry.get("prediction_date")
                        for entry in staged_entries.values()
                        if entry.get("metadata", {}).get("target_asset")
                        in (None, target_asset)
                        and entry.get("prediction_date")
                    }
                )
                if not staged_dates:
                    continue
                for staged_date in staged_dates:
                    cursor.execute(
                        "SELECT date, close FROM prices WHERE symbol = ? "
                        "AND date >= ? ORDER BY date ASC LIMIT 1",
                        (target_asset, staged_date),
                    )
                    start_row = cursor.fetchone()
                    if (
                        not start_row
                        or start_row[0] == latest_date
                        or float(start_row[1]) <= 0
                    ):
                        continue
                    start_price = float(start_row[1])
                    forward_return = (float(latest_close) / start_price) - 1.0
                    cursor.execute(
                        "SELECT MIN(date), COUNT(*) FROM prices WHERE symbol = ? "
                        "AND date > ? AND date <= ?",
                        (target_asset, start_row[0], latest_date),
                    )
                    realized_range_row = cursor.fetchone()
                    realized_start_date = (
                        str(realized_range_row[0])
                        if realized_range_row and realized_range_row[0]
                        else None
                    )
                    realized_horizon_sessions = int(realized_range_row[1] or 0)
                    n_resolved = monitor.resolve_staged(
                        forward_return,
                        resolved_date=str(latest_date),
                        realized_start_date=realized_start_date,
                        target_asset=target_asset,
                        realized_horizon_sessions=realized_horizon_sessions,
                    )
                    if n_resolved:
                        logger.info(
                            "IC decay: resolved %d staged predictions "
                            "(%s → %s %s, forward return=%.4f%%)",
                            n_resolved, staged_date, latest_date, target_asset,
                            forward_return * 100,
                        )

            # Phase 2: Stage canonical current predictions (per-signal upsert).
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

            # Alternative data: canonical SPY-facing value (projection boundary).
            alt = output.get("alternative_data")
            if isinstance(alt, dict) and alt.get("spy_value") is not None:
                predictions["alternative_data"] = float(alt["spy_value"])

            # Behavioral sentiment: normalized equity shift (capped ±5% → [-1, 1]).
            beh = output.get("behavioral_sentiment")
            if isinstance(beh, dict) and beh.get("equity_shift_pct") is not None:
                try:
                    predictions["behavioral_sentiment"] = max(
                        -1.0, min(1.0, float(beh["equity_shift_pct"]) / 5.0)
                    )
                except (TypeError, ValueError):
                    pass

            if predictions:
                monitor.stage_predictions(
                    predictions,
                    str(latest_spy_row[0]) if latest_spy_row else datetime.now(timezone.utc).strftime("%Y-%m-%d"),
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

        # threshold=0.5 keeps legacy dashboard wiring; max_lookback caps O(n)
        # collapse on multi-year SPY history (see BOCDDetector.DEFAULT_MAX_LOOKBACK).
        detector = BOCDDetector(
            hazard_rate=1.0 / 252,
            threshold=0.5,
            min_run_length=5,
            max_lookback=BOCDDetector.DEFAULT_MAX_LOOKBACK,
        )
        detector.fit(returns)

        signal = detector.get_signal()
        bocd_data = signal["bocd_detector"]
        bocd_data["timestamp"] = datetime.now().isoformat()

        return bocd_data

    @staticmethod
    def _coerce_vix_level(value: Any) -> Optional[float]:
        """Parse a positive finite VIX level, else None."""
        if value is None:
            return None
        try:
            level = float(value)
        except (TypeError, ValueError):
            return None
        if level != level or level <= 0:  # NaN or non-positive
            return None
        return level

    @classmethod
    def _enrich_regime_vix(
        cls,
        regime_data: Dict[str, Any],
        *,
        vix_term_structure: Any = None,
        behavioral_sentiment: Any = None,
    ) -> Dict[str, Any]:
        """Fill regime.vix from best available surface; disclose vix_source.

        Preference: existing market.db level on regime_data → vix_term_structure
        → behavioral_sentiment. Does not change live target_allocations authority.
        """
        out = dict(regime_data or {})
        existing = cls._coerce_vix_level(out.get("vix"))
        if existing is not None:
            out["vix"] = existing
            out.setdefault("vix_source", "market.db")
            return out

        # vix_term_structure payload (dict from VIXTermStructureSignal.to_dict)
        if isinstance(vix_term_structure, dict):
            for key in ("vix_spot", "vix", "spot"):
                level = cls._coerce_vix_level(vix_term_structure.get(key))
                if level is not None:
                    out["vix"] = level
                    out["vix_source"] = "vix_term_structure"
                    return out

        # behavioral_sentiment may nest vix under top-level or snapshot
        if isinstance(behavioral_sentiment, dict):
            candidates = [
                behavioral_sentiment.get("vix"),
                behavioral_sentiment.get("vix_level"),
            ]
            opts = behavioral_sentiment.get("options")
            if isinstance(opts, dict):
                candidates.append(opts.get("vix"))
            for cand in candidates:
                level = cls._coerce_vix_level(cand)
                if level is not None:
                    out["vix"] = level
                    out["vix_source"] = "behavioral_sentiment"
                    return out

        out["vix"] = None
        out["vix_source"] = "unavailable"
        return out

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
        vix_level = self._coerce_vix_level(vix_row[0] if vix_row else None)
        
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
            "vix_source": "market.db" if vix_level is not None else "unavailable",
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
        
        target_alloc = self._resolve_live_target_allocations_for_regime(current_regime)
        
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


    def _build_base_signal_sections(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Build the core signal sections before dashboard-level metadata."""
        return self._get_signal_section_builder().build_base_sections(context)

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
        return self._get_signal_section_builder().build_optional_sections(
            output,
            context,
        )

    def _apply_signal_postprocessors(
        self,
        output: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Apply staleness, monitoring, alerting, and final signal appenders."""
        return self._get_signal_section_builder().apply_postprocessors(
            output,
            context,
        )

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
        private_path = Path(DATA_DIR) / "signals.json"

        # Validate once (same callback as legacy save_results_json).
        try:
            output = validate_all_signals(output)
        except Exception as e:  # noqa: BLE001 — match save_results_json soft-fail
            logger.warning("Validation callback failed: %s", e)

        # Batch HK: serialize-once multi-dest (public + private + repo soft-mirror)
        # with authority gate + 0o644 mode. Stops Dual-wrote dual open('w') path
        # that re-sticky 0600 and content-drifts under concurrent partials.
        try:
            from src.monitor.signal_authority import (
                AuthorityValidationError,
                default_repo_signals_path,
                write_signals_multi_dest,
            )

            # Explicit repo_path so soft-mirror is not skipped under pytest
            # (auto soft-mirror is gated off when PYTEST_CURRENT_TEST is set).
            result = write_signals_multi_dest(
                output,
                public_path=out_path,
                private_path=private_path,
                repo_path=default_repo_signals_path(),
                soft_mirror_repo=True,
            )
            if result.wrote_public:
                logger.info("Generated signals.json → %s", out_path)
            if result.wrote_private:
                logger.info("Multi-dest private signals.json → %s", private_path)
            if result.wrote_repo:
                logger.info("Multi-dest repo soft-mirror signals.json → %s", result.repo_path)
            if result.skipped_reason:
                logger.warning(
                    "signals full-generate multi-dest partial skip: %s",
                    result.skipped_reason,
                )
        except AuthorityValidationError as exc:
            logger.error(
                "Refusing full generate signals write (authority gate): %s",
                exc,
            )
        except (OSError, TypeError, ValueError, ImportError) as exc:
            logger.warning(
                "signals multi-dest full generate failed (%s); falling back to "
                "legacy public save_results_json only",
                exc,
            )
            save_results_json(output, output_path=str(out_path))

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

        # Coverage fail → demote primary GARCH only on over-exceedance hard fail.
        # Under-exceedance is efficiency warning (over-conservative), not demotion.
        cov = garch_cvar.get("coverage_diagnostics")
        if isinstance(cov, dict):
            coverage_pass = cov.get("coverage_pass")
            direction = cov.get("coverage_direction") or cov.get("exceedance_bias")
            hard_fail = cov.get("coverage_hard_fail")
            if hard_fail is None:
                # Directed hard fail (over) OR legacy undirected coverage_pass=false
                # without direction metadata (treat as hard fail for safety).
                if direction in (None, "", "ok") and coverage_pass is False:
                    hard_fail = True
                    direction = direction or "over"  # assume over for legacy
                else:
                    hard_fail = bool(
                        coverage_pass is False and direction == "over"
                    )
            direction = direction or "ok"
            # Surface direction on risk metrics for operators
            garch_cvar["coverage_direction"] = direction
            garch_cvar["exceedance_bias"] = direction
            if hard_fail and garch_cvar.get("garch_active"):
                garch_cvar["garch_active"] = False
                garch_cvar["runtime_role"] = "advisory_degraded"
                garch_cvar["garch_active_reason"] = (
                    f"coverage_hard_fail (direction={direction}, "
                    f"coverage_pass={coverage_pass}); "
                    "over-exceedance — GARCH not primary risk authority"
                )
            elif cov.get("coverage_efficiency_warning") and garch_cvar.get(
                "garch_active"
            ):
                garch_cvar.setdefault("runtime_role", "primary")
                garch_cvar["garch_active_reason"] = (
                    f"coverage_efficiency_warning (direction={direction}); "
                    "under-exceedance — advisory capital inefficiency, "
                    "GARCH remains primary"
                )
            elif coverage_pass is True:
                garch_cvar.setdefault("runtime_role", "primary")
        return garch_cvar

    def _load_entropy_data(self) -> Dict:
        """Load entropy-based diversification metrics for dashboard (v3.22).

        Correlation-axis metrics are **not** hard-coded (prior 0.95 / 2.5 defaults
        looked like live diversification quality). Publish null + status until a
        real covariance path computes them.
        """
        entropy: Dict[str, Any] = {
            "shannon_entropy": None,
            "effective_n": None,
            "max_possible": None,
            "normalized_score": None,
            "concentration_risk": "unknown",
            "hhi_index": None,
            "correlation_entropy": None,
            "participation_ratio": None,
            "correlation_metrics_status": "unavailable",
            "status": "partial",
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
                        if metrics.get("shannon_entropy") is not None:
                            entropy["shannon_entropy"] = metrics.get("shannon_entropy")
                        if metrics.get("effective_n") is not None:
                            entropy["effective_n"] = metrics.get("effective_n")
                        if metrics.get("normalized_score") is not None:
                            entropy["normalized_score"] = metrics.get("normalized_score")
                        if metrics.get("hhi_index") is not None:
                            entropy["hhi_index"] = metrics.get("hhi_index")
                        if metrics.get("max_possible") is not None:
                            entropy["max_possible"] = metrics.get("max_possible")
                        # Derive H_max = ln(n) when shannon present but max missing
                        if (
                            entropy.get("max_possible") is None
                            and entropy.get("shannon_entropy") is not None
                        ):
                            try:
                                import math

                                from src.paths import BASE_ALLOCATION

                                n = len(BASE_ALLOCATION)
                                if n > 1:
                                    entropy["max_possible"] = round(math.log(n), 4)
                            except Exception:  # noqa: BLE001 — leave null
                                pass
                        # Only surface correlation metrics when actually computed
                        if metrics.get("correlation_entropy") is not None:
                            entropy["correlation_entropy"] = metrics.get("correlation_entropy")
                            entropy["correlation_metrics_status"] = "ok"
                        if metrics.get("participation_ratio") is not None:
                            entropy["participation_ratio"] = metrics.get("participation_ratio")
                            entropy["correlation_metrics_status"] = "ok"
                        
                        # Determine concentration risk from normalized score
                        score = entropy.get("normalized_score")
                        if isinstance(score, (int, float)):
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
                            entropy["status"] = "ok"
        except (OSError, json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
            logger.warning("Entropy metrics unavailable: %s", e)

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
        """Expose AIController runtime status without implying live routing authority.

        ``available`` means checkpoint-backed runtime readiness for shadow use,
        not merely that the MARL module imported. Controllers without a loaded
        checkpoint report available=false with reason, plus module_importable.
        """
        status = {
            "schema_version": "marl-runtime-status/v1",
            "available": False,
            "module_importable": False,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "runtime": DashboardGenerator._default_marl_runtime_status(),
            "execution_role": DashboardGenerator._marl_execution_role(),
        }

        try:
            from src.agents.ai_controller import AIController

            controller = AIController(use_signal_integrator=False)
            runtime_status = controller.get_status()
            if isinstance(runtime_status, dict):
                status["module_importable"] = True
                status["runtime"] = {
                    **DashboardGenerator._default_marl_runtime_status(),
                    **runtime_status,
                }
                checkpoint_loaded = bool(runtime_status.get("checkpoint_loaded"))
                # available = ready for shadow inference path (checkpoint present)
                status["available"] = checkpoint_loaded
                if not checkpoint_loaded:
                    status["reason"] = "checkpoint_not_loaded"
                else:
                    status["reason"] = "checkpoint_loaded"
        except SIGNAL_EXCEPTIONS as e:
            _log_signal_error("marl_status", e)
            status["error"] = str(e)
            status["reason"] = "controller_import_or_init_failed"

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
            
            asof = latest.get("date")
            # Wall-clock weekday lag vs today (UTC) for freeze detection
            lag_weekdays = 0
            status = "ok"
            reason = None
            if isinstance(asof, str) and len(asof) >= 10:
                try:
                    from datetime import date as _date
                    asof_d = _date.fromisoformat(asof[:10])
                    today = datetime.now(timezone.utc).date()
                    # Count Mon-Fri strictly after asof through today
                    cur = asof_d
                    from datetime import timedelta as _td
                    cur = cur + _td(days=1)
                    while cur <= today:
                        if cur.weekday() < 5:
                            lag_weekdays += 1
                        cur = cur + _td(days=1)
                except ValueError:
                    lag_weekdays = 0
            max_lag = int(os.environ.get("YIELD_CURVE_MAX_STALE_WEEKDAYS", "5"))
            if lag_weekdays > max_lag:
                status = "stale"
                reason = f"asof_lag_weekdays_{lag_weekdays}_gt_{max_lag}"

            result["yield_curve"] = {
                "spread2s10s": spread,
                "dgs2": latest.get("dgs2"),
                "dgs10": latest.get("dgs10"),
                "duration_regime": regime,
                "spread_history": spread_history,
                "asof": asof,
                "asof_lag_weekdays": lag_weekdays,
                "status": status,
                **({"reason": reason} if reason else {}),
                **({
                    "runtime_status": "stale",
                } if status == "stale" else {}),
                **{
                    key: value
                    for key, value in _yield_source_provenance(PUBLIC_DIR).items()
                    if value is not None
                },
            }
            
            # Calculate duration allocation based on regime (advisory sleeve —
            # never bare weights without provenance; not live order-routing authority)
            regime_allocations = {
                "steep": {"tlt": 0.70, "ief": 0.25, "shy": 0.05, "bil": 0.00},
                "normal": {"tlt": 0.50, "ief": 0.35, "shy": 0.15, "bil": 0.00},
                "flat": {"tlt": 0.30, "ief": 0.40, "shy": 0.25, "bil": 0.05},
                "inverted": {"tlt": 0.15, "ief": 0.25, "shy": 0.35, "bil": 0.25}
            }
            weights = regime_allocations.get(regime, regime_allocations["normal"])
            result["duration_allocation"] = _enrich_duration_allocation_provenance(
                {
                    "weights": weights,
                    # Flat keys retained for backward-compat consumers
                    **weights,
                    "duration_regime": regime,
                    "source": "yield_curve_regime_table",
                }
            )

        except (OSError, json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
            logger.warning("Failed to load yield curve data: %s", e)
        
        return result
    
    def generate_stats_json(self) -> Path:
        """Generate performance statistics."""
        cursor = self.conn.cursor()

        # Single batched query for all symbols instead of N+1 per-symbol queries
        # Champion book (live authority / paper) vs context benchmarks
        champion_symbols = ['SPY', 'GLD', 'TLT']
        context_symbols = ['QQQ', 'VIX']
        symbols = champion_symbols + context_symbols
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
        held_stats: Dict[str, Any] = {}
        context_stats: Dict[str, Any] = {}
        for symbol in symbols:
            prices = symbol_prices.get(symbol, [])
            if len(prices) >= 2:
                returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
                in_portfolio = symbol in champion_symbols
                entry = {
                    "30d_return": round((prices[-1] - prices[0]) / prices[0] * 100, 2),
                    "volatility": round(np.std(returns) * np.sqrt(252) * 100, 2) if returns else 0,
                    "current": prices[-1],
                    "in_portfolio": in_portfolio,
                    "not_in_portfolio": not in_portfolio,
                    "role": "held" if in_portfolio else "benchmark_or_context",
                }
                stats[symbol] = entry
                if in_portfolio:
                    held_stats[symbol] = entry
                else:
                    context_stats[symbol] = entry
        
        # Paper portfolio metrics with SPY comparison.
        # Prefer daily_pnl.jsonl session series (SSOT) over performance.jsonl
        # which historically interleaves evaluator micro-noise.
        paper_metrics = {}
        spy_comparison = None
        daily_returns: list[float] = []
        daily_values: list[float] = []
        return_source = "none"

        def _material_return(val: Any) -> bool:
            """Include real session returns; drop evaluator micro-noise (~1e-8).

            Exact 0.0 is a valid flat session (Sharpe 0). Tiny non-zero noise
            from intraday evaluator rows must not enter the metric series.
            """
            try:
                v = float(val)
            except (TypeError, ValueError):
                return False
            if v == 0.0:
                return True
            return abs(v) >= 1e-6

        pnl_log = DATA_DIR / "daily_pnl.jsonl"
        if pnl_log.exists():
            try:
                by_date: dict[str, dict] = {}
                with open(pnl_log) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            row = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if not isinstance(row, dict):
                            continue
                        d = str(row.get("date") or "")[:10]
                        if not d:
                            continue
                        by_date[d] = row
                ordered = [by_date[d] for d in sorted(by_date.keys())]
                # Keep only material session returns for Sharpe input
                for e in ordered:
                    if not _material_return(e.get("daily_return")):
                        continue
                    daily_returns.append(float(e["daily_return"]))
                    try:
                        daily_values.append(float(e.get("total_value") or 0))
                    except (TypeError, ValueError):
                        daily_values.append(0.0)
                if len(daily_returns) >= 3:
                    return_source = "daily_pnl.jsonl_session"
            except OSError:
                daily_returns, daily_values = [], []

        if return_source == "none":
            perf_log = DATA_DIR / "performance.jsonl"
            if perf_log.exists():
                with open(perf_log) as f:
                    tail_lines = deque(f, maxlen=500)
                    # Raw tail can be short after dedup; require only enough
                    # lines to form a 3-day session series after filters.
                    if len(tail_lines) >= 3:
                        raw_entries = []
                        for l in tail_lines:
                            try:
                                raw_entries.append(json.loads(l))
                            except json.JSONDecodeError:
                                continue
                        daily_entries = self._deduplicate_performance_entries_by_date(
                            raw_entries
                        )
                        for e in daily_entries:
                            dr = e.get("daily_return")
                            if dr is None or not _material_return(dr):
                                continue
                            daily_returns.append(float(dr))
                            try:
                                daily_values.append(float(e.get("total_value") or 0))
                            except (TypeError, ValueError):
                                daily_values.append(0.0)
                        if daily_returns:
                            return_source = "performance.jsonl_daily_dedup"

        if daily_returns and len(daily_returns) >= 3:
                        std_r = float(np.std(daily_returns))
                        raw_sharpe = (
                            float(np.mean(daily_returns) / std_r * np.sqrt(252))
                            if std_r > 0
                            else 0.0
                        )
                        # Honesty: same implausibility gate as graduation (raw > 3.0).
                        # Keep raw value; never coerce to 0.0 (looked like zero skill).
                        implausible = bool(raw_sharpe > 3.0)
                        n_pts = len(daily_returns)
                        paper_metrics = {
                            "sharpe": round(raw_sharpe, 2),
                            "sharpe_raw": round(raw_sharpe, 4),
                            "sharpe_implausible": implausible,
                            "sharpe_plausibility_status": (
                                "implausible_short_sample"
                                if implausible
                                else "ok"
                            ),
                            "sharpe_note": (
                                (
                                    f"implausible raw Sharpe {raw_sharpe:.2f} > 3.0 "
                                    f"over {n_pts} daily points "
                                    "(likely short-sample / low-vol artifact; "
                                    "not graduation-ready skill)"
                                )
                                if implausible
                                else None
                            ),
                            "total_return": (
                                round(
                                    (daily_values[-1] - daily_values[0])
                                    / daily_values[0]
                                    * 100,
                                    2,
                                )
                                if daily_values
                                and daily_values[0]
                                and daily_values[-1]
                                else None
                            ),
                            "max_value": round(max(daily_values), 2) if daily_values else None,
                            "min_value": round(min(daily_values), 2) if daily_values else None,
                            "days_tracked": n_pts,
                            "return_source": return_source,
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
        
        output = _stamp_generator_git_sha({
            "asset_stats": stats,
            "held_asset_stats": held_stats,
            "context_asset_stats": context_stats,
            "champion_symbols": list(champion_symbols),
            "paper_portfolio": paper_metrics,
            "spy_comparison": spy_comparison,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        })
        
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
        
        output = _stamp_generator_git_sha({
            "alerts": sorted(alerts, key=lambda x: x.get("timestamp", "") or "", reverse=True),
            "count": len(alerts),
            "generated_at": datetime.now(timezone.utc).isoformat()
        })
        
        out_path = PUBLIC_DIR / "alerts.json"
        private_alerts = Path(DATA_DIR) / "alerts.json"

        def _write_alerts_multi(payload: dict) -> None:
            """Serialize-once public + private + repo soft-mirror (Batch HN).

            repo_path left None so auto soft-mirror uses repo_filename and skips
            under pytest (avoids clobbering checkout public/data during tests).
            """
            try:
                from src.monitor.signal_authority import write_json_multi_dest

                write_json_multi_dest(
                    payload,
                    public_path=out_path,
                    private_path=private_alerts
                    if private_alerts.parent.is_dir() or private_alerts.exists()
                    else None,
                    soft_mirror_repo=True,
                    repo_filename="alerts.json",
                )
            except Exception as exc:  # noqa: BLE001 — fall back to legacy single dest
                logger.warning(
                    "alerts multi-dest write failed (%s); falling back to save_results_json",
                    exc,
                )
                save_results_json(payload, output_path=str(out_path))

        _write_alerts_multi(output)

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
                    output = _stamp_generator_git_sha({
                        "alerts": rebuilt,
                        "count": len(rebuilt),
                        "generated_at": datetime.now(timezone.utc).isoformat(),
                    })
                    _write_alerts_multi(output)
            except (OSError, json.JSONDecodeError, TypeError) as verify_exc:
                logger.error("alerts.json post-write kill verify failed: %s", verify_exc)
        
        return out_path

    @staticmethod
    def _empty_incident_summary() -> Dict[str, Any]:
        return {
            "schema_version": "incident-lifecycle/v1",
            "generated_at": datetime.now(timezone.utc).isoformat(),
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
        payload.setdefault("generated_at", datetime.now(timezone.utc).isoformat())
        payload.setdefault("open_count", 0)
        payload.setdefault("incidents", [])
        payload.setdefault("metrics", {
            "incident_frequency": 0,
            "open_count": int(payload.get("open_count", 0) or 0),
            "resolved_count": 0,
            "mean_mttr_seconds": None,
        })
        payload = _stamp_generator_git_sha(payload)

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
            "generated_at": datetime.now(timezone.utc).isoformat()
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
            data_dir=DATA_DIR,
            log_error=_log_signal_error,
        )

        # Kill authority + open incidents (same SSOT as data/health monitor)
        kill_fields = project_kill_switch_fields(load_kill_switch_payload(DATA_DIR))
        open_incidents = load_open_incidents_summary(DATA_DIR)
        health_data["kill_switch"] = kill_fields
        health_data["open_incidents"] = open_incidents

        # Additive bounded IC quality projection. Keep paper-control effect and
        # routing authority explicit; do not derive target allocations here.
        try:
            from src.monitor.ic_decay_monitor import (
                build_ic_decay_summary,
                compute_ic_decay_report,
                ic_control_projection,
            )

            control = ic_control_projection(kill_fields)
            health_data["ic_decay_summary"] = build_ic_decay_summary(
                compute_ic_decay_report(),
                evidence_generated_at=health_data["generated_at"],
                **control,
            )
        except Exception as exc:  # noqa: BLE001 — quality disclosure is additive
            logger.warning("IC quality summary projection skipped: %s", exc)

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

        # When dashboard regenerates from cleared incidents.json / kill SSOT,
        # also patch lagging monitor data/health.json so operators do not see
        # open_count=1 on the monitor report while incidents.json is 0.
        try:
            from src.monitor.health_check import reconcile_monitor_health_with_disk_ssot

            reconcile_monitor_health_with_disk_ssot(data_dir=DATA_DIR)
        except Exception as exc:  # noqa: BLE001 — never block public health write
            logger.warning("monitor health SSOT reconcile failed: %s", exc)

        health_data = _stamp_generator_git_sha(health_data)
        out_path = PUBLIC_DIR / "health.json"
        # Batch IE: serialize-once multi-dest for dashboard health.json
        # (public + repo soft-mirror @ 0o644). Never fan-out to private
        # DATA_DIR/health.json — that file is the monitor schema SSOT
        # (operational_readiness) and must not be overwritten by the
        # dashboard payload. Mirrors Batch IC public health merge contract.
        wrote = False
        try:
            from src.monitor.signal_authority import write_json_multi_dest

            result = write_json_multi_dest(
                health_data,
                public_path=out_path,
                private_path=None,
                soft_mirror_repo=True,
                repo_filename="health.json",
            )
            wrote = bool(result.wrote_public or result.wrote_repo)
            if result.skipped_reason:
                logger.warning(
                    "dashboard health multi-dest partial skip: %s",
                    result.skipped_reason,
                )
        except Exception as multi_exc:  # noqa: BLE001 — fall back below
            logger.warning(
                "dashboard health multi-dest failed (%s); falling back to save_results_json",
                multi_exc,
            )
            wrote = False
        if not wrote:
            save_results_json(health_data, output_path=str(out_path))

        return out_path
    
    def _generate_sector_momentum_signals(self, vix_level: Optional[float] = None) -> Optional[Dict]:
        """Generate sector rotation momentum signals from historical data."""
        try:
            from src.strategy.sector_momentum_calc import generate_sector_signals

            historical_path = PUBLIC_DIR.parent / "data" / "historical.json"

            # Pass through None when VIX is unknown — do not coerce to 0.0
            # (zero VIX looks like ultra-calm and misleads regime/threshold gates).
            signals = generate_sector_signals(historical_path, vix=vix_level)
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
            if isinstance(report, dict):
                report = _stamp_generator_git_sha(report)
            out_path = PUBLIC_DIR / "analytics.json"
            save_results_json(report, output_path=str(out_path))

            return out_path
        except (OSError, json.JSONDecodeError, KeyError, ValueError, TypeError, ImportError, RuntimeError) as e:
            # Fallback: empty analytics
            report = _stamp_generator_git_sha({
                "status": "error",
                "message": str(e),
                "generated_at": datetime.now(timezone.utc).isoformat(),
            })
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
            payload = _stamp_generator_git_sha(dashboard.to_dict())
            save_results_json(payload, output_path=str(public_path))
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
            sizing_data = _stamp_generator_git_sha({
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
                "generated_at": datetime.now(timezone.utc).isoformat(),
            })

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

            hedge_data = _stamp_generator_git_sha({
                **status,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "canonical_controller": "hedge_selector",
                "runtime_role": "diagnostic_cost_evidence",
                "live_authoritative": False,
                "routed": False,
            })

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
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }
            bl_data = _stamp_generator_git_sha(bl_data)

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

            turnover_data = _stamp_generator_git_sha({
                "schema_version": "turnover-validator/v1",
                "signals": production_signals,
                "synthetic_baselines": synthetic_baselines,
                "generated_at": datetime.now(timezone.utc).isoformat(),
            })

            out_path = PUBLIC_DIR / "turnover_validator.json"
            save_results_json(turnover_data, output_path=str(out_path))
            return out_path

        except SIGNAL_EXCEPTIONS as e:
            _log_signal_error("turnover_validator", e)
            return None

    @staticmethod
    def _normalize_gate_regime_name(regime: str | None) -> str:
        """Map live lowercase regimes to RegimeGate uppercase labels."""
        if not regime:
            return "NORMAL"
        name = str(regime).strip()
        if not name:
            return "NORMAL"
        # Gate rules use NORMAL/HIGH_VOL/…; live classify uses normal/vol_spike/…
        upper = name.upper()
        aliases = {
            "VOL_SPIKE": "HIGH_VOL",
            "VOLSPIKE": "HIGH_VOL",
            "HIGHVOL": "HIGH_VOL",
            "LOWVOL": "LOW_VOL",
            "LOW_VOL": "LOW_VOL",
            "HIGH_VOL": "HIGH_VOL",
            "CRISIS": "CRISIS",
            "RECOVERY": "RECOVERY",
            "NORMAL": "NORMAL",
        }
        return aliases.get(upper.replace("-", "_"), upper.replace("-", "_"))

    def _resolve_current_regime_for_gate(self) -> tuple[str, float, str]:
        """Resolve current regime + confidence for gate SSOT (not live order authority).

        Preference order:
        1. Live VIX/trend classifier via open DB connection (same as signals path)
        2. ensemble_voting on public/data signals.json
        3. regime_classifier_state.json (adaptive path; may be stale)
        4. Explicit default with disclosed source
        """
        # 1) Live VIX path when generator has a DB connection
        conn = getattr(self, "conn", None)
        if conn is not None:
            try:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT close FROM prices WHERE symbol = '^VIX' ORDER BY date DESC LIMIT 1"
                )
                vix_row = cursor.fetchone()
                vix_level = float(vix_row[0]) if vix_row and vix_row[0] is not None else None
                cursor.execute(
                    "SELECT regime FROM regime_log ORDER BY detected_at DESC LIMIT 1"
                )
                trend_row = cursor.fetchone()
                trend_regime = trend_row[0] if trend_row else "normal"
                live = classify_vix_regime(vix_level, trend_regime)
                conf = 0.7 if vix_level is not None else 0.55
                return self._normalize_gate_regime_name(live), conf, "classify_vix_regime"
            except Exception as exc:  # noqa: BLE001 — fall through to file SSOT
                logger.debug("regime_state: VIX path failed: %s", exc)

        # 2) Ensemble voting block on published signals
        for signals_path in (PUBLIC_DIR / "signals.json", DATA_DIR / "signals.json"):
            try:
                if not signals_path.exists():
                    continue
                with open(signals_path) as f:
                    signals = json.load(f)
                ensemble = signals.get("ensemble_voting") or {}
                if ensemble.get("regime") is not None:
                    conf_raw = ensemble.get("regime_confidence", 0.5)
                    try:
                        conf = float(conf_raw)
                    except (TypeError, ValueError):
                        conf = 0.5
                    return (
                        self._normalize_gate_regime_name(str(ensemble.get("regime"))),
                        conf,
                        "ensemble_voting",
                    )
                regime_block = signals.get("regime") or {}
                if isinstance(regime_block, dict) and regime_block.get("regime"):
                    return (
                        self._normalize_gate_regime_name(str(regime_block.get("regime"))),
                        0.6,
                        "signals.regime",
                    )
            except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
                logger.debug("regime_state: signals read failed (%s): %s", signals_path, exc)

        # 3) Adaptive classifier state (legacy parallel file)
        clf_path = DATA_DIR / "regime_classifier_state.json"
        try:
            if clf_path.exists():
                with open(clf_path) as f:
                    clf = json.load(f)
                regime = clf.get("current_regime") or clf.get("regime")
                if regime:
                    conf_raw = clf.get("confidence", 0.5)
                    if isinstance(clf.get("history"), list) and clf["history"]:
                        last = clf["history"][-1]
                        if isinstance(last, dict) and last.get("confidence") is not None:
                            conf_raw = last.get("confidence")
                    try:
                        conf = float(conf_raw)
                    except (TypeError, ValueError):
                        conf = 0.5
                    return (
                        self._normalize_gate_regime_name(str(regime)),
                        conf,
                        "regime_classifier_state",
                    )
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            logger.debug("regime_state: classifier read failed: %s", exc)

        return "NORMAL", 0.5, "default_missing_state"

    def _persist_regime_state(
        self,
        regime_name: str,
        confidence: float,
        source: str,
    ) -> Path:
        """Write DATA_DIR/regime_state.json SSOT for gate + graduation consumers."""
        regime_file = DATA_DIR / "regime_state.json"
        history: list = []
        previous = None
        if regime_file.exists():
            try:
                with open(regime_file) as f:
                    prior = json.load(f)
                previous = prior.get("regime")
                hist = prior.get("history")
                if isinstance(hist, list):
                    history = hist[-49:]  # keep last 50 after append
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                history = []

        now_iso = datetime.now().isoformat()
        history.append(
            {
                "timestamp": now_iso,
                "regime": regime_name,
                "confidence": confidence,
                "source": source,
            }
        )
        payload = {
            "regime": regime_name,
            "confidence": confidence,
            "source": source,
            "previous_regime": previous,
            "updated_at": now_iso,
            "schema_version": "regime-state/v1",
            "note": (
                "SSOT for dashboard regime_gate + graduation regime_coverage; "
                "not live order-routing authority (see regime_authority / target_allocations)."
            ),
            "history": history,
        }
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        save_results_json(payload, output_path=str(regime_file))

        # Append one regime_log line so graduation coverage can accumulate over cycles
        try:
            log_path = DATA_DIR / "regime_log.json"
            with open(log_path, "a", encoding="utf-8") as logf:
                logf.write(
                    json.dumps(
                        {
                            "regime": regime_name,
                            "confidence": confidence,
                            "source": source,
                            "detected_at": now_iso,
                        }
                    )
                    + "\n"
                )
        except OSError as exc:
            logger.debug("regime_state: regime_log append failed: %s", exc)

        return regime_file

    def generate_regime_gate_json(self) -> Optional[Path]:
        """Generate regime gate status data for dashboard."""
        try:
            from src.signals.regime_gate import RegimeGate

            gate = RegimeGate()
            summary = gate.get_gate_summary()

            # Resolve + persist SSOT so gate/graduation never depend on a dead path
            regime_name, regime_confidence, confidence_source = (
                self._resolve_current_regime_for_gate()
            )
            self._persist_regime_state(regime_name, regime_confidence, confidence_source)

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
                "confidence_source": confidence_source,
                "gate_rules": [
                    {"signal_name": sig, "off_regimes": sorted(regimes), "is_active": sig in active_signals}
                    for sig, regimes in summary.items()
                ],
                "active_signals": active_signals,
                "inactive_signals": inactive_signals,
                "min_dwell_days": gate.min_dwell_days,
                "generated_at": datetime.now(timezone.utc).isoformat(),
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

            gate_data = _stamp_generator_git_sha(gate_data)
            out_path = PUBLIC_DIR / "regime_gate.json"
            save_results_json(gate_data, output_path=str(out_path))
            return out_path

        except SIGNAL_EXCEPTIONS as e:
            _log_signal_error("regime_gate", e)
            return None

    # Cache last-known regime for _is_msm_gated resilience
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

            tsmom_data = _stamp_generator_git_sha({
                "composite_signal": float(np.mean([s.signal for s in signals])) if signals else 0.0,
                "speed_breakdown": speed_breakdown,
                "position_recommendation": "long" if np.mean([s.signal for s in signals]) > 0.1 else ("short" if np.mean([s.signal for s in signals]) < -0.1 else "neutral") if signals else "neutral",
                "confidence": min(1.0, abs(np.mean([s.vol_scaled_position for s in signals]))) if signals else 0.0,
                "standalone_sharpe": 0.96,
                "overlay_sharpe": 0.93,
                "health_score": 0.55,
                "is_gated_off": self._is_msm_gated(),
                "generated_at": datetime.now(timezone.utc).isoformat(),
            })

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
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }
            rv_data = _stamp_generator_git_sha(rv_data)

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
        """Generate graduation readiness progress for dashboard (dual private+public)."""
        return refresh_graduation_dual_surfaces(
            public_dir=PUBLIC_DIR,
            data_dir=DATA_DIR,
            paper_trading_builder=self._paper_trading_summary_for_dashboard,
            display_value=self._graduation_display_value,
        )


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
            data["generated_at"] = datetime.now(timezone.utc).isoformat()
            data = _stamp_generator_git_sha(data)
            save_results_json(data, output_path=str(output_path))
            return output_path

        except ImportError:
            logger.warning("scipy not available — skipping risk decomposition")
            fallback = _stamp_generator_git_sha({
                "status": "unavailable",
                "reason": "scipy not installed",
                "generated_at": datetime.now(timezone.utc).isoformat(),
            })
            save_results_json(fallback, output_path=str(output_path))
            return output_path

        except (OSError, ValueError, TypeError, RuntimeError) as e:
            logger.warning("Risk decomposition failed: %s", e)
            fallback = _stamp_generator_git_sha({
                "status": "error",
                "reason": str(e),
                "generated_at": datetime.now(timezone.utc).isoformat(),
            })
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
            # existing dashboard consumers. Task 5B: content files are already
            # in place; commit the index LAST, atomically, with generation
            # identity so an interrupted generation never advances it.
            index = build_public_data_index(paths, public_dir=PUBLIC_DIR)
            from src.monitor.health_check import commit_public_index

            commit_public_index(
                index,
                index_path=PUBLIC_DIR / "index.json",
                generation_id=_new_generation_id(),
            )
            _mirror_public_data_contract_files_to_dist(PUBLIC_DIR)
        finally:
            self.close()

        logger.info("Dashboard generation complete")

        # Option C (operator-approved 2026-08-11): record the public artifact
        # surface as an immutable generation + atomically flip the current
        # pointer. Guarded: a store failure must never fail the job.
        try:
            from src.dashboard.generation_store import GenerationStore

            GenerationStore(public_dir=PUBLIC_DIR).record(run_id=_new_generation_id())
        except Exception as e:  # noqa: BLE001 — observability must not fail the run
            logger.warning("generation record skipped (non-blocking): %s", e)

def _new_generation_id() -> str:
    """One generation identity for the committed public-data manifest."""
    import uuid as _uuid

    return (
        f"gen-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-"
        f"{_uuid.uuid4().hex[:8]}"
    )


def refresh_graduation_dual_surfaces(
    *,
    public_dir: Path | None = None,
    data_dir: Path | None = None,
    paper_trading_builder=None,
    display_value=None,
) -> Optional[Path]:
    """Compute graduation once; write public graduation.json + private report.

    Batch BP: single compute path so private/public readiness and CB criterion
    cannot invent different consecutive_ok values. Safe to call from health
    when ``.circuit_breaker.json`` consecutive_ok changes (no DB required).
    """
    try:
        from src.strategy.graduation_checklist import GraduationChecklist
        from src.paths import DATA_DIR as _DEFAULT_DATA, PUBLIC_DATA_DIR as _DEFAULT_PUB

        pub = Path(public_dir) if public_dir is not None else Path(_DEFAULT_PUB)
        # DATA_DIR for checklist is module-level; callers monkeypatch as needed.
        _ = data_dir  # reserved for future path injection

        checklist = GraduationChecklist()
        state = checklist._load_state()
        results = checklist.check(state)
        score = checklist.readiness_score(results)
        is_ready = checklist.is_graduation_ready(results)

        _display = display_value or (
            lambda v: str(v) if v is not None else ""
        )
        criteria_progress = []
        for name, result in results.items():
            criteria_progress.append({
                "name": name,
                "passed": result.passed,
                "value": result.value,
                "required": result.required,
                "description": result.description,
                "id": name,
                "label": result.description or name,
                "threshold": _display(result.required),
            })

        trading_days_result = results.get("min_trading_days")
        n_days = trading_days_result.value if trading_days_result is not None else 0
        min_trading_days = (
            trading_days_result.required
            if trading_days_result is not None
            else checklist.criteria["min_trading_days"]["value"]
        )
        manual_approval = results.get("manual_approval")
        if paper_trading_builder is not None:
            paper_trading = paper_trading_builder(
                state,
                days_elapsed=n_days,
                days_required=min_trading_days,
            )
        else:
            paper_trading = {
                "days_elapsed": n_days,
                "days_required": min_trading_days,
            }

        cb_result = results.get("circuit_breaker_confidence")
        graduation_data = _stamp_generator_git_sha({
            "readiness_score": score,
            "is_graduation_ready": is_ready,
            "manual_approval_required": True,
            "manual_approval_pending": not bool(
                manual_approval and manual_approval.passed
            ),
            "trading_days": n_days,
            "min_trading_days": min_trading_days,
            "criteria_met": sum(
                1 for n, r in results.items() if n != "manual_approval" and r.passed
            ),
            "criteria_total": sum(1 for n in results if n != "manual_approval"),
            "criteria": criteria_progress,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "readiness_pct": score,
            "eligible": is_ready,
            "paper_trading": paper_trading,
            "circuit_breaker_ssot": ".circuit_breaker.json",
            "circuit_breaker_consecutive_ok": (
                cb_result.value if cb_result is not None else None
            ),
        })

        out_path = pub / "graduation.json"
        save_results_json(graduation_data, output_path=str(out_path))
        try:
            checklist.save_report(results)
        except Exception as priv_exc:  # noqa: BLE001
            logger.warning("Private graduation report dual-write failed: %s", priv_exc)
        return out_path
    except Exception as e:  # noqa: BLE001 — health path must not crash
        try:
            _log_signal_error("graduation", e)
        except Exception:  # noqa: BLE001
            logger.warning("graduation dual-surface refresh failed: %s", e)
        return None

if __name__ == "__main__":
    from src.utils.log_config import configure_logging
    configure_logging()
    with DashboardGenerator() as gen:
        gen.run()

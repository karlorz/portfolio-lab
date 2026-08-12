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
from src.monitor.alerting import webhook_config_state
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
from src.dashboard.sections_alerts import _AlertsSectionsMixin
from src.dashboard.sections_overlay import _OverlaySectionsMixin
from src.dashboard.sections_stacking import _StackingSectionsMixin
from src.dashboard.sections_graduation import _GraduationExplainabilitySectionsMixin
from src.dashboard.sections_regime_gate import _RegimeGateStateMixin
from src.dashboard.sections_data_loaders import _DataLoaderSectionsMixin

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


class DashboardGenerator(_EnsembleSectionsMixin, _HedgeSectionsMixin, _RegimeAuthorityMixin, _AlertsSectionsMixin, _OverlaySectionsMixin, _StackingSectionsMixin, _GraduationExplainabilitySectionsMixin, _RegimeGateStateMixin, _DataLoaderSectionsMixin):
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


    def _build_base_signal_sections(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Build the core signal sections before dashboard-level metadata."""
        return self._get_signal_section_builder().build_base_sections(context)

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
        
        webhook_configured, webhook_source = webhook_config_state()
        output = _stamp_generator_git_sha({
            "alerts": sorted(alerts, key=lambda x: x.get("timestamp", "") or "", reverse=True),
            "count": len(alerts),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "alerting": {
                "webhook_configured": webhook_configured,
                "webhook_source": webhook_source,
            },
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



    def generate_graduation_json(self) -> Optional[Path]:
        """Generate graduation readiness progress for dashboard (dual private+public)."""
        return refresh_graduation_dual_surfaces(
            public_dir=PUBLIC_DIR,
            data_dir=DATA_DIR,
            paper_trading_builder=self._paper_trading_summary_for_dashboard,
            display_value=self._graduation_display_value,
        )




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

#!/usr/bin/env python3
"""
Portfolio-Lab Alpha: Dashboard Generator
Creates static dashboard from SQLite data for Vite/React app consumption.
"""

import json
import sqlite3
import logging
import os
from collections import deque
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

from src.paths import BASE_ALLOCATION, YIELDS_JSON, DATA_DIR, PUBLIC_DATA_DIR, MARKET_DB, REGIME_OVERRIDES, sqlite_connect
from src.utils import safe_get, classify_vix_regime
from src.backtest.metrics import save_results_json
from src.dashboard.public_data_index import build_public_data_index
from src.monitor.hermes_cron import (
    combine_scheduler_backends,
    load_hermes_portfolio_cron_jobs,
    load_local_cron_jobs,
    resolve_hermes_cron_jobs_path,
)
from src.monitor.signal_schemas import validate_all_signals, validate_signal

__all__ = [
    "DashboardGenerator",
    "PUBLIC_DIR",
    "DB_PATH",
]

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
        summary["cron_job_count"] = len(cron_jobs)
        summary["failed_cron_jobs"] = sum(
            1 for job in cron_jobs if isinstance(job, dict) and job.get("status") == "error"
        )

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

    return summary


def _classify_market_data_freshness(market_lag_days: int) -> str:
    """Classify a symbol by lag versus the provider's latest available date."""
    if market_lag_days <= 1:
        return "fresh"
    if market_lag_days <= 3:
        return "stale"
    return "critical"


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
            with open(perf_log) as f:
                for line in deque(f, maxlen=500):
                    try:
                        entry = json.loads(line)
                        paper_perf.append({
                            "t": entry.get("timestamp", "")[:10],
                            "v": entry.get("total_value", 0),
                            "r": entry.get("daily_return", 0)
                        })
                    except (OSError, json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
                        logger.warning("Failed to parse performance log entry: %s", e)

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

            # Phase 1: Resolve previously staged predictions
            if monitor.has_staged_predictions():
                staged_date = monitor.get_staged_date()
                if staged_date:
                    # Compute SPY forward return from staged date to latest
                    cursor = self.conn.cursor()
                    cursor.execute(
                        "SELECT date, close FROM prices WHERE symbol = 'SPY' "
                        "AND date >= ? ORDER BY date ASC LIMIT 1",
                        (staged_date,),
                    )
                    start_row = cursor.fetchone()
                    cursor.execute(
                        "SELECT date, close FROM prices WHERE symbol = 'SPY' "
                        "ORDER BY date DESC LIMIT 1",
                    )
                    end_row = cursor.fetchone()
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
            if isinstance(fred, dict) and fred.get("confidence") is not None:
                predictions["fred_macro"] = float(fred["confidence"])

            if predictions:
                monitor.stage_predictions(
                    predictions,
                    datetime.now().strftime("%Y-%m-%d"),
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

    def generate_signals_json(self) -> Path:
        """Generate current signals and allocations."""
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
        base_alloc = BASE_ALLOCATION
        target_alloc = REGIME_OVERRIDES.get(current_regime) or base_alloc
        
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

        # Add factor rotation signals if engine available
        factor_rotation_signal = None
        factor_rotation_result = None
        try:
            from src.strategy.factor_rotation import FactorMomentumEngine
            engine = FactorMomentumEngine()
            factor_rotation_result = engine.evaluate()
            if factor_rotation_result and "error" not in factor_rotation_result:
                factor_rotation_signal = {
                    "selected_factors": factor_rotation_result.get("selected_factors", []),
                    "allocation": factor_rotation_result.get("allocation", {}),
                    "signal_strength": factor_rotation_result.get("signal_strength", 0.0),
                    "recommendation": factor_rotation_result.get("recommendation", {}),
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
                source_breakdown = []
                for src in ensemble_result.source_votes:
                    source_breakdown.append({
                        "source": src.source.value if hasattr(src.source, 'value') else str(src.source),
                        "direction": "bullish" if src.value > 0 else ("bearish" if src.value < 0 else "neutral"),
                        "strength": round(abs(src.value), 3),
                        "confidence": round(src.confidence, 3),
                        "weight": round(src.weight, 3),
                    })
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
                    "num_sources": ensemble_result.num_sources,
                    "n_eff": round(getattr(ensemble_result, 'n_eff', 0), 2),
                    "weight_entropy": round(getattr(ensemble_result, 'weight_entropy', 0), 4),
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
                    'remaining_budget_pct': gate_result.metadata.get('remaining_budget_pct', 100),
                    'status': gate.get_status(),
                }
            else:
                # No positions — use gate status only
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
                    'remaining_budget_pct': 100,
                    'status': gate.get_status(),
                }
        except (OSError, json.JSONDecodeError, KeyError, ValueError, TypeError, ImportError, RuntimeError) as e:
            logger.warning("Dashboard generation error: %s", e)

        # Add alternative data signals (v2.60 Phase 3)
        alternative_data_signal = None
        try:
            alt_data_file = DATA_DIR / "signals" / "alternative_data_latest.json"
            if alt_data_file.exists():
                with open(alt_data_file) as f:
                    alt_data_raw = json.load(f)
                    alternative_data_signal = {
                        "regime": alt_data_raw.get("regime"),
                        "probability": alt_data_raw.get("probability"),
                        "confidence": alt_data_raw.get("confidence"),
                        "timestamp": alt_data_raw.get("timestamp"),
                        "components": {
                            "earnings": {
                                "score": safe_get(alt_data_raw, "raw_data", "earnings_sentiment"),
                                "confidence": safe_get(alt_data_raw, "raw_data", "earnings_confidence"),
                                "weight": safe_get(alt_data_raw, "raw_data", "weights", "earnings")
                            },
                            "news": {
                                "score": safe_get(alt_data_raw, "raw_data", "news_sentiment"),
                                "confidence": safe_get(alt_data_raw, "raw_data", "news_confidence"),
                                "weight": safe_get(alt_data_raw, "raw_data", "weights", "news")
                            },
                            "jobs": {
                                "score": safe_get(alt_data_raw, "raw_data", "jobs_signal"),
                                "confidence": safe_get(alt_data_raw, "raw_data", "jobs_confidence"),
                                "weight": safe_get(alt_data_raw, "raw_data", "weights", "jobs")
                            },
                            "social": {
                                "score": safe_get(alt_data_raw, "raw_data", "social_sentiment"),
                                "confidence": safe_get(alt_data_raw, "raw_data", "social_confidence"),
                                "weight": safe_get(alt_data_raw, "raw_data", "weights", "social")
                            }
                        },
                        "composite_score": safe_get(alt_data_raw, "raw_data", "composite_score"),
                        "z_score": safe_get(alt_data_raw, "raw_data", "z_score"),
                        "sources_count": safe_get(alt_data_raw, "raw_data", "sources_count"),
                        "data_freshness_hours": safe_get(alt_data_raw, "raw_data", "data_freshness_hours")
                    }
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
            }
        except SIGNAL_EXCEPTIONS as e:
            _log_signal_error("behavioral_sentiment", e)

        # Stacking ensemble dashboard data (v3.10)
        stacking_ensemble_dashboard = None
        try:
            from src.signals.stacking_integrator import StackingIntegrator

            integrator = StackingIntegrator()
            # Try to get a prediction (may use fallback if no model)
            prediction = integrator.predict({})
            stacking_ensemble_dashboard = {
                "active": True,
                "stacking_available": integrator.model is not None,
                "prediction_direction": prediction.direction if prediction else "neutral",
                "confidence": prediction.confidence if prediction else 0.5,
                "probability_bullish": prediction.probability_bullish if prediction else 0.33,
                "probability_bearish": prediction.probability_bearish if prediction else 0.33,
                "probability_neutral": prediction.probability_neutral if prediction else 0.34,
                "fallback_used": prediction.fallback_used if prediction else True,
                "model_version": prediction.model_version if prediction else "unknown",
                "voting_accuracy": 0.65,
                "stacking_accuracy": 0.76,
                "feature_count": 102,
                "latency_ms": prediction.latency_ms if prediction else 0.0,
                "backtest_finding": (
                    "+11% accuracy produces negligible Sharpe gain (2021-2026). "
                    "Signal frequency and shift magnitude are binding constraints."
                ),
            }
        except SIGNAL_EXCEPTIONS as e:
            _log_signal_error("stacking_ensemble", e)

        # Factor rotation dashboard data (v3.00) — reuses factor_rotation_result from above
        factor_rotation_dashboard = None
        try:
            from src.strategy.factor_rotation import FactorMomentumEngine

            if factor_rotation_result is not None and "error" not in factor_rotation_result:
                allocations = factor_rotation_result.get("allocation", {})
                factor_rotation_dashboard = {
                    "active": True,
                    "selected_factors": factor_rotation_result.get("selected_factors", []),
                    "signal_strength": round(factor_rotation_result.get("signal_strength", 0.0), 2),
                    "factor_allocations": allocations,
                    "backtest_finding": (
                        "Factor rotation reduces MaxDD by 5.8pp (2021-2026). "
                        "Defensive tool — best in high-vol regimes (Sharpe 1.474)."
                    ),
                }
        except SIGNAL_EXCEPTIONS as e:
            _log_signal_error("factor_rotation_dashboard", e)

        # Merge overlay dashboard data (collar, crypto, calendar, kurtosis, etc.)
        overlay_data = self._get_overlay_data()

        # Hedge selector recommendation
        hedge_selector_signal = None
        try:
            hedge_selector_signal = self._get_hedge_selector_signal(vix_level, current_regime)
        except MONITOR_EXCEPTIONS as e:
            _log_signal_error("hedge_selector", e)

        output = _attach_signal_metadata({
            "regime": validate_signal("regime", regime_data),
            "target_allocations": target_alloc,
            "current_positions": positions,
            "cash": round(cash, 2),
            "total_value": round(total_value, 2),
            "latest_prices": latest,
            "recent_orders": list(reversed(orders)),
            "ml_signals": self._generate_ml_signals(),
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
        })

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

        # Signal staleness detection (production readiness)
        output["staleness"] = self._check_signal_staleness(output)

        # Apply staleness-weighted decay to ensemble weights
        output = self._apply_staleness_decay(output)

        # FRED-MD macro regime signal
        try:
            from src.data.fred_data import get_fred_signal
            fred_signal = get_fred_signal()
            output["fred_macro"] = validate_signal("fred_macro", {
                "regime": fred_signal.regime,
                "confidence": fred_signal.confidence,
                "recession_probability": fred_signal.recession_probability,
                "inflation_pressure": fred_signal.inflation_pressure,
                "monetary_stance": fred_signal.monetary_stance,
                "manufacturing_health": fred_signal.manufacturing_health,
                "credit_conditions": fred_signal.credit_conditions,
                "indicators": fred_signal.indicators,
                "timestamp": fred_signal.timestamp,
            })
        except MONITOR_EXCEPTIONS as e:
            _log_signal_error("fred_macro", e)
            output["fred_macro"] = {
                "regime": "UNKNOWN",
                "confidence": 0.0,
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

        # Health check report
        try:
            from src.monitor.health_check import run_health_check
            output["health"] = _compact_health_summary(run_health_check())
        except Exception as e:
            output["health"] = _compact_health_summary({"status": "error", "error": str(e)})

        # Fire external alerts on staleness state transitions
        try:
            from src.monitor.alerting import check_staleness_and_alert
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

        out_path = PUBLIC_DIR / "signals.json"
        save_results_json(output, output_path=str(out_path), validator=validate_all_signals)

        return out_path

    def _load_broker_data(self) -> Dict:
        """Load broker position sync and order data for dashboard."""
        broker = {
            "connected": False,
            "positions": [],
            "drift": [],
            "recent_orders": [],
            "last_sync": None,
            "kill_switch": False,
        }

        # Check position sync log (tail read only)
        sync_log = DATA_DIR / "position_sync.jsonl"
        if sync_log.exists():
            try:
                with open(sync_log) as f:
                    tail = deque(f, maxlen=1)
                if tail:
                    last = json.loads(tail[0])
                    broker["connected"] = True
                    broker["last_sync"] = last.get("timestamp")
                    broker["positions"] = last.get("broker_positions", [])
                    broker["drift"] = last.get("drift", [])
            except (OSError, json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
                logger.warning("Failed to load position sync log: %s", e)

        # Check broker orders log (tail read only)
        orders_log = DATA_DIR / "broker_orders.jsonl"
        if orders_log.exists():
            try:
                with open(orders_log) as f:
                    recent = []
                    for line in deque(f, maxlen=10):
                        if line.strip():
                            recent.append(json.loads(line))
                    broker["recent_orders"] = list(reversed(recent))
            except (OSError, json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
                logger.warning("Failed to load broker orders log: %s", e)

        # Check kill switch
        kill_file = DATA_DIR / "kill_switch.json"
        if kill_file.exists():
            try:
                with open(kill_file) as f:
                    ks = json.load(f)
                broker["kill_switch"] = ks.get("enabled", False)
            except (OSError, json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
                logger.warning("Failed to load kill switch state: %s", e)

        return broker

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
        }

        # Compute conformal CVaR cross-check from SPY returns
        try:
            from src.monitor.conformal_risk import conformal_cvar, conformal_var
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
                if garch_cvar["conformal_var_95"] != 0:
                    garch_cvar["conformal_cvar_ratio"] = round(
                        garch_cvar["conformal_cvar_95"]
                        / garch_cvar["conformal_var_95"], 3,
                    )
        except (ImportError, ValueError, IndexError) as e:
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
        signals = {
            "available": False,
            "timestamp": None,
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
                    signals["available"] = True
                    signals["timestamp"] = datetime.now().isoformat()
                    signals["features"] = {
                        sym: {
                            "vix_level": feat.get("vix_level"),
                            "trend_direction": feat.get("trend_direction"),
                            "price_vs_sma20": feat.get("price_vs_sma20"),
                            "return_5d": feat.get("return_5d"),
                            "spy_correlation": feat.get("spy_correlation_20d"),
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
                    signals["grid_search"] = {
                        "available": True,
                        "timestamp": latest.get("timestamp"),
                        "top_allocation": latest.get("allocations"),
                        "sharpe": latest.get("sharpe"),
                        "volatility": latest.get("volatility"),
                    }
            except (OSError, json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
                logger.warning("Failed to load grid search results: %s", e)

        return signals
    
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
                "spread_history": spread_history
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

                    # Deduplicate to daily: keep last entry per calendar date
                    # (performance.jsonl contains intraday entries; raw count
                    # overstates trading days and inflates Sharpe)
                    daily_map: dict[str, dict] = {}
                    for idx, entry in enumerate(raw_entries):
                        ts = entry.get("timestamp", "")
                        date_key = ts[:10] if len(ts) >= 10 else ""
                        if not date_key:
                            # Fallback: entries without timestamps are
                            # treated as separate days
                            date_key = f"__no_ts_{idx}__"
                        daily_map[date_key] = entry
                    daily_entries = [daily_map[d] for d in sorted(daily_map)]

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
    
    def generate_alerts_json(self) -> Path:
        """Generate active alerts and notifications."""
        alerts = []
        
        # Check for promotion trigger
        promote_trigger = DATA_DIR / ".promote_to_live"
        if promote_trigger.exists():
            with open(promote_trigger) as f:
                data = json.load(f)
                alerts.append({
                    "level": "success",
                    "type": "graduation_candidate",
                    "title": "Paper Trading Graduation Ready",
                    "message": f"Sharpe: {safe_get(data, 'metrics', 'sharpe')}, ready for live approval",
                    "timestamp": data.get("timestamp"),
                    "requires_action": True
                })
        
        # Check for kill switch
        kill_file = DATA_DIR / "kill_switch.json"
        if kill_file.exists():
            with open(kill_file) as f:
                data = json.load(f)
                if data.get("enabled"):
                    mode = data.get("mode", "unknown")
                    alerts.append({
                        "level": "error",
                        "type": "kill_switch",
                        "title": f"{mode.upper()} Kill Switch Triggered",
                        "message": data.get("reason"),
                        "timestamp": data.get("timestamp"),
                        "requires_action": True
                    })
        
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
        
        # Check data quality
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
        
        output = {
            "alerts": sorted(alerts, key=lambda x: x.get("timestamp", ""), reverse=True),
            "count": len(alerts),
            "generated_at": datetime.now().isoformat()
        }
        
        out_path = PUBLIC_DIR / "alerts.json"
        save_results_json(output, output_path=str(out_path))
        
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
        
        # Get cron job status from project-local status file and Hermes, when available.
        scheduler_backends = {}
        cron_status_file = DATA_DIR / "cron_status.json"
        local_jobs, local_backend = load_local_cron_jobs(cron_status_file)
        health_data["cron_jobs"].extend(local_jobs)
        scheduler_backends["local"] = local_backend

        hermes_jobs_path = _resolve_hermes_cron_jobs_path()
        if hermes_jobs_path is not None:
            hermes_jobs, hermes_backend = load_hermes_portfolio_cron_jobs(hermes_jobs_path)
            health_data["cron_jobs"].extend(hermes_jobs)
            scheduler_backends["hermes"] = hermes_backend

        health_data["scheduler_status"] = combine_scheduler_backends(scheduler_backends)
        
        # Get data freshness from SQLite
        cursor = self.conn.cursor()
        cursor.execute("SELECT MAX(date) FROM prices")
        latest_market_date = cursor.fetchone()[0]
        latest_market_dt = None
        if latest_market_date:
            try:
                latest_market_dt = datetime.strptime(latest_market_date, "%Y-%m-%d")
            except (ValueError, TypeError) as e:
                logger.warning(
                    "Failed to parse latest market freshness date '%s': %s",
                    latest_market_date,
                    e,
                )

        cursor.execute("""
            SELECT symbol, MAX(date) as last_date 
            FROM prices 
            GROUP BY symbol
        """)
        for row in cursor.fetchall():
            sym, last_date = row
            if last_date:
                try:
                    last_dt = datetime.strptime(last_date, "%Y-%m-%d")
                    days_stale = (datetime.now() - last_dt).days
                    market_lag_days = (
                        max((latest_market_dt - last_dt).days, 0)
                        if latest_market_dt is not None
                        else days_stale
                    )
                    health_data["data_freshness"][sym] = {
                        "last_update": last_date,
                        "days_stale": days_stale,
                        "market_lag_days": market_lag_days,
                        "latest_available_market_date": latest_market_date,
                        "status": _classify_market_data_freshness(market_lag_days)
                    }
                except (ValueError, TypeError) as e:
                    logger.warning("Failed to parse data freshness date '%s': %s", last_date, e)

        # Get signal health from SignalHealthTracker
        try:
            from src.signals.health_tracker import SignalHealthTracker
            tracker = SignalHealthTracker()
            signal_health_report = tracker.get_health_report()
            health_data["signal_health"] = {
                "timestamp": signal_health_report.get("timestamp"),
                "summary": signal_health_report.get("summary", {}),
                "scores": signal_health_report.get("scores", {}),
                "alerts": signal_health_report.get("alerts", []),
                "overall_health": signal_health_report.get("overall_health", "unknown")
            }
        except SIGNAL_EXCEPTIONS as e:
            _log_signal_error("signal_health", e)
            health_data["signal_health"] = {
                "error": f"Failed to get signal health: {str(e)}",
                "status": "unavailable"
            }

        try:
            from src.data.fred_data import get_fred_md_cache_health
            from src.monitor.fred_readiness import assess_fred_readiness

            health_data["fred_readiness"] = assess_fred_readiness(get_fred_md_cache_health())
        except SIGNAL_EXCEPTIONS as e:
            _log_signal_error("fred_readiness", e)
            health_data["fred_readiness"] = {
                "status": "warning",
                "readiness": "unknown",
                "ready": True,
                "blocking": False,
                "reason": "readiness_check_unavailable",
                "message": f"FRED readiness check unavailable: {str(e)}",
                "remediation": "Verify fredapi availability and FRED readiness dependencies.",
            }

        try:
            from src.monitor.data_pipeline_slo import (
                build_data_pipeline_slo,
                load_public_index,
                load_rebalance_health,
                load_signal_staleness,
                load_source_manifest,
            )
            rebalance_health = load_rebalance_health(PUBLIC_DIR)

            health_data["data_pipeline_slo"] = build_data_pipeline_slo(
                health_data=health_data,
                source_manifest=load_source_manifest(PUBLIC_DIR),
                public_index=load_public_index(PUBLIC_DIR),
                signal_staleness=load_signal_staleness(PUBLIC_DIR),
                alpaca_feed_entitlement=rebalance_health.get("alpaca_feed_entitlement"),
                market_data_consistency=rebalance_health.get("market_data_consistency"),
            )
        except (ImportError, OSError, ValueError, TypeError) as e:
            health_data["data_pipeline_slo"] = {
                "schema_version": "data-pipeline-slo/v1",
                "status": "warning",
                "top_dimension": "unknown",
                "error": str(e),
            }

        # Overall system health
        stale_count = sum(1 for d in health_data["data_freshness"].values() if d.get("status") != "fresh")
        failed_jobs = sum(1 for j in health_data["cron_jobs"] if j.get("status") == "error")
        scheduler_status = health_data.get("scheduler_status", {}).get("status")
        slo_status = health_data.get("data_pipeline_slo", {}).get("status")
        backend_error = any(
            backend.get("status") == "error"
            for backend in health_data.get("scheduler_status", {}).get("backends", {}).values()
        )
        
        if health_data["system_status"] not in {"warning", "critical", "degraded"}:
            health_data["system_status"] = "healthy"
        if backend_error:
            health_data["system_status"] = "degraded"
        elif (
            scheduler_status in {"degraded", "warning", "unavailable"}
            or slo_status == "warning"
            or failed_jobs > 0
            or stale_count > 5
        ):
            health_data["system_status"] = "warning"
        if slo_status == "critical" or failed_jobs > 2 or stale_count > 10:
            health_data["system_status"] = "critical"
        
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
            return gen.OUTPUT_PATH
        except SIGNAL_EXCEPTIONS as e:
            _log_signal_error("overlay_dashboard_generate", e)
            return None

    def generate_adaptive_sizing_json(self) -> Optional[Path]:
        """Generate adaptive sizing data for dashboard."""
        try:
            from src.strategy.adaptive_sizing import AdaptiveSizer

            sizer = AdaptiveSizer()
            decision = sizer.compute_allocation()

            sizing_data = {
                "base_allocation": decision.base_allocation,
                "adjusted_allocation": decision.adjusted_allocation,
                "adjustments": decision.adjustments,
                "regime_adjustment": decision.regime_adjustment,
                "volatility_adjustment": decision.volatility_adjustment,
                "signal_adjustment": decision.signal_adjustment,
                "drawdown_adjustment": decision.drawdown_adjustment,
                "factors": asdict(decision.factors) if hasattr(decision.factors, '__dataclass_fields__') else {},
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
                bl_weights = {k.lower(): v for k, v in BASE_ALLOCATION.items()}
            if posterior_returns is None:
                posterior_returns = {}

            # Use base allocation as prior
            prior = {k.lower(): v for k, v in BASE_ALLOCATION.items()}

            # Build views list for panel consumption
            view_list = []
            if hasattr(views, 'absolute_views') and views.absolute_views:
                abs_views = views.absolute_views if isinstance(views.absolute_views, dict) else dict(zip(views.symbols, views.absolute_views))
                for i, sym in enumerate(views.symbols):
                    ret = abs_views.get(sym, 0.0)
                    conf = views.view_confidences[i] if i < len(views.view_confidences) else 0.5
                    view_list.append({
                        "signal_name": "ensemble_consensus",
                        "asset": sym,
                        "direction": "bullish" if ret > 0 else ("bearish" if ret < 0 else "neutral"),
                        "confidence": round(conf, 3),
                        "expected_return_delta": round(ret, 6),
                    })

            bl_data = {
                "prior_weights": prior,
                "posterior_weights": bl_weights,
                "posterior_returns": posterior_returns,
                "views": view_list,
                "tau": bl_input.get("tau", 0.15),
                "view_confidence_method": "idzorek",
                "optimization_available": result is not None,
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
            from src.strategy.turnover_validator import TurnoverValidator

            validator = TurnoverValidator()
            diagnostics = validator.get_state_diagnostics()

            turnover_data = {
                **diagnostics,
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
            all_signals = list(summary.keys())
            active_signals = gate.get_active_signal_names(
                all_signals + ["alt_data", "cross_asset_rv", "unified_overlay"],
                regime_name,
            )
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

    def _get_hedge_selector_signal(self, vix_level: Optional[float], regime: str) -> Optional[Dict]:
        """Get hedge selector recommendation for dashboard."""
        if vix_level is None:
            return None
        try:
            from src.strategy.hedge_selector import HedgeSelector
            selector = HedgeSelector()
            # Estimate confidence based on regime stability
            regime_confidence = 0.8 if regime in ["normal", "crisis"] else 0.6
            rec = selector.select(
                vix_level=vix_level,
                regime_confidence=regime_confidence,
                regime_label=regime
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
    }

    @staticmethod
    def _normalized_signal_timestamp(signal_block: Any, preferred_field: str) -> str | None:
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

            ts_str = self._normalized_signal_timestamp(signal_block, ts_field)
            signal_timestamps[signal_key] = ts_str

            if ts_str is None:
                if (
                    signal_key in self.OPTIONAL_SIGNAL_STALENESS_KEYS
                    and self._is_unavailable_signal_block(signal_block)
                ):
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
                age_seconds = (now - ts).total_seconds()
                age_hours = age_seconds / 3600.0
                signal_age_hours[signal_key] = round(age_hours, 2)

                # Exponential decay: fresh signals get 1.0, stale signals approach 0.0
                decay = _math.exp(-age_hours / tau_hours) if tau_hours > 0 else 1.0
                staleness_decay[signal_key] = round(decay, 4)

                if age_seconds > ttl_seconds:
                    stale_signals.append(signal_key)
            except (ValueError, TypeError):
                stale_signals.append(signal_key)
                signal_age_hours[signal_key] = None
                staleness_decay[signal_key] = 0.0

        healthy_count = len(timestamped_signals) - len(stale_signals) - len(unavailable_signals)
        return {
            "stale_signals": stale_signals,
            "unavailable_signals": unavailable_signals,
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
            total_weight = sum(s.get("weight", 0.0) for s in ensemble["source_breakdown"])
            if total_weight > 0:
                weighted_sum = sum(
                    s.get("value", 0.0) * s.get("weight", 0.0)
                    for s in ensemble["source_breakdown"]
                )
                ensemble["weighted_consensus"] = round(weighted_sum / total_weight, 4)
                ensemble["total_weight_after_decay"] = round(total_weight, 4)

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
                "signal_value": signal.composite_signal if hasattr(signal, 'composite_signal') else 0.0,
                "pairs": [p.to_dict() for p in signal.pair_signals] if hasattr(signal, 'pair_signals') and signal.pair_signals else [],
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

    def generate_graduation_json(self) -> Optional[Path]:
        """Generate graduation readiness progress for dashboard.

        Reads from .graduation_report.json (generated by graduation_checklist)
        and provides structured progress data for the dashboard UI.
        """
        try:
            from src.strategy.graduation_checklist import GraduationChecklist

            checklist = GraduationChecklist()
            state = checklist._load_state()
            results = checklist.check(state)
            score = checklist.readiness_score(results)
            is_ready = checklist.is_graduation_ready(results)

            # Build progress data for dashboard
            criteria_progress = []
            for name, result in results.items():
                criteria_progress.append({
                    "name": name,
                    "passed": result.passed,
                    "value": result.value,
                    "required": result.required,
                    "description": result.description,
                })

            # Estimate trading days
            portfolio = state.get("portfolio", {})
            history = portfolio.get("history", [])
            unique_dates = set()
            for entry in history:
                ts = entry.get("timestamp", "")
                date_key = ts[:10] if len(ts) >= 10 else ts
                unique_dates.add(date_key)
            n_days = len(unique_dates)

            graduation_data = {
                "readiness_score": score,
                "is_graduation_ready": is_ready,
                "manual_approval_required": True,
                "manual_approval_pending": True,  # Always pending unless approval file exists
                "trading_days": n_days,
                "min_trading_days": GraduationChecklist.MIN_OBSERVATION_DAYS,
                "criteria_met": sum(1 for n, r in results.items() if n != "manual_approval" and r.passed),
                "criteria_total": sum(1 for n in results if n != "manual_approval"),
                "criteria": criteria_progress,
                "generated_at": datetime.now().isoformat(),
            }

            out_path = PUBLIC_DIR / "graduation.json"
            save_results_json(graduation_data, output_path=str(out_path))

            return out_path

        except SIGNAL_EXCEPTIONS as e:
            _log_signal_error("graduation", e)
            return None

    def generate_explainability_json(self) -> Optional[Path]:
        """Copy latest explainability data to public data directory.

        The explainability files are generated by portfolio_explainability.py
        and stored in data/explainability/. This method copies the latest
        dated file to public/data/explainability/explainability_latest.json.
        """
        try:
            import shutil
            source_dir = DATA_DIR / "explainability"
            target_dir = PUBLIC_DIR / "explainability"
            target_dir.mkdir(parents=True, exist_ok=True)

            # Find the latest dated explainability file
            dated_files = sorted(source_dir.glob("explainability_*.json"), reverse=True)
            if not dated_files:
                logger.warning("No explainability data files found in %s", source_dir)
                return None

            latest = dated_files[0]
            target = target_dir / "explainability_latest.json"
            shutil.copy2(latest, target)
            logger.info("Copied %s → %s", latest.name, target)
            return target

        except (OSError, ValueError, TypeError) as e:
            logger.warning("Failed to copy explainability data: %s", e)
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
            paths = [
                self.generate_performance_json(),
                self.generate_signals_json(),
                self.generate_stats_json(),
                self.generate_alerts_json(),
                self.generate_health_json(),
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

            for p in paths:
                if p:
                    logger.info("Generated: %s", p)

            # Create a versioned public-data manifest while keeping files[] for
            # existing dashboard consumers.
            index = build_public_data_index(paths, public_dir=PUBLIC_DIR)
            save_results_json(index, output_path=str(PUBLIC_DIR / "index.json"))
        finally:
            self.close()

        logger.info("Dashboard generation complete")

if __name__ == "__main__":
    from src.utils.log_config import configure_logging
    configure_logging()
    with DashboardGenerator() as gen:
        gen.run()

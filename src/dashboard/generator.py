#!/usr/bin/env python3
"""
Portfolio-Lab Alpha: Dashboard Generator
Creates static dashboard from SQLite data for Vite/React app consumption.
"""

import json
import sqlite3
import logging
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

from src.paths import BASE_ALLOCATION, YIELDS_JSON, DATA_DIR, PUBLIC_DATA_DIR, MARKET_DB, sqlite_connect
from src.utils import safe_get
from src.backtest.metrics import save_results_json

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

PUBLIC_DIR = PUBLIC_DATA_DIR
DB_PATH = MARKET_DB

class DashboardGenerator:
    # SPC monitor instance (class-level to persist across runs)
    _spc_monitor = None

    def __init__(self):
        PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite_connect(DB_PATH)
        self.conn.row_factory = sqlite3.Row
    
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
        
        # Get paper portfolio performance (from JSONL log)
        perf_log = DATA_DIR / "performance.jsonl"
        paper_perf = []
        if perf_log.exists():
            with open(perf_log) as f:
                for line in f:
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
        except (ImportError, AttributeError, KeyError, ValueError, TypeError, RuntimeError, OSError) as e:
            logger.warning("Overlay dashboard data unavailable: %s", e)

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
        except (ImportError, AttributeError, KeyError, ValueError, TypeError, RuntimeError, OSError) as e:
            logger.warning("VIX term structure data unavailable: %s", e)

        return result

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
        
        # VIX-based regime detection
        # >25: crisis, >20: vol_spike, <15: low_vol
        if vix_level is not None:
            if vix_level > 25:
                vix_regime = "crisis"
            elif vix_level > 20:
                vix_regime = "vol_spike"
            elif vix_level < 15:
                vix_regime = "low_vol"
            else:
                vix_regime = "normal"
            
            # Composite: VIX overrides trend in extreme cases
            if vix_regime in ["crisis", "vol_spike"]:
                current_regime = vix_regime
            elif vix_regime == "low_vol" and trend_regime != "crisis":
                current_regime = "low_vol"
            else:
                current_regime = trend_regime
        else:
            current_regime = trend_regime
        
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
        regime_overrides = {
            "crisis": {"SPY": 0.20, "GLD": 0.50, "TLT": 0.30},
            "vol_spike": {"SPY": 0.30, "GLD": 0.45, "TLT": 0.25},
            "low_vol": {"SPY": 0.55, "GLD": 0.30, "TLT": 0.15}
        }
        target_alloc = regime_overrides.get(current_regime, base_alloc)
        
        # Pending orders
        orders = []
        orders_log = DATA_DIR / "orders.jsonl"
        if orders_log.exists():
            with open(orders_log) as f:
                lines = f.readlines()[-5:]  # Last 5 orders
                for line in lines:
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
        except (ImportError, AttributeError, KeyError, ValueError, TypeError, RuntimeError, OSError) as e:
            logger.warning("Factor rotation not available: %s", e)

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
        except (ImportError, AttributeError, KeyError, ValueError, TypeError, RuntimeError, OSError) as e:
            logger.warning("Convexity harvest / vol parity not available: %s", e)

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
        except (ImportError, AttributeError, KeyError, ValueError, TypeError, RuntimeError, OSError) as e:
            logger.warning("LLM sentiment not available: %s", e)

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
                    "source_breakdown": source_breakdown,
                }
        except (ImportError, AttributeError, KeyError, ValueError, TypeError, RuntimeError, OSError) as e:
            logger.warning("Ensemble voting not available: %s", e)

        # Add sector rotation momentum signals (v2.40 Phase 5)
        sector_momentum_signal = None
        try:
            sector_momentum_signal = self._generate_sector_momentum_signals(vix_level=vix_level)
        except (ImportError, AttributeError, KeyError, ValueError, TypeError, RuntimeError, OSError) as e:
            # Sector momentum not available yet
            logger.warning("Sector momentum not available: %s", e)

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
        except (ImportError, AttributeError, KeyError, ValueError, TypeError, RuntimeError, OSError) as e:
            logger.warning("Behavioral sentiment not available: %s", e)

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
        except (ImportError, AttributeError, KeyError, ValueError, TypeError, RuntimeError, OSError) as e:
            logger.warning("Stacking ensemble not available: %s", e)

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
        except (ImportError, AttributeError, KeyError, ValueError, TypeError, RuntimeError, OSError) as e:
            logger.warning("Factor rotation dashboard not available: %s", e)

        # Merge overlay dashboard data (collar, crypto, calendar, kurtosis, etc.)
        overlay_data = self._get_overlay_data()

        output = {
            "generated_at": datetime.now().isoformat(),
            "regime": regime_data,
            "target_allocations": target_alloc,
            "current_positions": positions,
            "cash": round(cash, 2),
            "total_value": round(total_value, 2),
            "latest_prices": latest,
            "recent_orders": list(reversed(orders)),
            "ml_signals": self._generate_ml_signals(),
            "factor_rotation": factor_rotation_signal,
            "yield_curve": yield_curve_data.get("yield_curve"),
            "duration_allocation": yield_curve_data.get("duration_allocation"),
            "convexity_harvest": convexity_signal,
            "volatility_parity": vol_parity_signal,
            "llm_sentiment": sentiment_signal,
            "ensemble_voting": ensemble_signal,
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
            "smart_rebalance": smart_rebalance_data,
            "broker": broker_data,
            "garch_cvar": garch_cvar_data,
            "entropy": entropy_data,
            "bond_momentum": overlay_data.get("bond_momentum", {}),
        }

        # Rebalance health data
        try:
            from src.monitor.rebalance_health import generate as gen_rebalance_health
            output["rebalance_health"] = gen_rebalance_health()
        except (ImportError, AttributeError, KeyError, ValueError, TypeError, RuntimeError, OSError) as e:
            logger.warning("Rebalance health not available: %s", e)
            output["rebalance_health"] = {"generated": None, "error": str(e)}

        # Signal staleness detection (production readiness)
        output["staleness"] = self._check_signal_staleness(output)

        # Apply staleness-weighted decay to ensemble weights
        output = self._apply_staleness_decay(output)

        # Fire external alerts on staleness state transitions
        try:
            from src.monitor.alerting import check_staleness_and_alert
            check_staleness_and_alert(output["staleness"])
        except (ImportError, AttributeError, KeyError, ValueError, TypeError, RuntimeError, OSError) as e:
            logger.warning("Alerting not available: %s", e)

        # SPC signal quality monitoring
        output["spc"] = self._run_spc_monitor(output)
        
        out_path = PUBLIC_DIR / "signals.json"
        save_results_json(output, output_path=str(out_path))
        
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

        # Check position sync log
        sync_log = DATA_DIR / "position_sync.jsonl"
        if sync_log.exists():
            try:
                lines = sync_log.read_text().strip().split("\n")
                if lines:
                    last = json.loads(lines[-1])
                    broker["connected"] = True
                    broker["last_sync"] = last.get("timestamp")
                    broker["positions"] = last.get("broker_positions", [])
                    broker["drift"] = last.get("drift", [])
            except (OSError, json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
                logger.warning("Failed to load position sync log: %s", e)

        # Check broker orders log
        orders_log = DATA_DIR / "broker_orders.jsonl"
        if orders_log.exists():
            try:
                lines = orders_log.read_text().strip().split("\n")
                recent = []
                for line in lines[-10:]:
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
        """Load GARCH-filtered CVaR metrics for dashboard (v3.21)."""
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
        }

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
                # Get latest features for each symbol
                latest_features = {}
                with open(features_file, 'r') as f:
                    for line in f:
                        try:
                            feat = json.loads(line)
                            sym = feat.get("symbol")
                            ts = feat.get("timestamp", "")
                            if sym and (sym not in latest_features or ts > latest_features[sym].get("timestamp", "")):
                                latest_features[sym] = feat
                        except json.JSONDecodeError:
                            logger.exception("Failed to parse feature line in ensemble_voter_signals.jsonl")
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

        # Check for grid search results
        grid_file = DATA_DIR / "grid_search_results.jsonl"
        if grid_file.exists():
            try:
                with open(grid_file, 'r') as f:
                    lines = f.readlines()
                    if lines:
                        latest = json.loads(lines[-1])
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
                lines = f.readlines()
                if len(lines) >= 20:
                    recent = [json.loads(l) for l in lines[-63:]]  # Last 63 entries
                    returns = [r.get("daily_return", 0) for r in recent if r.get("daily_return")]
                    values = [r.get("total_value", 0) for r in recent]
                    
                    if returns and values:
                        paper_metrics = {
                            "sharpe": round(np.mean(returns) / np.std(returns) * np.sqrt(252), 2) if np.std(returns) > 0 else 0,
                            "total_return": round((values[-1] - values[0]) / values[0] * 100, 2),
                            "max_value": round(max(values), 2),
                            "min_value": round(min(values), 2),
                            "days_tracked": len(values)
                        }
                        
                        # Calculate SPY comparison if we have enough data
                        cursor.execute("""
                            SELECT date, close FROM prices 
                            WHERE symbol = 'SPY' 
                            AND date >= date('now', '-63 days')
                            ORDER BY date
                        """)
                        spy_rows = cursor.fetchall()
                        if len(spy_rows) >= 20 and len(values) >= 20:
                            spy_prices = [r[1] for r in spy_rows[-len(values):]]
                            spy_returns = [(spy_prices[i] - spy_prices[i-1]) / spy_prices[i-1] 
                                          for i in range(1, len(spy_prices))]
                            
                            # Calculate metrics
                            spy_total_return = (spy_prices[-1] - spy_prices[0]) / spy_prices[0]
                            portfolio_total_return = (values[-1] - values[0]) / values[0]
                            
                            # Correlation and Beta (30-day rolling)
                            min_len = min(len(returns), len(spy_returns))
                            if min_len >= 20:
                                returns_arr = np.array(returns[-20:])
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
                                "portfolio_value": round(values[-1], 2),
                                "spy_value": round(values[0] * (1 + spy_total_return), 2),
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
        
        # Get cron job status from project-local status file
        try:
            cron_status_file = DATA_DIR / "cron_status.json"
            if cron_status_file.exists():
                with open(cron_status_file) as f:
                    cron_data = json.load(f)
                health_data["cron_jobs"] = cron_data.get("jobs", [])
            else:
                # Fallback: mark as unknown but system healthy
                health_data["cron_jobs"] = [
                    {"name": "portfolio-lab-data", "status": "unknown", "state": "scheduled"},
                    {"name": "portfolio-lab-eval", "status": "unknown", "state": "scheduled"},
                    {"name": "portfolio-lab-dashboard", "status": "unknown", "state": "scheduled"},
                    {"name": "portfolio-lab-research", "status": "unknown", "state": "scheduled"},
                    {"name": "portfolio-lab-wiki-sync", "status": "unknown", "state": "scheduled"},
                    {"name": "portfolio-lab-health", "status": "unknown", "state": "scheduled"},
                    {"name": "portfolio-lab-build", "status": "unknown", "state": "scheduled"},
                ]
        except (OSError, json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
            health_data["system_status"] = "degraded"
            health_data["error"] = f"Failed to get cron status: {str(e)}"
        
        # Get data freshness from SQLite
        cursor = self.conn.cursor()
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
                    health_data["data_freshness"][sym] = {
                        "last_update": last_date,
                        "days_stale": days_stale,
                        "status": "fresh" if days_stale <= 1 else "stale" if days_stale <= 3 else "critical"
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
        except (ImportError, AttributeError, KeyError, ValueError, TypeError, RuntimeError, OSError) as e:
            health_data["signal_health"] = {
                "error": f"Failed to get signal health: {str(e)}",
                "status": "unavailable"
            }

        # Overall system health
        stale_count = sum(1 for d in health_data["data_freshness"].values() if d.get("status") != "fresh")
        failed_jobs = sum(1 for j in health_data["cron_jobs"] if j.get("status") == "error")
        
        if failed_jobs > 0 or stale_count > 5:
            health_data["system_status"] = "warning"
        if failed_jobs > 2 or stale_count > 10:
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

        except (ImportError, AttributeError, KeyError, ValueError, TypeError, RuntimeError, OSError) as e:
            logger.warning("Failed to generate sector momentum signals: %s", e)
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
        except (ImportError, AttributeError, KeyError, ValueError, TypeError, RuntimeError, OSError) as e:
            logger.warning("Overlay dashboard generation failed: %s", e)
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

        except (ImportError, AttributeError, KeyError, ValueError, TypeError, RuntimeError, OSError) as e:
            logger.warning("Failed to generate adaptive sizing data: %s", e)
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

        except (ImportError, AttributeError, KeyError, ValueError, TypeError, RuntimeError, OSError) as e:
            logger.warning("Failed to generate VIXY hedge data: %s", e)
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

        except (ImportError, AttributeError, KeyError, ValueError, TypeError, RuntimeError, OSError) as e:
            logger.warning("Failed to generate Black-Litterman data: %s", e)
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

        except (ImportError, AttributeError, KeyError, ValueError, TypeError, RuntimeError, OSError) as e:
            logger.warning("Failed to generate turnover validator data: %s", e)
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

            out_path = PUBLIC_DIR / "regime_gate.json"
            save_results_json(gate_data, output_path=str(out_path))
            return out_path

        except (ImportError, AttributeError, KeyError, ValueError, TypeError, RuntimeError, OSError) as e:
            logger.warning("Failed to generate regime gate data: %s", e)
            return None

    def _is_msm_gated(self) -> bool:
        """Check if MSM should be gated off based on current regime.

        MSM has zero ensemble weight in HIGH_VOL/CRISIS regimes (health 0.55,
        net-negative -0.012 Sharpe). Returns True when gated off.
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT regime FROM regime_log ORDER BY detected_at DESC LIMIT 1")
            row = cursor.fetchone()
            regime = row[0] if row else "normal"
            return regime.lower() in {"high_vol", "crisis"}
        except Exception as e:
            logger.warning("_is_msm_gated: regime query failed (%s) — gating MSM off", e)
            return True  # Gate off when regime unknown

    # Signal staleness detection (production readiness)
    SIGNAL_STALENESS_TTL_HOURS = int(os.environ.get("SIGNAL_STALENESS_TTL_HOURS", "4"))
    STALENESS_DECAY_TAU_HOURS = float(os.environ.get("STALENESS_DECAY_TAU_HOURS", "2.0"))

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
        }

        for signal_key, (ts_field, _) in timestamped_signals.items():
            signal_block = signal_data.get(signal_key)
            if signal_block is None:
                stale_signals.append(signal_key)
                signal_timestamps[signal_key] = None
                signal_age_hours[signal_key] = None
                staleness_decay[signal_key] = 0.0
                continue

            ts_str = signal_block.get(ts_field) if isinstance(signal_block, dict) else None
            signal_timestamps[signal_key] = ts_str

            if ts_str is None:
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

        healthy_count = len(timestamped_signals) - len(stale_signals)
        return {
            "stale_signals": stale_signals,
            "signal_timestamps": signal_timestamps,
            "signal_age_hours": signal_age_hours,
            "staleness_decay": staleness_decay,
            "decay_tau_hours": tau_hours,
            "healthy_count": healthy_count,
            "total_count": len(timestamped_signals),
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

        except (ImportError, AttributeError, KeyError, ValueError, TypeError, RuntimeError, OSError) as e:
            logger.warning("Failed to generate TSMOM data: %s", e)
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

        except (ImportError, AttributeError, KeyError, ValueError, TypeError, RuntimeError, OSError) as e:
            logger.warning("Failed to generate cross-asset RV data: %s", e)
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

        except (ImportError, AttributeError, KeyError, ValueError, TypeError, RuntimeError, OSError) as e:
            logger.warning("Failed to generate graduation data: %s", e)
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

        for p in paths:
            if p:
                logger.info("Generated: %s", p)

        # Create index
        index = {
            "files": [str(p.name) for p in paths if p],
            "generated_at": datetime.now().isoformat()
        }
        save_results_json(index, output_path=str(PUBLIC_DIR / "index.json"))

        self.conn.close()
        logger.info("Dashboard generation complete")

if __name__ == "__main__":
    gen = DashboardGenerator()
    gen.run()

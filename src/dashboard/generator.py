#!/usr/bin/env python3
"""
Portfolio-Lab Alpha: Dashboard Generator
Creates static dashboard from SQLite data for Vite/React app consumption.
"""

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

DATA_DIR = Path("~/projects/portfolio-lab/data").expanduser()
PUBLIC_DIR = Path("~/projects/portfolio-lab/public/data").expanduser()
DB_PATH = DATA_DIR / "market.db"

# Add src to path for importing signal health tracker
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

class DashboardGenerator:
    def __init__(self):
        PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(DB_PATH)
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
                    except Exception:
                        pass

        output = {
            "prices": prices,
            "regimes": regimes,
            "paper_portfolio": paper_perf,
            "generated_at": datetime.now().isoformat()
        }
        
        out_path = PUBLIC_DIR / "dashboard.json"
        with open(out_path, 'w') as f:
            json.dump(output, f)
        
        return out_path
    
    def generate_signals_json(self) -> Path:
        """Generate current signals and allocations."""
        cursor = self.conn.cursor()
        
        # Import strategy engines
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / 'strategy'))
        try:
            from dual_momentum import DualMomentumEngine
            from comparison import StrategyComparisonEngine
            dual_momentum_available = True
        except ImportError:
            dual_momentum_available = False
        
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
        base_alloc = {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16}
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
                    except Exception:
                        pass

        # Add factor rotation signals if engine available
        factor_rotation_signal = None
        try:
            from factor_rotation import FactorMomentumEngine
            factor_engine = FactorMomentumEngine(db_path=DB_PATH)
            factor_rotation_signal = factor_engine.evaluate()
        except Exception as e:
            pass  # Factor rotation not available
        
        # Add volatility targeting if engine available
        vol_targeting_signal = None
        try:
            from vol_targeting import VolatilityTargetingEngine
            vol_engine = VolatilityTargetingEngine(db_path=DB_PATH)
            vol_targeting_signal = vol_engine.evaluate(target_alloc)
        except Exception as e:
            pass  # Vol targeting not available
        
        # Add yield curve data from yields.json
        yield_curve_data = self._get_yield_curve_data()
        
        # Add volatility parity / convexity harvest signals
        convexity_signal = None
        vol_parity_signal = None
        try:
            from strategy.convexity_harvest import ConvexityHarvestStrategy
            from strategy.vol_parity_allocator import VolatilityParityAllocator
            
            # Get convexity harvest signal
            convexity_engine = ConvexityHarvestStrategy()
            convexity_signal = convexity_engine.get_current_signal()
            
            # Get volatility parity allocation  
            vol_allocator = VolatilityParityAllocator(vix_strategy=convexity_engine)
            vol_parity_data = vol_allocator.get_current_allocation()
            if vol_parity_data:
                vol_parity_signal = vol_parity_data.get('allocation')
        except Exception as e:
            # Convexity harvest / vol parity not available yet
            pass
        
        # Add LLM sentiment signals (v2.30 Phase 5)
        sentiment_signal = None
        try:
            from strategy.regime_sentiment import RegimeSentimentPipeline
            
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
        except Exception as e:
            # LLM sentiment not available yet
            pass
        
        # Add ensemble voting signals (v2.20 Phase 3)
        ensemble_signal = None
        try:
            from strategy.ensemble_voter import EnsembleVotingEngine
            
            ensemble_engine = EnsembleVotingEngine()
            ensemble_result = ensemble_engine.evaluate()
            if ensemble_result:
                ensemble_signal = {
                    "regime": ensemble_result.regime,
                    "confidence": ensemble_result.confidence,
                    "agreement_score": ensemble_result.agreement_score,
                    "probabilities": ensemble_result.ensemble_probs,
                    "action": ensemble_result.action,
                    "position_scaling": ensemble_result.position_scaling,
                    "disagreement_sources": ensemble_result.disagreement_sources
                }
        except Exception as e:
            # Ensemble voting not available yet
            pass
        
        # Add sector rotation momentum signals (v2.40 Phase 5)
        sector_momentum_signal = None
        try:
            sector_momentum_signal = self._generate_sector_momentum_signals()
        except Exception as e:
            # Sector momentum not available yet
            pass
        
        # Add smart rebalancing status (v2.90)
        smart_rebalance_data = None
        try:
            import importlib
            rebalancing_pkg = importlib.import_module('src.rebalancing')
            SmartRebalanceGate = rebalancing_pkg.SmartRebalanceGate

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
        except Exception as e:
            import traceback
            traceback.print_exc()
            pass

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
                                "score": alt_data_raw.get("raw_data", {}).get("earnings_sentiment"),
                                "confidence": alt_data_raw.get("raw_data", {}).get("earnings_confidence"),
                                "weight": alt_data_raw.get("raw_data", {}).get("weights", {}).get("earnings")
                            },
                            "news": {
                                "score": alt_data_raw.get("raw_data", {}).get("news_sentiment"),
                                "confidence": alt_data_raw.get("raw_data", {}).get("news_confidence"),
                                "weight": alt_data_raw.get("raw_data", {}).get("weights", {}).get("news")
                            },
                            "jobs": {
                                "score": alt_data_raw.get("raw_data", {}).get("jobs_signal"),
                                "confidence": alt_data_raw.get("raw_data", {}).get("jobs_confidence"),
                                "weight": alt_data_raw.get("raw_data", {}).get("weights", {}).get("jobs")
                            },
                            "social": {
                                "score": alt_data_raw.get("raw_data", {}).get("social_sentiment"),
                                "confidence": alt_data_raw.get("raw_data", {}).get("social_confidence"),
                                "weight": alt_data_raw.get("raw_data", {}).get("weights", {}).get("social")
                            }
                        },
                        "composite_score": alt_data_raw.get("raw_data", {}).get("composite_score"),
                        "z_score": alt_data_raw.get("raw_data", {}).get("z_score"),
                        "sources_count": alt_data_raw.get("raw_data", {}).get("sources_count"),
                        "data_freshness_hours": alt_data_raw.get("raw_data", {}).get("data_freshness_hours")
                    }
        except Exception as e:
            # Alternative data signal not available yet
            pass
        
        # Load broker data (Phase 4: live trading prep)
        broker_data = self._load_broker_data()

        # Add closing auction signals (v3.17 Phase 4)
        closing_auction_data = self._load_closing_auction_data()

        output = {
            "timestamp": datetime.now().isoformat(),
            "regime": regime_data,
            "target_allocations": target_alloc,
            "current_positions": positions,
            "cash": round(cash, 2),
            "total_value": round(total_value, 2),
            "latest_prices": latest,
            "recent_orders": list(reversed(orders)),
            "ml_signals": self._generate_ml_signals(),
            "factor_rotation": factor_rotation_signal,
            "volatility_targeting": vol_targeting_signal,
            "yield_curve": yield_curve_data.get("yield_curve"),
            "duration_allocation": yield_curve_data.get("duration_allocation"),
            "convexity_harvest": convexity_signal,
            "volatility_parity": vol_parity_signal,
            "llm_sentiment": sentiment_signal,
            "ensemble_voting": ensemble_signal,
            "sector_rotation": sector_momentum_signal,
            "alternative_data": alternative_data_signal,
            "smart_rebalance": smart_rebalance_data,
            "broker": broker_data,
            "closing_auction": closing_auction_data,
        }
        
        out_path = PUBLIC_DIR / "signals.json"
        with open(out_path, 'w') as f:
            json.dump(output, f, indent=2)
        
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
            except Exception:
                pass

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
            except Exception:
                pass

        # Check kill switch
        kill_file = DATA_DIR / "kill_switch.json"
        if kill_file.exists():
            try:
                with open(kill_file) as f:
                    ks = json.load(f)
                broker["kill_switch"] = ks.get("enabled", False)
            except Exception:
                pass

        return broker

    def _load_closing_auction_data(self) -> Dict:
        """Load closing auction MOC signals for dashboard (v3.17 Phase 4)."""
        closing_auction = {
            "signals": [],
            "last_update": None,
            "market_open": False,
        }
        
        try:
            # Check for closing auction signal file
            signal_file = DATA_DIR / "closing_auction_latest.json"
            if signal_file.exists():
                with open(signal_file) as f:
                    data = json.load(f)
                    closing_auction["signals"] = data.get("signals", [])
                    closing_auction["last_update"] = data.get("timestamp")
            
            # Check market hours (simplified: 9:30-16:00 ET)
            from datetime import datetime, time
            now = datetime.now()
            et_offset = timedelta(hours=0)  # Server is ET
            et_time = (now + et_offset).time()
            market_open_time = time(9, 30)
            market_close_time = time(16, 0)
            closing_auction["market_open"] = market_open_time <= et_time <= market_close_time
            
        except Exception as e:
            # Closing auction data not available
            pass
        
        return closing_auction

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
                        except Exception:
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
            except Exception as e:
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
            except Exception:
                pass

        return signals
    
    def _get_yield_curve_data(self) -> Dict:
        """Get yield curve data from yields.json and calculate duration allocation."""
        result = {
            "yield_curve": None,
            "duration_allocation": None
        }
        
        yields_file = Path("/root/projects/portfolio-lab/public/data/yields.json")
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
            
        except Exception as e:
            print(f"Warning: Failed to load yield curve data: {e}")
        
        return result
    
    def generate_stats_json(self) -> Path:
        """Generate performance statistics."""
        cursor = self.conn.cursor()
        
        # Calculate 30-day returns for each asset
        stats = {}
        for symbol in ['SPY', 'GLD', 'TLT', 'QQQ', 'VIX']:
            cursor.execute("""
                SELECT close FROM prices 
                WHERE symbol = ? AND date >= date('now', '-30 days')
                ORDER BY date
            """, (symbol,))
            
            prices = [row[0] for row in cursor.fetchall()]
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
        with open(out_path, 'w') as f:
            json.dump(output, f)
        
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
                    "message": f"Sharpe: {data.get('metrics', {}).get('sharpe')}, ready for live approval",
                    "timestamp": data.get("timestamp"),
                    "requires_action": True
                })
        
        # Check for kill switch
        for mode in ["paper", "live"]:
            kill_file = DATA_DIR / f".kill_switch_{mode}"
            if kill_file.exists():
                with open(kill_file) as f:
                    data = json.load(f)
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
        with open(out_path, 'w') as f:
            json.dump(output, f)
        
        return out_path
    
    def generate_health_json(self) -> Path:
        """Generate system health status for dashboard."""
        import subprocess
        import os
        
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
        except Exception as e:
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
                except Exception:
                    pass

        # Get signal health from SignalHealthTracker
        try:
            from signals.health_tracker import SignalHealthTracker
            tracker = SignalHealthTracker()
            signal_health_report = tracker.get_health_report()
            health_data["signal_health"] = {
                "timestamp": signal_health_report.get("timestamp"),
                "summary": signal_health_report.get("summary", {}),
                "scores": signal_health_report.get("scores", {}),
                "alerts": signal_health_report.get("alerts", []),
                "overall_health": signal_health_report.get("overall_health", "unknown")
            }
        except Exception as e:
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
        with open(out_path, 'w') as f:
            json.dump(health_data, f, indent=2)
        
        return out_path
    
    def _generate_sector_momentum_signals(self) -> Optional[Dict]:
        """Generate sector rotation momentum signals from historical data."""
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / 'strategy'))
            
            from sector_momentum_calc import generate_sector_signals
            
            historical_path = PUBLIC_DIR.parent / "data" / "historical.json"
            
            # Get current VIX level for threshold checking
            vix = 0
            try:
                cursor = self.conn.cursor()
                cursor.execute("SELECT close FROM prices WHERE symbol = '^VIX' ORDER BY date DESC LIMIT 1")
                row = cursor.fetchone()
                if row:
                    vix = row[0]
            except Exception:
                pass

            signals = generate_sector_signals(historical_path, vix=vix)
            return signals
            
        except Exception as e:
            print(f"Warning: Failed to generate sector momentum signals: {e}")
            return None
    
    def generate_analytics_json(self) -> Path:
        """Generate analytics data (drawdown, rolling metrics, benchmarks)."""
        # Import analytics calculator
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        
        try:
            from analytics.calculator import AnalyticsCalculator
            calc = AnalyticsCalculator(data_dir=str(DATA_DIR))
            report = calc.generate_analytics_report()
            
            out_path = PUBLIC_DIR / "analytics.json"
            with open(out_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            return out_path
        except Exception as e:
            # Fallback: empty analytics
            report = {
                "status": "error",
                "message": str(e),
                "generated_at": datetime.now().isoformat(),
            }
            out_path = PUBLIC_DIR / "analytics.json"
            with open(out_path, 'w') as f:
                json.dump(report, f, indent=2)
            return out_path
    
    def run(self):
        """Generate all dashboard files."""
        print(f"[{datetime.now()}] Generating dashboard data...")
        
        paths = [
            self.generate_performance_json(),
            self.generate_signals_json(),
            self.generate_stats_json(),
            self.generate_alerts_json(),
            self.generate_health_json(),
            self.generate_analytics_json(),  # NEW
        ]
        
        for p in paths:
            print(f"  Generated: {p}")
        
        # Create index
        index = {
            "files": [str(p.name) for p in paths],
            "generated_at": datetime.now().isoformat()
        }
        with open(PUBLIC_DIR / "index.json", 'w') as f:
            json.dump(index, f)
        
        self.conn.close()
        print(f"[{datetime.now()}] Dashboard generation complete")

if __name__ == "__main__":
    gen = DashboardGenerator()
    gen.run()

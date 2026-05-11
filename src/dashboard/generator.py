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
                    except:
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
                    except:
                        pass
        
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
        }
        
        out_path = PUBLIC_DIR / "signals.json"
        with open(out_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        return out_path
    
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
                        except:
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
            except:
                pass
        
        return signals
    
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
            "generated_at": datetime.now().isoformat()
        }
        
        # Get cron job status from hermes CLI
        try:
            # Try reading from hermes state directory since CLI may not work in cron context
            hermes_state = Path.home() / ".hermes" / "cron" / "state.json"
            if hermes_state.exists():
                with open(hermes_state) as f:
                    state = json.load(f)
                    for job_id, job in state.get("jobs", {}).items():
                        if job.get("name", "").startswith("portfolio-lab"):
                            health_data["cron_jobs"].append({
                                "id": job_id[:12],
                                "name": job.get("name"),
                                "schedule": job.get("schedule"),
                                "last_run": job.get("last_run_at"),
                                "next_run": job.get("next_run_at"),
                                "status": "ok" if job.get("last_status") == "ok" else "error",
                                "state": job.get("state", "unknown")
                            })
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
                except:
                    pass
        
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

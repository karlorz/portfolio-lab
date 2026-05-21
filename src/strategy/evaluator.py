#!/usr/bin/env python3
"""
Portfolio-Lab Alpha: Strategy Evaluator
Runs continuously to evaluate signals, generate orders, route to paper or live.
"""

import os
import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, NamedTuple
import numpy as np

from src.paths import DATA_DIR, MARKET_DB

# Config
DB_PATH = MARKET_DB
ORDERS_LOG = DATA_DIR / "orders.jsonl"
PERFORMANCE_LOG = DATA_DIR / "performance.jsonl"

# Paper trading config (default)
PAPER_CONFIG = {
    "initial_capital": 100000,
    "max_position_pct": 0.4,  # Max 40% in any single asset
    "max_drawdown_pct": 0.15,  # Kill switch at 15% DD
    "rebalance_threshold": 0.10,  # 10% drift triggers rebalance
    "volatility_target": 0.12,  # 12% annual vol target
}

# Core strategy: SPY/GLD/TLT 46/38/16 with regime overrides
BASE_ALLOCATION = {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16}

REGIME_OVERRIDES = {
    "crisis": {"SPY": 0.20, "GLD": 0.50, "TLT": 0.30},  # Risk-off
    "vol_spike": {"SPY": 0.30, "GLD": 0.45, "TLT": 0.25},  # Defensive
    "low_vol": {"SPY": 0.55, "GLD": 0.30, "TLT": 0.15},  # Risk-on
}

class Position(NamedTuple):
    symbol: str
    shares: float
    avg_price: float
    current_price: float
    value: float
    weight: float
    unrealized_pnl: float

class Portfolio:
    def __init__(self, state_file: Path, mode: str = "paper"):
        self.state_file = state_file
        self.mode = mode  # "paper" or "live"
        self.cash = 0
        self.positions: Dict[str, Position] = {}
        self.history: List[Dict] = []
        self._load_state()
    
    def _load_state(self):
        if self.state_file.exists():
            with open(self.state_file) as f:
                state = json.load(f)
                self.cash = state.get("cash", PAPER_CONFIG["initial_capital"])
                self.positions = {k: Position(**v) for k, v in state.get("positions", {}).items()}
                self.history = state.get("history", [])
        else:
            self.cash = PAPER_CONFIG["initial_capital"]
    
    def save_state(self):
        state = {
            "cash": self.cash,
            "positions": {k: v._asdict() for k, v in self.positions.items()},
            "history": self.history[-100:],  # Keep last 100 snapshots
            "updated": datetime.now().isoformat(),
            "mode": self.mode
        }
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2, default=str)
    
    def total_value(self, prices: Dict[str, float]) -> float:
        position_value = sum(
            p.shares * prices.get(p.symbol, p.current_price) 
            for p in self.positions.values()
        )
        return self.cash + position_value
    
    def current_weights(self, prices: Dict[str, float]) -> Dict[str, float]:
        total = self.total_value(prices)
        if total == 0:
            return {}
        return {
            p.symbol: (p.shares * prices.get(p.symbol, p.current_price)) / total
            for p in self.positions.values()
        }
    
    def calculate_orders(self, target_weights: Dict[str, float], prices: Dict[str, float]) -> List[Dict]:
        """Generate orders to move from current to target allocation."""
        total = self.total_value(prices)
        current_weights = self.current_weights(prices)
        orders = []
        
        for symbol, target_w in target_weights.items():
            if symbol not in prices or prices[symbol] <= 0:
                continue
            
            current_w = current_weights.get(symbol, 0)
            drift = abs(target_w - current_w)
            
            # Only rebalance if drift exceeds threshold
            if drift > PAPER_CONFIG["rebalance_threshold"]:
                target_value = total * target_w
                current_value = self.positions[symbol].value if symbol in self.positions else 0
                delta_value = target_value - current_value
                
                if abs(delta_value) > 100:  # Min order size $100
                    shares = delta_value / prices[symbol]
                    orders.append({
                        "symbol": symbol,
                        "side": "buy" if shares > 0 else "sell",
                        "shares": abs(shares),
                        "estimated_price": prices[symbol],
                        "estimated_value": abs(delta_value),
                        "reason": f"rebalance_{'up' if shares > 0 else 'down'}",
                        "drift_before": drift
                    })
        
        return orders
    
    def execute_orders(self, orders: List[Dict], prices: Dict[str, float], slippage: float = 0.001):
        """Execute orders with slippage simulation."""
        executed = []
        
        for order in orders:
            symbol = order["symbol"]
            base_price = prices.get(symbol, order["estimated_price"])
            
            # Simulate slippage (0.1% for paper trading)
            fill_price = base_price * (1 + slippage if order["side"] == "buy" else 1 - slippage)
            fill_shares = order["shares"]
            fill_value = fill_shares * fill_price
            
            if order["side"] == "buy":
                if fill_value > self.cash:
                    # Partial fill
                    fill_shares = self.cash / fill_price
                    fill_value = fill_shares * fill_price
                
                self.cash -= fill_value
                
                if symbol in self.positions:
                    p = self.positions[symbol]
                    new_shares = p.shares + fill_shares
                    new_avg = (p.shares * p.avg_price + fill_value) / new_shares
                    self.positions[symbol] = Position(
                        symbol, new_shares, new_avg, fill_price,
                        new_shares * fill_price, 0, (fill_price - new_avg) * new_shares
                    )
                else:
                    self.positions[symbol] = Position(
                        symbol, fill_shares, fill_price, fill_price,
                        fill_value, 0, 0
                    )
            else:
                if symbol in self.positions and self.positions[symbol].shares >= fill_shares:
                    p = self.positions[symbol]
                    realized = (fill_price - p.avg_price) * fill_shares
                    new_shares = p.shares - fill_shares
                    
                    if new_shares > 0:
                        self.positions[symbol] = Position(
                            symbol, new_shares, p.avg_price, fill_price,
                            new_shares * fill_price, 0, (fill_price - p.avg_price) * new_shares
                        )
                    else:
                        del self.positions[symbol]
                    
                    self.cash += fill_value
            
            executed.append({
                **order,
                "fill_price": fill_price,
                "fill_shares": fill_shares,
                "fill_value": fill_value,
                "timestamp": datetime.now().isoformat()
            })
        
        return executed
    
    def check_risk_limits(self, prices: Dict[str, float]) -> Optional[str]:
        """Check if risk limits breached. Returns kill reason or None."""
        total = self.total_value(prices)
        
        # Drawdown check (need equity curve)
        if len(self.history) > 20:
            peak = max(h["total_value"] for h in self.history[-252:])  # 1 year lookback
            if total < peak * (1 - PAPER_CONFIG["max_drawdown_pct"]):
                return f"max_drawdown_{(peak - total) / peak:.2%}"
        
        # Position concentration check
        for p in self.positions.values():
            if p.weight > PAPER_CONFIG["max_position_pct"]:
                return f"max_position_{p.symbol}_{p.weight:.2%}"
        
        return None

def get_current_regime(conn: sqlite3.Connection) -> str:
    """Get latest detected regime using VIX thresholds and trend analysis."""
    cursor = conn.cursor()
    
    # Try to get VIX level
    cursor.execute("""
        SELECT close FROM prices 
        WHERE symbol = '^VIX' 
        ORDER BY date DESC LIMIT 1
    """)
    vix_row = cursor.fetchone()
    vix_level = vix_row[0] if vix_row else None
    
    # Try to get trend signal from regime_log (existing trend-based detection)
    cursor.execute("SELECT regime FROM regime_log ORDER BY detected_at DESC LIMIT 1")
    trend_row = cursor.fetchone()
    trend_regime = trend_row[0] if trend_row else "normal"
    
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
        # If VIX says crisis or vol_spike, trust it (market fear is immediate)
        # If VIX says low_vol, confirm with trend (avoid false calm)
        if vix_regime in ["crisis", "vol_spike"]:
            return vix_regime
        elif vix_regime == "low_vol" and trend_regime != "crisis":
            return "low_vol"
    
    # Fallback to trend-based or default
    return trend_regime if trend_row else "normal"

def get_latest_vix(conn: sqlite3.Connection) -> Optional[float]:
    """Get latest VIX level for display."""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT close FROM prices 
        WHERE symbol = '^VIX' 
        ORDER BY date DESC LIMIT 1
    """)
    row = cursor.fetchone()
    return row[0] if row else None

def get_latest_prices(conn: sqlite3.Connection) -> Dict[str, float]:
    """Get latest prices for all symbols."""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT symbol, close FROM prices 
        WHERE (symbol, date) IN (
            SELECT symbol, MAX(date) FROM prices GROUP BY symbol
        )
    """)
    return {row[0]: row[1] for row in cursor.fetchall()}

def calculate_performance(portfolio: Portfolio, prices: Dict[str, float]) -> Dict:
    """Calculate current performance metrics."""
    total = portfolio.total_value(prices)
    
    # Calculate daily return
    if portfolio.history:
        last_total = portfolio.history[-1]["total_value"]
        daily_return = (total - last_total) / last_total if last_total > 0 else 0
    else:
        daily_return = 0
    
    return {
        "timestamp": datetime.now().isoformat(),
        "total_value": total,
        "cash": portfolio.cash,
        "daily_return": daily_return,
        "positions_count": len(portfolio.positions),
        "mode": portfolio.mode
    }

def main():
    """Main evaluation loop."""
    print(f"[{datetime.now()}] Strategy Evaluator Starting")
    
    # Determine mode from environment
    mode = os.environ.get("ALPHALAB_MODE", "paper")
    state_file = DATA_DIR / f"portfolio_{mode}.json"
    
    conn = sqlite3.connect(DB_PATH)
    portfolio = Portfolio(state_file, mode)
    
    # Get current state
    prices = get_latest_prices(conn)
    regime = get_current_regime(conn)
    vix = get_latest_vix(conn)
    
    print(f"Mode: {mode}, Regime: {regime}, VIX: {vix:.2f}" if vix else f"Mode: {mode}, Regime: {regime}")
    print(f"Portfolio value: ${portfolio.total_value(prices):,.2f}")
    
    # Check kill switches
    kill_reason = portfolio.check_risk_limits(prices)
    if kill_reason:
        print(f"KILL SWITCH TRIGGERED: {kill_reason}")
        # In paper mode, just log and hold
        # In live mode, this would liquidate
        with open(DATA_DIR / f".kill_switch_{mode}", 'w') as f:
            json.dump({"reason": kill_reason, "timestamp": datetime.now().isoformat()}, f)
        return
    
    # Determine target allocation
    target_alloc = REGIME_OVERRIDES.get(regime, BASE_ALLOCATION)
    print(f"Target allocation: {target_alloc}")
    
    # Generate orders
    orders = portfolio.calculate_orders(target_alloc, prices)
    
    if orders:
        print(f"Generated {len(orders)} orders:")
        for o in orders:
            print(f"  {o['side'].upper()} {o['shares']:.2f} {o['symbol']} @ ${o['estimated_price']:.2f}")
        
        # Execute (paper trading with slippage)
        executed = portfolio.execute_orders(orders, prices)
        
        # Log orders
        with open(ORDERS_LOG, 'a') as f:
            for e in executed:
                f.write(json.dumps(e) + '\n')
        
        print(f"Executed {len(executed)} orders")
    else:
        print("No rebalancing needed")
    
    # Update and save state
    perf = calculate_performance(portfolio, prices)
    portfolio.history.append(perf)
    portfolio.save_state()
    
    # Log performance
    with open(PERFORMANCE_LOG, 'a') as f:
        f.write(json.dumps(perf) + '\n')
    
    # Check graduation criteria (paper mode only)
    if mode == "paper":
        check_graduation_criteria(portfolio)
    
    conn.close()
    print(f"[{datetime.now()}] Evaluation complete")

def _deduplicate_to_daily(history: List[Dict]) -> List[Dict]:
    """Filter history to keep only the last entry per trading day.

    History entries are recorded every ~30 minutes during market hours.
    Using raw entries for graduation metrics produces garbage results because
    most intra-day snapshots have daily_return=0.0 (price unchanged within day).
    This function groups by date and keeps the last snapshot per day.
    """
    daily: Dict[str, Dict] = {}
    for entry in history:
        ts = entry.get("timestamp", "")
        # Extract date from ISO timestamp: "2026-05-11T03:20:31" -> "2026-05-11"
        date_key = ts[:10] if len(ts) >= 10 else ts
        # Keep last entry per date (later timestamps overwrite earlier)
        daily[date_key] = entry
    # Return in chronological order
    sorted_dates = sorted(daily.keys())
    return [daily[d] for d in sorted_dates]


def check_graduation_criteria(portfolio: Portfolio):
    """Check if paper trading performance warrants live promotion.

    Uses trading-day-level data (deduplicates intra-day snapshots) and
    includes sanity validation to prevent false positives from near-zero
    standard deviation in intra-day return data.
    """
    MIN_DAYS = 63  # ~3 months
    MIN_SHARPE = 0.5
    MAX_DD = 0.15
    MIN_WIN_RATE = 0.45
    MAX_REALISTIC_SHARPE = 3.0  # Any Sharpe > 3.0 is unrealistic
    
    if len(portfolio.history) < MIN_DAYS:
        return
    
    # Deduplicate intra-day snapshots to trading-day-level data
    daily_history = _deduplicate_to_daily(portfolio.history)
    
    # Need at least MIN_DAYS trading days after dedup
    if len(daily_history) < MIN_DAYS:
        print(f"GRADUATION DEFERRED: Only {len(daily_history)} unique trading days "
              f"(need {MIN_DAYS}), skipping intra-day snapshots")
        return
    
    recent = daily_history[-MIN_DAYS:]
    returns = [h["daily_return"] for h in recent]
    
    # Calculate metrics
    total_return = (recent[-1]["total_value"] - recent[0]["total_value"]) / recent[0]["total_value"]
    
    # Volatility floor: prevent division-by-near-zero when intra-day return
    # data has been recorded but shows zero variation within each day
    daily_std = max(np.std(returns), 0.0001)
    sharpe = np.mean(returns) / daily_std * np.sqrt(252) if daily_std > 0 else 0
    
    peak = recent[0]["total_value"]
    max_dd = 0
    for h in recent:
        if h["total_value"] > peak:
            peak = h["total_value"]
        dd = (peak - h["total_value"]) / peak
        if dd > max_dd:
            max_dd = dd
    
    win_rate = sum(1 for r in returns if r > 0) / len(returns) if returns else 0
    
    # Sanity validation: reject unrealistic metrics before writing trigger
    if sharpe > MAX_REALISTIC_SHARPE:
        print(f"WARNING: Sharpe {sharpe:.2f} exceeds realistic maximum "
              f"{MAX_REALISTIC_SHARPE}. This is likely caused by intra-day "
              f"snapshot contamination. Skipping promotion.")
        return
    
    # Check criteria
    if sharpe > MIN_SHARPE and max_dd < MAX_DD and win_rate > MIN_WIN_RATE:
        print(f"GRADUATION CANDIDATE: Sharpe={sharpe:.2f}, DD={max_dd:.2%}, WinRate={win_rate:.2%}")
        
        # Create promotion trigger
        trigger = {
            "action": "promote_to_live",
            "metrics": {
                "sharpe": round(sharpe, 2),
                "max_drawdown": round(max_dd, 4),
                "win_rate": round(win_rate, 4),
                "total_return": round(total_return, 6)
            },
            "timestamp": datetime.now().isoformat(),
            "requires_approval": True
        }
        
        trigger_path = DATA_DIR / ".promote_to_live"
        with open(trigger_path, 'w') as f:
            json.dump(trigger, f, indent=2)
        
        print(f"Created promotion trigger: {trigger_path}")

if __name__ == "__main__":
    main()

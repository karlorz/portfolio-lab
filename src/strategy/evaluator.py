#!/usr/bin/env python3
"""
Portfolio-Lab Alpha: Strategy Evaluator
Runs continuously to evaluate signals, generate orders, route to paper or live.
"""

import os
import json
import logging
import sqlite3
from src.paths import sqlite_connect
from src.utils import classify_vix_regime
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, NamedTuple
import numpy as np

from src.paths import BASE_ALLOCATION, DATA_DIR, MARKET_DB, REGIME_OVERRIDES
from src.backtest.metrics import save_results_json
from enum import Enum


__all__ = ['ORDERS_LOG', 'PERFORMANCE_LOG', 'PAPER_CONFIG', 'REGIME_OVERRIDES', 'Position', 'Portfolio', 'get_current_regime', 'get_latest_vix', 'get_latest_prices', 'calculate_performance', 'check_graduation_criteria', 'KillSwitchLevel']


class KillSwitchLevel(Enum):
    """Graduated kill switch severity levels.

    Level 1: WARNING — reduce position sizes by 25%
    Level 2: RESTRICT — reduce position sizes by 50%
    Level 3: HALT — block all new orders, allow only reduce-only
    Level 4: LIQUIDATE — emergency liquidation of all positions
    """
    NONE = "none"
    WARNING = "warning"        # Level 1
    RESTRICT = "restrict"      # Level 2
    HALT = "halt"              # Level 3
    LIQUIDATE = "liquidate"    # Level 4


# Drawdown thresholds for graduated kill switch (env-var configurable)
KILL_SWITCH_THRESHOLDS = {
    "warning_drawdown_pct": float(os.environ.get("KILL_WARNING_DRAWDOWN_PCT", "0.10")),
    "restrict_drawdown_pct": float(os.environ.get("KILL_RESTRICT_DRAWDOWN_PCT", "0.15")),
    "halt_drawdown_pct": float(os.environ.get("KILL_HALT_DRAWDOWN_PCT", "0.20")),
    "liquidate_drawdown_pct": float(os.environ.get("KILL_LIQUIDATE_DRAWDOWN_PCT", "0.25")),
    "extreme_tail_cvar_ratio": float(os.environ.get("KILL_EXTREME_CVAR_RATIO", "3.0")),
}

logger = logging.getLogger(__name__)

# Config
DB_PATH = MARKET_DB
ORDERS_LOG = DATA_DIR / "orders.jsonl"
PERFORMANCE_LOG = DATA_DIR / "performance.jsonl"

# Max entries retained in performance log (~80 trading days at ~62/day).
# Well above the 63-day graduation window (2× headroom).
_MAX_PERFORMANCE_ENTRIES = int(os.getenv("MAX_PERFORMANCE_ENTRIES", "5000"))


def _prune_performance_log() -> None:
    """Truncate performance.jsonl to the most recent max entries.

    Runs after each append during paper evaluation. Keeps the log from growing
    unbounded (was 1128 entries for 18 trading dates). File I/O is cheap enough
    for a periodic windowed-prune on a small JSONL file.
    """
    try:
        logfile = Path(PERFORMANCE_LOG)
        if not logfile.exists():
            return
        with open(logfile) as f:
            lines = f.readlines()
        if len(lines) <= _MAX_PERFORMANCE_ENTRIES:
            return
        trimmed = len(lines) - _MAX_PERFORMANCE_ENTRIES
        with open(logfile, 'w') as f:
            f.writelines(lines[-_MAX_PERFORMANCE_ENTRIES:])
        logger.info("Pruned %d entries from performance log (retained %d)", trimmed, _MAX_PERFORMANCE_ENTRIES)
    except (OSError, IOError) as e:
        logger.warning("Failed to prune performance log: %s", e)


# Paper trading config (defaults — override via env vars)
PAPER_CONFIG = {
    "initial_capital": int(os.environ.get("PAPER_INITIAL_CAPITAL", "100000")),
    "max_position_pct": float(os.environ.get("PAPER_MAX_POSITION_PCT", "0.5")),
    "max_drawdown_pct": float(os.environ.get("PAPER_MAX_DRAWDOWN_PCT", "0.15")),
    "rebalance_threshold": float(os.environ.get("PAPER_REBALANCE_THRESHOLD", "0.10")),
    "periodic_rebalance_days": int(os.environ.get("PAPER_PERIODIC_REBALANCE_DAYS", "30")),
    "periodic_rebalance_drift": float(os.environ.get("PAPER_PERIODIC_REBALANCE_DRIFT", "0.02")),
    "volatility_target": float(os.environ.get("PAPER_VOLATILITY_TARGET", "0.12")),
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
        save_results_json(state, output_path=str(self.state_file))
    
    def total_value(self, prices: Dict[str, float]) -> float:
        position_value = sum(
            p.shares * prices.get(p.symbol, p.current_price) 
            for p in self.positions.values()
        )
        return self.cash + position_value

    def _get_daily_returns(self) -> np.ndarray:
        """Extract properly-deduped daily returns from snapshot history.

        The history may contain multiple intraday snapshots per trading day.
        This method deduplicates by date (YYYY-MM-DD), taking the LAST snapshot
        per day, then computes daily returns from the resulting time series.
        Falls back to using the stored daily_return field if only 1 day exists.
        """
        if len(self.history) < 2:
            return np.array([])

        # Group snapshots by trading date (last entry per day wins)
        from collections import OrderedDict
        daily = OrderedDict()
        for h in self.history:
            date_key = h.get("timestamp", "")[:10]  # YYYY-MM-DD
            daily[date_key] = h

        if len(daily) < 2:
            # Fallback: use stored daily_return values directly
            returns = np.array([h.get("daily_return", 0.0) for h in self.history[-252:]])
            # Filter out zero returns (intraday duplicate noise)
            returns = returns[np.abs(returns) > 1e-12]
            return returns if len(returns) >= 2 else np.array([])

        # Compute daily returns from sorted total_value series
        sorted_dates = list(daily.keys())
        values = np.array([daily[d]["total_value"] for d in sorted_dates])
        prev_values = values[:-1]
        curr_values = values[1:]
        daily_returns = np.where(prev_values > 0, (curr_values - prev_values) / prev_values, 0.0)

        return daily_returns

    def current_weights(self, prices: Dict[str, float]) -> Dict[str, float]:
        total = self.total_value(prices)
        if total == 0:
            return {}
        return {
            p.symbol: (p.shares * prices.get(p.symbol, p.current_price)) / total
            for p in self.positions.values()
        }

    def _days_since_last_rebalance(self) -> int:
        """Count days since the last rebalance order was executed.

        Scans order log for the most recent rebalance entry.
        Returns 999 if no rebalance found (forces periodic check).
        """
        try:
            if not ORDERS_LOG.exists():
                return 999
            last_rebalance_date = None
            with open(ORDERS_LOG) as f:
                for line in f:
                    entry = json.loads(line.strip())
                    if "rebalance" in entry.get("reason", ""):
                        ts = entry.get("timestamp", "")
                        if ts:
                            last_rebalance_date = ts[:10]
            if last_rebalance_date is None:
                return 999
            from datetime import date
            last = date.fromisoformat(last_rebalance_date)
            return (date.today() - last).days
        except (OSError, ValueError, KeyError, TypeError):
            return 999
    
    def calculate_orders(self, target_weights: Dict[str, float], prices: Dict[str, float]) -> List[Dict]:
        """Generate orders to move from current to target allocation."""
        total = self.total_value(prices)
        current_weights = self.current_weights(prices)
        orders = []

        # Check if periodic rebalance is warranted
        days_since_rebalance = self._days_since_last_rebalance()
        periodic_due = (
            days_since_rebalance >= PAPER_CONFIG["periodic_rebalance_days"]
            and days_since_rebalance > 0
        )
        periodic_drift = PAPER_CONFIG["periodic_rebalance_drift"]

        for symbol, target_w in target_weights.items():
            if symbol not in prices or prices[symbol] <= 0:
                continue

            current_w = current_weights.get(symbol, 0)
            drift = abs(target_w - current_w)

            # Rebalance if drift exceeds threshold, or periodic rebalance is due
            should_rebalance = (
                drift > PAPER_CONFIG["rebalance_threshold"]
                or (periodic_due and drift > periodic_drift)
            )

            if should_rebalance:
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
        """Check if risk limits breached. Returns kill reason or None.

        Uses graduated kill switch levels (KillSwitchLevel):
        - WARNING (10% DD): reduce position sizes by 25%
        - RESTRICT (15% DD): reduce position sizes by 50%
        - HALT (20% DD): block all new orders, reduce-only mode
        - LIQUIDATE (25% DD or extreme tail risk): emergency liquidation
        """
        total = self.total_value(prices)
        current_drawdown = 0.0

        # Drawdown check (need equity curve)
        if len(self.history) > 20:
            peak = max(h["total_value"] for h in self.history[-252:])  # 1 year lookback
            current_drawdown = (peak - total) / peak if peak > 0 else 0.0

            # Graduated kill switch by drawdown severity
            if current_drawdown >= KILL_SWITCH_THRESHOLDS["liquidate_drawdown_pct"]:
                return f"max_drawdown_{current_drawdown:.2%}"
            if current_drawdown >= KILL_SWITCH_THRESHOLDS["halt_drawdown_pct"]:
                return f"max_drawdown_{current_drawdown:.2%}"
            if current_drawdown >= KILL_SWITCH_THRESHOLDS["restrict_drawdown_pct"]:
                return f"max_drawdown_{current_drawdown:.2%}"
            if current_drawdown >= KILL_SWITCH_THRESHOLDS["warning_drawdown_pct"]:
                return f"max_drawdown_{current_drawdown:.2%}"

        # GARCH-CVaR tail risk check
        daily_returns = self._get_daily_returns()
        if len(daily_returns) >= 10:  # Minimum for EWMA fallback in GARCH-CVaR
            try:
                from src.monitor.garch_cvar import calculate_garch_cvar
                recent_returns = daily_returns[-min(252, len(daily_returns)):]
                current_dd = min(0.0, -current_drawdown) if current_drawdown > 0 else 0.0
                metrics = calculate_garch_cvar(
                    returns=recent_returns,
                    current_drawdown=current_dd,
                    max_drawdown=-PAPER_CONFIG["max_drawdown_pct"],
                )
                # Write GARCH health report for dashboard consumption
                self._write_garch_health_report(metrics)
                # Trigger kill if CVaR exceeds 3× VaR (extreme tail risk)
                if metrics.cvar_ratio > KILL_SWITCH_THRESHOLDS["extreme_tail_cvar_ratio"] and metrics.filter_active:
                    return f"extreme_tail_risk_cvar_ratio_{metrics.cvar_ratio:.1f}"
            except (KeyError, ValueError, TypeError, AttributeError, RuntimeError, ImportError, OSError) as e:
                logger.warning("GARCH-CVaR computation failed, skipping tail risk check: %s", e)

        # Position concentration check
        for p in self.positions.values():
            if p.weight > PAPER_CONFIG["max_position_pct"]:
                return f"max_position_{p.symbol}_{p.weight:.2%}"

        return None

    def _write_garch_health_report(self, metrics) -> None:
        """Write GARCH-CVaR metrics to .health_report.json for dashboard."""
        try:
            from dataclasses import asdict
            report_path = DATA_DIR / ".health_report.json"
            data = asdict(metrics) if hasattr(metrics, '__dataclass_fields__') else {}

            # Add portfolio entropy metrics from current allocation
            data["checks"] = data.get("checks", {})
            data["checks"]["portfolio_entropy"] = self._compute_portfolio_entropy()

            # Derive top-level status for unified dashboard compatibility
            tail = data.get("tail_severity", "normal")
            cvar_ratio = data.get("cvar_ratio", 1.0)
            if tail in ("extreme", "severe") or cvar_ratio > 3.0:
                data["status"] = "unhealthy"
            else:
                data["status"] = "healthy"
            data["summary"] = {"passed": 1, "total_checks": 1}

            save_results_json(data, output_path=str(report_path))
        except (OSError, json.JSONDecodeError, KeyError, ValueError, TypeError, AttributeError) as e:
            logger.warning("Failed to write GARCH health report: %s", e)

    @staticmethod
    def _compute_portfolio_entropy() -> dict:
        """Compute portfolio concentration metrics from BASE_ALLOCATION.

        Returns Shannon entropy, effective number of assets, normalized
        score (0-100), and Herfindahl-Hirschman Index (HHI).
        """
        import math
        from src.paths import BASE_ALLOCATION

        weights = list(BASE_ALLOCATION.values())
        n = len(weights)

        # Shannon entropy: H = -sum(w_i * ln(w_i))
        shannon = -sum(w * math.log(w) for w in weights if w > 0)

        # Effective number of assets: exp(H)
        effective_n = math.exp(shannon)

        # Normalized score: H / H_max * 100, where H_max = ln(n)
        h_max = math.log(n) if n > 1 else 1.0
        normalized_score = (shannon / h_max) * 100.0 if h_max > 0 else 0.0

        # HHI: sum(w_i^2) — ranges from 1/n (equal) to 1 (concentrated)
        hhi = sum(w * w for w in weights)

        return {
            "name": "portfolio_entropy",
            "status": "good" if normalized_score > 90 else "warning",
            "ok": normalized_score > 70,
            "metrics": {
                "shannon_entropy": round(shannon, 4),
                "effective_n": round(effective_n, 2),
                "normalized_score": round(normalized_score, 1),
                "hhi_index": round(hhi, 4),
            },
        }

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

    # Try to get trend signal from regime_log
    cursor.execute("SELECT regime FROM regime_log ORDER BY detected_at DESC LIMIT 1")
    trend_row = cursor.fetchone()
    trend_regime = trend_row[0] if trend_row else "normal"

    return classify_vix_regime(vix_level, trend_regime)

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

    # Calculate daily return relative to previous day's close.
    # Comparing against the last history entry (which may be intraday)
    # produces daily_return=0 for most snapshots. Instead, find the
    # close (16:00 or last entry) from the most recent completed day.
    daily_return = 0.0
    if portfolio.history:
        today = datetime.now().strftime("%Y-%m-%d")
        prev_day_total = None

        # Collect all non-today entries, grouped by date
        prev_entries: dict = {}
        for entry in portfolio.history:
            ts = entry.get("timestamp", "")
            entry_date = ts[:10] if len(ts) >= 10 else ""
            if entry_date and entry_date != today:
                prev_entries.setdefault(entry_date, []).append(entry)

        if prev_entries:
            # Prefer the most recent date with a 16:00 close entry
            for date in sorted(prev_entries.keys(), reverse=True):
                entries = prev_entries[date]
                for e in entries:
                    if "16:00" in e.get("timestamp", ""):
                        prev_day_total = e.get("total_value")
                        break
                if prev_day_total is not None:
                    break
            # Fallback: most recent non-today date, last entry
            if prev_day_total is None:
                latest_date = max(prev_entries.keys())
                prev_day_total = prev_entries[latest_date][-1].get("total_value")

        # Fallback: if no previous-day entry, use last entry (first day)
        if prev_day_total is None:
            prev_day_total = portfolio.history[-1].get("total_value", 0)
        if prev_day_total and prev_day_total > 0:
            daily_return = (total - prev_day_total) / prev_day_total

    return {
        "timestamp": datetime.now().isoformat(),
        "total_value": total,
        "cash": portfolio.cash,
        "daily_return": daily_return,
        "positions_count": len(portfolio.positions),
        "mode": portfolio.mode
    }


def classify_kill_level(reason: str) -> KillSwitchLevel:
    """Determine the graduated kill switch level from the breach reason.

    Levels are based on drawdown severity and risk type:
    - WARNING: drawdown 10-15% or moderate concentration breach
    - RESTRICT: drawdown 15-20%
    - HALT: drawdown 20-25% or max_drawdown breach
    - LIQUIDATE: drawdown >25% or extreme tail risk (CVaR ratio >3)
    """
    if not reason:
        return KillSwitchLevel.NONE

    # Extreme tail risk → LIQUIDATE
    if "extreme_tail_risk" in reason:
        return KillSwitchLevel.LIQUIDATE

    # Parse drawdown percentage from reason string
    if "max_drawdown_" in reason:
        try:
            dd_str = reason.split("max_drawdown_")[1].rstrip("%")
            raw_val = float(dd_str)
            # Convert to 0-1 fraction: formats like "-12.0%" or "12.0%"
            dd_pct = raw_val / 100 if abs(raw_val) > 1 else raw_val
            dd_abs = abs(dd_pct)

            if dd_abs >= KILL_SWITCH_THRESHOLDS["liquidate_drawdown_pct"]:
                return KillSwitchLevel.LIQUIDATE
            elif dd_abs >= KILL_SWITCH_THRESHOLDS["halt_drawdown_pct"]:
                return KillSwitchLevel.HALT
            elif dd_abs >= KILL_SWITCH_THRESHOLDS["restrict_drawdown_pct"]:
                return KillSwitchLevel.RESTRICT
            elif dd_abs >= KILL_SWITCH_THRESHOLDS["warning_drawdown_pct"]:
                return KillSwitchLevel.WARNING
        except (ValueError, IndexError):
            pass
        # Default for unparseable drawdown → HALT (fail-closed)
        return KillSwitchLevel.HALT

    # Position concentration → WARNING (less severe than drawdown)
    if "max_position_" in reason:
        return KillSwitchLevel.WARNING

    # Unknown breach → HALT (fail-closed)
    return KillSwitchLevel.HALT


def _kill_level_reduction(level: KillSwitchLevel) -> float:
    """Position size reduction fraction for each kill switch level.

    Returns 0.0 (no reduction) to 1.0 (full liquidation).
    """
    return {
        KillSwitchLevel.NONE: 0.0,
        KillSwitchLevel.WARNING: 0.25,
        KillSwitchLevel.RESTRICT: 0.50,
        KillSwitchLevel.HALT: 1.0,
        KillSwitchLevel.LIQUIDATE: 1.0,
    }.get(level, 1.0)


def main():
    """Main evaluation loop."""
    logger.info("Strategy Evaluator Starting")

    # Determine mode from environment
    mode = os.environ.get("ALPHALAB_MODE", "paper")
    state_file = DATA_DIR / f"portfolio_{mode}.json"

    with sqlite_connect(DB_PATH) as conn:
        portfolio = Portfolio(state_file, mode)

        # Get current state
        prices = get_latest_prices(conn)
        regime = get_current_regime(conn)
        vix = get_latest_vix(conn)

    if vix:
        logger.info("Mode: %s, Regime: %s, VIX: %.2f", mode, regime, vix)
    else:
        logger.info("Mode: %s, Regime: %s", mode, regime)
    logger.info("Portfolio value: $%.2f", portfolio.total_value(prices))

    # Check kill switches
    kill_reason = portfolio.check_risk_limits(prices)
    if kill_reason:
        kill_level = classify_kill_level(kill_reason)
        logger.critical("KILL SWITCH TRIGGERED: level=%s reason=%s", kill_level.value, kill_reason)
        # Write kill_switch.json with graduated level (read by order_router and dashboard)
        save_results_json({
            "enabled": True,
            "level": kill_level.value,
            "reason": kill_reason,
            "mode": mode,
            "timestamp": datetime.now().isoformat(),
            "position_reduction": _kill_level_reduction(kill_level),
        }, output_path=str(DATA_DIR / "kill_switch.json"))
        return

    # Clear stale kill switch if risk limits are no longer breached
    kill_file = DATA_DIR / "kill_switch.json"
    if kill_file.exists():
        kill_file.unlink(missing_ok=True)
        logger.info("Kill switch cleared for %s — risk limits no longer breached", mode)

    # Determine target allocation
    target_alloc = REGIME_OVERRIDES.get(regime) or BASE_ALLOCATION
    logger.info("Target allocation: %s", target_alloc)

    # Generate orders
    orders = portfolio.calculate_orders(target_alloc, prices)

    if orders:
        logger.info("Generated %d orders:", len(orders))
        for o in orders:
            logger.info("  %s %.2f %s @ $%.2f", o['side'].upper(), o['shares'], o['symbol'], o['estimated_price'])

        # Execute (paper trading with slippage)
        executed = portfolio.execute_orders(orders, prices)

        # Log orders
        with open(ORDERS_LOG, 'a') as f:
            for e in executed:
                f.write(json.dumps(e) + '\n')

        logger.info("Executed %d orders", len(executed))
    else:
        logger.info("No rebalancing needed")
    
    # Update and save state
    perf = calculate_performance(portfolio, prices)
    portfolio.history.append(perf)
    portfolio.save_state()
    
    # Log performance
    with open(PERFORMANCE_LOG, 'a') as f:
        f.write(json.dumps(perf) + '\n')

    _prune_performance_log()
    
    # Check graduation criteria (paper mode only)
    if mode == "paper":
        check_graduation_criteria(portfolio)
    
    logger.info("Evaluation complete")

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
    MIN_DAYS = int(os.environ.get("GRADUATION_MIN_DAYS", "63"))
    MIN_SHARPE = float(os.environ.get("GRADUATION_MIN_SHARPE", "0.5"))
    MAX_DD = float(os.environ.get("GRADUATION_MAX_DD", "0.15"))
    MIN_WIN_RATE = float(os.environ.get("GRADUATION_MIN_WIN_RATE", "0.45"))
    MAX_REALISTIC_SHARPE = float(os.environ.get("GRADUATION_MAX_REALISTIC_SHARPE", "3.0"))
    MIN_DSR = float(os.environ.get("GRADUATION_MIN_DSR", "0.50"))
    
    if len(portfolio.history) < MIN_DAYS:
        return
    
    # Deduplicate intra-day snapshots to trading-day-level data
    daily_history = _deduplicate_to_daily(portfolio.history)
    
    # Need at least MIN_DAYS trading days after dedup
    if len(daily_history) < MIN_DAYS:
        logger.info("GRADUATION DEFERRED: Only %d unique trading days (need %d), "
                     "skipping intra-day snapshots", len(daily_history), MIN_DAYS)
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
        logger.warning("Sharpe %.2f exceeds realistic maximum %.1f — likely intra-day "
                        "snapshot contamination. Skipping promotion.", sharpe, MAX_REALISTIC_SHARPE)
        return
    
    # Check criteria
    # DSR validation: confirm Sharpe survives multiple-testing correction
    # With 94 grid-search configs, DSR > 0.95 means the Sharpe is statistically
    # significant, not just the best of many trials
    # (MIN_DSR is now externalized via GRADUATION_MIN_DSR env var, set above)
    try:
        from src.backtest.metrics import compute_deflated_sharpe_ratio
        dsr = compute_deflated_sharpe_ratio(
            sharpe_ratio=sharpe, n_trials=94, n_observations=len(returns),
        )
    except (ImportError, ValueError, ZeroDivisionError, OverflowError):
        dsr = 0.0  # If DSR can't be computed, fail closed

    if sharpe > MIN_SHARPE and max_dd < MAX_DD and win_rate > MIN_WIN_RATE and dsr >= MIN_DSR:
        logger.info("GRADUATION CANDIDATE: Sharpe=%.2f, DD=%.2f%%, "
                     "WinRate=%.2f%%, DSR=%.2f", sharpe, max_dd * 100, win_rate * 100, dsr)

        # Create promotion trigger
        trigger = {
            "action": "promote_to_live",
            "metrics": {
                "sharpe": round(sharpe, 2),
                "max_drawdown": round(max_dd, 4),
                "win_rate": round(win_rate, 4),
                "total_return": round(total_return, 6),
                "dsr": round(dsr, 4),
            },
            "timestamp": datetime.now().isoformat(),
            "requires_approval": True
        }
        
        trigger_path = DATA_DIR / ".promote_to_live"
        save_results_json(trigger, output_path=str(trigger_path))
        
        logger.info("Created promotion trigger: %s", trigger_path)

if __name__ == "__main__":
    main()

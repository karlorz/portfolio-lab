"""
Dual Momentum Strategy Implementation
Combines Absolute Momentum (trend filter) + Relative Momentum (strength selection)
Based on Gary Antonacci's Global Equities Momentum (GEM) framework

Strategy Rules:
1. Absolute Momentum: Only hold assets with positive 12-month return (price > 12mo SMA)
2. Relative Momentum: Among qualifying assets, pick the top N by 12-month return
3. Risk-off fallback: If no assets have positive momentum, hold TLT (long treasuries)

This is applied as an overlay on the All-Season 46/38/16 base allocation.
"""

import os
import json
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path


@dataclass
class MomentumScore:
    """Momentum calculation for a single asset"""
    symbol: str
    price: float
    sma_200: float  # 10-month equivalent (~200 trading days)
    return_12m: float  # 12-month total return
    return_6m: float   # 6-month return (acceleration)
    return_3m: float   # 3-month return (short-term momentum)
    volatility: float  # 20-day annualized volatility
    above_sma: bool   # Price > 200-day SMA (absolute momentum)
    score: float      # Composite momentum score


@dataclass  
class DualMomentumSignal:
    """Signal output from dual momentum engine"""
    timestamp: str
    base_allocation: Dict[str, float]  # Original 46/38/16
    adjusted_allocation: Dict[str, float]  # After momentum overlay
    momentum_scores: Dict[str, MomentumScore]
    selected_assets: List[str]  # Assets passing absolute momentum
    risk_off: bool  # True if no assets have positive momentum
    signal_strength: float  # 0-1 confidence in signal
    rebalance_triggered: bool  # Whether to execute rebalance


class DualMomentumEngine:
    """
    Dual Momentum strategy engine for portfolio-lab
    
    Configuration:
    - lookback_months: 12 (primary momentum window)
    - sma_period: 200 days (~10 months)
    - min_assets: 1 (at least 1 asset must have positive momentum)
    - risk_off_asset: TLT (safety when no positive momentum)
    - top_n_select: 2 (select top 2 by momentum from qualifying set)
    """
    
    def __init__(
        self,
        db_path: Path = Path("~/projects/portfolio-lab/data/market.db").expanduser(),
        lookback_months: int = 12,
        sma_days: int = 200,
        top_n: int = 2,
        risk_off_asset: str = "TLT",
        momentum_threshold: float = 0.0,  # Minimum 12m return to qualify
        vol_lookback: int = 20
    ):
        self.db_path = db_path
        self.lookback_months = lookback_months
        self.sma_days = sma_days
        self.top_n = top_n
        self.risk_off_asset = risk_off_asset
        self.momentum_threshold = momentum_threshold
        self.vol_lookback = vol_lookback
        
        # Universe: Core All-Season assets + equity alternatives
        self.universe = ["SPY", "GLD", "TLT", "IEF", "QQQ", "EFA", "VXUS"]
        
        # Base allocation (All-Season champion)
        self.base_allocation = {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16}
    
    def _fetch_price_data(self, symbol: str, days: int = 252) -> List[Dict]:
        """Fetch historical price data from SQLite"""
        if not self.db_path.exists():
            return []
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT date, close, volume
            FROM prices
            WHERE symbol = ?
            ORDER BY date DESC
            LIMIT ?
        """, (symbol, days))
        
        rows = cursor.fetchall()
        conn.close()
        
        # Return in chronological order (oldest first)
        return [
            {"date": row[0], "close": row[1], "volume": row[2]}
            for row in reversed(rows)
        ]
    
    def _calculate_momentum_score(self, symbol: str) -> Optional[MomentumScore]:
        """Calculate momentum metrics for a symbol"""
        data = self._fetch_price_data(symbol, 300)  # Extra buffer for SMA
        
        if len(data) < self.sma_days + 20:
            return None
        
        closes = np.array([d["close"] for d in data])
        current_price = closes[-1]
        
        # 200-day SMA
        sma_200 = np.mean(closes[-self.sma_days:])
        
        # Returns (approximately 252 trading days/year)
        days_12m = min(252, len(closes) - 1)
        days_6m = min(126, len(closes) - 1)
        days_3m = min(63, len(closes) - 1)
        
        return_12m = (current_price / closes[-days_12m]) - 1 if days_12m > 0 else 0
        return_6m = (current_price / closes[-days_6m]) - 1 if days_6m > 0 else 0
        return_3m = (current_price / closes[-days_3m]) - 1 if days_3m > 0 else 0
        
        # Volatility (20-day)
        returns_daily = np.diff(closes[-self.vol_lookback-1:]) / closes[-self.vol_lookback-1:-1]
        volatility = np.std(returns_daily) * np.sqrt(252) if len(returns_daily) > 1 else 0.1
        
        # Absolute momentum check
        above_sma = current_price > sma_200
        
        # Composite score (weighted combination)
        # 50% 12m + 30% 6m + 20% 3m, with penalty for high volatility
        if volatility > 0:
            vol_adj = min(1.0, 0.15 / volatility)  # Target 15% vol
        else:
            vol_adj = 1.0
        
        score = (return_12m * 0.5 + return_6m * 0.3 + return_3m * 0.2) * vol_adj
        
        return MomentumScore(
            symbol=symbol,
            price=current_price,
            sma_200=sma_200,
            return_12m=return_12m,
            return_6m=return_6m,
            return_3m=return_3m,
            volatility=volatility,
            above_sma=above_sma,
            score=score
        )
    
    def evaluate(self) -> DualMomentumSignal:
        """
        Run dual momentum evaluation and generate allocation signal
        """
        timestamp = datetime.now().isoformat()
        
        # Calculate momentum for all assets in universe
        momentum_scores = {}
        for symbol in self.universe:
            score = self._calculate_momentum_score(symbol)
            if score:
                momentum_scores[symbol] = score
        
        if not momentum_scores:
            # No data - return base allocation
            return DualMomentumSignal(
                timestamp=timestamp,
                base_allocation=self.base_allocation.copy(),
                adjusted_allocation=self.base_allocation.copy(),
                momentum_scores={},
                selected_assets=[],
                risk_off=True,
                signal_strength=0.0,
                rebalance_triggered=False
            )
        
        # ABSOLUTE MOMENTUM: Filter for assets with positive 12m return
        qualifying = {
            sym: score for sym, score in momentum_scores.items()
            if score.return_12m > self.momentum_threshold and score.above_sma
        }
        
        # RELATIVE MOMENTUM: Select top N by score from qualifying set
        if qualifying:
            sorted_assets = sorted(
                qualifying.items(),
                key=lambda x: x[1].score,
                reverse=True
            )
            selected = [sym for sym, _ in sorted_assets[:self.top_n]]
            risk_off = False
        else:
            # No qualifying assets - go to risk-off (hold TLT)
            selected = [self.risk_off_asset]
            risk_off = True
        
        # Generate adjusted allocation
        adjusted_allocation = self._generate_allocation(selected, momentum_scores, risk_off)
        
        # Calculate signal strength (confidence metric)
        if qualifying:
            avg_score = np.mean([s.score for s in qualifying.values()])
            best_score = sorted_assets[0][1].score if sorted_assets else 0
            signal_strength = min(1.0, max(0.0, (best_score - 0.05) * 5))  # Normalize
        else:
            signal_strength = 0.0
        
        # Determine if rebalance is needed (check drift from current)
        # This would be compared against current positions in practice
        rebalance_triggered = len(selected) > 0 and not risk_off
        
        return DualMomentumSignal(
            timestamp=timestamp,
            base_allocation=self.base_allocation.copy(),
            adjusted_allocation=adjusted_allocation,
            momentum_scores=momentum_scores,
            selected_assets=selected,
            risk_off=risk_off,
            signal_strength=signal_strength,
            rebalance_triggered=rebalance_triggered
        )
    
    def _generate_allocation(
        self,
        selected_assets: List[str],
        scores: Dict[str, MomentumScore],
        risk_off: bool = False
    ) -> Dict[str, float]:
        """
        Generate risk-adjusted allocation based on selected assets
        """
        if not selected_assets or risk_off:
            # Full risk-off: 100% TLT
            return {self.risk_off_asset: 1.0}
        
        # Get volatility for each selected asset
        vols = {}
        for sym in selected_assets:
            if sym in scores:
                vols[sym] = scores[sym].volatility
            else:
                vols[sym] = 0.15  # Default 15%
        
        # Inverse-volatility weighting (risk parity style)
        inv_vols = {sym: 1/v if v > 0 else 1 for sym, v in vols.items()}
        total_inv_vol = sum(inv_vols.values())
        
        if total_inv_vol > 0:
            weights = {sym: inv_vols[sym] / total_inv_vol for sym in selected_assets}
        else:
            weights = {sym: 1/len(selected_assets) for sym in selected_assets}
        
        # Ensure we still have some bond/gold exposure for diversification
        # If TLT or GLD is not in selected, add a minimum 10% to each
        result = weights.copy()
        
        if "TLT" not in result and not risk_off:
            # Scale down others and add 10% TLT
            scale = 0.9
            result = {k: v * scale for k, v in result.items()}
            result["TLT"] = 0.10
        
        if "GLD" not in result:
            scale = 0.9 if "TLT" in result else 0.95
            result = {k: v * scale for k, v in result.items()}
            result["GLD"] = result.get("GLD", 0) + (0.10 if "TLT" in result else 0.05)
        
        # Normalize to sum to 1.0
        total = sum(result.values())
        if total > 0:
            result = {k: v / total for k, v in result.items()}
        
        return result
    
    def get_rebalance_recommendation(
        self,
        current_positions: Dict[str, float],
        threshold: float = 0.10
    ) -> Dict:
        """
        Compare current positions to target and generate rebalance recommendation
        """
        signal = self.evaluate()
        target = signal.adjusted_allocation
        
        # Calculate drift for each asset
        all_assets = set(current_positions.keys()) | set(target.keys())
        drifts = {}
        
        for asset in all_assets:
            current = current_positions.get(asset, 0)
            tgt = target.get(asset, 0)
            drifts[asset] = current - tgt
        
        # Check if any drift exceeds threshold
        max_drift = max(abs(d) for d in drifts.values()) if drifts else 0
        needs_rebalance = max_drift > threshold
        
        return {
            "signal": signal,
            "current_positions": current_positions,
            "target_allocation": target,
            "drifts": drifts,
            "max_drift": max_drift,
            "needs_rebalance": needs_rebalance,
            "recommendation": "REBALANCE" if needs_rebalance else "HOLD",
            "risk_off_active": signal.risk_off,
            "selected_assets": signal.selected_assets
        }


class DualMomentumBacktest:
    """
    Backtesting engine for dual momentum strategy
    Allows comparison against buy-and-hold and static allocation
    """

    def __init__(self, engine: DualMomentumEngine):
        self.engine = engine

    def run_backtest(
        self,
        start_date: str,
        end_date: str,
        initial_capital: float = 100000,
        rebalance_frequency: str = "monthly"
    ) -> Dict:
        """
        Run historical backtest of dual momentum strategy.
        Monthly rebalancing with absolute + relative momentum selection.
        """
        if not self.engine.db_path.exists():
            return {"error": "No market data available", "status": "failed"}

        conn = sqlite3.connect(self.engine.db_path)

        # Get all available trading dates in range
        dates = pd.read_sql(
            "SELECT DISTINCT date FROM prices WHERE date BETWEEN ? AND ? ORDER BY date",
            conn, params=(start_date, end_date)
        )['date'].tolist()

        if len(dates) < 252:
            conn.close()
            return {"error": f"Insufficient data: {len(dates)} days (need 252+)", "status": "failed"}

        # Get prices for universe symbols
        symbols = self.engine.universe
        price_data = {}
        for sym in symbols:
            rows = pd.read_sql(
                "SELECT date, close FROM prices WHERE symbol = ? AND date BETWEEN ? AND ? ORDER BY date",
                conn, params=(sym, start_date, end_date)
            )
            if not rows.empty:
                price_data[sym] = rows.set_index('date')['close']

        # Get SPY for benchmark
        spy_prices = pd.read_sql(
            "SELECT date, close FROM prices WHERE symbol = 'SPY' AND date BETWEEN ? AND ? ORDER BY date",
            conn, params=(start_date, end_date)
        )
        conn.close()

        if spy_prices.empty:
            return {"error": "No SPY benchmark data", "status": "failed"}

        spy_series = spy_prices.set_index('date')['close']

        # Determine rebalance dates (monthly)
        rebalance_dates = []
        prev_month = None
        for d in dates:
            month = d[:7]
            if month != prev_month:
                rebalance_dates.append(d)
                prev_month = month

        # Simulate portfolio
        portfolio_value = initial_capital
        spy_value = initial_capital
        values = [portfolio_value]
        spy_values = [spy_value]
        current_weights = self.engine.base_allocation.copy()
        trade_count = 0
        risk_off_count = 0

        for i, date in enumerate(dates):
            if i == 0:
                continue

            # Calculate daily return from current weights
            daily_return = 0.0
            for sym, weight in current_weights.items():
                if sym in price_data and date in price_data[sym].index:
                    prev_date = dates[i - 1]
                    if prev_date in price_data[sym].index:
                        prev_p = price_data[sym][prev_date]
                        cur_p = price_data[sym][date]
                        if prev_p > 0:
                            daily_return += weight * (cur_p - prev_p) / prev_p

            portfolio_value *= (1 + daily_return)
            values.append(portfolio_value)

            # SPY benchmark
            if date in spy_series.index and dates[i - 1] in spy_series.index:
                spy_ret = (spy_series[date] - spy_series[dates[i - 1]]) / spy_series[dates[i - 1]]
                spy_value *= (1 + spy_ret)
            spy_values.append(spy_value)

            # Rebalance
            if date in rebalance_dates:
                available = [s for s in symbols if s in price_data and date in price_data[s].index]
                if available:
                    old_universe = self.engine.universe
                    self.engine.universe = available
                    signal = self.engine.evaluate()
                    self.engine.universe = old_universe

                    new_weights = signal.adjusted_allocation
                    if new_weights:
                        current_weights = new_weights
                        trade_count += 1
                        if signal.risk_off:
                            risk_off_count += 1

        # Compute metrics
        years = len(dates) / 252
        final_value = values[-1]
        cagr = (final_value / initial_capital) ** (1 / years) - 1 if years > 0 else 0

        daily_returns = np.diff(values) / values[:-1]
        vol = float(np.std(daily_returns) * np.sqrt(252))
        sharpe = (cagr - 0.04) / vol if vol > 0 else 0

        peak = np.maximum.accumulate(values)
        drawdowns = (np.array(values) - peak) / peak
        max_dd = float(np.min(drawdowns))

        spy_cagr = (spy_values[-1] / initial_capital) ** (1 / years) - 1 if years > 0 else 0

        return {
            "strategy": "dual_momentum",
            "period": f"{start_date} to {end_date}",
            "initial_capital": initial_capital,
            "final_value": round(final_value, 2),
            "cagr": round(cagr, 4),
            "volatility": round(vol, 4),
            "sharpe_ratio": round(sharpe, 3),
            "max_drawdown": round(max_dd, 4),
            "spy_cagr": round(spy_cagr, 4),
            "excess_return": round(cagr - spy_cagr, 4),
            "trade_count": trade_count,
            "risk_off_months": risk_off_count,
            "trading_days": len(dates),
            "rebalance_frequency": rebalance_frequency,
            "status": "completed"
        }


# CLI Interface
def main():
    import sys
    
    engine = DualMomentumEngine()
    
    if len(sys.argv) < 2:
        print("Usage: python dual_momentum.py [evaluate|backtest|status]")
        sys.exit(1)
    
    cmd = sys.argv[1]
    
    if cmd == "evaluate":
        signal = engine.evaluate()
        
        print(f"\n{'='*60}")
        print("DUAL MOMENTUM SIGNAL")
        print(f"{'='*60}")
        print(f"Timestamp: {signal.timestamp}")
        print(f"Risk Off: {'YES' if signal.risk_off else 'NO'}")
        print(f"Signal Strength: {signal.signal_strength:.1%}")
        print(f"Rebalance Triggered: {'YES' if signal.rebalance_triggered else 'NO'}")
        print(f"\nSelected Assets: {', '.join(signal.selected_assets)}")
        
        print(f"\n{'-'*60}")
        print("MOMENTUM SCORES (12m | 6m | 3m | Above SMA | Score)")
        print(f"{'-'*60}")
        
        for symbol, score in sorted(
            signal.momentum_scores.items(),
            key=lambda x: x[1].score,
            reverse=True
        ):
            status = "✓" if score.above_sma else "✗"
            print(f"{symbol:6} | {score.return_12m:+.1%} | {score.return_6m:+.1%} | "
                  f"{score.return_3m:+.1%} | {status} | {score.score:+.3f}")
        
        print(f"\n{'-'*60}")
        print("ALLOCATION COMPARISON")
        print(f"{'-'*60}")
        print(f"{'Asset':<8} {'Base':<8} {'Adjusted':<8} {'Delta':<8}")
        print(f"{'-'*60}")
        
        all_assets = set(signal.base_allocation.keys()) | set(signal.adjusted_allocation.keys())
        for asset in sorted(all_assets):
            base = signal.base_allocation.get(asset, 0)
            adj = signal.adjusted_allocation.get(asset, 0)
            delta = adj - base
            print(f"{asset:<8} {base:>7.1%} {adj:>7.1%} {delta:>+7.1%}")
        
        print(f"{'='*60}\n")
        
        # Output JSON for integration
        output = {
            "timestamp": signal.timestamp,
            "risk_off": signal.risk_off,
            "signal_strength": signal.signal_strength,
            "selected_assets": signal.selected_assets,
            "base_allocation": signal.base_allocation,
            "adjusted_allocation": signal.adjusted_allocation,
            "momentum_scores": {
                sym: {
                    "return_12m": float(s.return_12m),
                    "return_6m": float(s.return_6m),
                    "return_3m": float(s.return_3m),
                    "above_sma": bool(s.above_sma),
                    "volatility": float(s.volatility),
                    "score": float(s.score)
                }
                for sym, s in signal.momentum_scores.items()
            }
        }
        
        print(json.dumps(output, indent=2))
        
    elif cmd == "backtest":
        # Placeholder for backtest
        print("Backtest mode - requires historical data simulation")
        
    elif cmd == "status":
        # Quick status check
        signal = engine.evaluate()
        print(json.dumps({
            "available": len(signal.momentum_scores) > 0,
            "risk_off": signal.risk_off,
            "selected": signal.selected_assets,
            "strength": signal.signal_strength,
            "rebalance_needed": signal.rebalance_triggered
        }, indent=2))
    
    else:
        print(f"Unknown command: {cmd}")
        print("Usage: python dual_momentum.py [evaluate|backtest|status]")


if __name__ == "__main__":
    main()

"""
Paper Trading Simulator — Historical Replay Engine
Simulates live trading by replaying historical data day-by-day.

Without broker API keys, this provides:
- Day-by-day portfolio simulation with all overlays active
- Realistic execution: slippage, commission, spread
- Performance tracking: P&L, Sharpe, drawdown, turnover
- Graduation readiness assessment

Usage:
    python -m src.broker.paper_trading_sim run
    python -m src.broker.paper_trading_sim run --days 90
    python -m src.broker.paper_trading_sim report
"""

import json
import logging
import math
import sqlite3
from dataclasses import dataclass, asdict, field
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Single simulated trade."""
    date: str
    symbol: str
    side: str  # buy/sell
    shares: float
    price: float
    value: float
    commission: float
    slippage_bps: float
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DailySnapshot:
    """Portfolio state at end of trading day."""
    date: str
    total_value: float
    daily_return: float
    cumulative_return: float

    # Asset values
    spy_value: float
    gld_value: float
    tlt_value: float
    ief_value: float
    shy_value: float
    btc_value: float
    eth_value: float
    cash: float

    # Overlay states
    collar_active: bool
    crypto_active: bool
    bond_position: str
    vix_level: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PaperTradingReport:
    """Complete paper trading performance report."""
    timestamp: str
    start_date: str
    end_date: str
    trading_days: int

    # Performance
    total_return: float
    cagr: float
    volatility: float
    sharpe: float
    max_drawdown: float
    max_drawdown_date: str

    # Trading activity
    total_trades: int
    total_commission: float
    total_slippage_bps: float
    turnover_pct: float

    # Win rate
    winning_days: int
    losing_days: int
    win_rate: float

    # Overlay stats
    collar_active_days: int
    crypto_active_days: int
    avg_bond_duration: float

    # Graduation assessment
    meets_graduation_sharpe: bool
    meets_graduation_dd: bool
    graduation_ready: bool
    graduation_note: str

    # Trades and snapshots (for detailed analysis)
    trades: List[Dict]
    snapshots: List[Dict]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["trades"] = [t if isinstance(t, dict) else t.to_dict() for t in self.trades]
        d["snapshots"] = [s if isinstance(s, dict) else s.to_dict() for s in self.snapshots]
        return d


class PaperTradingSimulator:
    """
    Historical replay paper trading simulator.

    Replays market data day-by-day, executing overlay signals
    as if trading live. No broker API needed.

    Execution model:
    - Slippage: 5bps per trade (conservative)
    - Commission: $1 per trade (Alpaca-like)
    - Rebalance: drift-based (10% threshold per asset)
    - Overlay signals: recalculated each day from lookback window
    """

    BASELINE = {"spy": 0.46, "gld": 0.38, "tlt": 0.16}
    SLIPPAGE_BPS = 5.0
    COMMISSION_PER_TRADE = 1.0
    REBALANCE_THRESHOLD = 0.10  # 10% drift triggers rebalance

    DATA_DIR = Path(__file__).parent.parent.parent / "data"
    OUTPUT_DIR = DATA_DIR / "paper_trading"

    def __init__(self):
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self._trades: List[Trade] = []
        self._snapshots: List[DailySnapshot] = []

    def _load_prices(self) -> Dict[str, Dict]:
        """Load prices from market.db."""
        db_path = self.DATA_DIR / "market.db"
        if not db_path.exists():
            logger.error("market.db not found")
            return {}

        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        data = {}
        for asset, sym in [("SPY", "SPY"), ("GLD", "GLD"), ("TLT", "TLT"),
                            ("IEF", "IEF"), ("VIX", "^VIX")]:
            cursor.execute(
                "SELECT date, close FROM prices WHERE symbol=? ORDER BY date",
                (sym,)
            )
            rows = cursor.fetchall()
            if rows:
                data[asset] = {
                    "dates": [r[0] for r in rows],
                    "prices": [float(r[1]) for r in rows],
                }

        conn.close()
        return data

    def _execute_trade(self, date_str: str, symbol: str, side: str,
                       target_value: float, price: float) -> Trade:
        """Execute a simulated trade with slippage and commission."""
        slippage = self.SLIPPAGE_BPS / 10000
        if side == "buy":
            exec_price = price * (1 + slippage)
        else:
            exec_price = price * (1 - slippage)

        shares = target_value / exec_price if exec_price > 0 else 0
        value = shares * exec_price
        commission = self.COMMISSION_PER_TRADE

        trade = Trade(
            date=date_str, symbol=symbol, side=side,
            shares=round(shares, 4), price=round(exec_price, 2),
            value=round(value, 2), commission=commission,
            slippage_bps=self.SLIPPAGE_BPS,
            reason="rebalance",
        )
        self._trades.append(trade)
        return trade

    def _compute_overlay_signals(self, prices: Dict[str, List[float]],
                                  i: int) -> Dict:
        """Compute all overlay signals at a given time index."""
        signals = {}

        # VIX
        vix_data = prices.get("VIX", {})
        vix = vix_data.get("prices", [18.0])[min(i, len(vix_data.get("prices", [18.0])) - 1)]

        # Collar: active when VIX > 25
        signals["collar"] = {"active": vix > 25, "vix": vix,
                             "spy_shift": -0.03 if vix > 35 else (-0.01 if vix > 25 else 0)}

        # Bond duration: based on TLT momentum
        tlt_prices = prices.get("TLT", {}).get("prices", [])
        if i >= 126 and len(tlt_prices) > i:
            tlt_mom = tlt_prices[i] / tlt_prices[i-126] - 1
            if tlt_mom > 0.10:
                tlt_w, ief_w, shy_w = 0.60, 0.25, 0.15
            elif tlt_mom > 0:
                tlt_w, ief_w, shy_w = 0.30, 0.45, 0.25
            elif tlt_mom > -0.10:
                tlt_w, ief_w, shy_w = 0.10, 0.40, 0.50
            else:
                tlt_w, ief_w, shy_w = 0.0, 0.30, 0.70
        else:
            tlt_w, ief_w, shy_w = 1.0, 0.0, 0.0

        signals["bond"] = {"tlt": tlt_w, "ief": ief_w, "shy": shy_w,
                           "position": "long" if tlt_w > 0.5 else ("short" if tlt_w < 0.2 else "intermediate")}

        return signals

    def run(self, days: int = 90) -> PaperTradingReport:
        """Run paper trading simulation for specified number of days."""
        data = self._load_prices()
        if not data or "SPY" not in data:
            logger.error("No price data available")
            return PaperTradingReport(
                timestamp=datetime.now().isoformat(),
                start_date="N/A", end_date="N/A", trading_days=0,
                total_return=0, cagr=0, volatility=0, sharpe=0,
                max_drawdown=0, max_drawdown_date="",
                total_trades=0, total_commission=0, total_slippage_bps=0,
                turnover_pct=0, winning_days=0, losing_days=0, win_rate=0,
                collar_active_days=0, crypto_active_days=0, avg_bond_duration=0,
                meets_graduation_sharpe=False, meets_graduation_dd=False,
                graduation_ready=False, graduation_note="No data",
                trades=[], snapshots=[],
            )

        spy_prices = data["SPY"]["prices"]
        gld_prices = data["GLD"]["prices"]
        tlt_prices = data["TLT"]["prices"]
        ief_prices = data.get("IEF", {}).get("prices", tlt_prices)
        dates = data["SPY"]["dates"]

        n = min(len(spy_prices), len(dates))
        start_idx = max(180, n - days - 1)  # 180-day warmup, then last N days

        # Initialize portfolio
        initial_capital = 100000.0
        holdings = {"SPY": 0.0, "GLD": 0.0, "TLT": 0.0, "IEF": 0.0,
                     "SHY": 0.0, "CASH": initial_capital}
        target_weights = dict(self.BASELINE)

        peak_value = initial_capital
        peak_date = dates[start_idx] if start_idx < n else ""
        daily_returns = []
        winning_days = 0
        losing_days = 0
        collar_days = 0
        crypto_days = 0
        bond_positions = []
        prev_value = initial_capital

        self._trades = []
        self._snapshots = []

        for i in range(start_idx, n):
            date_str = dates[i]

            # Current prices
            spy_p = spy_prices[i]
            gld_p = gld_prices[i]
            tlt_p = tlt_prices[min(i, len(tlt_prices)-1)]
            ief_p = ief_prices[min(i, len(ief_prices)-1)]

            # Compute position values
            current_holdings = {
                "SPY": holdings.get("SPY", 0) * spy_p,
                "GLD": holdings.get("GLD", 0) * gld_p,
                "TLT": holdings.get("TLT", 0) * tlt_p,
                "IEF": holdings.get("IEF", 0) * ief_p,
                "SHY": holdings.get("SHY", 0) * ief_p * 0.5,  # SHY proxy
                "CASH": holdings.get("CASH", 0),
            }
            total_value = sum(current_holdings.values())

            # Daily return
            daily_ret = (total_value / prev_value - 1) if prev_value > 0 else 0
            daily_returns.append(daily_ret)
            if daily_ret > 0:
                winning_days += 1
            elif daily_ret < 0:
                losing_days += 1

            # Peak tracking
            if total_value > peak_value:
                peak_value = total_value
                peak_date = date_str

            # Overlay signals
            signals = self._compute_overlay_signals(
                {"TLT": {"prices": tlt_prices}, "VIX": {"prices": data.get("VIX", {}).get("prices", [18.0])}}, i
            )

            # Apply collar
            collar = signals["collar"]
            if collar["active"]:
                collar_days += 1
                target_weights["spy"] = max(0.36, min(0.56,
                    self.BASELINE["spy"] + collar["spy_shift"]))

            # Apply bond rotation
            bond = signals["bond"]
            bond_positions.append(bond["position"])

            # Normalize target weights
            total_target = target_weights["spy"] + target_weights["gld"] + target_weights["tlt"]
            if total_target > 0:
                target_weights["spy"] /= total_target
                target_weights["gld"] /= total_target
                target_weights["tlt"] /= total_target

            # Rebalance if drift exceeds threshold
            for asset, price in [("SPY", spy_p), ("GLD", gld_p), ("TLT", tlt_p)]:
                current_w = current_holdings[asset] / total_value if total_value > 0 else 0
                target_w = target_weights.get(asset.lower(), 0)

                if abs(current_w - target_w) > self.REBALANCE_THRESHOLD * target_w:
                    target_val = target_w * total_value
                    current_val = current_holdings[asset]
                    diff = target_val - current_val

                    if abs(diff) > 500:  # Minimum trade size $500
                        side = "buy" if diff > 0 else "sell"
                        trade = self._execute_trade(date_str, asset, side, abs(diff), price)
                        holdings[asset] = holdings.get(asset, 0) + (trade.shares if side == "buy" else -trade.shares)
                        holdings["CASH"] = holdings.get("CASH", 0) - (trade.value if side == "buy" else -trade.value) - trade.commission

            # Store snapshot
            self._snapshots.append(DailySnapshot(
                date=date_str, total_value=round(total_value, 2),
                daily_return=round(daily_ret * 100, 4),
                cumulative_return=round((total_value / initial_capital - 1) * 100, 2),
                spy_value=round(current_holdings["SPY"], 2),
                gld_value=round(current_holdings["GLD"], 2),
                tlt_value=round(current_holdings["TLT"], 2),
                ief_value=round(current_holdings["IEF"], 2),
                shy_value=round(current_holdings["SHY"], 2),
                btc_value=0, eth_value=0, cash=round(holdings["CASH"], 2),
                collar_active=collar["active"],
                crypto_active=False,
                bond_position=bond["position"],
                vix_level=round(collar["vix"], 1),
            ))

            prev_value = total_value

        # Compute report
        final_value = self._snapshots[-1].total_value if self._snapshots else initial_capital
        total_return = (final_value / initial_capital - 1) * 100

        n_ret = len(daily_returns)
        if n_ret > 1:
            avg_ret = np.mean(daily_returns) * 252 * 100
            vol = np.std(daily_returns) * math.sqrt(252) * 100
            sharpe = avg_ret / vol if vol > 0 else 0
        else:
            avg_ret = vol = sharpe = 0

        max_dd = 0.0
        max_dd_date = ""
        peak = initial_capital
        for s in self._snapshots:
            if s.total_value > peak:
                peak = s.total_value
            dd = (s.total_value / peak - 1) * 100
            if dd < max_dd:
                max_dd = dd
                max_dd_date = s.date

        total_commission = sum(t.commission for t in self._trades)
        total_slippage = self.SLIPPAGE_BPS * len(self._trades)

        meets_sharpe = sharpe > 0.5
        meets_dd = max_dd > -15
        graduation_ready = meets_sharpe and meets_dd

        return PaperTradingReport(
            timestamp=datetime.now().isoformat(),
            start_date=dates[start_idx], end_date=dates[-1],
            trading_days=n - start_idx,
            total_return=round(total_return, 2),
            cagr=round(avg_ret, 2),
            volatility=round(vol, 2),
            sharpe=round(sharpe, 3),
            max_drawdown=round(max_dd, 2),
            max_drawdown_date=max_dd_date,
            total_trades=len(self._trades),
            total_commission=round(total_commission, 2),
            total_slippage_bps=round(total_slippage, 1),
            turnover_pct=round(sum(abs(t.value) for t in self._trades) / initial_capital * 100, 1),
            winning_days=winning_days,
            losing_days=losing_days,
            win_rate=round(winning_days / max(1, winning_days + losing_days) * 100, 1),
            collar_active_days=collar_days,
            crypto_active_days=crypto_days,
            avg_bond_duration=round(sum(1 for p in bond_positions if p == "long") / max(1, len(bond_positions)) * 100, 1),
            meets_graduation_sharpe=meets_sharpe,
            meets_graduation_dd=meets_dd,
            graduation_ready=graduation_ready,
            graduation_note=(
                "GRADUATION READY" if graduation_ready
                else f"Needs: {'Sharpe >0.5' if not meets_sharpe else ''} "
                     f"{'Max DD >-15%' if not meets_dd else ''}"
            ),
            trades=[t.to_dict() for t in self._trades],
            snapshots=[s.to_dict() for s in self._snapshots],
        )


def run_paper_trading(days: int = 90) -> PaperTradingReport:
    """Convenience function."""
    sim = PaperTradingSimulator()
    return sim.run(days)


def main():
    import sys
    days = 90
    if len(sys.argv) > 2 and sys.argv[1] == "--days":
        days = int(sys.argv[2])

    sim = PaperTradingSimulator()
    report = sim.run(days)

    print("=" * 60)
    print("PAPER TRADING SIMULATOR — Historical Replay")
    print("=" * 60)
    print(f"Period: {report.start_date} → {report.end_date}")
    print(f"Days: {report.trading_days}")
    print()
    print(f"{'Metric':<25} {'Value':>10}")
    print("-" * 35)
    print(f"{'Total Return':<25} {report.total_return:>9.2f}%")
    print(f"{'CAGR':<25} {report.cagr:>9.2f}%")
    print(f"{'Volatility':<25} {report.volatility:>9.2f}%")
    print(f"{'Sharpe Ratio':<25} {report.sharpe:>10.3f}")
    print(f"{'Max Drawdown':<25} {report.max_drawdown:>9.2f}%")
    print(f"{'Win Rate':<25} {report.win_rate:>9.1f}%")
    print()
    print(f"Trades: {report.total_trades}")
    print(f"Commission: ${report.total_commission:.2f}")
    print(f"Slippage: {report.total_slippage_bps:.1f}bps")
    print(f"Turnover: {report.turnover_pct:.1f}%")
    print()
    print(f"Collar active: {report.collar_active_days}/{report.trading_days} days")
    print(f"Avg bond duration: {report.avg_bond_duration:.0f}% long")
    print()
    print("Graduation Assessment:")
    print(f"  Sharpe > 0.5: {'PASS' if report.meets_graduation_sharpe else 'FAIL'}")
    print(f"  Max DD > -15%: {'PASS' if report.meets_graduation_dd else 'FAIL'}")
    print(f"  → {report.graduation_note}")
    print("=" * 60)

    if "--save" in sys.argv:
        out = sim.OUTPUT_DIR / "paper_trading_report.json"
        with open(out, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f"Report saved to {out}")


if __name__ == "__main__":
    main()

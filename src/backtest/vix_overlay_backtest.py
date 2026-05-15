"""
VIX Term Structure Overlay Backtest - v4.50 Phase 4 Implementation
Walk-forward backtest validation for VIX term structure tactical overlay.

Target: Validate +0.03 to +0.04 Sharpe improvement through drawdown avoidance.
Period: 2010-2026 (16+ years including multiple crisis periods)
"""

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for VIX overlay backtest."""
    start_date: str = "2010-01-01"
    end_date: str = "2026-05-15"
    initial_capital: float = 100000.0
    
    # Baseline allocation (46/38/16)
    base_spy_weight: float = 0.46
    base_gld_weight: float = 0.38
    base_tlt_weight: float = 0.16
    
    # Rebalancing
    rebalance_frequency: str = "monthly"  # monthly, weekly, daily
    transaction_cost_bps: float = 10.0  # 10 bps per trade
    
    # Overlay constraints
    max_daily_shift: float = 0.05  # 5% max daily shift
    min_holding_days: int = 5
    vix_spike_threshold: float = 0.50  # 50% single-day spike


@dataclass
class DailyReturn:
    """Single day return data."""
    date: str
    spy_return: float
    gld_return: float
    tlt_return: float
    vix_spot: Optional[float] = None
    vix3m: Optional[float] = None
    vix6m: Optional[float] = None


@dataclass
class BacktestResult:
    """Complete backtest results."""
    # Basic metrics
    total_return: float
    cagr: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    
    # Overlay-specific
    overlay_active_days: int
    baseline_sharpe: float
    sharpe_improvement: float
    
    # Crisis performance
    return_2008: Optional[float]
    return_2020: Optional[float]
    return_2022: Optional[float]
    
    # Trade stats
    total_rebalances: int
    avg_rebalance_size: float
    total_transaction_costs: float
    
    # Regime breakdown
    regime_returns: Dict[str, float]
    
    # Full equity curve
    equity_curve: List[Dict]


class VIXOverlayBacktester:
    """
    Walk-forward backtest for VIX term structure overlay.
    
    Simulates the overlay using historical VIX data and asset returns
    to validate expected Sharpe improvement.
    """
    
    # VIX regime thresholds (VIX3M/VIX ratio)
    EXTREME_CONTANGO_THRESHOLD = 1.15  # VIX3M/VIX > 1.15
    CONTANGO_THRESHOLD = 1.05  # VIX3M/VIX > 1.05
    FLAT_THRESHOLD = 0.95  # VIX3M/VIX around 1.0
    BACKWARDATION_THRESHOLD = 0.85  # VIX3M/VIX < 0.85
    
    def __init__(self, config: BacktestConfig = None):
        self.config = config or BacktestConfig()
        self.data: List[DailyReturn] = []
        
    def load_data(self, data_path: Optional[str] = None) -> bool:
        """
        Load historical price and VIX data.
        
        Attempts to load from:
        1. Provided data_path
        2. public/data/prices.json
        3. data/vix_term_structure.json
        """
        try:
            # Try to load from prices.json
            prices_path = Path("/root/projects/portfolio-lab/public/data/prices.json")
            if not prices_path.exists():
                prices_path = Path("public/data/prices.json")
            
            if prices_path.exists():
                with open(prices_path) as f:
                    prices_data = json.load(f)
                
                # Extract daily returns from price data
                self._process_price_data(prices_data)
                logger.info(f"Loaded {len(self.data)} days of price data")
                return True
            
            logger.error("No price data found")
            return False
            
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            return False
    
    def _process_price_data(self, prices_data: Dict):
        """Process raw price data into daily returns."""
        # Get date range from SPY data (list format)
        spy_data = prices_data.get("SPY", [])
        gld_data = prices_data.get("GLD", [])
        tlt_data = prices_data.get("TLT", [])
        
        if not spy_data:
            logger.error("No SPY data found")
            return
        
        # Build date-indexed data
        dates = [p["d"] for p in spy_data]
        
        # Create price lookup dictionaries
        spy_prices = {p["d"]: p["p"] for p in spy_data}
        gld_prices = {p["d"]: p["p"] for p in gld_data}
        tlt_prices = {p["d"]: p["p"] for p in tlt_data}
        
        # Calculate daily returns
        for i, date in enumerate(dates[1:], 1):
            prev_date = dates[i-1]
            
            spy_prev = spy_prices.get(prev_date)
            spy_curr = spy_prices.get(date)
            gld_prev = gld_prices.get(prev_date)
            gld_curr = gld_prices.get(date)
            tlt_prev = tlt_prices.get(prev_date)
            tlt_curr = tlt_prices.get(date)
            
            if all([spy_prev, spy_curr, gld_prev, gld_curr, tlt_prev, tlt_curr]):
                spy_ret = (spy_curr - spy_prev) / spy_prev
                gld_ret = (gld_curr - gld_prev) / gld_prev
                tlt_ret = (tlt_curr - tlt_prev) / tlt_prev
                
                self.data.append(DailyReturn(
                    date=date,
                    spy_return=spy_ret,
                    gld_return=gld_ret,
                    tlt_return=tlt_ret
                ))
    
    def calculate_vix_signal(self, vix_spot: float, vix3m: float) -> Tuple[str, float]:
        """
        Calculate VIX term structure signal.
        
        Returns:
            (regime, signal_value) where signal_value is -1.0 to +1.0
        """
        if vix_spot <= 0 or vix3m <= 0:
            return "unknown", 0.0
        
        ratio = vix3m / vix_spot
        
        # Regime classification
        if ratio > self.EXTREME_CONTANGO_THRESHOLD:
            regime = "extreme_contango"
            # Normalize to 0 to +1 range
            signal = min(1.0, (ratio - 1.15) / 0.3 + 0.5)
        elif ratio > self.CONTANGO_THRESHOLD:
            regime = "contango"
            signal = (ratio - 1.05) / 0.1 * 0.5
        elif ratio > self.FLAT_THRESHOLD:
            regime = "flat"
            signal = (ratio - 0.95) / 0.1 * 0.2 - 0.1
        elif ratio > self.BACKWARDATION_THRESHOLD:
            regime = "backwardation"
            signal = (ratio - 0.85) / 0.1 * -0.5 - 0.2
        else:
            regime = "extreme_backwardation"
            signal = max(-1.0, (ratio - 0.85) / 0.3 * -0.5 - 0.5)
        
        return regime, signal
    
    def get_allocation_shifts(self, signal_value: float) -> Tuple[float, float, float]:
        """
        Get recommended allocation shifts based on signal.
        
        Returns:
            (spy_shift, gld_shift, tlt_shift) in percentage points
        """
        # Signal to allocation mapping
        # +1.0 (extreme contango): SPY +10%, GLD -5%, TLT -5%
        # -1.0 (extreme backwardation): SPY -10%, GLD +5%, TLT +5%
        
        spy_shift = signal_value * 0.10  # ±10% for SPY
        gld_shift = -signal_value * 0.05  # ∓5% for GLD
        tlt_shift = -signal_value * 0.05  # ∓5% for TLT
        
        return spy_shift, gld_shift, tlt_shift
    
    def run_backtest(self) -> Optional[BacktestResult]:
        """
        Run walk-forward backtest with VIX overlay.
        
        Simulates both baseline (46/38/16) and overlay-enhanced portfolios.
        """
        if not self.data:
            logger.error("No data loaded")
            return None
        
        # Filter to backtest period
        start_dt = datetime.strptime(self.config.start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(self.config.end_date, "%Y-%m-%d")
        
        backtest_data = [
            d for d in self.data
            if start_dt <= datetime.strptime(d.date, "%Y-%m-%d") <= end_dt
        ]
        
        if not backtest_data:
            logger.error("No data in backtest period")
            return None
        
        logger.info(f"Running backtest on {len(backtest_data)} days")
        
        # Initialize portfolios
        base_capital = self.config.initial_capital
        overlay_capital = self.config.initial_capital
        
        base_weights = {
            "spy": self.config.base_spy_weight,
            "gld": self.config.base_gld_weight,
            "tlt": self.config.base_tlt_weight
        }
        
        overlay_weights = dict(base_weights)
        
        # Tracking variables
        base_equity = [base_capital]
        overlay_equity = [overlay_capital]
        overlay_active_days = 0
        total_rebalances = 0
        rebalance_sizes = []
        total_costs = 0.0
        
        holding_days = 0
        last_signal = 0.0
        
        # Regime tracking
        regime_returns = {
            "extreme_contango": [],
            "contango": [],
            "flat": [],
            "backwardation": [],
            "extreme_backwardation": []
        }
        
        # Crisis tracking
        returns_2008 = []
        returns_2020 = []
        returns_2022 = []
        
        for day in backtest_data:
            day_date = datetime.strptime(day.date, "%Y-%m-%d")
            year = day_date.year
            
            # Calculate baseline return
            base_ret = (
                base_weights["spy"] * day.spy_return +
                base_weights["gld"] * day.gld_return +
                base_weights["tlt"] * day.tlt_return
            )
            base_capital *= (1 + base_ret)
            base_equity.append(base_capital)
            
            # Calculate VIX signal (simulate with heuristics since we don't have historical VIX3M)
            # In real implementation, this would use actual VIX3M data
            regime, signal = self._simulate_vix_signal(day, day_date)
            
            # Check if we should apply overlay
            apply_overlay = True
            
            # Minimum holding period
            if holding_days < self.config.min_holding_days and abs(signal - last_signal) < 0.3:
                apply_overlay = False
            
            # Check for VIX spike (simulated)
            if self._is_vix_spike(day, backtest_data):
                apply_overlay = False
            
            if apply_overlay and abs(signal) > 0.2:
                overlay_active_days += 1
                
                # Get target shifts
                spy_shift, gld_shift, tlt_shift = self.get_allocation_shifts(signal)
                
                # Apply max daily shift constraint
                current_spy = overlay_weights["spy"]
                target_spy = self.config.base_spy_weight + spy_shift
                allowed_shift = np.sign(target_spy - current_spy) * min(
                    abs(target_spy - current_spy),
                    self.config.max_daily_shift
                )
                new_spy = current_spy + allowed_shift
                
                # Calculate other weights maintaining relative ratios
                total_shift = new_spy - self.config.base_spy_weight
                new_gld = self.config.base_gld_weight - total_shift * 0.5
                new_tlt = self.config.base_tlt_weight - total_shift * 0.5
                
                # Calculate turnover and costs
                turnover = abs(new_spy - overlay_weights["spy"])
                if turnover > 0.001:  # 0.1% threshold
                    total_rebalances += 1
                    rebalance_sizes.append(turnover)
                    cost = turnover * self.config.transaction_cost_bps / 10000 * overlay_capital
                    total_costs += cost
                    overlay_capital -= cost
                
                overlay_weights["spy"] = new_spy
                overlay_weights["gld"] = new_gld
                overlay_weights["tlt"] = new_tlt
                
                holding_days = 0
                last_signal = signal
            else:
                holding_days += 1
            
            # Normalize weights to sum to 1
            total_weight = sum(overlay_weights.values())
            if abs(total_weight - 1.0) > 0.001:
                for k in overlay_weights:
                    overlay_weights[k] /= total_weight
            
            # Calculate overlay return
            overlay_ret = (
                overlay_weights["spy"] * day.spy_return +
                overlay_weights["gld"] * day.gld_return +
                overlay_weights["tlt"] * day.tlt_return
            )
            overlay_capital *= (1 + overlay_ret)
            overlay_equity.append(overlay_capital)
            
            # Track regime returns
            if regime in regime_returns:
                regime_returns[regime].append(overlay_ret)
            
            # Track crisis periods
            if year == 2008:
                returns_2008.append(overlay_ret)
            elif year == 2020 and day_date.month >= 2:
                returns_2020.append(overlay_ret)
            elif year == 2022:
                returns_2022.append(overlay_ret)
        
        # Calculate metrics
        base_returns = self._calculate_returns_from_equity(base_equity)
        overlay_returns = self._calculate_returns_from_equity(overlay_equity)
        
        base_metrics = self._calculate_metrics(base_returns)
        overlay_metrics = self._calculate_metrics(overlay_returns)
        
        # Average regime returns
        avg_regime_returns = {}
        for regime, rets in regime_returns.items():
            if rets:
                avg_regime_returns[regime] = np.mean(rets) * 252  # Annualized
        
        result = BacktestResult(
            total_return=(overlay_equity[-1] / self.config.initial_capital - 1) * 100,
            cagr=overlay_metrics["cagr"],
            volatility=overlay_metrics["volatility"],
            sharpe_ratio=overlay_metrics["sharpe"],
            max_drawdown=overlay_metrics["max_dd"],
            overlay_active_days=overlay_active_days,
            baseline_sharpe=base_metrics["sharpe"],
            sharpe_improvement=overlay_metrics["sharpe"] - base_metrics["sharpe"],
            return_2008=self._annualize_returns(returns_2008) if returns_2008 else None,
            return_2020=self._annualize_returns(returns_2020) if returns_2020 else None,
            return_2022=self._annualize_returns(returns_2022) if returns_2022 else None,
            total_rebalances=total_rebalances,
            avg_rebalance_size=np.mean(rebalance_sizes) if rebalance_sizes else 0,
            total_transaction_costs=total_costs,
            regime_returns=avg_regime_returns,
            equity_curve=[
                {
                    "date": backtest_data[min(i, len(backtest_data)-1)].date if i > 0 else backtest_data[0].date,
                    "baseline": base_equity[i],
                    "overlay": overlay_equity[i]
                }
                for i in range(0, len(overlay_equity), max(1, len(overlay_equity)//252))  # Sample ~252 points
            ]
        )
        
        return result
    
    def _simulate_vix_signal(self, day: DailyReturn, day_date: datetime) -> Tuple[str, float]:
        """
        Simulate VIX signal from asset returns.
        
        In production, this would use actual VIX3M/VIX data.
        For backtest, we infer from volatility patterns.
        """
        # Use TLT return as a proxy for bond volatility
        # Use SPY return magnitude for equity volatility
        
        spy_vol_proxy = abs(day.spy_return) * 100  # Approx daily vol in %
        
        # Simulate VIX level based on recent volatility
        if spy_vol_proxy > 2.0:  # High vol day
            vix_spot = 25.0 + spy_vol_proxy * 5
        elif spy_vol_proxy > 1.0:
            vix_spot = 18.0 + spy_vol_proxy * 7
        else:
            vix_spot = 14.0 + spy_vol_proxy * 4
        
        # Simulate VIX3M (typically smoother than spot)
        vix3m = vix_spot * (0.9 + np.random.random() * 0.2)  # 0.9 to 1.1 ratio
        
        return self.calculate_vix_signal(vix_spot, vix3m)
    
    def _is_vix_spike(self, day: DailyReturn, all_data: List[DailyReturn]) -> bool:
        """Check if this day had a VIX spike (>50% single-day increase)."""
        day_idx = all_data.index(day) if day in all_data else -1
        if day_idx < 1:
            return False
        
        # Simulate VIX spike detection from return patterns
        prev_day = all_data[day_idx - 1]
        spy_drop = day.spy_return < -0.03  # 3% SPY drop
        vol_increase = abs(day.spy_return) > abs(prev_day.spy_return) * 2
        
        return spy_drop and vol_increase
    
    def _calculate_returns_from_equity(self, equity: List[float]) -> List[float]:
        """Calculate daily returns from equity curve."""
        returns = []
        for i in range(1, len(equity)):
            ret = (equity[i] - equity[i-1]) / equity[i-1]
            returns.append(ret)
        return returns
    
    def _calculate_metrics(self, returns: List[float]) -> Dict:
        """Calculate performance metrics from returns."""
        if not returns:
            return {"cagr": 0, "volatility": 0, "sharpe": 0, "max_dd": 0}
        
        returns_array = np.array(returns)
        
        # Annualized return (assuming 252 trading days)
        total_return = np.prod(1 + returns_array) - 1
        n_years = len(returns) / 252
        cagr = (1 + total_return) ** (1/n_years) - 1 if n_years > 0 else 0
        
        # Annualized volatility
        volatility = np.std(returns_array) * np.sqrt(252) * 100
        
        # Sharpe ratio (assuming 0% risk-free rate for simplicity)
        sharpe = (cagr * 100) / volatility if volatility > 0 else 0
        
        # Max drawdown
        equity = [1.0]
        for ret in returns:
            equity.append(equity[-1] * (1 + ret))
        
        peak = equity[0]
        max_dd = 0
        for val in equity:
            if val > peak:
                peak = val
            dd = (peak - val) / peak
            max_dd = max(max_dd, dd)
        
        return {
            "cagr": cagr * 100,
            "volatility": volatility,
            "sharpe": sharpe,
            "max_dd": -max_dd * 100
        }
    
    def _annualize_returns(self, returns: List[float]) -> float:
        """Annualize a list of daily returns."""
        if not returns:
            return 0.0
        total = np.prod([1 + r for r in returns]) - 1
        n_years = len(returns) / 252
        return ((1 + total) ** (1/n_years) - 1) * 100 if n_years > 0 else 0.0
    
    def save_results(self, result: BacktestResult, output_path: str = None):
        """Save backtest results to JSON."""
        if output_path is None:
            output_path = "/root/projects/portfolio-lab/data/backtest_results/vix_overlay_backtest.json"
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(asdict(result), f, indent=2, default=str)
        
        logger.info(f"Results saved to {output_path}")
    
    def print_report(self, result: BacktestResult):
        """Print formatted backtest report."""
        print("\n" + "="*60)
        print("VIX TERM STRUCTURE OVERLAY BACKTEST RESULTS")
        print("="*60)
        print(f"Period: {self.config.start_date} to {self.config.end_date}")
        print(f"Initial Capital: ${self.config.initial_capital:,.2f}")
        print()
        
        print("PERFORMANCE METRICS")
        print("-"*60)
        print(f"Total Return:        {result.total_return:>8.2f}%")
        print(f"CAGR:                {result.cagr:>8.2f}%")
        print(f"Volatility:          {result.volatility:>8.2f}%")
        print(f"Sharpe Ratio:        {result.sharpe_ratio:>8.3f}")
        print(f"Max Drawdown:        {result.max_drawdown:>8.2f}%")
        print()
        
        print("OVERLAY IMPACT")
        print("-"*60)
        print(f"Baseline Sharpe:     {result.baseline_sharpe:>8.3f}")
        print(f"Overlay Sharpe:      {result.sharpe_ratio:>8.3f}")
        print(f"Improvement:         {result.sharpe_improvement:>+8.3f}  {'✓' if result.sharpe_improvement > 0.02 else '✗'}")
        print(f"Overlay Active Days: {result.overlay_active_days:>8} ({result.overlay_active_days/len(self.data)*100:.1f}%)")
        print()
        
        print("CRISIS PERFORMANCE")
        print("-"*60)
        if result.return_2008:
            print(f"2008 GFC:            {result.return_2008:>8.2f}%")
        if result.return_2020:
            print(f"2020 COVID:          {result.return_2020:>8.2f}%")
        if result.return_2022:
            print(f"2022 Rate Hikes:     {result.return_2022:>8.2f}%")
        print()
        
        print("TRADE STATISTICS")
        print("-"*60)
        print(f"Total Rebalances:    {result.total_rebalances:>8}")
        print(f"Avg Rebalance Size:  {result.avg_rebalance_size*100:>8.2f}%")
        print(f"Transaction Costs:   ${result.total_transaction_costs:>8.2f}")
        print()
        
        print("REGIME PERFORMANCE (Annualized)")
        print("-"*60)
        for regime, ret in sorted(result.regime_returns.items()):
            print(f"{regime:20s} {ret:>8.2f}%")
        
        print("="*60)
        
        # Success criteria check
        print("\nSUCCESS CRITERIA")
        print("-"*60)
        checks = [
            ("Sharpe > Baseline + 0.03", result.sharpe_improvement >= 0.03),
            ("Max DD < -25%", result.max_drawdown > -25),
            ("Rebalances < 50/yr", result.total_rebalances < 800),  # 16 years * 50
        ]
        for desc, passed in checks:
            print(f"{'✓' if passed else '✗'} {desc}")
        print("="*60)


def main():
    """Run VIX overlay backtest."""
    backtester = VIXOverlayBacktester()
    
    if not backtester.load_data():
        logger.error("Failed to load data")
        return 1
    
    result = backtester.run_backtest()
    if not result:
        logger.error("Backtest failed")
        return 1
    
    backtester.print_report(result)
    backtester.save_results(result)
    
    return 0


if __name__ == "__main__":
    exit(main())

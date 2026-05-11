"""
Performance analytics calculations: drawdown, rolling metrics, benchmarks.
"""
import json
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path


@dataclass
class DrawdownPoint:
    date: str
    value: float
    peak: float
    drawdown: float  # Negative value (e.g., -0.15 = -15%)
    drawdown_pct: float
    days_since_peak: int
    is_recovery: bool


@dataclass
class RollingMetrics:
    date: str
    sharpe_63d: Optional[float]
    sharpe_126d: Optional[float]
    sharpe_252d: Optional[float]
    volatility_63d: Optional[float]
    returns_63d: Optional[float]


@dataclass
class BenchmarkSeries:
    symbol: str
    dates: List[str]
    values: List[float]  # Normalized to 100
    cagr: float
    volatility: float
    max_drawdown: float


@dataclass
class CrisisPeriod:
    name: str
    start_date: str
    end_date: str
    description: str
    spy_return: float
    portfolio_return: Optional[float]


class AnalyticsCalculator:
    """
    Calculates performance analytics from portfolio and price data.
    
    Generates drawdown series, rolling metrics, and benchmark comparisons
    for dashboard visualization.
    """
    
    # Crisis periods for reference
    CRISIS_PERIODS = [
        CrisisPeriod(
            name="GFC 2008",
            start_date="2008-09-01",
            end_date="2009-03-31",
            description="Global Financial Crisis",
            spy_return=-0.47,
            portfolio_return=None,  # Would be calculated
        ),
        CrisisPeriod(
            name="COVID 2020",
            start_date="2020-02-19",
            end_date="2020-03-23",
            description="COVID-19 Market Crash",
            spy_return=-0.34,
            portfolio_return=None,
        ),
        CrisisPeriod(
            name="Rate Hikes 2022",
            start_date="2022-01-01",
            end_date="2022-10-12",
            description="Fed Rate Hike Cycle",
            spy_return=-0.25,
            portfolio_return=None,
        ),
    ]
    
    def __init__(self, data_dir: str = "~/projects/portfolio-lab/data"):
        self.data_dir = Path(data_dir).expanduser()
        self.performance_file = self.data_dir / "performance.jsonl"
    
    def load_performance_data(self) -> List[Dict]:
        """Load paper portfolio performance history."""
        if not self.performance_file.exists():
            return []
        
        data = []
        with open(self.performance_file, 'r') as f:
            for line in f:
                try:
                    data.append(json.loads(line))
                except:
                    continue
        return data
    
    def calculate_drawdown_series(
        self,
        performance_data: Optional[List[Dict]] = None
    ) -> List[DrawdownPoint]:
        """
        Calculate underwater curve (drawdown series).
        
        Returns list of drawdown points showing peak-to-trough decline.
        """
        if performance_data is None:
            performance_data = self.load_performance_data()
        
        if not performance_data:
            return []
        
        # Sort by date
        sorted_data = sorted(
            performance_data,
            key=lambda x: x.get('timestamp', x.get('date', ''))
        )
        
        drawdowns = []
        peak_value = sorted_data[0].get('total_value', 100000)
        peak_date = sorted_data[0].get('timestamp', sorted_data[0].get('date', ''))[:10]
        days_since_peak = 0
        
        for i, point in enumerate(sorted_data):
            value = point.get('total_value', 0)
            date = point.get('timestamp', point.get('date', ''))[:10]
            
            # Update peak if new high
            if value > peak_value:
                peak_value = value
                peak_date = date
                days_since_peak = 0
            else:
                days_since_peak += 1
            
            # Calculate drawdown
            if peak_value > 0:
                dd = (value - peak_value) / peak_value
                dd_pct = dd * 100
            else:
                dd = 0
                dd_pct = 0
            
            # Recovery if within 1% of peak
            is_recovery = value >= peak_value * 0.99
            
            drawdowns.append(DrawdownPoint(
                date=date,
                value=value,
                peak=peak_value,
                drawdown=dd,
                drawdown_pct=round(dd_pct, 2),
                days_since_peak=days_since_peak,
                is_recovery=is_recovery,
            ))
        
        return drawdowns
    
    def calculate_max_drawdown(self, drawdown_series: List[DrawdownPoint]) -> Dict:
        """Calculate maximum drawdown statistics."""
        if not drawdown_series:
            return {"max_drawdown": 0, "max_drawdown_date": None, "recovery_date": None}
        
        # Find maximum drawdown
        max_dd_point = min(drawdown_series, key=lambda x: x.drawdown)
        max_dd = max_dd_point.drawdown
        max_dd_date = max_dd_point.date
        
        # Find recovery date (first date after max DD where is_recovery is True)
        recovery_date = None
        found_max = False
        for point in drawdown_series:
            if point.date == max_dd_date:
                found_max = True
            if found_max and point.is_recovery:
                recovery_date = point.date
                break
        
        # Calculate underwater duration
        if recovery_date:
            underwater_days = (
                datetime.strptime(recovery_date, "%Y-%m-%d") - 
                datetime.strptime(max_dd_date, "%Y-%m-%d")
            ).days
        else:
            # Still underwater
            underwater_days = (
                datetime.now() - datetime.strptime(max_dd_date, "%Y-%m-%d")
            ).days
        
        return {
            "max_drawdown": round(max_dd * 100, 2),  # Percentage
            "max_drawdown_date": max_dd_date,
            "recovery_date": recovery_date,
            "underwater_days": underwater_days,
            "peak_value": max_dd_point.peak,
            "trough_value": max_dd_point.value,
        }
    
    def calculate_rolling_sharpe(
        self,
        window_days: int = 63,
        performance_data: Optional[List[Dict]] = None
    ) -> List[Dict]:
        """
        Calculate rolling Sharpe ratio over performance history.
        
        Args:
            window_days: Rolling window in trading days (63, 126, 252)
        """
        if performance_data is None:
            performance_data = self.load_performance_data()
        
        if len(performance_data) < window_days + 1:
            return []
        
        # Sort by date
        sorted_data = sorted(
            performance_data,
            key=lambda x: x.get('timestamp', x.get('date', ''))
        )
        
        rolling_metrics = []
        
        for i in range(window_days, len(sorted_data)):
            window = sorted_data[i - window_days:i]
            
            # Calculate daily returns
            returns = []
            for j in range(1, len(window)):
                prev_value = window[j-1].get('total_value', 0)
                curr_value = window[j].get('total_value', 0)
                if prev_value > 0:
                    ret = (curr_value - prev_value) / prev_value
                    returns.append(ret)
            
            if len(returns) < 2:
                continue
            
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            
            if std_return > 0:
                # Annualized Sharpe (assuming 252 trading days)
                sharpe = (mean_return / std_return) * np.sqrt(252)
            else:
                sharpe = 0
            
            date = sorted_data[i].get('timestamp', sorted_data[i].get('date', ''))[:10]
            
            rolling_metrics.append({
                "date": date,
                "sharpe": round(sharpe, 2),
                "volatility": round(std_return * np.sqrt(252) * 100, 2),
                "mean_return": round(mean_return * 100, 3),
                "window_days": window_days,
            })
        
        return rolling_metrics
    
    def calculate_all_rolling_metrics(
        self,
        performance_data: Optional[List[Dict]] = None
    ) -> Dict[str, List[Dict]]:
        """Calculate rolling metrics for all windows."""
        windows = [63, 126, 252]  # 3 months, 6 months, 1 year
        
        return {
            f"sharpe_{w}d": self.calculate_rolling_sharpe(w, performance_data)
            for w in windows
        }
    
    def calculate_benchmark_comparison(
        self,
        performance_data: Optional[List[Dict]] = None,
        db_path: str = "~/projects/portfolio-lab/data/market.db"
    ) -> Dict[str, Dict]:
        """
        Compare portfolio performance to benchmarks.
        
        Benchmarks: SPY (S&P 500), 60/40, All Weather
        """
        if performance_data is None:
            performance_data = self.load_performance_data()
        
        if not performance_data:
            return {}
        
        # Get date range
        sorted_data = sorted(
            performance_data,
            key=lambda x: x.get('timestamp', x.get('date', ''))
        )
        
        start_date = sorted_data[0].get('timestamp', sorted_data[0].get('date', ''))[:10]
        end_date = sorted_data[-1].get('timestamp', sorted_data[-1].get('date', ''))[:10]
        
        start_value = sorted_data[0].get('total_value', 100000)
        end_value = sorted_data[-1].get('total_value', 100000)
        
        portfolio_return = (end_value - start_value) / start_value if start_value > 0 else 0
        
        # Calculate daily returns for volatility
        returns = []
        for i in range(1, len(sorted_data)):
            prev = sorted_data[i-1].get('total_value', 0)
            curr = sorted_data[i].get('total_value', 0)
            if prev > 0:
                returns.append((curr - prev) / prev)
        
        portfolio_vol = np.std(returns) * np.sqrt(252) * 100 if returns else 0
        
        # Calculate max drawdown
        dd_series = self.calculate_drawdown_series(performance_data)
        max_dd = self.calculate_max_drawdown(dd_series)
        
        result = {
            "portfolio": {
                "start_date": start_date,
                "end_date": end_date,
                "start_value": start_value,
                "end_value": end_value,
                "total_return": round(portfolio_return * 100, 2),
                "cagr": None,  # Would need to calculate based on actual days
                "volatility": round(portfolio_vol, 2),
                "max_drawdown": max_dd.get("max_drawdown", 0),
                "sharpe": None,  # Full period Sharpe
            }
        }
        
        return result
    
    def generate_analytics_report(self) -> Dict:
        """Generate complete analytics report."""
        perf_data = self.load_performance_data()
        
        if not perf_data:
            return {
                "status": "no_data",
                "message": "No performance data available",
                "generated_at": datetime.now().isoformat(),
            }
        
        # Calculate all metrics
        drawdown_series = self.calculate_drawdown_series(perf_data)
        max_dd_stats = self.calculate_max_drawdown(drawdown_series)
        rolling_metrics = self.calculate_all_rolling_metrics(perf_data)
        benchmark_comparison = self.calculate_benchmark_comparison(perf_data)
        
        # Convert drawdown series to dict format
        drawdown_dicts = [
            {
                "date": d.date,
                "value": round(d.value, 2),
                "peak": round(d.peak, 2),
                "drawdown": round(d.drawdown * 100, 2),  # As percentage
                "days_since_peak": d.days_since_peak,
                "is_recovery": d.is_recovery,
            }
            for d in drawdown_series
        ]
        
        # Crisis period performance (would need historical backtest data)
        crisis_summary = []
        for crisis in self.CRISIS_PERIODS:
            crisis_summary.append({
                "name": crisis.name,
                "period": f"{crisis.start_date} to {crisis.end_date}",
                "description": crisis.description,
                "spy_return": round(crisis.spy_return * 100, 1),
                "portfolio_return": None,  # Would require historical simulation
            })
        
        return {
            "status": "success",
            "generated_at": datetime.now().isoformat(),
            "data_points": len(perf_data),
            "date_range": {
                "start": perf_data[0].get('timestamp', '')[:10] if perf_data else None,
                "end": perf_data[-1].get('timestamp', '')[:10] if perf_data else None,
            },
            "drawdown": {
                "series": drawdown_dicts,
                "max_drawdown": max_dd_stats,
            },
            "rolling_metrics": rolling_metrics,
            "benchmark_comparison": benchmark_comparison,
            "crisis_periods": crisis_summary,
        }


def main():
    """CLI for analytics calculator."""
    import sys
    
    calc = AnalyticsCalculator()
    
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        
        if cmd == "drawdown":
            series = calc.calculate_drawdown_series()
            stats = calc.calculate_max_drawdown(series)
            print(json.dumps(stats, indent=2))
            
        elif cmd == "rolling":
            metrics = calc.calculate_all_rolling_metrics()
            for window, data in metrics.items():
                if data:
                    latest = data[-1]
                    print(f"{window}: Sharpe={latest['sharpe']}, Vol={latest['volatility']}%")
                    
        elif cmd == "report":
            report = calc.generate_analytics_report()
            print(json.dumps(report, indent=2, default=str))
            
        else:
            print(f"Unknown command: {cmd}")
            print("Commands: drawdown, rolling, report")
    else:
        # Default: full report
        report = calc.generate_analytics_report()
        print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()

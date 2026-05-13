#!/usr/bin/env python3
"""
Portfolio-Lab v2.54: Fed Policy Overlay

Real-time Federal Reserve policy regime detection and tactical allocation
based on Fed Funds rate, inflation, and real yield data from FRED.

Sources:
- Federal Reserve Economic Data (FRED) - St. Louis Fed
- Goldman Sachs Research: "Asset Allocation in a Higher-Rate World" (2024)
- AQR: "2024 Capital Market Assumptions"

Regime Classification:
    EASING:     Real rates negative, Fed cutting or on hold after cuts
    TIGHTENING: Fed actively hiking, positive real rates rising
    NEUTRAL:    Real rates modestly positive, inflation near target
    UNCERTAIN:  Mixed signals, policy path unclear

Tactical Allocation Shifts (from base 46/38/16):
    EASING:     SPY +5%, GLD +5%, TLT -5%, CASH -5%
                (Risk-on, inflation hedge via gold)
    TIGHTENING: SPY -10%, GLD +10%, TLT 0%, CASH 0%
                (Defensive, gold outperforms in rate hikes)
    NEUTRAL:    Base allocation (no shift)
    UNCERTAIN:  SPY -5%, GLD +10%, TLT -5%, CASH 0%
                (Maximum gold, reduced directional exposure)

Real Rate Calculation:
    Real Rate = Nominal Yield - Expected Inflation (TIPS breakeven or CPI YoY)

Usage:
    python -m src.signals.fed_policy_overlay fetch
    python -m src.signals.fed_policy_overlay regime
    python -m src.signals.fed_policy_overlay allocate --portfolio 46/38/16
    python -m src.signals.fed_policy_overlay backtest --start 2005-01-01

Reference:
    FRED API: https://fred.stlouisfed.org/
    Target Sharpe improvement: 0.96 -> 1.05 (combined with TSMOM + HMM)
"""

import numpy as np
import pandas as pd
import json
import argparse
import sys
import requests
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


# FRED Series IDs
FRED_SERIES = {
    'FEDFUNDS': 'Federal Funds Effective Rate',
    'CPIAUCSL': 'Consumer Price Index (All Urban)', 
    'CPILFESL': 'Core CPI (Less Food & Energy)',
    'T10YIE': '10-Year Breakeven Inflation Rate',
    'T5YIE': '5-Year Breakeven Inflation Rate',
    'DFII10': '10-Year TIPS Yield',
    'DFII5': '5-Year TIPS Yield',
    'DGS10': '10-Year Treasury Yield',
    'DGS2': '2-Year Treasury Yield',
    'UNRATE': 'Unemployment Rate',
    'PAYEMS': 'Total Nonfarm Payrolls',
    'INDPRO': 'Industrial Production',
}

# Cache directory
DATA_DIR = Path("~/projects/portfolio-lab/data").expanduser()
FRED_CACHE = DATA_DIR / "fred_data.json"


def fetch_fred_series(series_id: str, start_date: str = "2000-01-01") -> Optional[pd.DataFrame]:
    """
    Fetch FRED series data via direct CSV download.
    
    Args:
        series_id: FRED series identifier
        start_date: Start date for data retrieval (YYYY-MM-DD)
        
    Returns:
        DataFrame with 'date' and 'value' columns, or None on error
    """
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Parse CSV
        from io import StringIO
        df = pd.read_csv(StringIO(response.text))
        
        # Rename columns
        df.columns = ['date', 'value']
        df['date'] = pd.to_datetime(df['date'])
        
        # Filter by start date
        df = df[df['date'] >= start_date]
        
        # Convert value to numeric (handle missing values)
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        df = df.dropna()
        
        return df
        
    except Exception as e:
        print(f"Error fetching {series_id}: {e}")
        return None


def fetch_all_fred_data(cache_path: Path = FRED_CACHE, force_refresh: bool = False) -> Dict[str, pd.DataFrame]:
    """
    Fetch all relevant FRED series and cache results.
    
    Priority series:
        - FEDFUNDS: Current policy rate
        - CPIAUCSL: Inflation gauge
        - T10YIE: Inflation expectations
        - DFII10: Real 10-year yield
        - DGS10: Nominal 10-year yield
    """
    # Check cache
    if not force_refresh and cache_path.exists():
        cache_age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
        if cache_age < timedelta(hours=24):
            with open(cache_path) as f:
                cached = json.load(f)
            print(f"Using cached FRED data (age: {cache_age})")
            return {k: pd.DataFrame(v) for k, v in cached.items()}
    
    # Priority series to fetch
    priority_series = ['FEDFUNDS', 'CPIAUCSL', 'T10YIE', 'DFII10', 'DGS10', 'DGS2']
    
    data = {}
    for series_id in priority_series:
        print(f"Fetching {series_id}...")
        df = fetch_fred_series(series_id)
        if df is not None:
            data[series_id] = df
            print(f"  Got {len(df)} observations, latest: {df.iloc[-1]['date'].strftime('%Y-%m-%d')} = {df.iloc[-1]['value']:.2f}")
        else:
            print(f"  Failed to fetch {series_id}")
    
    # Cache results
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache_data = {k: v.to_dict('records') for k, v in data.items()}
    with open(cache_path, 'w') as f:
        json.dump(cache_data, f, indent=2, default=str)
    
    return data


def calculate_inflation_yoy(cpi_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate year-over-year CPI inflation rate."""
    df = cpi_df.copy()
    df['inflation_yoy'] = df['value'].pct_change(periods=12) * 100
    return df.dropna()


def calculate_real_rate(nominal_df: pd.DataFrame, inflation_df: pd.DataFrame, 
                        merge_how: str = 'outer') -> pd.DataFrame:
    """
    Calculate real interest rate: nominal - inflation.
    
    Handles different frequencies by forward-filling inflation to match nominal.
    """
    # Merge on date
    merged = pd.merge(
        nominal_df[['date', 'value']].rename(columns={'value': 'nominal'}),
        inflation_df[['date', 'inflation_yoy']].rename(columns={'inflation_yoy': 'inflation'}),
        on='date',
        how=merge_how
    )
    
    # Forward fill inflation to match nominal frequency (daily vs monthly)
    merged['inflation'] = merged['inflation'].ffill()
    
    # Calculate real rate
    merged['real_rate'] = merged['nominal'] - merged['inflation']
    
    return merged.dropna()


@dataclass
class FedPolicyRegime:
    """Current Federal Reserve policy regime classification."""
    
    timestamp: str
    regime: str  # EASING, TIGHTENING, NEUTRAL, UNCERTAIN
    
    # Key metrics
    fed_funds_rate: float
    inflation_yoy: float
    real_rate_10y: float
    real_rate_short: float
    breakeven_10y: float
    
    # Additional context
    yield_curve_10y2y: float
    unemployment: Optional[float] = None
    
    # Classification confidence
    confidence: float = 0.0
    regime_factors: Dict[str, float] = None
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    def is_divergence_risk(self) -> bool:
        """Check if there's divergence between short and long real rates."""
        return abs(self.real_rate_short - self.real_rate_10y) > 1.0
    
    def get_allocation_shift(self) -> Dict[str, float]:
        """Get recommended allocation shift from base 46/38/16."""
        shifts = {
            'EASING': {'SPY': +0.05, 'GLD': +0.05, 'TLT': -0.05, 'CASH': -0.05},
            'TIGHTENING': {'SPY': -0.10, 'GLD': +0.10, 'TLT': 0.0, 'CASH': 0.0},
            'NEUTRAL': {'SPY': 0.0, 'GLD': 0.0, 'TLT': 0.0, 'CASH': 0.0},
            'UNCERTAIN': {'SPY': -0.05, 'GLD': +0.10, 'TLT': -0.05, 'CASH': 0.0},
        }
        return shifts.get(self.regime, shifts['NEUTRAL'])


def classify_fed_regime(
    fed_funds: float,
    inflation_yoy: float,
    real_rate_10y: float,
    real_rate_short: Optional[float] = None,
    yield_curve_slope: Optional[float] = None,
    rate_change_6m: float = 0.0
) -> Tuple[str, float, Dict[str, float]]:
    """
    Classify Federal Reserve policy regime based on key indicators.
    
    Classification Rules (simplified from research):
    
    EASING:
        - Real rates negative OR Fed cutting (rate_change_6m < -0.5)
        - Inflation > 2% (some tolerance for overshoot)
        
    TIGHTENING:
        - Real rates > 1.5% AND rising
        - Fed hiking (rate_change_6m > 0.5)
        - Inflation > 2.5% OR strong labor market
        
    NEUTRAL:
        - Real rates 0.5% to 1.5%
        - Inflation 1.5% to 2.5%
        - Yield curve normal (positive slope)
        
    UNCERTAIN:
        - Mixed signals (e.g., high real rates but Fed cutting)
        - Inverted yield curve (recession fear)
        - Policy path unclear
    
    Returns:
        (regime_name, confidence_score, factor_scores)
    """
    factors = {}
    
    # Factor 1: Real rate level
    if real_rate_short is not None:
        real_short = real_rate_short
    else:
        real_short = fed_funds - inflation_yoy
    
    factors['real_rate_level'] = real_short
    
    # Factor 2: Rate trajectory
    factors['rate_change_6m'] = rate_change_6m
    
    # Factor 3: Inflation vs target
    inflation_gap = inflation_yoy - 2.0
    factors['inflation_gap'] = inflation_gap
    
    # Factor 4: Yield curve
    if yield_curve_slope is not None:
        factors['yield_curve'] = yield_curve_slope
    else:
        factors['yield_curve'] = 0.0
    
    # Scoring for each regime
    scores = {'EASING': 0, 'TIGHTENING': 0, 'NEUTRAL': 0, 'UNCERTAIN': 0}
    
    # EASING signals
    if real_short < 0:
        scores['EASING'] += 2
    elif real_short < 0.5:
        scores['EASING'] += 1
        
    if rate_change_6m < -0.25:
        scores['EASING'] += 2
    elif rate_change_6m < 0:
        scores['EASING'] += 1
        
    if inflation_yoy > 3.0 and fed_funds < inflation_yoy:
        scores['EASING'] += 1  # Real rates negative, Fed behind curve
    
    # TIGHTENING signals  
    if real_short > 1.5:
        scores['TIGHTENING'] += 2
    elif real_short > 1.0:
        scores['TIGHTENING'] += 1
        
    if rate_change_6m > 0.5:
        scores['TIGHTENING'] += 2
    elif rate_change_6m > 0.25:
        scores['TIGHTENING'] += 1
        
    if inflation_yoy > 3.0 and fed_funds > 4.0:
        scores['TIGHTENING'] += 1
    
    # NEUTRAL signals
    if 0.5 <= real_short <= 1.5 and 1.5 <= inflation_yoy <= 2.5:
        scores['NEUTRAL'] += 2
    
    if abs(rate_change_6m) < 0.25:
        scores['NEUTRAL'] += 1
        
    if yield_curve_slope is not None and 0 < yield_curve_slope < 2:
        scores['NEUTRAL'] += 1
    
    # UNCERTAIN signals (mixed or extreme)
    if scores['EASING'] > 0 and scores['TIGHTENING'] > 0:
        scores['UNCERTAIN'] += 2
        
    if yield_curve_slope is not None and yield_curve_slope < -0.5:
        scores['UNCERTAIN'] += 1  # Inverted curve = uncertainty
        
    if abs(inflation_yoy - 2.0) > 2.0:
        scores['UNCERTAIN'] += 1  # Extreme inflation/deflation
    
    # Determine regime
    max_score = max(scores.values())
    if max_score == 0:
        regime = 'NEUTRAL'
        confidence = 0.5
    else:
        # Get regime with max score
        regime = max(scores.keys(), key=lambda k: scores[k])
        
        # Confidence based on score margin
        second_best = sorted(scores.values(), reverse=True)[1] if len(scores) > 1 else 0
        margin = max_score - second_best
        confidence = min(1.0, 0.5 + margin * 0.15)
    
    return regime, confidence, factors


class FedPolicyOverlay:
    """
    Federal Reserve policy overlay for tactical allocation.
    
    Fetches real-time data from FRED and determines policy regime,
    then recommends allocation shifts from base 46/38/16.
    """
    
    def __init__(self, cache_path: Path = FRED_CACHE):
        self.cache_path = cache_path
        self.data: Dict[str, pd.DataFrame] = {}
        self.current_regime: Optional[FedPolicyRegime] = None
        
    def fetch_data(self, force_refresh: bool = False) -> Dict[str, pd.DataFrame]:
        """Fetch all required FRED data."""
        self.data = fetch_all_fred_data(self.cache_path, force_refresh)
        return self.data
    
    def detect_regime(self, timestamp: Optional[str] = None) -> Optional[FedPolicyRegime]:
        """
        Detect current Federal Reserve policy regime.
        """
        if not self.data:
            self.fetch_data()
        
        # Extract latest values
        fed_funds_df = self.data.get('FEDFUNDS')
        cpi_df = self.data.get('CPIAUCSL')
        tips_10y_df = self.data.get('DFII10')
        nominal_10y_df = self.data.get('DGS10')
        nominal_2y_df = self.data.get('DGS2')
        breakeven_df = self.data.get('T10YIE')
        
        if fed_funds_df is None or fed_funds_df.empty:
            print("Error: No Fed Funds data available")
            return None
        
        # Latest Fed Funds
        latest_fed = fed_funds_df.iloc[-1]
        fed_funds_rate = latest_fed['value']
        
        # Calculate inflation YoY from CPI
        if cpi_df is not None and len(cpi_df) > 12:
            cpi_inflation = calculate_inflation_yoy(cpi_df)
            inflation_yoy = cpi_inflation.iloc[-1]['inflation_yoy']
        else:
            # Fallback to breakeven
            if breakeven_df is not None:
                inflation_yoy = breakeven_df.iloc[-1]['value']
            else:
                inflation_yoy = 2.0  # Default assumption
        
        # Real rates
        if tips_10y_df is not None:
            real_rate_10y = tips_10y_df.iloc[-1]['value']
        elif nominal_10y_df is not None and breakeven_df is not None:
            # Approximate: nominal - breakeven
            real_rate_10y = nominal_10y_df.iloc[-1]['value'] - breakeven_df.iloc[-1]['value']
        else:
            real_rate_10y = fed_funds_rate - inflation_yoy
        
        # Short-term real rate
        real_rate_short = fed_funds_rate - inflation_yoy
        
        # Yield curve slope
        if nominal_10y_df is not None and nominal_2y_df is not None:
            latest_10y = nominal_10y_df.iloc[-1]['value']
            # Find closest 2y date
            closest_2y = nominal_2y_df[nominal_2y_df['date'] <= nominal_10y_df.iloc[-1]['date']]
            if not closest_2y.empty:
                latest_2y = closest_2y.iloc[-1]['value']
                yield_curve_slope = latest_10y - latest_2y
            else:
                yield_curve_slope = None
        else:
            yield_curve_slope = None
        
        # Rate change over 6 months
        if len(fed_funds_df) > 6:
            rate_6m_ago_idx = max(0, len(fed_funds_df) - 7)
            rate_6m_ago = fed_funds_df.iloc[rate_6m_ago_idx]['value']
            rate_change_6m = fed_funds_rate - rate_6m_ago
        else:
            rate_change_6m = 0.0
        
        # Classify regime
        regime, confidence, factors = classify_fed_regime(
            fed_funds=fed_funds_rate,
            inflation_yoy=inflation_yoy,
            real_rate_10y=real_rate_10y,
            real_rate_short=real_rate_short,
            yield_curve_slope=yield_curve_slope,
            rate_change_6m=rate_change_6m
        )
        
        timestamp = timestamp or datetime.now().isoformat()
        
        self.current_regime = FedPolicyRegime(
            timestamp=timestamp,
            regime=regime,
            fed_funds_rate=fed_funds_rate,
            inflation_yoy=inflation_yoy,
            real_rate_10y=real_rate_10y,
            real_rate_short=real_rate_short,
            breakeven_10y=breakeven_df.iloc[-1]['value'] if breakeven_df is not None else 2.0,
            yield_curve_10y2y=yield_curve_slope if yield_curve_slope is not None else 0.0,
            confidence=confidence,
            regime_factors=factors
        )
        
        return self.current_regime
    
    def get_allocation_recommendation(
        self,
        base_allocation: Dict[str, float] = None
    ) -> Dict:
        """
        Get allocation recommendation based on Fed policy regime.
        """
        if self.current_regime is None:
            self.detect_regime()
        
        if self.current_regime is None:
            return {"error": "Unable to detect regime"}
        
        base = base_allocation or {'SPY': 0.46, 'GLD': 0.38, 'TLT': 0.16}
        shifts = self.current_regime.get_allocation_shift()
        
        # Apply shifts
        recommended = {}
        for asset, base_weight in base.items():
            shift = shifts.get(asset, 0.0)
            recommended[asset] = max(0.05, min(0.90, base_weight + shift))
        
        # Normalize to 1.0
        total = sum(recommended.values())
        if total > 0:
            for asset in recommended:
                recommended[asset] /= total
        
        # Calculate deltas
        deltas = {asset: recommended[asset] - base[asset] for asset in base}
        
        return {
            'timestamp': self.current_regime.timestamp,
            'strategy': 'Fed Policy Overlay v2.54',
            'regime': self.current_regime.regime,
            'regime_confidence': round(self.current_regime.confidence, 3),
            'base_allocation': base,
            'recommended_allocation': {k: round(v, 4) for k, v in recommended.items()},
            'deltas': {k: round(v, 4) for k, v in deltas.items()},
            'key_metrics': {
                'fed_funds': round(self.current_regime.fed_funds_rate, 2),
                'inflation_yoy': round(self.current_regime.inflation_yoy, 2),
                'real_rate_10y': round(self.current_regime.real_rate_10y, 2),
                'real_rate_short': round(self.current_regime.real_rate_short, 2),
                'yield_curve_10y2y': round(self.current_regime.yield_curve_10y2y, 2)
            },
            'regime_factors': self.current_regime.regime_factors,
            'divergence_risk': self.current_regime.is_divergence_risk()
        }


def main():
    parser = argparse.ArgumentParser(description="Fed Policy Overlay v2.54")
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # fetch command
    fetch_parser = subparsers.add_parser('fetch', help='Fetch FRED data')
    fetch_parser.add_argument('--force', action='store_true', help='Force refresh')
    
    # regime command
    regime_parser = subparsers.add_parser('regime', help='Show current Fed regime')
    regime_parser.add_argument('--refresh', action='store_true', help='Refresh data first')
    
    # allocate command
    allocate_parser = subparsers.add_parser('allocate', help='Get allocation recommendation')
    allocate_parser.add_argument('--portfolio', default='46/38/16', help='Base allocation')
    allocate_parser.add_argument('--refresh', action='store_true')
    
    # backtest command (placeholder)
    backtest_parser = subparsers.add_parser('backtest', help='Backtest regime strategy')
    backtest_parser.add_argument('--start', default='2005-01-01', help='Start date')
    backtest_parser.add_argument('--end', help='End date')
    
    # status command
    status_parser = subparsers.add_parser('status', help='Show overlay status')
    
    args = parser.parse_args()
    
    if args.command == 'fetch':
        overlay = FedPolicyOverlay()
        data = overlay.fetch_data(force_refresh=args.force)
        print(f"\nFetched {len(data)} series:")
        for series_id, df in data.items():
            print(f"  {series_id}: {len(df)} obs, latest={df.iloc[-1]['date'].strftime('%Y-%m-%d')}")
    
    elif args.command == 'regime':
        overlay = FedPolicyOverlay()
        if args.refresh:
            overlay.fetch_data(force_refresh=True)
        
        regime = overlay.detect_regime()
        if regime:
            print(json.dumps(regime.to_dict(), indent=2, default=str))
        else:
            print("Error: Could not detect regime")
    
    elif args.command == 'allocate':
        overlay = FedPolicyOverlay()
        if args.refresh:
            overlay.fetch_data(force_refresh=True)
        
        # Parse allocation
        parts = args.portfolio.split('/')
        base_alloc = {
            'SPY': float(parts[0]) / 100,
            'GLD': float(parts[1]) / 100,
            'TLT': float(parts[2]) / 100,
        }
        
        result = overlay.get_allocation_recommendation(base_alloc)
        print(json.dumps(result, indent=2, default=str))
    
    elif args.command == 'backtest':
        print("Backtest functionality: TBD")
        print("Historical FRED data needed for full backtest")
        print("Use 'fetch' to download data first")
    
    elif args.command == 'status':
        overlay = FedPolicyOverlay()
        data = overlay.fetch_data()
        
        print("Fed Policy Overlay v2.54 - Status")
        print("=" * 40)
        print(f"Cache: {FRED_CACHE}")
        print(f"Cache exists: {FRED_CACHE.exists()}")
        
        if FRED_CACHE.exists():
            cache_age = datetime.now() - datetime.fromtimestamp(FRED_CACHE.stat().st_mtime)
            print(f"Cache age: {cache_age}")
        
        print(f"\nData series available:")
        for series_id, df in data.items():
            if not df.empty:
                latest = df.iloc[-1]
                print(f"  {series_id}: {latest['date'].strftime('%Y-%m-%d')} = {latest['value']:.2f}")
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()

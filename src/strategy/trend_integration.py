#!/usr/bin/env python3
"""
Portfolio-Lab v2.30: Trend Integration Module

Multi-strategy hedge fund replication with CTA trend-following, 
momentum, and carry factor exposure via liquid ETFs.

Based on Q3 2026 deep research:
- DBMF (iMGP DBi): 13.84% 2025 returns, 0.85% ER, trend replication
- CTA (Simplify): 0.75% ER, Altis Partners models, multi-factor
- HFMF (Unlimited): 0.95% ER, 2x volatility target, sector rotation

Usage:
    from src.strategy.trend_integration import TrendReplicationStrategy, CTAOverlay
    
    strategy = TrendReplicationStrategy()
    weights = strategy.calculate_overlay(
        base_allocation={"SPY": 0.46, "GLD": 0.38, "TLT": 0.16},
        trend_strength=0.6,
        vol_regime="normal"
    )

CLI:
    python -m src.strategy.trend_integration analyze --portfolio 100000 --regime normal
    python -m src.strategy.trend_integration signals --lookback 90
    python -m src.strategy.trend_integration backtest --start 2020-01-01
"""

import json
import sqlite3
import sys
import argparse
import statistics
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import math

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# ---------------------------------------------------------------------------
# Constants and Configuration
# ---------------------------------------------------------------------------

DATA_DIR = Path("~/projects/portfolio-lab/data").expanduser()
TREND_DB = DATA_DIR / "trend_signals.db"

# Trend-following parameters
TREND_LOOKBACKS = [20, 60, 120, 252]  # 1m, 3m, 6m, 12m
TREND_SIGNAL_THRESHOLD = 0.1  # Min trend strength for signal

# CTA/Hedge Fund Replication ETFs (as of Q2 2026)
REPLICATION_ETFS = {
    "DBMF": {
        "name": "iMGP DBi Managed Futures Strategy",
        "category": "trend_following",
        "expense_ratio": 0.0085,
        "target_vol": 0.10,
        "strategy": "replication",
        "benchmark": "SG_CTA_Index",
        "markets": ["equities", "bonds", "commodities", "currencies"],
        "avg_annual_return": 0.098,
        "max_drawdown": -0.15,
        "correlation_spy": 0.05,
    },
    "CTA": {
        "name": "Simplify Managed Futures Strategy",
        "category": "multi_factor",
        "expense_ratio": 0.0075,
        "target_vol": 0.12,
        "strategy": "active_cta",
        "models": ["Altis_Partners"],
        "factors": ["trend", "carry", "correlation"],
        "avg_annual_return": 0.088,
        "max_drawdown": -0.18,
        "correlation_spy": 0.12,
    },
    "KMLM": {
        "name": "KraneShares Mount Lucas Managed Futures",
        "category": "rules_based",
        "expense_ratio": 0.0095,
        "target_vol": 0.10,
        "strategy": "rules_based",
        "avg_annual_return": 0.075,
        "max_drawdown": -0.12,
        "correlation_spy": 0.08,
    },
    "HFMF": {
        "name": "Unlimited HFMF Managed Futures",
        "category": "enhanced_vol",
        "expense_ratio": 0.0095,
        "target_vol": 0.20,  # 2x sector vol
        "strategy": "enhanced",
        "avg_annual_return": 0.12,
        "max_drawdown": -0.25,
        "correlation_spy": 0.10,
    },
}

# Volatility regime configurations
VOL_REGIMES = {
    "low": {
        "max_overlay": 0.20,
        "leverage_factor": 1.3,
        "trend_threshold": 0.05,
    },
    "normal": {
        "max_overlay": 0.15,
        "leverage_factor": 1.0,
        "trend_threshold": 0.10,
    },
    "high": {
        "max_overlay": 0.10,
        "leverage_factor": 0.7,
        "trend_threshold": 0.15,
    },
    "extreme": {
        "max_overlay": 0.05,
        "leverage_factor": 0.5,
        "trend_threshold": 0.20,
    },
}

# Trend signal weights by lookback
LOOKBACK_WEIGHTS = {
    20: 0.10,   # 1 month
    60: 0.25,   # 3 month
    120: 0.30,  # 6 month
    252: 0.35,  # 12 month
}

# Carry factor markets (futures curve analysis)
CARRY_MARKETS = {
    "commodities": ["GLD", "USO", "DBA"],
    "rates": ["TLT", "IEF", "SHY"],
    "currencies": ["UUP", "FXE", "FXY"],
}


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class TrendSignal:
    """Individual trend signal for an asset."""
    ticker: str
    timestamp: str
    
    # Trend metrics by lookback
    trend_1m: float  # 20-day
    trend_3m: float  # 60-day
    trend_6m: float  # 120-day
    trend_12m: float  # 252-day
    
    # Composite
    composite_trend: float  # Weighted average
    trend_strength: float  # Absolute magnitude
    trend_direction: str  # bullish, bearish, neutral
    
    # Momentum quality
    momentum_consistency: float  # % of lookbacks agreeing
    sharpe_of_trends: float  # Sharpe across lookbacks
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CarrySignal:
    """Carry signal from futures curve analysis."""
    ticker: str
    timestamp: str
    
    # Carry metrics
    roll_yield_annual: float  # Annualized roll yield
    curve_shape: str  # contango, backwardation, flat
    
    # Signal
    carry_signal: float  # -1.0 to +1.0
    carry_quality: float  # Confidence 0.0 to 1.0
    
    # Risk
    volatility_annual: float
    carry_per_vol: float  # Carry per unit vol
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CTAAllocation:
    """CTA/hedge fund replication allocation."""
    portfolio_value: float
    timestamp: str
    
    # Overlay sizing
    base_allocation: Dict[str, float]
    overlay_pct: float  # % of portfolio to CTA overlay
    overlay_usd: float
    
    # ETF breakdown
    replication_etfs: Dict[str, Dict[str, Any]]  # ETF code -> allocation details
    
    # Risk metrics
    expected_vol: float
    expected_return: float
    correlation_to_base: float
    diversification_ratio: float
    
    # Trend context
    vol_regime: str
    trend_strength_avg: float
    signal_confidence: float
    
    # Rebalancing
    rebalance_triggered: bool
    rebalance_threshold: float = 0.02  # 2%
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TrendBacktestResult:
    """Backtest results for trend strategy."""
    start_date: str
    end_date: str
    
    # Performance
    total_return: float
    annualized_return: float
    annualized_vol: float
    sharpe_ratio: float
    max_drawdown: float
    
    # Comparison
    spy_return: float
    spy_vol: float
    spy_sharpe: float
    
    # Risk metrics
    correlation_to_spy: float
    correlation_to_bonds: float
    correlation_to_gold: float
    
    # Drawdown analysis
    recovery_time_avg: float
    drawdown_events: int
    
    # Regime performance
    performance_by_regime: Dict[str, Dict[str, float]]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Database Setup
# ---------------------------------------------------------------------------

def init_database():
    """Initialize trend signals database."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(TREND_DB)
    cursor = conn.cursor()
    
    # Trend signals table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS trend_signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            trend_1m REAL,
            trend_3m REAL,
            trend_6m REAL,
            trend_12m REAL,
            composite_trend REAL,
            trend_strength REAL,
            trend_direction TEXT,
            momentum_consistency REAL,
            metadata TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(ticker, timestamp)
        )
    """)
    
    # Carry signals table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS carry_signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            roll_yield_annual REAL,
            curve_shape TEXT,
            carry_signal REAL,
            carry_quality REAL,
            volatility_annual REAL,
            carry_per_vol REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(ticker, timestamp)
        )
    """)
    
    # CTA allocation history
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS cta_allocations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            portfolio_value REAL,
            overlay_pct REAL,
            overlay_usd REAL,
            replication_etfs TEXT,
            expected_vol REAL,
            expected_return REAL,
            diversification_ratio REAL,
            vol_regime TEXT,
            signal_confidence REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Performance tracking
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS trend_performance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            etf_code TEXT NOT NULL,
            nav REAL,
            aum REAL,
            ytd_return REAL,
            expense_ratio REAL,
            UNIQUE(date, etf_code)
        )
    """)
    
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Trend Signal Generator
# ---------------------------------------------------------------------------

class TrendSignalGenerator:
    """Generate trend-following signals across multiple timeframes."""
    
    def __init__(self):
        init_database()
        self.market_db = DATA_DIR / "market.db"
    
    def calculate_trend_signal(self, ticker: str) -> Optional[TrendSignal]:
        """Calculate comprehensive trend signal for a ticker."""
        if not self.market_db.exists():
            return None
        
        conn = sqlite3.connect(self.market_db)
        cursor = conn.cursor()
        
        # Get price history
        cursor.execute("""
            SELECT date, close FROM prices 
            WHERE symbol = ? 
            AND date >= date('now', '-400 days')
            ORDER BY date DESC
        """, (ticker,))
        
        rows = cursor.fetchall()
        conn.close()
        
        if len(rows) < 252:
            return None
        
        # Reverse to chronological order
        prices = [r[1] for r in reversed(rows)]
        
        # Calculate trend for each lookback
        trends = {}
        for days in TREND_LOOKBACKS:
            if len(prices) >= days:
                current = prices[-1]
                past = prices[-days]
                trend_return = (current / past) - 1
                trends[days] = trend_return
            else:
                trends[days] = 0.0
        
        # Calculate composite trend
        weighted_trend = sum(
            trends[days] * LOOKBACK_WEIGHTS[days]
            for days in trends.keys()
        )
        
        # Trend strength
        trend_strength = abs(weighted_trend)
        
        # Direction
        if weighted_trend > TREND_SIGNAL_THRESHOLD:
            direction = "bullish"
        elif weighted_trend < -TREND_SIGNAL_THRESHOLD:
            direction = "bearish"
        else:
            direction = "neutral"
        
        # Momentum consistency (what % of lookbacks agree with composite)
        agreeing = sum(
            1 for days, trend in trends.items()
            if (weighted_trend > 0 and trend > 0) or (weighted_trend < 0 and trend < 0)
        )
        consistency = agreeing / len(trends) if trends else 0.0
        
        # Sharpe of trends (trend returns / volatility)
        if len(trends) > 1:
            trend_values = list(trends.values())
            trend_vol = statistics.stdev(trend_values) if len(trend_values) > 1 else 0.0001
            sharpe_of_trends = statistics.mean(trend_values) / trend_vol if trend_vol > 0 else 0.0
        else:
            sharpe_of_trends = 0.0
        
        signal = TrendSignal(
            ticker=ticker,
            timestamp=datetime.now().isoformat(),
            trend_1m=round(trends[20], 4),
            trend_3m=round(trends[60], 4),
            trend_6m=round(trends[120], 4),
            trend_12m=round(trends[252], 4),
            composite_trend=round(weighted_trend, 4),
            trend_strength=round(trend_strength, 4),
            trend_direction=direction,
            momentum_consistency=round(consistency, 4),
            sharpe_of_trends=round(sharpe_of_trends, 4),
            metadata={
                "price_current": prices[-1],
                "price_1m_ago": prices[-20] if len(prices) >= 20 else None,
                "lookbacks_available": len(trends),
            }
        )
        
        # Store to database
        self._store_signal(signal)
        
        return signal
    
    def _store_signal(self, signal: TrendSignal):
        """Store trend signal to database."""
        conn = sqlite3.connect(TREND_DB)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO trend_signals 
            (ticker, timestamp, trend_1m, trend_3m, trend_6m, trend_12m,
             composite_trend, trend_strength, trend_direction,
             momentum_consistency, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            signal.ticker, signal.timestamp,
            signal.trend_1m, signal.trend_3m, signal.trend_6m, signal.trend_12m,
            signal.composite_trend, signal.trend_strength, signal.trend_direction,
            signal.momentum_consistency, json.dumps(signal.metadata)
        ))
        
        conn.commit()
        conn.close()
    
    def calculate_carry_signal(self, ticker: str) -> Optional[CarrySignal]:
        """Calculate carry signal from implied roll yield."""
        # Simplified: use recent performance vs expected
        # In production, would analyze futures curve
        
        try:
            trend = self.calculate_trend_signal(ticker)
            if not trend:
                return None
            
            # Proxy carry: if trending up with momentum = possible backwardation
            if trend.trend_direction == "bullish" and trend.trend_strength > 0.15:
                roll_yield = 0.03  # Assume 3% annual roll yield
                curve = "backwardation"
                carry_sig = 0.5
            elif trend.trend_direction == "bearish" and trend.trend_strength > 0.15:
                roll_yield = -0.02  # Contango (negative roll)
                curve = "contango"
                carry_sig = -0.3
            else:
                roll_yield = 0.01
                curve = "flat"
                carry_sig = 0.0
            
            # Estimate volatility from trend Sharpe
            vol = 0.15  # Default 15%
            if trend.sharpe_of_trends != 0:
                vol = abs(trend.composite_trend / trend.sharpe_of_trends)
                vol = min(0.50, max(0.05, vol))  # Cap between 5-50%
            
            return CarrySignal(
                ticker=ticker,
                timestamp=datetime.now().isoformat(),
                roll_yield_annual=roll_yield,
                curve_shape=curve,
                carry_signal=carry_sig,
                carry_quality=0.6 if curve in ["backwardation", "contango"] else 0.3,
                volatility_annual=round(vol, 4),
                carry_per_vol=round(roll_yield / vol, 4) if vol > 0 else 0.0,
            )
            
        except Exception as e:
            return None
    
    def get_trend_regime(self) -> str:
        """Determine overall market trend regime."""
        # Check major assets
        assets = ["SPY", "GLD", "TLT"]
        
        signals = []
        for asset in assets:
            sig = self.calculate_trend_signal(asset)
            if sig:
                signals.append(sig)
        
        if not signals:
            return "normal"
        
        # Average trend strength
        avg_strength = statistics.mean([s.trend_strength for s in signals])
        
        # Count directions
        bullish = sum(1 for s in signals if s.trend_direction == "bullish")
        bearish = sum(1 for s in signals if s.trend_direction == "bearish")
        
        # Determine regime
        if avg_strength > 0.20 and bullish >= 2:
            return "strong_trend"
        elif avg_strength > 0.20 and bearish >= 2:
            return "bear_trend"
        elif avg_strength < 0.05:
            return "choppy"
        else:
            return "normal"


# ---------------------------------------------------------------------------
# Trend Replication Strategy
# ---------------------------------------------------------------------------

class TrendReplicationStrategy:
    """
    CTA/Hedge fund replication strategy with dynamic overlay sizing.
    
    Implements:
    - Trend-following signals across 20/60/120/252 day lookbacks
    - Multi-factor approach (trend + carry + correlation)
    - Volatility targeting based on regime
    - ETF-based implementation for liquidity
    """
    
    def __init__(self):
        init_database()
        self.generator = TrendSignalGenerator()
        self.market_db = DATA_DIR / "market.db"
    
    def calculate_overlay(
        self,
        portfolio_value: float,
        base_allocation: Dict[str, float],
        vol_regime: str = "normal",
        trend_strength: Optional[float] = None,
    ) -> CTAAllocation:
        """
        Calculate CTA overlay allocation.
        
        Args:
            portfolio_value: Total portfolio value
            base_allocation: Base portfolio weights (e.g., SPY/GLD/TLT)
            vol_regime: low/normal/high/extreme volatility
            trend_strength: Override calculated trend strength
        
        Returns:
            CTAAllocation with ETF breakdown
        """
        # Get regime config
        regime_config = VOL_REGIMES.get(vol_regime, VOL_REGIMES["normal"])
        
        # Calculate trend signals if not provided
        if trend_strength is None:
            trend_regime = self.generator.get_trend_regime()
            
            # Get trend strength from main assets
            signals = []
            for ticker in base_allocation.keys():
                sig = self.generator.calculate_trend_signal(ticker)
                if sig:
                    signals.append(sig.trend_strength)
            
            trend_strength = statistics.mean(signals) if signals else 0.10
        
        # Determine overlay size
        max_overlay = regime_config["max_overlay"]
        
        # Scale by trend strength (more trend = more allocation)
        # Only allocate if trend strength exceeds threshold
        if trend_strength >= regime_config["trend_threshold"]:
            trend_scalar = min(1.0, trend_strength / 0.20)  # Max at 20% trend
            overlay_pct = max_overlay * trend_scalar
        else:
            overlay_pct = 0.0  # No allocation in weak trends
        
        overlay_usd = portfolio_value * overlay_pct
        
        # Allocate across replication ETFs
        replication_allocs = self._allocate_replication_etfs(
            overlay_usd, vol_regime, trend_strength
        )
        
        # Calculate expected metrics
        expected_return = self._calculate_expected_return(replication_allocs)
        expected_vol = self._calculate_expected_vol(replication_allocs, regime_config)
        
        # Correlation to base portfolio (typically low)
        correlation = 0.10  # CTA funds typically 0.05-0.15 correlation to SPY
        
        # Diversification ratio
        if expected_vol > 0:
            diversification = 1.0 / (1.0 + correlation)  # Higher = more diversifying
        else:
            diversification = 1.0
        
        return CTAAllocation(
            portfolio_value=portfolio_value,
            timestamp=datetime.now().isoformat(),
            base_allocation=base_allocation,
            overlay_pct=round(overlay_pct, 4),
            overlay_usd=round(overlay_usd, 2),
            replication_etfs=replication_allocs,
            expected_vol=round(expected_vol, 4),
            expected_return=round(expected_return, 4),
            correlation_to_base=round(correlation, 4),
            diversification_ratio=round(diversification, 4),
            vol_regime=vol_regime,
            trend_strength_avg=round(trend_strength, 4),
            signal_confidence=round(trend_strength * 2, 4),  # Proxy
            rebalance_triggered=False,
        )
    
    def _allocate_replication_etfs(
        self,
        overlay_usd: float,
        vol_regime: str,
        trend_strength: float,
    ) -> Dict[str, Dict[str, Any]]:
        """Allocate overlay across replication ETFs."""
        allocations = {}
        
        # Base weights
        if trend_strength > 0.15:
            # Strong trend: favor pure trend-followers
            weights = {
                "DBMF": 0.50,
                "KMLM": 0.30,
                "CTA": 0.20,
            }
        elif trend_strength > 0.10:
            # Moderate: balanced approach
            weights = {
                "DBMF": 0.35,
                "CTA": 0.35,
                "KMLM": 0.30,
            }
        else:
            # Weak trend: favor multi-factor
            weights = {
                "CTA": 0.50,
                "DBMF": 0.30,
                "KMLM": 0.20,
            }
        
        # Adjust for volatility regime
        if vol_regime == "high":
            # Reduce HFMF (higher vol), increase KMLM (more conservative)
            weights["KMLM"] = weights.get("KMLM", 0) + 0.10
            if "HFMF" in weights:
                weights["HFMF"] -= 0.10
        
        # Calculate allocations
        for etf, weight in weights.items():
            if etf in REPLICATION_ETFS:
                etf_config = REPLICATION_ETFS[etf]
                alloc_usd = overlay_usd * weight
                
                allocations[etf] = {
                    "allocation_pct": round(weight, 4),
                    "allocation_usd": round(alloc_usd, 2),
                    "expense_ratio": etf_config["expense_ratio"],
                    "target_vol": etf_config["target_vol"],
                    "expected_return": etf_config["avg_annual_return"],
                    "correlation_spy": etf_config["correlation_spy"],
                }
        
        return allocations
    
    def _calculate_expected_return(
        self,
        replication_allocs: Dict[str, Dict[str, Any]]
    ) -> float:
        """Calculate weighted expected return."""
        if not replication_allocs:
            return 0.0
        
        weighted_return = sum(
            details["allocation_pct"] * details["expected_return"]
            for details in replication_allocs.values()
        )
        
        return weighted_return
    
    def _calculate_expected_vol(
        self,
        replication_allocs: Dict[str, Dict[str, Any]],
        regime_config: Dict[str, float]
    ) -> float:
        """Calculate expected portfolio volatility."""
        if not replication_allocs:
            return 0.0
        
        # Weighted average vol
        base_vol = sum(
            details["allocation_pct"] * details["target_vol"]
            for details in replication_allocs.values()
        )
        
        # Adjust for leverage factor
        return base_vol * regime_config["leverage_factor"]
    
    def get_current_vix_level(self) -> Optional[float]:
        """Get current VIX level for volatility regime."""
        try:
            conn = sqlite3.connect(self.market_db)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT close FROM prices 
                WHERE symbol = 'VIX' 
                ORDER BY date DESC 
                LIMIT 1
            """)
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return float(row[0])
        except:
            pass
        
        return None
    
    def determine_vol_regime(self) -> str:
        """Determine volatility regime from VIX."""
        vix = self.get_current_vix_level()
        
        if vix is None:
            return "normal"
        
        if vix > 35:
            return "extreme"
        elif vix > 25:
            return "high"
        elif vix < 15:
            return "low"
        else:
            return "normal"


# ---------------------------------------------------------------------------
# CLI Interface
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Trend Integration v2.30")
    parser.add_argument("command", choices=["analyze", "signals", "backtest"])
    parser.add_argument("--portfolio", type=float, default=100000, help="Portfolio value")
    parser.add_argument("--regime", default="auto", choices=["low", "normal", "high", "extreme", "auto"])
    parser.add_argument("--lookback", type=int, default=252, help="Days for signal analysis")
    parser.add_argument("--start", default="2020-01-01", help="Backtest start date")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    
    args = parser.parse_args()
    
    strategy = TrendReplicationStrategy()
    generator = TrendSignalGenerator()
    
    if args.command == "analyze":
        # Determine regime
        if args.regime == "auto":
            vol_regime = strategy.determine_vol_regime()
        else:
            vol_regime = args.regime
        
        # Calculate overlay
        base = {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16}
        allocation = strategy.calculate_overlay(
            portfolio_value=args.portfolio,
            base_allocation=base,
            vol_regime=vol_regime,
        )
        
        if args.json:
            print(json.dumps(allocation.to_dict(), indent=2))
        else:
            print(f"\n📈 Trend Integration v2.30 Analysis")
            print(f"   Portfolio Value: ${args.portfolio:,.2f}")
            print(f"   Volatility Regime: {vol_regime.upper()}")
            print(f"   VIX Level: {strategy.get_current_vix_level() or 'N/A'}")
            print(f"\n   CTA Overlay Allocation:")
            print(f"   • Overlay %: {allocation.overlay_pct:.1%}")
            print(f"   • Overlay $: ${allocation.overlay_usd:,.2f}")
            print(f"\n   ETF Allocation:")
            for etf, details in allocation.replication_etfs.items():
                print(f"   • {etf}: {details['allocation_pct']:.0%} (${details['allocation_usd']:,.0f})")
                print(f"     - Expense Ratio: {details['expense_ratio']:.2%}")
                print(f"     - Expected Return: {details['expected_return']:.1%}")
            print(f"\n   Expected Metrics:")
            print(f"   • Expected Vol: {allocation.expected_vol:.1%}")
            print(f"   • Expected Return: {allocation.expected_return:.1%}")
            print(f"   • Correlation to Base: {allocation.correlation_to_base:.1%}")
            print(f"   • Diversification Ratio: {allocation.diversification_ratio:.2f}")
    
    elif args.command == "signals":
        print(f"\n📈 Trend Signals (Lookback: {args.lookback} days)\n")
        
        assets = ["SPY", "QQQ", "GLD", "TLT", "IWM", "VTI"]
        for ticker in assets:
            signal = generator.calculate_trend_signal(ticker)
            if signal:
                emoji = "📈" if signal.trend_direction == "bullish" else "📉" if signal.trend_direction == "bearish" else "➡️"
                print(f"   {emoji} {ticker:5} | Composite: {signal.composite_trend:+.1%} | "
                      f"1M: {signal.trend_1m:+.1%} | 12M: {signal.trend_12m:+.1%} | "
                      f"Strength: {signal.trend_strength:.2f}")
            else:
                print(f"   ⚠️  {ticker:5} | No signal available")
        
        # Carry signals
        print(f"\n   Carry Signals:")
        for ticker in ["GLD", "TLT"]:
            carry = generator.calculate_carry_signal(ticker)
            if carry:
                print(f"   • {ticker}: {carry.carry_signal:+.2f} ({carry.curve_shape}, "
                      f"roll: {carry.roll_yield_annual:.1%})")
    
    elif args.command == "backtest":
        print(f"\n📉 Trend Strategy Backtest ({args.start} to present)")
        print(f"\n   Simulation Parameters:")
        print(f"   • Trend Lookbacks: 20, 60, 120, 252 days")
        print(f"   • Overlay Range: 0-15%")
        print(f"   • ETFs: DBMF, CTA, KMLM")
        print(f"\n   Note: Backtest requires historical ETF data")
        print(f"   Run 'fetch-data' to update market database")


if __name__ == "__main__":
    main()

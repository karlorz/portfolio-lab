"""
CTA Trend-Following Overlay Strategy
Multi-timeframe trend detection with volatility targeting

Research: CME Group 2024, Graham Capital, Quantica Capital Q1 2025
- Volatility targeting reduces max drawdown by ~30%
- Multi-timeframe approach improves robustness
- Crisis alpha during equity/bond stress periods

Implementation:
- 3 timeframes: 20d (fast), 60d (medium), 120d (slow)
- Volatility targeting position sizing
- Equal risk allocation across markets
- Ensemble trend scoring
"""

import os
import json
import sqlite3
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, NamedTuple
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrendSignal:
    """Trend signal from a single timeframe"""
    timeframe: int  # days
    score: float  # -1 to +1 (bearish to bullish)
    strength: float  # 0 to 1 (conviction)
    price_vs_sma: float  # percentage above/below SMA
    regime: str  # "uptrend", "downtrend", "chop"


@dataclass
class CTAPosition:
    """CTA-style position with trend and vol targeting"""
    symbol: str
    asset_class: str
    base_weight: float
    trend_score: float  # -1 to +1 ensemble score
    trend_strength: float  # 0 to 1
    realized_vol: float  # annualized
    target_vol: float
    position_scalar: float  # vol targeting multiplier
    final_weight: float  # base * trend * vol_scalar
    signal: str  # "long", "short", "neutral"
    last_update: str


class CTATrendEngine:
    """
    CTA-style trend-following with multi-timeframe ensemble
    
    Timeframes:
    - Short (20d): Fast signals, more whipsaw, early entry
    - Medium (60d): Balanced, primary signal
    - Long (120d): Slow, major trends only, confirmation
    
    Position Sizing:
    - Equal risk allocation base
    - Volatility targeting adjustment
    - Trend strength conviction multiplier
    """
    
    # Timeframe configuration
    TIMEFRAMES = {
        "short": {"days": 20, "weight": 0.25, "threshold": 0.3},
        "medium": {"days": 60, "weight": 0.50, "threshold": 0.2},
        "long": {"days": 120, "weight": 0.25, "threshold": 0.1},
    }
    
    # Universe by asset class
    UNIVERSE = {
        # Equities
        "SPY": {"class": "equity", "alt": "QQQ", "base_weight": 0.15},
        "QQQ": {"class": "equity", "alt": "SPY", "base_weight": 0.10},
        "IWM": {"class": "equity", "alt": None, "base_weight": 0.05},
        "EFA": {"class": "equity", "alt": "VXUS", "base_weight": 0.05},
        "VXUS": {"class": "equity", "alt": "EFA", "base_weight": 0.05},
        # Bonds  
        "TLT": {"class": "bond", "alt": "IEF", "base_weight": 0.10},
        "IEF": {"class": "bond", "alt": "TLT", "base_weight": 0.05},
        "HYG": {"class": "credit", "alt": "LQD", "base_weight": 0.05},
        "LQD": {"class": "credit", "alt": "HYG", "base_weight": 0.05},
        # Commodities
        "GLD": {"class": "commodity", "alt": None, "base_weight": 0.10},
        "DBC": {"class": "commodity", "alt": None, "base_weight": 0.05},
        # Alternatives
        "VIX": {"class": "vol", "alt": None, "base_weight": 0.02},  # For regime detection
    }
    
    # Risk parameters
    TARGET_VOL = 0.10  # 10% annual volatility target
    MAX_LEVERAGE = 2.0
    MIN_LEVERAGE = 0.25
    MAX_POSITION_RISK = 0.15  # Max 15% risk per position
    REBALANCE_DAYS = 5  # Weekly rebalancing
    
    def __init__(
        self,
        db_path: Path = Path("~/projects/portfolio-lab/data/market.db").expanduser()
    ):
        self.db_path = db_path
        self.vol_lookback = 20
        
    def _fetch_data(self, symbol: str, days: int = 200) -> List[Dict]:
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
        
        return [
            {"date": row[0], "close": row[1], "volume": row[2]}
            for row in reversed(rows)
        ]
    
    def _calculate_sma(self, prices: np.ndarray, period: int) -> float:
        """Calculate simple moving average"""
        if len(prices) < period:
            return prices[-1] if len(prices) > 0 else 0
        return np.mean(prices[-period:])
    
    def _calculate_volatility(self, prices: np.ndarray, days: int = 20) -> float:
        """Calculate annualized realized volatility"""
        if len(prices) < days + 1:
            return 0.15  # Default 15%
        
        returns = np.diff(prices[-days-1:]) / prices[-days-1:-1]
        daily_vol = np.std(returns)
        annual_vol = daily_vol * np.sqrt(252)
        
        return max(annual_vol, 0.05)  # Floor at 5%
    
    def _calculate_trend_signal(
        self,
        symbol: str,
        timeframe_name: str,
        prices: np.ndarray
    ) -> Optional[TrendSignal]:
        """
        Calculate trend signal for a specific timeframe
        
        Score: -1 (strong bearish) to +1 (strong bullish)
        Based on price position relative to SMA and standard deviation
        """
        config = self.TIMEFRAMES[timeframe_name]
        period = config["days"]
        
        if len(prices) < period + 20:
            return None
        
        current_price = prices[-1]
        sma = self._calculate_sma(prices, period)
        
        # Calculate price position relative to SMA
        price_std = np.std(prices[-period:])
        if price_std == 0:
            price_std = current_price * 0.01  # 1% default
        
        # Normalized distance from SMA (z-score like)
        price_vs_sma = (current_price - sma) / price_std
        
        # Convert to -1 to +1 score with sigmoid-like compression
        # Score saturates around +/- 2 standard deviations
        score = np.tanh(price_vs_sma * 0.5)
        
        # Strength based on consistency of trend
        # Check if price has been above/below SMA consistently
        recent_prices = prices[-20:]
        recent_sma = [np.mean(prices[max(0, i-period):i]) for i in range(-20, 0)]
        above_count = sum(1 for p, s in zip(recent_prices, recent_sma) if p > s)
        consistency = abs(above_count - 10) / 10  # 0-1, higher = more consistent
        
        # Combine for strength
        strength = min(abs(score) * 0.7 + consistency * 0.3, 1.0)
        
        # Determine regime
        threshold = config["threshold"]
        if score > threshold:
            regime = "uptrend"
        elif score < -threshold:
            regime = "downtrend"
        else:
            regime = "chop"
        
        return TrendSignal(
            timeframe=period,
            score=score,
            strength=strength,
            price_vs_sma=(current_price / sma - 1) * 100,
            regime=regime
        )
    
    def _ensemble_trend_score(self, signals: List[TrendSignal]) -> Tuple[float, float, str]:
        """
        Combine multi-timeframe signals into ensemble score
        
        Returns: (ensemble_score, strength, consensus_regime)
        """
        if not signals:
            return 0.0, 0.0, "neutral"
        
        # Weight by timeframe configuration
        weighted_score = 0.0
        total_weight = 0.0
        
        for signal in signals:
            weight = self.TIMEFRAMES["short"]["weight"]
            if signal.timeframe == 60:
                weight = self.TIMEFRAMES["medium"]["weight"]
            elif signal.timeframe == 120:
                weight = self.TIMEFRAMES["long"]["weight"]
            
            weighted_score += signal.score * signal.strength * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0.0, 0.0, "neutral"
        
        ensemble_score = weighted_score / total_weight
        
        # Calculate ensemble strength (agreement across timeframes)
        if len(signals) >= 2:
            signs = [np.sign(s.score) for s in signals]
            sign_agreement = abs(sum(signs)) / len(signs)  # 0 = disagreement, 1 = full agreement
            avg_strength = np.mean([s.strength for s in signals])
            strength = sign_agreement * avg_strength
        else:
            strength = signals[0].strength if signals else 0
        
        # Determine consensus regime
        regimes = [s.regime for s in signals]
        if regimes.count("uptrend") >= 2:
            consensus = "uptrend"
        elif regimes.count("downtrend") >= 2:
            consensus = "downtrend"
        else:
            consensus = "mixed"
        
        return ensemble_score, strength, consensus
    
    def _calculate_position_scalar(
        self,
        realized_vol: float,
        trend_strength: float
    ) -> float:
        """
        Calculate position size scalar based on vol targeting and trend conviction
        
        Base: target_vol / realized_vol
        Adjustment: multiply by trend_strength (reduce size when uncertain)
        """
        if realized_vol <= 0:
            return 1.0
        
        # Volatility targeting
        vol_scalar = self.TARGET_VOL / realized_vol
        
        # Apply trend conviction (reduce exposure when trend is weak/unclear)
        conviction_multiplier = 0.5 + (trend_strength * 0.5)  # 0.5 to 1.0
        
        # Combined scalar
        scalar = vol_scalar * conviction_multiplier
        
        # Apply bounds
        scalar = max(scalar, self.MIN_LEVERAGE)
        scalar = min(scalar, self.MAX_LEVERAGE)
        
        return scalar
    
    def analyze_symbol(self, symbol: str) -> Optional[CTAPosition]:
        """
        Full CTA analysis for a single symbol
        
        Returns CTAPosition with trend signal and position sizing
        """
        if symbol not in self.UNIVERSE:
            return None
        
        # Fetch data (need 120 days + buffer)
        data = self._fetch_data(symbol, 150)
        if len(data) < 120:
            return None
        
        prices = np.array([d["close"] for d in data])
        asset_info = self.UNIVERSE[symbol]
        
        # Calculate trend signals for each timeframe
        signals = []
        for tf_name in ["short", "medium", "long"]:
            signal = self._calculate_trend_signal(symbol, tf_name, prices)
            if signal:
                signals.append(signal)
        
        if not signals:
            return None
        
        # Ensemble score
        trend_score, trend_strength, consensus = self._ensemble_trend_score(signals)
        
        # Calculate volatility
        realized_vol = self._calculate_volatility(prices, self.vol_lookback)
        
        # Position scalar (vol targeting + trend conviction)
        position_scalar = self._calculate_position_scalar(realized_vol, trend_strength)
        
        # Base weight from universe config
        base_weight = asset_info["base_weight"]
        
        # Final weight with trend direction
        # For long-only implementation: scale by positive trend only
        # Full implementation could allow short via inverse ETFs
        trend_direction = max(trend_score, 0) if trend_score > 0 else 0  # Long only
        
        final_weight = base_weight * position_scalar * (0.5 + trend_direction * 0.5)
        
        # Cap at max position risk
        max_weight = self.MAX_POSITION_RISK
        final_weight = min(final_weight, max_weight)
        
        # Signal classification
        if trend_score > 0.3 and trend_strength > 0.5:
            signal = "long"
        elif trend_score < -0.3 and trend_strength > 0.5:
            signal = "short"  # Would use inverse ETF in practice
        else:
            signal = "neutral"
        
        return CTAPosition(
            symbol=symbol,
            asset_class=asset_info["class"],
            base_weight=base_weight,
            trend_score=trend_score,
            trend_strength=trend_strength,
            realized_vol=realized_vol,
            target_vol=self.TARGET_VOL,
            position_scalar=position_scalar,
            final_weight=final_weight,
            signal=signal,
            last_update=datetime.now().isoformat()
        )
    
    def evaluate(self) -> Dict:
        """
        Run full CTA evaluation across universe
        
        Returns allocation recommendation and signal summary
        """
        timestamp = datetime.now().isoformat()
        
        # Analyze all symbols in universe
        positions = {}
        for symbol in self.UNIVERSE.keys():
            position = self.analyze_symbol(symbol)
            if position:
                positions[symbol] = position
        
        if not positions:
            return {
                "timestamp": timestamp,
                "error": "Insufficient data for CTA analysis",
                "positions": {},
                "allocation": {},
                "summary": {}
            }
        
        # Normalize weights to sum to 1.0
        total_weight = sum(p.final_weight for p in positions.values())
        if total_weight > 0:
            normalized_allocation = {
                sym: round(pos.final_weight / total_weight, 4)
                for sym, pos in positions.items()
            }
        else:
            normalized_allocation = {}
        
        # Count signals by type
        signal_counts = {"long": 0, "short": 0, "neutral": 0}
        for pos in positions.values():
            signal_counts[pos.signal] = signal_counts.get(pos.signal, 0) + 1
        
        # Calculate portfolio-level metrics
        avg_trend_score = np.mean([p.trend_score for p in positions.values()])
        avg_trend_strength = np.mean([p.trend_strength for p in positions.values()])
        avg_vol = np.mean([p.realized_vol for p in positions.values()])
        
        # Risk distribution by asset class
        asset_class_weights = {}
        for pos in positions.values():
            ac = pos.asset_class
            asset_class_weights[ac] = asset_class_weights.get(ac, 0) + pos.final_weight
        
        return {
            "timestamp": timestamp,
            "strategy": "cta_trend_overlay",
            "target_volatility": self.TARGET_VOL,
            "positions": {
                sym: {
                    "asset_class": pos.asset_class,
                    "base_weight": pos.base_weight,
                    "trend_score": round(pos.trend_score, 4),
                    "trend_strength": round(pos.trend_strength, 4),
                    "realized_vol": round(pos.realized_vol, 4),
                    "position_scalar": round(pos.position_scalar, 4),
                    "final_weight": round(pos.final_weight, 4),
                    "signal": pos.signal
                }
                for sym, pos in positions.items()
            },
            "allocation": normalized_allocation,
            "summary": {
                "total_positions": len(positions),
                "signal_counts": signal_counts,
                "avg_trend_score": round(avg_trend_score, 4),
                "avg_trend_strength": round(avg_trend_strength, 4),
                "avg_realized_vol": round(avg_vol, 4),
                "asset_class_distribution": {
                    k: round(v / total_weight, 4) if total_weight > 0 else 0
                    for k, v in asset_class_weights.items()
                }
            }
        }
    
    def get_crisis_alpha_signals(self) -> Dict[str, str]:
        """
        Identify which assets would provide crisis alpha
        
        Based on correlation to SPY and trend characteristics
        """
        spy_data = self._fetch_data("SPY", 60)
        if len(spy_data) < 30:
            return {}
        
        spy_prices = np.array([d["close"] for d in spy_data])
        spy_return = (spy_prices[-1] / spy_prices[0]) - 1
        
        crisis_signals = {}
        
        for symbol in ["GLD", "TLT", "DBC"]:
            if symbol not in self.UNIVERSE:
                continue
                
            data = self._fetch_data(symbol, 60)
            if len(data) < 30:
                continue
            
            prices = np.array([d["close"] for d in data])
            
            # Calculate correlation to SPY
            min_len = min(len(spy_prices), len(prices))
            if min_len < 20:
                continue
            
            spy_returns = np.diff(spy_prices[-min_len:]) / spy_prices[-min_len:-1]
            sym_returns = np.diff(prices[-min_len:]) / prices[-min_len:-1]
            
            correlation = np.corrcoef(spy_returns, sym_returns)[0, 1]
            
            # Crisis alpha candidate: negative correlation + positive trend
            position = self.analyze_symbol(symbol)
            if position and correlation < 0 and position.trend_score > 0:
                crisis_signals[symbol] = {
                    "correlation_to_spy": round(correlation, 4),
                    "trend_score": round(position.trend_score, 4),
                    "rationale": "Negative correlation, positive trend"
                }
        
        return crisis_signals


if __name__ == "__main__":
    # Quick test
    engine = CTATrendEngine()
    result = engine.evaluate()
    print(json.dumps(result, indent=2))

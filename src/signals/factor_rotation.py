"""
Factor Rotation Signal Generator for Quality-Momentum Overlay (v3.00)

Phase 2 of v3.00 Factor Rotation implementation.
Generates quality-weighted momentum signals and factor allocation recommendations.

Key Components:
- QualityMomentumCalculator: Computes quality scores and momentum signals
- FactorRotationSignal: Generates allocation recommendations across MTUM/QUAL/USMV/VLUE
- Regime-based factor weighting
"""

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, NamedTuple
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime classification."""
    BULL = "bull"
    BEAR = "bear"
    NEUTRAL = "neutral"
    HIGH_VOL = "high_vol"
    CRISIS = "crisis"


@dataclass
class QualityScore:
    """Quality metrics for a factor ETF."""
    symbol: str
    date: str
    roe: float  # Return on equity
    debt_equity: float  # Debt to equity ratio
    earnings_stability: float  # Earnings variance (lower is better)
    profitability: float  # Gross margin stability
    composite_score: float  # Weighted average (0-1 scale)
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass  
class FactorAllocation:
    """Factor ETF allocation for a given regime."""
    mtum_pct: float  # Momentum
    qual_pct: float  # Quality
    usmv_pct: float  # Low Volatility
    vlue_pct: float  # Value
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @property
    def total(self) -> float:
        return self.mtum_pct + self.qual_pct + self.usmv_pct + self.vlue_pct


@dataclass
class RotationSignal:
    """Output signal from factor rotation analysis."""
    date: str
    regime: str
    quality_momentum_score: float  # Blended Q+M score (-1 to +1)
    confidence: float  # Signal confidence (0-1)
    factor_allocations: Dict[str, float]  # Symbol -> percentage
    equity_adjustment: float  # Recommended ± adjustment to base equity %
    rationale: List[str]  # Human-readable explanation
    
    def to_dict(self) -> Dict:
        return {
            "date": self.date,
            "regime": self.regime,
            "quality_momentum_score": self.quality_momentum_score,
            "confidence": self.confidence,
            "factor_allocations": self.factor_allocations,
            "equity_adjustment": self.equity_adjustment,
            "rationale": self.rationale,
        }


# Regime-based factor allocations (from research)
REGIME_ALLOCATIONS = {
    MarketRegime.BULL: FactorAllocation(
        mtum_pct=0.60,
        qual_pct=0.25,
        usmv_pct=0.10,
        vlue_pct=0.05,
    ),
    MarketRegime.BEAR: FactorAllocation(
        mtum_pct=0.10,
        qual_pct=0.40,
        usmv_pct=0.40,
        vlue_pct=0.10,
    ),
    MarketRegime.NEUTRAL: FactorAllocation(
        mtum_pct=0.35,
        qual_pct=0.35,
        usmv_pct=0.20,
        vlue_pct=0.10,
    ),
    MarketRegime.HIGH_VOL: FactorAllocation(
        mtum_pct=0.15,
        qual_pct=0.30,
        usmv_pct=0.45,
        vlue_pct=0.10,
    ),
    MarketRegime.CRISIS: FactorAllocation(
        mtum_pct=0.05,
        qual_pct=0.35,
        usmv_pct=0.50,
        vlue_pct=0.10,
    ),
}

# Quality scoring weights
QUALITY_WEIGHTS = {
    "roe": 0.30,
    "debt_equity": 0.25,
    "earnings_stability": 0.25,
    "profitability": 0.20,
}

# Factor ETF metadata
FACTOR_ETFS = {
    "MTUM": {"factor": "momentum", "expense": 0.0015},
    "QUAL": {"factor": "quality", "expense": 0.0015},
    "USMV": {"factor": "low_vol", "expense": 0.0015},
    "VLUE": {"factor": "value", "expense": 0.0015},
}


class QualityMomentumCalculator:
    """
    Calculates quality scores and momentum signals for factor rotation.
    
    Implements Asness QMJ (Quality Minus Junk) methodology adapted for ETFs.
    """
    
    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or Path("data/factors")
        self.db_path = self.data_dir / "factor_data.db"
        self.signals_dir = Path("data/signals")
        self.signals_dir.mkdir(parents=True, exist_ok=True)
        
    def compute_quality_score(
        self,
        symbol: str,
        roe: float,
        debt_equity: float,
        earnings_stability: float,
        profitability: float,
    ) -> float:
        """
        Compute composite quality score from components.
        
        Args:
            symbol: ETF symbol
            roe: Return on equity (higher is better)
            debt_equity: Debt/equity ratio (lower is better, inverted)
            earnings_stability: Earnings variance (lower is better, inverted)
            profitability: Gross margin stability (higher is better)
            
        Returns:
            Composite quality score (0-1 scale, 1 = highest quality)
        """
        # Normalize inputs to 0-1 scale (simplified normalization)
        # In production, these would use cross-sectional rankings
        
        # ROE: typical range 0-30%, cap at 0.5 for extreme values
        roe_score = min(roe / 0.30, 1.0) if roe > 0 else 0
        
        # Debt/Equity: typical range 0-2.0, invert so lower is better
        de_score = max(0, 1.0 - (debt_equity / 2.0)) if debt_equity >= 0 else 0
        
        # Earnings stability: assume already normalized (lower variance = higher score)
        es_score = max(0, 1.0 - earnings_stability)
        
        # Profitability: assume already normalized
        prof_score = max(0, min(1.0, profitability))
        
        # Weighted composite
        composite = (
            QUALITY_WEIGHTS["roe"] * roe_score +
            QUALITY_WEIGHTS["debt_equity"] * de_score +
            QUALITY_WEIGHTS["earnings_stability"] * es_score +
            QUALITY_WEIGHTS["profitability"] * prof_score
        )
        
        return round(composite, 4)
    
    def get_quality_scores(
        self,
        date: Optional[str] = None,
        lookback_days: int = 30,
    ) -> Dict[str, QualityScore]:
        """
        Retrieve quality scores for all factor ETFs.
        
        Args:
            date: Target date (YYYY-MM-DD), defaults to latest
            lookback_days: How far back to look for data
            
        Returns:
            Dict mapping symbol -> QualityScore
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
            
        scores = {}
        
        if not self.db_path.exists():
            logger.warning(f"Database not found: {self.db_path}")
            return scores
            
        with sqlite3.connect(self.db_path) as conn:
            for symbol in FACTOR_ETFS.keys():
                cursor = conn.execute(
                    """
                    SELECT roe, debt_equity, earnings_stability, profitability, composite_score
                    FROM quality_scores
                    WHERE symbol = ? AND date <= ?
                    ORDER BY date DESC
                    LIMIT 1
                    """,
                    (symbol, date)
                )
                row = cursor.fetchone()
                
                if row:
                    scores[symbol] = QualityScore(
                        symbol=symbol,
                        date=date,
                        roe=row[0] or 0,
                        debt_equity=row[1] or 0,
                        earnings_stability=row[2] or 0,
                        profitability=row[3] or 0,
                        composite_score=row[4] or 0.5,
                    )
                else:
                    # Default neutral score if no data
                    scores[symbol] = QualityScore(
                        symbol=symbol,
                        date=date,
                        roe=0.15,
                        debt_equity=0.5,
                        earnings_stability=0.5,
                        profitability=0.5,
                        composite_score=0.5,
                    )
                    
        return scores
    
    def compute_momentum_signal(
        self,
        symbol: str,
        date: Optional[str] = None,
        lookback_months: int = 12,
    ) -> float:
        """
        Compute momentum signal from price returns.
        
        Uses 12-month return minus last month (TSMOM style).
        
        Args:
            symbol: ETF symbol
            date: End date for calculation
            lookback_months: Months to look back
            
        Returns:
            Momentum score (-1 to +1, normalized)
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
            
        if not self.db_path.exists():
            return 0.0
            
        # Calculate returns
        end_date = datetime.strptime(date, "%Y-%m-%d")
        start_date = end_date - timedelta(days=lookback_months * 30)
        skip_date = end_date - timedelta(days=30)
        
        with sqlite3.connect(self.db_path) as conn:
            # Get price at start (formation period)
            cursor = conn.execute(
                "SELECT close FROM factor_prices WHERE symbol = ? AND date <= ? ORDER BY date DESC LIMIT 1",
                (symbol, start_date.strftime("%Y-%m-%d"))
            )
            start_row = cursor.fetchone()
            
            # Get price at skip date (exclude last month)
            cursor = conn.execute(
                "SELECT close FROM factor_prices WHERE symbol = ? AND date <= ? ORDER BY date DESC LIMIT 1",
                (symbol, skip_date.strftime("%Y-%m-%d"))
            )
            skip_row = cursor.fetchone()
            
            # Get current price
            cursor = conn.execute(
                "SELECT close FROM factor_prices WHERE symbol = ? AND date <= ? ORDER BY date DESC LIMIT 1",
                (symbol, date)
            )
            current_row = cursor.fetchone()
            
        if not all([start_row, skip_row, current_row]):
            return 0.0
            
        # TSMOM: 12mo return excluding last month
        try:
            formation_return = (skip_row[0] - start_row[0]) / start_row[0] if start_row[0] > 0 else 0
            # Normalize to -1 to +1 scale (assuming ±50% annual returns = ±1)
            signal = max(-1.0, min(1.0, formation_return / 0.50))
            return round(signal, 4)
        except (ZeroDivisionError, TypeError):
            return 0.0
    
    def compute_quality_momentum_blend(
        self,
        quality_score: float,
        momentum_signal: float,
        regime: MarketRegime,
        quality_weight: Optional[float] = None,
    ) -> float:
        """
        Blend quality and momentum signals based on regime.
        
        Args:
            quality_score: Quality score (0-1)
            momentum_signal: Momentum signal (-1 to +1)
            regime: Current market regime
            quality_weight: Override quality weight (uses regime default if None)
            
        Returns:
            Blended signal (-1 to +1)
        """
        if quality_weight is None:
            # Regime-based quality weight
            quality_weights = {
                MarketRegime.BULL: 0.30,
                MarketRegime.BEAR: 0.70,
                MarketRegime.NEUTRAL: 0.50,
                MarketRegime.HIGH_VOL: 0.65,
                MarketRegime.CRISIS: 0.80,
            }
            quality_weight = quality_weights.get(regime, 0.50)
        
        # Normalize quality to -1 to +1 scale (0.5 = 0)
        quality_normalized = (quality_score - 0.5) * 2
        
        # Weighted blend
        blended = quality_weight * quality_normalized + (1 - quality_weight) * momentum_signal
        
        return round(max(-1.0, min(1.0, blended)), 4)
    
    def detect_regime(
        self,
        date: Optional[str] = None,
        vix_level: Optional[float] = None,
        trend_strength: Optional[float] = None,
    ) -> MarketRegime:
        """
        Detect market regime based on VIX and trend strength.
        
        Args:
            date: Analysis date
            vix_level: VIX index level (fetched if not provided)
            trend_strength: Trend strength metric (fetched if not provided)
            
        Returns:
            MarketRegime classification
        """
        # Simplified regime detection
        # In production, would use HMM regime detector output
        
        if vix_level is None:
            vix_level = 20.0  # Default assumption
            
        if trend_strength is None:
            trend_strength = 0.0  # Neutral
            
        # Regime classification rules
        if vix_level > 35:
            return MarketRegime.CRISIS
        elif vix_level > 25:
            return MarketRegime.HIGH_VOL
        elif trend_strength > 0.3 and vix_level < 20:
            return MarketRegime.BULL
        elif trend_strength < -0.3:
            return MarketRegime.BEAR
        else:
            return MarketRegime.NEUTRAL
    
    def generate_rotation_signal(
        self,
        date: Optional[str] = None,
        regime: Optional[MarketRegime] = None,
        vix_level: Optional[float] = None,
    ) -> RotationSignal:
        """
        Generate factor rotation signal for given date.
        
        Args:
            date: Target date (defaults to today)
            regime: Override regime detection
            vix_level: VIX level for regime detection
            
        Returns:
            RotationSignal with allocations and recommendations
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
            
        # Detect or use provided regime
        if regime is None:
            regime = self.detect_regime(date, vix_level)
            
        # Get quality scores
        quality_scores = self.get_quality_scores(date)
        
        # Compute momentum signals
        momentum_signals = {
            symbol: self.compute_momentum_signal(symbol, date)
            for symbol in FACTOR_ETFS.keys()
        }
        
        # Compute quality-momentum blends for each ETF
        qm_scores = {}
        for symbol in FACTOR_ETFS.keys():
            quality = quality_scores.get(symbol, QualityScore(symbol, date, 0, 0, 0, 0, 0.5)).composite_score
            momentum = momentum_signals.get(symbol, 0)
            qm_scores[symbol] = self.compute_quality_momentum_blend(quality, momentum, regime)
        
        # Get regime-based allocations
        allocations = REGIME_ALLOCATIONS[regime]
        
        # Calculate factor allocations
        factor_allocations = {
            "MTUM": round(allocations.mtum_pct, 4),
            "QUAL": round(allocations.qual_pct, 4),
            "USMV": round(allocations.usmv_pct, 4),
            "VLUE": round(allocations.vlue_pct, 4),
        }
        
        # Compute composite quality-momentum score (weighted average)
        total_score = sum(qm_scores.values())
        composite_qm = total_score / len(qm_scores) if qm_scores else 0
        
        # Determine equity adjustment based on regime and quality
        if regime == MarketRegime.BULL and composite_qm > 0.3:
            equity_adj = 0.05  # Increase equity
        elif regime == MarketRegime.BEAR and composite_qm < -0.3:
            equity_adj = -0.10  # Decrease equity
        elif regime == MarketRegime.CRISIS:
            equity_adj = -0.15  # Defensive
        elif regime == MarketRegime.HIGH_VOL:
            equity_adj = -0.05  # Slight defensive
        else:
            equity_adj = 0.0
            
        # Build rationale
        rationale = [
            f"Regime: {regime.value}",
            f"Quality-Momentum composite: {composite_qm:+.2f}",
            f"MTUM Q-M: {qm_scores.get('MTUM', 0):+.2f}, QUAL Q-M: {qm_scores.get('QUAL', 0):+.2f}",
            f"Quality scores: MTUM={quality_scores.get('MTUM', QualityScore('MTUM', date, 0,0,0,0,0.5)).composite_score:.2f}, "
            f"QUAL={quality_scores.get('QUAL', QualityScore('QUAL', date, 0,0,0,0,0.5)).composite_score:.2f}",
        ]
        
        # Confidence based on data availability
        confidence = 0.7 if all(qm_scores.values()) else 0.5
        
        return RotationSignal(
            date=date,
            regime=regime.value,
            quality_momentum_score=composite_qm,
            confidence=confidence,
            factor_allocations=factor_allocations,
            equity_adjustment=equity_adj,
            rationale=rationale,
        )
    
    def save_signal(self, signal: RotationSignal, filename: Optional[str] = None) -> Path:
        """
        Save rotation signal to JSON.
        
        Args:
            signal: RotationSignal to save
            filename: Override filename (defaults to factor_rotation_YYYY-MM-DD.json)
            
        Returns:
            Path to saved file
        """
        if filename is None:
            filename = f"factor_rotation_{signal.date}.json"
            
        filepath = self.signals_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(signal.to_dict(), f, indent=2)
            
        logger.info(f"Saved rotation signal to {filepath}")
        return filepath
    
    def get_latest_signal(self) -> Optional[RotationSignal]:
        """
        Retrieve the most recent rotation signal.
        
        Returns:
            RotationSignal or None if no signals found
        """
        if not self.signals_dir.exists():
            return None
            
        signal_files = sorted(self.signals_dir.glob("factor_rotation_*.json"), reverse=True)
        
        if not signal_files:
            return None
            
        with open(signal_files[0]) as f:
            data = json.load(f)
            
        return RotationSignal(
            date=data["date"],
            regime=data["regime"],
            quality_momentum_score=data["quality_momentum_score"],
            confidence=data["confidence"],
            factor_allocations=data["factor_allocations"],
            equity_adjustment=data["equity_adjustment"],
            rationale=data["rationale"],
        )


class FactorRotationIntegrator:
    """
    Integrates factor rotation signals with the ensemble voter.
    
    Provides standardized interface for signal integration.
    """
    
    def __init__(self):
        self.calculator = QualityMomentumCalculator()
        
    def get_signal_for_ensemble(
        self,
        date: Optional[str] = None,
    ) -> Dict:
        """
        Get factor rotation signal formatted for ensemble voter.
        
        Args:
            date: Target date
            
        Returns:
            Signal dictionary with standardized fields
        """
        signal = self.calculator.generate_rotation_signal(date)
        
        return {
            "source": "FACTOR_ROTATION",
            "date": signal.date,
            "direction": signal.regime,
            "strength": abs(signal.quality_momentum_score),
            "signal_value": signal.quality_momentum_score,  # -1 to +1
            "confidence": signal.confidence,
            "factor_allocations": signal.factor_allocations,
            "equity_adjustment": signal.equity_adjustment,
            "rationale": signal.rationale,
        }
    
    def get_backtest_allocations(
        self,
        start_date: str,
        end_date: str,
        rebalance_freq: str = "monthly",
    ) -> List[Dict]:
        """
        Generate historical allocations for backtesting.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            rebalance_freq: Rebalancing frequency (monthly, quarterly)
            
        Returns:
            List of allocation dictionaries with dates
        """
        allocations = []
        
        current = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        while current <= end:
            date_str = current.strftime("%Y-%m-%d")
            
            try:
                signal = self.calculator.generate_rotation_signal(date_str)
                allocations.append({
                    "date": date_str,
                    "regime": signal.regime,
                    "allocations": signal.factor_allocations,
                    "equity_adjustment": signal.equity_adjustment,
                    "quality_momentum_score": signal.quality_momentum_score,
                })
            except Exception as e:
                logger.warning(f"Failed to generate signal for {date_str}: {e}")
                
            # Advance date
            if rebalance_freq == "monthly":
                current = current + timedelta(days=30)
            elif rebalance_freq == "quarterly":
                current = current + timedelta(days=90)
            else:
                current = current + timedelta(days=30)
                
        return allocations


# CLI interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Factor Rotation Signal Generator")
    parser.add_argument("--date", type=str, help="Target date (YYYY-MM-DD)")
    parser.add_argument("--regime", type=str, choices=[r.value for r in MarketRegime], help="Override regime")
    parser.add_argument("--backtest", action="store_true", help="Run backtest mode")
    parser.add_argument("--start", type=str, help="Backtest start date")
    parser.add_argument("--end", type=str, help="Backtest end date")
    parser.add_argument("--output", type=str, help="Output file path")
    
    args = parser.parse_args()
    
    calc = QualityMomentumCalculator()
    
    if args.backtest and args.start and args.end:
        integrator = FactorRotationIntegrator()
        allocations = integrator.get_backtest_allocations(args.start, args.end)
        
        output = {
            "start_date": args.start,
            "end_date": args.end,
            "allocation_count": len(allocations),
            "allocations": allocations,
        }
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(output, f, indent=2)
            print(f"Saved backtest to {args.output}")
        else:
            print(json.dumps(output, indent=2))
            
    else:
        regime = MarketRegime(args.regime) if args.regime else None
        signal = calc.generate_rotation_signal(args.date, regime)
        
        output = signal.to_dict()
        
        if args.output:
            calc.save_signal(signal, args.output)
        else:
            print(json.dumps(output, indent=2))

"""
Factor Premia Signal Generator (v4.10 Phase 2)

Implements cross-sectional factor ranking for risk premia harvesting:
- Value (VLUE): Long cheap assets, short expensive
- Momentum (MTUM): Cross-sectional momentum
- Quality (QUAL): Profitable, low-debt firms
- Low Volatility (USMV): Low volatility vs high volatility

Target: +0.03 to +0.04 Sharpe improvement with 10-15% factor allocation.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum
import json


class FactorType(Enum):
    VALUE = "value"
    MOMENTUM = "momentum"
    QUALITY = "quality"
    LOW_VOL = "low_vol"


@dataclass
class FactorSignal:
    """Individual factor signal for an ETF."""
    etf: str
    factor: FactorType
    score: float  # Normalized 0-100
    rank: int     # Cross-sectional rank (1 = highest score)
    z_score: float
    recommendation: str  # "overweight", "neutral", "underweight"
    confidence: float    # 0-100 based on data quality


@dataclass
class FactorEnsemble:
    """Composite factor ensemble signal."""
    timestamp: str
    signals: Dict  # factor -> list of ETF signals
    composite_scores: Dict[str, float]  # etf -> weighted composite
    top_pick: Optional[str]
    bottom_pick: Optional[str]
    regime_adjustment: Dict  # factor weights by regime
    burn_in_progress: float  # 0-100, how complete is burn-in


class FactorPremiaCalculator:
    """
    Cross-sectional factor ranking for risk premia harvesting.
    
    Uses ETF-based factor exposure (MTUM, VLUE, USMV, QUAL) with:
    - 12-month lookback for momentum
    - 3-month average for signal smoothing
    - Regime-based weight adjustments
    - Correlation-based factor capping
    """
    
    FACTOR_ETFS = {
        FactorType.MOMENTUM: "MTUM",
        FactorType.VALUE: "VLUE",
        FactorType.QUALITY: "QUAL",
        FactorType.LOW_VOL: "USMV",
    }
    
    # ETF-specific configuration
    ETF_CONFIG = {
        "MTUM": {
            "formation_period": 252,  # 12 months
            "skip_period": 21,        # 1 month skip
            "vol_target": 0.15,
            "max_allocation": 0.05,   # 5% max
        },
        "VLUE": {
            "formation_period": 252,
            "skip_period": 21,
            "vol_target": 0.12,
            "max_allocation": 0.05,
            "min_holding_months": 12,  # Value needs patience
        },
        "QUAL": {
            "formation_period": 252,
            "skip_period": 21,
            "vol_target": 0.12,
            "max_allocation": 0.03,   # 3% max
        },
        "USMV": {
            "formation_period": 252,
            "skip_period": 21,
            "vol_target": 0.10,
            "max_allocation": 0.03,
        },
    }
    
    # Regime-based factor weight multipliers
    REGIME_MULTIPLIERS = {
        "early_cycle": {
            FactorType.MOMENTUM: 1.5,
            FactorType.VALUE: 0.8,
            FactorType.QUALITY: 0.7,
            FactorType.LOW_VOL: 0.5,
        },
        "mid_cycle": {
            FactorType.MOMENTUM: 1.0,
            FactorType.VALUE: 1.0,
            FactorType.QUALITY: 1.0,
            FactorType.LOW_VOL: 1.0,
        },
        "late_cycle": {
            FactorType.MOMENTUM: 0.7,
            FactorType.VALUE: 1.2,
            FactorType.QUALITY: 1.3,
            FactorType.LOW_VOL: 1.4,
        },
        "recession": {
            FactorType.MOMENTUM: 0.3,
            FactorType.VALUE: 0.6,
            FactorType.QUALITY: 1.2,
            FactorType.LOW_VOL: 1.5,
        },
    }
    
    def __init__(self, prices_df: Optional[pd.DataFrame] = None):
        """
        Initialize with price data.
        
        Args:
            prices_df: DataFrame with dates as index and ETF columns (MTUM, VLUE, QUAL, USMV)
        """
        self.prices = prices_df
        self.signals_history: List[FactorEnsemble] = []
        self.correlation_matrix: Optional[pd.DataFrame] = None
        
    def load_data(self, data_path: str) -> None:
        """Load price data from CSV or parquet."""
        if data_path.endswith('.parquet'):
            self.prices = pd.read_parquet(data_path)
        else:
            self.prices = pd.read_csv(data_path, index_col=0, parse_dates=True)
        
        # Ensure we have required columns
        required = set(self.FACTOR_ETFS.values())
        available = set(self.prices.columns)
        missing = required - available
        if missing:
            raise ValueError(f"Missing required ETFs: {missing}")
    
    def calculate_momentum_score(self, etf: str, 
                                  lookback: int = 252,
                                  skip: int = 21) -> Tuple[float, float]:
        """
        Calculate time-series momentum score (TSMOM style).
        
        Returns:
            (score, annualized_vol)
        """
        if self.prices is None or len(self.prices) < lookback + skip:
            return 0.0, 0.15
        
        prices = self.prices[etf].dropna()
        if len(prices) < lookback + skip:
            return 0.0, 0.15
        
        # Formation period return (skip most recent month)
        end_price = prices.iloc[-(skip + 1)]
        start_price = prices.iloc[-(lookback + skip)]
        formation_return = (end_price / start_price) - 1
        
        # Volatility scaling (20-day realized vol, annualized)
        recent_returns = prices.pct_change().iloc[-20:]
        vol = recent_returns.std() * np.sqrt(252)
        
        if vol == 0:
            vol = 0.15  # Default vol
        
        # Vol-scaled momentum signal
        vol_target = self.ETF_CONFIG[etf]["vol_target"]
        score = formation_return / vol * vol_target
        
        return score, vol
    
    def calculate_cross_sectional_rank(self, factor: FactorType) -> List[FactorSignal]:
        """
        Calculate cross-sectional ranking for a single factor.
        
        For ETFs, we use momentum-style scoring as a proxy for factor exposure.
        In production, this would use factor loadings from risk model.
        
        Returns:
            List of FactorSignal sorted by score (highest first)
        """
        etf = self.FACTOR_ETFS[factor]
        scores = {}
        
        # Calculate raw scores
        for factor_type, ticker in self.FACTOR_ETFS.items():
            score, vol = self.calculate_momentum_score(ticker)
            # Store both raw momentum and factor-adjusted
            scores[ticker] = {
                'raw_score': score,
                'vol': vol,
                'factor': factor_type,
            }
        
        # For cross-sectional ranking, we rank ETFs by their momentum
        # This creates a relative value signal
        raw_scores = [s['raw_score'] for s in scores.values()]
        
        if len(raw_scores) < 2 or np.std(raw_scores) == 0:
            # Not enough data for ranking
            return []
        
        # Calculate z-scores
        mean_score = np.mean(raw_scores)
        std_score = np.std(raw_scores)
        
        signals = []
        for ticker, data in scores.items():
            z_score = (data['raw_score'] - mean_score) / std_score if std_score > 0 else 0
            
            # Normalize to 0-100 scale
            normalized = 50 + z_score * 25  # Mean=50, 1 SD = 25 points
            normalized = max(0, min(100, normalized))  # Clip to 0-100
            
            # Determine recommendation
            if z_score > 0.5:
                recommendation = "overweight"
            elif z_score < -0.5:
                recommendation = "underweight"
            else:
                recommendation = "neutral"
            
            # Confidence based on data length
            data_days = len(self.prices) if self.prices is not None else 0
            confidence = min(100, data_days / 2.52)  # 252 days = 100% confidence
            
            signal = FactorSignal(
                etf=ticker,
                factor=factor,
                score=normalized,
                rank=0,  # Set after sorting
                z_score=z_score,
                recommendation=recommendation,
                confidence=confidence,
            )
            signals.append(signal)
        
        # Sort by score descending and assign ranks
        signals.sort(key=lambda x: x.score, reverse=True)
        for i, signal in enumerate(signals, 1):
            signal.rank = i
        
        return signals
    
    def calculate_factor_correlations(self) -> pd.DataFrame:
        """
        Calculate correlation matrix between factor ETFs.
        Used for diversification monitoring.
        """
        if self.prices is None or self.prices.empty:
            return pd.DataFrame()
        
        # Check if all required ETF columns are present
        required_etfs = list(self.FACTOR_ETFS.values())
        missing_etfs = [etf for etf in required_etfs if etf not in self.prices.columns]
        if missing_etfs:
            return pd.DataFrame()
        
        # Check for sufficient data (need at least 2 rows for pct_change to produce 1 row, 3+ for meaningful corr)
        if len(self.prices) < 3:
            return pd.DataFrame()
        
        returns = self.prices[required_etfs].pct_change().dropna()
        
        # Use 6-month rolling correlation
        if len(returns) > 126:
            returns = returns.iloc[-126:]
        
        self.correlation_matrix = returns.corr()
        return self.correlation_matrix
    
    def check_factor_crowding(self) -> Dict:
        """
        Check for factor crowding (high correlation between factors).
        
        Returns:
            Dict with crowding alerts and recommended caps
        """
        if self.correlation_matrix is None:
            self.calculate_factor_correlations()
        
        if self.correlation_matrix is None:
            return {"status": "unknown", "alerts": []}
        
        alerts = []
        caps = {}
        
        corr_matrix = self.correlation_matrix
        factor_etfs = list(self.FACTOR_ETFS.values())
        
        for i, etf1 in enumerate(factor_etfs):
            for j, etf2 in enumerate(factor_etfs):
                if i < j:
                    corr = corr_matrix.loc[etf1, etf2]
                    if corr > 0.8:
                        alerts.append(f"HIGH CORRELATION: {etf1}-{etf2} = {corr:.2f}")
                        caps[etf1] = caps.get(etf1, 0.05) * 0.8
                        caps[etf2] = caps.get(etf2, 0.05) * 0.8
                    elif corr > 0.7:
                        alerts.append(f"ELEVATED CORRELATION: {etf1}-{etf2} = {corr:.2f}")
        
        status = "normal"
        if alerts:
            has_high = any("HIGH" in str(a) for a in alerts)
            status = "critical" if has_high else "elevated"
        
        return {
            "status": status,
            "alerts": alerts,
            "recommended_caps": caps,
            "correlation_matrix": corr_matrix.to_dict(),
        }
    
    def get_regime_weights(self, regime: str = "mid_cycle") -> Dict[FactorType, float]:
        """
        Get factor weights adjusted for economic regime.
        
        Default is mid_cycle (equal weights).
        """
        base_weights = {
            FactorType.MOMENTUM: 0.30,
            FactorType.VALUE: 0.25,
            FactorType.QUALITY: 0.25,
            FactorType.LOW_VOL: 0.20,
        }
        
        multipliers = self.REGIME_MULTIPLIERS.get(regime, self.REGIME_MULTIPLIERS["mid_cycle"])
        
        # Apply multipliers
        adjusted = {factor: base_weights[factor] * multipliers[factor] 
                   for factor in base_weights}
        
        # Normalize to sum to 1
        total = sum(adjusted.values())
        return {factor: w/total for factor, w in adjusted.items()}
    
    def calculate_composite_scores(self, regime: str = "mid_cycle") -> Dict[str, float]:
        """
        Calculate composite factor scores for each ETF.
        
        Uses regime-adjusted weights to combine factor signals.
        """
        weights = self.get_regime_weights(regime)
        
        # Get all factor signals
        all_signals = {}
        for factor in FactorType:
            signals = self.calculate_cross_sectional_rank(factor)
            all_signals[factor] = {s.etf: s for s in signals}
        
        # Calculate weighted composite for each ETF
        composite = {}
        for etf in self.FACTOR_ETFS.values():
            score = 0
            total_weight = 0
            
            for factor, weight in weights.items():
                if factor in all_signals and etf in all_signals[factor]:
                    signal = all_signals[factor][etf]
                    score += signal.score * weight
                    total_weight += weight
            
            if total_weight > 0:
                composite[etf] = score / total_weight
            else:
                composite[etf] = 50  # Neutral
        
        return composite
    
    def generate_ensemble(self, regime: str = "mid_cycle") -> FactorEnsemble:
        """
        Generate complete factor ensemble signal.
        
        This is the main entry point for factor premia signals.
        """
        # Calculate individual factor rankings
        signals = {}
        for factor in FactorType:
            signals[factor] = self.calculate_cross_sectional_rank(factor)
        
        # Calculate composite scores
        composite = self.calculate_composite_scores(regime)
        
        # Determine top/bottom picks
        if composite:
            sorted_etfs = sorted(composite.items(), key=lambda x: x[1], reverse=True)
            top_pick = sorted_etfs[0][0] if sorted_etfs[0][1] > 60 else None
            bottom_pick = sorted_etfs[-1][0] if sorted_etfs[-1][1] < 40 else None
        else:
            top_pick = None
            bottom_pick = None
        
        # Calculate burn-in progress
        data_days = len(self.prices) if self.prices is not None else 0
        burn_in = min(100, data_days / 2.52)  # 252 trading days = 100%
        
        ensemble = FactorEnsemble(
            timestamp=datetime.now().isoformat(),
            signals=signals,
            composite_scores=composite,
            top_pick=top_pick,
            bottom_pick=bottom_pick,
            regime_adjustment=self.get_regime_weights(regime),
            burn_in_progress=burn_in,
        )
        
        self.signals_history.append(ensemble)
        return ensemble
    
    def get_allocation_recommendations(self, 
                                       total_factor_budget: float = 0.15,
                                       regime: str = "mid_cycle") -> Dict[str, float]:
        """
        Generate allocation recommendations for factor ETFs.
        
        Args:
            total_factor_budget: Maximum total factor allocation (default 15%)
            regime: Economic regime for weight adjustment
            
        Returns:
            Dict mapping ETF to recommended allocation
        """
        ensemble = self.generate_ensemble(regime)
        crowding = self.check_factor_crowding()
        
        allocations = {}
        
        # Start with zero allocation
        for etf in self.FACTOR_ETFS.values():
            allocations[etf] = 0.0
        
        # Apply recommendations based on composite scores
        for etf, score in ensemble.composite_scores.items():
            config = self.ETF_CONFIG[etf]
            max_alloc = config["max_allocation"]
            
            # Apply crowding cap if applicable
            if etf in crowding.get("recommended_caps", {}):
                max_alloc = min(max_alloc, crowding["recommended_caps"][etf])
            
            # Score-based allocation
            if score > 75:  # Strong overweight signal
                allocations[etf] = max_alloc
            elif score > 60:  # Moderate overweight
                allocations[etf] = max_alloc * 0.6
            elif score < 25:  # Strong underweight (avoid)
                allocations[etf] = 0.0
            elif score < 40:  # Moderate underweight
                allocations[etf] = max_alloc * 0.2
            else:  # Neutral
                allocations[etf] = max_alloc * 0.4
        
        # Normalize to total budget
        total = sum(allocations.values())
        if total > 0:
            scale = min(1.0, total_factor_budget / total)
            allocations = {etf: alloc * scale for etf, alloc in allocations.items()}
        
        # Ensure minimum allocation threshold (0.5%)
        min_threshold = 0.005
        allocations = {etf: (alloc if alloc >= min_threshold else 0.0) 
                      for etf, alloc in allocations.items()}
        
        return allocations
    
    def to_dict(self, ensemble: FactorEnsemble) -> Dict:
        """Convert ensemble to dictionary for serialization."""
        return {
            "timestamp": ensemble.timestamp,
            "signals": {
                factor.value: [
                    {
                        "etf": s.etf,
                        "score": s.score,
                        "rank": s.rank,
                        "z_score": s.z_score,
                        "recommendation": s.recommendation,
                        "confidence": s.confidence,
                    }
                    for s in signals
                ]
                for factor, signals in ensemble.signals.items()
            },
            "composite_scores": ensemble.composite_scores,
            "top_pick": ensemble.top_pick,
            "bottom_pick": ensemble.bottom_pick,
            "regime_weights": {k.value: v for k, v in ensemble.regime_adjustment.items()},
            "burn_in_progress": ensemble.burn_in_progress,
        }
    
    def get_current_summary(self) -> Dict:
        """Get current factor premia summary for dashboard display."""
        ensemble = self.generate_ensemble()
        allocations = self.get_allocation_recommendations()
        crowding = self.check_factor_crowding()
        
        # Calculate signal strength
        avg_confidence = np.mean([
            s.confidence for factor_signals in ensemble.signals.values() 
            for s in factor_signals
        ]) if ensemble.signals else 0
        
        return {
            "timestamp": ensemble.timestamp,
            "burn_in_progress": ensemble.burn_in_progress,
            "composite_scores": ensemble.composite_scores,
            "top_pick": ensemble.top_pick,
            "bottom_pick": ensemble.bottom_pick,
            "recommended_allocations": allocations,
            "total_factor_allocation": sum(allocations.values()),
            "crowding_status": crowding["status"],
            "crowding_alerts": crowding["alerts"],
            "avg_confidence": avg_confidence,
            "signal_ready": ensemble.burn_in_progress >= 100,
        }


def load_factor_data(data_dir: str = "data") -> pd.DataFrame:
    """
    Load factor ETF price data from the data directory.
    
    Expected files:
    - factor_etfs/mtum_prices.csv
    - factor_etfs/vlue_prices.csv
    - factor_etfs/qual_prices.csv
    - factor_etfs/usmv_prices.csv
    """
    import os
    
    etfs = ["MTUM", "VLUE", "QUAL", "USMV"]
    prices = {}
    
    for etf in etfs:
        # Try multiple file formats
        paths = [
            os.path.join(data_dir, f"factor_etfs/{etf.lower()}_prices.csv"),
            os.path.join(data_dir, f"factor_etfs/{etf}_prices.csv"),
            os.path.join(data_dir, f"prices_{etf}.csv"),
        ]
        
        df = None
        for path in paths:
            if os.path.exists(path):
                df = pd.read_csv(path, index_col=0, parse_dates=True)
                break
        
        if df is None:
            # Fallback: try to extract from main prices database
            main_path = os.path.join(data_dir, "prices.json")
            if os.path.exists(main_path):
                # Would need to implement extraction from compact format
                pass
            continue
        
        # Use 'adj close' or 'close'
        if 'adj close' in df.columns:
            prices[etf] = df['adj close']
        elif 'close' in df.columns:
            prices[etf] = df['close']
        else:
            prices[etf] = df.iloc[:, 0]  # First column
    
    if not prices:
        raise ValueError(f"No factor ETF data found in {data_dir}")
    
    df = pd.DataFrame(prices)
    df = df.dropna(how='any')  # Require all ETFs to have data
    return df


def main():
    """CLI entry point for factor premia signal generation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Factor Premia Signal Generator")
    parser.add_argument("--data-dir", default="data", help="Data directory path")
    parser.add_argument("--regime", default="mid_cycle", 
                       choices=["early_cycle", "mid_cycle", "late_cycle", "recession"],
                       help="Economic regime")
    parser.add_argument("--output", help="Output JSON file path")
    parser.add_argument("--format", choices=["summary", "full"], default="summary",
                       help="Output format")
    
    args = parser.parse_args()
    
    try:
        # Load data
        prices = load_factor_data(args.data_dir)
        calculator = FactorPremiaCalculator(prices)
        
        # Generate signals
        if args.format == "summary":
            result = calculator.get_current_summary()
        else:
            ensemble = calculator.generate_ensemble(args.regime)
            result = calculator.to_dict(ensemble)
        
        # Output
        json_output = json.dumps(result, indent=2, default=str)
        
        if args.output:
            with open(args.output, 'w') as f:
                f.write(json_output)
            print(f"Output written to {args.output}")
        else:
            print(json_output)
            
    except Exception as e:
        print(f"Error: {e}", file=__import__('sys').stderr)
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
